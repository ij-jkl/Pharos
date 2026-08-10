"""One part, one conversation: the agent loop, and the budget check that bounds it.

The loop is ordinary — send, receive tool calls, execute, append, repeat — and everything
interesting is in when it refuses to send.

Before every request the whole conversation is counted, catalogue included, and compared
against a ceiling that already has the hand-off reserve subtracted from it. If the next round
trip would cross that line the loop stops and asks for the hand-off instead. So a part never
sends a request it has not already proven fits, and the last thing it does always has room to
happen. That ordering is the whole design: a run that discovers it is out of window while
trying to report that it is out of window has lost the work.

The count is a floor, on the same terms as everywhere else in Pharos. Serialised JSON is not
byte-identical to what the model's chat template produces, and the backend adds its own
scaffolding Pharos cannot see. That is why the ceiling keeps a margin rather than aiming at
the last token, and why ``prompt_eval_count`` from the response is recorded and reported: it
is the ground truth, and a run that drifts from its own estimate should say so.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from pharos.agent.tools import ToolBox, catalogue_text

_logger = logging.getLogger("pharos.agent")

# Because the response is STREAMED, `read` is the gap allowed between two chunks, not the
# budget for the whole reply. A model generating steadily can take as long as it likes; one
# that has stopped producing anything is dead, and three minutes is generous for a first
# token after a cold prompt eval. The old non-streaming form needed 900s here, during which a
# stuck run was indistinguishable from a working one.
_STALL_SECONDS = 180.0
_TIMEOUT = httpx.Timeout(connect=5.0, read=_STALL_SECONDS, write=60.0, pool=5.0)

# How often the run says it is still alive while a reply streams in.
_PROGRESS_SECONDS = 5.0

# What a chat template spends per message on top of its content: the role marker and the
# turn delimiters. Small, and deliberately a floor — SAFETY_MARGIN covers the rest.
_PER_MESSAGE_TOKENS = 8

# Never ask for a reply cap below this: a handful of tokens cannot express even a refusal, and
# the ceiling check has already decided there is room to work.
_MIN_REPLY_ROOM = 256

# The estimate is a floor and the chat template adds scaffolding it cannot see, so the ceiling
# stays this far clear of the real budget. Roughly one long tool result of slack.
SAFETY_MARGIN = 256

# A model that keeps calling tools without converging is burning the window; stop and hand off
# rather than let it spin. Reached in practice only when a part was scoped too wide.
_MAX_STEPS = 40

# Consecutive failed tool calls before a part is abandoned. A model that has decided the
# project has a Models/ folder it does not have will keep trying variations of the same wrong
# path until the window runs out — one real run made eleven straight refused writes. Every one
# of those is correctly refused and the repository is never at risk, but grinding to the
# ceiling and then handing off nothing wastes minutes and tells the user less than stopping
# does. A handful of failures in a row is a part that has lost the plot, not a rough patch.
_MAX_CONSECUTIVE_FAILURES = 6

# Counted into the ceiling like everything else, and counted EXACTLY: Pharos wrote this
# client, so unlike the proxy's learned estimate for Continue or Cursor there is no guess in
# a run's own overhead.
#
# Every line here earned its place by a run failing without it. A model handed a bare task
# and a tool catalogue will happily describe the refactor it would perform, in detail, and
# call nothing — which reads as success and changes not one byte.
_SYSTEM_PROMPT = """You are a coding agent working directly in a local repository.

You change files ONLY by calling tools. Prose does not reach the disk: describing an edit,
quoting the new file contents in your reply, or explaining what you would do all leave the
repository exactly as it was. If the task asks for a change, you must call write_file.

How to work:
- Given a directory, call list_dir on it first to find out what is in it.
- Call read_file before you change a file. Never write a file you have not read.
- To change part of an existing file, use replace_lines with the line numbers read_file
  showed you. It is the tool to reach for: you send only the lines you are changing.
- Use write_file only for a NEW file, or when you are genuinely replacing a whole small one.
  It takes the COMPLETE contents, not a fragment.
- The "NNN| " prefixes in a read are line numbers, not file content. Never write them back.
- Do not ask questions. Nobody can answer; make the best change you can and note any
  assumption in your final message.
- If your change refers to a type, class or file that does not exist yet, create it with
  write_file in this same run. A dangling reference does not build, and leaving one behind
  turns a refactor into broken code.
- When the work is done, reply with a short summary and call no tools.

If, having looked, you conclude nothing needs changing, say exactly: NO CHANGES NEEDED."""

# One corrective nudge when a part ends having written nothing. Small models answer in prose
# far more often than they refuse, and a single reminder converts most of those; a second one
# never has, so there is no loop here.
_NO_WRITE_NUDGE = (
    "You have not called write_file, so nothing has changed on disk and the task is not done. "
    "The files listed in your scope above have NOT been handled yet — any hand-off you were "
    "given describes work on OTHER files. Open your own files with read_file and make the "
    "edit now. Only after you have read them, if no file needs changing, reply with exactly: "
    "NO CHANGES NEEDED."
)

_HANDOFF_REQUEST = (
    "Stop here. Do not call any more tools. In at most 10 lines, write the hand-off for the "
    "next part: what you changed, what you did not get to, and anything it needs to know. "
    "Write only about the files YOU were given. Do not say the overall task is complete — "
    "later parts cover files you have never seen, and a hand-off that claims completion "
    "stops them from doing their half."
)


def system_prompt_for(workspace_root: object) -> str:
    """The exact system prompt a session will send for this workspace.

    Shared with the planner so both subtract the identical figure — see
    ``pharos.agent.tools.agent_overhead_tokens``.
    """
    return (
        f"{_SYSTEM_PROMPT}\n\nThe workspace root is {workspace_root}. "
        f"Every path you pass to a tool is relative to it."
    )


@dataclass
class PartResult:
    """What one part's conversation produced."""

    text: str  # the model's closing message — the hand-off, when there is a next part
    steps: int
    files_written: list[str]
    peak_tokens: int  # highest projected conversation size reached
    reported_tokens: int | None  # prompt_eval_count from the last response: ground truth
    stopped_early: bool  # the budget ceiling cut the loop short
    scope_refusals: list[str] = field(default_factory=list)
    error: str | None = None


class AgentSession:
    """Drives one part to completion against the backend, inside a fixed token ceiling."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        model: str,
        toolbox: ToolBox,
        count: Callable[[str], int],
        usable_budget: int,
        handoff_reserve: int,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._toolbox = toolbox
        self._count = count
        self._on_event = on_event or (lambda _message: None)
        self._tools = toolbox.catalogue()
        self._tool_names = {t["function"]["name"] for t in self._tools}
        self._system_prompt = (
            f"{_SYSTEM_PROMPT}\n\nThe workspace root is {toolbox.workspace.root}. "
            f"Every path you pass to a tool is relative to it."
        )
        # The catalogue and the system prompt together are this client's whole overhead, and
        # both are counted rather than estimated — see the note above _SYSTEM_PROMPT.
        self._catalogue_tokens = count(catalogue_text(self._tools)) + count(self._system_prompt)
        # The hand-off has to fit AFTER the conversation that produced it, so it comes out of
        # the ceiling up front rather than being hoped for at the end.
        self._ceiling = usable_budget - handoff_reserve - SAFETY_MARGIN
        self._messages: list[dict[str, Any]] = []

    @property
    def catalogue_tokens(self) -> int:
        """What this client's own tool catalogue costs — exact, because Pharos wrote it."""
        return self._catalogue_tokens

    @property
    def ceiling(self) -> int:
        return self._ceiling

    def _projected(self) -> int:
        """Tokens the next request would carry: the conversation plus the catalogue.

        Counted over the message CONTENT, not over the JSON envelope. Serialising first looked
        equivalent and is not: JSON escapes every quote and newline, so a C# file costs about
        twice as much as JSON text as it does as text. A real run estimated 7,252 tokens for a
        conversation the backend then counted at 3,164, and being wrong by that much in the
        safe direction still costs the user real parts and real hand-offs.

        Each message pays a small constant for the role marker and delimiters its chat template
        adds, which keeps this a floor on what the backend will see rather than a hope.
        Tool-call arguments ARE serialised, because that is genuinely how they travel.
        """
        total = self._catalogue_tokens
        for message in self._messages:
            total += _PER_MESSAGE_TOKENS
            content = message.get("content")
            if isinstance(content, str) and content:
                total += self._count(content)
            calls = message.get("tool_calls")
            if calls:
                total += self._count(json.dumps(calls, separators=(",", ":")))
        return total

    async def run(self, part_body: str) -> PartResult:
        self._messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": part_body},
        ]
        nudged = False
        failures = 0
        peak = self._projected()
        reported: int | None = None
        stopped_early = False
        steps = 0

        if peak > self._ceiling:
            return PartResult(
                text="",
                steps=0,
                files_written=[],
                peak_tokens=peak,
                reported_tokens=None,
                stopped_early=True,
                error=(
                    f"the part itself is {peak:,} tokens against a {self._ceiling:,} ceiling, "
                    f"so there is no room to work in. Lower the scope, or raise the window."
                ),
            )

        while steps < _MAX_STEPS:
            projected = self._projected()
            peak = max(peak, projected)
            if projected > self._ceiling:
                # Refusing BEFORE the send is the guarantee: nothing that has not been proven
                # to fit is ever put on the wire.
                self._on_event(
                    f"window ceiling reached ({projected:,} of {self._ceiling:,}) — handing off"
                )
                stopped_early = True
                break

            try:
                message, reported = await self._chat(self._tools)
            except (httpx.HTTPError, ValueError) as exc:
                return PartResult(
                    text="",
                    steps=steps,
                    files_written=list(self._toolbox.files_written),
                    peak_tokens=peak,
                    reported_tokens=reported,
                    stopped_early=False,
                    scope_refusals=list(self._toolbox.scope_refusals),
                    error=f"backend call failed: {_describe(exc)}",
                )

            calls = message.get("tool_calls") or []
            recovered = False
            if not calls:
                calls = recover_tool_calls(str(message.get("content") or ""), self._tool_names)
                recovered = bool(calls)
                if recovered:
                    self._on_event(f"recovered {len(calls)} tool call(s) written as text")
            self._messages.append(_assistant_message(message))
            if not calls and not self._toolbox.files_written and not nudged:
                nudged = True
                text = str(message.get("content") or "")
                # "NO CHANGES NEEDED" is only credible from a part that actually looked. A
                # part that opened none of its own files is answering from the hand-off above
                # it, which describes somebody else's files.
                looked = bool(self._toolbox.files_read)
                if not looked or "NO CHANGES NEEDED" not in text.upper():
                    self._on_event("answered without editing — asking once for the actual edit")
                    _logger.info("no-edit reply (read %s): %r", self._toolbox.files_read, text)
                    self._messages.append({"role": "user", "content": _NO_WRITE_NUDGE})
                    continue
            if not calls:
                # No tool calls means the model considers the part done; its text is the
                # hand-off the part body asked for.
                return PartResult(
                    text=str(message.get("content") or "").strip(),
                    steps=steps,
                    files_written=list(self._toolbox.files_written),
                    peak_tokens=peak,
                    reported_tokens=reported,
                    stopped_early=False,
                    scope_refusals=list(self._toolbox.scope_refusals),
                )

            steps += 1
            for call in calls:
                if self._execute(call, as_user=recovered):
                    failures = 0
                else:
                    failures += 1
            if failures >= _MAX_CONSECUTIVE_FAILURES:
                self._on_event(
                    f"{failures} tool calls in a row failed — abandoning this part rather than "
                    f"grinding to the ceiling"
                )
                stopped_early = True
                break

        if steps >= _MAX_STEPS:
            stopped_early = True
            self._on_event(f"stopped after {_MAX_STEPS} tool rounds without converging")

        text, reported = await self._request_handoff(reported)
        return PartResult(
            text=text,
            steps=steps,
            files_written=list(self._toolbox.files_written),
            peak_tokens=peak,
            reported_tokens=reported,
            stopped_early=stopped_early,
            scope_refusals=list(self._toolbox.scope_refusals),
        )

    def _execute(self, call: dict[str, Any], *, as_user: bool = False) -> bool:
        """Run one tool call and append its result. Returns whether the call succeeded."""
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            # OpenAI-shaped backends send arguments as a JSON string; Ollama sends an object.
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        room = max(self._ceiling - self._projected(), 0)
        result = self._toolbox.dispatch(name, arguments, room=room, count=self._count)
        detail = result.wrote or arguments.get("path") or ""
        self._on_event(f"{'ok ' if result.ok else '!! '}{name} {detail}".rstrip())
        _logger.info("tool %s(%s) ok=%s", name, arguments, result.ok)
        if as_user:
            # A model whose template never emitted a structured call will not reliably render
            # a "tool" turn back either, so the result goes in as ordinary user text. Same
            # content, a shape the template definitely understands.
            body = f"TOOL RESULT ({name}):\n{result.text}"
            self._messages.append({"role": "user", "content": body})
        else:
            self._messages.append({"role": "tool", "tool_name": name, "content": result.text})
        return result.ok

    async def _request_handoff(self, reported: int | None) -> tuple[str, int | None]:
        """Ask for the closing summary, with no tools offered so it cannot start working again."""
        self._messages.append({"role": "user", "content": _HANDOFF_REQUEST})
        try:
            message, reported = await self._chat(None)
        except (httpx.HTTPError, ValueError) as exc:
            return f"(hand-off unavailable: {_describe(exc)})", reported
        return str(message.get("content") or "").strip(), reported

    async def _chat(self, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any], int | None]:
        """One turn, streamed — for the user's sake and for the run's.

        Streaming is not a cosmetic choice here. With ``stream: false`` a model rewriting a
        4,000-token file produces nothing observable for minutes, and the only report of a run
        in trouble is a timeout after a quarter of an hour. Both look identical from outside:
        a wedged machine with the GPU pinned and no output. Reading the response as it arrives
        turns that into a live token count, and — more usefully — turns httpx's read timeout
        from "the whole call took too long" into "no token has arrived in ``_STALL_SECONDS``",
        which is the condition actually worth failing on.

        ``num_predict`` is set to the room left under the ceiling. The model then cannot
        generate its way out of the window even in principle, rather than being asked nicely
        not to. A reply cut short by that limit is reported rather than passed off as
        complete — a truncated write_file call is exactly the kind of silent damage this
        project exists to refuse.
        """
        room = max(self._ceiling - self._projected(), _MIN_REPLY_ROOM)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "options": {"num_predict": room},
        }
        if tools:
            payload["tools"] = tools

        content: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        message: dict[str, Any] = {"role": "assistant"}
        prompt_eval: int | None = None
        truncated = False
        chunks = 0
        started = last_report = time.monotonic()

        async with self._client.stream(
            "POST", "/api/chat", json=payload, timeout=_TIMEOUT
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if "error" in data:
                    raise ValueError(str(data["error"]))
                part = data.get("message")
                if isinstance(part, dict):
                    piece = part.get("content")
                    if isinstance(piece, str) and piece:
                        content.append(piece)
                        chunks += 1
                    if part.get("tool_calls"):
                        tool_calls.extend(part["tool_calls"])
                    if part.get("role"):
                        message["role"] = part["role"]
                if data.get("done"):
                    value = data.get("prompt_eval_count")
                    prompt_eval = value if isinstance(value, int) else None
                    truncated = data.get("done_reason") == "length"
                now = time.monotonic()
                if now - last_report >= _PROGRESS_SECONDS:
                    last_report = now
                    self._on_event(
                        f"generating… {chunks:,} chunks, {now - started:.0f}s "
                        f"(cap {room:,} tokens)"
                    )

        message["content"] = "".join(content)
        if tool_calls:
            message["tool_calls"] = tool_calls
        if truncated:
            # Say it rather than hand back a half-written file as though it were finished.
            self._on_event(f"reply hit the {room:,}-token cap and was cut off")
            message["content"] += (
                f"\n\n[Pharos] This reply was cut off at the {room}-token limit — "
                f"what remains of the window."
            )
        return message, prompt_eval


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def recover_tool_calls(content: str, known: set[str]) -> list[dict[str, Any]]:
    """Parse tool calls a model wrote as TEXT instead of putting in ``tool_calls``.

    Not every template that advertises tool support actually emits Ollama's structured field.
    qwen2.5-coder:14b, for one, replies with a fenced ```json block holding {"name", "arguments"}
    and leaves tool_calls null — so a run against it looks exactly like a model that refuses to
    use its tools, and changes nothing while reporting success.

    Deliberately strict, because the cost of a false positive is executing something the model
    never meant as a call: the payload must be a JSON object (or a list of them) carrying an
    "arguments" mapping and a "name" that is actually in this part's catalogue. Prose that
    merely quotes JSON does not match, and neither does a call to a tool that does not exist.
    """
    payloads: list[Any] = []
    for block in _FENCE.findall(content):
        try:
            payloads.append(json.loads(block))
        except ValueError:
            continue
    if not payloads:
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            try:
                payloads.append(json.loads(stripped))
            except ValueError:
                return []

    calls: list[dict[str, Any]] = []
    for payload in payloads:
        for item in payload if isinstance(payload, list) else [payload]:
            if not isinstance(item, dict):
                continue
            # Some templates nest it one level down under "function".
            nested = item.get("function")
            body: dict[str, Any] = nested if isinstance(nested, dict) else item
            name, arguments = body.get("name"), body.get("arguments")
            if isinstance(name, str) and name in known and isinstance(arguments, dict):
                calls.append({"function": {"name": name, "arguments": arguments}})
    return calls


def _describe(exc: Exception) -> str:
    """An exception rendered so it is actually actionable.

    httpx's timeout and protocol errors stringify to the empty string, so the obvious
    f"{exc}" produced "backend call failed:" and nothing else — a report that tells you a run
    died and refuses to say how. The class name is the part that identifies the failure.
    """
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """The assistant turn to append, keeping only what the next request needs.

    A thinking model's ``thinking`` field is dropped: the backend does not want it echoed
    back, and it is pure cost in a window this tight.
    """
    kept: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        kept["tool_calls"] = message["tool_calls"]
    return kept
