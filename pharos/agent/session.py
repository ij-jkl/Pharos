"""One part, one conversation: the agent loop, and the budget check that bounds it.

The loop is ordinary — send, receive tool calls, execute, append, repeat — and everything
interesting is in when it refuses to send.

Before every request the whole conversation is counted, catalogue included, and compared
against a ceiling that already has the hand-off reserve subtracted from it. If the next round
trip would cross that line the loop stops and asks for the hand-off instead. So a part never
sends a request it has not already proven fits, and the last thing it does always has room to
happen. That ordering is the whole design: a run that discovers it is out of window while
trying to report that it is out of window has lost the work.

The count is close, and slightly LOW: the chat template's own scaffolding is applied
server-side and is invisible from here. SAFETY_MARGIN is subtracted from every ceiling as a
first guess at that residue, but a constant cannot be right for every template -- calibrated
against one model at 0.87-0.98x, it was measured at 0.81-0.92x on qwen3.5-9b, whose worst
request fell 275 tokens short of a 256-token margin.

So the margin is a starting point, not the answer. ``prompt_eval_count`` comes back with every
response and says exactly how short the estimate fell for the template actually loaded, and
those pairs were already being recorded. ``_template_factor`` feeds them back, so the ceiling
tightens to whatever the backend has really been counting, and only ever tightens.

Erring high is the safe direction — a part sized against an inflated estimate fits, where the
reverse eventually overflows — but it is not free: it makes parts smaller and runs longer than
they need to be. So the ratio is measured against ``prompt_eval_count`` on every part and
published in the scorecard rather than quietly absorbed. Calling this a "floor" would be the
comfortable word and the wrong one.
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

from pharos.agent.ledger import FileChange, names_its_work
from pharos.agent.tools import ToolBox, catalogue_text, normalise

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

# How Pharos marks its own words inside a model's reply, so they can be told apart again.
_PHAROS_NOTE = "[Pharos]"

# --- compaction -------------------------------------------------------------------------------
#
# What a part spends its window on is overwhelmingly tool results: a read_file of a 700-line
# module is thousands of tokens that stay in the conversation for the rest of the part, long
# after the model has finished editing that file. Without compaction the ceiling simply stops
# the part there and asks for the hand-off -- correct, and expensive, because most of what
# filled the window is no longer being used.
#
# `--compact` replaces the OLDEST tool results with a stub naming what was dropped, oldest
# first, until the projection clears the ceiling again. Four rules keep it honest:
#
#   * only tool results are ever touched. The system prompt, the part body, every user turn
#     and everything the model itself said stay exactly as they were: compaction must not be
#     able to change what the part was asked to do.
#   * the message stays in place, stubbed rather than removed. A tool result deleted out from
#     under the assistant turn that called for it leaves a tool_call with no answer, which is
#     a malformed conversation, not a smaller one.
#   * the most recent results are never touched -- they are what the model is working on.
#   * bounded. A part that compacts, re-reads what it just dropped, and compacts again is
#     grinding, so after _MAX_COMPACTIONS rounds the ceiling goes back to stopping the part.
#
# The stub says what was dropped, which is the whole difference between compaction and a
# silently truncated context: the model can see that it read something and that the text is
# gone, rather than quietly losing the middle of its own history.
_COMPACT_KEEP_RECENT = 2
_MAX_COMPACTIONS = 3
# Below this a stub is not worth the confusion it causes: it costs tokens of its own, and a
# result small enough to be near it was never what filled the window.
_COMPACT_MIN_RECLAIM = 200
# The shape a tool result takes when the model's template could not emit a structured call;
# shared with _execute so the two spellings cannot drift apart.
_TOOL_RESULT_PREFIX = "TOOL RESULT ("

# The estimate is a floor and the chat template adds scaffolding it cannot see, so the ceiling
# stays this far clear of the real budget. Roughly one long tool result of slack.
SAFETY_MARGIN = 256

# How many times a part may be reminded that it stopped short. One was not enough: a part
# that wrote one of four files, was reminded, wrote a second and stopped had spent the only
# ask available. Each further reminder has to be earned by the previous one having produced a
# write, so a part that ignores the reminder is not asked a third time, and the cost of a
# part that simply will not finish stays bounded.
_MAX_NUDGES = 4

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


def _incomplete_nudge(missing: list[str], unread: list[str]) -> str:
    """Name the files still outstanding. A generic reminder is useless to a model that has
    already written something and believes it is finished."""
    listed = ", ".join(missing[:8]) + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
    never_opened = (
        f" You have not even read: {', '.join(unread[:8])}." if unread else ""
    )
    return (
        f"You are not done. These files are yours for this part and are still unchanged on "
        f"disk: {listed}.{never_opened} Read each one and make the edit with replace_lines or "
        f"write_file. Any hand-off you were given describes work on OTHER files. If a file "
        f"genuinely needs no change, name it and say why — do not skip it silently."
    )


_HANDOFF_REQUEST = (
    "Stop here. Do not call any more tools. In at most 10 lines, write the hand-off for the "
    "next part: NAME each file you changed and say what you did to it, then anything the next "
    "part needs to know. Write only about the files YOU were given. Do not say the overall "
    "task is complete — later parts cover files you have never seen, and a hand-off that "
    "claims completion stops them from doing their half. Do NOT write 'NO CHANGES NEEDED' "
    "here: that phrase is for declining work you were asked to do, and this is a report on "
    "work you have already done."
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
    # Everything below exists so a run can be SCORED rather than just watched. A part that
    # "finished" tells you nothing on its own; whether it wrote what it owned, handed the
    # thread on, and had room to breathe is what says the run held together.
    truncated: bool = False  # the backend dropped context: its own count fell mid-conversation
    ceiling: int = 0  # what this part was allowed, so peak means something as a fraction
    # (our projection, the backend's prompt_eval_count) for each request, paired at the moment
    # it was sent. Comparing a part's PEAK against whichever count came back last compares two
    # different requests and inflates the ratio — measured 2.56x that way against a true 1.3x.
    drift_samples: list[tuple[int, int]] = field(default_factory=list)
    nudges: int = 0  # how many times it stopped short and had to be asked again
    # Whether Pharos had to ASK for the hand-off rather than being handed one. Reported, never
    # netted off: a hand-off that had to be requested is still a hand-off the model wrote, and
    # a run where every one of them had to be asked for is a different run from one where none
    # did. See ``_ensure_handoff``.
    handoff_requested: bool = False
    scoped: list[str] = field(default_factory=list)  # the files this part owned
    handoff_tokens: int = 0  # size of the hand-off it produced
    # What the dispatcher saw land on disk, with the lines each write added. Pharos's own
    # record of the part, independent of anything the part says about itself, and what the
    # run carries to the next part -- see `pharos.agent.ledger`.
    changes: list[FileChange] = field(default_factory=list)
    # Requests this part sent with no correction to its ceiling at all -- nothing measured
    # yet in this part, and nothing remembered from an earlier run. See AgentSession.exposed.
    exposed_requests: int = 0
    # Compaction, both halves. ``compacted_tokens`` is what stubbing stale tool results
    # actually gave back; ``reclaimable_tokens`` is what it WOULD have given back on a part
    # that hit the ceiling with --compact off, so the run can say what it declined to do.
    compacted_tokens: int = 0
    compactions: int = 0
    reclaimable_tokens: int = 0
    compacted: list[str] = field(default_factory=list)  # what was stubbed, in order
    # Tool calls the window actually stopped -- refused for room, and still refused after any
    # compaction retry. Kept apart from every other refusal because it is the case the ceiling
    # check never sees: the part is comfortably inside its window and cannot open the next
    # file all the same, so it does NOT stop early, and the scorecard reported it as nothing.
    room_refusals: int = 0
    # How the model asked for its tools. A part that produced no NATIVE call has told you
    # something about the model rather than about the task -- see AgentSession, and the
    # scorecard row that reads these two together.
    native_calls: int = 0
    recovered_calls: int = 0
    # Files this part actually OPENED, as the dispatcher saw it -- not what the model says it
    # looked at. A scoped file that was read and left alone is a different outcome from one
    # nobody ever opened, and coverage alone cannot tell them apart.
    files_read: list[str] = field(default_factory=list)

    @property
    def nudged(self) -> bool:
        return self.nudges > 0


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
        num_ctx: int | None = None,
        template_offset: int | None = None,
        compact: bool = False,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        # Sent on EVERY request, not just the load. Ollama treats a differing options set as a
        # different configuration and reloads the model to serve it — so a run that loaded at
        # 16,384 and then asked for num_predict alone got the model reloaded at the backend's
        # default 4,096, while the ceiling went on being enforced against 16,384. The backend
        # then truncated silently. That is the exact failure this project exists to expose,
        # happening to its own client, and it is only visible because prompt_eval_count starts
        # falling while the conversation grows.
        self._num_ctx = num_ctx
        self._toolbox = toolbox
        self._count = count
        self._on_event = on_event or (lambda _message: None)
        self._tools = toolbox.catalogue()
        self._tool_names = {t["function"]["name"] for t in self._tools}
        self._system_prompt = (
            f"{_SYSTEM_PROMPT}\n\nThe workspace root is {toolbox.workspace.root}. "
            f"Every path you pass to a tool is relative to it."
        )
        # The CATALOGUE only. The system prompt is messages[0] and is counted there like any
        # other message; adding it here as well charged every request for it twice — 353
        # phantom tokens per projection on a real workspace, which is most of the gap between
        # what Pharos projected and what the backend reported. The planner subtracts both
        # (agent_overhead_tokens) because there it is genuinely fixed cost that no message
        # carries yet.
        self._catalogue_tokens = count(catalogue_text(self._tools))
        # The hand-off has to fit AFTER the conversation that produced it, so it comes out of
        # the ceiling up front rather than being hoped for at the end.
        self._ceiling = usable_budget - handoff_reserve - SAFETY_MARGIN
        # Kept, because the hand-off request is allowed to generate INTO it -- see _chat.
        self._handoff_reserve = max(handoff_reserve, 0)
        # What previous runs measured this model's chat template to cost, if anything has. A
        # part starts with no responses of its own, so without this its FIRST request is the
        # one request of the part enforced against an uncorrected ceiling.
        self._seed = max(template_offset or 0, 0)
        self._exposed = 0
        self._compact_enabled = compact
        self._messages: list[dict[str, Any]] = []
        # message index -> the path that tool call was about, so a stub can name it. Kept
        # beside the conversation rather than inside it: self._messages goes on the wire
        # verbatim, and a private key of ours has no business in somebody's request body.
        self._result_paths: dict[int, str] = {}
        self._compacted: list[str] = []
        self._compacted_tokens = 0
        self._compactions = 0
        self._room_refusals = 0
        self._native_calls = 0
        self._recovered_calls = 0
        self._drift: list[tuple[int, int]] = []
        self._largest_tooled = 0
        self.truncated_by_backend = False

    @property
    def catalogue_tokens(self) -> int:
        """What this client's own tool catalogue costs — exact, because Pharos wrote it."""
        return self._catalogue_tokens

    @property
    def ceiling(self) -> int:
        return self._ceiling

    def _projected(self, with_tools: bool = True) -> int:
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
        total = self._catalogue_tokens if with_tools else 0
        for message in self._messages:
            total += _PER_MESSAGE_TOKENS
            content = message.get("content")
            if isinstance(content, str) and content:
                total += self._count(content)
            calls = message.get("tool_calls")
            if calls:
                total += self._count(json.dumps(calls, separators=(",", ":")))
        return total

    def _template_offset(self) -> int:
        """How many tokens the backend's count has run above ours, for THIS conversation.

        The chat template is applied server-side and is invisible from here, so the projection
        is a floor by construction and ``SAFETY_MARGIN`` is a constant guessing how short it
        falls. The exact answer arrives with every response: ``prompt_eval_count`` is ground
        truth for the model and template actually loaded, and the pairs are already recorded
        for the drift line.

        TOKENS, not a ratio, and that is measured rather than assumed. Seventeen paired
        requests over conversations from 1,235 to 3,604 tokens put the gap at 263-314 tokens
        across the whole range -- flat, while the ratio it implied fell from 1.22x to 1.09x as
        the conversation grew. Fixed scaffolding is what a fixed number describes; scaling it
        would reserve three times what is needed on a large conversation and more on a larger.

        The worst gap seen is the one that counts -- a ceiling holds at the worst case or it
        does not hold. ``self._seed`` is what earlier runs measured, so the first request of a
        part is covered too. Never below zero: a backend counting fewer tokens than we did is
        not licence to fit more in, only a sign we were being cautious.
        """
        gaps = [reported - projected for projected, reported in self._drift if projected > 0]
        return max([0, self._seed, *gaps])

    @property
    def exposed(self) -> int:
        """Requests this part sent with no correction in force at all.

        The number the template memory exists to drive to zero. Not the same question as the
        drift line below, which measures the ESTIMATOR and is expected to read short: this
        asks whether any request was enforced against a ceiling with nothing but a constant
        behind it. Without the template memory that is the first request of every part.
        """
        return self._exposed

    def _projected_for_ceiling(self) -> int:
        """The projection the ceiling is enforced against: ours, plus what the backend has
        actually been counting on top. Deliberately not what ``peak_tokens`` reports, which
        stays the raw estimate so the drift line keeps measuring the estimator, not itself."""
        return self._projected() + self._template_offset()

    async def run(self, part_body: str) -> PartResult:
        self._messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": part_body},
        ]
        self._drift = []
        self._largest_tooled = 0
        self._result_paths = {}
        self._compacted = []
        self._compacted_tokens = 0
        self._compactions = 0
        self._room_refusals = 0
        self._native_calls = 0
        self._recovered_calls = 0
        # Both of these are per-part measurements that PartResult reports, and both were
        # missing from the reset above while every other counter beside them was in it. A
        # session is built per part today, so nothing carried -- but the block exists because
        # run() is meant to be re-entrant, and two figures that silently accumulate across
        # calls are the shape of a number that reads plausibly and is the sum of a run.
        self._exposed = 0
        self.truncated_by_backend = False
        nudges = 0
        written_at_last_nudge = 0
        read_at_last_nudge = 0
        failures = 0
        scoped = sorted(self._toolbox.scope) if self._toolbox.scope else []
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
            if self._projected_for_ceiling() > self._ceiling and self._compact_enabled:
                # Compaction runs BEFORE the refusal and cannot skip it: whatever it gives
                # back, the projection is measured again against the same ceiling, and a part
                # that still does not fit stops exactly where it would have stopped.
                reclaimed = self._compact()
                if reclaimed:
                    projected = self._projected()
                    self._on_event(
                        f"ceiling reached — compacted {reclaimed:,} tokens of stale tool "
                        f"results, back to {projected:,} of {self._ceiling:,}"
                    )
            if self._projected_for_ceiling() > self._ceiling:
                # Refusing BEFORE the send is the guarantee: nothing that has not been proven
                # to fit is ever put on the wire.
                stale = 0 if self._compact_enabled else self._reclaimable()
                offer = (
                    f" — {stale:,} of it is tool results this part finished with; --compact "
                    f"would give them back"
                    if stale
                    else ""
                )
                self._on_event(
                    f"window ceiling reached ({projected:,} of {self._ceiling:,}) — "
                    f"handing off{offer}"
                )
                stopped_early = True
                break

            # Only here, past both checks, is this conversation going on the wire — and only
            # what went on the wire can be a fraction of the ceiling that let it through.
            # Measured at the top of the loop instead, `peak` also caught the moment a tool
            # result had pushed the conversation OVER the ceiling, which is the state the two
            # checks above exist to undo: a run reported `headroom 104%`, a fraction of a
            # limit that by construction cannot be exceeded. The excursion is not lost — it is
            # what the compaction and hand-off lines above report, by name and in tokens.
            peak = max(peak, projected)

            try:
                message, reported = await self._chat(self._tools)
            except (httpx.HTTPError, ValueError) as exc:
                return self._result(
                    text="",
                    steps=steps,
                    peak=peak,
                    reported=reported,
                    nudges=nudges,
                    scoped=scoped,
                    error=f"backend call failed: {_describe(exc)}",
                )

            calls = message.get("tool_calls") or []
            self._native_calls += len(calls)
            recovered = False
            if not calls:
                calls = recover_tool_calls(str(message.get("content") or ""), self._tool_names)
                recovered = bool(calls)
                if recovered:
                    self._recovered_calls += len(calls)
                    self._on_event(f"recovered {len(calls)} tool call(s) written as text")
            self._messages.append(_assistant_message(message))
            if not calls and nudges < _MAX_NUDGES:
                # Not "wrote nothing" — "did not write everything it owns". A part that edits
                # two of its four files and stops looks like success to an all-or-nothing
                # check, and partial coverage is the failure mode that actually happens: runs
                # here have landed between 31% and 70%. The reminder names the specific files
                # still outstanding, because "you have not called write_file" tells a model
                # that just called write_file twice nothing it can act on.
                written = {normalise(p) for p in self._toolbox.files_written}
                missing = [p for p in scoped if normalise(p) not in written]
                text = str(message.get("content") or "")
                # "NO CHANGES NEEDED" is only credible about files it actually opened. A part
                # answering from the hand-off above it is describing somebody else's work.
                read = {normalise(p) for p in self._toolbox.files_read}
                unread = [p for p in missing if normalise(p) not in read]
                declined = "NO CHANGES NEEDED" in text.upper() and not unread
                # Progress is a write OR a file newly opened. Counting only writes cost a real
                # run two files: part 1 was reminded, went and READ the second of its two
                # files, wrote nothing, and was never asked again — because opening a file it
                # had not seen did not count as movement. It plainly is movement; a model that
                # has just read the file is one reminder away from editing it. What must not
                # earn another ask is a part that did nothing at all, and that is still the
                # test.
                progressed = (
                    len(self._toolbox.files_written) > written_at_last_nudge
                    or len(self._toolbox.files_read) > read_at_last_nudge
                )
                if missing and not declined and (nudges == 0 or progressed):
                    nudges += 1
                    written_at_last_nudge = len(self._toolbox.files_written)
                    read_at_last_nudge = len(self._toolbox.files_read)
                    self._on_event(
                        f"stopped with {len(missing)} of {len(scoped)} file(s) unchanged — "
                        f"asking again ({nudges} of {_MAX_NUDGES})"
                    )
                    _logger.info("incomplete part (read %s, wrote %s): %r",
                                 sorted(read), sorted(written), text)
                    self._messages.append(
                        {"role": "user", "content": _incomplete_nudge(missing, unread)}
                    )
                    continue
                # An unrestricted part has no list to check against, so fall back to the
                # all-or-nothing question rather than never asking at all.
                unscoped_and_idle = (
                    not scoped
                    and not self._toolbox.files_written
                    and "NO CHANGES NEEDED" not in text.upper()
                )
                if unscoped_and_idle and nudges == 0:
                    nudges += 1
                    self._on_event("answered without editing — asking once for the edit")
                    self._messages.append({"role": "user", "content": _NO_WRITE_NUDGE})
                    continue
            if not calls:
                # No tool calls means the model considers the part done. Its parting message
                # is only a hand-off if it happens to read like one -- see _ensure_handoff.
                text, reported, asked = await self._ensure_handoff(message, reported)
                return self._result(
                    text=text,
                    steps=steps,
                    peak=peak,
                    reported=reported,
                    nudges=nudges,
                    scoped=scoped,
                    handoff_requested=asked,
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
        return self._result(
            text=text,
            steps=steps,
            peak=peak,
            reported=reported,
            nudges=nudges,
            scoped=scoped,
            stopped_early=stopped_early,
            handoff_requested=True,
        )

    def _result(
        self,
        *,
        text: str,
        steps: int,
        peak: int,
        reported: int | None,
        nudges: int,
        scoped: list[str],
        stopped_early: bool = False,
        error: str | None = None,
        handoff_requested: bool = False,
    ) -> PartResult:
        """One place where a finished part is described, whichever way it finished.

        A part can end four ways and three of them wrote out the same twenty-odd fields by
        hand. Every field read off ``self`` is the same in all three, so the only thing that
        repetition could ever produce is disagreement between them -- and it did, twice in one
        afternoon: ``files_read`` and ``handoff_requested`` were each added to some of the
        exits and not others, and a part that ended down the forgotten path reported a default
        instead of what happened. What varies genuinely is the handful of arguments above.

        The fourth exit -- a part whose own body will not fit under the ceiling -- stays
        written out where it happens and deliberately does not come through here. Nothing has
        run at that point: there is no conversation, no drift sample and no request, so it
        reports ``ceiling=0`` and the scorecard leaves it out of ``peak_fraction`` rather than
        recording a part that exceeded a ceiling it never sent anything against.
        """
        return PartResult(
            text=text,
            steps=steps,
            files_written=list(self._toolbox.files_written),
            peak_tokens=peak,
            reported_tokens=reported,
            stopped_early=stopped_early,
            scope_refusals=list(self._toolbox.scope_refusals),
            error=error,
            changes=self._toolbox.changes(),
            exposed_requests=self._exposed,
            ceiling=self._ceiling,
            compacted_tokens=self._compacted_tokens,
            compactions=self._compactions,
            reclaimable_tokens=(0 if self._compact_enabled else self._reclaimable()),
            compacted=list(self._compacted),
            room_refusals=self._room_refusals,
            native_calls=self._native_calls,
            recovered_calls=self._recovered_calls,
            files_read=list(self._toolbox.files_read),
            nudges=nudges,
            scoped=scoped,
            drift_samples=list(self._drift),
            truncated=self.truncated_by_backend,
            handoff_tokens=self._count(text),
            handoff_requested=handoff_requested,
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

        room = max(self._ceiling - self._projected_for_ceiling(), 0)
        result = self._toolbox.dispatch(name, arguments, room=room, count=self._count)
        if self._compact_enabled and not result.ok and result.needed_room is not None:
            # The commonest shape of the problem, and the one the ceiling check never sees:
            # the part is comfortably inside its window and still cannot open the next file,
            # because the window is full of files it finished with ten steps ago.
            reclaimed = self._compact(headroom=result.needed_room)
            if reclaimed:
                self._on_event(
                    f"no room for {name} — compacted {reclaimed:,} tokens and retried"
                )
                room = max(self._ceiling - self._projected_for_ceiling(), 0)
                result = self._toolbox.dispatch(name, arguments, room=room, count=self._count)
        # Counted AFTER any retry, so it means "the window stopped this call" rather than
        # "the window was tight for a moment". With --compact on and a retry that went
        # through, nothing was refused and nothing should be reported as refused.
        if not result.ok and result.needed_room is not None:
            self._room_refusals += 1
        detail = result.wrote or arguments.get("path") or ""
        self._on_event(f"{'ok ' if result.ok else '!! '}{name} {detail}".rstrip())
        if result.ok:
            _logger.info("tool %s(%s) ok", name, _loggable(arguments))
        else:
            # The reason, not just the fact. Six consecutive failures abandoned a part and the
            # log said only which tool and which path — so the one question worth asking, why,
            # needed the run reproducing. A refusal already carries its own explanation; it
            # just was not being written down.
            _logger.warning(
                "tool %s(%s) REFUSED: %s", name, _loggable(arguments), result.text[:400]
            )
        if as_user:
            # A model whose template never emitted a structured call will not reliably render
            # a "tool" turn back either, so the result goes in as ordinary user text. Same
            # content, a shape the template definitely understands.
            body = f"{_TOOL_RESULT_PREFIX}{name}):\n{result.text}"
            self._messages.append({"role": "user", "content": body})
        else:
            self._messages.append({"role": "tool", "tool_name": name, "content": result.text})
        self._result_paths[len(self._messages) - 1] = str(arguments.get("path") or "")
        return result.ok

    # --- compaction ---------------------------------------------------------------------

    def _tool_result_indices(self) -> list[int]:
        """Where the tool results sit, in order. Both shapes a result can take count."""
        found: list[int] = []
        for index, message in enumerate(self._messages):
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "tool" or (role == "user" and content.startswith(_TOOL_RESULT_PREFIX)):
                found.append(index)
        return found

    def _compactable(self) -> list[int]:
        """Tool results old enough AND large enough to stub.

        Size first, and it is not an optimisation. Not every tool result is a file: a refusal
        for space, a write confirmation, a failed call are all tool results and all tiny.
        Counting them into the "most recent" window meant three consecutive refusals could
        push the one real file the part had read out of protection and get it stubbed, while
        the three messages saying "there was no room" were carefully preserved. Measured
        exactly that way on the first run of this code.
        """
        substantial = [
            index
            for index in self._tool_result_indices()
            if self._count(str(self._messages[index].get("content") or ""))
            > _COMPACT_MIN_RECLAIM
        ]
        if len(substantial) <= _COMPACT_KEEP_RECENT:
            return []
        return substantial[:-_COMPACT_KEEP_RECENT]

    def _reclaimable(self) -> int:
        """What compaction would give back right now, without doing it.

        Reported when --compact is off and the ceiling has just stopped a part, because "your
        window filled up" and "your window filled up with files you finished with eleven steps
        ago" are different things to be told.
        """
        total = 0
        for index in self._compactable():
            content = str(self._messages[index].get("content") or "")
            total += max(self._count(content) - self._count(self._stub(index, 0)), 0)
        return total

    def _stub(self, index: int, dropped: int) -> str:
        """What replaces a tool result: what it was, how big it was, and that it can be had
        again. A model that can see it read something and that the text is gone can decide
        what to do about it; a context silently truncated underneath it cannot."""
        message = self._messages[index]
        name = str(message.get("tool_name") or "") or _result_tool_name(message)
        about = f"{name} {self._result_paths.get(index, '')}".strip() or "a tool call"
        body = (
            f"[pharos dropped the {dropped:,}-token result of {about} to make room in the "
            f"window. It was read earlier in this part. Call it again only if you still need "
            f"its contents.]"
        )
        if message.get("role") == "tool":
            return body
        # Keep the prefix, so the message still reads as a tool result to everything above.
        return f"{_TOOL_RESULT_PREFIX}{name or 'tool'}):\n{body}"

    def _compact(self, headroom: int = 0) -> int:
        """Stub stale tool results until there is room again. Returns the tokens reclaimed.

        ``headroom`` is what has to fit ON TOP of the conversation afterwards: zero when the
        ceiling itself has been reached, and the size of the file when a read was refused for
        space. Both are the same question asked at two moments.

        Oldest first, and it stops the moment there is enough: a part that needed 300 tokens
        does not lose the last four files it read to get them.
        """
        if self._compactions >= _MAX_COMPACTIONS:
            return 0
        reclaimed = 0
        for index in self._compactable():
            if self._projected_for_ceiling() + headroom <= self._ceiling:
                break
            message = self._messages[index]
            before = self._count(str(message.get("content") or ""))
            stub = self._stub(index, before)
            saved = before - self._count(stub)
            if saved < _COMPACT_MIN_RECLAIM:
                continue
            message["content"] = stub
            reclaimed += saved
            name = str(message.get("tool_name") or "") or _result_tool_name(message)
            self._compacted.append(f"{name} {self._result_paths.get(index, '')}".strip())
        if reclaimed:
            self._compactions += 1
            self._compacted_tokens += reclaimed
            # The truncation detector rests on "a conversation only grows, so the backend's
            # count for it can only grow". Compaction is Pharos deliberately making the
            # conversation smaller, which breaks that premise -- and both features shipped in
            # the same version, so they met for the first time on a real run. Measured in
            # DESKTOP_VALIDATION §25: one line reclaimed 1,955 tokens, and the next two
            # requests were reported as BACKEND TRUNCATED, blaming the user's backend for
            # dropping context Pharos had just dropped itself.
            #
            # The high-water mark is dropped rather than adjusted by ``reclaimed``. That
            # figure is in OUR vocabulary and the mark is in the backend's, so subtracting one
            # from the other would leave a mark that is wrong by the drift between them -- and
            # wrong in the direction that cries wolf again, only more quietly. Zero rearms on
            # the very next tooled request, which costs exactly one request of blindness, at
            # the one moment the conversation is at its smallest and truncation is least
            # possible.
            self._largest_tooled = 0
        return reclaimed

    async def _ensure_handoff(
        self, message: dict[str, Any], reported: int | None
    ) -> tuple[str, int | None, bool]:
        """The parting message, or an explicit request when it is not a hand-off.

        A part can end two ways, and until this existed only one of them was ever ASKED for a
        hand-off. A part cut short by the step limit gets ``_HANDOFF_REQUEST`` -- the
        instruction that says NAME each file you changed. A part that simply stops calling
        tools, which is the ordinary way a part ends, got nothing: whatever it happened to say
        in the same breath as its last tool call was taken as the hand-off. So the careful
        instruction was reserved for the exit that went badly, and the common exit was left to
        luck.

        Measured on the six-part run in DESKTOP_VALIDATION §25, where no part was abandoned
        and every one of them therefore took the unasked path: 5 hand-offs expected, 3
        produced, 2 of those 3 thin. Two parts replied to their own last tool result with
        nothing at all, and an empty string is not a hand-off however generously it is read.

        The test is the same one the scorecard applies -- does the text name a file this part
        changed -- so a part that volunteered a real hand-off keeps its own words and costs
        nothing extra. Anything else is asked once, with the proper instruction.

        This does not make continuity green by construction. Being asked is not the same as
        answering: the model still has to name its files, ``names_its_work`` still judges the
        reply it gives, and a part that answers the request with prose about nothing still
        counts as thin. What it removes is an instrument that measured two different protocols
        and reported the difference as a property of the model.

        A part that wrote nothing and said so is left alone. It has no work to name, the
        request explicitly forbids the phrase it correctly used, and asking would only talk it
        out of a true answer.
        """
        text = str(message.get("content") or "").strip()
        said = model_words(text)
        if said and names_its_work(said, self._toolbox.files_written):
            return text, reported, False
        asked, reported = await self._request_handoff(reported)
        return (asked or text), reported, True

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
        has_tools = bool(tools)
        # Counted before the send, because that is when the exposure happens. A factor of
        # exactly 1.0 means nothing corrected this request's ceiling: no measurement from an
        # earlier request in this part, and nothing remembered from an earlier run. Only
        # SAFETY_MARGIN stood behind it, and nine live runs measured that constant ~290 tokens
        # short on this model.
        if self._template_offset() <= 0:
            self._exposed += 1
        # Two things the tool-less hand-off request must not be charged for.
        #
        # The catalogue is not sent with it, so projecting one costs it room that will
        # genuinely be free. And the ceiling has ``handoff_reserve`` subtracted already,
        # precisely so the hand-off has somewhere to go -- measuring the reply against the
        # ceiling hands it everything EXCEPT the space set aside for it, which is the one
        # request that space exists for.
        #
        # Both together are why a part that ended at 99% of its ceiling was asked for its
        # hand-off with 443 tokens against a 500-token reserve, and answered with nothing at
        # all: the reply was cut off before a word of it arrived, and what the run recorded as
        # that part's hand-off was Pharos's own note saying so (DESKTOP_VALIDATION §27).
        limit = self._ceiling + (0 if has_tools else self._handoff_reserve)
        room = max(limit - self._projected(with_tools=has_tools), _MIN_REPLY_ROOM)
        # Paired below with whatever prompt_eval_count comes back for THIS request.
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "options": _request_options(room, self._num_ctx),
        }
        if tools:
            payload["tools"] = tools

        # Counted on the same terms as the request being sent. The hand-off goes out with no
        # tool catalogue, so charging our side for one would make that sample incomparable and
        # inflate the drift figure for every part that ends that way.
        projected_now = self._projected(with_tools=has_tools)
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

        if prompt_eval:
            # A conversation only grows, so the backend's count for it can only grow. When it
            # falls, the backend is no longer evaluating everything it was sent — it is
            # dropping the oldest tokens to fit a window smaller than the one Pharos measured.
            # Detected rather than inferred: this is the silent context loss the whole project
            # is about, and Pharos must not be the last to notice it in its own client.
            # Only against requests of the same shape. A tool-less request legitimately
            # carries a few hundred fewer tokens, and reading that as context loss would make
            # the one warning that must never cry wolf fire on every part that hands off.
            previous = self._largest_tooled if has_tools else 0
            if has_tools:
                self._largest_tooled = max(self._largest_tooled, prompt_eval)
            if previous and prompt_eval < previous:
                self.truncated_by_backend = True
                self._on_event(
                    f"BACKEND TRUNCATED: it counted {prompt_eval:,} tokens for a conversation "
                    f"it counted {previous:,} for earlier — the loaded window is smaller than "
                    f"the one this run measured, and context is being dropped"
                )
                _logger.warning(
                    "backend truncation: prompt_eval fell %d -> %d while ours rose to %d",
                    previous, prompt_eval, projected_now,
                )
            self._drift.append((projected_now, prompt_eval))
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


def _result_tool_name(message: dict[str, Any]) -> str:
    """The tool a user-shaped result came from: the name inside `TOOL RESULT (name):`."""
    content = str(message.get("content") or "")
    if not content.startswith(_TOOL_RESULT_PREFIX):
        return ""
    closing = content.find(")", len(_TOOL_RESULT_PREFIX))
    return content[len(_TOOL_RESULT_PREFIX) : closing] if closing > 0 else ""


def _loggable(arguments: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments with any file content shortened.

    A write_file call carries an entire source file in `content`. Logging that verbatim turns
    the run log into a second copy of the repository and buries the one field that identifies
    the call, which is the path.
    """
    trimmed: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > 120:
            trimmed[key] = f"<{len(value)} chars>"
        else:
            trimmed[key] = value
    return trimmed


def model_words(text: str) -> str:
    """``text`` with Pharos's own annotations stripped -- what the MODEL actually said.

    A reply cut off at its token limit gets a note appended saying so, and a reply cut off
    before it produced a single character is then made ENTIRELY of that note. Left unexamined
    it reads as a hand-off: non-empty, in the part's own text field, counted as produced. It
    is Pharos talking to itself.

    Used wherever the question is about the model rather than about the run. Nothing is
    removed from what gets displayed or carried -- the note is the useful part for a person
    reading the log.
    """
    kept = [line for line in text.splitlines() if not line.lstrip().startswith(_PHAROS_NOTE)]
    return chr(10).join(kept).strip()


def _request_options(room: int, num_ctx: int | None) -> dict[str, Any]:
    """Per-request options. ``num_ctx`` has to be repeated or the backend reloads without it."""
    options: dict[str, Any] = {"num_predict": room}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    return options


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
