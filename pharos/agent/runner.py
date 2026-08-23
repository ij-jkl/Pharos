"""Orchestration: pre-flight the task, divide it, then run each part as its own conversation.

The division is ``pharos split`` called as a library, unmodified. That is the point — the plan
a user can inspect with ``pharos split`` is the same plan the runner executes, so the dry run
and the real run cannot disagree. No model is consulted about what the task means; parts are
groups of the files the prompt named, packed by scope and position.

Traffic goes through the Pharos proxy by default rather than straight at the backend, so a run
appears in the dashboard like any other client and feeds the same calibration record. The
proxy does not know or care that Pharos wrote this client: it observes and forwards, exactly
as it does for Continue or Cursor. When the proxy is not running the runner falls back to the
backend directly and says so, because refusing to work without the dashboard would be a silly
dependency for a CLI.

Between parts the only thing carried forward is the hand-off the previous part wrote — a few
hundred tokens, bounded by ``handoff_reserve``. Each part otherwise starts from an empty
conversation. That is what keeps part 5 as affordable as part 1, and it is the difference
between this and an agent that simply runs until it dies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from pharos.agent.session import SAFETY_MARGIN, AgentSession, PartResult
from pharos.agent.tools import (
    ToolBox,
    agent_overhead_tokens,
    normalise,
    scope_from_part_files,
    workspace_root,
)
from pharos.agent.workspace import (
    NotARepository,
    Undo,
    Workspace,
    changed_files,
    current_branch,
    git_guard,
)
from pharos.config import PharosConfig
from pharos.preflight.check import CheckReport, Verdict, run_check
from pharos.preflight.split import PartFile, SplitMode, SplitPlan, build_plan
from pharos.tokenizer.gguf import GgufTokenizer
from pharos.tokenizer.resolver import resolve_gguf_path

_logger = logging.getLogger("pharos.agent")

_HEURISTIC_CHARS_PER_TOKEN = 4


@dataclass
class RunOutcome:
    """Everything the report needs, and the exit code the CLI derives from it."""

    report: CheckReport | None = None
    plan: SplitPlan | None = None
    parts: list[PartResult] = field(default_factory=list)
    branch: str | None = None
    base_branch: str | None = None  # what the run branched FROM, for the undo line
    undo: Undo | None = None  # set only when there was no git to fall back on
    files_changed: list[str] = field(default_factory=list)
    counts_exact: bool = False
    divided: bool = True  # False when --no-split ran the whole task as one conversation
    catalogue_tokens: int = 0
    repair_parts: int = 0  # extra parts run over files the plan assigned and nobody changed
    handoff_reserve: int = 0  # what each part held back, so the scorecard can judge overruns
    via_proxy: bool = True
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.parts) and all(p.error is None for p in self.parts)


def _counter(config: PharosConfig) -> tuple[Callable[[str], int], bool]:
    """The token counter for a run: the model's own GGUF vocabulary, or chars/4 and say so."""
    gguf = resolve_gguf_path(config)
    if gguf is None:
        return (lambda text: max(len(text) // _HEURISTIC_CHARS_PER_TOKEN, 0), False)
    tokenizer = GgufTokenizer(gguf)
    return (tokenizer.count, True)


async def _reachable(client: httpx.AsyncClient) -> bool:
    try:
        response = await client.get("/api/tags", timeout=httpx.Timeout(3.0))
    except httpx.HTTPError:
        return False
    return response.status_code < 500


# Ollama unloads an idle model after five minutes, which is shorter than the gap between
# planning a run and finishing it. Asking for the model to stay resident for the run's
# duration is the difference between "it just works" and a run that dies halfway because the
# window it measured against evaporated.
_KEEP_ALIVE = "30m"


def _load_payload(config: PharosConfig) -> dict[str, Any]:
    """The smallest request that makes the backend hold the model at the wanted window."""
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": "ok"}],
        "stream": False,
        "keep_alive": _KEEP_ALIVE,
    }
    if config.num_ctx is not None:
        payload["options"] = {"num_ctx": config.num_ctx}
    return payload


async def _ensure_window(
    config: PharosConfig,
    prompt: str,
    report: CheckReport,
    say: Callable[[str], None],
) -> CheckReport:
    """Reload the model when the resident window is not the one the user configured.

    Only ever when ``num_ctx`` is set explicitly. Pharos reports on the window it measures, so
    changing one behind the user's back would make its own headline number a fiction; changing
    one the user asked for, and saying so, does not.
    """
    profile = report.profile
    if config.num_ctx is None or profile is None or not profile.backend.reachable:
        return report
    loaded = profile.backend.loaded_ctx
    if loaded == config.num_ctx:
        return report

    say(
        f"resident window is {loaded or 'unknown'}, pharos.toml asks for {config.num_ctx:,} "
        f"— reloading the model"
    )
    try:
        async with httpx.AsyncClient(base_url=config.backend_url) as client:
            response = await client.post(
                "/api/chat",
                json=_load_payload(config),
                timeout=httpx.Timeout(connect=5.0, read=900.0, write=60.0, pool=5.0),
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        say(f"could not reload at num_ctx={config.num_ctx}: {exc}")
        return report
    return await run_check(config, prompt)


async def _load_and_recheck(
    config: PharosConfig,
    prompt: str,
    report: CheckReport,
    say: Callable[[str], None],
) -> CheckReport:
    """No model resident is a fixable state, not a refusal — load the configured one and retry.

    Only when the backend is actually reachable and a model is named: a run must never invent
    a window, so an unreachable backend or an unnamed model still comes back indeterminate.
    The load goes straight to the backend rather than through the proxy, because it is a
    control action rather than the agent's own traffic and does not belong in the dashboard's
    request history.
    """
    profile = report.profile
    reachable = profile.backend.reachable if profile is not None else False
    if not reachable or not config.model:
        return report

    say(f"no model resident — loading {config.model} (a large model takes a while the first time)")
    payload = _load_payload(config)
    try:
        async with httpx.AsyncClient(base_url=config.backend_url) as client:
            response = await client.post(
                "/api/chat", json=payload, timeout=httpx.Timeout(connect=5.0, read=900.0,
                                                                 write=60.0, pool=5.0)
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        say(f"could not load {config.model}: {exc}")
        return report

    say(f"loaded {config.model}")
    return await run_check(config, prompt)


async def run_task(
    config: PharosConfig,
    prompt: str,
    *,
    dry_run: bool = False,
    use_git: bool = True,
    divide: bool = True,
    on_event: Callable[[str], None] | None = None,
) -> RunOutcome:
    """Pre-flight, divide and execute ``prompt`` in the configured workspace.

    ``divide=False`` runs the whole task as one undivided conversation. It exists to be the
    control in a comparison: same model, same tools, same task, only the division removed. A
    claim that splitting rescues a task nobody watched fail is not worth much.
    """
    say = on_event or (lambda _message: None)
    outcome = RunOutcome()

    root = workspace_root(config.target_folder)
    workspace = Workspace(root)

    report = await run_check(config, prompt)
    outcome.report = report
    outcome.counts_exact = report.counts_exact

    if report.verdict is Verdict.INDETERMINATE:
        report = await _load_and_recheck(config, prompt, report, say)
        outcome.report = report
        outcome.counts_exact = report.counts_exact
    if report.verdict is Verdict.INDETERMINATE:
        outcome.error = (
            f"no verdict — {report.verdict_detail}. A run cannot promise to stay inside a "
            f"window nobody can measure."
        )
        return outcome

    report = await _ensure_window(config, prompt, report, say)
    outcome.report = report
    outcome.counts_exact = report.counts_exact
    budget = report.profile.budget if report.profile is not None else None
    usable = budget.usable_budget if budget is not None else None
    if usable is None:
        outcome.error = "no usable budget could be determined for the loaded model."
        return outcome

    # What this client itself costs, measured now so the plan and the session agree on it.
    count, _exact = _counter(config)
    overhead = agent_overhead_tokens(workspace, count)
    outcome.catalogue_tokens = overhead

    # One part or many, the execution path is identical — a task that fits is simply a plan
    # of one, so there is no separate untested branch for the common case.
    plan = build_plan(
        config,
        prompt,
        report,
        target=_edit_target(config, usable, overhead),
        # Two limits, one plan: the window, and how many files a model gets through in one
        # sitting. Only the first is measurable; see PharosConfig.max_files_per_part.
        max_files=config.max_files_per_part,
    )
    outcome.plan = plan
    bodies: list[tuple[str, list[PartFile] | None]]
    outcome.divided = divide
    outcome.handoff_reserve = config.handoff_reserve
    if not divide:
        say("running undivided (--no-split): one conversation, no scope enforcement")
        bodies = [(prompt, None)]
    elif plan.already_fits:
        bodies = [(prompt, None)]
    elif plan.ok and plan.mode is not SplitMode.NONE:
        bodies = _bodies(plan)
    else:
        loaded = report.profile.backend.loaded_ctx if report.profile else None
        advice = ""
        if loaded and loaded <= 8192:
            advice = (
                f" The loaded window is only {loaded:,} tokens, which is the backend's "
                f"cautious default rather than what this card can hold — set num_ctx in "
                f"pharos.toml (16384 is a reasonable start) and run again."
            )
        outcome.error = f"no plan could be built — {plan.reason or 'unknown reason'}.{advice}"
        return outcome

    if dry_run:
        say(f"dry run — {len(bodies)} part(s) planned, nothing executed")
        return outcome

    undo: Undo | None = None
    if use_git:
        try:
            outcome.base_branch = current_branch(root)
            outcome.branch = git_guard(root)
            if outcome.branch:
                say(f"working on branch {outcome.branch}")
        except NotARepository:
            # Not a repository is not a reason to refuse — it is a reason to bring an undo of
            # our own. A dirty REPOSITORY still stops the run; that exception is not caught.
            stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            undo = Undo(root, root / ".pharos" / f"undo-{stamp}")
            outcome.undo = undo
            say(f"no git here — originals will be copied to {undo.directory} before any write")

    proxy_url = f"http://{config.proxy_host}:{config.proxy_port}"

    async with httpx.AsyncClient(base_url=proxy_url) as probe:
        outcome.via_proxy = await _reachable(probe)
    base_url = proxy_url if outcome.via_proxy else config.backend_url
    say(
        f"routing through the Pharos proxy at {proxy_url}"
        if outcome.via_proxy
        else f"proxy not running — going straight to {config.backend_url}, so this run will "
        f"not appear in the dashboard"
    )

    model = report.model or config.model
    if not model:
        outcome.error = "no model to run against; set `model` in pharos.toml."
        return outcome

    handoff: str | None = None
    async with httpx.AsyncClient(base_url=base_url) as client:

        async def run_part(
            label: str, body: str, files: list[PartFile] | None, index: int, carried: str | None
        ) -> PartResult:
            """One part, start to finish. The repair pass goes through here too, so a repaired
            file is bounded, scoped and scored on exactly the same terms as a planned one."""
            toolbox = ToolBox(
                workspace=workspace,
                scope=scope_from_part_files(files) if files else None,
                undo=undo,
            )
            session = AgentSession(
                client=client,
                model=model,
                toolbox=toolbox,
                count=count,
                usable_budget=usable,
                handoff_reserve=config.handoff_reserve,
                num_ctx=config.num_ctx,
                on_event=_prefixed(say, label),
            )
            text = _with_handoff(body, carried, index, files)
            _logger.info("%s body: %s", label, text)
            result = await session.run(text)
            _logger.info(
                "%s finished: steps=%d wrote=%s handoff=%r",
                label, result.steps, result.files_written, result.text,
            )
            # The pairs behind the drift figure, so an outlier can be identified instead of
            # theorised about. The scorecard reduces these to a range and a shortfall; when
            # the range is wide the only useful question is which request did it, and that
            # needs the absolute numbers, not the ratio.
            if result.drift_samples:
                _logger.info(
                    "%s drift (ours, backend): %s",
                    label,
                    ", ".join(f"({a}, {b})" for a, b in result.drift_samples),
                )
            outcome.parts.append(result)
            return result

        failed = False
        for index, (body, files) in enumerate(bodies, start=1):
            say(f"part {index} of {len(bodies)}")
            result = await run_part(f"part {index}", body, files, index, handoff)
            if result.error:
                say(f"  [part {index}] failed: {result.error}")
                failed = True
                break
            carried, cut = (
                _cap_handoff(result.text, config.handoff_reserve, count)
                if result.text
                else ("", False)
            )
            if cut:
                say(
                    f"  [part {index}] hand-off cut to the "
                    f"{config.handoff_reserve:,}-token reserve"
                )
            handoff = carried or None

        # A repair pass over whatever the plan assigned and no part actually changed.
        #
        # Not persuasion, and not a retry of a failed call: those files were somebody's scope
        # and were left alone, usually because the part that owned them decided it had
        # finished. A fresh conversation holding only the leftovers is the same medicine that
        # took coverage from 46% to 92% — parts finish about two files, so give a part two
        # files. Deliberately ONE round: a second would be chasing a model that has declined
        # the same work twice, and an unbounded repair loop is how a run stops having a
        # knowable cost.
        if not failed and divide and config.repair_pass:
            missed = _untouched(outcome.parts)
            if missed:
                say(f"{len(missed)} file(s) the plan assigned were never changed - repairing")
                chunks = _chunks(missed, config.max_files_per_part)
                for offset, chunk in enumerate(chunks, start=1):
                    label = f"repair {offset} of {len(chunks)}"
                    say(label)
                    outcome.repair_parts += 1
                    result = await run_part(label, _repair_body(prompt, chunk), chunk, 1, None)
                    if result.error:
                        say(f"  [{label}] failed: {result.error}")
                        break

    if undo is not None:
        outcome.files_changed = sorted(undo.saved)
    else:
        outcome.files_changed = changed_files(root) if use_git else []
    return outcome


def _edit_target(config: PharosConfig, usable: int, overhead: int) -> int:
    """Tokens of content one part may hold — half what `pharos split` would allow, on purpose.

    `pharos split` sizes a part for a prompt that gets PASTED: each file's content appears in
    the window exactly once. An agentic run pays for the same file twice — once in the tool
    result that reads it, and again in the assistant turn that writes it back — so a part
    packed to the pasting target is on course to hit its ceiling partway through the first
    edit, every time. Halving the content budget is the mechanical correction, and it is why
    a run can report more parts than `pharos split` shows for the identical task.

    Deliberately crude. A task that only reads pays double for nothing, which costs parts but
    never correctness; the alternative is asking a model to predict how much it intends to
    write, and that is the kind of guess Pharos does not make.

    ``overhead`` is this client's own system prompt and tool catalogue. It has to come out
    here as well as in the session, or the plan hands the session parts it cannot afford —
    the planner promising room the executor has already spent. Unlike the proxy's learned
    figure for a third-party agent, this one is counted exactly.
    """
    return max((usable - config.handoff_reserve - SAFETY_MARGIN - overhead) // 2, 1)


def _prefixed(say: Callable[[str], None], label: str) -> Callable[[str], None]:
    """Tag a session's commentary with the part it came from."""

    def emit(message: str) -> None:
        say(f"  [{label}] {message}")

    return emit


def _untouched(parts: list[PartResult]) -> list[PartFile]:
    """Files some part was given and no part changed, in the order they were assigned."""
    written = {normalise(path) for part in parts for path in part.files_written}
    missed: dict[str, PartFile] = {}
    for part in parts:
        for path in part.scoped:
            key = normalise(path)
            if key not in written and key not in missed:
                missed[key] = PartFile(display=path, tokens=0)
    return list(missed.values())


def _chunks(files: list[PartFile], size: int) -> list[list[PartFile]]:
    """Split the leftovers into parts of the same size the planner uses."""
    step = max(size, 1)
    return [files[i : i + step] for i in range(0, len(files), step)]


def _repair_body(prompt: str, files: list[PartFile]) -> str:
    """A part whose whole job is work an earlier part was given and did not do.

    It says so plainly rather than posing as a fresh assignment. A model told these were
    missed has a reason to look; one told they are simply "the task" may conclude, exactly as
    the earlier part did, that the job is already complete.
    """
    listed = chr(10).join(f"  - {f.display}" for f in files)
    return (
        "[Pharos] REPAIR PASS. An earlier part of this run was given the files below and left "
        "them unchanged on disk. Nothing else is in scope for you." + chr(10) * 2
        + listed + chr(10) * 2
        + "Read each one and apply the task to it. Do not assume it is already done - it is "
        "not, and that is why you are seeing it. If a file genuinely needs no change, say "
        "which and why." + chr(10) * 2
        + "--- TASK ---" + chr(10) + prompt.strip()
    )


def _bodies(plan: SplitPlan) -> list[tuple[str, list[PartFile] | None]]:
    """(part body, scoped files) pairs. Text-mode parts carry no file scope to enforce."""
    return [
        (part.body, part.files if plan.mode is SplitMode.SCOPE else None) for part in plan.parts
    ]


def _cap_handoff(handoff: str, reserve: int, count: Callable[[str], int]) -> tuple[str, bool]:
    """Hold a hand-off to the room the plan reserved for it.

    Every part's ceiling was computed with ``handoff_reserve`` subtracted for exactly this
    text. A real run produced one of 1,804 tokens against a 500-token reserve and forwarded it
    whole, so the next part ran under a projection that was wrong by 1,300 tokens before it
    read anything.

    Cut from the END, and say so in the text. The opening of a hand-off is what it did; the
    tail is elaboration, so a reader — human or model — loses the least that way. Silently
    truncating would be the context loss this project refuses, which is why the marker is
    part of the forwarded text rather than only a line in the report.
    """
    if reserve <= 0 or count(handoff) <= reserve:
        return handoff, False
    # Cut by characters against a token budget, then walk back until it actually fits.
    kept = handoff
    while kept and count(kept) > reserve:
        kept = kept[: int(len(kept) * 0.8)] if len(kept) > 40 else ""
    marker = (
        f"\n[Pharos] The rest of this hand-off was cut: it was {count(handoff):,} tokens "
        f"against the {reserve:,} reserved for it, and the part below was planned around that "
        f"reserve. Ask for what is missing if you need it."
    )
    return kept.rstrip() + marker, True


def _with_handoff(
    body: str, handoff: str | None, index: int, files: list[PartFile] | None = None
) -> str:
    """The prompt a part actually receives: what it inherits, what it owes, and its scope.

    Two things the splitter's body does not say, both learned from watching runs.

    A hand-off pasted above a task reads as a verdict on that task. Part 1 reported the work
    complete and parts 2, 3 and 4 did nothing at all — correctly, by their reading. It is
    labelled as context about OTHER files, and the point is repeated after it, because the
    last thing read is the thing obeyed.

    And a list of files reads as material, not as a checklist. Runs settle around half the
    scope and stop, satisfied. The reminder that catches that only fires AFTER the model has
    decided it is finished, which is late and costs a round trip; stating the count up front
    is the cheap half of the same idea. The count, not the names — the names are in the scope
    block immediately below, and repeating them would spend the one budget this project exists
    to protect.
    """
    sections: list[str] = []
    if handoff and index > 1:
        sections.append(
            f"--- HAND-OFF FROM PART {index - 1} (context only) ---\n{handoff.strip()}\n"
            f"--- END HAND-OFF ---\n"
            f"That hand-off describes work on DIFFERENT files. The files scoped to you below "
            f"have NOT been done yet — do them now, whatever it says about progress."
        )
    if files:
        total = len(files)
        subject = "file" if total == 1 else "files"
        sections.append(
            f"YOUR TARGET: {total} {subject}. This part is not finished until every one of the "
            f"{total} in-scope {subject} listed below has been CHANGED on disk. Work through "
            f"them one at a time — read it, edit it, then move to the next — and do not stop "
            f"after the first. If one genuinely needs no change, say which and why; do not "
            f"skip it silently."
        )
    sections.append(body)
    return "\n\n".join(sections)


def workspace_for(config: PharosConfig) -> Path:
    return workspace_root(config.target_folder)
