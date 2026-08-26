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

Between parts two things are carried forward, together bounded by ``handoff_reserve``: the
hand-off the previous part wrote, and Pharos's own record of what has actually landed on disk.
The second exists because the first is unreliable in a measured way — parts change files and
then report "NO CHANGES NEEDED" — and Pharos does not have to take a model's word for what a
model just did. Each part otherwise starts from an empty conversation. That is what keeps part
5 as affordable as part 1, and it is the difference between this and an agent that simply runs
until it dies.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from pharos.agent.audit import (
    PartAudit,
    TreeIndex,
    audit_part,
    diff_trees,
    index_tree,
    own_paths,
)
from pharos.agent.ledger import LEDGER_SHARE, Ledger
from pharos.agent.review import Finding, Review, ask, batches, collect, parse, validate
from pharos.agent.session import SAFETY_MARGIN, AgentSession, PartResult
from pharos.agent.tools import (
    ToolBox,
    agent_overhead_tokens,
    normalise,
    scope_from_part_files,
    workspace_root,
)
from pharos.agent.verify import (
    CheckOutcome,
    Damage,
    DamageWatch,
    Verification,
    baseline,
    detect_commands,
    run_command,
    syntax_state,
    verify,
)
from pharos.agent.workspace import (
    NotARepository,
    Undo,
    Workspace,
    changed_files,
    current_branch,
    git_guard,
)
from pharos.calibration import (
    TEMPLATE_OFFSET_CAP,
    TemplateCost,
    remember_run,
    remembered_cost,
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
    # Pharos's record of the run, carried between parts. `ledger_files` counts the files some
    # later part was actually told about, not everything the run eventually changed.
    ledger_on: bool = False
    ledger_files: int = 0
    # Which part broke what, as each part finished. The end-of-run checks say whether the
    # project is broken; this says by whom, which on a five-part run is the more useful half.
    damage: list[Damage] = field(default_factory=list)
    stopped_on_break: str | None = None  # the part that ended the run by breaking something
    # Every file the PLAN assigned, whether or not its part ever ran. Coverage is measured
    # against this rather than against the parts that executed: a run halted after part 1 of
    # three otherwise reports 100%, because the four files nobody attempted are not in any
    # part's scope. True of a run stopped by an error too, and has been since v0.4.
    planned_files: list[str] = field(default_factory=list)
    # What earlier runs measured this model's chat template to cost, and whether any request
    # still went out with nothing correcting its ceiling.
    # What was IN FORCE during this run, and what the memory holds after it. Two different
    # facts: the first says how this run's ceilings were computed, the second what the next
    # run will start from. On a model's first run the seed is None and the memory is not.
    template_cost: TemplateCost | None = None
    template_learned: TemplateCost | None = None
    exposed_requests: int = 0
    via_proxy: bool = True
    # What the DISK says each part did, against what the part said it did. Empty when the
    # audit was turned off; see `pharos.agent.audit`.
    audits: list[PartAudit] = field(default_factory=list)
    # One model's opinion of the diff, when --review asked for one. Deliberately the last
    # thing computed and the last thing printed: nothing above it may depend on it.
    review: Review | None = None
    # What the project's own checks said afterwards. None when verification was off or
    # the run never got as far as executing anything.
    verification: Verification | None = None
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
    exclude: list[str] | None = None,
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
    return await run_check(config, prompt, exclude=exclude)


async def _load_and_recheck(
    config: PharosConfig,
    prompt: str,
    report: CheckReport,
    say: Callable[[str], None],
    exclude: list[str] | None = None,
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
    return await run_check(config, prompt, exclude=exclude)


def written_by_parts(parts: list[PartResult]) -> list[str]:
    """Every path the dispatcher recorded itself writing, deduplicated.

    Pharos's own account of what it did, not the model's: ``ToolBox.files_written`` is appended
    after a write succeeds. Weaker than a diff — a write that restored a file's original bytes
    still counts — and the only answer available when there is no git and no snapshot.
    """
    return sorted({normalise(path) for part in parts for path in part.files_written})


async def run_task(
    config: PharosConfig,
    prompt: str,
    *,
    dry_run: bool = False,
    use_git: bool = True,
    verify_work: bool = True,
    divide: bool = True,
    semantic: bool = False,
    reserve_reads: bool = False,
    compact: bool = False,
    audit: bool = True,
    review: bool = False,
    exclude: list[str] | None = None,
    use_ledger: bool = True,
    stop_on_break: bool = False,
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

    report = await run_check(config, prompt, exclude=exclude)
    outcome.report = report
    outcome.counts_exact = report.counts_exact

    if report.verdict is Verdict.INDETERMINATE:
        report = await _load_and_recheck(config, prompt, report, say, exclude=exclude)
        outcome.report = report
        outcome.counts_exact = report.counts_exact
    if report.verdict is Verdict.INDETERMINATE:
        outcome.error = (
            f"no verdict — {report.verdict_detail}. A run cannot promise to stay inside a "
            f"window nobody can measure."
        )
        return outcome

    report = await _ensure_window(config, prompt, report, say, exclude=exclude)
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
        # Not when the parts are being thrown away. --no-split still builds a plan, so the
        # report can say what it declined, but paying a backend call to arrange parts nobody
        # will run is a round trip for a footnote.
        semantic=semantic and divide,
        # Size the parts against what agents opened unprompted on past runs, not only against
        # what this part names. Off by default: it is the one input to a plan that is a
        # prediction rather than a measurement of the part in front of it.
        reserve_reads=reserve_reads and divide,
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
        outcome.error = f"no plan could be built — {plan.reason or _why_no_plan(plan)}.{advice}"
        return outcome

    outcome.planned_files = [f.display for _, files in bodies if files for f in files]

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

    # The checks, and their answers BEFORE anything is written. Taken here because a moment
    # later the model starts editing, and a baseline captured after that measures nothing.
    checks: list[str] = []
    was_passing: dict[str, CheckOutcome] = {}
    parses_now: dict[str, str | None] = {}
    per_part_checks: list[str] = []
    if config.verify and verify_work:
        checks = (
            list(config.verify_commands)
            if config.verify_commands is not None
            else detect_commands(root)
        )
        parses_now = syntax_state(root, outcome.planned_files)
        if checks:
            say(f"baseline: {', '.join(checks)}")
            was_passing = await asyncio.to_thread(
                baseline, root, checks, timeout=config.verify_timeout_seconds
            )
            for command, before in was_passing.items():
                if before.skipped is not None:
                    say(f"  {command} could not be baselined ({before.skipped})")
                elif not before.ok:
                    say(f"  {command} was already failing - it will not be charged to this run")
            per_part_checks = _cheap_enough(was_passing, config.per_part_check_seconds)
            if per_part_checks:
                spent = sum(was_passing[c].duration for c in per_part_checks)
                say(
                    f"  running after every part as well: {', '.join(per_part_checks)} "
                    f"({spent:.1f}s the baseline took)"
                )

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

    # Parse what each part wrote the moment it finishes, against the state immediately
    # before it. Costs milliseconds beside the minute a part takes, and it is the only way to
    # name the part responsible: the end-of-run check knows the project is broken and cannot
    # know who did it.
    watch = (
        DamageWatch(root, parses_now, was_passing)
        if (config.verify and verify_work)
        else None
    )
    halt_on_break = stop_on_break and watch is not None

    # What previous runs learned about this model's chat template. A part starts with no
    # responses of its own, so without this its first request is enforced against a ceiling
    # that has only SAFETY_MARGIN behind it -- and that constant measured 288-297 tokens short
    # on nine live runs. Seeding costs nothing and is not a new measurement: it is the same
    # correction the session already makes for itself, one request earlier.
    memory = Path(config.template_memory_file)
    known = remembered_cost(memory, model)
    outcome.template_cost = known
    seed = known.offset if known else None
    if known is not None:
        say(
            f"template memory: {model} counted {known.offset:,} tokens above our projection "
            f"over {known.runs} previous run(s) — every part's ceiling starts corrected"
        )
        if known.capped:
            say(
                f"  that measurement wanted more than the {TEMPLATE_OFFSET_CAP:,}-token cap "
                f"and was held to it — a gap that size is not template scaffolding"
            )

    handoff: str | None = None
    # Pharos's own account of the run, appended to as each write lands. Off it goes back to
    # v0.5 behaviour, where the model's prose was the only thread between parts — which is
    # what the coverage figures in DESKTOP_VALIDATION were measured against.
    record = Ledger()
    # label -> (tree before the part, tree after its tools). Closed out once the project's own
    # checks have run, because a check command that rewrites files is doing so during the run
    # and nothing before v1.0 noticed.
    pending: dict[str, tuple[TreeIndex, TreeIndex]] = {}
    # What Pharos itself writes into the workspace, so the audit stops reporting its own log
    # as a change nobody claimed. The undo directory goes in too: it exists to hold a copy of
    # every original the run is about to overwrite, which is the largest false finding
    # available. Computed once -- none of these move during a run.
    ours = own_paths(config, root, *([undo.directory] if undo is not None else []))

    def close_audit(label: str, result: PartResult) -> None:
        snapshots = pending.pop(label, None)
        if snapshots is None:
            return
        before, after_tools = snapshots
        after_checks = index_tree(root, ignore=ours)
        outcome.audits.append(
            audit_part(
                label,
                changed=diff_trees(before, after_tools),
                claimed=list(result.files_written),
                scoped=list(result.scoped) or None,
                by_checks=diff_trees(after_tools, after_checks),
                truncated=before.truncated or after_tools.truncated,
            )
        )

    ledger_on = config.handoff_ledger and use_ledger and divide
    outcome.ledger_on = ledger_on
    ledger_text = ""

    async with httpx.AsyncClient(base_url=base_url) as client:

        async def run_part(
            label: str,
            body: str,
            files: list[PartFile] | None,
            index: int,
            carried: str | None,
            already_done: str = "",
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
                template_offset=seed,
                # Reclaim the window a part filled with files it has finished with, rather
                # than stopping it there. Off unless asked: it changes what the model can see
                # of its own history, and everything else a part does is additive.
                compact=compact,
                on_event=_prefixed(say, label),
            )
            text = _with_handoff(body, carried, index, files, already_done)
            _logger.info("%s body: %s", label, text)
            before = index_tree(root, ignore=ours) if audit else TreeIndex()
            result = await session.run(text)
            if audit:
                # Taken here, BEFORE the checks run, so this diff is the part's own doing.
                # What the checks themselves write is a separate question, asked below.
                pending[label] = (before, index_tree(root, ignore=ours))
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
            if ledger_text:
                outcome.ledger_files = max(outcome.ledger_files, len(record.files))
            result = await run_part(f"part {index}", body, files, index, handoff, ledger_text)
            # Parse BEFORE the error check. A part that wrote a broken file and then died on
            # the next call still broke that file: the write landed, and attributing it to
            # nobody because the part failed afterwards would lose the one fact worth having
            # about the failure.
            broke = (
                await _broke(
                    watch, f"part {index}", result.files_written, say, outcome,
                    root=root, checks=per_part_checks,
                    check_timeout=config.verify_timeout_seconds,
                )
                if watch is not None
                else False
            )
            close_audit(f"part {index}", result)
            if result.error:
                say(f"  [part {index}] failed: {result.error}")
                failed = True
                break
            if broke and halt_on_break:
                outcome.stopped_on_break = f"part {index}"
                say(
                    f"  [part {index}] stopping here (--stop-on-break): the parts after this "
                    f"one would inherit a tree that does not parse"
                )
                break
            record.record(f"part {index}", result.changes)
            ledger_text, carried, cut, prose_reserve = _carry(
                record, result.text, config.handoff_reserve, count, ledger=ledger_on
            )
            if cut:
                say(f"  [part {index}] hand-off cut to the {prose_reserve:,} tokens left of "
                    f"the {config.handoff_reserve:,}-token reserve")
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
        # Not after a stop-on-break either. The sweep exists to finish work the plan left
        # undone, and running it over a tree that no longer parses is the compounding the
        # flag was set to prevent.
        if not failed and outcome.stopped_on_break is None and divide and config.repair_pass:
            missed = _untouched(outcome.parts)
            if missed:
                say(f"{len(missed)} file(s) the plan assigned were never changed - repairing")
                chunks = _chunks(missed, config.max_files_per_part)
                for offset, chunk in enumerate(chunks, start=1):
                    label = f"repair {offset} of {len(chunks)}"
                    say(label)
                    outcome.repair_parts += 1
                    # The record goes to repair parts as well. Their scope is by definition
                    # the files NOT in it, so nothing is withheld from them, and a sweep that
                    # knows the conventions the run has already settled on writes code that
                    # matches. Without it a repair part starts completely blind.
                    if ledger_text:
                        outcome.ledger_files = max(outcome.ledger_files, len(record.files))
                    result = await run_part(
                        label, _repair_body(prompt, chunk), chunk, 1, None, ledger_text
                    )
                    if watch is not None:
                        await _broke(
                            watch, label, result.files_written, say, outcome,
                            root=root, checks=per_part_checks,
                            check_timeout=config.verify_timeout_seconds,
                        )
                    close_audit(label, result)
                    record.record(label, result.changes)
                    ledger_text, _, _, _ = _carry(
                        record, "", config.handoff_reserve, count, ledger=ledger_on
                    )
                    if result.error:
                        say(f"  [{label}] failed: {result.error}")
                        break

    if undo is not None:
        outcome.files_changed = sorted(undo.saved)
    elif use_git:
        outcome.files_changed = changed_files(root)
    else:
        # --no-git, so there is no diff to ask and no snapshot to list. This used to be an
        # empty list, which was not "we do not know" but a positive claim that nothing was
        # written — printed as "Nothing was written" under a run that had just correctly
        # edited both its files. Worse, it went to the verifier as the set of written files,
        # so --no-git quietly turned the syntax check off: a run could break every file in
        # the tree and be told there was nothing in a format it could parse.
        #
        # The dispatcher records each path as it writes it, so this is Pharos's own account
        # of what it did rather than the model's. Weaker than a diff — a write that restored
        # a file's original bytes still counts here — and much better than silence.
        outcome.files_changed = written_by_parts(outcome.parts)

    outcome.exposed_requests = sum(p.exposed_requests for p in outcome.parts)
    # Fold this run's measurements back in, so the next one starts corrected. One number per
    # run -- its worst ratio -- so a long run cannot outvote a short one.
    gaps = [
        theirs - ours
        for part in outcome.parts
        for ours, theirs in part.drift_samples
        if ours > 0 and theirs > 0
    ]
    if gaps:
        outcome.template_learned = remember_run(memory, model, gaps)

    if config.verify and verify_work:
        say("verifying")
        outcome.verification = await asyncio.to_thread(
            verify,
            root,
            written=outcome.files_changed,
            commands=checks,
            command_baseline=was_passing,
            syntax_baseline=parses_now,
            timeout=config.verify_timeout_seconds,
        )
        for check in outcome.verification.newly_broken:
            say(f"  {check.name} FAILED - it passed before this run")

    # LAST, and after the verdict is already settled. Everything above this line is a
    # measurement; what follows is an opinion, and it is not allowed to touch any of them.
    if review and not outcome.files_changed:
        # Asked for and not silently skipped. A flag that produces no output when the answer
        # is "there was nothing to look at" is indistinguishable from one that failed.
        outcome.review = Review(note="the run changed no files, so there was no diff to read")
    elif review:
        say("reviewing the diff (an opinion — it changes nothing above)")
        outcome.review = await asyncio.to_thread(
            review_run,
            config,
            root,
            model,
            outcome.files_changed,
            count,
            usable,
            undo=undo,
        )
        found = outcome.review
        if found is not None and found.findings:
            say(f"  {len(found.findings)} finding(s) survived checking")
    return outcome


def review_run(
    config: PharosConfig,
    root: Path,
    model: str,
    changed: list[str],
    count: Callable[[str], int],
    usable: int,
    *,
    undo: Undo | None = None,
) -> Review:
    """Collect the run's diff, put it to the backend in batches, and check what comes back.

    Every failure lands in the same place: a Review carrying a note and no findings. A review
    that could not happen must read as a review that could not happen, never as a clean bill
    of health -- the two are opposite conclusions and would print almost identically.
    """
    originals = (
        {key: undo.original(key) for key in undo.saved} if undo is not None else None
    )
    diffs = collect(root, changed, originals=originals)
    if not diffs:
        return Review(
            unreviewed=list(changed),
            note="no diff could be read for the files this run changed",
        )
    # The same halving the planner uses, for the same reason: the diff goes in and the
    # findings come back out of the same window.
    budget = max(usable // 2, 1)
    packed, oversized = batches(diffs, budget, count)
    findings: list[Finding] = []
    reviewed: list[str] = []
    discarded = 0
    problems: list[str] = []
    for batch in packed:
        reply, error = ask(config, batch, model)
        if error is not None or reply is None:
            problems.append(error or "no reply")
            continue
        raw, parse_error = parse(reply)
        if parse_error is not None:
            problems.append(parse_error)
            continue
        kept, dropped = validate(raw, batch)
        findings.extend(kept)
        discarded += dropped
        reviewed.extend(diff.display for diff in batch)
    note = None
    if oversized:
        note = (
            f"{len(oversized)} file(s) had a diff too large to review in one call and were "
            f"left out rather than shown in half: {', '.join(oversized[:3])}"
        )
    if problems:
        detail = "; ".join(sorted(set(problems))[:2])
        note = f"{note + ' — ' if note else ''}{len(problems)} batch(es) failed: {detail}"
    return Review(
        findings=findings,
        reviewed=sorted(set(reviewed)),
        unreviewed=sorted(set(oversized)),
        discarded=discarded,
        note=note,
    )


def _why_no_plan(plan: SplitPlan) -> str:
    """Explain a plan that EXISTS and does not fit.

    ``reason`` is set only when no plan could be built at all. A scope plan whose parts came
    out over budget explains itself through the parts, and the runner was rendering that case
    as "unknown reason" — the one thing this project may not say about its own refusal. Seen
    live on a giant prompt: eighteen parts, nine of them over, and a message that named none
    of it.
    """
    over = [part for part in plan.parts if not part.fits]
    if over:
        worst = max(over, key=lambda part: part.over_by)
        which = ", ".join(str(part.index) for part in over[:4])
        more = f" and {len(over) - 4} more" if len(over) > 4 else ""
        detail = (
            f"{len(over)} of {len(plan.parts)} part(s) came out over the "
            f"{plan.target_per_part:,}-token {plan.target_label} — part {which}{more}, the "
            f"worst by {worst.over_by:,} tokens"
        )
    elif not plan.parts:
        detail = "the planner produced no parts at all"
    else:
        detail = "the plan was rejected with no cause recorded, which is itself a bug"
    notes = "; ".join(plan.notes[:2])
    return f"{detail}{' — ' + notes if notes else ''}"


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


def _cheap_enough(measured: dict[str, CheckOutcome], budget: float) -> list[str]:
    """Which checks are quick enough to also run between parts, decided by measurement.

    The baseline has just run every one of them, so how long each takes on this machine and
    this repository is already known and costs nothing to read. Asking the user instead would
    be asking them to guess a number about their own suite that Pharos has the answer to.

    A check that could not be baselined is excluded whatever its duration: without a "before"
    there is nothing to compare a per-part result against, and reporting a failure with no
    baseline as damage would name a part for a state it may have inherited.
    """
    return [
        name
        for name, outcome in measured.items()
        if outcome.skipped is None and outcome.duration <= budget
    ]


async def _broke(
    watch: DamageWatch,
    label: str,
    written: list[str],
    say: Callable[[str], None],
    outcome: RunOutcome,
    *,
    root: Path,
    checks: list[str],
    check_timeout: float,
) -> bool:
    """Check what this part did, say what it broke, and keep the running tally.

    The parser first, always. Then whichever project checks the baseline measured as cheap
    enough, off the event loop so a run in the TUI keeps drawing while they execute.
    """
    ran: dict[str, CheckOutcome] = {}
    if checks and written:
        # No writes means nothing this part could have broken, and the checks would only
        # re-measure the previous part's answer at the price of running them again.
        ran = await asyncio.to_thread(
            lambda: {c: run_command(root, c, timeout=check_timeout) for c in checks}
        )
    found = watch.after_part(label, written, ran)
    for entry in found:
        say(f"  [{label}] BROKE {entry.subject}: {_one_line(entry.error)}")
    outcome.damage = watch.damage
    return bool(found)


def _one_line(detail: str) -> str:
    """A tool's excerpt is several lines; a progress line is one."""
    first = next((line for line in detail.splitlines() if line.strip()), "")
    return first if len(first) <= 120 else first[:119] + chr(8230)


def _carry(
    record: Ledger,
    prose: str,
    reserve: int,
    count: Callable[[str], int],
    *,
    ledger: bool,
) -> tuple[str, str, bool, int]:
    """What the next part inherits, and the arithmetic that keeps it inside one reserve.

    Two things cross the gap between parts and they share a single budget, because that budget
    is what every part's ceiling was computed against: adding the record on top of the reserve
    rather than inside it would make each part smaller than the plan promised, silently, which
    is the failure this project exists to expose.

    The record goes first and is capped at ``LEDGER_SHARE`` of the reserve. First because it
    is the half that cannot be wrong; capped because the model's hand-off carries intent no
    mechanical record reconstructs, and starving it to fit a long list of filenames would
    trade the irreplaceable half for the reproducible one.

    Returns the record, the hand-off, whether the hand-off was cut, and what it was cut to.
    """
    text = record.render(count=count, budget=int(reserve * LEDGER_SHARE)) if ledger else ""
    prose_reserve = max(reserve - count(text), 0)
    carried, cut = _cap_handoff(prose, prose_reserve, count) if prose else ("", False)
    return text, carried, cut, prose_reserve


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

    The marker is counted against the reserve too, which it was not until v0.6. Trimming the
    prose to exactly the reserve and then appending forty tokens of explanation overran the
    budget by forty tokens on every cut hand-off — the same failure the function exists to
    prevent, committed by the fix for it, and invisible because nothing measured the total
    that crossed the gap. The record carried beside the hand-off now does measure it.
    """
    if reserve <= 0 or count(handoff) <= reserve:
        return handoff, False
    marker = (
        f"\n[Pharos] The rest of this hand-off was cut: it was {count(handoff):,} tokens "
        f"against the {reserve:,} reserved for it, and the part below was planned around that "
        f"reserve. Ask for what is missing if you need it."
    )
    room = reserve - count(marker)
    if room <= 0:
        # No room to explain the cut at length. Say the shortest true thing instead: that
        # there was a hand-off and it did not fit. Dropping it silently would leave the next
        # part unable to tell an empty hand-off from a discarded one.
        brief = f"[Pharos] Hand-off dropped: {count(handoff):,} tokens against {reserve:,}."
        return (brief, True) if count(brief) <= reserve else ("", True)
    # Cut by characters against a token budget, then walk back until it actually fits.
    kept = handoff
    while kept and count(kept) > room:
        kept = kept[: int(len(kept) * 0.8)] if len(kept) > 40 else ""
    return kept.rstrip() + marker, True


def _with_handoff(
    body: str,
    handoff: str | None,
    index: int,
    files: list[PartFile] | None = None,
    already_done: str = "",
) -> str:
    """The prompt a part actually receives: what it inherits, what it owes, and its scope.

    Three things the splitter's body does not say, all learned from watching runs.

    A hand-off pasted above a task reads as a verdict on that task. Part 1 reported the work
    complete and parts 2, 3 and 4 did nothing at all — correctly, by their reading. It is
    labelled as context about OTHER files, and the point is repeated after it, because the
    last thing read is the thing obeyed.

    And that hand-off is a model's account of its own work, which is exactly the claim least
    worth taking on trust: measured across sixteen runs, parts changed files and then reported
    "NO CHANGES NEEDED". So Pharos's own record of what landed goes in underneath it, and it
    goes SECOND on purpose — where the two disagree, the one read last is the one obeyed, and
    the one read last is the one that cannot be wrong.

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
    if already_done:
        sections.append(already_done)
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
