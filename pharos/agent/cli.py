"""`pharos run` — the command line around the runner.

Exit codes match the rest of the toolchain: 0 the run covered its scope and no part failed ·
1 it did not · 2 indeterminate (no model resident, backend unreachable, a config error, or a
repository too dirty to write to safely).

Zero means the WORK was done, not that the process survived. A run whose parts all report
"done" while three of twenty files were never written has not succeeded, and exiting 0 on it
would make the code useless as a gate and would flatter exactly the failure this tool exists
to expose. It still says nothing about whether the edits are correct — see the scorecard.

The running commentary goes to stderr and the summary to stdout, so `pharos run ... > log`
keeps the report and still shows progress live.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from pharos.agent.audit import FILE_CAP
from pharos.agent.review import SEVERITIES, Review
from pharos.agent.runner import RunOutcome, first_line, run_task
from pharos.agent.scorecard import Scorecard, score, to_dict
from pharos.agent.session import SAFETY_MARGIN, PartResult
from pharos.agent.tools import workspace_root
from pharos.agent.verify import Verification
from pharos.agent.workspace import GitGuardError
from pharos.config import ConfigError, load_config
from pharos.console import force_utf8
from pharos.log import configure_logging
from pharos.paths import shorten_path
from pharos.preflight.split import Grouping, SplitMode


def main(argv: list[str] | None = None) -> int:
    force_utf8(sys.stdout, sys.stderr)

    parser = argparse.ArgumentParser(
        prog="pharos run",
        description="Run a coding task that does not fit in one context window, by dividing "
        "it into parts that do and executing each in its own conversation.",
        epilog="Name files and folders in backticks — they are what the task is divided by. "
        "With no prompt and no --file, the prompt is read from stdin.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("prompt", nargs="?", help="the task to carry out")
    group.add_argument("--file", type=Path, help="read the task from a file instead")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="pre-flight and plan only; print the parts and change nothing",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="run the whole task in one conversation, undivided — the control case for "
        "showing what the division actually buys",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the scorecard as JSON on stdout, so a script or CI step can assert on a run",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="skip the clean-tree check and the run branch (you lose the undo)",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="let the model group the files into parts instead of packing them in the order "
        "they were named; the projections and refusals are unchanged either way",
    )
    parser.add_argument(
        "--reserve-reads",
        action="store_true",
        help="size the parts against what agents on this model historically opened on their "
        "own, not just against what the part names",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATH",
        help="drop a named file from every part's scope (repeatable) — for the file a long "
        "prompt names in order to FORBID it",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="after the run, show the diff to the model and print what it says. An opinion, "
        "printed under the verdict and unable to change it",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="skip indexing the tree around each part (you lose the report of changes no "
        "tool of the run claimed, and of what your own checks rewrote)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="when a part fills its window, stub the tool results it has finished with and "
        "keep going, instead of stopping there and handing off",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the project's own checks afterwards (they run twice, so a slow suite costs)",
    )
    parser.add_argument(
        "--stop-on-break",
        action="store_true",
        help="end the run at the first part that leaves a file it wrote unparseable, instead "
        "of carrying on into the parts after it",
    )
    parser.add_argument(
        "--no-ledger",
        action="store_true",
        help="do not carry Pharos's record of what landed on disk between parts; the model's "
        "own hand-off becomes the only thread",
    )
    args = parser.parse_args(argv)
    if args.stop_on_break and args.no_verify:
        # The per-part parse belongs to verification, so one flag would silently disable the
        # other. An accepted flag that does nothing is worse than a rejected one.
        parser.error("--stop-on-break needs the checks that --no-verify turns off")

    console = Console(stderr=args.json)
    progress = Console(stderr=True)

    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        return 2

    if args.file is not None:
        try:
            prompt = args.file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            console.print(f"[bold red]Cannot read the task file:[/] {exc}")
            return 2
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        if sys.stdin is None or sys.stdin.isatty():
            progress.print("[dim]Type the task, then Ctrl-Z Enter (Windows) / Ctrl-D (Unix).[/]")
        prompt = sys.stdin.read() if sys.stdin is not None else ""
    if not prompt.strip():
        console.print("[bold red]Empty task.[/] Pass it as an argument, with --file, or on stdin.")
        return 2

    root = workspace_root(config.target_folder)
    # A run is long, mostly unattended, and its interesting details (which files were read,
    # what each part handed off) scroll past or never reach the terminal at all. Without this
    # the only way to work out why a part did nothing is to reproduce it. The terminal keeps
    # the summary; the file keeps the evidence.
    log_path = configure_logging(config.log_file)
    # Both are long absolute paths and the line holds two of them, so each gets half the
    # width rather than being wrapped through the middle of a directory name.
    half = max(progress.width // 2 - 12, 24)
    where = shorten_path(root, half)
    logged = shorten_path(log_path, half)
    progress.print(
        f"[dim]workspace {where} {chr(183)} log {logged}[/]", no_wrap=True, overflow="ignore"
    )

    status = RunStatus()

    def report(message: str) -> None:
        line = status.event(message)
        if line is not None:
            progress.print(line)

    # The live line is a terminal affordance. Redirected to a file it would be thousands of
    # cursor moves, so there it degrades to the printed events alone, which is what a log
    # wants anyway.
    animate = progress.is_terminal and not args.dry_run

    live: AbstractContextManager[object] = (
        Live(status, console=progress, refresh_per_second=10, transient=True)
        if animate
        else nullcontext()
    )

    try:
        with live:
            outcome = asyncio.run(
                run_task(
                    config,
                    prompt,
                    dry_run=args.dry_run,
                    use_git=not args.no_git,
                    verify_work=not args.no_verify,
                    divide=not args.no_split,
                    semantic=args.semantic,
                    reserve_reads=args.reserve_reads,
                    compact=args.compact,
                    audit=not args.no_audit,
                    review=args.review,
                    exclude=args.exclude,
                    use_ledger=not args.no_ledger,
                    stop_on_break=args.stop_on_break or config.stop_on_break,
                    on_event=report,
                )
            )
    except GitGuardError as exc:
        console.print(f"[bold red]Refusing to run:[/] {exc}")
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/] Files already written are on the run branch.")
        return 1

    return _render(console, outcome, dry_run=args.dry_run, as_json=args.json, root=str(root))


# Presentation only, and deliberately restrained. A bar makes a fraction legible at a glance;
# it must never make one look better than it is, so the thresholds match the words beside it
# and a partial run is coloured as a partial run.
_BAR_WIDTH = 24


# The live commentary is the only thing on screen for minutes at a time, so it is worth
# colouring by kind. Everything stays one line and nothing is hidden: a refusal a user cannot
# see is a refusal they will rediscover as a mystery in the scorecard.
_LIVE_MARKS: tuple[tuple[str, str, str], ...] = (
    ("BACKEND TRUNCATED", "bold red", "!"),
    ("failed", "red", "x"),
    ("!! ", "yellow", "!"),
    ("REFUSED", "yellow", "!"),
    ("ok write_file", "green", "+"),
    ("ok replace_lines", "green", "+"),
    ("ok ", "cyan", chr(183)),
    ("asking again", "yellow", chr(183)),
    ("answered without editing", "yellow", chr(183)),
    ("stopped with", "yellow", chr(183)),
    ("in a row failed", "yellow", "!"),
    ("ceiling reached", "yellow", "!"),
    ("recovered", "dim", chr(183)),
    ("generating", "dim", chr(183)),
)


_LABELLED = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$", re.DOTALL)


def _live(message: str) -> str:
    """One progress line, marked by what it is.

    Everything user- or model-supplied is escaped before it goes near markup. A part label is
    written "[part 3]" and a model can hand back any path it likes, both of which Rich reads
    as tags — the label was being silently swallowed, and a file called `[draft].cs` would
    have taken a line of output with it.

    Structural lines (a new part beginning) get weight; routine tool chatter stays quiet, so
    the refusals and the truncation warning are what the eye catches.
    """
    stripped = message.strip()
    if stripped.startswith(("part ", "repair ")) and " of " in stripped:
        return f"[bold cyan]  {escape(stripped)}[/]"

    label, body = "", stripped
    match = _LABELLED.match(message)
    if match:
        label, body = match.group(1), match.group(2).strip()

    style, mark = "", ""
    for needle, needle_style, needle_mark in _LIVE_MARKS:
        if needle in body:
            style, mark = needle_style, needle_mark
            break
    body = body.removeprefix("ok ").removeprefix("!! ")

    if not label:
        # A run-level line: the workspace, the window being reloaded, where the undo went.
        # It belongs at the same indent as the header, not tucked under a part.
        return f"[{style or 'dim'}]  {escape(body)}[/]"
    return (
        f"    [{style or 'dim'}]{mark or chr(183)}[/] "
        f"[dim]{escape(label)}[/] [{style or 'dim'}]{escape(body)}[/]"
    )


def _bar(fraction: float) -> str:
    filled = round(fraction * _BAR_WIDTH)
    style = "green" if fraction >= 0.999 else "yellow" if fraction >= 0.6 else "red"
    return f"[{style}]{chr(9608) * filled}[/][dim]{chr(9617) * (_BAR_WIDTH - filled)}[/]"


def _headline(card: Scorecard) -> tuple[str, str, str]:
    """The one-word verdict, its colour, and the sentence that qualifies it."""
    if card.failed_parts:
        return "FAILED", "bold red", "a part could not finish"
    if card.stopped_on_break is not None:
        # Not FAILED (nothing errored) and not INCOMPLETE (the untouched files were never
        # attempted). The run was halted, on purpose, and the word has to say so or the
        # coverage figure below it reads as a model that gave up.
        return (
            "STOPPED",
            "bold red",
            f"{card.stopped_on_break} left a file it wrote unparseable, and --stop-on-break "
            f"ended the run there",
        )
    if card.coverage is None:
        return "DONE", "bold green", "one unrestricted part; nothing to measure coverage against"
    if card.complete:
        note = "every file the plan assigned was changed"
        if card.abandoned_parts:
            note += f", though {card.abandoned_parts} part(s) stopped early"
        return "COMPLETE", "bold green", note
    written, total = card.written_files, card.scoped_files
    if written >= total and card.verification is not None and not card.verification.ok:
        # Coverage is full, so "N of M files were never changed" would read "0 of 6" beside a
        # 100% bar. The run is not incomplete for want of writing; it is broken for what it
        # wrote, and the verdict has to say which.
        names = ", ".join(check.name for check in card.verification.newly_broken)
        return "BROKEN", "bold red", f"every file was changed, but {names} now fails"
    return "INCOMPLETE", "bold yellow", f"{total - written} of {total} files were never changed"


def _part_glyph(result: PartResult, scoped: int, written: int) -> tuple[str, str]:
    if result.error:
        return chr(10007), "red"
    if result.truncated:
        return "!", "red"
    if scoped and written == 0:
        return chr(10007), "yellow"
    if scoped and written < scoped:
        return chr(126), "yellow"
    return chr(10003), "green"


def _shorten(paths: list[str], limit: int = 2) -> str:
    """File names without their directories: the report is a summary, not an inventory."""
    names = [p.rsplit("/", 1)[-1].rsplit(chr(92), 1)[-1] for p in paths]
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" +{len(names) - limit}"


def _scope_label(file: object) -> str:
    """A part's file as a reader wants it: the path, and the line range only when there is one.

    ``PartFile.label()`` spells out "(whole file - from <the directory it came from>)" for
    every entry, which is the right thing in the splitter's own report and pure noise in a
    list of four. The directory is already in the path.
    """
    display = str(getattr(file, "display", file))
    start = getattr(file, "line_start", None)
    end = getattr(file, "line_end", None)
    return f"{display}  [dim]lines {start}-{end}[/]" if start else display


class RunStatus:
    """A single line that keeps moving while a run works.

    A run is minutes of near-silence punctuated by events. Printed events alone cannot tell a
    user whether the gap between two of them is a model thinking or a process wedged, and that
    was the honest complaint: there is no way to tell a slow run from a dead one. A spinner
    that keeps turning answers it without anybody having to ask, and the elapsed clock makes
    "slow" measurable rather than a feeling.

    Everything it shows is real. The phase is the last thing that actually happened, the
    counters are actual writes and actual tool calls, and the clocks are wall time. A progress
    BAR would be the obvious thing here and would be a lie: the number of tool calls a part
    needs is not knowable in advance, so the only honest shapes are a spinner and a count.
    """

    def __init__(self) -> None:
        self.part = ""
        self.phase = "starting"
        self.started = time.monotonic()
        self.part_started = time.monotonic()
        self.files_written = 0
        self.tool_calls = 0
        self.trouble = ""
        self._spinner = Spinner("dots", style="cyan")

    def event(self, message: str) -> str | None:
        """Fold one event into the status. Returns a line to scroll, or None to stay quiet."""
        stripped = message.strip()
        match = _LABELLED.match(message)
        body = match.group(2).strip() if match else stripped

        if stripped.startswith(("part ", "repair ")) and " of " in stripped:
            self.part = stripped
            self.part_started = time.monotonic()
            self.phase = "starting"
            return _live(message)

        if body.startswith("generating"):
            # Already once every few seconds and identical each time; the spinner is saying
            # the same thing more cheaply, so it stays out of the scrollback.
            self.phase = body.replace(chr(8230), "").replace("...", "").strip()
            return None

        if body.startswith("ok "):
            self.tool_calls += 1
            action = body[3:]
            if action.startswith(("write_file", "replace_lines")):
                self.files_written += 1
            self.phase = action
            return _live(message)

        if body.startswith("!! ") or "REFUSED" in body:
            self.tool_calls += 1
            self.phase = body.removeprefix("!! ")
            self.trouble = self.phase
            return _live(message)

        if "TRUNCATED" in body or "in a row failed" in body or "ceiling reached" in body:
            self.trouble = body
            self.phase = body
            return _live(message)

        self.phase = body or self.phase
        return _live(message)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render to fit, whatever the terminal is.

        Exactly one line, always. A status that wraps becomes a jumping two- or three-line
        block on a narrow terminal, which is worse than no status at all — and `no_wrap` on the
        Text is not enough, because the spinner re-renders it. Truncating against the real
        width is, so the width has to come from the console rather than be assumed.
        """
        looks_like_path = "/" in self.phase or chr(92) in self.phase
        phase = _shorten([self.phase], limit=1) if looks_like_path else self.phase
        head = (self.part or "working").replace(" of ", "/")

        text = Text.from_markup(
            f"[bold cyan]{escape(head)}[/] [dim]{chr(183)}[/] {escape(phase)} "
            f"[dim]{chr(183)} {_clock(time.monotonic() - self.part_started)} in part "
            f"{chr(183)} {_clock(time.monotonic() - self.started)} total "
            f"{chr(183)} {self.files_written} written {chr(183)} {self.tool_calls} calls[/]"
        )
        # Two columns for the spinner and its space.
        text.truncate(max(options.max_width - 2, 12), overflow="ellipsis")
        self._spinner.style = "yellow" if self.trouble else "cyan"
        self._spinner.update(text=text)
        yield self._spinner


def _clock(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m{rest:02d}s" if minutes else f"{rest}s"


def _render_header(console: Console, outcome: RunOutcome, root: str) -> None:
    report = outcome.report
    model = (report.model if report else None) or "no model"
    exactness = "exact" if outcome.counts_exact else "heuristic chars/4"
    window = ""
    if report is not None and report.profile is not None:
        budget = report.profile.budget
        if budget is not None and budget.loaded_ctx:
            window = f" {chr(183)} window {budget.loaded_ctx:,}"

    console.print()
    console.print(f"[bold cyan]  Pharos run[/]  [dim]{model} {chr(183)} {exactness}{window}[/]")
    console.print(
        f"[dim]  {shorten_path(root, max(console.width - 4, 24))}[/]",
        no_wrap=True,
        overflow="ignore",
    )


def _render_plan(console: Console, outcome: RunOutcome) -> None:
    plan = outcome.plan
    if plan is None:
        return
    if not outcome.divided:
        declined = len(plan.parts)
        console.print(
            f"[yellow]  undivided[/] [dim](--no-split){chr(160)}"
            + (f" {chr(183)} declined a {declined}-part plan" if declined else "")
            + "[/]"
        )
    elif plan.mode is not SplitMode.NONE:
        console.print(
            f"[dim]  divided into {len(plan.parts)} parts of "
            f"{chr(8804)} {plan.target_per_part:,} tokens[/]"
        )
    elif plan.already_fits:
        console.print("[dim]  fits in one window[/]")
    if plan.grouping_note is not None:
        # A run whose parts a model chose is a different run. It says so, whichever way the
        # proposal went, because the note is the only place the fallback is visible.
        by_model = plan.grouping is Grouping.SEMANTIC
        console.print(
            f"[{'cyan' if by_model else 'yellow'}]  "
            f"{'grouped by meaning' if by_model else 'grouped by position'}[/] "
            f"[dim]{chr(183)} {plan.grouping_note}[/]"
        )


def _render_parts(console: Console, outcome: RunOutcome) -> None:
    """One line per part: what it was given, what it did, and how close it came to the edge."""
    table = Table.grid(padding=(0, 1))
    table.add_column(width=1)                       # indent, to sit under the header
    table.add_column(width=1)                       # glyph
    table.add_column(min_width=11)                  # which part
    table.add_column(justify="right", min_width=5)  # files
    table.add_column(justify="right", min_width=7)  # rounds
    table.add_column(style="dim", overflow="ellipsis")  # detail

    planned = len(outcome.parts) - outcome.repair_parts
    for index, result in enumerate(outcome.parts, start=1):
        scoped, written = len(result.scoped), len(result.files_written)
        glyph, style = _part_glyph(result, scoped, written)
        name = (
            f"part {index}/{planned}"
            if index <= planned
            else f"repair {index - planned}"
        )
        files = f"{written}/{scoped}" if scoped else str(written)
        detail = _shorten(result.files_written) if result.files_written else ""
        if result.error:
            detail = result.error
        elif result.truncated:
            detail = "backend dropped context"
        elif result.stopped_early:
            detail = "hit the window ceiling"
        elif scoped and written == 0:
            detail = "wrote nothing"
        table.add_row(
            "",
            f"[{style}]{glyph}[/]",
            name,
            f"[{style}]{files}[/]",
            f"[dim]{result.steps} rnd[/]",
            detail,
        )
    console.print()
    console.print(table)


def _add_untouched_row(body: Table, card: Scorecard) -> None:
    """Split the misses into the two things they can be.

    Coverage is deliberately written-over-scoped and is not softened here. What this adds is
    the fact coverage cannot carry: whether a file nobody wrote was one a part opened and left
    alone, or one nobody ever looked at. Measured on a nineteen-file run where three of the
    three "misses" were one-line `__init__.py` modules a part had opened and correctly found
    nothing to do in -- 83% coverage, and no way to see that from the number.
    """
    if not card.untouched:
        return
    examined = card.examined_untouched
    unopened = [path for path in card.untouched if path not in set(examined)]
    parts = []
    if examined:
        parts.append(f"{len(examined)} opened and left alone")
    if unopened:
        parts.append(f"[yellow]{len(unopened)} never opened[/]")
    shown = _first_few(card.untouched, 4)
    body.add_row(
        "unchanged",
        f"{len(card.untouched)} scoped file(s) nobody wrote — {', '.join(parts)}  "
        f"[dim]{shown}[/]",
    )


def _add_tool_call_row(body: Table, card: Scorecard) -> None:
    """Say when the model never produced a structured tool call.

    Silent in the normal case. When it fires it is usually the whole explanation for a run
    that covered nothing, and it is a fact about the model rather than a guess: the requests
    carried a tool catalogue and no reply came back with a call in it.
    """
    if not card.tool_calls_unsupported:
        if card.recovered_calls:
            body.add_row(
                "tool calls",
                f"[dim]{card.native_calls} native, {card.recovered_calls} recovered from "
                f"text — this model emits some of its calls as prose[/]",
            )
        return
    recovered = (
        f" {card.recovered_calls} were recovered from text, which is a fallback and not a fix."
        if card.recovered_calls
        else ""
    )
    body.add_row(
        "tool calls",
        f"[red]none — the model never returned a structured tool call[/]\n"
        f"[dim]Every request carried the catalogue.{recovered} A model whose template does "
        f"not support tool calling cannot drive a run, whatever the window says; try one "
        f"that does before reading anything else on this card.[/]",
    )


def _add_audit_row(body: Table, card: Scorecard) -> None:
    """What the filesystem said, against what the run said about itself.

    Silent when it agrees, which is the common case and not worth a line. Everything it can
    report is a fact about the disk, and none of it is a judgement of the code: a change
    nobody claimed may be perfectly fine, and the point is that it stops being invisible.
    """
    if not card.audits:
        return
    findings: list[str] = []
    if card.absent_writes:
        findings.append(
            f"[red]{len(card.absent_writes)} write(s) reported that the disk does not "
            f"show[/]  [dim]{_first_few(card.absent_writes)}[/]"
        )
    if card.out_of_scope_changes:
        findings.append(
            f"[yellow]{len(card.out_of_scope_changes)} file(s) changed outside the part's "
            f"scope[/]  [dim]{_first_few(card.out_of_scope_changes)}[/]"
        )
    if card.unattributed_changes:
        findings.append(
            f"[yellow]{len(card.unattributed_changes)} change(s) no tool of the run "
            f"claimed[/]  [dim]{_first_few(card.unattributed_changes)}[/]"
        )
    if card.check_writes:
        findings.append(
            f"{len(card.check_writes)} file(s) your own checks rewrote  "
            f"[dim]{_first_few(card.check_writes)} {chr(183)} a formatter or a snapshot "
            f"test, not the model[/]"
        )
    if any(a.truncated for a in card.audits):
        findings.append(
            f"[dim]the tree was too large to index in full ({FILE_CAP:,} files) {chr(183)} "
            f"anything past the cap is outside this report[/]"
        )
    if not findings:
        changed = sum(len(a.changed.paths) for a in card.audits)
        body.add_row(
            "audit",
            f"[green]{chr(10003)}[/] [dim]{changed} file change(s), every one of them "
            f"claimed by the part that made it[/]",
        )
        return
    body.add_row("audit", chr(10).join(findings))


def _first_few(paths: list[str], limit: int = 3) -> str:
    shown = ", ".join(paths[:limit])
    return shown if len(paths) <= limit else f"{shown} and {len(paths) - limit} more"


def _add_verification_row(body: Table, verification: Verification | None) -> None:
    """What the project's own checks said, and which of them this run actually broke.

    Three states are kept apart on purpose. A check that never ran is not a pass, a check that
    was already failing is not this run's doing, and only the difference between the two
    sweeps is charged to the model.
    """
    if verification is None:
        return
    if not verification.ran:
        reasons = {check.skipped for check in verification.skipped if check.skipped}
        detail = "; ".join(sorted(reasons)) or "nothing to check"
        body.add_row("verification", f"[dim]not run {chr(183)} {detail}[/]")
        return

    broken = verification.newly_broken
    if broken:
        lines = [
            f"[red]{len(broken)} check(s) this run broke[/]  "
            f"[dim]passed before, failing now[/]"
        ]
        for check in broken:
            lines.append(f"[red]{chr(10007)}[/] {check.name}")
            for line in check.detail.splitlines()[:3]:
                lines.append(f"    [dim]{line.strip()}[/]")
    else:
        passed = ", ".join(check.name for check in verification.passing) or "none"
        lines = [f"[green]{chr(10003)}[/] [dim]{passed}[/]"]

    for check in verification.already_failing:
        moved = " and its output has changed" if check.changed_while_failing else ""
        style = "yellow" if check.changed_while_failing else "dim"
        lines.append(f"[{style}]{check.name} was already failing before the run{moved}[/]")
    for check in verification.unattributable:
        lines.append(
            f"[yellow]{check.name} is failing, but its baseline never ran - "
            f"cannot say this run caused it[/]"
        )
    for check in verification.skipped:
        lines.append(f"[dim]{check.name} skipped {chr(183)} {check.skipped}[/]")
    if verification.unchecked_files:
        count = len(verification.unchecked_files)
        lines.append(f"[dim]{count} written file(s) in no format this can parse[/]")
    body.add_row("verification", chr(10).join(lines))


def _render_scorecard(console: Console, card: Scorecard) -> None:
    """The five questions, in the order they matter when a run disappoints."""
    word, style, qualifier = _headline(card)
    body = Table.grid(padding=(0, 2))
    body.add_column(min_width=11)
    body.add_column()

    body.add_row(f"[{style}]{word}[/]", f"[dim]{qualifier}[/]")
    if card.coverage is not None:
        body.add_row(
            "coverage",
            f"{_bar(card.coverage)}  {card.coverage:.0%}  "
            f"[dim]{card.written_files} of {card.scoped_files} files[/]",
        )
        for path in card.untouched[:4]:
            body.add_row("", f"[dim]untouched  {path}[/]")
        if len(card.untouched) > 4:
            body.add_row("", f"[dim]... {len(card.untouched) - 4} more[/]")

    if card.repair_parts:
        plan = card.plan_coverage
        share = f" {chr(183)} the plan alone reached {plan:.0%}" if plan is not None else ""
        body.add_row(
            "repair",
            f"{card.repair_parts} extra part(s) rescued {card.rescued} file(s){share}",
        )

    if card.handoffs_expected:
        ok = card.kept_the_thread
        mark = chr(10003) if ok else "!"
        colour = "green" if ok else "yellow"
        body.add_row(
            "continuity",
            f"[{colour}]{mark}[/] {card.handoffs_produced}/{card.handoffs_expected} hand-offs "
            f"[dim]{chr(183)} largest {card.largest_handoff:,} of "
            f"{card.handoff_reserve:,} reserved[/]",
        )
        if card.thin_handoffs:
            # Two different facts, and the second must not read as an excuse for the first.
            # A part that wrote files and reported nothing still failed to report; that the
            # run carried the context anyway is worth saying in the same breath and not
            # instead.
            rescued = (
                f" [dim]{chr(183)} Pharos carried {card.ledger_files} changed file(s) "
                f"forward regardless[/]"
                if card.ledger_on and card.ledger_files
                else ""
            )
            body.add_row(
                "",
                f"[yellow]{card.thin_handoffs} part(s) changed files and reported almost "
                f"nothing[/]{rescued}",
            )
        if card.handoffs_requested:
            # Not a fault, and not folded into the count above it. A part whose parting
            # message named none of its own work was asked for one properly, and answered; a
            # run where that happened five times out of five held the thread only because it
            # was prompted to at every step, which is a different run from one where it did
            # not need to be.
            body.add_row(
                "",
                f"[dim]{card.handoffs_requested} of {card.handoffs_expected} had to be asked "
                f"for {chr(183)} the rest were volunteered[/]",
            )
        if card.handoff_overruns:
            body.add_row("", f"[yellow]{card.handoff_overruns} overran the reserve[/]")
        if card.revisits:
            body.add_row(
                "",
                f"[yellow]{len(card.revisits)} revisit(s)[/] [dim]a part reached for work "
                f"another part owned[/]",
            )
    if card.invented:
        body.add_row(
            "wandering",
            f"[dim]{len(card.invented)} path(s) in no part's scope {chr(183)} refused, "
            f"nothing touched[/]",
        )

    if card.damage:
        # Which part, not just which file. The verification block below says the project is
        # broken; on a five-part run this is the half that says whose diff to read.
        for entry in card.damage[:4]:
            fixed = (
                f" [dim]{chr(183)} repaired by {entry.repaired_by}[/]"
                if entry.repaired_by
                else ""
            )
            colour = "yellow" if entry.repaired_by else "red"
            body.add_row(
                "damage" if entry is card.damage[0] else "",
                f"[{colour}]{entry.label} broke {escape(entry.subject)}[/]{fixed}\n"
                f"  [dim]{escape(first_line(entry.error, 96))}[/]",
            )
        if len(card.damage) > 4:
            body.add_row("", f"[dim]... {len(card.damage) - 4} more[/]")

    if card.truncated_parts:
        body.add_row(
            "[bold red]TRUNCATED[/]",
            f"[red]the backend dropped context in {card.truncated_parts} part(s)[/] "
            f"[dim]its own count fell as the conversation grew, so the loaded window is "
            f"smaller than the one measured[/]",
        )

    body.add_row(
        "headroom",
        f"{_bar(min(card.peak_fraction, 1.0))}  {card.peak_fraction:.0%}  "
        f"[dim]of a part's ceiling, at the worst part[/]",
    )
    if card.drift_high is not None and card.drift_low is not None:
        span = (
            f"{card.drift_low:.2f}x"
            if round(card.drift_low, 2) == round(card.drift_high, 2)
            else f"{card.drift_low:.2f}-{card.drift_high:.2f}x"
        )
        # Past the margin no longer means a ceiling was enforced against a number below the
        # real prompt: after the first response the ceiling scales by the ratio measured here.
        # It still means the estimate needs the correction, which is worth seeing.
        corrected = (
            "every request corrected"
            if not card.exposed_requests
            else f"{card.exposed_requests} request(s) went out uncorrected"
        )
        # The margin is not the only thing behind the ceiling. When the remembered template
        # cost covers the shortfall AND nothing went out uncorrected, a live run was never
        # actually enforced against a number below the real prompt -- so it does not get the
        # yellow. Printing the alarm and "every request corrected" side by side, both true,
        # left a reader unable to tell which one described their run.
        covered = (
            card.template_offset is not None
            and not card.exposed_requests
            and card.worst_shortfall <= SAFETY_MARGIN + card.template_offset
        )
        if card.under_counted and covered:
            note = (
                f"[dim]short by {card.worst_shortfall:,} tokens, past the {SAFETY_MARGIN}-token "
                f"margin on its own {chr(183)} covered by the +{card.template_offset:,} "
                f"remembered, and {corrected}[/]"
            )
        elif card.under_counted:
            note = (
                f"[yellow]short by {card.worst_shortfall:,} tokens, past the "
                f"{SAFETY_MARGIN}-token margin[/] [dim]{chr(183)} {corrected}[/]"
            )
        else:
            note = (
                f"[dim]worst shortfall {card.worst_shortfall:,} tokens, inside the "
                f"{SAFETY_MARGIN}-token margin[/]"
            )
        body.add_row(
            "drift",
            f"{span} [dim]over {card.drift_samples} requests[/] {chr(183)} {note}",
        )
    if card.template_offset is not None:
        # What the ceiling was actually enforced against, which the drift line above cannot
        # say: it reports the raw estimator on purpose, so that it goes on measuring the
        # estimator rather than the correction applied to it.
        body.add_row(
            "template",
            f"+{card.template_offset:,} tokens [dim]remembered over {card.template_runs} "
            f"previous run(s) {chr(183)} every part's ceiling started corrected[/]",
        )
    if card.nudged_parts or card.abandoned_parts:
        body.add_row(
            "convergence",
            f"[dim]{card.nudged_parts} part(s) needed a reminder, "
            f"{card.abandoned_parts} stopped early"
            + (
                f" ({card.stopped_with_work_done} of them with every file they owned "
                f"already written)"
                if card.stopped_with_work_done
                else ""
            )
            + "[/]",
        )
    if card.compacted_tokens:
        body.add_row(
            "compaction",
            f"{card.compacted_tokens:,} tokens reclaimed [dim]across "
            f"{card.compacted_parts} part(s) {chr(183)} tool results those parts had "
            f"finished with, stubbed rather than dropped[/]",
        )
    elif card.reclaimable_tokens:
        # The number that decides whether --compact is worth turning on here, and it can only
        # be known by a run that did NOT use it.
        body.add_row(
            "compaction",
            f"[yellow]not asked for[/] [dim]{chr(183)} {card.reclaimable_tokens:,} tokens of "
            f"the window that stopped a part were tool results it had finished with; "
            f"--compact would have given them back[/]",
        )
    _add_untouched_row(body, card)
    _add_tool_call_row(body, card)
    _add_audit_row(body, card)
    _add_verification_row(body, card.verification)

    console.print()
    console.print(
        Padding(
            Panel(
                body,
                title="Scorecard",
                title_align="left",
                border_style=style,
                padding=(1, 2),
                expand=False,
            ),
            (0, 0, 0, 2),
        )
    )


def _render_footer(console: Console, outcome: RunOutcome) -> None:
    """What to do next, which is always the same two things: look at it, or undo it."""
    lines: list[str] = []
    if outcome.files_changed:
        lines.append(f"[bold]{len(outcome.files_changed)} file(s) changed on disk.[/]")
    else:
        lines.append("[yellow]Nothing was written.[/]")

    if outcome.branch:
        base = outcome.base_branch or "-"
        lines.append(f"[dim]review[/]  git diff {base}")
        lines.append(f"[dim]undo[/]    git checkout {base} && git branch -D {outcome.branch}")
    elif outcome.undo is not None:
        lines.append(f"[dim]review[/]  compare against {outcome.undo.directory}")
        lines.append("[dim]undo[/]    copy that folder back over the workspace")

    # Said last because it is the caveat a reader should leave with: Pharos proved the task
    # fit and ran, and has no opinion at all about whether the code is right.
    lines.append("")
    lines.append("[dim]Pharos checked that it fit and that it ran. Reading the diff is yours.[/]")
    if not outcome.via_proxy:
        lines.append("[dim]Ran without the proxy, so the dashboard recorded nothing.[/]")
    if not outcome.counts_exact:
        lines.append(
            "[yellow]Counts were heuristic[/][dim] - no GGUF resolved, so the ceiling this run "
            "held itself to was an estimate.[/]"
        )
    console.print()
    for line in lines:
        console.print(f"  {line}" if line else "")
    console.print()


def _review_summary(review: Review | None) -> dict[str, object] | None:
    """The review as plain data. Its own key, so nothing consuming the scorecard picks it up
    by accident and starts gating on a model's opinion of somebody's code."""
    if review is None:
        return None
    return {
        "ran": review.ran,
        "findings": [
            {"file": f.file, "line": f.line, "severity": f.severity, "note": f.note}
            for f in review.findings
        ],
        "by_severity": review.by_severity,
        "reviewed": list(review.reviewed),
        "unreviewed": list(review.unreviewed),
        "discarded": review.discarded,
        "note": review.note,
        "advisory": True,
    }


def _render_review(console: Console, review: Review | None) -> None:
    """The opinion, printed last and labelled as one.

    The heading says what it is every single time. A finding here has been checked for
    pointing at a real changed line and for nothing else — it has not been checked for being
    RIGHT, and there is no way to check that from here.
    """
    if review is None:
        return
    console.print()
    if not review.ran:
        detail = review.note or "there was nothing to review"
        console.print(f"  [bold]Review[/] [dim]{chr(183)} not run: {detail}[/]")
        console.print()
        return

    lines: list[str] = []
    ordered = sorted(
        review.findings, key=lambda f: (SEVERITY_ORDER.index(f.severity), f.file, f.line)
    )
    for finding in ordered:
        colour = {"bug": "red", "risk": "yellow", "note": "cyan"}[finding.severity]
        lines.append(
            f"[{colour}]{finding.severity:<4}[/] {finding.file}:{finding.line}  {finding.note}"
        )
    if not lines:
        lines.append(
            f"[dim]nothing reported across {len(review.reviewed)} changed file(s)[/]"
        )
    footnotes: list[str] = []
    if review.discarded:
        # Said out loud rather than swallowed: it is the measurement of how much of this
        # particular answer was invented, and it is the reason the check exists.
        footnotes.append(
            f"{review.discarded} finding(s) discarded — they named a file this run did not "
            f"change, or a line the diff does not contain"
        )
    if review.unreviewed:
        footnotes.append(f"not reviewed: {', '.join(review.unreviewed[:4])}")
    if review.note:
        footnotes.append(review.note)
    for note in footnotes:
        lines.append(f"[dim]{note}[/]")

    console.print(
        Padding(
            Panel(
                chr(10).join(lines),
                title="Review — one model's opinion, not a measurement",
                title_align="left",
                border_style="dim",
                padding=(1, 2),
                expand=False,
            ),
            (0, 0, 0, 2),
        )
    )
    console.print(
        "  [dim]It did not decide anything above. The verdict, the coverage and the "
        "verification were settled before it was asked.[/]"
    )
    console.print()


SEVERITY_ORDER = list(SEVERITIES)


def _plan_summary(outcome: RunOutcome) -> dict[str, object] | None:
    """How the parts were arranged, for a reader that is a script.

    The human output says whether a model grouped the parts and why, on every run. The JSON
    said nothing at all, so a CI step consuming it could not tell a model-arranged plan from a
    position-packed one -- which is the single outcome --semantic is not allowed to have, and
    it had it in the only format a machine reads.

    Part bodies are left out on purpose: they can run to several KB each and, unlike in
    `pharos check --json`, they have already been executed by the time anyone reads this.
    """
    plan = outcome.plan
    if plan is None:
        return None
    return {
        "mode": plan.mode.value,
        "divided": outcome.divided,
        "grouping": plan.grouping.value,
        "grouping_note": plan.grouping_note,
        "target_per_part": plan.target_per_part,
        "handoff_reserve": plan.handoff_reserve,
        "parts": [
            {
                "index": part.index,
                "title": part.title,
                "projected_tokens": part.projected_tokens,
                "files": [f.display for f in part.files],
            }
            for part in plan.parts
        ],
    }


def _render(
    console: Console,
    outcome: RunOutcome,
    *,
    dry_run: bool,
    as_json: bool = False,
    root: str = "",
) -> int:
    report = outcome.report
    _render_header(console, outcome, root or str(workspace_root(None)))

    if outcome.error is not None:
        console.print()
        console.print(f"  [bold red]Did not run[/]  {outcome.error}")
        console.print()
        return 2 if report is None or not outcome.parts else 1

    _render_plan(console, outcome)

    if dry_run:
        plan = outcome.plan
        console.print()
        if plan is not None and plan.parts:
            table = Table.grid(padding=(0, 2))
            table.add_column(width=1)
            table.add_column(min_width=11)
            table.add_column(justify="right")
            table.add_column(style="dim", overflow="ellipsis")
            for part in plan.parts:
                files = part.files or []
                table.add_row(
                    "",
                    f"part {part.index}/{part.total}"
                    + (f" {chr(183)} {part.title}" if part.title else ""),
                    f"{chr(8805)} {part.projected_tokens:,}",
                    _scope_label(files[0]) if files else "text segment",
                )
                for extra in files[1:]:
                    table.add_row("", "", "", _scope_label(extra))
            console.print(table)
            # Otherwise two parts holding near-identical files show wildly different
            # projections and nothing on screen says why: every part after the first is
            # counted as arriving with the previous part's hand-off already in its window.
            if len(plan.parts) > 1 and plan.handoff_reserve:
                console.print(
                    f"[dim]  parts after the first include {plan.handoff_reserve:,} tokens "
                    f"of room for the hand-off they arrive with[/]"
                )
        console.print()
        console.print("  [yellow]Dry run[/] [dim]nothing was executed, no files were touched[/]")
        console.print()
        if as_json:
            # `--dry-run --json` used to return here having written nothing at all: the human
            # table went to stderr (where --json puts it) and stdout stayed empty, so a CI step
            # piping into `jq` got a parse error from a command that had exited 0. The plan is
            # exactly the payload worth having -- it answers "what would this run do?" without
            # running it.
            sys.stdout.write(
                json.dumps(
                    {
                        "dry_run": True,
                        "plan": _plan_summary(outcome),
                        "planned_files": outcome.planned_files,
                    },
                    indent=2,
                )
                + "\n"
            )
        return 0

    card = score(
        outcome.parts,
        handoff_reserve=outcome.handoff_reserve,
        repair_parts=outcome.repair_parts,
        verification=outcome.verification,
        ledger_on=outcome.ledger_on,
        ledger_files=outcome.ledger_files,
        damage=outcome.damage,
        stopped_on_break=outcome.stopped_on_break,
        planned_files=outcome.planned_files,
        template_offset=outcome.template_cost.offset if outcome.template_cost else None,
        template_runs=outcome.template_cost.runs if outcome.template_cost else 0,
        audits=outcome.audits,
    )
    if as_json:
        # stdout belongs to the payload alone, exactly as `pharos check --json` treats it.
        payload: dict[str, object] = dict(to_dict(card))
        payload["plan"] = _plan_summary(outcome)
        # Beside the scorecard, never inside it: nothing in `card` was allowed to see this.
        payload["review"] = _review_summary(outcome.review)
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0 if card.complete else 1

    _render_parts(console, outcome)
    _render_scorecard(console, card)
    _render_review(console, outcome.review)
    _render_footer(console, outcome)
    return 0 if card.complete else 1
