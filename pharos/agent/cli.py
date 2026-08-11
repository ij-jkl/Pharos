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
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table

from pharos.agent.runner import RunOutcome, run_task
from pharos.agent.scorecard import Scorecard, score, to_dict
from pharos.agent.session import SAFETY_MARGIN, PartResult
from pharos.agent.tools import workspace_root
from pharos.agent.workspace import GitGuardError
from pharos.config import ConfigError, load_config
from pharos.console import force_utf8
from pharos.log import configure_logging
from pharos.preflight.split import SplitMode


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
    args = parser.parse_args(argv)

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
    progress.print(f"[dim]workspace {root} · log {log_path}[/]")

    try:
        outcome = asyncio.run(
            run_task(
                config,
                prompt,
                dry_run=args.dry_run,
                use_git=not args.no_git,
                divide=not args.no_split,
                on_event=lambda message: progress.print(_live(message)),
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
    if card.coverage is None:
        return "DONE", "bold green", "one unrestricted part; nothing to measure coverage against"
    if card.complete:
        return "COMPLETE", "bold green", "every file the plan assigned was changed"
    written, total = card.written_files, card.scoped_files
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
    console.print(f"[dim]  {root}[/]")


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
            body.add_row(
                "",
                f"[yellow]{card.thin_handoffs} part(s) changed files and reported almost "
                f"nothing[/]",
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
        note = (
            f"[yellow]short by {card.worst_shortfall:,} tokens, past the "
            f"{SAFETY_MARGIN}-token margin[/]"
            if card.under_counted
            else f"[dim]worst shortfall {card.worst_shortfall:,} tokens, inside the "
                 f"{SAFETY_MARGIN}-token margin[/]"
        )
        body.add_row(
            "drift",
            f"{span} [dim]over {card.drift_samples} requests[/] {chr(183)} {note}",
        )
    if card.nudged_parts or card.abandoned_parts:
        body.add_row(
            "convergence",
            f"[dim]{card.nudged_parts} part(s) needed a reminder, "
            f"{card.abandoned_parts} stopped early[/]",
        )

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


def _render_footer(console: Console, outcome: RunOutcome, card: Scorecard) -> None:
    """What to do next, which is always the same two things: look at it, or undo it."""
    lines: list[str] = []
    if outcome.files_changed:
        lines.append(f"[bold]{len(outcome.files_changed)} file(s) changed on disk.[/]")
    else:
        lines.append("[yellow]Nothing was written.[/]")

    if outcome.branch:
        lines.append("[dim]review[/]  git diff main")
        lines.append(f"[dim]undo[/]    git checkout main && git branch -D {outcome.branch}")
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
                    f"part {part.index}/{part.total}",
                    f"{chr(8805)} {part.projected_tokens:,}",
                    _scope_label(files[0]) if files else "text segment",
                )
                for extra in files[1:]:
                    table.add_row("", "", "", _scope_label(extra))
            console.print(table)
        console.print()
        console.print("  [yellow]Dry run[/] [dim]nothing was executed, no files were touched[/]")
        console.print()
        return 0

    card = score(
        outcome.parts,
        handoff_reserve=outcome.handoff_reserve,
        repair_parts=outcome.repair_parts,
    )
    if as_json:
        # stdout belongs to the payload alone, exactly as `pharos check --json` treats it.
        sys.stdout.write(json.dumps(to_dict(card), indent=2) + "\n")
        return 0 if card.complete else 1

    _render_parts(console, outcome)
    _render_scorecard(console, card)
    _render_footer(console, outcome, card)
    return 0 if card.complete else 1

    console.print()
    _render_scorecard(console, card)

    if outcome.files_changed:
        console.print(f"[bold]Changed {len(outcome.files_changed)} file(s):[/]")
        for path in outcome.files_changed:
            console.print(f"  {path}")
    else:
        console.print("[yellow]No files changed.[/]")

    if outcome.branch:
        console.print(
            f"\n[dim]On branch {outcome.branch} — review with `git diff main`, "
            f"bin it with `git checkout main && git branch -D {outcome.branch}`.[/]"
        )
    elif outcome.undo is not None:
        console.print()
        console.print(f"[dim]{outcome.undo.restore_hint()}[/]")
    if not outcome.via_proxy:
        console.print("[dim]Ran without the proxy, so nothing was recorded in the dashboard.[/]")
    if not outcome.counts_exact:
        console.print(
            "[yellow]Counts were heuristic[/] — no GGUF resolved for this model, so the "
            "ceiling the run held itself to was an estimate, not a measurement."
        )
    console.print()
    return 0 if card.complete else 1
