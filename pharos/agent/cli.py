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
import sys
from pathlib import Path

from rich.console import Console

from pharos.agent.runner import RunOutcome, run_task
from pharos.agent.scorecard import Scorecard, score, to_dict
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
                on_event=lambda message: progress.print(f"[dim]{message}[/]"),
            )
        )
    except GitGuardError as exc:
        console.print(f"[bold red]Refusing to run:[/] {exc}")
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/] Files already written are on the run branch.")
        return 1

    return _render(console, outcome, dry_run=args.dry_run, as_json=args.json)


def _ground_truth(reported: int | None) -> str:
    """The backend's own prompt_eval_count beside our estimate, when it gave us one."""
    return f"  [dim](backend counted {reported:,})[/]" if reported else ""


def _render_scorecard(console: Console, card: Scorecard) -> None:
    """The five questions, in the order they matter when a run disappoints."""
    console.print("[bold]Scorecard[/]")

    coverage = card.coverage
    if coverage is None:
        console.print(
            f"  coverage     [dim]not applicable — one unrestricted part, "
            f"{card.written_files} file(s) written[/]"
        )
    else:
        style = "green" if coverage == 1.0 else ("yellow" if coverage >= 0.5 else "red")
        console.print(
            f"  coverage     [{style}]{card.written_files} of {card.scoped_files} "
            f"scoped files written[/] ({coverage:.0%})"
        )
        for path in card.untouched[:6]:
            console.print(f"                 [dim]untouched: {path}[/]")
        if len(card.untouched) > 6:
            console.print(f"                 [dim]... {len(card.untouched) - 6} more[/]")

    if card.handoffs_expected:
        style = "green" if card.kept_the_thread else "yellow"
        console.print(
            f"  continuity   [{style}]{card.handoffs_produced} of {card.handoffs_expected} "
            f"hand-offs produced[/]; largest {card.largest_handoff:,} of "
            f"{card.handoff_reserve:,} reserved"
        )
        if card.handoff_overruns:
            console.print(
                f"                 [yellow]{card.handoff_overruns} overran the reserve[/] — "
                f"raise handoff_reserve, or the next part starts with a truncated thread"
            )
        if card.thin_handoffs:
            console.print(
                f"                 [yellow]{card.thin_handoffs} part(s) changed files and "
                f"reported almost nothing[/] — the next part started without knowing what "
                f"had been done"
            )
        if card.revisits:
            console.print(
                f"                 [yellow]{len(card.revisits)} revisit(s)[/]: a part reached "
                f"for work another part already owned — {', '.join(card.revisits[:3])}"
            )

    if card.invented:
        console.print(
            f"  wandering    [dim]{len(card.invented)} path(s) in no part's scope "
            f"({', '.join(card.invented[:3])}) — refused, nothing was touched[/]"
        )

    console.print(f"  headroom     peak used {card.peak_fraction:.0%} of a part's ceiling")
    if card.drift_high is not None and card.drift_low is not None:
        # A range, not a number: high is wasted room, and anything under 1.0 means a ceiling
        # was enforced against an estimate below the real prompt, which is not a ceiling.
        span = (
            f"{card.drift_low:.2f}x"
            if round(card.drift_low, 2) == round(card.drift_high, 2)
            else f"{card.drift_low:.2f}-{card.drift_high:.2f}x"
        )
        note = (
            "  [yellow](one or more requests came in UNDER the real prompt)[/]"
            if card.under_counted
            else "  [dim](always above the real prompt: conservative)[/]"
        )
        at_peak = (
            f", {card.drift_at_peak:.2f}x on the largest"
            if card.drift_at_peak is not None
            else ""
        )
        console.print(
            f"  drift        our estimate ran {span} the backend's count "
            f"over {card.drift_samples} request(s){at_peak}{note}"
        )
    if card.nudged_parts or card.abandoned_parts:
        console.print(
            f"  convergence  [dim]{card.nudged_parts} part(s) needed a nudge, "
            f"{card.abandoned_parts} stopped early[/]"
        )
    console.print()


def _render(console: Console, outcome: RunOutcome, *, dry_run: bool, as_json: bool = False) -> int:
    console.print()
    report = outcome.report
    plan = outcome.plan

    if report is not None:
        exactness = "exact" if outcome.counts_exact else "heuristic chars/4"
        console.print(
            f"[bold]Pharos run[/] · {report.model or 'no model'} · tokenizer {exactness}"
        )
        console.print(f"[dim]floor {report.floor:,} · verdict {report.verdict.value}[/]")

    if outcome.error is not None:
        console.print(f"\n[bold red]Did not run:[/] {outcome.error}")
        return 2 if report is None or not outcome.parts else 1

    if not outcome.divided:
        parts = len(plan.parts) if plan is not None else 0
        declined = f" — the plan it declined to use had {parts} parts" if parts else ""
        console.print(f"[yellow]ran undivided (--no-split)[/]{declined}")
    elif plan is not None and plan.mode is not SplitMode.NONE:
        console.print(
            f"[dim]divided into {len(plan.parts)} parts of ≤ {plan.target_per_part:,} "
            f"tokens ({plan.target_label})[/]"
        )
    elif plan is not None and plan.already_fits:
        console.print("[dim]fits in one window — run as a single part[/]")

    if dry_run:
        console.print("\n[yellow]Dry run[/] — nothing was executed and no files were touched.")
        if plan is not None:
            for part in plan.parts:
                scope = ", ".join(f.label() for f in part.files) or "text segment"
                console.print(
                    f"  part {part.index}/{part.total}  "
                    f"≥ {part.projected_tokens:,}  {scope}"
                )
        return 0

    console.print()
    for index, result in enumerate(outcome.parts, start=1):
        status = "[red]failed[/]" if result.error else "[green]done[/]"
        console.print(
            f"part {index}  {status}  {result.steps} tool round(s)  "
            f"peak ~{result.peak_tokens:,} tokens" + _ground_truth(result.reported_tokens)
        )
        if result.stopped_early:
            console.print("  [yellow]window ceiling reached — handed off early[/]")
        if not result.files_written and not result.error:
            # "done" with nothing written is the quiet failure of a small model: it reads its
            # file, describes the change in prose and never calls write_file. Saying so per
            # part is the difference between a run you can trust and a summary that flatters
            # it — the totals at the bottom would otherwise absorb it without a trace.
            console.print("  [yellow]wrote nothing[/] — the model answered without editing")
        for refused in dict.fromkeys(result.scope_refusals):
            console.print(f"  [yellow]blocked out-of-scope read:[/] {refused}")
        if result.error:
            console.print(f"  [red]{result.error}[/]")

    card = score(outcome.parts, handoff_reserve=outcome.handoff_reserve)
    if as_json:
        # stdout belongs to the payload alone, exactly as `pharos check --json` treats it.
        sys.stdout.write(json.dumps(to_dict(card), indent=2) + "\n")
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
