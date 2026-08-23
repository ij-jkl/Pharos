"""`pharos check` / `pharos split` — pre-flight a prompt, and cut it up when it does not fit.

Exit codes for `check`: 0 the floor fits the usable budget · 1 it exceeds · 2 indeterminate
(backend unreachable, no model resident, or a config error). For `split` (and `check --split`):
0 a plan was produced whose every part fits · 1 no plan, or a part that still does not fit ·
2 indeterminate. Advisory only: neither command changes a file, and nothing is sent
anywhere unless you pass --semantic, which asks the configured backend to group the
files and sends it the task, the filenames and the first five lines of each.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pharos.config import ConfigError, load_config
from pharos.console import force_utf8
from pharos.preflight.check import CheckReport, CountedFile, Verdict, run_check
from pharos.preflight.extract import ALL_CANDIDATES, DIRECTORY_FILE_CAP
from pharos.preflight.split import Grouping, SplitMode, SplitPlan, build_plan

_EXIT_BY_VERDICT = {Verdict.FITS: 0, Verdict.EXCEEDS: 1, Verdict.INDETERMINATE: 2}
# Bumped when a field is removed or its meaning changes, so a wrapper can refuse politely
# rather than silently misread a number. Added fields do not bump it.
_JSON_SCHEMA_VERSION = 1


def main(argv: list[str] | None = None, *, always_split: bool = False) -> int:
    # Before argparse: --help and usage errors are rendered too, and the epilog carries an
    # em dash, so a redirected `--help` would crash before reaching any of the work below.
    force_utf8(sys.stdout, sys.stderr)

    prog = "pharos split" if always_split else "pharos check"
    description = (
        "Cut a prompt that does not fit into ordered sub-prompts that do."
        if always_split
        else "Check an intended prompt against the context budget before pasting it."
    )
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog="With no prompt and no --file, the prompt is read from stdin — pipe it, or "
        "paste it and end with Ctrl-Z Enter (Windows) / Ctrl-D (Unix).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("prompt", nargs="?", help="the prompt text to check")
    group.add_argument("--file", type=Path, help="read the prompt from a file instead")
    if not always_split:
        parser.add_argument(
            "--split",
            action="store_true",
            help="when the floor exceeds the budget, print a plan of sub-prompts that fit",
        )
    parser.add_argument(
        "--target",
        type=int,
        help="judge against this many tokens instead of the live budget — the whole prompt "
        "for a check, each part for a split (lets you work with no backend running)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="write each part to DIR/part-01.txt … ready to paste one at a time",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the part bodies (pipe-friendly); suppresses the report",
    )
    parser.add_argument(
        "--resolve",
        action="append",
        metavar="REF=PATH",
        default=[],
        help="answer an ambiguous reference, e.g. --resolve utils.py=tests/utils.py, or "
        "--resolve utils.py=* to count every candidate (repeatable)",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="choose interactively between the candidates of each ambiguous reference",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="ask the backend which files belong together instead of packing them in the "
        "order they were named; falls back to position packing and says so if the proposal "
        "fails any check. Sends the task, the filenames and the first five lines of each "
        "file to the backend"
        + ("" if always_split else " (requires --split)"),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the report (and plan) as JSON on stdout instead of a table",
    )
    args = parser.parse_args(argv)
    want_split = always_split or getattr(args, "split", False)
    # An accepted flag that does nothing is worse than a rejected one: it looks like it worked.
    if args.semantic and not want_split:
        parser.error("--semantic groups the parts of a split, so it needs --split")

    # With --json, stdout belongs to the payload alone: prompts, warnings and the picker menu
    # go to stderr, or a caller piping into `jq` gets a parse error instead of a report.
    console = Console(stderr=args.json)
    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        return 2

    if args.file is not None:
        try:
            prompt = args.file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            console.print(f"[bold red]Cannot read prompt file:[/] {exc}")
            return 2
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = _read_stdin(console)
        if prompt is None:
            return 2

    try:
        resolve = _parse_resolve(args.resolve)
    except ValueError as exc:
        console.print(f"[bold red]Bad --resolve:[/] {exc}")
        return 2

    from_stdin = args.file is None and args.prompt is None
    skip_profile = args.target is not None
    report = asyncio.run(
        run_check(
            config, prompt, skip_profile=skip_profile, resolve=resolve, target=args.target
        )
    )

    if args.pick and report.extraction.ambiguous:
        picked = _pick_interactively(console, report, from_stdin=from_stdin)
        if picked:
            resolve = {**resolve, **picked}
            report = asyncio.run(
                # Reuse the probe already paid for: re-running the check must not mean a
                # second round-trip to the backend (which can mean a second model load).
                run_check(
                    config,
                    prompt,
                    profile=report.profile,
                    skip_profile=skip_profile,
                    resolve=resolve,
                    target=args.target,
                )
            )

    quiet_output = args.json or (want_split and args.quiet)
    if not quiet_output:
        _render(console, report)

    if not want_split:
        if args.json:
            _emit_json(report_to_dict(report))
        elif report.verdict is Verdict.EXCEEDS:
            console.print(
                "[dim]Run the same prompt through `pharos split` (or add --split) to cut it "
                "into sub-prompts that fit.[/]\n"
            )
        return _EXIT_BY_VERDICT[report.verdict]

    plan = build_plan(config, prompt, report, target=args.target, semantic=args.semantic)
    if args.json:
        _emit_json({**report_to_dict(report), "plan": plan_to_dict(plan)})
    elif args.quiet:
        for part in plan.parts:
            console.print(part.body, markup=False, highlight=False)
        if not plan.parts:
            console.print(f"pharos: no plan — {plan.reason}", markup=False)
    else:
        _render_plan(console, plan)
    if args.out is not None and plan.parts:
        _write_parts(console, plan, args.out, quiet=quiet_output)

    if plan.indeterminate:
        return 2
    return 0 if plan.ok else 1


def _read_stdin(console: Console) -> str | None:
    """Read the prompt from stdin: piped input, or an interactive paste ended with EOF.

    Pasting is the motion this tool exists for, and shell-quoting a multi-line prompt is the
    friction it is supposed to remove. An empty read is an error, not an empty prompt — a
    verdict on nothing would be a confident, useless number.
    """
    if sys.stdin is None:
        console.print("[bold red]No prompt:[/] give one as an argument, or use --file.")
        return None
    if sys.stdin.isatty():
        console.print(
            "[dim]Paste the prompt, then press Ctrl-Z Enter (Windows) or Ctrl-D (Unix).[/]"
        )
    try:
        text = sys.stdin.read()
    except (KeyboardInterrupt, UnicodeDecodeError) as exc:
        console.print(f"[bold red]Could not read the prompt from stdin:[/] {exc}")
        return None
    if not text.strip():
        console.print(
            "[bold red]Empty prompt.[/] Pass it as an argument, with --file, or on stdin."
        )
        return None
    return text


def split_main(argv: list[str] | None = None) -> int:
    """`pharos split` — the same machinery, always splitting."""
    return main(argv, always_split=True)


def _parse_resolve(pairs: list[str]) -> dict[str, str]:
    """Parse ``--resolve REF=PATH`` flags. Splits on the FIRST '=' — paths may contain more."""
    out: dict[str, str] = {}
    for pair in pairs:
        raw, sep, path = pair.partition("=")
        if not sep or not raw.strip() or not path.strip():
            raise ValueError(f"expected REF=PATH, got {pair!r}")
        out[raw.strip()] = path.strip()
    return out


def _pick_interactively(
    console: Console,
    report: CheckReport,
    *,
    from_stdin: bool,
    ask: Callable[[str], str] | None = None,
) -> dict[str, str]:
    """Ask which candidate each ambiguous reference meant. Returns raw -> chosen path.

    ``ask`` is injected so the prompt/answer loop can be driven in a test without patching
    builtins — the behaviour under test is the loop, not Python's ``input``. It defaults to
    None rather than to ``input`` itself: a default argument binds at definition time, which
    would quietly capture the original builtin and make ``input`` unpatchable for everyone
    else.
    """
    ask = ask or input
    if from_stdin:
        console.print(
            "[yellow]--pick needs a terminal to ask on, and the prompt came from stdin.[/] "
            "Use --resolve REF=PATH instead."
        )
        return {}
    if not sys.stdin.isatty():
        console.print("[yellow]--pick needs an interactive terminal.[/] Use --resolve REF=PATH.")
        return {}

    chosen: dict[str, str] = {}
    for raw, candidates in report.extraction.ambiguous.items():
        console.print(f"\n[bold]{raw}[/] matches {len(candidates)} files:")
        for i, candidate in enumerate(candidates, start=1):
            console.print(f"  [bold]{i}[/]  {_display(candidate, report.root)}")
        console.print(f"  [bold]a[/]  all {len(candidates)}")
        try:
            answer = ask(f"  which one? [1-{len(candidates)}, a=all, blank to skip] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]cancelled — the rest stay uncounted[/]")
            break
        if answer.lower() == "a":
            chosen[raw] = ALL_CANDIDATES
            continue
        if not answer.isdigit() or not 1 <= int(answer) <= len(candidates):
            console.print("  [dim]skipped — it stays uncounted[/]")
            continue
        chosen[raw] = str(candidates[int(answer) - 1])
    return chosen


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _emit_json(payload: dict[str, object]) -> None:
    """Write JSON straight to stdout: Rich would wrap long lines and corrupt it."""
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")


def report_to_dict(report: CheckReport) -> dict[str, object]:
    """The pre-flight report as plain data — every number carrying its provenance."""
    budget = report.profile.budget if report.profile is not None else None
    return {
        "schema": _JSON_SCHEMA_VERSION,
        "root": str(report.root),
        "model": report.model,
        "counts_exact": report.counts_exact,
        "tokenizer": "gguf" if report.counts_exact else "heuristic-chars-4",
        "prompt_tokens": report.prompt_tokens,
        "files": [_file_to_dict(f) for f in report.files],
        "directories": [
            {
                "path": d.display,
                "tokens": d.tokens,
                "truncated": d.truncated,
                "skipped_non_text": d.skipped_non_text,
                "files": [_file_to_dict(f) for f in d.files],
            }
            for d in report.directories
        ],
        "overhead": (
            None
            if report.overhead is None
            else {"tokens": report.overhead.tokens, "provenance": report.overhead.provenance}
        ),
        "floor": report.floor,
        "ceiling": report.ceiling,
        "budget": (
            None
            if budget is None or budget.usable_budget is None
            else {
                "usable": budget.usable_budget,
                "loaded_ctx": budget.loaded_ctx,
                "response_reserve": budget.response_reserve,
                "warn_tokens": budget.warn_tokens,
            }
        ),
        "verdict": report.verdict.value,
        "verdict_detail": report.verdict_detail,
        "warnings": [w for w in (report.directory_warning, report.reserve_warning) if w],
        "uncounted": {
            "ambiguous": {
                raw: [str(p) for p in paths]
                for raw, paths in report.extraction.ambiguous.items()
            },
            "missing": list(report.extraction.missing),
            "binary": list(report.skipped_binary),
        },
    }


def _file_to_dict(counted: CountedFile) -> dict[str, object]:
    return {
        "path": counted.display,
        "tokens": counted.tokens,
        "found_by_search": counted.found_by_search,
        "note": counted.note,
    }


def plan_to_dict(plan: SplitPlan) -> dict[str, object]:
    """The split plan as plain data, part bodies included so a wrapper can send them."""
    return {
        "mode": plan.mode.value,
        "ok": plan.ok,
        "already_fits": plan.already_fits,
        "indeterminate": plan.indeterminate,
        "reason": plan.reason,
        "notes": list(plan.notes),
        "target_per_part": plan.target_per_part,
        "target_label": plan.target_label,
        "handoff_reserve": plan.handoff_reserve,
        "grouping": plan.grouping.value,
        "grouping_note": plan.grouping_note,
        "parts": [
            {
                "index": part.index,
                "total": part.total,
                "title": part.title,
                "projected_tokens": part.projected_tokens,
                "fits": part.fits,
                "over_by": part.over_by,
                "files": [
                    {
                        "path": f.display,
                        "tokens": f.tokens,
                        "line_start": f.line_start,
                        "line_end": f.line_end,
                        "total_lines": f.total_lines,
                        "source_dir": f.source_dir,
                    }
                    for f in part.files
                ],
                "body": part.body,
            }
            for part in plan.parts
        ],
    }


def _render(console: Console, report: CheckReport) -> None:
    console.print()
    exactness = "gguf (exact)" if report.counts_exact else "heuristic chars/4 — no GGUF resolved"
    header = f"[bold]Pharos pre-flight[/] · {report.model or 'no model'} · tokenizer {exactness}"
    console.print(header)
    if report.root_is_fallback:
        console.print(
            f"[yellow]target_folder is not set — resolving paths against the current "
            f"directory ({report.root})[/]"
        )
    console.print()

    table = Table.grid(padding=(0, 2))
    table.add_column()
    table.add_column(justify="right")
    table.add_column(style="dim")

    label = "exact" if report.counts_exact else "heuristic"
    table.add_row("prompt text", f"{report.prompt_tokens:,}", label)
    for entry in report.files:
        note = f"{label} · found by search" if entry.found_by_search else label
        if entry.note:
            note = f"{note} · {entry.note}"
        table.add_row(entry.display, f"{entry.tokens:,}", note)
    if report.overhead is not None:
        table.add_row(
            "client overhead",
            f"~{report.overhead.tokens:,}",
            f"estimate — {report.overhead.provenance}",
        )
    for directory in report.directories:
        detail = f"estimate — {len(directory.files)} text files"
        if directory.truncated:
            detail += f", capped at {DIRECTORY_FILE_CAP}"
        if directory.skipped_non_text:
            detail += f", {directory.skipped_non_text} non-text skipped"
        table.add_row(
            f"{directory.display} (directory, if fully read)", f"+{directory.tokens:,}", detail
        )
    console.print(table)

    console.print()
    console.print(f"[bold]FLOOR   ≥ {report.floor:,} tokens[/]")
    if report.directory_tokens:
        console.print(
            f"[bold]CEILING ≤ {report.ceiling:,} tokens[/] "
            f"[dim]if every named directory is read in full[/]"
        )
    lower_bound_note = (
        "A lower bound: exact for the files named above; the agent may read files "
        "it was not told about, and those are not in this number."
        if report.counts_exact
        else "A lower bound, and heuristic at that: no GGUF tokenizer resolved, so every "
        "count above is chars/4."
    )
    console.print(f"[dim]{lower_bound_note}[/]")
    if report.overhead is None:
        console.print(
            "[dim]Client overhead unknown — run traffic through the Pharos proxy to calibrate "
            "it, or set client_overhead_tokens in pharos.toml. The floor omits it.[/]"
        )

    console.print()
    _render_budget(console, report)
    if report.directory_warning:
        console.print(f"[yellow]⚠ {report.directory_warning}[/]")
    if report.reserve_warning:
        console.print(f"[yellow]⚠ {report.reserve_warning}[/]")
    _render_uncounted(console, report)
    console.print()


def _render_budget(console: Console, report: CheckReport) -> None:
    budget = report.profile.budget if report.profile is not None else None
    if budget is not None and budget.usable_budget is not None:
        console.print(
            f"Budget    {budget.usable_budget:,} usable  "
            f"[dim](loaded {budget.loaded_ctx:,} − reserve {budget.response_reserve:,})[/]"
        )
    styles = {
        Verdict.FITS: ("green", "FITS"),
        Verdict.EXCEEDS: ("bold red", "EXCEEDS"),
        Verdict.INDETERMINATE: ("yellow", "NO VERDICT"),
    }
    style, word = styles[report.verdict]
    console.print(Text.assemble("Verdict   ", (word, style), f" — {report.verdict_detail}"))


def _render_uncounted(console: Console, report: CheckReport) -> None:
    ex = report.extraction
    empty_dirs = [d for d in report.directories if not d.files]
    if not (empty_dirs or ex.ambiguous or ex.missing or report.skipped_binary):
        return
    console.print()
    console.print("[bold]Not counted:[/]")
    for directory in empty_dirs:
        console.print(
            f"  {directory.display}  [dim]directory — no text files under it "
            f"({directory.skipped_non_text} non-text skipped)[/]"
        )
    for raw, candidates in ex.ambiguous.items():
        shown = " · ".join(_display(c, report.root) for c in candidates[:4])
        more = f" (+{len(candidates) - 4} more)" if len(candidates) > 4 else ""
        console.print(f"  {raw}  [dim]ambiguous: {shown}{more}[/]")
    if ex.ambiguous:
        example = next(iter(ex.ambiguous))
        console.print(
            f"  [dim]resolve with --pick, "
            f"--resolve {example}={_display(ex.ambiguous[example][0], report.root)}, "
            f"or --resolve {example}=* to count them all[/]"
        )
    for raw in ex.missing:
        console.print(f"  {raw}  [dim]not found under {report.root}[/]")
    for display in report.skipped_binary:
        console.print(f"  {display}  [dim]binary — no text token count exists for it[/]")


_MODE_HEADLINE = {
    SplitMode.SCOPE: (
        "Split by SCOPE — every part repeats the task and narrows it to a subset of the "
        "files. Run them in order, pasting each part's hand-off above the next."
    ),
    SplitMode.TEXT: (
        "Split by TEXT — the pasted text itself is what overflows, so it is cut into ordered "
        "segments. Paste them in order into the same conversation; only the last one asks "
        "for the work."
    ),
}


def _render_plan(console: Console, plan: SplitPlan) -> None:
    console.print()
    if plan.mode is SplitMode.NONE:
        style = "yellow" if plan.already_fits else "bold red"
        console.print(f"[{style}]No split plan[/] — {plan.reason}")
        for note in plan.notes:
            console.print(f"[dim]  · {note}[/]")
        console.print()
        return

    console.print(
        f"[bold]Split plan[/] · {len(plan.parts)} parts · "
        f"≤ {plan.target_per_part:,} tokens each [dim]({plan.target_label})[/]"
    )
    console.print(f"[dim]{_MODE_HEADLINE[plan.mode]}[/]")
    if plan.grouping_note is not None:
        # Whichever way it went. A plan that quietly used a model, or quietly did not, is the
        # one outcome --semantic is not allowed to have.
        by_model = plan.grouping is Grouping.SEMANTIC
        marker = "grouped by meaning" if by_model else "grouped by position"
        style = "cyan" if by_model else "yellow"
        console.print(f"[{style}]{marker}[/] [dim]— {plan.grouping_note}[/]")
    if plan.handoff_reserve:
        console.print(
            f"[dim]Each part holds back {plan.handoff_reserve:,} tokens for the hand-off "
            f"pasted above it, and parts 2+ count it.[/]"
        )
    console.print()

    table = Table.grid(padding=(0, 2))
    table.add_column()
    table.add_column(justify="right")
    table.add_column(style="dim")
    for part in plan.parts:
        scope = ", ".join(f.label() for f in part.files) or "text segment"
        status = "" if part.fits else f"STILL OVER by {part.over_by:,}"
        table.add_row(
            f"part {part.index}/{part.total}",
            f"≥ {part.projected_tokens:,}",
            f"{scope}{'  ' + status if status else ''}",
        )
    console.print(table)

    console.print()
    console.print(
        "[dim]Each projection is a floor on the same terms as the verdict above: it holds "
        "while the agent stays inside the part's scope.[/]"
    )
    for note in plan.notes:
        console.print(f"[yellow]⚠ {note}[/]")
    if not plan.ok:
        console.print(
            "[bold red]⚠ At least one part still exceeds the budget[/] — it cannot be cut "
            "any smaller without splitting inside a line."
        )

    console.print()
    for part in plan.parts:
        console.print(
            Panel(
                Text(part.body),
                title=(
                    f"part {part.index} of {part.total}"
                    + (f" - {part.title}" if part.title else "")
                ),
                title_align="left",
                border_style="green" if part.fits else "red",
            )
        )
    console.print()


def _write_parts(console: Console, plan: SplitPlan, out: Path, *, quiet: bool) -> None:
    try:
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for part in plan.parts:
            path = out / f"part-{part.index:02d}.txt"
            path.write_text(part.body, encoding="utf-8")
            written.append(path)
    except OSError as exc:
        console.print(f"[bold red]Could not write parts:[/] {exc}")
        return
    if not quiet:
        console.print(f"Wrote {len(written)} parts to {out}")
        console.print()
