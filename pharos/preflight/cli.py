"""`pharos check` — render the pre-flight report and exit with a scriptable code.

Exit codes: 0 the floor fits the usable budget · 1 it exceeds · 2 indeterminate (backend
unreachable, no model resident, or a config error). Advisory only: nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from pharos.config import ConfigError, load_config
from pharos.preflight.check import CheckReport, Verdict, run_check

_EXIT_BY_VERDICT = {Verdict.FITS: 0, Verdict.EXCEEDS: 1, Verdict.INDETERMINATE: 2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pharos check",
        description="Check an intended prompt against the context budget before pasting it.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("prompt", nargs="?", help="the prompt text to check")
    group.add_argument("--file", type=Path, help="read the prompt from a file instead")
    args = parser.parse_args(argv)

    console = Console()
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
    else:
        prompt = args.prompt

    report = asyncio.run(run_check(config, prompt))
    _render(console, report)
    return _EXIT_BY_VERDICT[report.verdict]


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
        table.add_row(entry.display, f"{entry.tokens:,}", note)
    if report.overhead is not None:
        table.add_row(
            "client overhead",
            f"~{report.overhead.tokens:,}",
            f"estimate — {report.overhead.provenance}",
        )
    console.print(table)

    console.print()
    console.print(f"[bold]FLOOR   ≥ {report.floor:,} tokens[/]")
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
    if not (ex.directories or ex.ambiguous or ex.missing or report.skipped_binary):
        return
    console.print()
    console.print("[bold]Not counted:[/]")
    for entry in ex.directories:
        console.print(f"  {entry.raw}  [dim]directory — name files explicitly to include them[/]")
    for raw, candidates in ex.ambiguous.items():
        shown = " · ".join(str(c) for c in candidates[:4])
        more = f" (+{len(candidates) - 4} more)" if len(candidates) > 4 else ""
        console.print(f"  {raw}  [dim]ambiguous: {shown}{more}[/]")
    for raw in ex.missing:
        console.print(f"  {raw}  [dim]not found under {report.root}[/]")
    for display in report.skipped_binary:
        console.print(f"  {display}  [dim]binary — no text token count exists for it[/]")
