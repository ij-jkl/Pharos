"""`pharos-profile` — detect and print the environment profile, then exit.

Checkpoint-2 deliverable: GPU (or N/A), model / quant / weight size, advertised vs loaded
context + MISMATCH flag, usable budget, VRAM headroom. Runs cleanly even with no GPU and no
reachable backend (this laptop's current double-degraded state).
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pharos.config import ConfigError, load_config
from pharos.console import force_utf8
from pharos.profiler.profiler import build_profile
from pharos.profiler.types import BudgetReport, EnvironmentProfile


def main() -> None:
    """Console-script entry point for `pharos-profile`."""
    force_utf8(sys.stdout, sys.stderr)
    console = Console()
    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        raise SystemExit(2) from exc
    profile = asyncio.run(build_profile(config))
    _render(console, profile)


def _render(console: Console, profile: EnvironmentProfile) -> None:
    console.print()
    console.print(_summary_panel(profile))
    if profile.ctx_mismatch:
        console.print(_mismatch_banner(profile))
    console.print()


def _mib(value: int | None) -> str:
    return f"{value:,} MiB" if value is not None else "N/A"


def _tok(value: int | None) -> str:
    return f"{value:,}" if value is not None else "N/A"


def _summary_panel(profile: EnvironmentProfile) -> Panel:
    gpu = profile.gpu
    backend = profile.backend

    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="bold cyan", no_wrap=True)
    grid.add_column()

    if gpu.available:
        grid.add_row("GPU", f"{gpu.name} · free {_mib(gpu.free_mib)} / {_mib(gpu.total_mib)}")
    else:
        grid.add_row("GPU", f"[dim]N/A — {gpu.detail}[/]")

    if backend.reachable:
        grid.add_row("Backend", f"[green]reachable[/] · {backend.base_url}")
    else:
        grid.add_row("Backend", f"[red]UNREACHABLE[/] · {backend.base_url}")

    grid.add_row("Model", _model_line(profile))
    _add_context_row(grid, profile)
    _add_budget_rows(grid, profile.budget, gpu_available=gpu.available)

    return Panel(
        grid,
        title="[bold]Pharos — environment profile[/]",
        border_style="cyan",
        expand=False,
    )


def _model_line(profile: EnvironmentProfile) -> str:
    backend = profile.backend
    if not backend.model:
        return "[dim]none loaded[/]"
    bits = [backend.model]
    if backend.quantization:
        bits.append(backend.quantization)
    if backend.parameter_size:
        bits.append(backend.parameter_size)
    if backend.architecture:
        bits.append(f"arch {backend.architecture}")
    return " · ".join(bits)


def _add_context_row(grid: Table, profile: EnvironmentProfile) -> None:
    backend = profile.backend
    adv, loaded = backend.advertised_max_ctx, backend.loaded_ctx
    if adv is None and loaded is None:
        return
    line = f"advertised {_tok(adv)}  |  loaded {_tok(loaded)}"
    if profile.ctx_mismatch and profile.ctx_mismatch_ratio is not None:
        line += f"  [bold yellow]← MISMATCH ({profile.ctx_mismatch_ratio:.1%} of advertised)[/]"
    grid.add_row("Context", line)


def _add_budget_rows(grid: Table, budget: BudgetReport, *, gpu_available: bool) -> None:
    if budget.usable_budget is not None:
        detail = f"[dim](loaded {_tok(budget.loaded_ctx)} - reserve {budget.response_reserve})[/]"
        grid.add_row("Usable budget", f"{_tok(budget.usable_budget)} tokens  {detail}")
        pct = f"{budget.warn_threshold:.0%} / {budget.alert_threshold:.0%}"
        grid.add_row(
            "Warn / Alert",
            f"{_tok(budget.warn_tokens)} / {_tok(budget.alert_tokens)} tokens  [dim]({pct})[/]",
        )
    else:
        grid.add_row("Usable budget", "[dim]N/A — no loaded context detected[/]")

    if budget.kv_estimate_mib is not None:
        kv = f"~{budget.kv_estimate_mib:,.0f} MiB for {_tok(budget.loaded_ctx)} ctx"
        grid.add_row("KV cache", f"[dim]{kv} ({kv_provenance(budget)})[/]")

    if gpu_available:
        line = f"{_mib(budget.vram_free_mib)} free [dim](measured)[/]"
        if budget.vram_headroom_tokens is not None:
            source = "KV derived" if budget.kv_rate_derived else "KV configured"
            line += (
                f" · ≈{_tok(budget.vram_headroom_tokens)} more ctx tokens "
                f"[dim](estimate · {source} · "
                f"{budget.vram_safety_margin_mib} MiB margin held back)[/]"
            )
        else:
            line += " · [dim]headroom N/A — no model resident to measure against[/]"
        grid.add_row("VRAM headroom", line)


def kv_provenance(budget: BudgetReport) -> str:
    """Where the KV rate came from — the model's own metadata, or the configured fallback.

    The two differ by 5-14x on real models, and the rate drives the headroom advice, so which
    one produced a figure is part of the figure.
    """
    rate = budget.kv_mib_per_1k
    if rate is None:
        return "estimate"
    if budget.kv_rate_derived:
        return f"derived · {rate:,.0f} MiB/1K from model metadata"
    return f"estimate · configured {rate:,.0f} MiB/1K"


def _mismatch_banner(profile: EnvironmentProfile) -> Panel:
    advertised = profile.backend.advertised_max_ctx
    loaded = profile.backend.loaded_ctx
    ratio = profile.ctx_mismatch_ratio
    assert advertised is not None and loaded is not None and ratio is not None
    body = Text.from_markup(
        f"The model advertises a max context of [bold]{advertised:,}[/] tokens, but only "
        f"[bold]{loaded:,}[/] are actually loaded ([bold]{ratio:.1%}[/] of capacity).\n"
        + mismatch_advice(advertised, profile.budget.achievable_ctx_estimate, rich_markup=True)
    )
    return Panel(body, title="[bold red]⚠  CONTEXT MISMATCH[/]", border_style="red", expand=False)


def mismatch_advice(advertised: int, achievable: int | None, *, rich_markup: bool) -> str:
    """The banner's action line, honest about hardware limits.

    Advising "raise num_ctx to the advertised max" when VRAM cannot hold it is a confidently
    wrong actionable number — following it would OOM or spill to system RAM. When the
    accountant's (margin-adjusted) ceiling is below the advertised window, point at THAT.
    Shared by the CLI and TUI banners so the advice can never diverge between them.
    """
    b = ("[bold]", "[/]") if rich_markup else ("", "")
    if achievable is not None and achievable < advertised:
        return (
            f"Raise Ollama's {b[0]}num_ctx{b[1]} toward ≈{achievable:,} — the hardware ceiling "
            f"on this machine (estimate); the advertised {advertised:,} does not fit in VRAM."
        )
    return (
        f"You are running with far less context than the model supports — raise Ollama's "
        f"{b[0]}num_ctx{b[1]} to use more."
    )
