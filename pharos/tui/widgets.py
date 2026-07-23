"""TUI widgets: stats header, mismatch banner, context/VRAM gauges, throughput, event log.

Honest-labeling rule: a token count is rendered with its provenance everywhere it appears —
exact counts plain green, estimates prefixed "~" in yellow with their source — because a
number shown without its label is a confident wrong number. All widgets keep a plain-text
mirror (``last_text`` / ``plain``) so tests can assert on rendered content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.widgets import RichLog, Static

from pharos.profiler.types import BudgetReport, EnvironmentProfile, GpuInfo

_BAR_WIDTH = 30
_LOG_MIRROR_LINES = 1000


@dataclass(frozen=True, slots=True)
class ContextUsage:
    """The latest known context usage: a token count plus the provenance of that number."""

    tokens: int
    exact: bool
    source: str  # "gguf" | "request" | "heuristic" | "reconciled"


def token_label(tokens: int, *, exact: bool, source: str) -> Text:
    """Render a count with its provenance — the one place the estimate/exact marking lives."""
    if exact:
        suffix = "exact" if source == "reconciled" else f"exact · {source}"
        return Text.assemble((f"{tokens:,}", "bold green"), (f" ({suffix})", "green"))
    if source == "heuristic":
        return Text.assemble((f"~{tokens:,}", "bold yellow"), (" (heuristic chars/4)", "yellow"))
    return Text.assemble((f"~{tokens:,}", "bold yellow"), (f" (estimate · {source})", "yellow"))


def render_bar(fraction: float | None, *, warn: float, alert: float) -> Text:
    """A threshold-colored block bar; a dim rule when the fraction is unknowable."""
    if fraction is None:
        return Text("─" * _BAR_WIDTH, style="dim")
    fraction = max(0.0, min(fraction, 1.0))
    filled = round(fraction * _BAR_WIDTH)
    style = "green"
    if fraction >= alert:
        style = "red"
    elif fraction >= warn:
        style = "yellow"
    bar = Text()
    bar.append("█" * filled, style=style)
    bar.append("░" * (_BAR_WIDTH - filled), style="grey37")
    return bar


class StatsHeader(Static):
    """Proxy address, backend reachability, model / quant / arch, GPU name."""

    last_text: str = ""

    def show(self, profile: EnvironmentProfile | None, proxy_addr: str) -> None:
        text = Text()
        text.append("PHAROS ", style="bold cyan")
        text.append(f"listening {proxy_addr}   ", style="dim")
        if profile is None:
            text.append("probing environment…", style="dim")
        else:
            backend = profile.backend
            text.append("backend ", style="dim")
            if backend.reachable:
                text.append("● reachable", style="bold green")
            else:
                text.append("✖ UNREACHABLE", style="bold red")
            text.append(f" {backend.base_url}", style="dim")
            text.append("\n")
            bits = [backend.model or "no model loaded"]
            if backend.quantization:
                bits.append(backend.quantization)
            if backend.parameter_size:
                bits.append(backend.parameter_size)
            if backend.architecture:
                bits.append(f"arch {backend.architecture}")
            text.append(" · ".join(bits))
            gpu = profile.gpu
            if gpu.available:
                text.append(f"   GPU {gpu.name}", style="cyan")
            else:
                text.append("   GPU N/A", style="dim")
        self.last_text = text.plain
        self.update(text)


class MismatchBanner(Static):
    """The headline feature, persistently visible: advertised vs actually-loaded context."""

    last_text: str = ""

    def show(self, profile: EnvironmentProfile | None) -> None:
        backend = profile.backend if profile is not None else None
        if (
            profile is None
            or not profile.ctx_mismatch
            or backend is None
            or backend.advertised_max_ctx is None
            or backend.loaded_ctx is None
        ):
            self.last_text = ""
            self.update("")
            self.set_class(False, "visible")
            return
        ratio = profile.ctx_mismatch_ratio
        pct = f" ({ratio:.1%} of capacity)" if ratio is not None else ""
        text = Text(
            f"⚠ CONTEXT MISMATCH — advertised {backend.advertised_max_ctx:,} · "
            f"loaded {backend.loaded_ctx:,}{pct} — raise num_ctx to use the full window",
            style="bold",
        )
        self.last_text = text.plain
        self.update(text)
        self.set_class(True, "visible")


class ContextGauge(Static):
    """Context usage against the usable budget, with the count's provenance always shown."""

    last_text: str = ""

    def show(self, usage: ContextUsage | None, budget: BudgetReport | None) -> None:
        text = Text()
        text.append("CTX  ", style="bold cyan")
        if usage is None:
            text.append("no requests yet", style="dim")
        else:
            text.append_text(token_label(usage.tokens, exact=usage.exact, source=usage.source))
        usable = budget.usable_budget if budget is not None else None
        if budget is not None and usable:
            fraction = (usage.tokens / usable) if usage is not None else 0.0
            text.append("  ")
            text.append_text(
                render_bar(fraction, warn=budget.warn_threshold, alert=budget.alert_threshold)
            )
            text.append(
                f"  / {usable:,} usable · warn {budget.warn_tokens:,}"
                f" · alert {budget.alert_tokens:,}",
                style="dim",
            )
        else:
            text.append("  · usable budget N/A — no loaded context detected", style="dim")
        self.last_text = text.plain
        self.update(text)


class VramGauge(Static):
    """Measured VRAM plus the estimated token headroom that makes it actionable."""

    last_text: str = ""

    def show(self, gpu: GpuInfo | None, budget: BudgetReport | None) -> None:
        text = Text()
        text.append("VRAM ", style="bold cyan")
        if gpu is None or not gpu.available:
            text.append("N/A — no NVIDIA GPU detected", style="dim")
        else:
            used = gpu.used_mib
            total = gpu.total_mib
            fraction = (used / total) if used is not None and total else None
            text.append(f"{used:,} / {total:,} MiB used  " if used and total else "")
            text.append_text(render_bar(fraction, warn=0.80, alert=0.90))
            if gpu.free_mib is not None:
                text.append(f"  free {gpu.free_mib:,} MiB (measured)", style="dim")
            headroom = budget.vram_headroom_tokens if budget is not None else None
            if headroom is not None:
                text.append(f" · ≈{headroom:,} more ctx tokens fit", style="yellow")
                text.append(" (estimate)", style="dim yellow")
        self.last_text = text.plain
        self.update(text)


class ThroughputLine(Static):
    """tok/s, the last request's outcome, and the running dropped-events total."""

    last_text: str = ""

    def show(self, tok_s: float | None, last: Text | None, dropped_total: int) -> None:
        text = Text()
        text.append("RATE ", style="bold cyan")
        text.append(f"{tok_s:.1f} tok/s" if tok_s is not None else "— tok/s")
        text.append("   last: ", style="dim")
        text.append_text(last if last is not None else Text("—", style="dim"))
        if dropped_total:
            text.append(f"   ⚠ {dropped_total} events dropped total", style="bold yellow")
        self.last_text = text.plain
        self.update(text)


class EventLog(RichLog):
    """Scrolling event log; keeps a bounded plain-text mirror for tests."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(max_lines=_LOG_MIRROR_LINES, wrap=False, markup=False, **kwargs)
        self.plain: list[str] = []

    def line(self, text: Text) -> None:
        self.plain.append(text.plain)
        if len(self.plain) > _LOG_MIRROR_LINES:
            del self.plain[: len(self.plain) - _LOG_MIRROR_LINES]
        self.write(text)
