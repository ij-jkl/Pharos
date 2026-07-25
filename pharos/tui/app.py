"""The Textual dashboard: a pure event-bus consumer beside an out-of-band profiler loop.

The TUI must never backpressure the proxy: it reads from its own bounded bus queue, and the
bus drops the oldest events for a lagging consumer. Any such gap is surfaced honestly — a
warning line in the log plus a running total in the RATE line — never hidden. Environment
facts (GPU, backend, budget) come from periodic profiler probes, off the request path.
"""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Footer

from pharos.config import PharosConfig
from pharos.events import (
    EventBus,
    InputCounted,
    PharosEvent,
    RequestAborted,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
)
from pharos.profiler.profiler import build_profile
from pharos.profiler.types import EnvironmentProfile
from pharos.tui.widgets import (
    ContextGauge,
    ContextUsage,
    EventLog,
    MismatchBanner,
    StatsHeader,
    ThroughputLine,
    VramGauge,
    token_label,
)

_PROFILE_REFRESH_S = 5.0


class PharosApp(App[None]):
    """htop-style dashboard for the proxy: gauges on top, scrolling event log below."""

    CSS_PATH = "styles.tcss"
    TITLE = "Pharos"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, config: PharosConfig, bus: EventBus) -> None:
        super().__init__()
        self._config = config
        self._bus = bus
        self._events = bus.subscribe()
        self._proxy_addr = f"{config.proxy_host}:{config.proxy_port}"
        self._profile: EnvironmentProfile | None = None
        self._usage: ContextUsage | None = None
        self._tok_s: float | None = None
        self._last_summary: Text | None = None
        self._dropped_total = 0

    def compose(self) -> ComposeResult:
        yield StatsHeader(id="stats")
        yield MismatchBanner(id="mismatch")
        yield ContextGauge(id="ctx")
        yield VramGauge(id="vram")
        yield ThroughputLine(id="throughput")
        yield EventLog(id="log")
        yield Footer()

    def on_mount(self) -> None:
        self._render_all()
        self.run_worker(self._consume_events(), group="events")
        self.run_worker(self._refresh_profile(), group="profile", exclusive=True)
        self.set_interval(_PROFILE_REFRESH_S, self._schedule_profile_refresh)

    # --- out-of-band environment probing ---------------------------------------------------

    def _schedule_profile_refresh(self) -> None:
        self.run_worker(self._refresh_profile(), group="profile", exclusive=True)

    async def _refresh_profile(self) -> None:
        self._profile = await build_profile(self._config)
        self._render_all()

    # --- event consumption (never blocks the proxy; it publishes fire-and-forget) ----------

    async def _consume_events(self) -> None:
        while True:
            event = await self._events.get()
            self._handle_event(event)
            dropped = self._bus.dropped_count(self._events)
            if dropped > self._dropped_total:
                lost = dropped - self._dropped_total
                self._dropped_total = dropped
                self._log_line(
                    Text.assemble(
                        ("⚠ ", "bold yellow"),
                        (
                            f"{lost} event(s) dropped — UI fell behind "
                            "(drop-oldest queue); totals may lag",
                            "yellow",
                        ),
                    )
                )
                self._render_gauges()

    def _handle_event(self, event: PharosEvent) -> None:
        if isinstance(event, RequestStarted):
            line = Text.assemble(
                (f"#{event.request_id} ", "bold"),
                ("→ ", "cyan"),
                (f"{event.method} {event.path}", ""),
            )
            extra = f" · {event.model}" if event.model else ""
            if event.stream:
                extra += " · stream"
            line.append(extra, "dim")
            self._log_line(line)
        elif isinstance(event, InputCounted):
            self._usage = ContextUsage(tokens=event.tokens, exact=event.exact, source=event.source)
            line = Text.assemble((f"#{event.request_id} ", "bold"), ("✎ input ", "cyan"))
            line.append_text(token_label(event.tokens, exact=event.exact, source=event.source))
            self._log_line(line)
            self._render_gauges()
        elif isinstance(event, RequestCompleted):
            self._apply_completion(event)
        elif isinstance(event, RequestAborted):
            self._last_summary = Text(f"#{event.request_id} ✕ ABORTED", style="bold red")
            self._log_line(
                Text.assemble(
                    (f"#{event.request_id} ", "bold"),
                    ("✕ ABORTED", "bold red"),
                    (
                        f" after {event.duration_s:.1f}s — stream ended early "
                        f"(status {event.status_code} was already sent)",
                        "red",
                    ),
                )
            )
            self._render_gauges()
        elif isinstance(event, RequestFailed):
            self._last_summary = Text(f"#{event.request_id} ✕ failed", style="red")
            self._log_line(
                Text.assemble(
                    (f"#{event.request_id} ", "bold"),
                    ("✕ failed: ", "bold red"),
                    (event.error, "red"),
                )
            )
            self._render_gauges()

    def _apply_completion(self, event: RequestCompleted) -> None:
        if event.prompt_eval_count is not None:
            total = event.prompt_eval_count + (event.eval_count or 0)
            self._usage = ContextUsage(tokens=total, exact=True, source="reconciled")
        elif self._usage is not None and event.eval_count:
            # No reconciliation reported: fold the output into the standing estimate.
            self._usage = ContextUsage(
                tokens=self._usage.tokens + event.eval_count,
                exact=False,
                source=self._usage.source,
            )
        if event.tokens_per_second is not None:
            self._tok_s = event.tokens_per_second

        # A 4xx/5xx is a completed *rejection*, not a success: it must never wear the green ✓.
        ok = event.status_code < 400
        glyph, style = ("✓ ", "bold green") if ok else ("✕ ", "bold red")
        line = Text.assemble(
            (f"#{event.request_id} ", "bold"),
            (glyph, style),
            (f"{event.status_code}", "green" if ok else "red"),
            (" · in ", "dim"),
        )
        if event.prompt_eval_count is not None:
            line.append_text(token_label(event.prompt_eval_count, exact=True, source="reconciled"))
        else:
            line.append("— not reported; estimate stands", "yellow")
        line.append(" · out ", "dim")
        line.append(f"{event.eval_count:,}" if event.eval_count is not None else "—")
        if event.tokens_per_second is not None:
            line.append(f" · {event.tokens_per_second:.1f} tok/s", "cyan")
        line.append(f" · {event.duration_s:.2f}s", "dim")
        summary_glyph, summary_style = ("✓", "green") if ok else ("✕", "bold red")
        self._last_summary = Text(
            f"#{event.request_id} {summary_glyph} {event.status_code}", style=summary_style
        )
        self._log_line(line)
        self._render_gauges()

    # --- rendering --------------------------------------------------------------------------

    def _render_all(self) -> None:
        self.query_one(StatsHeader).show(self._profile, self._proxy_addr)
        self.query_one(MismatchBanner).show(self._profile)
        self._render_gauges()

    def _render_gauges(self) -> None:
        budget = self._profile.budget if self._profile is not None else None
        gpu = self._profile.gpu if self._profile is not None else None
        self.query_one(ContextGauge).show(self._usage, budget)
        self.query_one(VramGauge).show(gpu, budget)
        self.query_one(ThroughputLine).show(self._tok_s, self._last_summary, self._dropped_total)

    def _log_line(self, text: Text) -> None:
        line = Text(datetime.now().strftime("%H:%M:%S") + " ", style="dim")
        line.append_text(text)
        self.query_one(EventLog).line(line)
