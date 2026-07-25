"""TUI tests (headless, via Textual's pilot): double degradation, honest count labeling,
the persistent mismatch banner, aborted-request rendering, and dropped-event surfacing.

The profiler is stubbed — these tests must not touch the network or the host GPU."""

from __future__ import annotations

import pytest

from pharos.accountant import Accountant
from pharos.config import PharosConfig
from pharos.events import (
    EventBus,
    InputCounted,
    RequestAborted,
    RequestCompleted,
    RequestStarted,
)
from pharos.profiler.types import BackendInfo, EnvironmentProfile, GpuInfo
from pharos.tui.app import PharosApp
from pharos.tui.widgets import (
    ContextGauge,
    EventLog,
    MismatchBanner,
    StatsHeader,
    ThroughputLine,
    VramGauge,
)

CONFIG = PharosConfig()


def degraded_profile() -> EnvironmentProfile:
    """This laptop: no NVIDIA GPU, no reachable backend."""
    gpu = GpuInfo(available=False, detail="no NVIDIA GPU detected")
    backend = BackendInfo(reachable=False, base_url="http://localhost:11434", detail="unreachable")
    budget = Accountant(CONFIG).report(loaded_ctx=None, gpu=gpu)
    return EnvironmentProfile(
        gpu=gpu, backend=backend, budget=budget, ctx_mismatch=False, ctx_mismatch_ratio=None
    )


def mismatched_profile() -> EnvironmentProfile:
    """The desktop scenario: GPU present, model loaded far below its advertised window."""
    gpu = GpuInfo(
        available=True,
        name="NVIDIA GeForce RTX 3060",
        total_mib=12288,
        used_mib=4288,
        free_mib=8000,
        source="nvml",
    )
    backend = BackendInfo(
        reachable=True,
        base_url="http://localhost:11434",
        model="qwen3.5:9b",
        architecture="qwen3",
        quantization="Q4_K_M",
        parameter_size="9.0B",
        advertised_max_ctx=262144,
        loaded_ctx=32768,
    )
    budget = Accountant(CONFIG).report(loaded_ctx=32768, gpu=gpu)
    return EnvironmentProfile(
        gpu=gpu,
        backend=backend,
        budget=budget,
        ctx_mismatch=True,
        ctx_mismatch_ratio=32768 / 262144,
    )


def _patch_profile(monkeypatch: pytest.MonkeyPatch, profile: EnvironmentProfile) -> None:
    async def fake_build_profile(config: PharosConfig) -> EnvironmentProfile:
        return profile

    monkeypatch.setattr("pharos.tui.app.build_profile", fake_build_profile)


async def test_tui_runs_under_double_degradation(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_profile(monkeypatch, degraded_profile())
    app = PharosApp(config=CONFIG, bus=EventBus())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        assert "UNREACHABLE" in app.query_one(StatsHeader).last_text
        assert "GPU N/A" in app.query_one(StatsHeader).last_text
        assert "N/A — no NVIDIA GPU" in app.query_one(VramGauge).last_text
        assert "usable budget N/A" in app.query_one(ContextGauge).last_text
        assert not app.query_one(MismatchBanner).has_class("visible")


async def test_mismatch_banner_is_persistently_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_profile(monkeypatch, mismatched_profile())
    app = PharosApp(config=CONFIG, bus=EventBus())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        banner = app.query_one(MismatchBanner)
        assert banner.has_class("visible")
        assert "CONTEXT MISMATCH" in banner.last_text
        assert "262,144" in banner.last_text
        assert "32,768" in banner.last_text
        # Achievable (32,768 + 234,000) exceeds the advertised max here, so the banner may
        # honestly point at the full window rather than a hardware ceiling.
        assert "full window" in banner.last_text
        # VRAM gauge is actionable: measured free VRAM plus the estimated token headroom
        # (which holds back the 512 MiB safety margin: (8000-512)/32*1000).
        vram = app.query_one(VramGauge).last_text
        assert "8,000" in vram
        assert "234,000" in vram
        assert "estimate" in vram


def vram_constrained_profile() -> EnvironmentProfile:
    """This desktop for real: ~10 GB of weights resident, the advertised max unreachable."""
    gpu = GpuInfo(
        available=True,
        name="NVIDIA GeForce RTX 3060",
        total_mib=12288,
        used_mib=11000,
        free_mib=1288,
        source="nvml",
    )
    backend = BackendInfo(
        reachable=True,
        base_url="http://localhost:11434",
        model="qwen3.5-9b-heretic:latest",
        advertised_max_ctx=262144,
        loaded_ctx=32768,
    )
    budget = Accountant(CONFIG).report(loaded_ctx=32768, gpu=gpu)
    return EnvironmentProfile(
        gpu=gpu,
        backend=backend,
        budget=budget,
        ctx_mismatch=True,
        ctx_mismatch_ratio=32768 / 262144,
    )


async def test_banner_points_at_hardware_ceiling_when_advertised_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telling the user to raise num_ctx to a window VRAM cannot hold is worse than silence."""
    _patch_profile(monkeypatch, vram_constrained_profile())
    app = PharosApp(config=CONFIG, bus=EventBus())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        banner = app.query_one(MismatchBanner).last_text
        # achievable = 32,768 + (1288-512)/32*1000 = 57,018 — well short of 262,144.
        assert "57,018" in banner
        assert "ceiling" in banner
        assert "full window" not in banner


async def test_rejected_request_never_wears_the_green_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4xx is a completed rejection: in a tool about making failures visible, it gets its
    own glyph, not the success one."""
    _patch_profile(monkeypatch, mismatched_profile())
    bus = EventBus()
    app = PharosApp(config=CONFIG, bus=bus)
    async with app.run_test() as pilot:
        await pilot.pause()
        bus.publish(
            RequestCompleted(
                request_id=3,
                status_code=400,
                prompt_eval_count=None,
                eval_count=None,
                eval_duration_ns=None,
                tokens_per_second=None,
                duration_s=0.1,
            )
        )
        await pilot.pause()
        log_text = "\n".join(app.query_one(EventLog).plain)
        assert "✕ 400" in log_text
        assert "✓" not in log_text
        assert "✕ 400" in app.query_one(ThroughputLine).last_text


async def test_estimate_then_reconciled_exact_in_gauge_and_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_profile(monkeypatch, mismatched_profile())
    bus = EventBus()
    app = PharosApp(config=CONFIG, bus=bus)
    async with app.run_test() as pilot:
        await pilot.pause()
        bus.publish(
            RequestStarted(
                request_id=1,
                method="POST",
                path="/api/chat",
                endpoint="ollama-chat",
                model="qwen3.5:9b",
                stream=True,
            )
        )
        bus.publish(InputCounted(request_id=1, tokens=1500, exact=False, source="gguf"))
        await pilot.pause()
        gauge = app.query_one(ContextGauge).last_text
        assert "~1,500" in gauge  # the estimate marker must be in the GAUGE, not only the log
        assert "estimate" in gauge

        bus.publish(
            RequestCompleted(
                request_id=1,
                status_code=200,
                prompt_eval_count=1720,
                eval_count=80,
                eval_duration_ns=2_000_000_000,
                tokens_per_second=40.0,
                duration_s=2.5,
            )
        )
        await pilot.pause()
        gauge = app.query_one(ContextGauge).last_text
        assert "1,800" in gauge  # reconciled: 1720 in + 80 out
        assert "exact" in gauge
        assert "~" not in gauge.split("/")[0]  # no estimate marker once reconciled
        assert "40.0 tok/s" in app.query_one(ThroughputLine).last_text
        log_text = "\n".join(app.query_one(EventLog).plain)
        assert "~1,500 (estimate · gguf)" in log_text
        assert "1,720 (exact)" in log_text


async def test_heuristic_counts_are_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_profile(monkeypatch, degraded_profile())
    bus = EventBus()
    app = PharosApp(config=CONFIG, bus=bus)
    async with app.run_test() as pilot:
        await pilot.pause()
        bus.publish(InputCounted(request_id=1, tokens=42, exact=False, source="heuristic"))
        await pilot.pause()
        gauge = app.query_one(ContextGauge).last_text
        assert "~42" in gauge
        assert "heuristic" in gauge


async def test_aborted_request_is_visually_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_profile(monkeypatch, mismatched_profile())
    bus = EventBus()
    app = PharosApp(config=CONFIG, bus=bus)
    async with app.run_test() as pilot:
        await pilot.pause()
        bus.publish(RequestAborted(request_id=7, status_code=200, duration_s=0.8))
        await pilot.pause()
        log_text = "\n".join(app.query_one(EventLog).plain)
        assert "ABORTED" in log_text
        assert "✓" not in log_text  # never rendered as a normal completion
        assert "ABORTED" in app.query_one(ThroughputLine).last_text


async def test_dropped_events_are_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_profile(monkeypatch, degraded_profile())
    bus = EventBus(queue_size=2)
    app = PharosApp(config=CONFIG, bus=bus)  # subscribes at construction
    # Overflow the TUI's queue before its consumer ever runs: 3 of 5 events are dropped.
    for i in range(5):
        bus.publish(InputCounted(request_id=i, tokens=i, exact=False, source="heuristic"))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        log_text = "\n".join(app.query_one(EventLog).plain)
        assert "dropped" in log_text  # the gap is loud, not silent
        assert "3" in app.query_one(ThroughputLine).last_text.split("dropped")[0][-10:]
