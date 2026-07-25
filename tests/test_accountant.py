"""Accountant budget-math tests (usable_budget, thresholds, KV estimate). Checkpoint 2."""

from __future__ import annotations

import pytest

from pharos.accountant import Accountant
from pharos.config import PharosConfig
from pharos.profiler.types import GpuInfo


def test_usable_budget_and_thresholds() -> None:
    cfg = PharosConfig(response_reserve=1024, warn_threshold=0.8, alert_threshold=0.9)
    report = Accountant(cfg).report(loaded_ctx=32768, gpu=GpuInfo(available=False))
    assert report.usable_budget == 31744
    assert report.warn_tokens == int(31744 * 0.8)
    assert report.alert_tokens == int(31744 * 0.9)


def test_usable_budget_clamped_non_negative() -> None:
    cfg = PharosConfig(response_reserve=5000)
    report = Accountant(cfg).report(loaded_ctx=1000, gpu=GpuInfo(available=False))
    assert report.usable_budget == 0


def test_none_loaded_ctx_yields_none() -> None:
    report = Accountant(PharosConfig()).report(loaded_ctx=None, gpu=GpuInfo(available=False))
    assert report.usable_budget is None
    assert report.warn_tokens is None
    assert report.alert_tokens is None
    assert report.kv_estimate_mib is None


def test_kv_estimate() -> None:
    cfg = PharosConfig(kv_mib_per_1k=32)
    report = Accountant(cfg).report(loaded_ctx=32768, gpu=GpuInfo(available=False))
    assert report.kv_estimate_mib == pytest.approx(32 * 32768 / 1000)


def test_vram_fields_present_only_when_gpu_available() -> None:
    gpu = GpuInfo(available=True, total_mib=12288, used_mib=4288, free_mib=8000)
    on = Accountant(PharosConfig()).report(loaded_ctx=4096, gpu=gpu)
    assert on.vram_free_mib == 8000
    assert on.vram_total_mib == 12288

    off = Accountant(PharosConfig()).report(loaded_ctx=4096, gpu=GpuInfo(available=False))
    assert off.vram_free_mib is None
    assert off.vram_total_mib is None
    assert off.vram_headroom_tokens is None


def test_vram_headroom_holds_back_the_safety_margin() -> None:
    gpu = GpuInfo(available=True, total_mib=12288, used_mib=4288, free_mib=8000)
    report = Accountant(PharosConfig(kv_mib_per_1k=32)).report(loaded_ctx=4096, gpu=gpu)
    # (8000 - 512 margin) MiB / 32 MiB-per-1K * 1000 — headroom must never spend the last
    # free byte: advice derived from it has to survive being followed.
    assert report.vram_headroom_tokens == 234_000
    assert report.vram_safety_margin_mib == 512


def test_vram_headroom_with_margin_disabled() -> None:
    gpu = GpuInfo(available=True, total_mib=12288, used_mib=4288, free_mib=8000)
    cfg = PharosConfig(kv_mib_per_1k=32, vram_safety_margin_mib=0)
    report = Accountant(cfg).report(loaded_ctx=4096, gpu=gpu)
    assert report.vram_headroom_tokens == 250_000


def test_vram_headroom_clamps_when_margin_exceeds_free() -> None:
    """Nearly full card: the honest headroom is zero, not negative."""
    gpu = GpuInfo(available=True, total_mib=12288, used_mib=12000, free_mib=288)
    report = Accountant(PharosConfig(kv_mib_per_1k=32)).report(loaded_ctx=4096, gpu=gpu)
    assert report.vram_headroom_tokens == 0
    assert report.achievable_ctx_estimate == 4096  # you are already at the ceiling


def test_achievable_ctx_is_loaded_plus_headroom() -> None:
    gpu = GpuInfo(available=True, total_mib=12288, used_mib=4288, free_mib=8000)
    report = Accountant(PharosConfig(kv_mib_per_1k=32)).report(loaded_ctx=4096, gpu=gpu)
    assert report.achievable_ctx_estimate == 4096 + 234_000


def test_achievable_ctx_is_none_without_gpu_or_model() -> None:
    no_gpu = Accountant(PharosConfig()).report(loaded_ctx=4096, gpu=GpuInfo(available=False))
    assert no_gpu.achievable_ctx_estimate is None
    gpu = GpuInfo(available=True, total_mib=12288, used_mib=1000, free_mib=11000)
    unloaded = Accountant(PharosConfig()).report(loaded_ctx=None, gpu=gpu)
    assert unloaded.achievable_ctx_estimate is None


def test_vram_headroom_is_none_without_a_resident_model() -> None:
    """Free VRAM cannot answer "how much more context fits" until the weights are loaded.

    With nothing resident, that free VRAM still has to absorb the model itself before a single
    context token exists, so the arithmetic overstates wildly — ~20x on the real hardware
    (335,718 tokens claimed vs 16,625 reachable once loaded). None is the honest answer, and
    it sits beside an equally honest "usable budget N/A".
    """
    gpu = GpuInfo(available=True, total_mib=12288, used_mib=288, free_mib=6400)
    report = Accountant(PharosConfig(kv_mib_per_1k=32)).report(loaded_ctx=None, gpu=gpu)
    assert report.usable_budget is None
    assert report.vram_headroom_tokens is None
    # The measured facts are still reported — only the derived projection is withheld.
    assert report.vram_free_mib == 6400
    assert report.vram_total_mib == 12288
