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


def test_vram_headroom_tokens_estimate() -> None:
    gpu = GpuInfo(available=True, total_mib=12288, used_mib=4288, free_mib=8000)
    report = Accountant(PharosConfig(kv_mib_per_1k=32)).report(loaded_ctx=4096, gpu=gpu)
    # 8000 MiB free / 32 MiB-per-1K-tokens * 1000 = 250k more ctx tokens (estimate).
    assert report.vram_headroom_tokens == 250_000


def test_vram_headroom_needs_no_loaded_ctx() -> None:
    gpu = GpuInfo(available=True, total_mib=12288, used_mib=288, free_mib=6400)
    report = Accountant(PharosConfig(kv_mib_per_1k=32)).report(loaded_ctx=None, gpu=gpu)
    assert report.usable_budget is None
    assert report.vram_headroom_tokens == 200_000
