"""What the VRAM store will and will not fit a KV rate from.

The interesting tests here are the refusals. A slope through the wrong points is not a smaller
answer than no slope at all, it is a confident wrong one — and it drives the headroom advice,
which is the figure this project has been most badly wrong about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharos.accountant import Accountant
from pharos.config import PharosConfig
from pharos.profiler.types import GpuInfo
from pharos.vram import (
    FIT_FLOOR_CTX,
    MIN_FIT_POINTS,
    measured_rate,
    record_vram_sample,
)

_BYTES_PER_MIB = 1024 * 1024
# 32 MiB per 1K tokens, the rate qwen3.5-9b actually measured on an RTX 3060.
_RATE_BYTES_PER_TOKEN = 32 * _BYTES_PER_MIB / 1000
_WEIGHTS = 6_000_000_000


def _vram_at(ctx: int) -> int:
    """What a fully resident model would occupy at `ctx`, on a perfectly linear card."""
    return int(_WEIGHTS + _RATE_BYTES_PER_TOKEN * ctx)


def _record(store: Path, *ctxs: int, model: str = "m", digest: str = "d") -> None:
    for ctx in ctxs:
        size = _vram_at(ctx)
        record_vram_sample(
            store,
            model,
            digest=digest,
            loaded_ctx=ctx,
            size_vram_bytes=size,
            weight_size_bytes=size,
        )


def test_three_windows_give_the_rate_the_card_actually_charges(tmp_path: Path) -> None:
    store = tmp_path / "vram.json"
    _record(store, 8192, 16384, 32768)

    fitted = measured_rate(store, "m", digest="d")

    assert fitted is not None
    assert fitted.mib_per_1k == pytest.approx(32.0, rel=0.01)
    assert fitted.points == 3
    assert fitted.span == "8,192-32,768"


def test_two_windows_are_not_enough(tmp_path: Path) -> None:
    """One reading is an anecdote and two cannot outvote a bad one."""
    store = tmp_path / "vram.json"
    _record(store, 8192, 32768)
    assert measured_rate(store, "m", digest="d") is None
    assert MIN_FIT_POINTS == 3


def test_windows_below_the_floor_are_kept_but_never_fitted(tmp_path: Path) -> None:
    """Below about 8K the VRAM/context line bends: a fixed allocation is still changing size,
    and a slope fitted through that region is not the KV rate. Measured by hand before this
    module existed — see the validation table in `_extract_kv_bytes_per_token`."""
    store = tmp_path / "vram.json"
    _record(store, 512, 1024, 2048, 4096)

    assert measured_rate(store, "m", digest="d") is None
    # The readings are still on disk; they were refused for the fit, not thrown away.
    saved = json.loads(store.read_text(encoding="utf-8"))
    assert len(saved["models"]["m"]["windows"]) == 4
    assert all(int(ctx) < FIT_FLOOR_CTX for ctx in saved["models"]["m"]["windows"])


def test_three_windows_too_close_together_are_refused(tmp_path: Path) -> None:
    """Noise in size_vram dominates a slope fitted across a narrow span."""
    store = tmp_path / "vram.json"
    _record(store, 8192, 9216, 10240)
    assert measured_rate(store, "m", digest="d") is None


def test_a_partly_offloaded_model_is_never_recorded(tmp_path: Path) -> None:
    """`size_vram != size` is Ollama saying part of the model is in RAM. Its VRAM does not move
    with context in any way worth fitting, and one such point poisons the line."""
    store = tmp_path / "vram.json"
    for ctx in (8192, 16384, 32768):
        record_vram_sample(
            store,
            "m",
            digest="d",
            loaded_ctx=ctx,
            size_vram_bytes=_vram_at(ctx) // 2,  # half on the card
            weight_size_bytes=_vram_at(ctx),
        )
    assert measured_rate(store, "m", digest="d") is None
    assert not store.exists() or json.loads(store.read_text(encoding="utf-8"))["models"] == {}


def test_an_unknown_residency_is_not_good_enough(tmp_path: Path) -> None:
    """A sample that MIGHT be half in RAM would sit in the store indistinguishable from a
    clean one, so it is not recorded at all."""
    store = tmp_path / "vram.json"
    record_vram_sample(
        store, "m", digest="d", loaded_ctx=16384,
        size_vram_bytes=_vram_at(16384), weight_size_bytes=None,
    )
    assert measured_rate(store, "m", digest="d") is None


def test_a_new_digest_discards_what_was_learned_about_the_old_file(tmp_path: Path) -> None:
    """A model re-pulled under the same tag can be a different file with a different cache."""
    store = tmp_path / "vram.json"
    _record(store, 8192, 16384, 32768, digest="old")
    assert measured_rate(store, "m", digest="old") is not None

    _record(store, 65536, digest="new")
    assert measured_rate(store, "m", digest="new") is None, "the old windows must not carry over"


def test_asking_for_a_digest_that_does_not_match_measures_nothing(tmp_path: Path) -> None:
    store = tmp_path / "vram.json"
    _record(store, 8192, 16384, 32768, digest="d")
    assert measured_rate(store, "m", digest="other") is None


def test_an_implausible_slope_is_discarded(tmp_path: Path) -> None:
    """Something other than the KV cache moved between these readings."""
    store = tmp_path / "vram.json"
    for ctx, size in ((8192, 6_000_000_000), (16384, 6_000_000_100), (32768, 6_000_000_200)):
        record_vram_sample(
            store, "m", digest="d", loaded_ctx=ctx, size_vram_bytes=size, weight_size_bytes=size
        )
    # ~0.01 MiB/1K: far below anything a real cache costs.
    assert measured_rate(store, "m", digest="d") is None


def test_a_shrinking_line_is_discarded(tmp_path: Path) -> None:
    """VRAM that falls as the window grows is not describing a cache."""
    store = tmp_path / "vram.json"
    for ctx, size in ((8192, 7_000_000_000), (16384, 6_500_000_000), (32768, 6_000_000_000)):
        record_vram_sample(
            store, "m", digest="d", loaded_ctx=ctx, size_vram_bytes=size, weight_size_bytes=size
        )
    assert measured_rate(store, "m", digest="d") is None


def test_a_repeated_window_updates_rather_than_duplicates(tmp_path: Path) -> None:
    """The dashboard reprobes every few seconds; the same window must not become 200 points."""
    store = tmp_path / "vram.json"
    for _ in range(5):
        _record(store, 16384)
    saved = json.loads(store.read_text(encoding="utf-8"))
    assert list(saved["models"]["m"]["windows"]) == ["16384"]


def test_a_missing_store_measures_nothing_and_does_not_raise(tmp_path: Path) -> None:
    assert measured_rate(tmp_path / "absent.json", "m") is None


def test_a_corrupt_store_is_a_log_line_and_nothing_more(tmp_path: Path) -> None:
    """Same rule as every other store here: it makes the report better when it is there and
    must never be able to stop one happening."""
    store = tmp_path / "vram.json"
    store.write_text("{not json at all", encoding="utf-8")

    assert measured_rate(store, "m") is None
    record_vram_sample(
        store, "m", digest="d", loaded_ctx=16384,
        size_vram_bytes=_vram_at(16384), weight_size_bytes=_vram_at(16384),
    )


def test_nothing_is_recorded_without_a_model_or_a_window(tmp_path: Path) -> None:
    store = tmp_path / "vram.json"
    record_vram_sample(
        store, None, digest="d", loaded_ctx=16384, size_vram_bytes=1, weight_size_bytes=1
    )
    record_vram_sample(
        store, "m", digest="d", loaded_ctx=None, size_vram_bytes=1, weight_size_bytes=1
    )
    assert not store.exists()


# --- how the accountant ranks the three answers -------------------------------------------------


def test_a_measurement_outranks_the_models_own_metadata() -> None:
    """Not the obvious order, and the one the numbers argue for: derivation counts the cache
    and nothing else, and read 1-4% UNDER what the card actually gave up on all three
    architectures it was validated against. Under-counting overstates headroom."""
    report = Accountant(PharosConfig(kv_mib_per_1k=26)).report(
        loaded_ctx=8192,
        gpu=GpuInfo(available=False),
        kv_bytes_per_token=147_456,  # 140.6 MiB/1K derived
        measured_mib_per_1k=142.6,
    )
    assert report.kv_rate_source == "measured"
    assert report.kv_mib_per_1k == pytest.approx(142.6)
    # The loser is kept, because the two agreeing is worth seeing and disagreeing more so.
    assert report.kv_derived_mib_per_1k == pytest.approx(147_456 * 1000 / _BYTES_PER_MIB)


def test_derivation_still_beats_the_constant_when_nothing_is_measured() -> None:
    """Every model on a fresh install, and every model whose windows have not moved."""
    report = Accountant(PharosConfig(kv_mib_per_1k=26)).report(
        loaded_ctx=8192,
        gpu=GpuInfo(available=False),
        kv_bytes_per_token=147_456,
        measured_mib_per_1k=None,
    )
    assert report.kv_rate_source == "derived"


def test_a_measurement_rescues_an_architecture_derivation_refuses() -> None:
    """The case this exists for: a hybrid SSM stack publishes too little to derive from, so the
    figure fell back to one constant for every model -- measured 4-7x wrong."""
    report = Accountant(PharosConfig(kv_mib_per_1k=33)).report(
        loaded_ctx=32768,
        gpu=GpuInfo(available=True, total_mib=12288, used_mib=8000, free_mib=4288),
        kv_bytes_per_token=None,  # refused: hybrid SSM
        measured_mib_per_1k=28.6,
    )
    assert report.kv_rate_source == "measured"
    assert report.kv_mib_per_1k == pytest.approx(28.6)
    assert report.kv_derived_mib_per_1k is None
    # And the headroom it drives moves with it, rather than with the constant.
    assert report.vram_headroom_tokens == int((4288 - 512) / 28.6 * 1000)


def test_the_fit_survives_a_curve_that_bends(tmp_path: Path) -> None:
    """Measured on a live RTX 3060: `qwen3.5:9b` is exactly linear inside a regime and then
    changes slope -- 33.20 MiB/1K from 8K to 32K, 28.56 from 32K to 64K. A straight line
    through all four points describes neither piece exactly, and is kept anyway because the
    bend is downward, so the fitted rate over-charges rather than over-promises.
    """
    store = tmp_path / "vram.json"
    windows = {8192: 5_880_141_577, 16384: 6_165_354_249, 32768: 6_735_779_593,
               65536: 7_717_236_243}
    for ctx, size in windows.items():
        record_vram_sample(
            store, "qwen3.5:9b", digest="d",
            loaded_ctx=ctx, size_vram_bytes=size, weight_size_bytes=size,
        )

    fitted = measured_rate(store, "qwen3.5:9b", digest="d")
    assert fitted is not None
    marginal = (windows[65536] - windows[32768]) / (65536 - 32768) * 1000 / _BYTES_PER_MIB
    assert marginal == pytest.approx(28.56, abs=0.05)
    # Over-charging is the safe direction: headroom is spent faster than it truly costs.
    assert fitted.mib_per_1k > marginal
    assert fitted.points == 4


def test_the_fit_reproduces_the_hand_measurement_it_replaces(tmp_path: Path) -> None:
    """The readings below are what `/api/ps` actually returned for `qwen3.5-9b-heretic` on an
    RTX 3060, taken by three ordinary profiles at 8K, 16K and 32K.

    The validation table in `_extract_kv_bytes_per_token` records 32.23 MiB/1K for this
    architecture, arrived at by hand with curl before this module existed. Fitting the same
    windows automatically gives the same number to two decimal places — which is the claim this
    whole module rests on: that the measurement was always available for free.
    """
    store = tmp_path / "vram.json"
    for ctx, size in ((8192, 8_857_657_015), (16384, 9_134_481_079), (32768, 9_688_129_207)):
        record_vram_sample(
            store, "qwen3.5-9b-heretic:latest", digest="d",
            loaded_ctx=ctx, size_vram_bytes=size, weight_size_bytes=size,
        )

    fitted = measured_rate(store, "qwen3.5-9b-heretic:latest", digest="d")
    assert fitted is not None
    assert fitted.mib_per_1k == pytest.approx(32.23, abs=0.01)
    # And it lands ABOVE what the metadata implies (31.25), which is the direction that keeps
    # the headroom estimate from over-promising.
    assert fitted.mib_per_1k > 31.25
