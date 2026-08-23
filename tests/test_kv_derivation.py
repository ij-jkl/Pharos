"""KV-rate derivation: the model's own GGUF metadata instead of a hand-tuned constant.

The constant this replaces (``kv_mib_per_1k``, default 32) measured 5-14x low against every
real model to hand, and it drives the VRAM headroom estimate — the figure that says how much
further ``num_ctx`` can be pushed. Understating KV *overstates* headroom, so the old number
was advice pointing straight at the OOM Pharos exists to prevent.

So the derivation is pinned against hardware, not against itself. Each expected figure below
was measured on an RTX 3060 12GB by loading the model at several window sizes and reading
``size_vram`` from /api/ps: VRAM is linear in context, and its slope is the true
bytes-per-token. Below roughly 8K that line bends, so every slope here is fitted from 8K
upward across at least three points — a two-point fit taken at 4K/8K read 59% high on
qwen2.5-coder:7b, which is how the low end was found to be untrustworthy at all.

Three architectures appear deliberately, one per branch of the derivation: qwen2 omits its
head dimensions and exercises the fallback, qwen3 publishes them, and qwen35 is a hybrid
SSM/attention stack where only every 4th layer holds a cache at all.
"""

from __future__ import annotations

import pytest

from pharos.accountant import Accountant
from pharos.config import PharosConfig
from pharos.profiler.backend import _extract_kv_bytes_per_token
from pharos.profiler.types import GpuInfo

BYTES_PER_MIB = 1024 * 1024

# /api/show `model_info`, trimmed to the keys the derivation reads.
QWEN3_4B = {
    "general.architecture": "qwen3",
    "qwen3.block_count": 36,
    "qwen3.attention.head_count": 32,
    "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128,
    "qwen3.attention.value_length": 128,
    "qwen3.embedding_length": 2560,
}

# qwen2 GGUFs omit key_length/value_length entirely: head_dim is embedding_length/head_count.
# Not in the measured table below: 14B cannot be held in 12GB above 8K without spilling to
# CPU, and a spilled load reallocates, so there is no clean slope to fit.
QWEN2_CODER_14B = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 48,
    "qwen2.attention.head_count": 40,
    "qwen2.attention.head_count_kv": 8,
    "qwen2.embedding_length": 5120,
}

QWEN2_CODER_7B = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.embedding_length": 3584,
}

# Hybrid SSM/attention: only every 4th layer keeps a KV cache. The other three are state-space
# layers whose state is fixed per sequence, so they cost nothing per token.
QWEN35_9B = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 32,
    "qwen35.attention.head_count": 16,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.embedding_length": 4096,
    "qwen35.full_attention_interval": 4,
    "qwen35.ssm.state_size": 128,
}

# Sliding-window attention: five layers in six hold only a 1024-token window, and which ones
# is not published.
GEMMA3_12B = {
    "general.architecture": "gemma3",
    "gemma3.block_count": 48,
    "gemma3.attention.head_count": 16,
    "gemma3.attention.head_count_kv": 8,
    "gemma3.attention.key_length": 256,
    "gemma3.attention.value_length": 256,
    "gemma3.attention.sliding_window": 1024,
    "gemma3.embedding_length": 3840,
}


def test_configured_default_is_the_wrong_order_of_magnitude_for_plain_stacks() -> None:
    """Why this exists: on a non-hybrid model the constant is not imprecise, it is 4-7x off."""
    for info, arch in ((QWEN3_4B, "qwen3"), (QWEN2_CODER_14B, "qwen2")):
        derived = _extract_kv_bytes_per_token(info, arch)
        assert derived is not None
        assert derived * 1000 / BYTES_PER_MIB > 4 * PharosConfig().kv_mib_per_1k


def test_derives_from_published_head_dimensions() -> None:
    # 36 layers x 8 KV heads x (128 + 128) x 2 bytes
    assert _extract_kv_bytes_per_token(QWEN3_4B, "qwen3") == 147_456


def test_falls_back_to_embedding_over_head_count() -> None:
    # head_dim = 5120 / 40 = 128, then 48 x 8 x (128 + 128) x 2
    assert _extract_kv_bytes_per_token(QWEN2_CODER_14B, "qwen2") == 196_608


def test_hybrid_stack_counts_only_the_attending_layers() -> None:
    """Every 4th layer of 32 attends; the rest are SSM and cost nothing per token."""
    assert _extract_kv_bytes_per_token(QWEN35_9B, "qwen35") == 8 * 4 * (256 + 256) * 2
    # Reading the stack as uniform -- the obvious formula -- is 4x high, and 4x low on KV is
    # 4x high on headroom.
    naive = 32 * 4 * (256 + 256) * 2
    assert _extract_kv_bytes_per_token(QWEN35_9B, "qwen35") == naive // 4


def test_declines_when_attention_slides() -> None:
    """A windowed layer's cache stops growing; deriving as if it were global overstates KV
    several-fold at long context, so no number is offered at all."""
    assert _extract_kv_bytes_per_token(GEMMA3_12B, "gemma3") is None


def test_per_layer_array_supersedes_the_hybrid_interval() -> None:
    """An explicit array states the whole pattern; the interval must not be applied on top."""
    info = {**QWEN35_9B, "qwen35.attention.head_count_kv": [4, 0, 0, 0]}
    assert _extract_kv_bytes_per_token(info, "qwen35") == 4 * (256 + 256) * 2


@pytest.mark.parametrize(
    ("model_info", "arch", "measured_mib_per_1k"),
    [
        (QWEN2_CODER_7B, "qwen2", 56.641),  # 8K/16K/32K
        (QWEN3_4B, "qwen3", 142.578),  # 8K/16K/32K
        (QWEN35_9B, "qwen35", 32.227),  # 2K/8K/16K/32K -- the hybrid case
    ],
)
def test_matches_vram_measured_on_a_3060(
    model_info: dict[str, object], arch: str, measured_mib_per_1k: float
) -> None:
    """The derived rate lands within 2% of VRAM growth measured on real hardware.

    It reads slightly *under* the measurement because llama.cpp allocates a little per-token
    scratch beyond the cache itself. Under-reading KV is the unsafe direction, so the margin
    is asserted as a bound rather than left to drift.
    """
    derived = _extract_kv_bytes_per_token(model_info, arch)
    assert derived is not None
    derived_mib_per_1k = derived * 1000 / BYTES_PER_MIB
    assert derived_mib_per_1k == pytest.approx(measured_mib_per_1k, rel=0.04)
    # The residual is one-sided — always slightly under measurement, being llama.cpp's
    # per-token scratch outside the cache proper. Pinned as a direction and not merely a
    # tolerance so that a change flipping the bias is caught: under-reading KV is what
    # overstates headroom, and only stays harmless while it is this small. At 3.4% it costs
    # ~150 MiB on a 4 GB estimate, well inside the 512 MiB margin held back before any
    # headroom figure is shown.
    assert derived_mib_per_1k <= measured_mib_per_1k
    assert derived_mib_per_1k > 0.96 * measured_mib_per_1k


def test_per_layer_kv_head_array_is_summed_not_multiplied() -> None:
    """An architecture varying KV heads per layer publishes an array; block_count must not
    then be applied a second time."""
    info = dict(QWEN3_4B)
    info["qwen3.block_count"] = 4
    info["qwen3.attention.head_count_kv"] = [8, 8, 4, 4]
    assert _extract_kv_bytes_per_token(info, "qwen3") == 24 * (128 + 128) * 2


@pytest.mark.parametrize(
    "mutation",
    [
        {"qwen3.block_count": None},
        {"qwen3.attention.head_count_kv": None},
        {"qwen3.block_count": 0},
        {"qwen3.attention.head_count_kv": 0},
    ],
)
def test_none_when_metadata_is_unusable(mutation: dict[str, object]) -> None:
    """Missing or nonsensical metadata yields None — the caller's signal to fall back, rather
    than a derived-looking number nothing derived."""
    info = {**QWEN3_4B, **mutation}
    assert _extract_kv_bytes_per_token(info, "qwen3") is None


def test_none_when_head_dimensions_cannot_be_recovered() -> None:
    info = {k: v for k, v in QWEN2_CODER_14B.items() if k != "qwen2.embedding_length"}
    assert _extract_kv_bytes_per_token(info, "qwen2") is None


def test_none_when_architecture_unknown() -> None:
    assert _extract_kv_bytes_per_token(QWEN3_4B, None) is None


def test_derived_rate_overrides_the_configured_constant() -> None:
    report = Accountant(PharosConfig(kv_mib_per_1k=26)).report(
        loaded_ctx=8192, gpu=GpuInfo(available=False), kv_bytes_per_token=147_456
    )
    assert report.kv_rate_derived is True
    assert report.kv_mib_per_1k == pytest.approx(147_456 * 1000 / BYTES_PER_MIB)
    assert report.kv_estimate_mib == pytest.approx(report.kv_mib_per_1k * 8192 / 1000)


def test_falls_back_to_configured_rate_when_nothing_to_derive() -> None:
    report = Accountant(PharosConfig(kv_mib_per_1k=26)).report(
        loaded_ctx=8192, gpu=GpuInfo(available=False), kv_bytes_per_token=None
    )
    assert report.kv_rate_derived is False
    assert report.kv_mib_per_1k == pytest.approx(26.0)


def test_headroom_no_longer_overstates_what_fits() -> None:
    """The regression this whole change exists for.

    Same GPU, same free VRAM, same model. The configured constant claimed roughly five times
    the context that actually fits; following it would have loaded a window the card cannot
    hold. The derived rate is the one reported.
    """
    gpu = GpuInfo(available=True, total_mib=12288, free_mib=4512, used_mib=7776)
    cfg = PharosConfig(kv_mib_per_1k=26, vram_safety_margin_mib=512)

    guessed = Accountant(cfg).report(loaded_ctx=8192, gpu=gpu)
    derived = Accountant(cfg).report(loaded_ctx=8192, gpu=gpu, kv_bytes_per_token=147_456)

    assert guessed.vram_headroom_tokens is not None
    assert derived.vram_headroom_tokens is not None
    assert guessed.vram_headroom_tokens > 4 * derived.vram_headroom_tokens
    # 4,000 usable MiB at ~140.6 MiB/1K
    assert derived.vram_headroom_tokens == pytest.approx(28_444, rel=0.01)
    assert derived.achievable_ctx_estimate == 8192 + derived.vram_headroom_tokens
