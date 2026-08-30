"""Typed, immutable results for the profiler.

Pure data holders — no logic, no local imports — so every other profiler/accountant module can
depend on them without import cycles. ``GpuInfo.available`` and ``BackendInfo.reachable`` are the
two graceful-degradation flags: a machine with no NVIDIA GPU and/or no reachable backend still
yields a complete profile instead of crashing.

The model and context facts all come from the same ``/api/show`` + ``/api/ps`` probe, so they
share one flat ``BackendInfo`` record rather than being split across several — there is no state
in which one of them is known and another is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Where the KV rate every VRAM figure is computed from actually came from. Ordered by how
# directly each one answers the question: a fit over this machine's own readings, the
# model's published metadata, or the constant in pharos.toml.
KvRateSource = Literal["measured", "derived", "configured"]


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """A snapshot of the primary GPU, or ``available=False`` when none was detected."""

    available: bool
    name: str | None = None
    total_mib: int | None = None
    used_mib: int | None = None
    free_mib: int | None = None
    source: str | None = None  # "nvml" | "nvidia-smi"
    detail: str | None = None  # reason when unavailable


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """What the backend probe learned about the loaded model and its context window."""

    reachable: bool
    base_url: str
    model: str | None = None
    architecture: str | None = None
    quantization: str | None = None
    parameter_size: str | None = None
    weight_size_bytes: int | None = None
    size_vram_bytes: int | None = None
    advertised_max_ctx: int | None = None  # from GGUF metadata via /api/show
    loaded_ctx: int | None = None  # the ACTUAL loaded window via /api/ps
    # Identifies the FILE behind a tag, so a model re-pulled under the same name does not
    # inherit VRAM readings taken against a different one.
    digest: str | None = None
    # KV-cache bytes per context token, derived from the same /api/show metadata block as
    # advertised_max_ctx. None when the model did not publish enough to compute it.
    kv_bytes_per_token: int | None = None
    detail: str | None = None  # reason when unreachable


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """The accountant's output: usable budget, warning lines and VRAM figures."""

    loaded_ctx: int | None
    response_reserve: int
    usable_budget: int | None
    warn_threshold: float
    alert_threshold: float
    warn_tokens: int | None
    alert_tokens: int | None
    kv_estimate_mib: float | None  # estimated, not measured
    vram_free_mib: int | None  # measured (None when no GPU)
    vram_total_mib: int | None
    # ESTIMATE: additional ctx tokens that fit in free VRAM after the safety margin is held back
    vram_headroom_tokens: int | None
    vram_safety_margin_mib: int = 0  # MiB deliberately excluded from the headroom estimate
    # ESTIMATE: loaded_ctx + headroom — how far num_ctx can actually go on this hardware
    achievable_ctx_estimate: int | None = None
    # The KV rate every figure above was computed with, and where it came from. The three
    # sources differ by up to 7x in practice, so the number is never shown unlabelled.
    kv_mib_per_1k: float | None = None
    kv_rate_source: KvRateSource = "configured"
    # What the model's metadata implied, kept even when a measurement outranked it: the two
    # agreeing is worth seeing, and the two disagreeing is worth seeing more.
    kv_derived_mib_per_1k: float | None = None
    # How a measurement was arrived at, for a report that has to say where a number came
    # from -- e.g. "4 windows, 8,192-65,536".
    kv_measured_detail: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    """The full picture: GPU + backend + budget, plus the headline context-mismatch flag."""

    gpu: GpuInfo
    backend: BackendInfo
    budget: BudgetReport
    ctx_mismatch: bool
    ctx_mismatch_ratio: float | None  # loaded / advertised, when both are known
