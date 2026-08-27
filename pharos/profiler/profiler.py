"""Compose the GPU and backend probes into an EnvironmentProfile and flag the context mismatch.

The advertised-vs-loaded context mismatch is the single most valuable thing Pharos surfaces, so
it is flagged loudly wherever Pharos runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from pharos.accountant import Accountant
from pharos.config import PharosConfig
from pharos.profiler.backend import probe_backend
from pharos.profiler.gpu import probe_gpu
from pharos.profiler.types import BackendInfo, EnvironmentProfile
from pharos.vram import measured_rate, record_vram_sample


def detect_ctx_mismatch(backend: BackendInfo) -> tuple[bool, float | None]:
    """Return (is_mismatch, loaded/advertised ratio) when both context sizes are known."""
    advertised = backend.advertised_max_ctx
    loaded = backend.loaded_ctx
    if advertised is None or loaded is None or advertised <= 0:
        return False, None
    return loaded < advertised, loaded / advertised


async def build_profile(config: PharosConfig) -> EnvironmentProfile:
    """Probe the GPU and backend and assemble the full environment profile.

    Every probe also files what the loaded model occupied at its current window. Nothing about
    the profile depends on that, and one reading answers nothing on its own -- but a profile
    runs on every `pharos check`, every `pharos run` and every few seconds of the dashboard,
    so the windows a user actually loads accumulate, and `pharos.vram` fits the real cost of a
    context token out of them. It is the same measurement the validation table was built from
    by hand, taken for free.
    """
    gpu = await asyncio.to_thread(probe_gpu)
    backend = await probe_backend(config)
    mismatch, ratio = detect_ctx_mismatch(backend)

    store = Path(config.vram_memory_file)
    await asyncio.to_thread(
        record_vram_sample,
        store,
        backend.model,
        digest=backend.digest,
        loaded_ctx=backend.loaded_ctx,
        size_vram_bytes=backend.size_vram_bytes,
        weight_size_bytes=backend.weight_size_bytes,
    )
    fitted = await asyncio.to_thread(
        measured_rate, store, backend.model, digest=backend.digest
    )

    budget = Accountant(config).report(
        loaded_ctx=backend.loaded_ctx,
        gpu=gpu,
        kv_bytes_per_token=backend.kv_bytes_per_token,
        measured_mib_per_1k=fitted.mib_per_1k if fitted else None,
    )
    if fitted is not None:
        budget = replace(
            budget,
            kv_measured_detail=f"{fitted.points} windows, {fitted.span}",
        )
    return EnvironmentProfile(
        gpu=gpu,
        backend=backend,
        budget=budget,
        ctx_mismatch=mismatch,
        ctx_mismatch_ratio=ratio,
    )
