"""Compose the GPU and backend probes into an EnvironmentProfile and flag the context mismatch.

The advertised-vs-loaded context mismatch is the single most valuable thing Pharos surfaces, so
it is flagged loudly wherever Pharos runs.
"""

from __future__ import annotations

import asyncio

from pharos.accountant import Accountant
from pharos.config import PharosConfig
from pharos.profiler.backend import probe_backend
from pharos.profiler.gpu import probe_gpu
from pharos.profiler.types import BackendInfo, EnvironmentProfile


def detect_ctx_mismatch(backend: BackendInfo) -> tuple[bool, float | None]:
    """Return (is_mismatch, loaded/advertised ratio) when both context sizes are known."""
    advertised = backend.advertised_max_ctx
    loaded = backend.loaded_ctx
    if advertised is None or loaded is None or advertised <= 0:
        return False, None
    return loaded < advertised, loaded / advertised


async def build_profile(config: PharosConfig) -> EnvironmentProfile:
    """Probe the GPU and backend and assemble the full environment profile."""
    gpu = await asyncio.to_thread(probe_gpu)
    backend = await probe_backend(config)
    mismatch, ratio = detect_ctx_mismatch(backend)
    budget = Accountant(config).report(
        loaded_ctx=backend.loaded_ctx,
        gpu=gpu,
        kv_bytes_per_token=backend.kv_bytes_per_token,
    )
    return EnvironmentProfile(
        gpu=gpu,
        backend=backend,
        budget=budget,
        ctx_mismatch=mismatch,
        ctx_mismatch_ratio=ratio,
    )
