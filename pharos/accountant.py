"""The single source of truth for budget math.

Computes usable_budget (= loaded_ctx - response_reserve) and the VRAM headroom / KV-cache
estimate. No other component does budget arithmetic. Backend-agnostic: the math never cares
whether Ollama or llama.cpp sits behind the proxy.
"""

from __future__ import annotations

from pharos.config import PharosConfig
from pharos.profiler.types import BudgetReport, GpuInfo

_BYTES_PER_MIB = 1024 * 1024


class Accountant:
    """Turns a loaded context size + GPU snapshot into a BudgetReport."""

    def __init__(self, config: PharosConfig) -> None:
        self._reserve = config.response_reserve
        self._warn = config.warn_threshold
        self._alert = config.alert_threshold
        self._kv_mib_per_1k = config.kv_mib_per_1k
        self._vram_margin_mib = config.vram_safety_margin_mib

    def report(
        self,
        *,
        loaded_ctx: int | None,
        gpu: GpuInfo,
        kv_bytes_per_token: int | None = None,
    ) -> BudgetReport:
        """Compute the budget for a given loaded context window and GPU snapshot.

        ``kv_bytes_per_token`` comes from the model's own GGUF metadata and, when present,
        replaces the configured ``kv_mib_per_1k`` for every figure below. The constant was a
        hand-tuned guess that measured 5-14x low on real models, and it drives the headroom
        estimate — so a low KV rate overstates how much context still fits, which is the one
        direction this number must never be wrong in. The config value stays as the fallback
        for backends that publish nothing to derive from.
        """
        kv_rate, kv_derived = self._kv_rate(kv_bytes_per_token)
        usable: int | None = None
        warn_tokens: int | None = None
        alert_tokens: int | None = None
        kv_estimate: float | None = None
        if loaded_ctx is not None:
            usable = max(loaded_ctx - self._reserve, 0)
            warn_tokens = int(usable * self._warn)
            alert_tokens = int(usable * self._alert)
            kv_estimate = kv_rate * loaded_ctx / 1000
        headroom_tokens: int | None = None
        achievable_ctx: int | None = None
        free_mib = gpu.free_mib if gpu.available else None
        if free_mib is not None and loaded_ctx is not None:
            # ESTIMATE (the same KV rate the figure above uses): tokens that still fit before
            # the KV cache would exhaust measured free VRAM. Never presented as measured.
            #
            # Requires a resident model. With nothing loaded, free VRAM still has to absorb
            # the weights before a single context token exists, so the same arithmetic
            # overstates wildly — measured at ~20x on a 12GB card with a 9B Q8_0 (335,718
            # tokens claimed, 16,625 actually reachable once resident). None is the honest
            # answer; the renderers say why.
            #
            # The safety margin is held back first: driver allocations and fragmentation claim
            # VRAM without warning, so a headroom figure that spends the last free byte is an
            # OOM invitation. Advice derived from this number must survive being followed.
            usable_free_mib = max(free_mib - self._vram_margin_mib, 0)
            headroom_tokens = int(usable_free_mib / kv_rate * 1000)
            # How far num_ctx can ACTUALLY be raised on this hardware, as opposed to the
            # advertised maximum the mismatch banner would otherwise point at.
            achievable_ctx = loaded_ctx + headroom_tokens
        return BudgetReport(
            loaded_ctx=loaded_ctx,
            response_reserve=self._reserve,
            usable_budget=usable,
            warn_threshold=self._warn,
            alert_threshold=self._alert,
            warn_tokens=warn_tokens,
            alert_tokens=alert_tokens,
            kv_estimate_mib=kv_estimate,
            vram_free_mib=gpu.free_mib if gpu.available else None,
            vram_total_mib=gpu.total_mib if gpu.available else None,
            vram_headroom_tokens=headroom_tokens,
            vram_safety_margin_mib=self._vram_margin_mib,
            achievable_ctx_estimate=achievable_ctx,
            kv_mib_per_1k=kv_rate,
            kv_rate_derived=kv_derived,
        )

    def _kv_rate(self, kv_bytes_per_token: int | None) -> tuple[float, bool]:
        """The KV rate to compute with, and whether it was derived or merely configured."""
        if kv_bytes_per_token is not None and kv_bytes_per_token > 0:
            return kv_bytes_per_token * 1000 / _BYTES_PER_MIB, True
        return self._kv_mib_per_1k, False
