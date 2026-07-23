"""The single source of truth for budget math.

Computes usable_budget (= loaded_ctx - response_reserve) and the VRAM headroom / KV-cache
estimate. No other component does budget arithmetic. Backend-agnostic: the math never cares
whether Ollama or llama.cpp sits behind the proxy.
"""

from __future__ import annotations

from pharos.config import PharosConfig
from pharos.profiler.types import BudgetReport, GpuInfo


class Accountant:
    """Turns a loaded context size + GPU snapshot into a BudgetReport."""

    def __init__(self, config: PharosConfig) -> None:
        self._reserve = config.response_reserve
        self._warn = config.warn_threshold
        self._alert = config.alert_threshold
        self._kv_mib_per_1k = config.kv_mib_per_1k

    def report(self, *, loaded_ctx: int | None, gpu: GpuInfo) -> BudgetReport:
        """Compute the budget for a given loaded context window and GPU snapshot."""
        usable: int | None = None
        warn_tokens: int | None = None
        alert_tokens: int | None = None
        kv_estimate: float | None = None
        if loaded_ctx is not None:
            usable = max(loaded_ctx - self._reserve, 0)
            warn_tokens = int(usable * self._warn)
            alert_tokens = int(usable * self._alert)
            kv_estimate = self._kv_mib_per_1k * loaded_ctx / 1000
        headroom_tokens: int | None = None
        free_mib = gpu.free_mib if gpu.available else None
        if free_mib is not None:
            # ESTIMATE (same kv_mib_per_1k the KV figure uses): tokens that still fit before
            # the KV cache would exhaust measured free VRAM. Never presented as measured.
            headroom_tokens = int(free_mib / self._kv_mib_per_1k * 1000)
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
        )
