"""Ollama backend adapter — the one place that knows Ollama's wire format.

Reads /api/tags, /api/show and /api/ps to detect the loaded model, its quantization, the
advertised max context (GGUF metadata) and the ACTUAL loaded context window. Degrades to
BackendInfo(reachable=False, ...) when the backend cannot be reached rather than raising:
every caller is a report that still has to print.

Field names — CONFIRMED against Ollama 0.31.1 (RTX 3060, qwen3.5-9b-heretic, 2026-07-24):

* /api/ps  -> the loaded window is a TOP-LEVEL ``context_length`` on the model entry
  (e.g. 32768). Its ``details`` object carries no context key at all.
* /api/show -> the advertised max is ``model_info["<arch>.context_length"]``, where
  ``<arch>`` is ``model_info["general.architecture"]`` (e.g. "qwen35" -> 262144).

``context``/``num_ctx`` are kept as unconfirmed fallbacks for other Ollama versions. A
``details.context_length`` fallback was deliberately REMOVED: the sibling /api/tags payload
uses that exact key for the ADVERTISED max, so honouring it here would report
advertised-as-loaded, yield ratio 1.0 and silently suppress the mismatch banner. A fallback
that can be confidently wrong is worse than no fallback.

Model names are matched tag-insensitively (``pharos.naming.normalize_model_name``): Ollama
accepts an untagged name everywhere but reports ``name:tag`` back from /api/ps.
"""

from __future__ import annotations

from typing import Any

import httpx

from pharos.config import PharosConfig
from pharos.naming import normalize_model_name
from pharos.profiler.types import BackendInfo

_DEFAULT_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
# "context_length" is confirmed (Ollama 0.31.1); the rest are version fallbacks. Never add
# "details.context_length" here — in /api/tags that key holds the advertised max, not the
# loaded window, and reporting it as loaded would silently suppress the mismatch banner.
_LOADED_CTX_KEYS = ("context_length", "context", "num_ctx")


class OllamaBackend:
    """Thin async adapter over the Ollama HTTP API."""

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base = base_url.rstrip("/")
        self._client = client

    async def tags(self) -> dict[str, Any]:
        resp = await self._client.get(f"{self._base}/api/tags", timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return _as_dict(resp.json())

    async def ps(self) -> dict[str, Any]:
        resp = await self._client.get(f"{self._base}/api/ps", timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return _as_dict(resp.json())

    async def show(self, model: str) -> dict[str, Any]:
        resp = await self._client.post(
            f"{self._base}/api/show", json={"model": model}, timeout=_DEFAULT_TIMEOUT
        )
        resp.raise_for_status()
        return _as_dict(resp.json())


async def probe_backend(
    config: PharosConfig, client: httpx.AsyncClient | None = None
) -> BackendInfo:
    """Detect the loaded model, its quant, advertised max ctx and actual loaded ctx.

    Never raises for an unreachable backend — returns BackendInfo(reachable=False, ...) instead.
    """
    base_url = config.backend_url
    own_client = client is None
    client = client if client is not None else httpx.AsyncClient()
    backend = OllamaBackend(base_url, client)
    try:
        try:
            ps = await backend.ps()
        except httpx.HTTPError as exc:
            return BackendInfo(
                reachable=False,
                base_url=base_url,
                detail=f"backend unreachable at {base_url} ({type(exc).__name__})",
            )
        return await _build_info(backend, base_url, ps, config.model)
    finally:
        if own_client:
            await client.aclose()


async def _build_info(
    backend: OllamaBackend, base_url: str, ps: dict[str, Any], preferred: str | None
) -> BackendInfo:
    chosen, ps_model = _select_model(_as_list(ps.get("models")), preferred)

    loaded_ctx = _extract_loaded_ctx(ps_model) if ps_model else None
    size_vram = _as_int(ps_model.get("size_vram")) if ps_model else None
    weight_size = _as_int(ps_model.get("size")) if ps_model else None
    digest = _as_str(ps_model.get("digest")) if ps_model else None
    quant = _dig_str(ps_model, "details", "quantization_level")
    param_size = _dig_str(ps_model, "details", "parameter_size")

    architecture: str | None = None
    advertised: int | None = None
    kv_bytes_per_token: int | None = None
    if chosen is not None:
        show = await _try_show(backend, chosen)
        if show is not None:
            model_info = _as_dict(show.get("model_info"))
            architecture = _as_str(model_info.get("general.architecture"))
            advertised = _extract_advertised_ctx(model_info, architecture)
            kv_bytes_per_token = _extract_kv_bytes_per_token(model_info, architecture)
            details = _as_dict(show.get("details"))
            quant = _as_str(details.get("quantization_level")) or quant
            param_size = _as_str(details.get("parameter_size")) or param_size

    if weight_size is None and chosen is not None:
        weight_size = await _try_weight_from_tags(backend, chosen)

    return BackendInfo(
        reachable=True,
        base_url=base_url,
        model=chosen,
        architecture=architecture,
        quantization=quant,
        parameter_size=param_size,
        weight_size_bytes=weight_size,
        size_vram_bytes=size_vram,
        advertised_max_ctx=advertised,
        loaded_ctx=loaded_ctx,
        digest=digest,
        kv_bytes_per_token=kv_bytes_per_token,
    )


async def _try_show(backend: OllamaBackend, model: str) -> dict[str, Any] | None:
    try:
        return await backend.show(model)
    except httpx.HTTPError:
        return None


async def _try_weight_from_tags(backend: OllamaBackend, model: str) -> int | None:
    try:
        tags = await backend.tags()
    except httpx.HTTPError:
        return None
    match = _select_model(_as_list(tags.get("models")), model)[1]
    return _as_int(match.get("size")) if match else None


def _select_model(
    models: list[Any], preferred: str | None
) -> tuple[str | None, dict[str, Any] | None]:
    entries = [_as_dict(m) for m in models]
    if preferred is not None:
        wanted = normalize_model_name(preferred)
        for entry in entries:
            candidates = (_as_str(entry.get("model")), _as_str(entry.get("name")))
            if any(c is not None and normalize_model_name(c) == wanted for c in candidates):
                return preferred, entry
        return preferred, None
    if entries:
        first = entries[0]
        return _as_str(first.get("model")) or _as_str(first.get("name")), first
    return None, None


def _extract_loaded_ctx(model: dict[str, Any]) -> int | None:
    """Loaded window from an /api/ps entry; None when the backend does not report one.

    None is the honest answer here — it degrades to "loaded N/A" rather than inventing a
    number that would make the advertised-vs-loaded comparison quietly meaningless.
    """
    for key in _LOADED_CTX_KEYS:
        value = _as_int(model.get(key))
        if value is not None:
            return value
    return None


def _extract_advertised_ctx(model_info: dict[str, Any], arch: str | None) -> int | None:
    """Advertised max from /api/show ``model_info``, by arch prefix then any suffix match."""
    if arch is not None:
        value = _as_int(model_info.get(f"{arch}.context_length"))
        if value is not None:
            return value
    for key, raw in model_info.items():
        if key.endswith(".context_length"):
            value = _as_int(raw)
            if value is not None:
                return value
    return None


# f16 is llama.cpp's default KV-cache element type and what Ollama loads with unless
# OLLAMA_KV_CACHE_TYPE overrides it. A quantized cache (q8_0) halves the real figure, so
# assuming f16 over-estimates KV — the safe direction for a headroom number, which must never
# advise its way into the OOM it exists to prevent.
_KV_ELEMENT_BYTES = 2


def _extract_kv_bytes_per_token(model_info: dict[str, Any], arch: str | None) -> int | None:
    """KV-cache bytes per context token, derived from the model's own GGUF metadata.

    ``attending_layers x kv_heads x (key_length + value_length) x element_bytes``. Every term
    is published by /api/show in the same ``model_info`` dict the advertised context comes
    from, which is what lets a hand-tuned constant be replaced by the model's own arithmetic.

    Validated against ``/api/ps size_vram`` on an RTX 3060 12GB: VRAM is linear in context,
    so loading the same model at several windows gives the true rate as the slope. Below about
    8K the line bends -- a fixed allocation still changing size -- so every figure here is
    fitted from 8K upward, across at least three points, in MiB per 1K tokens.

    ===================  =========  =========  =======  ===================
    model                predicted   measured    error  windows fitted
    ===================  =========  =========  =======  ===================
    qwen2.5-coder:7b         54.69      56.64    -3.4%  8K / 16K / 32K
    qwen3-4b                140.63     142.58    -1.4%  8K / 16K / 32K
    qwen3.5-9b (hybrid)      31.25      32.23    -3.0%  2K / 8K / 16K / 32K
    ===================  =========  =========  =======  ===================

    Three architectures, one per branch below: qwen2 omits its head dimensions and takes the
    fallback, qwen3 publishes them, qwen35 is the hybrid stack. All three read 1-4% *under*
    measurement -- llama.cpp's per-token scratch outside the cache proper. That residual
    slightly overstates headroom, by ~150 MiB on a 4 GB estimate, which is well inside the
    512 MiB ``vram_safety_margin_mib`` held back before any headroom figure is shown. The
    constant this replaces was 4-7x out, far past what any margin absorbs.

    qwen2.5-coder:14b is deliberately absent: it cannot be held in 12GB above 8K without
    spilling to CPU, and a spilled load reallocates, so no clean slope exists to fit here.

    Not every architecture is derivable, and the ones that are not return None so the caller
    falls back to the configured constant. Guessing here is worse than declining: an
    understated rate overstates VRAM headroom, which is advice that ends in the OOM Pharos
    exists to prevent.
    """
    if arch is None:
        return None
    if model_info.get(f"{arch}.attention.sliding_window") is not None:
        # Sliding-window attention caps each windowed layer's cache instead of letting it grow
        # with context, and which layers are windowed is not published (Gemma3 windows five of
        # every six). Deriving as though every layer were global overstates KV several-fold on
        # a long context, so this declines rather than guesses.
        return None
    layers = _as_int(model_info.get(f"{arch}.block_count"))
    kv_heads_raw = model_info.get(f"{arch}.attention.head_count_kv")
    if layers is None or layers <= 0 or kv_heads_raw is None:
        return None

    # head_count_kv is a scalar for a uniform stack, or a per-layer array for architectures
    # that vary it. An array already states the whole pattern, so it supersedes both
    # block_count and the hybrid interval below rather than combining with them; a zero entry
    # is a layer holding no KV cache at all, which is a legitimate count and not an error.
    if isinstance(kv_heads_raw, list):
        per_layer = [_as_int(value) for value in kv_heads_raw]
        if not per_layer or any(value is None or value < 0 for value in per_layer):
            return None
        total_kv_heads = sum(value for value in per_layer if value is not None)
    else:
        kv_heads = _as_int(kv_heads_raw)
        if kv_heads is None or kv_heads <= 0:
            return None
        # Hybrid stacks interleave state-space layers with attention, keeping a KV cache only
        # on every Nth. An SSM layer's state is fixed per sequence and does not grow with
        # context, so it belongs to the constant footprint rather than the per-token rate.
        # Ignoring this measured 4x high on qwen3.5-9b -- the error that made the hand-tuned
        # constant it replaces look plausible.
        interval = _as_int(model_info.get(f"{arch}.full_attention_interval"))
        if interval is not None and interval > 1:
            layers = layers // interval
        if layers <= 0:
            return None
        total_kv_heads = kv_heads * layers
    if total_kv_heads <= 0:
        return None

    key_len = _as_int(model_info.get(f"{arch}.attention.key_length"))
    value_len = _as_int(model_info.get(f"{arch}.attention.value_length"))
    if key_len is None or value_len is None:
        # Older GGUFs (qwen2 among them) omit both and imply embedding_length / head_count.
        fallback = _head_dim_fallback(model_info, arch)
        if fallback is None:
            return None
        key_len = fallback if key_len is None else key_len
        value_len = fallback if value_len is None else value_len
    if key_len <= 0 or value_len <= 0:
        return None
    return total_kv_heads * (key_len + value_len) * _KV_ELEMENT_BYTES


def _head_dim_fallback(model_info: dict[str, Any], arch: str) -> int | None:
    """Head dimension implied by embedding_length / head_count, for GGUFs omitting key_length."""
    embedding = _as_int(model_info.get(f"{arch}.embedding_length"))
    heads = _as_int(model_info.get(f"{arch}.attention.head_count"))
    if embedding is None or heads is None or embedding <= 0 or heads <= 0:
        return None
    dim = embedding // heads
    return dim if dim > 0 else None


# --- defensive coercion helpers (backend JSON is untyped) ---------------------------------------


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _dig(model: dict[str, Any], *keys: str) -> object:
    current: object = model
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _dig_str(model: dict[str, Any] | None, *keys: str) -> str | None:
    if model is None:
        return None
    return _as_str(_dig(model, *keys))
