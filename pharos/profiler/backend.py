"""Ollama backend adapter — a thin seam for a future backend-agnostic layer.

Reads /api/tags, /api/show and /api/ps to detect the loaded model, its quantization, the
advertised max context (GGUF metadata) and the ACTUAL loaded context window. Degrades to
BackendInfo(reachable=False, ...) when the backend cannot be reached (e.g. the dev laptop with
no Ollama running) rather than raising.

Field extraction is deliberately defensive: exact JSON key names vary across Ollama versions and
get confirmed on the desktop. The /api/ps loaded-context value is looked up under several
candidate keys, and the advertised max is found by architecture prefix or any *.context_length.
"""

from __future__ import annotations

from typing import Any

import httpx

from pharos.config import PharosConfig
from pharos.profiler.types import BackendInfo

_DEFAULT_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
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
    quant = _dig_str(ps_model, "details", "quantization_level")
    param_size = _dig_str(ps_model, "details", "parameter_size")

    architecture: str | None = None
    advertised: int | None = None
    if chosen is not None:
        show = await _try_show(backend, chosen)
        if show is not None:
            model_info = _as_dict(show.get("model_info"))
            architecture = _as_str(model_info.get("general.architecture"))
            advertised = _extract_advertised_ctx(model_info, architecture)
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
        for entry in entries:
            if preferred in (entry.get("model"), entry.get("name")):
                return preferred, entry
        return preferred, None
    if entries:
        first = entries[0]
        return _as_str(first.get("model")) or _as_str(first.get("name")), first
    return None, None


def _extract_loaded_ctx(model: dict[str, Any]) -> int | None:
    for key in _LOADED_CTX_KEYS:
        value = _as_int(model.get(key))
        if value is not None:
            return value
    return _as_int(_dig(model, "details", "context_length"))


def _extract_advertised_ctx(model_info: dict[str, Any], arch: str | None) -> int | None:
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
