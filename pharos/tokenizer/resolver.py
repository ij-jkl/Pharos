"""Resolve the GGUF path used for exact token counting.

Primary and robust: an explicit ``gguf_path`` in pharos.toml. Optional fallback: Ollama's
local blob store — ``manifests/<registry>/<namespace>/<name>/<tag>`` is a JSON manifest whose
model layer digest names a file under ``blobs/`` (``sha256-...``, no .gguf extension); the
store root honors ``OLLAMA_MODELS``.

Every failure path returns None: no GGUF just means input counting degrades to the labeled
heuristic — never a crash, and never a hard dependency on Ollama's storage layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pharos.config import PharosConfig

_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"


def resolve_gguf_path(config: PharosConfig, model: str | None = None) -> Path | None:
    """Resolve the GGUF for ``model`` (or ``config.model``); None when nothing resolves."""
    if config.gguf_path:
        path = Path(config.gguf_path)
        return path if path.is_file() else None
    name = model or config.model
    if not name:
        return None
    return resolve_from_store(name)


def resolve_from_store(model: str, root: Path | None = None) -> Path | None:
    """Look ``model`` (``name[:tag]``, tag defaults to "latest") up in an Ollama blob store."""
    store = root if root is not None else _default_store_root()
    manifest_path = _find_manifest(store / "manifests", model)
    if manifest_path is None:
        return None
    digest = _model_layer_digest(manifest_path)
    if digest is None:
        return None
    blob_name = digest.replace(":", "-")
    if "/" in blob_name or "\\" in blob_name:
        return None
    blob = store / "blobs" / blob_name
    return blob if blob.is_file() else None


def _default_store_root() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env) if env else Path.home() / ".ollama" / "models"


def _find_manifest(manifests: Path, model: str) -> Path | None:
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    if not name or not manifests.is_dir():
        return None
    try:
        # The registry host / namespace prefix varies, so match on the name/tag suffix.
        candidates = sorted(manifests.glob(f"**/{name}/{tag}"))
    except (OSError, ValueError):
        return None
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _model_layer_digest(manifest_path: Path) -> str | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        return None
    for layer in layers:
        if isinstance(layer, dict) and layer.get("mediaType") == _MODEL_MEDIA_TYPE:
            digest = layer.get("digest")
            if isinstance(digest, str) and digest:
                return digest
    return None
