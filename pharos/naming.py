"""Ollama model-name handling, in one place.

Ollama accepts an untagged name on every endpoint but reports the resolved ``name:tag`` back
from /api/ps and /api/tags. Comparing those two forms with ``==`` silently fails, which is how
the context-mismatch banner came to be suppressed for the most natural pharos.toml value (see
DESKTOP_VALIDATION.md item 1). Every name comparison goes through here.

``pharos.tokenizer.resolver._find_manifest`` applies the same "tag defaults to latest" rule,
but splits name from tag for path globbing rather than comparing, so it keeps its own parsing.
"""

from __future__ import annotations

DEFAULT_TAG = "latest"


def normalize_model_name(name: str) -> str:
    """Return ``name`` with an explicit tag, defaulting to "latest".

    A registry host with a port ("localhost:5000/foo") carries no tag: its last colon precedes
    a "/", so it is treated as untagged rather than split at the port.
    """
    _, sep, tail = name.rpartition(":")
    if sep and tail and "/" not in tail:
        return name
    return f"{name}:{DEFAULT_TAG}"


def same_model(left: str | None, right: str | None) -> bool:
    """True when both names refer to the same model once tags are normalized.

    None on either side means "unknown", which is never treated as a match: an unidentified
    model must not be silently assumed to be the configured one.
    """
    if left is None or right is None:
        return False
    return normalize_model_name(left) == normalize_model_name(right)
