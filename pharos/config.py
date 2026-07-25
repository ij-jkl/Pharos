"""Load and validate `pharos.toml` into a typed config object (Pydantic v2, read via tomllib).

A single config file is the seam that makes moving laptop -> desktop a change here, never in
source. `pharos.toml` is gitignored; ship `pharos.toml.example` instead.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(Exception):
    """Raised when pharos.toml is missing (when required), malformed, or fails validation."""


class PharosConfig(BaseModel):
    """Typed Pharos configuration; defaults mirror pharos.toml.example."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    backend_url: str = "http://localhost:11434"
    model: str | None = None
    gguf_path: str | None = None
    response_reserve: int = Field(default=1024, ge=0)
    warn_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    alert_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    kv_mib_per_1k: float = Field(default=32.0, gt=0.0)
    # Free VRAM held back before any headroom estimate: driver allocations, fragmentation and
    # display compositing all claim memory with no warning, and advice that consumes the last
    # free byte is an OOM invitation, not guidance.
    vram_safety_margin_mib: int = Field(default=512, ge=0)
    proxy_host: str = "127.0.0.1"
    proxy_port: int = Field(default=11435, ge=1, le=65535)
    log_file: str = Field(default="pharos.log", min_length=1)
    target_folder: str | None = None
    # Pre-flight: tokens the coding agent adds on top of a pasted prompt (system prompt, tool
    # catalogue, injected context). When set it overrides the value learned from observed
    # traffic; when neither exists the pre-flight floor omits overhead and says so.
    client_overhead_tokens: int | None = Field(default=None, ge=0)
    # Where the proxy records per-request token counts (counts only, never text) so the
    # pre-flight check can calibrate against real observed traffic.
    observations_file: str = Field(default="pharos_observations.json", min_length=1)

    @model_validator(mode="after")
    def _warn_below_alert(self) -> Self:
        if self.warn_threshold > self.alert_threshold:
            raise ValueError("warn_threshold must be <= alert_threshold")
        return self


def load_config(path: str | Path | None = None) -> PharosConfig:
    """Load pharos.toml into a PharosConfig.

    With ``path=None``, look for ``pharos.toml`` in the current directory; if it is absent, return
    an all-defaults config so the profiler still runs on a bare machine. A malformed or invalid
    file raises ConfigError, as does an explicitly given path that does not exist.
    """
    if path is None:
        default_path = Path("pharos.toml")
        if not default_path.exists():
            return PharosConfig()
        path = default_path
    else:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")

    try:
        with path.open("rb") as handle:
            data: dict[str, Any] = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc

    try:
        return PharosConfig(**data)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}: {exc}") from exc
