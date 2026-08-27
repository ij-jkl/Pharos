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
    """Typed Pharos configuration.

    Defaults mirror pharos.toml.example, which documents the reasoning and the measurements
    behind each one. The comments here carry only what a reader of this file needs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    backend_url: str = "http://localhost:11434"
    model: str | None = None
    gguf_path: str | None = None
    response_reserve: int = Field(default=1024, ge=0)
    warn_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    alert_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    # Fallback only: normally derived per model from GGUF metadata and labelled as such.
    kv_mib_per_1k: float = Field(default=32.0, gt=0.0)
    # Free VRAM held back before any headroom estimate. Driver allocations, fragmentation and
    # display compositing all claim memory without warning, so advice that spends the last free
    # byte is an OOM invitation rather than guidance.
    vram_safety_margin_mib: int = Field(default=512, ge=0)
    proxy_host: str = "127.0.0.1"
    proxy_port: int = Field(default=11435, ge=1, le=65535)
    log_file: str = Field(default="pharos.log", min_length=1)
    target_folder: str | None = None
    # Tokens the coding agent adds on top of a pasted prompt (system prompt, tool catalogue).
    # Set, it overrides the value learned from observed traffic; with neither, the pre-flight
    # floor omits overhead and says so.
    client_overhead_tokens: int | None = Field(default=None, ge=0)
    # Tokens the agent pulls in ON ITS OWN over one task. Same override rule; with neither, the
    # check prints FLOOR and CEILING and says the middle is unknown rather than standing a guess
    # in the gap.
    agent_read_tokens: int | None = Field(default=None, ge=0)
    # Where the proxy records per-request token counts (counts only, never text) for the
    # pre-flight to calibrate against.
    observations_file: str = Field(default="pharos_observations.json", min_length=1)
    # Tokens held back in every scope part for the previous part's hand-off. A knob rather than
    # a constant: measured hand-offs to one "at most 10 lines" instruction ranged 64-328 tokens.
    handoff_reserve: int = Field(default=500, ge=0)
    # Carry Pharos's own record of what landed on disk from part to part, alongside the model's
    # hand-off. It SHARES handoff_reserve rather than adding to it, taking at most half, so
    # turning it on cannot make a part overrun the budget its ceiling was computed from.
    handoff_ledger: bool = True
    # The window `pharos run` asks the backend to load. Left unset, whatever is already loaded is
    # used and measured as-is: Pharos never silently changes a window it is also reporting on.
    num_ctx: int | None = Field(default=None, ge=256)
    # Most files `pharos run` puts in one part. It bounds how much WORK a part contains, not how
    # many tokens -- the limit belongs to the model, not the hardware, and no token count
    # predicts it. Measured: a part completes 1.5-2.0 files whatever it is given.
    max_files_per_part: int = Field(default=2, ge=1)
    # How many parts beyond the mechanical count a semantic grouping may spend to keep related
    # files together. Grouping by meaning packs worse than first-fit almost by definition, so
    # some slack is needed; unbounded slack lets a model "group" one file per part.
    semantic_max_extra_parts: int = Field(default=2, ge=0)
    # Which model answers the grouping question. Unset, it is the model doing the work, which
    # costs no swap -- a different grouper evicts the run's model on a single-model card.
    semantic_model: str | None = None
    # Go back over any assigned file no part changed, in a fresh conversation holding only the
    # leftovers. One round, never more: an unbounded repair loop is how a run stops having a
    # knowable cost.
    repair_pass: bool = True
    # Re-run the project's own checks afterwards and report what the run broke. Mechanical only:
    # exit codes from the repository's tools, never a model's opinion. Every check also runs
    # BEFORE the first part, so a suite that was already red is never charged to the run.
    verify: bool = True
    # The checks to run. Unset, Pharos detects the ones this repository configures and that are
    # installed; set, the list is used verbatim, which is how any other ecosystem is checked.
    verify_commands: list[str] | None = None
    # Per-command ceiling. A check that outruns it is reported as skipped, never as passing: a
    # run must not be able to turn a slow suite green by waiting.
    verify_timeout_seconds: float = Field(default=300.0, gt=0.0)
    # End the run at the first part that leaves a file it wrote unparseable. Off by default, and
    # that is a judgement: a broken file is sometimes repaired by a later part or by the repair
    # sweep, and every published coverage figure was measured on runs that ran to the end.
    stop_on_break: bool = False
    # A project check also runs after EVERY part when the baseline measured it at or below this,
    # so a part that breaks the linter is named the way a part that breaks the parser is. The
    # parser alone misses half the damage. Set it to 0 to check only that files still parse.
    per_part_check_seconds: float = Field(default=3.0, ge=0.0)
    # Where the profiler remembers what each model occupied at each window, so the VRAM cost
    # of a context token can be fitted from this machine's own readings rather than taken
    # from a constant. Two numbers per window, safe to delete: it is re-measured by use.
    vram_memory_file: str = Field(default="pharos_vram.json", min_length=1)
    # Where `pharos run` remembers how many tokens the backend's own count runs above its
    # projection, per model. Counts only, and safe to delete: the next run measures it again.
    template_memory_file: str = Field(default="pharos_templates.json", min_length=1)

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
