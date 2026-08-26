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
    # Tokens `pharos split` holds back in every scope part for the previous part's hand-off.
    # Measured across two real agent runs (qwen3.5-9b): 64, 74, 83, 107, 160, 161, 162, 168 and
    # 328 tokens — for the SAME instruction ("at most 10 lines"). A model's idea of ten lines is
    # not a constant, so this is a knob rather than a magic number: 500 clears every hand-off
    # observed, and an agent that writes essays needs it raised.
    handoff_reserve: int = Field(default=500, ge=0)
    # Carry Pharos's own record of what has landed on disk from part to part, alongside the
    # hand-off the model writes. On by default because the model's half is measurably
    # unreliable: across sixteen live runs (DESKTOP_VALIDATION §14) parts changed files and
    # then reported "NO CHANGES NEEDED", so the next part was handed a summary that was not
    # thin but wrong. The record cannot be wrong — it is written by the dispatcher as each
    # write succeeds, not by a model describing itself afterwards.
    #
    # It shares `handoff_reserve` with the prose rather than adding to it, taking at most half,
    # so turning it on cannot make a part overrun the budget its ceiling was computed from.
    #
    # Turn it off to see what the model's own hand-offs achieve alone, which is what every
    # coverage figure recorded before v0.6 measures.
    handoff_ledger: bool = True
    # The context window `pharos run` asks the backend to load the model with. Ollama picks a
    # conservative VRAM-based default (4,096 on a 12 GB card), which is too small to plan an
    # agentic edit against: the part scaffold alone fills it. Left unset, whatever is already
    # loaded is used and measured as-is — Pharos never silently changes a window it is also
    # reporting on. Set it and `pharos run` loads the model with that window and says so.
    num_ctx: int | None = Field(default=None, ge=256)
    # Most files `pharos run` puts in one part. It bounds how much WORK a part contains, not
    # how many tokens — the limit it exists for belongs to the model, not the hardware, and no
    # token count predicts it.
    #
    # Measured rather than guessed, on the same 13-file task run four times against
    # qwen2.5-coder:14b in a 16K window. A part completes about 1.5 to 2.0 files and then stops
    # believing itself finished, whatever it was given: 1.5, 2.0, 1.5 files per part when parts
    # held four. Window pressure was never the cause — peak usage sat at 42-51% of the ceiling
    # throughout. Nor was persuasion the answer: stating the target up front, naming the
    # outstanding files, and asking again all failed to push a part past roughly two.
    #
    # So the fix is arithmetic. At four files per part that task covered 46%, 62%, 46%. At two,
    # 92%. The cost is more parts, which is more hand-offs and more chances to drop the thread
    # — worth watching in the scorecard, and worth raising for a model that finishes more.
    max_files_per_part: int = Field(default=2, ge=1)
    # How many parts beyond the mechanical count a semantic grouping may spend to keep related
    # files together. Grouping by meaning packs worse than first-fit almost by definition —
    # coherence and density want different answers — so some slack has to be allowed or the
    # feature can never do anything. Unbounded slack is the failure mode though: a model that
    # returns one file per part has "grouped" nothing and bought a hand-off for each one.
    #
    # This is a stated preference, not a measurement, exactly like max_files_per_part's
    # existence is (its VALUE was measured; this one's has not been). Two extra parts is enough
    # for a 6-file task to split 3/3 where position packed 2/2/2, and not enough to degenerate.
    semantic_max_extra_parts: int = Field(default=2, ge=0)
    # Which model answers the grouping question. Unset, it is the model doing the work.
    #
    # Worth setting, because the two jobs want different models and the obvious default is not
    # the best one. Grouping is small and structured — partition six filenames, name each
    # group — and on the same task, measured on this machine (see DESKTOP_VALIDATION.md §18):
    #
    #   qwen2.5-coder:7b    4.2s   exact coverage, and the only clean db/http split
    #   qwen2.5-coder:14b  14.1s   exact coverage, incoherent groups
    #   gemma3:12b         20.9s   exact coverage, incoherent groups
    #   qwen3.5:9b         13.3s   DROPPED a file — rejected, fell back to position
    #   llama3.2:1b        10.2s   dropped four files — rejected
    #
    # The smallest coding model won outright and was three times faster than the model that
    # would otherwise have been asked. Bigger did not mean better here; it meant slower and,
    # for the run's own model, wrong. Nothing is lost when it is wrong — the proposal is
    # rejected and the mechanical grouping stands — but a call that always fails is a call
    # not worth making.
    #
    # It is not free, though, and that was not measured when the above was written. A 12 GB
    # card holds one model of this size, so a different grouper EVICTS the run's model and the
    # next operation reloads it: 4-5s each way here. Leaving this unset avoids the swap.
    # Which way that trades depends on how good the grouping is on your files, which is the
    # thing nothing here can measure for you.
    semantic_model: str | None = None
    # After the plan has run, go back over any assigned file that no part actually
    # changed, in a fresh conversation holding only the leftovers. One round, never more:
    # a second would be chasing a model that has declined the same work twice, and an
    # unbounded repair loop is how a run stops having a knowable cost. Turn it off to see
    # what the plan alone achieves - which is what the coverage figures above measure.
    repair_pass: bool = True
    # After the run, re-run the project's own checks and report what the run broke. Mechanical
    # only: exit codes from tools the repository already configures, never a model's opinion of
    # the code. Every check is also run BEFORE the first part, so a suite that was already red
    # is reported as such and never charged to the run.
    verify: bool = True
    # The checks to run. Unset, Pharos detects the ones this repository configures and that are
    # installed (ruff, pytest). Set it and the list is used verbatim -- which is how any other
    # ecosystem gets checked, since detection deliberately invents nothing.
    verify_commands: list[str] | None = None
    # Per-command ceiling. A check that outruns it is reported as skipped, never as passing: a
    # run must not be able to turn a slow suite into a green tick by waiting.
    verify_timeout_seconds: float = Field(default=300.0, gt=0.0)

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
