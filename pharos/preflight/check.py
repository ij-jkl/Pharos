"""Assemble a pre-flight report: named files, exact counts, learned overhead, budget, verdict.

All budget arithmetic stays in the Accountant (via ``build_profile``); all token counting goes
through the existing GGUF tokenizer and content-hash cache. This module only composes.

THE FLOOR IS A LOWER BOUND, never a prediction. Exact for the prompt text and the files the
user explicitly named; the agent may read anything else it likes, and that is not in the
number. Client overhead, when known, is a labeled estimate learned from observed traffic
(see ``pharos.calibration``) or pinned in config — with neither, the floor omits it and the
report says so rather than inventing a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pharos.calibration import (
    Observation,
    OverheadEstimate,
    estimate_client_overhead,
    estimate_typical_output,
    load_observations,
)
from pharos.config import PharosConfig
from pharos.preflight.extract import Extraction, extract_references
from pharos.profiler.profiler import build_profile
from pharos.profiler.types import EnvironmentProfile
from pharos.tokenizer.cache import TokenCountCache
from pharos.tokenizer.gguf import GgufTokenizer, TokenCounter
from pharos.tokenizer.resolver import resolve_gguf_path

_HEURISTIC_CHARS_PER_TOKEN = 4  # same crude fallback the proxy uses, same label


class Verdict(Enum):
    FITS = "fits"
    EXCEEDS = "exceeds"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class CountedFile:
    display: str  # path as shown to the user (relative to root where possible)
    tokens: int
    found_by_search: bool


@dataclass(frozen=True, slots=True)
class CheckReport:
    """Everything the verdict renderer needs; no budget math happens after this point."""

    root: Path
    root_is_fallback: bool  # True when target_folder was unset and cwd stood in
    model: str | None
    counts_exact: bool  # False -> every count below is the labeled chars/4 heuristic
    prompt_tokens: int
    files: list[CountedFile]
    skipped_binary: list[str]
    extraction: Extraction  # directories / ambiguous / missing, surfaced by the renderer
    overhead: OverheadEstimate | None
    floor: int  # prompt + files + overhead-if-known: the "at least" number
    profile: EnvironmentProfile | None
    verdict: Verdict
    verdict_detail: str
    reserve_warning: str | None


async def run_check(
    config: PharosConfig,
    prompt: str,
    *,
    profile: EnvironmentProfile | None = None,
    skip_profile: bool = False,
) -> CheckReport:
    """Run the pre-flight for ``prompt``. ``profile`` injects a probe result (tests)."""
    root_is_fallback = config.target_folder is None
    # Resolved so "not found under ." never appears — the absolute root is the useful message.
    root = (Path(config.target_folder) if config.target_folder else Path.cwd()).resolve()

    if profile is None and not skip_profile:
        profile = await build_profile(config)
    backend_model = profile.backend.model if profile is not None else None
    model = config.model or backend_model

    counter, counts_exact = _resolve_counter(config, model)
    cache = TokenCountCache()
    identity = str(getattr(counter, "path", "")) if counter is not None else "heuristic"

    def count(text: str) -> int:
        if counter is not None:
            return cache.get_or_count(text, counter.count, identity=identity)
        return (len(text) + _HEURISTIC_CHARS_PER_TOKEN - 1) // _HEURISTIC_CHARS_PER_TOKEN

    prompt_tokens = count(prompt)
    extraction = extract_references(prompt, root)

    files: list[CountedFile] = []
    skipped_binary: list[str] = []
    for ref in extraction.files:
        display = _display_path(ref.path, root)
        try:
            content = ref.path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            # A binary file has no text token count; a fabricated one would poison the floor.
            skipped_binary.append(display)
            continue
        except OSError:
            extraction.missing.append(ref.raw)
            continue
        files.append(
            CountedFile(
                display=display, tokens=count(content), found_by_search=ref.found_by_search
            )
        )

    observations = load_observations(Path(config.observations_file))
    overhead = _overhead(config, observations, model)
    floor = prompt_tokens + sum(f.tokens for f in files) + (overhead.tokens if overhead else 0)

    verdict, detail = _verdict(floor, profile)
    reserve_warning = _reserve_warning(config, observations, model)

    return CheckReport(
        root=root,
        root_is_fallback=root_is_fallback,
        model=model,
        counts_exact=counts_exact,
        prompt_tokens=prompt_tokens,
        files=files,
        skipped_binary=skipped_binary,
        extraction=extraction,
        overhead=overhead,
        floor=floor,
        profile=profile,
        verdict=verdict,
        verdict_detail=detail,
        reserve_warning=reserve_warning,
    )


def _resolve_counter(config: PharosConfig, model: str | None) -> tuple[TokenCounter | None, bool]:
    gguf = resolve_gguf_path(config, model)
    if gguf is None:
        return None, False
    return GgufTokenizer(gguf), True


def _overhead(
    config: PharosConfig, observations: list[Observation], model: str | None
) -> OverheadEstimate | None:
    if config.client_overhead_tokens is not None:
        return OverheadEstimate(
            tokens=config.client_overhead_tokens,
            provenance="configured (client_overhead_tokens in pharos.toml)",
        )
    return estimate_client_overhead(observations, model)


def _verdict(floor: int, profile: EnvironmentProfile | None) -> tuple[Verdict, str]:
    if profile is None or not profile.backend.reachable:
        return Verdict.INDETERMINATE, "backend unreachable — floor printed, no budget to compare"
    budget = profile.budget
    if budget.usable_budget is None:
        return Verdict.INDETERMINATE, "no model resident — floor printed; load the model first"
    if floor > budget.usable_budget:
        over = floor - budget.usable_budget
        return Verdict.EXCEEDS, (
            f"the floor alone is {over:,} over the usable budget; anything the agent reads "
            f"on its own makes it worse"
        )
    margin_to_warn = (budget.warn_tokens or budget.usable_budget) - floor
    if margin_to_warn < 0:
        return Verdict.FITS, (
            f"fits, but already {-margin_to_warn:,} past the warn threshold "
            f"({budget.warn_tokens:,}) before the agent reads anything on its own"
        )
    return Verdict.FITS, f"floor leaves {margin_to_warn:,} before the warn threshold"


def _reserve_warning(
    config: PharosConfig, observations: list[Observation], model: str | None
) -> str | None:
    typical = estimate_typical_output(observations, model)
    if typical is None or config.response_reserve >= typical:
        return None
    return (
        f"response_reserve {config.response_reserve:,} is below the observed typical output "
        f"of ~{typical:,} tokens (median) — the usable budget may be optimistic"
    )


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)
