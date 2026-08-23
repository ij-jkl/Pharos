"""Pre-flight tests: reference extraction, resolution against a real tree, and the composed
check report — floor math, overhead provenance, verdicts and the reserve warning.

``run_check`` is exercised with an injected EnvironmentProfile (no network) and configs whose
model resolves no GGUF (no dependence on the host's Ollama store): counts are heuristic
chars/4, which is exactly what the report must then say.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pharos.accountant import Accountant
from pharos.calibration import Observation, ObservationRecorder
from pharos.config import PharosConfig
from pharos.preflight.check import Verdict, run_check
from pharos.preflight.extract import extract_references
from pharos.profiler.types import BackendInfo, EnvironmentProfile, GpuInfo

# --- extraction: which spans become candidates ------------------------------------------------


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "auth" / "login.py").write_text("def login(): pass\n", encoding="utf-8")
    (tmp_path / "src" / "auth" / "session.py").write_text("SESSION = {}\n", encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "utils.py").write_text("B = 2\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("all:\n\techo hi\n", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe binary")
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("ignored\n", encoding="utf-8")
    return tmp_path


def test_extracts_backticked_quoted_and_bare_references(tree: Path) -> None:
    prompt = (
        "Refactor `src/auth/login.py` and \"src/auth/session.py\" plus src/utils.py "
        "before v0.2 ships, e.g. today."
    )
    result = extract_references(prompt, tree)
    found = {f.raw for f in result.files}
    assert found == {"src/auth/login.py", "src/auth/session.py", "src/utils.py"}
    assert result.missing == []  # "v0.2" and "e.g." never became candidates


def test_bare_token_needs_separator_or_known_extension(tree: Path) -> None:
    result = extract_references("look at login.py and at Makefile", tree)
    assert [f.raw for f in result.files] == ["login.py"]  # unique basename -> found by search
    assert result.files[0].found_by_search is True
    # Bare "Makefile" has no separator and no known extension: not a candidate at all.
    assert result.missing == []


def test_quoted_extensionless_name_resolves(tree: Path) -> None:
    result = extract_references("update the `Makefile` please", tree)
    assert [f.raw for f in result.files] == ["Makefile"]


def test_quoted_prose_is_dropped_silently(tree: Path) -> None:
    result = extract_references("the word 'hello' and `uv sync` are not files", tree)
    assert result.files == []
    assert result.missing == []  # non-path-like failures are prose, not noise


def test_ambiguous_basename_is_reported_not_guessed(tree: Path) -> None:
    result = extract_references("fix utils.py", tree)
    assert result.files == []
    assert "utils.py" in result.ambiguous
    assert len(result.ambiguous["utils.py"]) == 2


def test_dir_qualified_reference_disambiguates(tree: Path) -> None:
    result = extract_references("fix tests/utils.py", tree)
    assert [f.raw for f in result.files] == ["tests/utils.py"]
    assert result.ambiguous == {}


def test_missing_pathlike_reference_is_reported(tree: Path) -> None:
    result = extract_references("see docs/design.md for the plan", tree)
    assert result.files == []
    assert result.missing == ["docs/design.md"]


def test_directory_reference_is_reported_not_expanded(tree: Path) -> None:
    result = extract_references("read src/auth/ carefully", tree)
    assert result.files == []
    # raw is shown as written in the prompt, trailing slash and all.
    assert [d.raw for d in result.directories] == ["src/auth/"]


def test_same_file_referenced_twice_counts_once(tree: Path) -> None:
    result = extract_references("`src/utils.py` then src/utils.py again", tree)
    assert len(result.files) == 1


def test_dotfile_is_path_like(tree: Path) -> None:
    result = extract_references("respect .gitignore", tree)
    assert [f.raw for f in result.files] == [".gitignore"]


def test_ignored_dirs_are_pruned_from_search(tree: Path) -> None:
    result = extract_references("open dep.py", tree)
    assert result.files == []
    assert result.missing == ["dep.py"]  # node_modules is never searched


def test_trailing_punctuation_is_stripped(tree: Path) -> None:
    result = extract_references("start with src/utils.py, then stop.", tree)
    assert [f.raw for f in result.files] == ["src/utils.py"]


# --- run_check: composition ---------------------------------------------------------------------


def _profile(*, loaded_ctx: int | None, reachable: bool = True) -> EnvironmentProfile:
    gpu = GpuInfo(available=True, name="RTX 3060", total_mib=12288, used_mib=11000, free_mib=1288)
    backend = BackendInfo(
        reachable=reachable,
        base_url="http://localhost:11434",
        model="test-model-not-installed" if reachable else None,
        advertised_max_ctx=262144 if reachable else None,
        loaded_ctx=loaded_ctx,
    )
    budget = Accountant(_config(Path("."))).report(loaded_ctx=loaded_ctx, gpu=gpu)
    return EnvironmentProfile(
        gpu=gpu,
        backend=backend,
        budget=budget,
        ctx_mismatch=loaded_ctx is not None,
        ctx_mismatch_ratio=None,
    )


def _config(root: Path, **overrides: object) -> PharosConfig:
    base: dict[str, object] = {
        "model": "test-model-not-installed",  # resolves no GGUF -> heuristic counting
        "target_folder": str(root),
        "observations_file": str(root / "obs.json"),
        "response_reserve": 1024,
    }
    base.update(overrides)
    return PharosConfig(**base)  # type: ignore[arg-type]


async def test_floor_sums_prompt_files_and_overhead(tree: Path) -> None:
    config = _config(tree, client_overhead_tokens=1000)
    prompt = "fix `src/auth/login.py` now"
    report = await run_check(config, prompt, profile=_profile(loaded_ctx=32768))

    assert report.counts_exact is False  # no GGUF -> labeled heuristic
    file_tokens = (len("def login(): pass\n") + 3) // 4
    prompt_tokens = (len(prompt) + 3) // 4
    assert [f.tokens for f in report.files] == [file_tokens]
    assert report.overhead is not None
    assert report.overhead.tokens == 1000
    assert "configured" in report.overhead.provenance
    assert report.floor == prompt_tokens + file_tokens + 1000
    assert report.verdict is Verdict.FITS


async def test_learned_overhead_used_when_config_not_set(tree: Path) -> None:
    recorder = ObservationRecorder(tree / "obs.json")
    recorder.add(
        Observation(
            ts=1.0,
            endpoint="openai-chat",
            model="test-model-not-installed:latest",
            input_tokens=9_000,
            input_exact=True,
            user_tokens=500,
            messages=1,
            output_tokens=700,
        )
    )
    recorder._flush_sync()

    report = await run_check(_config(tree), "check `Makefile`", profile=_profile(loaded_ctx=32768))
    assert report.overhead is not None
    assert report.overhead.tokens == 8_500
    assert "learned" in report.overhead.provenance


async def test_no_overhead_source_leaves_floor_without_it(tree: Path) -> None:
    report = await run_check(_config(tree), "check `Makefile`", profile=_profile(loaded_ctx=32768))
    assert report.overhead is None  # renderer says so; nothing is invented


async def test_exceeds_verdict(tree: Path) -> None:
    big = tree / "big.txt"
    big.write_text("x" * 200_000, encoding="utf-8")  # 50k heuristic tokens
    report = await run_check(
        _config(tree), "summarize big.txt", profile=_profile(loaded_ctx=32768)
    )
    assert report.verdict is Verdict.EXCEEDS
    assert "over the usable budget" in report.verdict_detail


async def test_indeterminate_when_no_model_resident(tree: Path) -> None:
    report = await run_check(
        _config(tree), "check `Makefile`", profile=_profile(loaded_ctx=None)
    )
    assert report.verdict is Verdict.INDETERMINATE
    assert report.floor > 0  # the floor still prints; only the comparison is withheld


async def test_indeterminate_when_backend_unreachable(tree: Path) -> None:
    report = await run_check(
        _config(tree),
        "check `Makefile`",
        profile=_profile(loaded_ctx=None, reachable=False),
    )
    assert report.verdict is Verdict.INDETERMINATE
    assert "unreachable" in report.verdict_detail


async def test_binary_file_is_skipped_not_counted(tree: Path) -> None:
    report = await run_check(
        _config(tree), "describe logo.png", profile=_profile(loaded_ctx=32768)
    )
    assert report.files == []
    assert report.skipped_binary == ["logo.png"]


async def test_reserve_warning_from_observed_output(tree: Path) -> None:
    recorder = ObservationRecorder(tree / "obs.json")
    for output in (1_500, 1_700, 1_900):
        recorder.add(
            Observation(
                ts=1.0,
                endpoint="ollama-chat",
                model="test-model-not-installed:latest",
                input_tokens=100,
                input_exact=True,
                user_tokens=50,
                messages=1,
                output_tokens=output,
            )
        )
    recorder._flush_sync()

    report = await run_check(
        _config(tree), "check `Makefile`", profile=_profile(loaded_ctx=32768)
    )
    assert report.reserve_warning is not None
    assert "1,700" in report.reserve_warning  # the median
    assert "1,024" in report.reserve_warning


async def test_unresolved_references_reach_the_report(tree: Path) -> None:
    report = await run_check(
        _config(tree),
        "fix utils.py and docs/gone.md under src/auth/",
        profile=_profile(loaded_ctx=32768),
    )
    assert "utils.py" in report.extraction.ambiguous
    assert report.extraction.missing == ["docs/gone.md"]
    assert [d.raw for d in report.extraction.directories] == ["src/auth/"]


# ------------------------------------------------------------------- an explicit target


@pytest.mark.anyio
async def test_target_gives_a_verdict_with_no_backend_at_all(tmp_path: Path) -> None:
    """`--target N` was accepted by check and silently ignored: 500 and 1,000,000 agreed.

    Judging a prompt against a window you have not loaded is the whole reason the flag
    exists, and check is where you would most want it -- there is nothing to plan, just a
    number to be under.
    """
    (tmp_path / "big.py").write_text("x" * 4000, encoding="utf-8")  # ~1,000 heuristic tokens
    config = _config(tmp_path)
    prompt = "Refactor big.py"

    over = await run_check(config, prompt, skip_profile=True, target=300)
    assert over.verdict is Verdict.EXCEEDS
    assert "over the requested target (300)" in over.verdict_detail

    under = await run_check(config, prompt, skip_profile=True, target=50_000)
    assert under.verdict is Verdict.FITS
    assert "requested target (50,000)" in under.verdict_detail

    # Without one, no backend still means no verdict — the target is the only thing that
    # can stand in for a measurement, and inventing a default would be a guess.
    silent = await run_check(config, prompt, skip_profile=True)
    assert silent.verdict is Verdict.INDETERMINATE


@pytest.mark.anyio
async def test_target_overrides_a_live_budget(tmp_path: Path) -> None:
    """Asked for a number, answer against that number — not against what happens to be loaded."""
    (tmp_path / "big.py").write_text("x" * 4000, encoding="utf-8")
    config = _config(tmp_path)
    profile = _profile(loaded_ctx=131_072)  # roomy: the live budget would say FITS
    report = await run_check(config, "Refactor big.py", profile=profile, target=300)
    assert report.verdict is Verdict.EXCEEDS
    assert "requested target" in report.verdict_detail
