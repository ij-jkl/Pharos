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


# --- the third number (v1.0) ----------------------------------------------------------------


async def test_a_pinned_prediction_is_not_described_as_something_agents_did(
    tmp_path: Path,
) -> None:
    """A configured number has no conversations behind it, and the prose must not pretend."""
    (tmp_path / "one.py").write_text("x = 1\n" * 200, encoding="utf-8")
    config = _config(tmp_path, agent_read_tokens=3000)
    report = await run_check(config, "Refactor `one.py`", skip_profile=True, target=1000)

    assert report.reads is not None and not report.reads.learned
    assert report.expected == report.floor + 3000
    assert report.expected_warning is not None
    assert "historically" not in report.expected_warning
    assert "0 conversations" not in report.expected_warning
    assert "agent_read_tokens" in report.expected_warning


async def test_with_nothing_learned_the_check_prints_two_numbers_not_three(
    tmp_path: Path,
) -> None:
    (tmp_path / "one.py").write_text("x = 1\n", encoding="utf-8")
    report = await run_check(
        _config(tmp_path), "Refactor `one.py`", skip_profile=True, target=100_000
    )
    assert report.reads is None
    assert report.expected is None
    assert report.expected_warning is None


async def test_the_expected_warning_only_fires_when_the_floor_itself_fits(
    tmp_path: Path,
) -> None:
    """A floor that already exceeds is a plain EXCEEDS; adding "and it gets worse" to it is
    noise on top of a verdict that has already been given."""
    (tmp_path / "one.py").write_text("x = 1\n" * 200, encoding="utf-8")
    config = _config(tmp_path, agent_read_tokens=3000)
    report = await run_check(config, "Refactor `one.py`", skip_profile=True, target=10)
    assert report.verdict is Verdict.EXCEEDS
    assert report.expected_warning is None


def await_check(root: Path, prompt: str, **kw: object):
    """Synchronous wrapper: these assertions are about extraction, not about async."""
    import asyncio

    return asyncio.run(
        run_check(_config(root), prompt, skip_profile=True, target=100_000, **kw)  # type: ignore[arg-type]
    )


def test_a_format_spec_in_backticks_is_not_a_missing_file(tmp_path: Path) -> None:
    """A prompt ABOUT string formatting spells out `%.2f` and `:.2f`, and both used to surface
    in the verdict as files that could not be found. The module's stated bias is against false
    positives, and no dotfile convention starts a name with a digit."""
    prompt = "Carry the spec across: `%.2f` becomes `:.2f`, `%-30s` becomes `:<30`."
    report = await_check(tmp_path, prompt)
    assert report.extraction.missing == []


def test_a_real_dotfile_is_still_found_by_name(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    report = await_check(tmp_path, "Update `.gitignore` please")
    assert [f.display for f in report.files] == [".gitignore"]


def test_a_missing_dotfile_is_still_reported(tmp_path: Path) -> None:
    report = await_check(tmp_path, "Update `.env` please")
    assert report.extraction.missing == [".env"]


# --- --exclude: the file a prompt names in order to forbid it ---------------------------------
#
# Found on a real giant prompt. It said "Do not touch `shop/notifications/templates.py`" and
# "Do not touch the tests. `tests/test_shop.py`" — and both were duly extracted, counted into
# the floor, and scoped to a part, whose scope block then told the agent it MAY open them. The
# difference between naming a file as work and naming it as a prohibition is in the meaning of
# the sentence, and nothing here reads meaning. So it is a flag, like --resolve.


def test_an_excluded_file_leaves_the_floor(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x = 1\n" * 50, encoding="utf-8")
    (tmp_path / "leave.py").write_text("y = 2\n" * 50, encoding="utf-8")
    prompt = "Refactor `keep.py`. Do not touch `leave.py`."

    both = await_check(tmp_path, prompt)
    assert sorted(f.display for f in both.files) == ["keep.py", "leave.py"]

    one = await_check(tmp_path, prompt, exclude=["leave.py"])
    assert [f.display for f in one.files] == ["keep.py"]
    assert one.floor < both.floor


def test_an_excluded_file_is_listed_never_silently_dropped(tmp_path: Path) -> None:
    (tmp_path / "leave.py").write_text("y = 2\n", encoding="utf-8")
    report = await_check(tmp_path, "Do not touch `leave.py`.", exclude=["leave.py"])
    assert report.excluded == ["leave.py"]


def test_an_exclusion_reaches_inside_a_named_directory(tmp_path: Path) -> None:
    """"Everything in src/ except the generated one" is a sentence Pharos has to honour."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("x = 1\n" * 50, encoding="utf-8")
    (tmp_path / "src" / "generated.py").write_text("y = 2\n" * 50, encoding="utf-8")

    whole = await_check(tmp_path, "Refactor everything in `src/`")
    pruned = await_check(tmp_path, "Refactor everything in `src/`", exclude=["generated.py"])
    assert whole.directory_tokens > pruned.directory_tokens
    assert "src/generated.py" in [p.replace(chr(92), "/") for p in pruned.excluded]


def test_a_bare_filename_and_a_full_path_both_exclude(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "leave.py").write_text("y = 2\n", encoding="utf-8")
    prompt = "Do not touch `pkg/leave.py`."
    assert await_check(tmp_path, prompt, exclude=["leave.py"]).files == []
    assert await_check(tmp_path, prompt, exclude=["pkg/leave.py"]).files == []


def test_excluding_something_nobody_named_changes_nothing(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    report = await_check(tmp_path, "Refactor `keep.py`", exclude=["absent.py"])
    assert [f.display for f in report.files] == ["keep.py"]
    assert report.excluded == []


# --- one display path, after two copies had already drifted ---------------------------------------


def test_a_path_inside_the_workspace_is_shown_relative(tmp_path: Path) -> None:
    from pharos.paths import display_path

    assert display_path(tmp_path / "src" / "a.py", tmp_path) == str(Path("src/a.py"))


def test_a_path_outside_the_workspace_is_left_absolute(tmp_path: Path) -> None:
    """Never a misleading relative answer: `../../elsewhere/a.py` reads like a project file."""
    from pharos.paths import display_path

    outside = tmp_path.parent / "elsewhere" / "a.py"
    assert display_path(outside, tmp_path / "work") == str(outside)


def test_display_survives_a_root_that_needs_resolving(tmp_path: Path) -> None:
    """The two copies of this disagreed on exactly this point -- one resolved the root before
    comparing and the other did not, so the same file could appear by name in the verdict and
    by absolute path in the plan printed beside it."""
    from pharos.paths import display_path

    root = tmp_path / "work"
    (root / "src").mkdir(parents=True)
    target = (root / "src" / "a.py").resolve()
    # A root the user typed that is not in resolved form: both spellings must agree.
    assert display_path(target, root) == display_path(target, root.resolve())
