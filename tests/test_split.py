"""Split tests: scope packing, file slicing, text segmentation, and the CLI's exit codes.

Same discipline as the pre-flight tests — an injected EnvironmentProfile (no network) and a
model that resolves no GGUF, so counting is the deterministic chars/4 heuristic and every
assertion about token arithmetic is reproducible on any machine.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pharos.accountant import Accountant
from pharos.calibration import ReadEstimate
from pharos.config import PharosConfig
from pharos.preflight.check import run_check
from pharos.preflight.cli import main as check_main
from pharos.preflight.cli import split_main
from pharos.preflight.split import SplitMode, build_plan
from pharos.profiler.types import BackendInfo, EnvironmentProfile, GpuInfo


def _config(root: Path, **overrides: object) -> PharosConfig:
    base: dict[str, object] = {
        "model": "test-model-not-installed",  # resolves no GGUF -> heuristic counting
        "target_folder": str(root),
        "observations_file": str(root / "obs.json"),
        "response_reserve": 1024,
    }
    base.update(overrides)
    return PharosConfig(**base)  # type: ignore[arg-type]


def _profile(root: Path, *, loaded_ctx: int | None, reachable: bool = True) -> EnvironmentProfile:
    gpu = GpuInfo(available=True, name="RTX 3060", total_mib=12288, used_mib=11000, free_mib=1288)
    backend = BackendInfo(
        reachable=reachable,
        base_url="http://localhost:11434",
        model="test-model-not-installed" if reachable else None,
        advertised_max_ctx=262144 if reachable else None,
        loaded_ctx=loaded_ctx,
    )
    return EnvironmentProfile(
        gpu=gpu,
        backend=backend,
        budget=Accountant(_config(root)).report(loaded_ctx=loaded_ctx, gpu=gpu),
        ctx_mismatch=False,
        ctx_mismatch_ratio=None,
    )


async def _plan(root: Path, prompt: str, *, loaded_ctx: int | None = 8192, **overrides: object):
    config = _config(root, **overrides)
    report = await run_check(config, prompt, profile=_profile(root, loaded_ctx=loaded_ctx))
    return report, build_plan(config, prompt, report)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Three files of ~1,500 heuristic tokens each (6,000 chars): no two fit one small part.

    Sized to clear the 3,072-token usable budget these tests plan against (4,096 loaded minus
    a 1,024 reserve) with room to spare, not to sit on it. At the original 4,000 characters the
    three files came to 3,000 tokens — 72 short — and the test only saw "exceeds" on Windows,
    where write_text had silently inflated each file by its 666 CRLFs. See conftest's
    ``lf_only_writes`` for why that is no longer possible.
    """
    (tmp_path / "src").mkdir()
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (tmp_path / "src" / name).write_text("x = 1\n" * 1000, encoding="utf-8")
    return tmp_path


# --- scope mode ---------------------------------------------------------------------------


async def test_scope_split_covers_every_file_exactly_once(tree: Path) -> None:
    prompt = "Refactor `src/alpha.py`, `src/beta.py` and `src/gamma.py` consistently."
    report, plan = await _plan(tree, prompt, loaded_ctx=4096)

    assert report.verdict.value == "exceeds"
    assert plan.mode is SplitMode.SCOPE
    assert plan.ok
    scoped = [f.display.replace("\\", "/") for part in plan.parts for f in part.files]
    assert sorted(scoped) == ["src/alpha.py", "src/beta.py", "src/gamma.py"]
    assert len(plan.parts) > 1


async def test_every_part_is_projected_within_the_target(tree: Path) -> None:
    prompt = "Refactor `src/alpha.py`, `src/beta.py` and `src/gamma.py`."
    _, plan = await _plan(tree, prompt, loaded_ctx=4096)

    assert plan.target_per_part > 0
    for part in plan.parts:
        assert part.projected_tokens <= plan.target_per_part
        assert part.fits and part.over_by == 0


async def test_part_body_states_scope_and_defers_the_rest(tree: Path) -> None:
    prompt = "Refactor `src/alpha.py`, `src/beta.py` and `src/gamma.py`."
    _, plan = await _plan(tree, prompt, loaded_ctx=4096)

    first = plan.parts[0]
    assert "Part 1 of" in first.body
    assert "IN SCOPE" in first.body and "OUT OF SCOPE" in first.body
    assert prompt.strip() in first.body  # the task travels with every part, verbatim
    in_scope = {f.display for f in first.files}
    deferred = {f.display for part in plan.parts[1:] for f in part.files} - in_scope
    assert deferred  # the fixture guarantees more than one part
    for display in deferred:
        assert display in first.body  # deferred files are named, never silently dropped


async def test_file_larger_than_a_part_is_sliced_into_contiguous_line_ranges(
    tmp_path: Path,
) -> None:
    (tmp_path / "huge.py").write_text("value = 1\n" * 4000, encoding="utf-8")
    _, plan = await _plan(tmp_path, "Rewrite `huge.py`.", loaded_ctx=8192)

    assert plan.mode is SplitMode.SCOPE
    slices = [f for part in plan.parts for f in part.files]
    assert len(slices) > 1
    assert all(f.is_slice for f in slices)
    assert slices[0].line_start == 1
    for previous, nxt in zip(slices, slices[1:], strict=False):
        assert nxt.line_start == previous.line_end + 1  # contiguous, no lines lost
    assert slices[-1].line_end == 4000
    assert "lines 1-" in plan.parts[0].body
    # The scope block grows a line per slice, so the packer must have measured it, not guessed.
    assert all(part.projected_tokens <= plan.target_per_part for part in plan.parts)


async def test_slices_of_one_file_only_move_forward_through_the_parts(tmp_path: Path) -> None:
    """A part must never hold two disjoint windows of the same file — reading order wins."""
    (tmp_path / "huge.py").write_text("value = 1\n" * 4000, encoding="utf-8")
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")
    _, plan = await _plan(tmp_path, "Rewrite `huge.py` and `small.py`.", loaded_ctx=8192)

    for part in plan.parts:
        spans = [f for f in part.files if f.display.endswith("huge.py")]
        assert len(spans) <= 1
    seen = [f.line_start for part in plan.parts for f in part.files if f.is_slice]
    assert seen == sorted(seen)


async def test_later_parts_reserve_and_count_the_hand_off(tree: Path) -> None:
    prompt = "Refactor `src/alpha.py`, `src/beta.py` and `src/gamma.py`."
    _, plan = await _plan(tree, prompt, loaded_ctx=4096)

    assert plan.handoff_reserve > 0
    # Part 1 has nothing pasted above it; every later part carries the previous hand-off, and
    # its projection must include text the part itself does not contain.
    content = [
        part.projected_tokens - sum(f.tokens for f in part.files) for part in plan.parts
    ]
    assert content[1] - content[0] == plan.handoff_reserve
    assert all(part.projected_tokens <= plan.target_per_part for part in plan.parts)


# --- named directories ----------------------------------------------------------------------


@pytest.fixture
def dir_tree(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    for i in range(6):
        (tmp_path / "src" / f"mod{i}.py").write_text(("y = 2\n" * 400)[:2400], encoding="utf-8")
    (tmp_path / "src" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n binary")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "mod0.cpython-312.pyc").write_bytes(b"\x00cached")
    return tmp_path


async def test_named_directory_is_counted_as_a_ceiling_not_into_the_floor(dir_tree: Path) -> None:
    report, _ = await _plan(dir_tree, "Refactor everything in `src/`.", loaded_ctx=32768)

    assert len(report.directories) == 1
    expanded = report.directories[0]
    assert len(expanded.files) == 6  # the .png is skipped, __pycache__ is pruned
    assert expanded.skipped_non_text == 1
    # The floor is what the prompt guarantees; a directory is not a guarantee that it is read.
    assert report.floor == report.prompt_tokens
    assert report.directory_tokens > 3000
    assert report.ceiling == report.floor + report.directory_tokens


async def test_directory_contents_become_scope_units(dir_tree: Path) -> None:
    _, plan = await _plan(dir_tree, "Refactor everything in `src/`.", loaded_ctx=4096)

    assert plan.mode is SplitMode.SCOPE
    scoped = [f for part in plan.parts for f in part.files]
    assert len(scoped) == 6
    assert all(f.source_dir is not None for f in scoped)
    assert "· from src" in plan.parts[0].body  # the origin travels with the label
    assert all(part.fits for part in plan.parts)


async def test_a_floor_that_fits_still_splits_when_the_directory_does_not(
    dir_tree: Path,
) -> None:
    """The floor is not the number that has to fit — the ceiling is."""
    # Six ~600-token files against a 3,072-token usable budget: the floor is trivial, the
    # directory is not.
    report, plan = await _plan(dir_tree, "Refactor everything in `src/`.", loaded_ctx=4096)

    assert report.verdict.value == "fits"  # the floor is a handful of tokens
    assert report.directory_warning is not None
    assert plan.mode is SplitMode.SCOPE and plan.ok


# --- text mode ----------------------------------------------------------------------------


async def test_text_mode_when_the_prompt_itself_overflows(tmp_path: Path) -> None:
    prompt = "\n\n".join(f"Paragraph {i}: " + "log line content " * 60 for i in range(40))
    _, plan = await _plan(tmp_path, prompt, loaded_ctx=4096)

    assert plan.mode is SplitMode.TEXT
    assert len(plan.parts) > 1
    assert all(part.fits for part in plan.parts)
    assert "Do NOT act on it yet" in plan.parts[0].body
    assert "final segment" in plan.parts[-1].body


async def test_text_segments_preserve_the_content_in_order(tmp_path: Path) -> None:
    prompt = "\n\n".join(f"MARKER{i} " + "filler " * 60 for i in range(40))
    _, plan = await _plan(tmp_path, prompt, loaded_ctx=4096)

    joined = "".join(part.body for part in plan.parts)
    for i in range(40):
        assert f"MARKER{i} " in joined
    positions = [joined.index(f"MARKER{i} ") for i in range(40)]
    assert positions == sorted(positions)  # order is the whole point of a text split


# --- refusals: cases where no honest plan exists --------------------------------------------


async def test_no_plan_when_the_prompt_already_fits(tree: Path) -> None:
    _, plan = await _plan(tree, "Rename a variable.", loaded_ctx=32768)

    assert plan.mode is SplitMode.NONE
    assert plan.already_fits and plan.ok
    assert "nothing to split" in (plan.reason or "")


async def test_no_plan_without_a_budget_to_plan_against(tree: Path) -> None:
    _, plan = await _plan(tree, "Refactor `src/alpha.py` and `src/beta.py`.", loaded_ctx=None)

    assert plan.mode is SplitMode.NONE
    assert plan.indeterminate and not plan.ok
    assert "--target" in (plan.reason or "")


async def test_target_override_plans_without_a_backend(tree: Path) -> None:
    config = _config(tree)
    prompt = "Refactor `src/alpha.py`, `src/beta.py` and `src/gamma.py`."
    report = await run_check(config, prompt, skip_profile=True)
    plan = build_plan(config, prompt, report, target=1600)

    assert plan.mode is SplitMode.SCOPE
    assert plan.target_per_part == 1600
    assert plan.target_label == "requested target"
    # Three ~1,500-token files against a 1,600-token ceiling: no two share a part, and the
    # count is not pinned harder than that — the scaffold's size is an implementation detail.
    assert len(plan.parts) >= 3
    assert {f.display for part in plan.parts for f in part.files} == {
        str(Path("src/alpha.py")), str(Path("src/beta.py")), str(Path("src/gamma.py"))
    }


async def test_overhead_that_swallows_the_window_is_refused_not_faked(tree: Path) -> None:
    _, plan = await _plan(
        tree,
        "Refactor `src/alpha.py`.",
        loaded_ctx=2048,
        client_overhead_tokens=1000,
    )

    assert plan.mode is SplitMode.NONE
    assert not plan.ok
    assert "no split can help" in (plan.reason or "")


async def test_oversized_notebook_is_refused_with_the_real_reason(tmp_path: Path) -> None:
    """A notebook is counted by cell source, so a line range into it names nothing on disk."""
    (tmp_path / "big.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": ["x = 1\n" * 3000]}]}),
        encoding="utf-8",
    )
    _, plan = await _plan(tmp_path, "Rewrite `big.ipynb`.", loaded_ctx=4096)

    assert plan.mode is SplitMode.NONE and not plan.ok
    # Not the "names no countable files" answer — the file is right there, it just cannot be cut.
    assert "no countable files" not in (plan.reason or "")
    assert any("cannot be cut by line range" in note for note in plan.notes)


async def test_scope_mode_refuses_when_no_files_can_be_narrowed(tmp_path: Path) -> None:
    _, plan = await _plan(
        tmp_path, "Do the thing.", loaded_ctx=2048, client_overhead_tokens=1500
    )

    assert plan.mode is SplitMode.NONE
    assert "no countable files" in (plan.reason or "") or "no split can help" in (plan.reason or "")


# --- CLI ------------------------------------------------------------------------------------


def _cd(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    (root / "pharos.toml").write_text(
        f'model = "test-model-not-installed"\ntarget_folder = {str(root)!r}\n'
        f'observations_file = {str(root / "obs.json")!r}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(root)


def test_split_cli_writes_one_file_per_part(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cd(monkeypatch, tree)
    out = tree / "parts"
    code = split_main(
        [
            "Refactor `src/alpha.py`, `src/beta.py` and `src/gamma.py`.",
            "--target",
            "1600",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    written = sorted(p.name for p in out.iterdir())
    assert len(written) >= 3
    assert written == [f"part-{i:02d}.txt" for i in range(1, len(written) + 1)]  # no gaps
    assert "IN SCOPE" in (out / "part-01.txt").read_text(encoding="utf-8")


def test_split_cli_exits_1_when_no_plan_exists(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cd(monkeypatch, tree)
    # A 150-token ceiling cannot even hold the scaffold: no honest plan, exit 1.
    assert split_main(["Refactor `src/alpha.py`.", "--target", "150"]) == 1


def test_check_cli_still_exits_on_the_verdict(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cd(monkeypatch, tree)
    # The probe may or may not find a live backend on the host, so the verdict is not fixed;
    # what must hold is that `check` still runs unchanged and returns a scriptable code.
    assert check_main(["Refactor `src/alpha.py`."]) in {0, 1, 2}


# --- room for what the agent opens on its own (--reserve-reads) -----------------------------


async def test_reserve_reads_takes_the_prediction_off_every_part(tree: Path) -> None:
    """The room comes out of the ceiling once, before anything is packed against it."""
    prompt = "Refactor `src/alpha.py`, `src/beta.py` and `src/gamma.py` consistently."
    config = _config(tree)
    report = await run_check(config, prompt, profile=_profile(tree, loaded_ctx=4096))
    plain = build_plan(config, prompt, report)
    learned = replace(report, reads=ReadEstimate(500, 4, 300, 700, "observed · test"))
    reserved = build_plan(config, prompt, learned, reserve_reads=True)

    assert reserved.target_per_part == plain.target_per_part - 500
    assert reserved.reads_reserved == 500
    assert any("held back from every part" in note for note in reserved.notes)
    assert len(reserved.parts) >= len(plain.parts)


async def test_reserve_reads_with_nothing_learned_says_so_and_packs_as_usual(tree: Path) -> None:
    prompt = "Refactor `src/alpha.py`, `src/beta.py` and `src/gamma.py` consistently."
    config = _config(tree)
    report = await run_check(config, prompt, profile=_profile(tree, loaded_ctx=4096))
    plain = build_plan(config, prompt, report)
    asked = build_plan(config, prompt, report, reserve_reads=True)

    assert asked.target_per_part == plain.target_per_part
    assert asked.reads_reserved == 0
    assert any("cannot size yet" in note for note in asked.notes)


async def test_reserve_reads_without_a_split_is_rejected_not_ignored(tree: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        check_main(["a prompt", "--reserve-reads"])
    assert exit_info.value.code == 2
