"""Change auditing: what the disk says, against what the run says about itself.

Every assertion here is about a DISAGREEMENT. A run whose ledger and filesystem agree is the
uninteresting case and is covered once; the rest of the file is the four ways they can come
apart, because those are the ones nothing before v1.0 could see.
"""

from __future__ import annotations

import io
import itertools
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from rich.console import Console
from rich.table import Table

from pharos.accountant import Accountant
from pharos.agent import runner
from pharos.agent.audit import (
    PartAudit,
    TreeDiff,
    TreeIndex,
    audit_part,
    diff_trees,
    index_tree,
    own_paths,
)
from pharos.agent.cli import _add_audit_row
from pharos.agent.runner import RunOutcome
from pharos.agent.scorecard import Scorecard
from pharos.agent.session import PartResult
from pharos.config import PharosConfig
from pharos.preflight.check import Verdict, run_check
from pharos.profiler.types import BackendInfo, EnvironmentProfile, GpuInfo

_TICK = itertools.count(1)


def _touch(path: Path, text: str) -> None:
    """Write, and force the stamp forward.

    Not a convenience. Windows updates its system clock about every 15 ms, so two writes in
    one test land on the SAME st_mtime_ns — and a rewrite that keeps the file's length would
    then be invisible to the audit for reasons that have nothing to do with the audit. Every
    write here gets its own second, so what the tests measure is the diffing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    stamp = time.time_ns() + next(_TICK) * 1_000_000_000
    os.utime(path, ns=(stamp, stamp))


def _tree(root: Path) -> Path:
    _touch(root / "src" / "alpha.py", "alpha = 1\n")
    _touch(root / "src" / "beta.py", "beta = 2\n")
    return root


# --- indexing ---------------------------------------------------------------------------


def test_the_index_sees_every_file_under_the_root(tmp_path: Path) -> None:
    _tree(tmp_path)
    index = index_tree(tmp_path)
    assert set(index.entries) == {"src/alpha.py", "src/beta.py"}
    assert not index.truncated


def test_the_index_prunes_what_every_other_walk_prunes(tmp_path: Path) -> None:
    """A .git or a __pycache__ churns constantly; reporting it would bury the real finding."""
    _tree(tmp_path)
    _touch(tmp_path / ".git" / "index", "x")
    _touch(tmp_path / "__pycache__" / "alpha.pyc", "x")
    _touch(tmp_path / ".venv" / "lib" / "thing.py", "x")
    assert set(index_tree(tmp_path).entries) == {"src/alpha.py", "src/beta.py"}


def test_a_tree_over_the_cap_says_so_instead_of_stopping_quietly(tmp_path: Path) -> None:
    for i in range(12):
        _touch(tmp_path / f"file{i}.txt", "x")
    index = index_tree(tmp_path, cap=5)
    assert index.truncated and len(index.entries) == 5


def test_an_unreadable_entry_is_skipped_rather_than_raised_on(tmp_path: Path) -> None:
    """A file that vanishes between the walk and the stat is a race, not a finding."""
    _tree(tmp_path)
    assert index_tree(tmp_path / "nowhere").entries == {}


# --- diffing ----------------------------------------------------------------------------


def test_a_diff_separates_created_modified_and_deleted(tmp_path: Path) -> None:
    _tree(tmp_path)
    before = index_tree(tmp_path)
    _touch(tmp_path / "src" / "alpha.py", "alpha = 99\n")
    _touch(tmp_path / "src" / "gamma.py", "gamma = 3\n")
    (tmp_path / "src" / "beta.py").unlink()

    changed = diff_trees(before, index_tree(tmp_path))
    assert changed.created == ("src/gamma.py",)
    assert changed.modified == ("src/alpha.py",)
    assert changed.deleted == ("src/beta.py",)
    assert changed.paths == ("src/alpha.py", "src/beta.py", "src/gamma.py")


def test_a_rewrite_of_identical_length_is_still_a_change(tmp_path: Path) -> None:
    """Same size, different content — caught by the stamp, which is why size alone is not it."""
    _tree(tmp_path)
    before = index_tree(tmp_path)
    _touch(tmp_path / "src" / "alpha.py", "alpha = 9\n")
    assert diff_trees(before, index_tree(tmp_path)).modified == ("src/alpha.py",)


def test_an_untouched_tree_diffs_to_nothing(tmp_path: Path) -> None:
    _tree(tmp_path)
    index = index_tree(tmp_path)
    assert not diff_trees(index, index_tree(tmp_path))


# --- reconciliation -----------------------------------------------------------------------


def _diff(*modified: str) -> TreeDiff:
    """A diff in which every named path was modified, without touching a real filesystem."""
    return diff_trees(
        TreeIndex(entries=dict.fromkeys(modified, (1, 1))),
        TreeIndex(entries=dict.fromkeys(modified, (2, 2))),
    )


def test_a_run_that_agrees_with_the_disk_is_clean() -> None:
    audit = audit_part(
        "part 1",
        changed=_diff("src/alpha.py"),
        claimed=["src/alpha.py"],
        scoped=["src/alpha.py"],
    )
    assert audit.clean
    assert not (audit.unattributed or audit.absent or audit.out_of_scope)


def test_a_change_no_tool_claimed_is_unattributed() -> None:
    """An editor left open on the workspace, a git hook, a generated file. Not necessarily
    wrong — necessarily worth knowing, because coverage is computed from claims."""
    audit = audit_part(
        "part 1",
        changed=_diff("src/alpha.py", "src/generated.py"),
        claimed=["src/alpha.py"],
        scoped=None,
    )
    assert audit.unattributed == ("src/generated.py",)
    assert not audit.clean


def test_a_write_that_never_landed_is_reported_as_absent() -> None:
    """The one finding that says the RUN was wrong about itself, not the tree."""
    audit = audit_part(
        "part 1",
        changed=_diff("src/alpha.py"),
        claimed=["src/alpha.py", "src/beta.py"],
        scoped=None,
    )
    assert audit.absent == ("src/beta.py",)


def test_a_change_outside_the_parts_scope_is_named() -> None:
    audit = audit_part(
        "part 2",
        changed=_diff("src/alpha.py", "src/deferred.py"),
        claimed=["src/alpha.py"],
        scoped=["src/alpha.py"],
    )
    assert audit.out_of_scope == ("src/deferred.py",)


def test_an_unrestricted_part_cannot_be_out_of_scope() -> None:
    """There is no list to be outside of, and inventing one would be the same mistake the
    coverage figure refuses to make."""
    audit = audit_part(
        "part 1",
        changed=_diff("anything.py"),
        claimed=[],
        scoped=None,
    )
    assert audit.out_of_scope == ()


def test_what_the_projects_own_checks_rewrote_is_kept_apart() -> None:
    """A formatter in a test command edits the user's tree during a Pharos run. It is not the
    model's doing and must not be scored as one — but it is not nothing, either."""
    audit = audit_part(
        "part 1",
        changed=_diff("src/alpha.py"),
        claimed=["src/alpha.py"],
        scoped=["src/alpha.py"],
        by_checks=_diff("src/alpha.py", "tests/__snapshots__/one.txt"),
    )
    assert audit.by_checks.paths == ("src/alpha.py", "tests/__snapshots__/one.txt")
    assert not audit.clean
    assert not audit.unattributed  # the checks are a separate question from the part's tools


def test_separators_do_not_decide_whether_a_claim_matches() -> None:
    """The claim comes from the dispatcher and the change from os.walk; on Windows those two
    disagree about the separator, and a run would have reported every write twice."""
    audit = audit_part(
        "part 1",
        changed=_diff("src/alpha.py"),
        claimed=["src" + chr(92) + "alpha.py"],
        scoped=["src" + chr(92) + "alpha.py"],
    )
    assert audit.clean


# --- how it reads --------------------------------------------------------------------------


def _rendered(card: Scorecard) -> str:
    console = Console(file=io.StringIO(), width=110, force_terminal=False)
    table = Table.grid(padding=(0, 2))
    table.add_column()
    table.add_column()
    _add_audit_row(table, card)
    console.print(table)
    return console.file.getvalue()  # type: ignore[attr-defined]


def _card(*audits: PartAudit) -> Scorecard:
    return Scorecard(
        parts=len(audits), parts_that_wrote=0, scoped_files=0, written_files=0,
        audits=list(audits),
    )


def test_an_audit_that_agrees_prints_one_quiet_line() -> None:
    text = _rendered(
        _card(
            audit_part(
                "part 1",
                changed=_diff("src/alpha.py"),
                claimed=["src/alpha.py"],
                scoped=["src/alpha.py"],
            )
        )
    )
    assert "claimed by the part that made it" in text


def test_a_run_with_no_audit_prints_no_row_at_all() -> None:
    assert _rendered(_card()).strip() == ""


def test_every_finding_reaches_the_report() -> None:
    text = _rendered(
        _card(
            audit_part(
                "part 1",
                changed=_diff("src/alpha.py", "src/stray.py"),
                claimed=["src/alpha.py", "src/ghost.py"],
                scoped=["src/alpha.py"],
                by_checks=_diff("src/formatted.py"),
            )
        )
    )
    assert "src/ghost.py" in text  # claimed, never landed
    assert "src/stray.py" in text  # landed, nobody claimed it
    assert "src/formatted.py" in text  # your own checks wrote it
    assert "outside the part" in text


# --- inside a real run ------------------------------------------------------------------------
#
# Everything above tests the audit in isolation. This drives `run_task` itself, because the
# glue is where an audit gets attached to the wrong part or to no part at all — and a report
# that silently audits nothing looks exactly like a report that found nothing.


class _StubSession:
    """Stands in for AgentSession: writes what it was told to, and one file it was not."""

    written: list[str] = []
    stray: str | None = None

    def __init__(self, **kwargs: object) -> None:
        self._root = Path(str(kwargs["toolbox"].workspace.root))  # type: ignore[union-attr]

    async def run(self, part_body: str) -> PartResult:
        for name in self.written:
            path = self._root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("changed = True" + chr(10), encoding="utf-8")
        if self.stray is not None:
            (self._root / self.stray).write_text("nobody = 1" + chr(10), encoding="utf-8")
        return PartResult(
            text="done",
            steps=1,
            files_written=list(self.written),
            peak_tokens=10,
            reported_tokens=None,
            stopped_early=False,
            scoped=list(self.written),
        )


async def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, stray: str | None, audit: bool = True
) -> RunOutcome:
    (tmp_path / "src").mkdir(exist_ok=True)
    for name in ("alpha.py", "extra.py"):
        _touch(tmp_path / "src" / name, "x = 1" + chr(10))

    config = PharosConfig(  # type: ignore[call-arg]
        model="test-model",
        target_folder=str(tmp_path),
        observations_file=str(tmp_path / "obs.json"),
        template_memory_file=str(tmp_path / "templates.json"),
        verify=False,
    )
    prompt = "Refactor `src/alpha.py`"
    report = await run_check(config, prompt, skip_profile=True, target=100_000)
    report = replace(report, profile=_profile(config))

    async def _check(*args: object, **kwargs: object) -> object:
        return report

    async def _window(*args: object, **kwargs: object) -> object:
        return report

    async def _reachable(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(runner, "run_check", _check)
    monkeypatch.setattr(runner, "_ensure_window", _window)
    monkeypatch.setattr(runner, "_reachable", _reachable)
    monkeypatch.setattr(_StubSession, "written", ["src/alpha.py"])
    monkeypatch.setattr(_StubSession, "stray", stray)
    monkeypatch.setattr(runner, "AgentSession", _StubSession)
    return await runner.run_task(config, prompt, use_git=False, audit=audit)


def _profile(config: PharosConfig) -> EnvironmentProfile:
    gpu = GpuInfo(available=False, name=None, total_mib=None, used_mib=None, free_mib=None)
    backend = BackendInfo(
        reachable=True,
        base_url=config.backend_url,
        model="test-model",
        advertised_max_ctx=100_000,
        loaded_ctx=100_000,
    )
    return EnvironmentProfile(
        gpu=gpu,
        backend=backend,
        budget=Accountant(config).report(loaded_ctx=100_000, gpu=gpu),
        ctx_mismatch=False,
        ctx_mismatch_ratio=None,
    )


async def test_a_run_audits_every_part_it_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = await _run(tmp_path, monkeypatch, stray=None)
    assert outcome.audits, "the run executed a part and audited nothing"
    assert outcome.audits[0].part == "part 1"
    assert "src/alpha.py" in outcome.audits[0].changed.paths
    assert outcome.audits[0].clean


async def test_a_run_names_the_file_nothing_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: a write that happened and that no tool of the run reported."""
    outcome = await _run(tmp_path, monkeypatch, stray="src/extra.py")
    assert outcome.audits[0].unattributed == ("src/extra.py",)
    assert not outcome.audits[0].clean


async def test_no_audit_leaves_the_report_empty_rather_than_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = await _run(tmp_path, monkeypatch, stray="src/extra.py", audit=False)
    assert outcome.audits == []


# --- what Pharos writes itself ----------------------------------------------------------------
#
# Every live run of v1.0 reported `pharos.log` and `pharos_observations.json` as unattributed
# AND out of scope AND written by the checks: three findings, all false, sitting beside the
# real ones. They are configured paths, so excluding them is exact rather than a guess at what
# a Pharos file looks like.


def _own_config(root: Path, **overrides: object) -> PharosConfig:
    base: dict[str, object] = {
        "target_folder": str(root),
        "log_file": str(root / "pharos.log"),
        "observations_file": str(root / "pharos_observations.json"),
        "template_memory_file": str(root / "pharos_templates.json"),
    }
    base.update(overrides)
    return PharosConfig(**base)  # type: ignore[arg-type]


def test_pharos_own_files_are_named_from_the_config_not_guessed(tmp_path: Path) -> None:
    keys = own_paths(_own_config(tmp_path), tmp_path)
    assert keys == {"pharos.log", "pharos_observations.json", "pharos_templates.json"}


def test_a_store_configured_outside_the_workspace_is_simply_absent(tmp_path: Path) -> None:
    """Nothing to exclude: the walk was never going to see it."""
    elsewhere = tmp_path.parent / "somewhere-else.json"
    keys = own_paths(_own_config(tmp_path, observations_file=str(elsewhere)), tmp_path)
    assert "pharos.log" in keys
    assert not any("somewhere-else" in key for key in keys)


def test_the_index_skips_what_pharos_writes(tmp_path: Path) -> None:
    _tree(tmp_path)
    _touch(tmp_path / "pharos.log", "chatter")
    _touch(tmp_path / "pharos_observations.json", "[]")
    ignore = own_paths(_own_config(tmp_path), tmp_path)

    assert set(index_tree(tmp_path).entries) > {"src/alpha.py", "src/beta.py"}
    assert set(index_tree(tmp_path, ignore=ignore).entries) == {"src/alpha.py", "src/beta.py"}


def test_the_run_no_longer_reports_its_own_log_as_a_change_nobody_claimed(
    tmp_path: Path,
) -> None:
    """The defect, end to end: the log moves during the part, and is not a finding."""
    _tree(tmp_path)
    ignore = own_paths(_own_config(tmp_path), tmp_path)
    before = index_tree(tmp_path, ignore=ignore)
    _touch(tmp_path / "pharos.log", "the run said things")
    _touch(tmp_path / "src" / "alpha.py", "alpha = 2\n")

    audit = audit_part(
        "part 1",
        changed=diff_trees(before, index_tree(tmp_path, ignore=ignore)),
        claimed=["src/alpha.py"],
        scoped=["src/alpha.py"],
    )
    assert audit.clean
    assert audit.changed.paths == ("src/alpha.py",)


def test_the_undo_directory_is_not_walked_at_all(tmp_path: Path) -> None:
    """It holds a copy of every original the run is about to overwrite — the largest false
    finding available, and indexing it would double the audit's cost to produce it."""
    _tree(tmp_path)
    undo = tmp_path / ".pharos" / "undo-2026-01-01"
    for name in ("alpha.py", "beta.py"):
        _touch(undo / "src" / name, "the original")
    ignore = own_paths(_own_config(tmp_path), tmp_path, tmp_path / ".pharos")

    indexed = index_tree(tmp_path, ignore=ignore)
    assert set(indexed.entries) == {"src/alpha.py", "src/beta.py"}


# --- a run that cannot get a verdict still reports what it found ----------------------------------


async def test_an_indeterminate_run_still_carries_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_task` fills in `outcome.report` three times on the way down, and the first two are
    not redundant: an indeterminate verdict returns between them.

    Without those assignments the caller gets an outcome with an error and no report at all,
    and `pharos run` prints a bare failure where it should print the pre-flight that produced
    it. Nothing else in the suite covered the early return, and it is the kind of line that
    reads like a leftover -- so this pins it.
    """
    config = PharosConfig(  # type: ignore[call-arg]
        model="test-model",
        target_folder=str(tmp_path),
        observations_file=str(tmp_path / "obs.json"),
        template_memory_file=str(tmp_path / "templates.json"),
        verify=False,
    )
    # No profile and no target: there is no window to judge the floor against.
    report = await run_check(config, "Do a thing", skip_profile=True)
    assert report.verdict is Verdict.INDETERMINATE

    async def _check(*args: object, **kwargs: object) -> object:
        return report

    monkeypatch.setattr(runner, "run_check", _check)
    monkeypatch.setattr(runner, "_load_and_recheck", _check)
    outcome = await runner.run_task(config, "Do a thing", use_git=False)

    assert outcome.error is not None and "no verdict" in outcome.error
    assert outcome.report is report
    assert outcome.counts_exact == report.counts_exact
