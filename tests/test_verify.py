"""Verification: the project's own checks, re-run, with a baseline so only new breakage counts.

The gap this closes was measured, not imagined. A real run wrote all six of its assigned files,
scored 100% coverage, and left ``sorted(total.items(), ...)`` where the variable is ``totals``
— COMPLETE, and a NameError. Ruff calls that F821 in milliseconds, so the check stays
mechanical throughout and no model is asked anything.

Three properties are load-bearing and each is pinned below: a check that never ran is not a
pass, a check that was already failing is not the run's fault, and only the difference between
the two sweeps may fail the run.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from pharos.agent.cli import _headline
from pharos.agent.scorecard import Scorecard, to_dict
from pharos.agent.verify import (
    CheckOutcome,
    Verification,
    _excerpt,
    detect_commands,
    run_command,
    syntax_check,
    syntax_state,
    tokenise,
    verify,
)

BROKEN_PY = "def f(:\n"
GOOD_PY = "def f():\n    return 1\n"


def _py(code: str) -> str:
    """A portable command line running ``code`` in this interpreter."""
    exe = f'"{sys.executable}"' if os.name == "nt" else shlex.quote(sys.executable)
    return f'{exe} -c "{code}"'


# --- syntax, which needs no tooling at all -------------------------------------------------------


def test_syntax_state_reports_each_format_it_can_parse(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text(GOOD_PY, encoding="utf-8")
    (tmp_path / "bad.py").write_text(BROKEN_PY, encoding="utf-8")
    (tmp_path / "ok.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("{not json}", encoding="utf-8")
    (tmp_path / "bad.toml").write_text("key = = 1", encoding="utf-8")
    state = syntax_state(tmp_path, ["ok.py", "bad.py", "ok.json", "bad.json", "bad.toml"])
    assert state["ok.py"] is None
    assert state["ok.json"] is None
    assert state["bad.py"] and state["bad.json"] and state["bad.toml"]


def test_syntax_state_leaves_out_what_it_cannot_answer(tmp_path: Path) -> None:
    """A C# file and a file that does not exist are both absent, not recorded as broken.

    Before a run, a file the plan is about to create does not exist; calling that a failure
    would make every newly created file look like damage the run did.
    """
    (tmp_path / "Program.cs").write_text("class P {", encoding="utf-8")
    state = syntax_state(tmp_path, ["Program.cs", "not-created-yet.py"])
    assert state == {}


def test_syntax_check_passes_when_everything_parses(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(GOOD_PY, encoding="utf-8")
    outcome, unchecked = syntax_check(tmp_path, ["a.py"], baseline={})
    assert outcome.ok and outcome.skipped is None and not unchecked


def test_syntax_check_fails_on_breakage_the_run_caused(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(BROKEN_PY, encoding="utf-8")
    outcome, _ = syntax_check(tmp_path, ["a.py"], baseline={"a.py": None})
    assert outcome.newly_broken
    assert "a.py" in outcome.detail


def test_syntax_check_does_not_blame_the_run_for_a_file_already_broken(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(BROKEN_PY, encoding="utf-8")
    outcome, _ = syntax_check(tmp_path, ["a.py"], baseline={"a.py": "was already broken"})
    assert not outcome.ok
    assert outcome.pre_existing and not outcome.newly_broken


def test_syntax_check_reports_unparseable_formats_rather_than_passing_them(
    tmp_path: Path,
) -> None:
    (tmp_path / "Program.cs").write_text("class P {", encoding="utf-8")
    outcome, unchecked = syntax_check(tmp_path, ["Program.cs"], baseline={})
    assert outcome.skipped is not None
    assert unchecked == ("Program.cs",)


# --- running the project's own tools -------------------------------------------------------------


def test_run_command_passes_on_exit_zero(tmp_path: Path) -> None:
    assert run_command(tmp_path, _py("pass"), timeout=60).ok


def test_run_command_fails_on_nonzero_and_keeps_the_output(tmp_path: Path) -> None:
    outcome = run_command(tmp_path, _py("import sys; print('boom'); sys.exit(1)"), timeout=60)
    assert not outcome.ok
    assert "boom" in outcome.detail


def test_a_missing_tool_is_skipped_never_passed(tmp_path: Path) -> None:
    """Silently passing a check that never executed is worse than not checking."""
    outcome = run_command(tmp_path, "definitely-not-a-real-tool --check", timeout=60)
    assert outcome.skipped is not None
    assert not outcome.newly_broken


def test_a_timeout_is_skipped_never_passed(tmp_path: Path) -> None:
    """A run must not be able to turn a slow suite green by outwaiting it."""
    outcome = run_command(tmp_path, _py("import time; time.sleep(30)"), timeout=0.5)
    assert outcome.skipped is not None and "timed out" in outcome.skipped
    assert not outcome.newly_broken


def test_an_unparseable_command_is_skipped(tmp_path: Path) -> None:
    assert run_command(tmp_path, 'ruff "unclosed', timeout=5).skipped is not None


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ruff check .", ["ruff", "check", "."]),
        ('lint --msg "two words"', ["lint", "--msg", "two words"]),
    ],
)
def test_tokenise_handles_quoting(command: str, expected: list[str]) -> None:
    assert tokenise(command) == expected


@pytest.mark.skipif(os.name != "nt", reason="backslash paths are a Windows concern")
def test_tokenise_keeps_windows_path_separators() -> None:
    """POSIX-mode shlex eats these, turning a real tool into "not on PATH"."""
    assert tokenise(r"C:\tools\lint.exe --fix") == [r"C:\tools\lint.exe", "--fix"]


# --- detection invents nothing -------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not on PATH here")
def test_detects_a_tool_the_project_configures(tmp_path: Path) -> None:
    """Detection is configuration AND installation, so this test needs the tool.

    Without the guard it asserts that a binary is on the PATH of whoever runs the suite,
    which is not a claim about Pharos: it failed on this machine purely for being invoked
    from a shell where the virtualenv had not been activated.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n", encoding="utf-8")
    assert "ruff check ." in detect_commands(tmp_path)


def test_detects_nothing_for_a_project_that_configures_nothing(tmp_path: Path) -> None:
    """Pharos never decides what your build is; an unconfigured project gets verify_commands."""
    (tmp_path / "main.py").write_text(GOOD_PY, encoding="utf-8")
    assert detect_commands(tmp_path) == []


# --- the baseline, which is what lets this gate an exit code --------------------------------------


def test_a_check_already_failing_is_not_charged_to_the_run(tmp_path: Path) -> None:
    """Without this, verification would fail every run on any repository with a red test."""
    command = _py("import sys; sys.exit(1)")
    (tmp_path / "a.py").write_text(GOOD_PY, encoding="utf-8")
    result = verify(
        tmp_path,
        written=["a.py"],
        commands=[command],
        command_baseline={command: CheckOutcome(name=command, ok=False)},
        syntax_baseline={},
        timeout=60,
    )
    assert result.ok
    assert [c.name for c in result.already_failing] == [command]
    assert not result.newly_broken


def test_a_check_the_run_broke_fails_the_verification(tmp_path: Path) -> None:
    command = _py("import sys; sys.exit(1)")
    (tmp_path / "a.py").write_text(GOOD_PY, encoding="utf-8")
    result = verify(
        tmp_path,
        written=["a.py"],
        commands=[command],
        command_baseline={command: CheckOutcome(name=command, ok=True)},
        syntax_baseline={},
        timeout=60,
    )
    assert not result.ok
    assert [c.name for c in result.newly_broken] == [command]


def test_a_failure_with_no_usable_baseline_is_not_pinned_on_the_run(tmp_path: Path) -> None:
    """A baseline that was itself skipped reports ok=True, because a check that did not run is
    not a failure. Reducing it to that bool would let a skipped baseline pose as a passing one
    and the run would be blamed for a regression nobody ever measured."""
    command = _py("import sys; sys.exit(1)")
    (tmp_path / "a.py").write_text(GOOD_PY, encoding="utf-8")
    result = verify(
        tmp_path,
        written=["a.py"],
        commands=[command],
        command_baseline={command: CheckOutcome(name=command, ok=True, skipped="timed out")},
        syntax_baseline={},
        timeout=60,
    )
    assert not result.newly_broken
    assert [c.name for c in result.unattributable] == [command]
    assert result.ok and not result.compared


def test_commands_are_skipped_when_the_run_wrote_nothing(tmp_path: Path) -> None:
    """Nothing written means nothing can have broken; re-running the suite would only
    re-measure the baseline at the cost of running it twice."""
    command = _py("import sys; sys.exit(1)")
    result = verify(
        tmp_path,
        written=[],
        commands=[command],
        command_baseline={command: CheckOutcome(name=command, ok=True)},
        syntax_baseline={},
        timeout=60,
    )
    assert result.ok
    assert [c.skipped for c in result.checks if c.name == command] == ["the run wrote no files"]


def test_a_red_repository_getting_redder_is_reported_not_hidden(tmp_path: Path) -> None:
    """A pass/fail baseline cannot tell "unchanged" from "made worse" once a check is already
    failing, so a run that adds an error to an already-red lint gate would otherwise be
    invisible. It is not charged to the run -- attribution needs a measurement nobody has --
    but the output moving is said out loud."""
    command = _py("import sys; print('two errors'); sys.exit(1)")
    (tmp_path / "a.py").write_text(GOOD_PY, encoding="utf-8")
    result = verify(
        tmp_path,
        written=["a.py"],
        commands=[command],
        command_baseline={
            command: CheckOutcome(name=command, ok=False, detail="one error")
        },
        syntax_baseline={},
        timeout=60,
    )
    assert result.ok  # still not blamed on the run
    assert [c.name for c in result.worsened] == [command]


def test_an_unchanged_failure_is_not_reported_as_worsened(tmp_path: Path) -> None:
    command = _py("import sys; print('same'); sys.exit(1)")
    (tmp_path / "a.py").write_text(GOOD_PY, encoding="utf-8")
    result = verify(
        tmp_path,
        written=["a.py"],
        commands=[command],
        command_baseline={command: CheckOutcome(name=command, ok=False, detail="same")},
        syntax_baseline={},
        timeout=60,
    )
    assert result.ok and not result.worsened


def test_ran_separates_nothing_broken_from_nothing_checked(tmp_path: Path) -> None:
    """A CI step has to be able to tell those apart; ``ok`` alone cannot."""
    nothing = verify(
        tmp_path, written=[], commands=[], command_baseline={}, syntax_baseline={}, timeout=5
    )
    assert nothing.ok and not nothing.ran


# --- and it decides the verdict -------------------------------------------------------------------


def _card(verification: Verification | None) -> Scorecard:
    return Scorecard(
        parts=1,
        parts_that_wrote=1,
        scoped_files=1,
        written_files=1,
        verification=verification,
    )


def test_full_coverage_is_not_complete_when_the_run_broke_the_build() -> None:
    """The exact shape of the run this exists for: every file written, project broken."""
    broken = Verification(checks=(CheckOutcome(name="ruff check .", ok=False, detail="F821"),))
    card = _card(broken)
    assert card.coverage == 1.0
    assert not card.complete


def test_full_coverage_stays_complete_when_the_repository_arrived_red() -> None:
    already = Verification(
        checks=(CheckOutcome(name="pytest -q", ok=False, pre_existing=True),)
    )
    assert _card(already).complete


def test_verification_is_reported_in_the_json_payload() -> None:
    broken = Verification(checks=(CheckOutcome(name="ruff check .", ok=False, detail="F821"),))
    payload = to_dict(_card(broken))["verification"]
    assert isinstance(payload, dict)
    assert payload["ok"] is False and payload["ran"] is True
    assert payload["newly_broken"] == [{"name": "ruff check .", "detail": "F821"}]
    assert to_dict(_card(None))["verification"] is None


# --- reporting the failure usefully --------------------------------------------------------------


def test_short_output_is_kept_whole() -> None:
    assert _excerpt("one" + chr(10) + "two") == "one" + chr(10) + "two"


def test_long_output_is_excerpted_from_both_ends() -> None:
    """Which end holds the answer is tool-specific: ruff leads with the rule code, pytest
    trails with the summary. The first version kept only the tail and scrolled ruff's
    diagnostic off the top."""
    excerpt = _excerpt(chr(10).join(f"line {n}" for n in range(1, 31)))
    assert "line 1" in excerpt
    assert "line 30" in excerpt
    assert "24 more line(s)" in excerpt


def test_verdict_says_broken_not_incomplete_when_every_file_was_written() -> None:
    """"INCOMPLETE - 0 of 6 files were never changed" beside a 100% bar is a contradiction,
    and it is what the first run of this printed. The run is not short of writes; it is broken
    for what it wrote."""
    broken = Verification(checks=(CheckOutcome(name="ruff check .", ok=False, detail="F821"),))
    verdict, _, note = _headline(_card(broken))
    assert verdict == "BROKEN"
    assert "ruff check ." in note
    assert "never changed" not in note


def test_verdict_stays_complete_when_verification_passed() -> None:
    clean = Verification(checks=(CheckOutcome(name="ruff check .", ok=True),))
    assert _headline(_card(clean))[0] == "COMPLETE"


def test_an_excerpt_never_ends_on_an_empty_diagram_opener() -> None:
    """ruff draws the offending source under a lone `|`. Cutting the head at four lines landed
    on that opener, so a live run printed a diagnostic, its location, and then a bare `|` with
    nothing under it -- scaffolding for a picture the cut had already removed."""
    output = chr(10).join(
        [
            "F821 Undefined name `Customer`",
            "--> src/orders.py:5:56",
            "  |",
            "  |",
            "5 | def f(c: Customer) -> None:",
            "  |          ^^^^^^^^",
            "help: add an import",
            "Found 1 error.",
        ]
    )
    excerpt = _excerpt(output)

    assert excerpt.splitlines()[1].strip() != "|"
    assert "|" + chr(10) + "..." not in excerpt
    # The diagnostic and its location are what the head is for, and both survive.
    assert "F821 Undefined name `Customer`" in excerpt
    assert "--> src/orders.py:5:56" in excerpt


def test_a_short_output_keeps_every_useful_line() -> None:
    """Trimming must only remove scaffolding, never a line carrying a message."""
    output = chr(10).join(["E501 line too long", "--> a.py:1:101", "help: shorten it"])
    assert _excerpt(output) == output
