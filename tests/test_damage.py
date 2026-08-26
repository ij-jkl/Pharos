"""Which part broke it: per-part attribution, repairs, and the halt.

The end-of-run check already answers "is the project broken". These are about the other half
— who did it — and about the two ways that answer can be wrong: charging a part for damage it
inherited, and hiding damage because a later part happened to clean it up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pharos.agent.runner import _cheap_enough
from pharos.agent.scorecard import score
from pharos.agent.session import PartResult
from pharos.agent.verify import CheckOutcome, Damage, DamageWatch, syntax_state


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("b = 2\n", encoding="utf-8")
    return tmp_path


def _break(root: Path, name: str) -> None:
    (root / name).write_text("def broken(\n", encoding="utf-8")


def _mend(root: Path, name: str) -> None:
    (root / name).write_text("mended = 1\n", encoding="utf-8")


def _part(**kwargs: object) -> PartResult:
    base: dict[str, object] = {
        "text": "",
        "steps": 1,
        "files_written": [],
        "peak_tokens": 10,
        "reported_tokens": None,
        "stopped_early": False,
    }
    base.update(kwargs)
    return PartResult(**base)  # type: ignore[arg-type]


# --- attribution --------------------------------------------------------------------------


def test_a_part_that_breaks_a_file_is_named(root: Path) -> None:
    watch = DamageWatch(root, syntax_state(root, ["src/a.py", "src/b.py"]))
    _break(root, "src/a.py")
    found = watch.after_part("part 1", ["src/a.py"])
    assert [(d.label, d.path) for d in found] == [("part 1", "src/a.py")]
    assert "was never closed" in found[0].error or "(" in found[0].error


def test_a_part_that_breaks_nothing_is_named_for_nothing(root: Path) -> None:
    watch = DamageWatch(root, syntax_state(root, ["src/a.py"]))
    (root / "src" / "a.py").write_text("a = 2\n", encoding="utf-8")
    assert watch.after_part("part 1", ["src/a.py"]) == []
    assert watch.damage == []


def test_a_file_that_arrived_broken_is_not_charged_to_the_part(root: Path) -> None:
    """The baseline rule that lets verification gate an exit code without being a nuisance."""
    _break(root, "src/a.py")
    watch = DamageWatch(root, syntax_state(root, ["src/a.py"]))
    (root / "src" / "a.py").write_text("def still_broken(\n", encoding="utf-8")
    assert watch.after_part("part 1", ["src/a.py"]) == []


def test_a_new_file_created_broken_is_charged(root: Path) -> None:
    """It did not exist to be fine before, and the part is still the one that broke it."""
    watch = DamageWatch(root, {})
    _break(root, "src/new.py")
    found = watch.after_part("part 1", ["src/new.py"])
    assert [d.path for d in found] == ["src/new.py"]


def test_the_second_part_is_not_charged_for_the_first_part_s_damage(root: Path) -> None:
    """Attribution is against the state immediately before the part, not the run baseline."""
    watch = DamageWatch(root, syntax_state(root, ["src/a.py"]))
    _break(root, "src/a.py")
    watch.after_part("part 1", ["src/a.py"])
    assert watch.after_part("part 2", ["src/a.py"]) == []
    assert [d.label for d in watch.damage] == ["part 1"]


def test_a_format_with_no_parser_is_neither_broken_nor_fine(root: Path) -> None:
    (root / "notes.md").write_text("# not parsed\n", encoding="utf-8")
    watch = DamageWatch(root, {})
    assert watch.after_part("part 1", ["notes.md"]) == []


def test_json_and_toml_are_parsed_too(root: Path) -> None:
    (root / "data.json").write_text('{"a": 1}\n', encoding="utf-8")
    watch = DamageWatch(root, syntax_state(root, ["data.json"]))
    (root / "data.json").write_text("{not json\n", encoding="utf-8")
    assert [d.path for d in watch.after_part("part 1", ["data.json"])] == ["data.json"]


# --- repairs ------------------------------------------------------------------------------


def test_a_later_part_that_fixes_it_is_credited(root: Path) -> None:
    watch = DamageWatch(root, syntax_state(root, ["src/a.py"]))
    _break(root, "src/a.py")
    watch.after_part("part 1", ["src/a.py"])
    _mend(root, "src/a.py")
    watch.after_part("part 3", ["src/a.py"])

    assert len(watch.damage) == 1
    assert watch.damage[0].label == "part 1"
    assert watch.damage[0].repaired_by == "part 3"
    assert watch.outstanding == []


def test_damage_that_nobody_came_back_for_stays_outstanding(root: Path) -> None:
    watch = DamageWatch(root, syntax_state(root, ["src/a.py"]))
    _break(root, "src/a.py")
    watch.after_part("part 1", ["src/a.py"])
    assert [d.path for d in watch.outstanding] == ["src/a.py"]


def test_broken_repaired_and_broken_again_is_two_entries(root: Path) -> None:
    watch = DamageWatch(root, syntax_state(root, ["src/a.py"]))
    _break(root, "src/a.py")
    watch.after_part("part 1", ["src/a.py"])
    _mend(root, "src/a.py")
    watch.after_part("part 2", ["src/a.py"])
    _break(root, "src/a.py")
    watch.after_part("part 3", ["src/a.py"])

    assert [(d.label, d.repaired_by) for d in watch.damage] == [
        ("part 1", "part 2"),
        ("part 3", None),
    ]


# --- the scorecard --------------------------------------------------------------------------


def test_the_scorecard_names_the_parts_that_broke_something() -> None:
    damage = [
        Damage("part 1", "src/a.py", "boom"),
        Damage("part 2", "src/b.py", "boom", repaired_by="repair 1"),
        Damage("part 1", "src/c.py", "boom"),
    ]
    card = score([_part()], handoff_reserve=500, damage=damage)
    assert card.broke_the_build == ["part 1"]  # once, not twice
    assert [d.path for d in card.transient_damage] == ["src/b.py"]


def test_a_run_with_no_damage_names_nobody() -> None:
    card = score([_part()], handoff_reserve=500)
    assert card.broke_the_build == []
    assert card.damage == []


def test_repaired_damage_does_not_count_against_the_run() -> None:
    damage = [Damage("part 1", "src/a.py", "boom", repaired_by="part 2")]
    card = score([_part()], handoff_reserve=500, damage=damage)
    assert card.broke_the_build == []
    assert card.transient_damage


def test_the_json_carries_the_attribution() -> None:
    from pharos.agent.scorecard import to_dict

    damage = [Damage("part 2", "src/a.py", "unexpected indent", repaired_by=None)]
    payload = to_dict(score([_part()], handoff_reserve=500, damage=damage,
                            stopped_on_break="part 2"))
    assert payload["broke_the_build"] == ["part 2"]
    assert payload["stopped_on_break"] == "part 2"
    assert payload["damage"] == [
        {
            "part": "part 2",
            "check": "syntax",
            "file": "src/a.py",
            "error": "unexpected indent",
            "repaired_by": None,
        }
    ]


# --- the verdict ------------------------------------------------------------------------------


def test_a_halted_run_reads_as_stopped_not_failed() -> None:
    """It did not fail — nothing errored — and it is not incomplete by the model's doing."""
    from pharos.agent.cli import _headline

    card = score(
        [_part(files_written=["src/a.py"], scoped=["src/a.py", "src/b.py"])],
        handoff_reserve=500,
        damage=[Damage("part 1", "src/a.py", "boom")],
        stopped_on_break="part 1",
    )
    word, _style, why = _headline(card)
    assert word == "STOPPED"
    assert "part 1" in why


def test_an_unstopped_broken_run_still_reads_as_incomplete() -> None:
    from pharos.agent.cli import _headline

    card = score(
        [_part(files_written=["src/a.py"], scoped=["src/a.py", "src/b.py"])],
        handoff_reserve=500,
        damage=[Damage("part 1", "src/a.py", "boom")],
    )
    assert _headline(card)[0] == "INCOMPLETE"


# --- coverage on a run that did not reach every part -------------------------------------------


def test_a_halted_run_measures_coverage_against_the_plan_not_the_parts_that_ran() -> None:
    """It read 100% before this: the four files nobody attempted were in no part's scope, so
    the denominator shrank with the numerator and the bar stayed full beside STOPPED."""
    card = score(
        [_part(files_written=["src/a.py", "src/b.py"], scoped=["src/a.py", "src/b.py"])],
        handoff_reserve=500,
        planned_files=["src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py", "src/f.py"],
        stopped_on_break="part 1",
    )
    assert card.scoped_files == 6
    assert card.written_files == 2
    assert card.coverage is not None and round(card.coverage, 2) == 0.33
    assert card.untouched == ["src/c.py", "src/d.py", "src/e.py", "src/f.py"]


def test_a_run_that_reached_every_part_is_unaffected() -> None:
    card = score(
        [_part(files_written=["src/a.py"], scoped=["src/a.py"])],
        handoff_reserve=500,
        planned_files=["src/a.py"],
    )
    assert card.coverage == 1.0


def test_without_a_plan_the_parts_are_still_the_denominator() -> None:
    """`--no-split` runs pass no plan, and inventing one would be worse than admitting it."""
    card = score([_part(files_written=["src/a.py"])], handoff_reserve=500)
    assert card.coverage is None


def test_a_part_that_errored_after_writing_is_still_charged(root: Path) -> None:
    """The write landed. Attributing it to nobody because the part died afterwards would lose
    the one fact worth having about the failure."""
    watch = DamageWatch(root, syntax_state(root, ["src/a.py"]))
    _break(root, "src/a.py")
    found = watch.after_part("part 2", ["src/a.py"])
    assert [d.label for d in found] == ["part 2"]


# --- the project's own checks, per part (v0.8) -------------------------------------------------


def _ok(name: str, *, seconds: float = 0.1) -> CheckOutcome:
    return CheckOutcome(name=name, ok=True, duration=seconds)


def _failing(name: str, detail: str = "boom", *, seconds: float = 0.1) -> CheckOutcome:
    return CheckOutcome(name=name, ok=False, detail=detail, duration=seconds)


def test_a_part_that_breaks_a_project_check_is_named(root: Path) -> None:
    watch = DamageWatch(root, {}, {"ruff": _ok("ruff")})
    found = watch.after_part("part 2", ["src/a.py"], {"ruff": _failing("ruff", "E402 here")})
    assert [(d.label, d.check, d.error) for d in found] == [("part 2", "ruff", "E402 here")]
    assert found[0].subject == "ruff"  # no file, so the check names itself
    assert found[0].path == ""


def test_a_check_already_red_on_arrival_is_nobody_s(root: Path) -> None:
    watch = DamageWatch(root, {}, {"pytest": _failing("pytest")})
    assert watch.after_part("part 1", ["src/a.py"], {"pytest": _failing("pytest")}) == []


def test_only_the_part_that_broke_it_is_charged(root: Path) -> None:
    watch = DamageWatch(root, {}, {"ruff": _ok("ruff")})
    watch.after_part("part 1", ["src/a.py"], {"ruff": _failing("ruff")})
    watch.after_part("part 2", ["src/b.py"], {"ruff": _failing("ruff")})
    assert [d.label for d in watch.damage] == ["part 1"]


def test_a_later_part_that_makes_the_check_pass_again_is_credited(root: Path) -> None:
    watch = DamageWatch(root, {}, {"ruff": _ok("ruff")})
    watch.after_part("part 1", ["src/a.py"], {"ruff": _failing("ruff")})
    watch.after_part("part 2", ["src/b.py"], {"ruff": _ok("ruff")})
    assert watch.damage[0].repaired_by == "part 2"
    assert watch.outstanding == []


def test_a_check_that_could_not_run_says_nothing_either_way(root: Path) -> None:
    """Leaving the remembered state alone keeps the next part compared against a real
    measurement rather than against a gap."""
    watch = DamageWatch(root, {}, {"ruff": _ok("ruff")})
    skipped = CheckOutcome(name="ruff", ok=True, skipped="timed out after 3s")
    assert watch.after_part("part 1", ["src/a.py"], {"ruff": skipped}) == []
    found = watch.after_part("part 2", ["src/b.py"], {"ruff": _failing("ruff")})
    assert [d.label for d in found] == ["part 2"]


def test_a_file_break_and_a_check_break_in_one_part_are_both_recorded(root: Path) -> None:
    watch = DamageWatch(root, syntax_state(root, ["src/a.py"]), {"ruff": _ok("ruff")})
    _break(root, "src/a.py")
    found = watch.after_part("part 1", ["src/a.py"], {"ruff": _failing("ruff")})
    assert sorted(d.check for d in found) == ["ruff", "syntax"]


def test_a_repair_matches_the_check_it_belongs_to(root: Path) -> None:
    """A file repaired must not be credited against an outstanding command failure."""
    watch = DamageWatch(root, syntax_state(root, ["src/a.py"]), {"ruff": _ok("ruff")})
    _break(root, "src/a.py")
    watch.after_part("part 1", ["src/a.py"], {"ruff": _failing("ruff")})
    _mend(root, "src/a.py")
    watch.after_part("part 2", ["src/a.py"], {"ruff": _failing("ruff")})

    by_check = {d.check: d.repaired_by for d in watch.damage}
    assert by_check == {"syntax": "part 2", "ruff": None}


# --- which checks are cheap enough ---------------------------------------------------------


def test_the_baseline_decides_which_checks_run_per_part() -> None:
    measured = {"ruff check .": _ok("ruff check .", seconds=0.4),
                "pytest -q": _ok("pytest -q", seconds=12.0)}
    assert _cheap_enough(measured, 3.0) == ["ruff check ."]


def test_a_check_with_no_baseline_never_runs_per_part() -> None:
    """Without a "before" there is nothing to compare against, whatever it costs."""
    measured = {"ruff": CheckOutcome(name="ruff", ok=True, skipped="not on PATH", duration=0.0)}
    assert _cheap_enough(measured, 3.0) == []


def test_a_budget_of_zero_keeps_the_parser_alone() -> None:
    assert _cheap_enough({"ruff": _ok("ruff", seconds=0.01)}, 0.0) == []


def test_a_check_that_was_failing_at_baseline_still_runs_per_part() -> None:
    """It ran, so it has a "before" — one that says red, which is what stops it being charged."""
    assert _cheap_enough({"pytest": _failing("pytest", seconds=0.5)}, 3.0) == ["pytest"]
