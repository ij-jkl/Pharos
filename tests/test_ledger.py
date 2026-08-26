"""The run's own record of what landed, and what it is allowed to change.

Two things are under test and the second matters as much as the first: that the record is
exact, and that it does not leak into the numbers that measure the model. A ledger that
quietly made `thin_handoffs` go to zero would be a worse feature than no ledger at all.

Counting is a stand-in ``len(text) // 4`` throughout, so every budget assertion is exact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pharos.agent.ledger import (
    LEDGER_SHARE,
    SAMPLE_LINES,
    FileChange,
    Ledger,
    added_lines,
    names_its_work,
)
from pharos.agent.runner import _carry, _with_handoff
from pharos.agent.scorecard import score
from pharos.agent.session import PartResult
from pharos.agent.tools import ToolBox
from pharos.agent.workspace import Workspace
from pharos.preflight.split import PartFile


def _count(text: str) -> int:
    return len(text) // 4


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
    (tmp_path / "src" / "beta.py").write_text("beta = 2\nbeta_two = 3\n", encoding="utf-8")
    return Workspace(tmp_path)


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


# --- the diff itself --------------------------------------------------------------------


def test_added_lines_reports_only_what_is_new() -> None:
    assert added_lines("a\nb\n", "a\nNEW\nb\n") == ["NEW"]


def test_added_lines_is_empty_when_nothing_changed() -> None:
    assert added_lines("a\nb\n", "a\nb\n") == []


def test_added_lines_counts_a_duplicated_line_as_an_addition() -> None:
    """A set difference would miss this; a real diff does not."""
    assert added_lines("a\n", "a\na\n") == ["a"]


def test_added_lines_ignores_a_line_that_only_moved() -> None:
    assert added_lines("a\nb\n", "b\na\n") in ([["a"]] and [["a"], ["b"]])


def test_added_lines_treats_a_new_file_as_all_additions() -> None:
    assert added_lines("", "one\ntwo\n") == ["one", "two"]


def test_a_deletion_adds_nothing() -> None:
    assert added_lines("a\nb\n", "a\n") == []


# --- the dispatcher records as it writes --------------------------------------------------


def test_write_file_records_the_lines_it_added(workspace: Workspace) -> None:
    box = ToolBox(workspace=workspace)
    box.dispatch("write_file", {"path": "src/alpha.py", "content": "alpha = 1\nLAYER = 'a'\n"},
                 room=1000, count=_count)
    changes = box.changes()
    assert [c.display for c in changes] == ["src/alpha.py"]
    assert changes[0].added == ("LAYER = 'a'",)
    assert changes[0].added_total == 1


def test_replace_lines_records_the_lines_it_added(workspace: Workspace) -> None:
    box = ToolBox(workspace=workspace)
    box.dispatch(
        "replace_lines",
        {"path": "src/beta.py", "line_start": 1, "line_end": 1, "content": "beta = 99\nX = 1"},
        room=1000,
        count=_count,
    )
    changes = box.changes()
    assert changes[0].added == ("beta = 99", "X = 1")
    assert changes[0].added_total == 2


def test_a_new_file_is_recorded_against_an_empty_before(workspace: Workspace) -> None:
    box = ToolBox(workspace=workspace)
    box.dispatch("write_file", {"path": "src/gamma.py", "content": "one\ntwo\n"},
                 room=1000, count=_count)
    assert box.changes()[0].added == ("one", "two")


def test_a_refused_write_records_nothing(workspace: Workspace) -> None:
    """The scope layer refuses, so nothing landed and nothing may be claimed."""
    box = ToolBox(workspace=workspace, scope={})
    box.dispatch("write_file", {"path": "src/alpha.py", "content": "x = 1\n"},
                 room=1000, count=_count)
    assert box.changes() == []


def test_a_write_that_changed_nothing_records_no_additions(workspace: Workspace) -> None:
    """It still counts as a written file — the ledger just has nothing to show for it."""
    box = ToolBox(workspace=workspace)
    box.dispatch("write_file", {"path": "src/alpha.py", "content": "alpha = 1\n"},
                 room=1000, count=_count)
    assert box.files_written == ["src/alpha.py"]
    assert box.changes()[0].added == ()
    assert box.changes()[0].added_total == 0


def test_the_sample_is_capped_but_the_total_is_not(workspace: Workspace) -> None:
    body = "".join(f"line{i}\n" for i in range(SAMPLE_LINES * 3))
    box = ToolBox(workspace=workspace)
    box.dispatch("write_file", {"path": "src/big.py", "content": body}, room=1000, count=_count)
    change = box.changes()[0]
    assert len(change.added) == SAMPLE_LINES
    assert change.added_total == SAMPLE_LINES * 3
    assert change.sampled


def test_two_writes_to_one_file_accumulate(workspace: Workspace) -> None:
    box = ToolBox(workspace=workspace)
    box.dispatch("write_file", {"path": "src/alpha.py", "content": "alpha = 1\nA = 1\n"},
                 room=1000, count=_count)
    box.dispatch("write_file", {"path": "src/alpha.py", "content": "alpha = 1\nA = 1\nB = 2\n"},
                 room=1000, count=_count)
    change = box.changes()[0]
    assert change.added == ("A = 1", "B = 2")
    assert change.added_total == 2


# --- the record -----------------------------------------------------------------------------


def _ledger(*entries: tuple[str, list[FileChange]]) -> Ledger:
    record = Ledger()
    for label, changes in entries:
        record.record(label, changes)
    return record


def test_a_part_that_changed_nothing_adds_no_entry() -> None:
    record = Ledger()
    record.record("part 1", [])
    assert record.entries == []
    assert record.render(count=_count, budget=1000) == ""


def test_the_record_names_the_files_and_shows_the_lines() -> None:
    record = _ledger(("part 1", [FileChange("src/a.py", ("LAYER = 'a'",), 1)]))
    text = record.render(count=_count, budget=1000)
    assert "src/a.py" in text
    assert "LAYER = 'a'" in text
    assert "+1 line" in text


def test_the_record_says_when_it_is_showing_a_sample() -> None:
    record = _ledger(("part 1", [FileChange("src/a.py", ("one", "two"), 9)]))
    text = record.render(count=_count, budget=1000)
    assert "+9 lines, 2 shown" in text


def test_the_record_says_the_files_are_not_the_reader_s() -> None:
    record = _ledger(("part 1", [FileChange("src/a.py", (), 0)]))
    text = record.render(count=_count, budget=1000)
    assert "do not open or change them" in text.lower()


def test_files_are_listed_once_across_parts_in_first_touched_order() -> None:
    record = _ledger(
        ("part 1", [FileChange("src/b.py", ("b1",), 1)]),
        ("part 2", [FileChange("src/a.py", ("a1",), 1), FileChange("src/b.py", ("b2",), 1)]),
    )
    assert record.files == ["src/b.py", "src/a.py"]
    text = record.render(count=_count, budget=1000)
    assert text.count("src/b.py") == 1
    # both parts' additions to the same file survive the merge
    assert "b1" in text and "b2" in text
    assert "+2 lines" in text


# --- the budget ------------------------------------------------------------------------------


def test_a_tight_budget_drops_the_sample_before_the_names() -> None:
    last = f"line{SAMPLE_LINES - 1}"
    record = _ledger(
        ("part 1", [FileChange("src/a.py", tuple(f"line{i}" for i in range(SAMPLE_LINES)),
                               SAMPLE_LINES)])
    )
    full = record.render(count=_count, budget=10_000)
    assert last in full
    squeezed = record.render(count=_count, budget=_count(full) - 3)
    assert "src/a.py" in squeezed
    assert last not in squeezed


def test_a_very_tight_budget_still_names_what_it_can_and_counts_the_rest() -> None:
    record = _ledger(
        ("part 1", [FileChange(f"src/file_{i}_with_a_long_name.py", (), 0) for i in range(8)])
    )
    text = record.render(count=_count, budget=60)
    assert _count(text) <= 60
    assert "more" in text


def test_a_budget_of_nothing_returns_nothing() -> None:
    record = _ledger(("part 1", [FileChange("src/a.py", ("x",), 1)]))
    assert record.render(count=_count, budget=0) == ""


def test_the_record_never_exceeds_the_budget_it_was_given() -> None:
    record = _ledger(
        ("part 1", [FileChange(f"src/f{i}.py", ("a", "b", "c"), 3) for i in range(12)])
    )
    for budget in (5, 20, 50, 120, 400, 4000):
        assert _count(record.render(count=_count, budget=budget)) <= budget


def test_the_share_leaves_the_prose_at_least_half_the_reserve() -> None:
    """The invariant the runner relies on to keep a hand-off from being squeezed to nothing."""
    assert 0.0 < LEDGER_SHARE <= 0.5


# --- what the part actually receives ----------------------------------------------------------


def test_the_record_goes_below_the_hand_off_and_above_the_target() -> None:
    """Where the two disagree, the one read last wins — and the record cannot be wrong."""
    text = _with_handoff(
        "BODY",
        "NO CHANGES NEEDED",
        2,
        [PartFile(display="src/c.py", tokens=10)],
        "RECORD-BLOCK",
    )
    assert text.index("NO CHANGES NEEDED") < text.index("RECORD-BLOCK")
    assert text.index("RECORD-BLOCK") < text.index("YOUR TARGET")


def test_a_part_with_no_record_reads_exactly_as_it_did_before() -> None:
    with_empty = _with_handoff("BODY", "prose", 2, None, "")
    without = _with_handoff("BODY", "prose", 2, None)
    assert with_empty == without


def test_the_record_reaches_a_first_part_that_has_no_hand_off() -> None:
    """A repair part is index 1 with nothing carried, and still needs the conventions."""
    text = _with_handoff("BODY", None, 1, None, "RECORD-BLOCK")
    assert "RECORD-BLOCK" in text
    assert "HAND-OFF FROM PART" not in text


# --- the rule that decides whether the model reported --------------------------------------


def test_naming_matches_on_the_basename() -> None:
    assert names_its_work("Added a constant to alpha.py", ["src/alpha.py"])


def test_naming_matches_a_windows_spelled_path() -> None:
    assert names_its_work("touched beta.py", [chr(92).join(["src", "beta.py"])])


def test_a_summary_that_names_nothing_it_wrote_does_not_count() -> None:
    assert not names_its_work("NO CHANGES NEEDED", ["src/alpha.py"])


def test_a_part_that_wrote_nothing_is_not_held_to_it() -> None:
    assert names_its_work("", [])


# --- and the scorecard keeps measuring the model ---------------------------------------------


def test_the_ledger_does_not_rescue_a_thin_hand_off() -> None:
    """The whole point. A part that wrote files and said nothing still reads as thin."""
    parts = [
        _part(text="NO CHANGES NEEDED", files_written=["src/a.py"], scoped=["src/a.py"]),
        _part(text="done", files_written=["src/b.py"], scoped=["src/b.py"]),
    ]
    card = score(parts, handoff_reserve=500, ledger_on=True, ledger_files=1)
    assert card.thin_handoffs == 1
    assert not card.kept_the_thread


def test_the_scorecard_reports_what_was_carried_beside_it() -> None:
    parts = [
        _part(text="NO CHANGES NEEDED", files_written=["src/a.py"], scoped=["src/a.py"]),
        _part(text="done", files_written=["src/b.py"], scoped=["src/b.py"]),
    ]
    card = score(parts, handoff_reserve=500, ledger_on=True, ledger_files=1)
    assert card.ledger_on
    assert card.ledger_files == 1


def test_a_run_without_the_ledger_says_so() -> None:
    card = score([_part()], handoff_reserve=500)
    assert not card.ledger_on
    assert card.ledger_files == 0


# --- one reserve, shared ------------------------------------------------------------------


def test_the_record_and_the_hand_off_together_stay_inside_one_reserve() -> None:
    """The invariant every part's ceiling was computed against."""
    record = _ledger(
        ("part 1", [FileChange(f"src/f{i}.py", ("aaaa", "bbbb", "cccc"), 3) for i in range(6)])
    )
    prose = "I did a great deal of work and here is a very long account of it. " * 40
    for reserve in (40, 120, 300, 500, 2000):
        text, carried, _cut, _left = _carry(record, prose, reserve, _count, ledger=True)
        assert _count(text) + _count(carried) <= reserve, reserve


def test_the_record_never_takes_more_than_its_share() -> None:
    record = _ledger(
        ("part 1", [FileChange(f"src/f{i}.py", ("aaaa", "bbbb"), 2) for i in range(20)])
    )
    text, _carried, _cut, _left = _carry(record, "short", 400, _count, ledger=True)
    assert _count(text) <= int(400 * LEDGER_SHARE)


def test_the_hand_off_keeps_at_least_half_the_reserve() -> None:
    record = _ledger(
        ("part 1", [FileChange(f"src/f{i}.py", ("aaaa", "bbbb"), 2) for i in range(20)])
    )
    _text, _carried, _cut, left = _carry(record, "short", 400, _count, ledger=True)
    assert left >= 200


def test_with_the_ledger_off_the_prose_gets_the_whole_reserve() -> None:
    record = _ledger(("part 1", [FileChange("src/a.py", ("x",), 1)]))
    text, _carried, _cut, left = _carry(record, "short", 400, _count, ledger=False)
    assert text == ""
    assert left == 400


def test_a_part_that_said_nothing_still_passes_the_record_on() -> None:
    """The measured failure: a part writes files and reports nothing. The thread survives."""
    record = _ledger(("part 1", [FileChange("src/a.py", ("LAYER = 1",), 1)]))
    text, carried, _cut, _left = _carry(record, "", 500, _count, ledger=True)
    assert carried == ""
    assert "src/a.py" in text and "LAYER = 1" in text


def test_blank_lines_are_counted_and_not_shown(workspace: Workspace) -> None:
    """Measured on the first live run: the record showed two blank lines after a constant,
    the next part read them as part of the pattern and reproduced them, and two of that run's
    lint failures were blank-line churn copied from one file to the next."""
    box = ToolBox(workspace=workspace)
    box.dispatch(
        "write_file",
        {"path": "src/alpha.py", "content": "LAYER = 'a'\n\n\nalpha = 1\n"},
        room=1000,
        count=_count,
    )
    change = box.changes()[0]
    assert change.added == ("LAYER = 'a'",)
    assert change.added_total == 3  # the blanks happened, and the total still says so


def test_the_record_says_how_many_it_is_showing_not_which() -> None:
    record = _ledger(("part 1", [FileChange("src/a.py", ("LAYER = 1",), 3)]))
    text = record.render(count=_count, budget=1000)
    assert "+3 lines, 1 shown" in text
