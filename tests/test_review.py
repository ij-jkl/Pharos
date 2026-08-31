"""Diff review: the model proposes, and every word of it is checked before anyone sees it.

The interesting tests here are the ones that THROW A FINDING AWAY. A review that prints what
the model said is trivial to write and worth nothing — the reason this module is allowed to
exist at all is that a finding pointing at a file nobody changed, or at a line the diff does
not contain, never reaches the screen.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from rich.console import Console

from pharos.agent.cli import _render_review
from pharos.agent.review import (
    FileDiff,
    Finding,
    Hunk,
    Review,
    ask,
    batches,
    collect,
    hunks_of,
    parse,
    request_payload,
    validate,
)
from pharos.config import PharosConfig


def _config(**overrides: object) -> PharosConfig:
    base: dict[str, object] = {"backend_url": "http://localhost:11434"}
    base.update(overrides)
    return PharosConfig(**base)  # type: ignore[arg-type]


_DIFF = """\
--- a/src/alpha.py
+++ b/src/alpha.py
@@ -10,6 +10,8 @@ def widen(values):
     total = 0
     for value in values:
-        total += value
+        total += value * 2
+    if total < 0:
+        raise ValueError(total)
     return total
"""


def _diff(display: str = "src/alpha.py", text: str = _DIFF) -> FileDiff:
    return FileDiff(display=display, text=text, hunks=hunks_of(text))


# --- reading the diff ---------------------------------------------------------------------


def test_hunks_are_numbered_in_the_new_file() -> None:
    """The only numbering a finding may use — the one a reader will open the file at."""
    assert hunks_of(_DIFF) == (Hunk(start=10, end=17),)


def test_a_single_line_hunk_has_a_length_of_one() -> None:
    text = "@@ -4 +4 @@\n-old\n+new\n"
    assert hunks_of(text) == (Hunk(start=4, end=4),)


def test_several_hunks_are_all_kept() -> None:
    text = "@@ -1,2 +1,2 @@\n x\n@@ -40,3 +50,4 @@\n y\n"
    assert hunks_of(text) == (Hunk(start=1, end=2), Hunk(start=50, end=53))


def test_prose_holds_no_hunks() -> None:
    assert hunks_of("this is not a diff at all") == ()


def test_a_workspace_without_git_diffs_against_the_undo_snapshot(tmp_path: Path) -> None:
    """The folder that is not a repository still gets a review: the originals are on disk."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    diffs = collect(tmp_path, ["src/alpha.py"], originals={"src/alpha.py": "a = 1\n"})
    assert len(diffs) == 1
    assert "+b = 2" in diffs[0].text
    assert diffs[0].hunks


def test_a_file_created_by_the_run_reads_as_wholly_added(tmp_path: Path) -> None:
    (tmp_path / "new.py").write_text("print(1)\n", encoding="utf-8")
    diffs = collect(tmp_path, ["new.py"], originals={})
    assert diffs and "+print(1)" in diffs[0].text


def test_a_file_that_did_not_change_produces_no_diff_to_review(tmp_path: Path) -> None:
    (tmp_path / "same.py").write_text("x = 1\n", encoding="utf-8")
    assert collect(tmp_path, ["same.py"], originals={"same.py": "x = 1\n"}) == []


# --- batching -----------------------------------------------------------------------------


def _count(text: str) -> int:
    return len(text) // 4


def test_diffs_are_packed_into_calls_that_fit() -> None:
    diffs = [_diff(f"src/f{i}.py") for i in range(4)]
    packed, oversized = batches(diffs, budget=_count(_DIFF) * 2, count=_count)
    assert oversized == []
    assert [len(batch) for batch in packed] == [2, 2]


def test_a_diff_too_large_for_one_call_is_left_out_rather_than_halved() -> None:
    """Half a diff reviewed as though it were whole is the failure read_file already refuses."""
    diffs = [_diff("src/huge.py"), _diff("src/small.py")]
    packed, oversized = batches(diffs, budget=_count(_DIFF) - 1, count=_count)
    assert oversized == ["src/huge.py", "src/small.py"]
    assert packed == []


# --- the request ---------------------------------------------------------------------------


def test_the_request_pins_the_shape_and_the_temperature() -> None:
    payload = request_payload(_config(), [_diff()], "test-model")
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["format"]["required"] == ["findings"]
    assert "src/alpha.py" in payload["messages"][0]["content"]
    assert "+        total += value * 2" in payload["messages"][0]["content"]


def test_an_unreachable_backend_is_an_error_string_not_an_exception() -> None:
    reply, error = ask(_config(backend_url="http://127.0.0.1:9"), [_diff()], "test-model")
    assert reply is None
    assert error


# --- parsing -------------------------------------------------------------------------------


def test_prose_instead_of_json_is_an_error() -> None:
    found, error = parse("I had a look and it seems fine to me!")
    assert found == [] and error == "the reply was not JSON"


def test_a_reply_with_no_findings_array_is_an_error_not_an_empty_review() -> None:
    found, error = parse(json.dumps({"result": "ok"}))
    assert found == [] and error == "the reply carried no findings array"


def test_an_empty_findings_list_parses_cleanly() -> None:
    """A model with nothing to say is a valid answer, and must not read as a failure."""
    found, error = parse(json.dumps({"findings": []}))
    assert found == [] and error is None


# --- the check that makes this admissible -----------------------------------------------------


def _raw(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "file": "src/alpha.py",
        "line": 12,
        "severity": "bug",
        "note": "the doubling changes the result for every existing caller",
    }
    base.update(overrides)
    return base


def test_a_finding_pointing_at_a_shown_line_survives() -> None:
    kept, discarded = validate([_raw()], [_diff()])
    assert discarded == 0
    assert kept[0].file == "src/alpha.py" and kept[0].line == 12


def test_a_finding_about_a_file_the_run_never_changed_is_discarded() -> None:
    kept, discarded = validate([_raw(file="src/imaginary.py")], [_diff()])
    assert kept == [] and discarded == 1


def test_a_finding_pointing_outside_the_diff_is_discarded() -> None:
    """The one that matters: a plausible line number is exactly what a reader would trust."""
    kept, discarded = validate([_raw(line=412)], [_diff()])
    assert kept == [] and discarded == 1


def test_a_severity_nobody_asked_for_is_discarded() -> None:
    kept, discarded = validate([_raw(severity="critical")], [_diff()])
    assert kept == [] and discarded == 1


def test_an_empty_note_is_discarded() -> None:
    kept, discarded = validate([_raw(note="   ")], [_diff()])
    assert kept == [] and discarded == 1


def test_a_line_that_is_not_a_number_is_discarded() -> None:
    kept, discarded = validate([_raw(line="around the middle")], [_diff()])
    assert kept == [] and discarded == 1


def test_a_boolean_is_not_a_line_number() -> None:
    """bool is an int in Python, and `"line": true` would otherwise sail through."""
    kept, discarded = validate([_raw(line=True)], [_diff()])
    assert kept == [] and discarded == 1


def test_the_separator_does_not_decide_whether_a_finding_matches() -> None:
    windows = "src" + chr(92) + "alpha.py"
    kept, _ = validate([_raw(file=windows)], [_diff()])
    assert kept and kept[0].file == "src/alpha.py"  # reported under the name we sent


def test_a_severity_in_the_wrong_case_still_counts() -> None:
    kept, discarded = validate([_raw(severity="BUG")], [_diff()])
    assert discarded == 0 and kept[0].severity == "bug"


def test_a_runaway_note_is_cut_rather_than_printed_whole() -> None:
    kept, _ = validate([_raw(note="x " * 400)], [_diff()])
    assert kept and len(kept[0].note) <= 300


def test_the_good_and_the_invented_are_separated_in_one_reply() -> None:
    """The realistic case: a model that is partly right, and the count of how partly."""
    kept, discarded = validate(
        [_raw(), _raw(line=999), _raw(file="nope.py"), _raw(line=14, severity="note")],
        [_diff()],
    )
    assert len(kept) == 2 and discarded == 2


# --- how it reads -----------------------------------------------------------------------------
#
# The panel is not decoration here. A review that could not happen must not read like a clean
# bill of health, and a finding must never be printed without the label saying what it is.


def _rendered(review: Review | None) -> str:
    """The panel, with Rich's line wrapping flattened — these assertions are about wording."""
    console = Console(file=io.StringIO(), width=110, force_terminal=False)
    _render_review(console, review)
    return " ".join(console.file.getvalue().split())  # type: ignore[attr-defined]


def test_no_review_prints_nothing() -> None:
    assert _rendered(None).strip() == ""


def test_a_review_that_could_not_run_says_so_rather_than_looking_clean() -> None:
    text = _rendered(Review(note="the backend returned an empty reply"))
    assert "not run" in text
    assert "empty reply" in text
    assert "nothing reported" not in text


def test_an_empty_review_is_labelled_as_an_opinion_all_the_same() -> None:
    text = _rendered(Review(reviewed=["src/alpha.py"]))
    assert "nothing reported" in text
    assert "opinion" in text
    assert "settled before it was asked" in text


def test_a_finding_is_printed_with_its_file_line_and_severity() -> None:
    text = _rendered(
        Review(
            findings=[Finding("src/alpha.py", 12, "bug", "the doubling breaks every caller")],
            reviewed=["src/alpha.py"],
        )
    )
    assert "src/alpha.py:12" in text
    assert "bug" in text
    assert "the doubling breaks every caller" in text


def test_what_was_thrown_away_is_printed_next_to_what_survived() -> None:
    """The measurement of how much of this particular answer was invented."""
    text = _rendered(
        Review(
            findings=[Finding("src/alpha.py", 12, "note", "fine")],
            reviewed=["src/alpha.py"],
            discarded=3,
            unreviewed=["src/enormous.py"],
        )
    )
    assert "3 finding(s) discarded" in text
    assert "src/enormous.py" in text


def test_a_run_that_changed_nothing_still_answers_the_flag() -> None:
    """A flag that prints nothing when the answer is "there was nothing to look at" is
    indistinguishable from one that quietly failed."""
    text = _rendered(Review(note="the run changed no files, so there was no diff to read"))
    assert "not run" in text and "no diff to read" in text


# --- a hunk that adds nothing holds no line ------------------------------------------------------


def test_a_pure_deletion_hunk_holds_no_line() -> None:
    """`+N,0` says the hunk adds nothing to the new file, and both git and difflib emit it.

    `max(length, 1)` was there for the `@@ -1 +1 @@` form, where an absent count means one
    line. Applied to an explicit zero it invented a line at N -- so a finding pointing at a
    file the run had emptied passed the very check that exists to stop a model naming a line
    it was never shown.
    """
    hunks = hunks_of("@@ -1,12 +0,0 @@\n-a\n-b\n")

    assert all(not hunk.holds(line) for hunk in hunks for line in range(14))


def test_an_absent_count_still_means_one_line() -> None:
    """The form `max(length, 1)` was protecting must keep working."""
    (hunk,) = hunks_of("@@ -1 +7 @@\n+x\n")

    assert hunk.holds(7)
    assert not hunk.holds(6) and not hunk.holds(8)


def test_a_finding_on_an_emptied_file_is_discarded() -> None:
    """End to end: the model reviews a file the run emptied and points inside it anyway."""
    diff = FileDiff(
        display="gone.py", text="@@ -1,3 +0,0 @@\n-a\n-b\n-c\n", hunks=hunks_of("@@ -1,3 +0,0 @@")
    )
    raw = [{"file": "gone.py", "line": 1, "severity": "bug", "note": "still wrong"}]

    kept, discarded = validate(raw, [diff])

    assert kept == []
    assert discarded == 1
