"""Scorecard tests: whether a run added up, judged without asking a model anything.

The distinction these exist to protect is between a part that lost the thread and a model
that was simply confused about the project. Both show up as refused tool calls; only the
first says the hand-off failed, and conflating them would make the continuity number
meaningless in exactly the runs where it matters.
"""

from __future__ import annotations

from pharos.agent.scorecard import score, to_dict
from pharos.agent.session import PartResult


def _part(
    *,
    scoped: list[str],
    wrote: list[str] | None = None,
    handoff: str = "did the thing",
    handoff_tokens: int = 60,
    refusals: list[str] | None = None,
    peak: int = 100,
    ceiling: int = 1000,
    reported: int | None = 100,
    nudged: bool = False,
    stopped: bool = False,
    error: str | None = None,
    drift_samples: list[tuple[int, int]] | None = None,
) -> PartResult:
    return PartResult(
        text=handoff,
        steps=1,
        files_written=wrote or [],
        peak_tokens=peak,
        reported_tokens=reported,
        stopped_early=stopped,
        scope_refusals=refusals or [],
        error=error,
        ceiling=ceiling,
        nudged=nudged,
        scoped=scoped,
        handoff_tokens=handoff_tokens,
        drift_samples=drift_samples or [],
    )


# --- coverage ---------------------------------------------------------------------------


def test_a_run_that_wrote_everything_it_owned_is_complete() -> None:
    parts = [
        _part(scoped=["a.py", "b.py"], wrote=["a.py", "b.py"]),
        _part(scoped=["c.py"], wrote=["c.py"]),
    ]
    card = score(parts, handoff_reserve=500)
    assert card.coverage == 1.0
    assert card.complete and card.kept_the_thread
    assert card.untouched == []


def test_parts_reporting_done_do_not_make_a_run_complete() -> None:
    """The failure this exists for: a part finishes by replying without calling a tool, which
    is exactly what a model does after describing an edit instead of making it."""
    parts = [
        _part(scoped=["a.py", "b.py"], wrote=["a.py"]),
        _part(scoped=["c.py"], wrote=[]),  # "done", wrote nothing
    ]
    card = score(parts, handoff_reserve=500)
    assert card.coverage == 1 / 3
    assert not card.complete
    assert card.untouched == ["b.py", "c.py"]
    assert card.parts_that_wrote == 1


def test_an_unrestricted_run_has_no_coverage_to_report() -> None:
    """One part with no scope has no denominator; inventing one would be worse than None."""
    card = score([_part(scoped=[], wrote=["a.py"])], handoff_reserve=500)
    assert card.coverage is None
    assert card.written_files == 1


# --- continuity -------------------------------------------------------------------------


def test_reaching_for_another_parts_file_counts_as_losing_the_thread() -> None:
    """A part redoing work already done is the fingerprint the hand-off did not carry."""
    parts = [
        _part(scoped=["a.py"], wrote=["a.py"]),
        _part(scoped=["b.py"], wrote=["b.py"], refusals=["a.py"]),
    ]
    card = score(parts, handoff_reserve=500)
    assert card.revisits == ["a.py"]
    assert card.invented == []
    assert not card.kept_the_thread


def test_an_invented_path_is_not_a_continuity_failure() -> None:
    """A real run tried to write to a Data/ folder that has never existed in the project.

    That is confusion about the repository, not about what has already been done, and counting
    it as lost continuity would make the number mean nothing.
    """
    parts = [
        _part(scoped=["a.py"], wrote=["a.py"]),
        _part(scoped=["b.py"], wrote=["b.py"], refusals=["Data/YourDbContext.cs"]),
    ]
    card = score(parts, handoff_reserve=500)
    assert card.revisits == []
    assert card.invented == ["Data/YourDbContext.cs"]
    assert card.kept_the_thread  # the thread held; the model just wandered


def test_a_missing_handoff_breaks_continuity() -> None:
    parts = [
        _part(scoped=["a.py"], wrote=["a.py"], handoff=""),
        _part(scoped=["b.py"], wrote=["b.py"]),
    ]
    card = score(parts, handoff_reserve=500)
    assert card.handoffs_expected == 1 and card.handoffs_produced == 0
    assert not card.kept_the_thread


def test_the_last_part_is_not_expected_to_hand_off() -> None:
    """It has nobody to hand to; demanding one would make every clean run look broken."""
    parts = [
        _part(scoped=["a.py"], wrote=["a.py"]),
        _part(scoped=["b.py"], wrote=["b.py"], handoff=""),
    ]
    assert score(parts, handoff_reserve=500).kept_the_thread


def test_a_handoff_over_the_reserve_is_reported() -> None:
    """The next part was planned around this many tokens; a bigger one arrives truncated."""
    parts = [
        _part(scoped=["a.py"], wrote=["a.py"], handoff_tokens=900),
        _part(scoped=["b.py"], wrote=["b.py"]),
    ]
    card = score(parts, handoff_reserve=500)
    assert card.handoff_overruns == 1
    assert not card.kept_the_thread


def test_a_single_part_run_expects_no_handoffs() -> None:
    card = score([_part(scoped=["a.py"], wrote=["a.py"], handoff="")], handoff_reserve=500)
    assert card.handoffs_expected == 0 and card.kept_the_thread


# --- headroom and drift -------------------------------------------------------------------


def test_headroom_is_the_worst_part_not_the_average() -> None:
    """One part scraping its ceiling is what breaks next time, however roomy the others were."""
    parts = [
        _part(scoped=["a.py"], wrote=["a.py"], peak=100, ceiling=1000),
        _part(scoped=["b.py"], wrote=["b.py"], peak=950, ceiling=1000),
    ]
    assert score(parts, handoff_reserve=500).peak_fraction == 0.95


def test_drift_pairs_each_projection_with_its_own_request() -> None:
    """Dividing a part's PEAK by whichever count arrived last compares two different requests.

    Measured that way a real run read 2.56x where its honest worst case was 1.53x, because the
    peak came from mid-loop and the reported count from the final, smaller call.
    """
    parts = [
        _part(scoped=["a.py"], wrote=["a.py"], peak=9_000,  # never used for drift
              drift_samples=[(1200, 1000), (2600, 2000)]),
    ]
    card = score(parts, handoff_reserve=500)
    assert card.drift_low == 1.2
    assert card.drift_high == 1.3
    assert card.drift_samples == 2
    assert not card.under_counted


def test_a_small_shortfall_is_normal_and_raises_nothing() -> None:
    """Being slightly under is the measured resting state, not an incident.

    The chat template adds scaffolding Pharos cannot see, so the projection sits about 40-50
    tokens below the backend's count whatever the conversation size. SAFETY_MARGIN exists for
    exactly that, and an alarm firing on every run would be noise.
    """
    parts = [_part(scoped=["a.py"], wrote=["a.py"], drift_samples=[(1_378, 1_430)])]
    card = score(parts, handoff_reserve=500)
    assert card.drift_low < 1.0  # under...
    assert card.worst_shortfall == 52
    assert not card.under_counted  # ...but well inside the margin


def test_a_shortfall_bigger_than_the_margin_raises_the_alarm() -> None:
    """Past that point the ceiling is being enforced against a number below the real prompt."""
    parts = [_part(scoped=["a.py"], wrote=["a.py"], drift_samples=[(9_000, 9_600)])]
    card = score(parts, handoff_reserve=500)
    assert card.worst_shortfall == 600
    assert card.under_counted


def test_over_counting_never_raises_the_alarm() -> None:
    """Wasted room is not a safety problem, so it must not read as one."""
    parts = [_part(scoped=["a.py"], wrote=["a.py"], drift_samples=[(2_000, 1_000)])]
    card = score(parts, handoff_reserve=500)
    assert card.worst_shortfall == 0 and not card.under_counted


def test_drift_is_absent_when_the_backend_never_said() -> None:
    parts = [_part(scoped=["a.py"], wrote=["a.py"], drift_samples=[])]
    card = score(parts, handoff_reserve=500)
    assert card.drift_high is None and card.drift_low is None
    assert not card.under_counted  # unknown must not read as an alarm


# --- the machine-readable form ------------------------------------------------------------


def test_the_json_form_carries_the_verdicts_a_script_would_assert_on() -> None:
    parts = [
        _part(scoped=["a.py"], wrote=["a.py"]),
        _part(scoped=["b.py"], wrote=[], refusals=["a.py"], nudged=True),
    ]
    payload = to_dict(score(parts, handoff_reserve=500))
    assert payload["complete"] is False
    assert payload["kept_the_thread"] is False
    assert payload["coverage"] == 0.5
    assert payload["revisits"] == ["a.py"]
    assert payload["nudged_parts"] == 1


def test_a_failed_part_stops_a_run_being_complete_even_at_full_coverage() -> None:
    parts = [_part(scoped=["a.py"], wrote=["a.py"], error="backend call failed: ReadTimeout")]
    card = score(parts, handoff_reserve=500)
    assert card.coverage == 1.0 and not card.complete


def test_a_partial_run_must_not_look_like_a_success() -> None:
    """`complete` is what the exit code keys off, so a flattering value here is a lie in CI."""
    parts = [_part(scoped=["a.py", "b.py"], wrote=["a.py"])]
    card = score(parts, handoff_reserve=500)
    assert not card.complete  # every part "done", one file never written


def test_an_unrestricted_run_that_wrote_something_counts_as_complete() -> None:
    """No scope means no coverage to fall short of; refusing to ever exit 0 would be useless."""
    card = score([_part(scoped=[], wrote=["a.py"])], handoff_reserve=500)
    assert card.coverage is None
    assert not card.failed_parts and not card.abandoned_parts


def test_a_part_that_wrote_files_and_said_nothing_lost_the_thread() -> None:
    """Measured on a real run: three of three hand-offs produced, largest SIX tokens against a
    500-token reserve, while coverage sat at 31%. The metric called that continuity."""
    parts = [
        _part(scoped=["a.py"], wrote=["a.py"], handoff="ok", handoff_tokens=6),
        _part(scoped=["b.py"], wrote=["b.py"]),
    ]
    card = score(parts, handoff_reserve=500)
    assert card.handoffs_produced == card.handoffs_expected == 1  # present...
    assert card.thin_handoffs == 1  # ...and carrying nothing
    assert not card.kept_the_thread


def test_a_part_that_changed_nothing_may_be_brief() -> None:
    """Silence is the honest answer when there was nothing to report; only work needs a summary."""
    parts = [
        _part(scoped=["a.py"], wrote=[], handoff="NO CHANGES NEEDED", handoff_tokens=4),
        _part(scoped=["b.py"], wrote=["b.py"]),
    ]
    card = score(parts, handoff_reserve=500)
    assert card.thin_handoffs == 0
    assert card.kept_the_thread
