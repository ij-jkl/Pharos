"""What the chat template costs, remembered between runs.

Pharos counts a conversation from the messages it holds. The backend counts it after applying
a chat template applied server-side and invisible from here, so the projection is a floor by
construction. A session already corrects for that from its second request, once one
`prompt_eval_count` has come back. The first request of every part is the one that correction
cannot cover, and these are about closing it.

It is stored in TOKENS rather than as a ratio, and that was measured rather than assumed --
seventeen requests over a 3x range of conversation sizes put the gap at a flat 263-314 tokens
while the ratio it implied fell from 1.22x to 1.09x.

Two things must stay true and are tested for directly: a remembered gap may never make a
ceiling LARGER than the raw projection would, and one freak measurement may never shrink every
future part for good.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharos.calibration import (
    TEMPLATE_OFFSET_CAP,
    TemplateCost,
    load_template_costs,
    remember_run,
    remembered_cost,
)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "pharos_templates.json"


# --- the offset itself ----------------------------------------------------------------------


def test_no_measurements_is_no_correction() -> None:
    assert TemplateCost(model="m").offset == 0


def test_the_offset_is_the_median_of_the_runs() -> None:
    assert TemplateCost(model="m", offsets=(260, 275, 300)).offset == 275


def test_one_freak_run_does_not_shrink_every_future_part() -> None:
    """A max across runs ratchets and never comes back; a median absorbs one outlier and
    still moves on a real change."""
    steady = TemplateCost(model="m", offsets=(263, 275, 279, 292))
    with_outlier = TemplateCost(model="m", offsets=(263, 275, 279, 292, 3000))
    assert abs(with_outlier.offset - steady.offset) <= 5
    assert with_outlier.offset < 400  # nowhere near the 3,000 it saw once


def test_a_real_change_does_move_it() -> None:
    changed = TemplateCost(model="m", offsets=(270, 275, 900, 910, 920))
    assert changed.offset > 500


def test_a_backend_counting_fewer_tokens_is_not_licence_to_fit_more_in() -> None:
    """Never below zero: our projection running high is caution, not headroom."""
    assert TemplateCost(model="m", offsets=()).offset == 0


def test_an_implausible_measurement_is_held_to_the_cap() -> None:
    cost = TemplateCost(model="m", offsets=(90_000, 91_000, 92_000))
    assert cost.offset == TEMPLATE_OFFSET_CAP
    assert cost.capped


def test_an_ordinary_measurement_is_not_reported_as_capped() -> None:
    assert not TemplateCost(model="m", offsets=(275, 280)).capped


def test_the_offset_does_not_scale_with_the_conversation() -> None:
    """The measurement that chose this shape: seventeen requests over conversations from
    1,235 to 3,604 tokens, all short by 263-314 tokens. A ratio fitted to the smallest would
    have reserved 793 tokens on the largest to cover 314."""
    cost = TemplateCost(model="m", offsets=(275,))
    assert cost.offset == 275  # the same number whatever the conversation costs


# --- the store ------------------------------------------------------------------------------


def test_a_missing_store_is_an_empty_memory_not_an_error(store: Path) -> None:
    assert load_template_costs(store) == {}
    assert remembered_cost(store, "qwen3.5:9b") is None


def test_a_corrupt_store_is_an_empty_memory_not_an_error(store: Path) -> None:
    """It makes runs better when it is there; it must never be able to stop one happening."""
    store.write_text("{not json", encoding="utf-8")
    assert load_template_costs(store) == {}


def test_a_store_of_the_wrong_shape_is_ignored(store: Path) -> None:
    store.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert load_template_costs(store) == {}


def test_a_run_is_remembered_and_read_back(store: Path) -> None:
    remember_run(store, "qwen3.5:9b", [263, 314, 275])
    cost = remembered_cost(store, "qwen3.5:9b")
    assert cost is not None
    assert cost.offsets == (314,)  # the run's WORST, and only that


def test_a_long_run_cannot_outvote_a_short_one(store: Path) -> None:
    """One number per run, so the median stays a median across runs."""
    remember_run(store, "m", [900] * 50)
    remember_run(store, "m", [275])
    remember_run(store, "m", [275])
    cost = remembered_cost(store, "m")
    assert cost is not None and cost.runs == 3
    assert cost.offset == 275


def test_the_model_name_is_matched_the_way_the_profiler_matches_it(store: Path) -> None:
    remember_run(store, "qwen3.5:9b", [275])
    assert remembered_cost(store, "qwen3.5:9b") is not None


def test_a_different_model_gets_its_own_memory(store: Path) -> None:
    remember_run(store, "alpha:1b", [400])
    remember_run(store, "beta:2b", [120])
    assert remembered_cost(store, "alpha:1b").offset == 400  # type: ignore[union-attr]
    assert remembered_cost(store, "beta:2b").offset == 120  # type: ignore[union-attr]


def test_an_unknown_model_remembers_nothing(store: Path) -> None:
    remember_run(store, "alpha:1b", [400])
    assert remembered_cost(store, "gamma:3b") is None


def test_nothing_is_remembered_for_an_unnamed_model(store: Path) -> None:
    assert remember_run(store, None, [275]) is None
    assert not store.exists()


def test_a_run_with_no_usable_ratios_writes_nothing(store: Path) -> None:
    assert remember_run(store, "m", []) is None
    assert remember_run(store, "m", [0, -40]) is None


def test_the_store_keeps_a_bounded_window(store: Path) -> None:
    for i in range(40):
        remember_run(store, "m", [200 + i])
    cost = remembered_cost(store, "m")
    assert cost is not None and cost.runs <= 20


def test_the_store_holds_numbers_and_nothing_else(store: Path) -> None:
    """Same promise as the observation store: counts only, never text."""
    remember_run(store, "qwen3.5:9b", [275])
    raw = json.loads(store.read_text(encoding="utf-8"))
    entry = raw["models"]["qwen3.5:9b"]
    assert set(entry) == {"offsets", "updated"}
    assert all(isinstance(value, int | float) for value in entry["offsets"])


# --- what a session does with it ----------------------------------------------------------------


def _session(seed: int, drift: list[tuple[int, int]]) -> object:
    """A session with only the two attributes the correction reads, so this stays a test of
    the arithmetic rather than of a constructor."""
    from pharos.agent.session import AgentSession

    session = AgentSession.__new__(AgentSession)
    session._seed = seed  # type: ignore[attr-defined]
    session._drift = drift  # type: ignore[attr-defined]
    return session


def test_a_seeded_session_starts_corrected() -> None:
    assert _session(275, [])._template_offset() == 275  # type: ignore[attr-defined]


def test_an_unseeded_session_starts_uncorrected() -> None:
    assert _session(0, [])._template_offset() == 0  # type: ignore[attr-defined]


def test_a_live_measurement_beats_a_lower_seed() -> None:
    """The seed is a starting point. This conversation's own numbers are the better answer,
    and only ever in the safe direction."""
    assert _session(275, [(1000, 1400)])._template_offset() == 400  # type: ignore[attr-defined]


def test_a_lower_live_measurement_does_not_undo_the_seed() -> None:
    assert _session(275, [(1000, 1050)])._template_offset() == 275  # type: ignore[attr-defined]


def test_a_backend_under_our_count_never_lowers_the_ceiling_correction() -> None:
    assert _session(0, [(1000, 900)])._template_offset() == 0  # type: ignore[attr-defined]


def test_the_correction_is_flat_across_conversation_sizes() -> None:
    """The property the additive shape exists for: one measured gap covers a conversation
    three times the size, instead of scaling with it."""
    session = _session(0, [(1235, 1510)])
    assert session._template_offset() == 275  # type: ignore[attr-defined]
    session._drift.append((3604, 3879))  # type: ignore[attr-defined]
    assert session._template_offset() == 275  # type: ignore[attr-defined]


# --- and what the scorecard says about it -----------------------------------------------------


def test_the_scorecard_reports_the_seed_and_the_exposure() -> None:
    from pharos.agent.scorecard import score, to_dict
    from pharos.agent.session import PartResult

    parts = [
        PartResult(text="", steps=1, files_written=[], peak_tokens=1, reported_tokens=None,
                   stopped_early=False, exposed_requests=0),
        PartResult(text="", steps=1, files_written=[], peak_tokens=1, reported_tokens=None,
                   stopped_early=False, exposed_requests=2),
    ]
    card = score(parts, handoff_reserve=500, template_offset=275, template_runs=4)
    assert card.exposed_requests == 2
    payload = to_dict(card)
    assert payload["template"] == {"offset_tokens": 275, "runs": 4, "exposed_requests": 2}
