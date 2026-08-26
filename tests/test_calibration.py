"""Calibration tests: the store, the recorder's debounce, and both estimators.

The estimator invariants matter more than the plumbing: the overhead estimate must stay a
floor (minimum, never mean), prefer first-turn-like records so long sessions cannot inflate
it, and never mix models.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pharos.calibration import (
    Observation,
    ObservationRecorder,
    _parse_record,
    estimate_agent_reads,
    estimate_client_overhead,
    estimate_typical_output,
    load_observations,
)


def _obs(
    *,
    input_tokens: int,
    user_tokens: int,
    messages: int = 1,
    model: str | None = "qwen3.5-9b-heretic:latest",
    exact: bool = True,
    output: int | None = 500,
    ts: float = 1000.0,
    agent_shaped: bool = True,
    user_exact: bool | None = None,
) -> Observation:
    return Observation(
        ts=ts,
        endpoint="openai-chat",
        model=model,
        input_tokens=input_tokens,
        input_exact=exact,
        user_tokens=user_tokens,
        messages=messages,
        output_tokens=output,
        agent_shaped=agent_shaped,
        user_exact=user_exact,
    )


# --- store ---------------------------------------------------------------------------------


def test_store_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    recorder = ObservationRecorder(path)
    recorder.add(_obs(input_tokens=100, user_tokens=10))
    recorder._flush_sync()
    loaded = load_observations(path)
    assert len(loaded) == 1
    assert loaded[0].input_tokens == 100
    assert loaded[0].user_tokens == 10


def test_store_is_bounded_and_keeps_newest(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    recorder = ObservationRecorder(path)
    for i in range(250):
        recorder.add(_obs(input_tokens=i, user_tokens=0, ts=float(i)))
    recorder._flush_sync()
    loaded = load_observations(path)
    assert len(loaded) == 200
    assert loaded[-1].input_tokens == 249  # newest survive, oldest are dropped
    assert loaded[0].input_tokens == 50


def test_store_tolerates_garbage(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    assert load_observations(path) == []
    path.write_text('[{"ts": 1, "endpoint": "x"}, "junk"]', encoding="utf-8")
    assert load_observations(path) == []  # partial records parse to nothing, not a crash
    assert load_observations(tmp_path / "absent.json") == []


def test_store_contains_counts_only(tmp_path: Path) -> None:
    """The privacy line: nothing in the file is text from a request."""
    path = tmp_path / "obs.json"
    recorder = ObservationRecorder(path)
    recorder.add(_obs(input_tokens=100, user_tokens=10))
    recorder._flush_sync()
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "ts", "endpoint", "model", "input_tokens", "input_exact",
        "user_tokens", "messages", "output_tokens", "agent_shaped", "user_exact",
    }
    assert set(raw[0].keys()) == allowed
    # Names alone would not catch a field that later starts carrying content, so the guard
    # checks shapes too: every value is a number, a flag, or the model name. `endpoint` and
    # `model` are the only strings, and neither comes from the body's text.
    for key, value in raw[0].items():
        if key in {"endpoint", "model"}:
            assert value is None or isinstance(value, str)
        else:
            assert isinstance(value, (int, float, bool)) or value is None


def test_recorder_debounce_asks_for_flush_at_threshold(tmp_path: Path) -> None:
    recorder = ObservationRecorder(tmp_path / "obs.json")
    flags = [recorder.add(_obs(input_tokens=i, user_tokens=0)) for i in range(5)]
    assert flags == [False, False, False, False, True]  # 5th record trips the batch flush
    # While a flush is marked in-flight, further adds do not re-trigger it.
    assert recorder.add(_obs(input_tokens=9, user_tokens=0)) is False


# --- overhead estimator ---------------------------------------------------------------------


def test_overhead_prefers_fewest_message_records() -> None:
    """History inflates client_added; only first-turn-like records estimate the fixed part."""
    observations = [
        _obs(input_tokens=10_500, user_tokens=500, messages=1),
        _obs(input_tokens=11_000, user_tokens=800, messages=1),
        _obs(input_tokens=40_000, user_tokens=300, messages=21),  # deep-session: ignored
    ]
    estimate = estimate_client_overhead(observations, "qwen3.5-9b-heretic")
    assert estimate is not None
    assert estimate.tokens == 10_000  # min over the messages==1 group only
    assert "fewest-message = 1" in estimate.provenance


def test_a_bare_poke_at_the_proxy_does_not_erase_the_learned_overhead() -> None:
    """Found live: four curl requests collapsed a real 1,800-token estimate to 10.

    The estimator minimises over the fewest-message records, and a hand-written curl has one
    message, no system prompt and no tools — so it wins that minimum and defines the overhead
    for every later pre-flight. A proxy sees more than one client; the estimate has to survive
    that. Agent-shaped records are therefore preferred over bare ones.
    """
    agent = _obs(input_tokens=1_235, user_tokens=3, messages=2, agent_shaped=True)
    pokes = [
        _obs(input_tokens=14, user_tokens=4, messages=1, agent_shaped=False) for _ in range(4)
    ]
    estimate = estimate_client_overhead([agent, *pokes], "qwen3.5-9b-heretic")
    assert estimate is not None
    assert estimate.tokens == 1_232  # the agent's real overhead, not the poke's 10
    assert "agent-shaped" in estimate.provenance


def test_bare_records_still_estimate_when_that_is_all_there_is() -> None:
    """Preferring agent-shaped records must not mean refusing to answer without them."""
    pokes = [_obs(input_tokens=100, user_tokens=10, messages=1, agent_shaped=False)]
    estimate = estimate_client_overhead(pokes, "qwen3.5-9b-heretic")
    assert estimate is not None
    assert estimate.tokens == 90
    assert "all," in estimate.provenance  # and it says the pool was not agent-shaped


def test_records_written_before_the_field_existed_still_parse() -> None:
    """Old stores must degrade to 'not agent-shaped', never to a crash."""
    legacy = {
        "ts": 1.0, "endpoint": "ollama-chat", "model": "qwen3.5-9b-heretic",
        "input_tokens": 500, "input_exact": True, "user_tokens": 50,
        "messages": 2, "output_tokens": 10,
    }
    parsed = _parse_record(legacy)
    assert parsed is not None and parsed.agent_shaped is False


def test_overhead_is_minimum_not_mean() -> None:
    observations = [
        _obs(input_tokens=10_000, user_tokens=0, messages=1),
        _obs(input_tokens=30_000, user_tokens=0, messages=1),
    ]
    estimate = estimate_client_overhead(observations, "qwen3.5-9b-heretic")
    assert estimate is not None
    assert estimate.tokens == 10_000  # a mean (20k) would break the floor promise


def test_overhead_prefers_exact_records() -> None:
    observations = [
        _obs(input_tokens=5_000, user_tokens=0, messages=1, exact=False),
        _obs(input_tokens=9_000, user_tokens=0, messages=1, exact=True),
    ]
    estimate = estimate_client_overhead(observations, "qwen3.5-9b-heretic")
    assert estimate is not None
    assert estimate.tokens == 9_000  # the smaller number loses: it is only an estimate
    assert "exact" in estimate.provenance


def test_overhead_filters_by_model_tag_insensitively() -> None:
    observations = [
        _obs(input_tokens=10_000, user_tokens=0, model="other-model:latest"),
        _obs(input_tokens=7_000, user_tokens=0, model="qwen3.5-9b-heretic:latest"),
    ]
    estimate = estimate_client_overhead(observations, "qwen3.5-9b-heretic")
    assert estimate is not None
    assert estimate.tokens == 7_000
    assert estimate_client_overhead(observations, "not-observed") is None


def test_overhead_empty_store_is_none() -> None:
    assert estimate_client_overhead([], "qwen3.5-9b-heretic") is None


# --- typical output --------------------------------------------------------------------------


def test_typical_output_is_median_of_matching_records() -> None:
    observations = [
        _obs(input_tokens=1, user_tokens=0, output=400),
        _obs(input_tokens=1, user_tokens=0, output=900),
        _obs(input_tokens=1, user_tokens=0, output=1600),
        _obs(input_tokens=1, user_tokens=0, output=None),  # unreported: excluded
        _obs(input_tokens=1, user_tokens=0, output=99_999, model="other:latest"),
    ]
    assert estimate_typical_output(observations, "qwen3.5-9b-heretic") == 900
    assert estimate_typical_output([], "qwen3.5-9b-heretic") is None


def test_a_record_from_the_wrong_vocabulary_is_not_learned_from(tmp_path: Path) -> None:
    """Overhead is input_tokens minus user_tokens; the two have to be commensurable.

    input_exact says the MINUEND came from the backend. The subtrahend comes from the one
    tokenizer the proxy has bound, so a request naming a different model produces an exact
    total minus a count in another vocabulary. Subtracting those yields an overhead that is
    confidently wrong, which is worse than a smaller sample.
    """
    good = _obs(input_tokens=2000, user_tokens=100, user_exact=True)
    wrong = _obs(input_tokens=2000, user_tokens=1900, user_exact=False)

    learned = estimate_client_overhead([good, wrong], "qwen3.5-9b-heretic")

    assert learned is not None
    assert learned.tokens == 1900  # the wrong-vocabulary record would have said 100


def test_records_written_before_the_flag_still_count(tmp_path: Path) -> None:
    """None means "never checked", which is not the same as "checked and failed".

    Treating an old store as untrusted would throw away everyone's calibration on upgrade;
    treating it as trusted would launder an unknown into a claim. It is kept, and the record
    says None rather than True so the distinction survives.
    """
    legacy = _obs(input_tokens=2000, user_tokens=100)
    assert legacy.user_exact is None

    learned = estimate_client_overhead([legacy], "qwen3.5-9b-heretic")
    assert learned is not None and learned.tokens == 1900


def test_the_flag_survives_a_round_trip_through_the_store(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    recorder = ObservationRecorder(path)
    recorder.add(_obs(input_tokens=10, user_tokens=1, user_exact=False))
    recorder._flush_sync()

    assert load_observations(path)[0].user_exact is False


def test_a_pool_of_only_wrong_vocabulary_records_says_so() -> None:
    """Refusing would be unhelpful; an unlabelled number would be the one thing not allowed."""
    only_wrong = [_obs(input_tokens=2000, user_tokens=1900, user_exact=False)]

    estimate = estimate_client_overhead(only_wrong, "qwen3.5-9b-heretic")

    assert estimate is not None and estimate.tokens == 100
    assert "different model's vocabulary" in estimate.provenance


def test_a_clean_pool_carries_no_caveat() -> None:
    """The warning has to mean something, so it must not appear on records that are fine."""
    clean = [_obs(input_tokens=2000, user_tokens=100, user_exact=True)]
    estimate = estimate_client_overhead(clean, "qwen3.5-9b-heretic")
    assert estimate is not None and "vocabulary" not in estimate.provenance


# --- what the agent opens on its own ----------------------------------------------------------


def _conversation(
    *,
    start: float,
    injections: list[int],
    output: int | None = 200,
    typed: int = 50,
    model: str | None = "qwen3.5-9b-heretic:latest",
    agent_shaped: bool = True,
    user_exact: bool | None = None,
) -> list[Observation]:
    """A conversation in which turn i injected exactly ``injections[i]`` tokens of its own.

    Built the way the arithmetic reads it: each request grows by the model's last reply, the
    user's new typing, and whatever the client pulled in unasked.
    """
    common = {"model": model, "agent_shaped": agent_shaped, "user_exact": user_exact}
    records = [
        _obs(input_tokens=2000, user_tokens=100, messages=1, ts=start, output=output, **common)
    ]
    for index, injected in enumerate(injections, start=1):
        previous = records[-1]
        records.append(
            _obs(
                input_tokens=previous.input_tokens + (output or 0) + typed + injected,
                user_tokens=previous.user_tokens + typed,
                messages=previous.messages + 2,
                ts=start + index * 10.0,
                output=output,
                **common,
            )
        )
    return records


def _three(**kw: object) -> list[Observation]:
    """Three conversations injecting 1,000 / 2,000 / 3,000 tokens — median 2,000."""
    return [
        record
        for index, total in enumerate((1000, 2000, 3000))
        for record in _conversation(start=index * 10_000.0, injections=[total], **kw)  # type: ignore[arg-type]
    ]


def test_reads_are_the_growth_that_was_neither_typed_nor_generated() -> None:
    three_alike = [
        record
        for index in range(3)
        for record in _conversation(start=index * 10_000.0, injections=[900, 600])
    ]
    estimate = estimate_agent_reads(three_alike, None)
    assert estimate is not None
    assert estimate.tokens == 1500  # 900 + 600, and nothing of the reply or the typing


def test_reads_are_the_median_across_conversations_not_the_largest() -> None:
    estimate = estimate_agent_reads(_three(), None)
    assert estimate is not None
    assert (estimate.tokens, estimate.low, estimate.high) == (2000, 1000, 3000)
    assert estimate.sessions == 3


def test_two_conversations_are_not_enough_to_learn_from() -> None:
    two = _conversation(start=0.0, injections=[1000]) + _conversation(
        start=10_000.0, injections=[2000]
    )
    assert estimate_agent_reads(two, None) is None


def test_a_conversation_that_only_grew_by_its_own_reply_teaches_nothing() -> None:
    quiet = [
        record
        for index in range(3)
        for record in _conversation(start=index * 10_000.0, injections=[0, 0])
    ]
    assert estimate_agent_reads(quiet, None) is None


def test_bare_pokes_teach_nothing_about_what_an_agent_opens() -> None:
    """The lesson the overhead estimator learned live, applied before it could happen again."""
    assert estimate_agent_reads(_three(agent_shaped=False), None) is None


def test_a_turn_whose_reply_was_never_counted_is_dropped_not_guessed() -> None:
    """With eval_count missing, the reply hides inside the growth and would bill as a read."""
    assert estimate_agent_reads(_three(output=None), None) is None


def test_a_pair_mixing_an_exact_input_with_an_estimated_one_is_dropped() -> None:
    mixed = [
        replace(obs, input_exact=index % 2 == 0)
        for index, obs in enumerate(_three())
    ]
    assert estimate_agent_reads(mixed, None) is None


def test_a_pair_counted_in_another_models_vocabulary_is_dropped() -> None:
    assert estimate_agent_reads(_three(user_exact=False), None) is None


def test_records_predating_the_vocabulary_flag_are_still_learned_from() -> None:
    assert estimate_agent_reads(_three(user_exact=None), None) is not None


def test_silence_longer_than_the_window_starts_a_new_conversation() -> None:
    """Two turns an hour apart are two tasks, and neither of them is a pair."""
    stretched = [replace(obs, ts=obs.ts * 1000.0) for obs in _three()]
    assert estimate_agent_reads(stretched, None) is None


def test_a_reply_that_is_not_replayed_reads_as_zero_never_as_negative() -> None:
    """A thinking model drops its reasoning from the history: we over-subtract, and clamp."""
    thinking = [
        record
        for index in range(3)
        for record in _conversation(start=index * 10_000.0, injections=[-5000], output=6000)
    ]
    assert estimate_agent_reads(thinking, None) is None


def test_reads_filter_by_model_like_every_other_estimator() -> None:
    other = _three(model="llama3:8b")
    assert estimate_agent_reads(other, "qwen3.5-9b-heretic") is None
    assert estimate_agent_reads(other, "llama3:8b") is not None


def test_a_learned_estimate_knows_it_was_learned() -> None:
    estimate = estimate_agent_reads(_three(), None)
    assert estimate is not None and estimate.learned
