"""Semantic grouping: the parsing, the checks that reject a proposal, and the fallback.

The feature is one model call, and its entire justification is that nothing it returns is
trusted. So these tests are mostly about *refusing* proposals — a dropped file, an invented
one, a group that does not fit, a part count that ran away — and about the plan still being
produced, correctly, every time one is refused.

Counting is the deterministic chars/4 heuristic here (a model name that resolves no GGUF), and
every plan is built against an explicit ``target``, so the arithmetic below is reproducible on
any machine with no backend running.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pharos.accountant import Accountant
from pharos.config import PharosConfig
from pharos.paths import normalise_display
from pharos.preflight import semantic, split
from pharos.preflight.check import run_check
from pharos.preflight.cli import main as check_main
from pharos.preflight.semantic import (
    FileBrief,
    Group,
    GroupRequest,
    Proposal,
    ask,
    brief,
    parse,
    request_payload,
    validate,
)
from pharos.preflight.split import Grouping, SplitMode, build_plan
from pharos.profiler.types import BackendInfo, EnvironmentProfile, GpuInfo

BS = chr(92)  # a literal backslash, kept out of every string below
_FILES = ("alpha.py", "beta.py", "gamma.py", "delta.py", "epsilon.py", "zeta.py")
_TARGET = 3000


# ----------------------------------------------------------------------------- fixtures


def _config(root: Path, **overrides: object) -> PharosConfig:
    base: dict[str, object] = {
        "model": "test-model-not-installed",  # resolves no GGUF -> heuristic counting
        "target_folder": str(root),
        "observations_file": str(root / "obs.json"),
        "response_reserve": 1024,
    }
    base.update(overrides)
    return PharosConfig(**base)  # type: ignore[arg-type]


def _profile(root: Path) -> EnvironmentProfile:
    gpu = GpuInfo(available=True, name="RTX 3060", total_mib=12288, used_mib=11000, free_mib=1288)
    backend = BackendInfo(
        reachable=True,
        base_url="http://localhost:11434",
        model="test-model-not-installed",
        advertised_max_ctx=262144,
        loaded_ctx=8192,
    )
    return EnvironmentProfile(
        gpu=gpu,
        backend=backend,
        budget=Accountant(_config(root)).report(loaded_ctx=8192, gpu=gpu),
        ctx_mismatch=False,
        ctx_mismatch_ratio=None,
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Six files of ~1,000 heuristic tokens each: two fit a part, three do not."""
    for name in _FILES:
        (tmp_path / name).write_text("x" * 4000, encoding="utf-8")
    return tmp_path


def _prompt() -> str:
    return "Refactor " + ", ".join(_FILES) + " to share one settings object."


async def _plan(root: Path, *, semantic_on: bool = True, **overrides: object):
    config = _config(root, **overrides)
    prompt = _prompt()
    report = await run_check(config, prompt, profile=_profile(root))
    return build_plan(config, prompt, report, target=_TARGET, semantic=semantic_on)


def _reply(groups: list[tuple[str, list[str]]]) -> str:
    return json.dumps({"groups": [{"title": t, "files": f} for t, f in groups]})


def _answers(reply: str | None, error: str | None = None):
    """Stand in for the backend call, recording that it was made."""
    calls: list[GroupRequest] = []

    def _ask(config: PharosConfig, request: GroupRequest, model: str):
        calls.append(request)
        return reply, error

    return _ask, calls


# -------------------------------------------------------------------------------- parse


def test_parse_reads_clean_json() -> None:
    proposal, error = parse(_reply([("the model", ["a.py"]), ("the view", ["b.py"])]))
    assert error is None
    assert proposal is not None
    assert [g.title for g in proposal.groups] == ["the model", "the view"]
    assert proposal.names == ["a.py", "b.py"]


def test_parse_survives_a_fenced_reply() -> None:
    """`format` should prevent this, but a backend that ignores it must not break the plan."""
    body = _reply([("core", ["a.py"])])
    proposal, error = parse(f"Sure! Here you go:\n```json\n{body}\n```\nHope that helps.")
    assert error is None
    assert proposal is not None
    assert proposal.names == ["a.py"]


def test_parse_matches_braces_inside_strings() -> None:
    raw = json.dumps({"groups": [{"title": "the {weird} one", "files": ["a}.py"]}]})
    proposal, error = parse(raw + " trailing prose }")
    assert error is None
    assert proposal is not None
    assert proposal.groups[0].title == "the {weird} one"
    assert proposal.names == ["a}.py"]


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("no json at all here", "not JSON"),
        ("{not valid json", "not JSON"),          # never closes: no object to read
        ('{"groups": [}', "not valid JSON"),      # closes, and is still not JSON
        ("[1, 2, 3]", "not JSON"),  # no object anywhere
        ('{"groups": []}', "no groups"),
        ('{"groups": "nope"}', "no groups"),
        ('{"groups": ["a"]}', "not an object"),
        ('{"groups": [{"title": "t"}]}', "listed no files"),
        ('{"groups": [{"title": "t", "files": [3]}]}', "not a filename"),
    ],
)
def test_parse_rejects(text: str, fragment: str) -> None:
    proposal, error = parse(text)
    assert proposal is None
    assert error is not None and fragment in error


def test_parse_cleans_the_title() -> None:
    raw = json.dumps(
        {
            "groups": [
                {"title": "  the\n  data   layer .", "files": ["a.py"]},
                {"title": "x" * 200, "files": ["b.py"]},
                {"title": 42, "files": ["c.py"]},
            ]
        }
    )
    proposal, error = parse(raw)
    assert error is None
    assert proposal is not None
    assert proposal.groups[0].title == "the data layer"
    assert len(proposal.groups[1].title) == semantic._TITLE_LIMIT
    assert proposal.groups[1].title.endswith("…")
    assert proposal.groups[2].title == ""  # a non-string title is dropped, not stringified


# ----------------------------------------------------------------------------- validate


def _request(**overrides: object) -> GroupRequest:
    base: dict[str, object] = {
        "task": "do the thing",
        "files": tuple(FileBrief(n, 100) for n in ("a.py", "b.py", "c.py")),
        "target_parts": 2,
        "max_parts": 4,
        "max_files": None,
        "per_part_budget": 250,
    }
    base.update(overrides)
    return GroupRequest(**base)  # type: ignore[arg-type]


def _proposal(*groups: list[str]) -> Proposal:
    return Proposal(groups=tuple(Group(title="", files=tuple(g)) for g in groups))


def test_validate_accepts_an_exact_partition() -> None:
    assert validate(_proposal(["a.py", "b.py"], ["c.py"]), _request()) is None


def test_validate_accepts_a_reordering() -> None:
    """Order is the model's to choose; only membership is checked against the scope."""
    assert validate(_proposal(["c.py"], ["b.py", "a.py"]), _request()) is None


def test_validate_rejects_a_dropped_file() -> None:
    reason = validate(_proposal(["a.py", "b.py"]), _request())
    assert reason is not None
    assert "dropped" in reason and "c.py" in reason


def test_validate_rejects_an_invented_file() -> None:
    reason = validate(_proposal(["a.py", "b.py"], ["c.py", "nowhere.py"]), _request())
    assert reason is not None
    assert "invented" in reason and "nowhere.py" in reason


def test_validate_rejects_a_repeated_file() -> None:
    reason = validate(_proposal(["a.py", "b.py"], ["b.py", "c.py"]), _request())
    assert reason is not None
    assert "repeated" in reason and "b.py" in reason


def test_validate_rejects_an_empty_group() -> None:
    reason = validate(_proposal(["a.py", "b.py", "c.py"], []), _request())
    assert reason is not None and "empty part" in reason


def test_validate_rejects_too_many_parts() -> None:
    reason = validate(_proposal(["a.py"], ["b.py"], ["c.py"]), _request(max_parts=2))
    assert reason is not None
    assert "3 parts against a ceiling of 2" in reason


def test_validate_ignores_size() -> None:
    """A group that is right and too big is the packer's problem, not a rejection."""
    assert validate(_proposal(["a.py", "b.py"], ["c.py"]), _request(max_files=1)) is None


def test_validate_names_at_most_three_offenders() -> None:
    files = tuple(FileBrief(f"f{i}.py", 10) for i in range(9))
    reason = validate(_proposal(["f0.py"]), _request(files=files, max_parts=9))
    assert reason is not None
    assert "and 5 more" in reason  # 8 dropped, 3 named


# ---------------------------------------------------------------------------------- ask


def _transport(handler) -> type[httpx.Client]:
    class _Client(httpx.Client):
        def __init__(self, **kwargs: object) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(**kwargs)  # type: ignore[arg-type]

    return _Client


def test_ask_returns_the_reply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"groups": []}'}})

    monkeypatch.setattr(semantic.httpx, "Client", _transport(handler))
    reply, error = ask(_config(tmp_path), _request(), "m")
    assert error is None
    assert reply == '{"groups": []}'
    # Determinism is a promise the module makes in its docstring; hold it to it.
    assert seen["options"] == {"temperature": 0.0, "seed": 0, "num_predict": 256 + 48 * 3}
    assert seen["stream"] is False
    # A thinking model that reasons for its whole budget answers nothing at all.
    assert seen["think"] is False
    assert seen["format"]["required"] == ["groups"]


def test_ask_sends_names_and_counts_but_never_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(200, json={"message": {"content": "{}"}})

    monkeypatch.setattr(semantic.httpx, "Client", _transport(handler))
    ask(_config(tmp_path), _request(), "m")
    body = sent[0]
    assert "a.py (100 tokens)" in body
    assert "do the thing" in body
    assert "xxxx" not in body  # no file bytes travel with the question


@pytest.mark.parametrize(
    ("response", "fragment"),
    [
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, json={"error": "model not found"}), "model not found"),
        (httpx.Response(200, json={"message": {"content": "  "}}), "empty reply"),
        (
            httpx.Response(200, json={"message": {"content": ""}, "done_reason": "length"}),
            "hit its length limit",
        ),
        (
            httpx.Response(
                200,
                json={"message": {"content": ""}, "done_reason": "length", "eval_count": 1024},
            ),
            "after 1,024 tokens",
        ),
        (
            httpx.Response(
                200, json={"message": {"content": '{\"grou'}, "done_reason": "length"}
            ),
            "cut off at its length limit",
        ),
        (httpx.Response(200, json={"nope": 1}), "empty reply"),
        (httpx.Response(200, content=b"not json"), "JSONDecodeError"),
    ],
)
def test_ask_turns_every_backend_failure_into_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: httpx.Response, fragment: str
) -> None:
    monkeypatch.setattr(semantic.httpx, "Client", _transport(lambda _: response))
    reply, error = ask(_config(tmp_path), _request(), "m")
    assert reply is None
    assert error is not None and fragment in error


def test_ask_survives_a_refused_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(semantic.httpx, "Client", _transport(handler))
    reply, error = ask(_config(tmp_path), _request(), "m")
    assert reply is None
    assert error is not None and "ConnectError" in error


# -------------------------------------------------------------------- plan integration


@pytest.mark.anyio
async def test_position_grouping_is_the_default(tree: Path) -> None:
    """No --semantic, no model call, no note: the previous behaviour, byte for byte."""
    plan = await _plan(tree, semantic_on=False)
    assert plan.mode is SplitMode.SCOPE
    assert plan.grouping is Grouping.POSITION
    assert plan.grouping_note is None
    assert all(part.title == "" for part in plan.parts)


@pytest.mark.anyio
async def test_a_good_proposal_regroups_the_parts(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = await _plan(tree, semantic_on=False)
    assert [len(p.files) for p in baseline.parts] == [2, 2, 2]
    assert [f.display for f in baseline.parts[0].files] == ["alpha.py", "beta.py"]

    ask_fn, calls = _answers(
        _reply(
            [
                ("the settings object", ["alpha.py", "zeta.py"]),
                ("its two readers", ["beta.py", "epsilon.py"]),
                ("the leftovers", ["gamma.py", "delta.py"]),
            ]
        )
    )
    monkeypatch.setattr(split, "ask", ask_fn)
    plan = await _plan(tree)

    assert plan.grouping is Grouping.SEMANTIC
    assert [f.display for f in plan.parts[0].files] == ["alpha.py", "zeta.py"]
    assert [p.title for p in plan.parts] == [
        "the settings object",
        "its two readers",
        "the leftovers",
    ]
    assert plan.ok and all(p.fits for p in plan.parts)
    # The question carried the mechanical answer's shape, so the model knows what to beat.
    assert calls[0].target_parts == 3
    assert calls[0].max_parts == 5  # 3 + semantic_max_extra_parts


@pytest.mark.anyio
async def test_the_title_reaches_the_part_and_is_paid_for(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model's free text in every part is still text in every part."""
    title = "the configuration surface"
    groups = [(title, ["alpha.py", "beta.py"]), ("", ["gamma.py", "delta.py"])]
    groups.append(("", ["epsilon.py", "zeta.py"]))
    monkeypatch.setattr(split, "ask", _answers(_reply(groups))[0])
    plan = await _plan(tree)

    assert title in plan.parts[0].body
    assert plan.parts[0].body.startswith(f"[Pharos] Part 1 of 3 — {title} —")
    assert plan.parts[1].body.startswith("[Pharos] Part 2 of 3 — this task")
    # Same files, same reserve: the projection gap is the title's own cost, not a rounding.
    assert plan.parts[0].projected_tokens > plan.parts[1].projected_tokens - 500


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reply", "error", "fragment"),
    [
        (None, "ConnectError: refused", "the backend could not be asked"),
        ("sorry, I cannot help with that", None, "was not JSON"),
        (_reply([("all of it", list(_FILES[:5]))]), None, "dropped"),
        (_reply([("invented", [*_FILES, "ghost.py"])]), None, "invented"),
        (_reply([(f"p{i}", [f]) for i, f in enumerate(_FILES)]), None, "against a ceiling of 5"),
    ],
)
async def test_every_refusal_falls_back_and_says_why(
    tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    reply: str | None,
    error: str | None,
    fragment: str,
) -> None:
    monkeypatch.setattr(split, "ask", _answers(reply, error)[0])
    plan = await _plan(tree)

    assert plan.grouping is Grouping.POSITION
    assert plan.grouping_note is not None
    assert fragment in plan.grouping_note
    # The note is the reason alone: the winning grouping is a field of its own, and a label
    # beside it in every renderer, so repeating it here said everything twice.
    assert "grouped by position" not in plan.grouping_note
    assert plan.grouping is Grouping.POSITION
    # The point of the fallback: the plan is exactly as good as it would have been.
    baseline = await _plan(tree, semantic_on=False)
    assert plan.ok
    assert [[f.display for f in p.files] for p in plan.parts] == [
        [f.display for f in p.files] for p in baseline.parts
    ]


@pytest.mark.anyio
async def test_a_bad_proposal_never_reaches_a_part_body(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected proposal's titles must not leak into the plan it did not produce."""
    monkeypatch.setattr(
        split, "ask", _answers(_reply([("SHOULD NOT APPEAR", list(_FILES[:5]))]))[0]
    )
    plan = await _plan(tree)
    assert all("SHOULD NOT APPEAR" not in part.body for part in plan.parts)
    assert all(part.title == "" for part in plan.parts)


@pytest.mark.anyio
async def test_no_model_configured_declines_without_calling(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ask_fn, calls = _answers(_reply([("x", list(_FILES))]))
    monkeypatch.setattr(split, "ask", ask_fn)
    plan = await _plan(tree, model=None)
    assert not calls
    assert plan.grouping is Grouping.POSITION
    assert plan.grouping_note is not None and "needs a model in pharos.toml" in plan.grouping_note


@pytest.mark.anyio
async def test_sliced_files_decline_without_calling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Line ranges have an order the file already fixed; a model can only get it wrong."""
    (tmp_path / "huge.py").write_text("line\n" * 4000, encoding="utf-8")
    (tmp_path / "small.py").write_text("y" * 400, encoding="utf-8")
    ask_fn, calls = _answers(_reply([("x", ["huge.py"])]))
    monkeypatch.setattr(split, "ask", ask_fn)

    config = _config(tmp_path)
    prompt = "Rewrite huge.py and small.py."
    report = await run_check(config, prompt, profile=_profile(tmp_path))
    plan = build_plan(config, prompt, report, target=_TARGET, semantic=True)

    assert any(f.is_slice for part in plan.parts for f in part.files)
    assert not calls
    assert plan.grouping is Grouping.POSITION
    assert plan.grouping_note is not None and "line ranges" in plan.grouping_note


@pytest.mark.anyio
async def test_a_text_split_says_semantic_does_not_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ask_fn, calls = _answers(_reply([("x", ["a.py"])]))
    monkeypatch.setattr(split, "ask", ask_fn)
    config = _config(tmp_path)
    prompt = "Summarise this log:\n\n" + "\n\n".join(f"event {i} " + "z" * 200 for i in range(80))
    report = await run_check(config, prompt, profile=_profile(tmp_path))
    plan = build_plan(config, prompt, report, target=_TARGET, semantic=True)

    assert plan.mode is SplitMode.TEXT
    assert not calls
    assert plan.grouping is Grouping.POSITION
    assert plan.grouping_note is not None and "does not apply to a text split" in plan.grouping_note


@pytest.mark.anyio
async def test_an_oversized_group_is_split_not_rejected(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three files of ~1,000 tokens exceed a part. The concern survives; the group is cut."""
    reply = _reply([("physics", list(_FILES[:3])), ("audio", list(_FILES[3:]))])
    monkeypatch.setattr(split, "ask", _answers(reply)[0])
    plan = await _plan(tree)

    assert plan.grouping is Grouping.SEMANTIC
    assert plan.grouping_note is not None
    assert "2 groups were too big for one part and were split in order." in plan.grouping_note
    # The model's order is preserved and no file crosses a concern boundary.
    assert [[f.display for f in p.files] for p in plan.parts] == [
        ["alpha.py", "beta.py"],
        ["gamma.py"],
        ["delta.py", "epsilon.py"],
        ["zeta.py"],
    ]
    assert [p.title for p in plan.parts] == [
        "physics (1 of 2)",
        "physics (2 of 2)",
        "audio (1 of 2)",
        "audio (2 of 2)",
    ]
    assert plan.ok


@pytest.mark.anyio
async def test_the_file_cap_splits_a_group_rather_than_losing_it(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pharos run` caps files per part for a reason the token budget cannot see."""
    reply = _reply([("too much", list(_FILES[:3])), ("rest", list(_FILES[3:]))])
    monkeypatch.setattr(split, "ask", _answers(reply)[0])
    config = _config(tree)
    prompt = _prompt()
    report = await run_check(config, prompt, profile=_profile(tree))
    plan = build_plan(config, prompt, report, target=_TARGET, max_files=2, semantic=True)

    assert plan.grouping is Grouping.SEMANTIC
    assert all(len(p.files) <= 2 for p in plan.parts)
    assert [[f.display for f in p.files] for p in plan.parts] == [
        ["alpha.py", "beta.py"],
        ["gamma.py"],
        ["delta.py", "epsilon.py"],
        ["zeta.py"],
    ]


@pytest.mark.anyio
async def test_repair_that_overruns_the_part_ceiling_is_rejected(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splitting buys parts, and the ceiling still has the last word.

    Position packs these six files 2/2/2, so with no slack the ceiling is 3. A 3-file group
    needs two parts on its own, which takes the repaired plan to four.
    """
    reply = _reply(
        [("three", list(_FILES[:3])), ("two", list(_FILES[3:5])), ("one", [_FILES[5]])]
    )
    monkeypatch.setattr(split, "ask", _answers(reply)[0])
    plan = await _plan(tree, semantic_max_extra_parts=0)

    assert plan.grouping is Grouping.POSITION
    assert plan.grouping_note is not None
    assert "keeping its groups intact needs 4 parts against a ceiling of 3" in plan.grouping_note


@pytest.mark.anyio
async def test_extra_parts_are_allowed_within_the_slack(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grouping by meaning packs worse than first-fit. Some of that is the price, not a bug."""
    groups = [(f"concern {i}", [f, g]) for i, (f, g) in enumerate([_FILES[:2], _FILES[2:4]])]
    groups += [("odd one out", [_FILES[4]]), ("the last", [_FILES[5]])]
    monkeypatch.setattr(split, "ask", _answers(_reply(groups))[0])
    plan = await _plan(tree)

    assert plan.grouping is Grouping.SEMANTIC
    assert len(plan.parts) == 4  # position gave 3; the config allows 3 + 2
    assert plan.grouping_note is not None
    assert "4 parts where position packing gave 3" in plan.grouping_note


@pytest.mark.anyio
async def test_the_slack_is_configurable_down_to_nothing(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    groups = [(f"concern {i}", [f, g]) for i, (f, g) in enumerate([_FILES[:2], _FILES[2:4]])]
    groups += [("odd one out", [_FILES[4]]), ("the last", [_FILES[5]])]
    monkeypatch.setattr(split, "ask", _answers(_reply(groups))[0])
    plan = await _plan(tree, semantic_max_extra_parts=0)

    assert plan.grouping is Grouping.POSITION
    assert plan.grouping_note is not None and "against a ceiling of 3" in plan.grouping_note


@pytest.mark.anyio
async def test_deferred_lists_follow_the_semantic_grouping(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scope block is the whole projection. Regrouping must move it too, or it lies."""
    monkeypatch.setattr(
        split,
        "ask",
        _answers(
            _reply(
                [
                    ("one", ["zeta.py", "alpha.py"]),
                    ("two", ["epsilon.py", "beta.py"]),
                    ("three", ["delta.py", "gamma.py"]),
                ]
            )
        )[0],
    )
    plan = await _plan(tree)
    body = plan.parts[0].body
    in_scope, out_of_scope = body.split("OUT OF SCOPE", 1)
    assert "zeta.py" in in_scope and "alpha.py" in in_scope
    assert "beta.py" not in in_scope
    for deferred in ("beta.py", "gamma.py", "delta.py", "epsilon.py"):
        assert deferred in out_of_scope


# ----------------------------------------------------------------------------- excerpts


def test_brief_keeps_the_first_lines_stripped_and_capped() -> None:
    text = "\n".join(["", "# header", "   indented()", "", "x" * 300, "d", "e", "f", "g"])
    got = brief("a.py", 10, text)
    parts = got.excerpt.split(" / ")
    assert len(parts) == semantic._EXCERPT_LINES
    assert parts[0] == "# header"
    assert parts[1] == "indented()"  # blank lines skipped, indentation dropped
    assert len(parts[2]) == semantic._EXCERPT_LINE_CHARS  # the long line is cut, not dropped
    assert parts[4] == "e"


def test_brief_without_text_is_name_only() -> None:
    assert brief("a.py", 10, None).excerpt == ""
    assert brief("a.py", 10, "").excerpt == ""
    assert brief("a.py", 10, "\n\n  \n").excerpt == ""


def test_the_listing_carries_excerpts(tmp_path: Path) -> None:
    files = (FileBrief("a.py", 10, "# render subsystem"), FileBrief("b.py", 20))
    request = _request(files=files)
    body = request_payload(_config(tmp_path), request, "m")["messages"][0]["content"]
    assert "- a.py (10 tokens)\n    # render subsystem" in body
    assert "- b.py (20 tokens)" in body
    assert "There are 2 files" in body  # the sentence that fixed coverage


def test_a_huge_excerpt_budget_drops_them_all_at_once(tmp_path: Path) -> None:
    """All or none: a grouping that changes with a file's position in the list is not one."""
    big = "QQQ" * (semantic._EXCERPT_TOTAL_CHARS // 2)
    files = tuple(FileBrief(f"f{i}.py", 10, big) for i in range(3))
    payload = request_payload(_config(tmp_path), _request(files=files), "m")
    body = payload["messages"][0]["content"]
    assert "QQQ" not in body
    for i in range(3):
        assert f"- f{i}.py (10 tokens)" in body


def test_no_file_cap_states_no_rule(tmp_path: Path) -> None:
    body = request_payload(_config(tmp_path), _request(max_files=None), "m")["messages"][0][
        "content"
    ]
    assert "file(s) in one part" not in body
    capped = request_payload(_config(tmp_path), _request(max_files=2), "m")["messages"][0][
        "content"
    ]
    assert "Put at most 2 file(s) in one part." in capped


@pytest.mark.anyio
async def test_the_head_of_each_file_reaches_the_question(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tree / "alpha.py").write_text("# the alpha concern\n" + "x" * 4000, encoding="utf-8")
    ask_fn, calls = _answers(_reply([("x", list(_FILES))]))
    monkeypatch.setattr(split, "ask", ask_fn)
    await _plan(tree)
    briefs = {f.display: f.excerpt for f in calls[0].files}
    assert briefs["alpha.py"].startswith("# the alpha concern")
    assert all(name in briefs for name in _FILES)


@pytest.mark.anyio
async def test_an_unreadable_file_still_reaches_the_question(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Name only, rather than dropped — or the checks would blame the model for our omission."""
    def explode(path: Path):
        raise OSError("gone")

    monkeypatch.setattr(split, "read_countable", explode)
    ask_fn, calls = _answers(_reply([("x", list(_FILES))]))
    monkeypatch.setattr(split, "ask", ask_fn)
    await _plan(tree)
    assert [f.display for f in calls[0].files] == list(_FILES)
    assert all(f.excerpt == "" for f in calls[0].files)


@pytest.mark.anyio
async def test_the_configured_grouper_model_is_the_one_asked(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grouping and doing the work are different jobs; see PharosConfig.semantic_model."""
    seen: list[str] = []

    def _ask(config: PharosConfig, request: GroupRequest, model: str):
        seen.append(model)
        return _reply([("x", list(_FILES))]), None

    monkeypatch.setattr(split, "ask", _ask)
    await _plan(tree, semantic_model="a-small-fast-one")
    assert seen == ["a-small-fast-one"]
    await _plan(tree)
    assert seen[-1] == "test-model-not-installed"  # falls back to the run's own model


def test_the_reply_budget_scales_but_is_capped() -> None:
    assert semantic.reply_budget(1) == 256 + 48
    assert semantic.reply_budget(6) == 256 + 48 * 6
    # Linear in the file count, and the file count has no ceiling of its own.
    assert semantic.reply_budget(10_000) == semantic._REPLY_MAX_TOKENS


@pytest.mark.anyio
async def test_no_split_does_not_pay_for_a_grouping(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-split throws the parts away; arranging them first is a round trip for a footnote."""
    from pharos.agent import runner

    config = _config(tree)
    prompt = _prompt()
    report = await run_check(config, prompt, profile=_profile(tree))

    async def _check(cfg: object, p: object, **kw: object):
        return report

    async def _window(cfg: object, p: object, rep: object, say: object):
        return report

    monkeypatch.setattr(runner, "run_check", _check)
    monkeypatch.setattr(runner, "_ensure_window", _window)
    ask_fn, calls = _answers(_reply([("x", list(_FILES))]))
    monkeypatch.setattr(split, "ask", ask_fn)

    seen: list[bool] = []
    real = runner.build_plan

    def spy(*args: object, **kwargs: object):
        seen.append(bool(kwargs.get("semantic")))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner, "build_plan", spy)

    await runner.run_task(
        config, prompt, dry_run=True, use_git=False, semantic=True, divide=False
    )
    assert seen == [False], "--no-split must not ask"
    assert not calls

    await runner.run_task(
        config, prompt, dry_run=True, use_git=False, semantic=True, divide=True
    )
    assert seen == [False, True], "dividing for real must ask"


@pytest.mark.anyio
async def test_an_untitled_group_is_not_numbered_into_existence(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank title numbered as " (1 of 2)" is truthy, and the renderer would print it."""
    monkeypatch.setattr(
        split, "ask", _answers(_reply([("", list(_FILES[:3])), ("", list(_FILES[3:]))]))[0]
    )
    plan = await _plan(tree)
    assert plan.grouping is Grouping.SEMANTIC
    assert len(plan.parts) == 4  # both groups were split, so numbering did apply
    assert [p.title for p in plan.parts] == ["", "", "", ""]
    for part in plan.parts:
        assert part.body.startswith(f"[Pharos] Part {part.index} of 4 — this task")


# ---------------------------------------------------------------------------------- CLI


def test_semantic_without_split_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    """An accepted flag that does nothing is worse than a rejected one: it looks like it worked."""
    with pytest.raises(SystemExit) as exit_info:
        check_main(["--semantic", "hello"])
    assert exit_info.value.code == 2
    assert "needs --split" in capsys.readouterr().err


def test_semantic_is_accepted_alongside_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`check --split --semantic` is the same command as `split --semantic`."""
    from pharos.preflight import cli

    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(split, "ask", _answers(None, "offline")[0])
    # No files named, so there is nothing to split and no grouping call -- the flag just parses.
    assert check_main(["--split", "--semantic", "--target", "4000", "hello"]) in (0, 1, 2)

# ------------------------------------------------------------------- path spellings


def test_validate_matches_across_separators() -> None:
    """The scope holds Windows separators; the model answers posix. Same files."""
    files = tuple(FileBrief(n, 100) for n in ("src\\a.py", "src\\b.py"))
    request = _request(files=files)
    assert validate(_proposal(["src/a.py"], ["src/b.py"]), request) is None
    assert validate(_proposal(["./src/a.py", "src\\b.py"]), request) is None


def test_validate_matches_case_insensitively_when_unambiguous() -> None:
    request = _request(files=(FileBrief("src/Store.py", 100),))
    assert validate(_proposal(["src/store.py"]), request) is None


def test_validate_still_catches_a_genuinely_invented_path() -> None:
    request = _request(
        files=(FileBrief("src\\a.py", 100), FileBrief("src\\b.py", 100))
    )
    reason = validate(_proposal(["src/a.py"], ["src/ghost.py"]), request)
    assert reason is not None
    assert "invented" in reason and "src/ghost.py" in reason
    assert "dropped" in reason and "src\\b.py" in reason


def test_the_question_asks_in_posix(tmp_path: Path) -> None:
    request = _request(files=(FileBrief("src\\deep\\a.py", 100),))
    body = request_payload(_config(tmp_path), request, "m")["messages"][0]["content"]
    assert "- src/deep/a.py (100 tokens)" in body
    assert "\\" not in body


@pytest.mark.anyio
async def test_a_nested_tree_groups_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this block exists to prevent: every file dropped AND invented, on Windows only.

    Every other fixture in this file is flat, so "alpha.py" has no separator to disagree about
    and the whole feature looked healthy while being unusable on any real project.
    """
    (tmp_path / "src").mkdir()
    names = [f"src/mod{i}.py" for i in range(6)]
    for name in names:
        (tmp_path / name).write_text("x" * 4000, encoding="utf-8")

    # The model answers in posix, which is NOT how the report spells them on Windows.
    monkeypatch.setattr(split, "ask", _answers(_reply([("a", names[:3]), ("b", names[3:])]))[0])
    config = _config(tmp_path)
    prompt = "Refactor " + ", ".join(names) + " to share one settings object."
    report = await run_check(config, prompt, profile=_profile(tmp_path))
    plan = build_plan(config, prompt, report, target=_TARGET, semantic=True)

    assert plan.grouping is Grouping.SEMANTIC, plan.grouping_note
    assert plan.ok
    grouped = [normalise_display(f.display) for part in plan.parts for f in part.files]
    assert sorted(grouped) == sorted(names)
