"""End-to-end: a real request through the proxy, whose observations drive a real pre-flight.

Everything else in this suite tests one seam. This file tests the loop the product actually
makes — Observe teaches Warn teaches Divide:

    a client sends a fat request through the proxy (respx-mocked backend)
        -> the proxy records an observation (counts only)
            -> `pharos check` learns the client overhead from it
                -> the floor lands over budget
                    -> `pharos split` cuts it into parts
                        -> re-checking a part's own text proves the part fits

That last step is the one that matters: the splitter's projection and the checker's floor are
computed by different code paths, and if they disagree the tool lies in exactly the situation
it exists for. So the parts are fed back through `run_check` and compared.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pharos.accountant import Accountant
from pharos.calibration import load_observations
from pharos.config import PharosConfig
from pharos.preflight.check import run_check
from pharos.preflight.cli import main as check_main
from pharos.preflight.cli import split_main
from pharos.preflight.split import SplitMode, build_plan
from pharos.profiler.types import BackendInfo, EnvironmentProfile, GpuInfo

BASE = "http://localhost:11434"

# The shape of a real coding-agent request: a system prompt and a tool catalogue the user
# never typed, plus a short user turn. That gap is what `client overhead` measures.
SYSTEM_PROMPT = "You are a coding agent. " + "Follow the project conventions carefully. " * 40
TOOL_CATALOGUE = [
    {
        "type": "function",
        "function": {
            "name": f"tool_{i}",
            "description": "Does a thing to a file. " * 12,
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
    for i in range(6)
]
USER_TURN = "fix the bug"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small project: four ~750-token modules under src/, plus a notebook and a decoy."""
    (tmp_path / "src").mkdir()
    for name in ("alpha.py", "beta.py", "gamma.py", "delta.py"):
        (tmp_path / "src" / name).write_text(("z = 3\n" * 500)[:3000], encoding="utf-8")
    (tmp_path / "src" / "notes.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": ["import os\n", "print(os.getcwd())\n"]},
                    {
                        "cell_type": "code",
                        "source": ["plot()\n"],
                        # A real notebook's bulk is output, not source: base64 image data.
                        "outputs": [{"data": {"image/png": "iVBORw0KGgo" + "A" * 20000}}],
                    },
                ],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "alpha.py").write_text("# a decoy with a colliding name\n", "utf-8")
    return tmp_path


def _config(root: Path, **overrides: object) -> PharosConfig:
    base: dict[str, object] = {
        "model": "test-model-not-installed",  # resolves no GGUF -> heuristic chars/4
        "target_folder": str(root),
        "observations_file": str(root / "obs.json"),
        "response_reserve": 1024,
    }
    base.update(overrides)
    return PharosConfig(**base)  # type: ignore[arg-type]


def _profile(root: Path, loaded_ctx: int) -> EnvironmentProfile:
    gpu = GpuInfo(available=True, name="RTX 3060", total_mib=12288, used_mib=11000, free_mib=1288)
    return EnvironmentProfile(
        gpu=gpu,
        backend=BackendInfo(
            reachable=True,
            base_url=BASE,
            model="test-model-not-installed",
            advertised_max_ctx=262144,
            loaded_ctx=loaded_ctx,
        ),
        budget=Accountant(_config(root)).report(loaded_ctx=loaded_ctx, gpu=gpu),
        ctx_mismatch=True,
        ctx_mismatch_ratio=0.125,
    )


async def _observe(respx_mock, make_proxy, project: Path) -> None:
    """Send one agent-shaped request through the proxy so an observation lands on disk."""
    respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "test-model-not-installed",
                "message": {"role": "assistant", "content": "done"},
                "done": True,
                "prompt_eval_count": 4321,  # the backend's own truth: reconciles to exact
                "eval_count": 180,
            },
        )
    )
    client = make_proxy(config=_config(project), record_observations=True)
    response = await client.post(
        "/api/chat",
        json={
            "model": "test-model-not-installed",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TURN},
            ],
            "tools": TOOL_CATALOGUE,
            "stream": False,
        },
    )
    assert response.status_code == 200
    await client._transport.app.state.pharos.aclose()  # type: ignore[attr-defined]


async def test_observed_traffic_teaches_the_preflight(respx_mock, make_proxy, project) -> None:
    await _observe(respx_mock, make_proxy, project)

    observations = load_observations(project / "obs.json")
    assert len(observations) == 1
    assert observations[0].input_exact  # reconciled against prompt_eval_count
    assert observations[0].input_tokens == 4321

    report = await run_check(
        _config(project), "fix the bug", profile=_profile(project, loaded_ctx=32768)
    )
    assert report.overhead is not None
    # The overhead is the gap between what the backend counted and what the user typed —
    # learned, not assumed, and it dwarfs the prompt itself.
    assert report.overhead.tokens > 4000
    assert "learned" in report.overhead.provenance
    assert report.floor == report.prompt_tokens + report.overhead.tokens


async def test_learned_overhead_turns_a_small_prompt_into_a_split(
    respx_mock, make_proxy, project
) -> None:
    """The headline case: 8 typed words that do not fit, and parts that do."""
    await _observe(respx_mock, make_proxy, project)
    config = _config(project)
    prompt = "Refactor everything in `src/` for consistency."

    report = await run_check(config, prompt, profile=_profile(project, loaded_ctx=8192))
    assert report.prompt_tokens < 20  # what the user actually typed
    assert report.ceiling > report.floor  # the directory is the rest of the story

    plan = build_plan(config, prompt, report)
    assert plan.mode is SplitMode.SCOPE
    assert plan.ok
    assert len(plan.parts) > 1

    # Every file the directory contributed is covered; a file too big for one part appears as
    # several line ranges, so compare the set of files, not the count of units.
    scoped = {f.display for part in plan.parts for f in part.files}
    assert scoped == {f.display for f in report.directories[0].files}


async def test_each_part_re_checks_as_fitting(respx_mock, make_proxy, project) -> None:
    """Feed every generated part back through the checker: projection vs. floor, independently.

    The splitter says a part costs N. The checker, given that part as a prompt, computes its
    own floor from scratch — different code, same tokenizer. If those disagree, the plan is
    fiction.

    The OUT OF SCOPE block is removed before re-checking, and that is the point rather than a
    convenience: it NAMES the deferred files, so a checker reading the part as prose counts
    every one of them. That is precisely the disobedient case the projection has never claimed
    to cover ("it holds while the agent stays inside the part's scope"). Stripping the block
    models the agent doing as it is told; the cost when it does not is asserted separately
    below, so the size of that exposure is written down rather than assumed away.
    """
    await _observe(respx_mock, make_proxy, project)
    config = _config(project)
    profile = _profile(project, loaded_ctx=8192)
    prompt = "Refactor everything in `src/` for consistency."

    report = await run_check(config, prompt, profile=profile)
    plan = build_plan(config, prompt, report)
    assert plan.parts

    budget = profile.budget.usable_budget
    assert budget is not None
    for part in plan.parts:
        # The part's own text names its in-scope files, so the checker re-derives them.
        recheck = await run_check(config, _obedient(part.body), profile=profile)
        if any(f.is_slice for f in part.files):
            # No comparison is meaningful here. The checker resolves a NAME and counts the
            # whole file, having no way to know the part scoped lines 1-427 of it; meanwhile
            # stripping the deferred block removes body text the projection did count. The two
            # errors pull opposite ways, so any inequality asserted here would be pinning
            # noise. The real claim — what a whole-file reader would cost — is asserted in
            # test_a_sliced_plan_says_what_a_whole_file_reader_would_cost.
            continue
        assert recheck.floor <= part.projected_tokens + plan.handoff_reserve
        assert recheck.floor <= budget
        assert recheck.verdict.value == "fits"


async def test_a_sliced_plan_says_what_a_whole_file_reader_would_cost(
    respx_mock, make_proxy, project
) -> None:
    """Found by testing: `read_file(path)` has no line-range argument on most agents."""
    await _observe(respx_mock, make_proxy, project)
    config = _config(project)
    prompt = "Refactor everything in `src/` for consistency."
    profile = _profile(project, loaded_ctx=4096)  # small enough to force slicing

    report = await run_check(config, prompt, profile=profile)
    plan = build_plan(config, prompt, report)
    if not any(f.is_slice for part in plan.parts for f in part.files):
        pytest.skip("this budget did not force a slice")
    assert any("whole files" in note for note in plan.notes)


def _obedient(body: str) -> str:
    """The part as an agent that respects the scope block would consume it."""
    lines = body.splitlines(keepends=True)
    out, skipping = [], False
    for line in lines:
        if line.startswith("OUT OF SCOPE"):
            skipping = True
        elif skipping and not line.startswith("  - "):
            skipping = False
        if not skipping:
            out.append(line)
    return "".join(out)


async def test_disobeying_the_scope_block_is_what_costs_the_overflow(
    respx_mock, make_proxy, project
) -> None:
    """Quantify the exposure the projection carries, instead of leaving it as a caveat.

    A part read literally — deferred filenames and all — costs far more than its projection.
    That gap IS the scope contract: it is the reason the part says "do not open these", and
    the reason the projection is documented as a floor conditional on obedience.
    """
    await _observe(respx_mock, make_proxy, project)
    config = _config(project)
    profile = _profile(project, loaded_ctx=8192)
    prompt = "Refactor everything in `src/` for consistency."

    plan = build_plan(config, prompt, await run_check(config, prompt, profile=profile))
    first = plan.parts[0]

    literal = await run_check(config, first.body, profile=profile)
    obedient = await run_check(config, _obedient(first.body), profile=profile)

    assert obedient.floor < literal.floor  # reading the deferred list is what blows it up
    assert literal.floor > first.projected_tokens
    # And the deferred NAMES themselves are negligible — it is their contents that cost.
    assert len(literal.files) > len(obedient.files)


async def test_notebook_is_counted_by_cell_source_not_raw_json(project: Path) -> None:
    """A notebook's bulk is base64 output; counting the file would overshoot by orders."""
    raw_chars = (project / "src" / "notes.ipynb").stat().st_size
    report = await run_check(
        _config(project),
        "explain `src/notes.ipynb`",
        profile=_profile(project, loaded_ctx=32768),
    )
    counted = next(f for f in report.files if f.display.endswith("notes.ipynb"))
    assert counted.note is not None and "cells" in counted.note
    assert counted.tokens < 30  # three lines of source
    assert counted.tokens < raw_chars // 4 // 100  # vs. what the raw JSON would have cost


async def test_ambiguous_reference_is_resolvable_and_never_guessed(project: Path) -> None:
    profile = _profile(project, loaded_ctx=32768)
    unresolved = await run_check(_config(project), "fix `alpha.py`", profile=profile)
    assert unresolved.files == []
    assert "alpha.py" in unresolved.extraction.ambiguous  # src/ and tests/ both match

    resolved = await run_check(
        _config(project),
        "fix `alpha.py`",
        profile=profile,
        resolve={"alpha.py": "src/alpha.py"},
    )
    assert [f.display for f in resolved.files] == [str(Path("src/alpha.py"))]
    assert resolved.extraction.ambiguous == {}
    assert resolved.floor > unresolved.floor


async def test_resolve_pointing_at_nothing_is_reported_not_ignored(project: Path) -> None:
    report = await run_check(
        _config(project),
        "fix `alpha.py`",
        profile=_profile(project, loaded_ctx=32768),
        resolve={"alpha.py": "src/nope.py", "beta.py": "src/beta.py"},
    )
    assert report.files == []
    assert any("nope.py" in m for m in report.extraction.missing)
    # beta.py was never in the prompt: a --resolve that resolves nothing is a typo, not a no-op.
    assert any("never names it" in m for m in report.extraction.missing)


# --- the CLI contract a wrapper script depends on ---------------------------------------------


def _cd(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    (root / "pharos.toml").write_text(
        f'model = "test-model-not-installed"\ntarget_folder = {str(root)!r}\n'
        f'observations_file = {str(root / "obs.json")!r}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(root)


def test_json_output_is_parseable_and_complete(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cd(monkeypatch, project)
    code = split_main(
        ["Refactor everything in `src/`.", "--target", "3000", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code in {0, 1}
    assert payload["schema"] == 1
    assert payload["tokenizer"] == "heuristic-chars-4"
    assert payload["ceiling"] > payload["floor"]
    assert payload["directories"][0]["path"] == "src"
    plan = payload["plan"]
    assert plan["mode"] == "scope"
    assert plan["parts"] and all(p["body"] for p in plan["parts"])
    assert sum(len(p["files"]) for p in plan["parts"]) == len(payload["directories"][0]["files"])


def test_json_stays_valid_when_there_is_no_plan(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cd(monkeypatch, project)
    split_main(["Rename a local variable.", "--target", "60000", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["mode"] == "none"
    assert payload["plan"]["already_fits"] is True
    assert payload["plan"]["parts"] == []


def test_check_json_carries_the_verdict_without_a_plan(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cd(monkeypatch, project)
    check_main(["Rename a local variable.", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "plan" not in payload
    assert payload["verdict"] in {"fits", "exceeds", "indeterminate"}


async def test_resolve_star_counts_every_candidate(project: Path) -> None:
    """The other honest answer to an ambiguous reference: 'I meant all of them'."""
    profile = _profile(project, loaded_ctx=32768)
    report = await run_check(
        _config(project), "fix `alpha.py`", profile=profile, resolve={"alpha.py": "*"}
    )
    displays = sorted(f.display.replace("\\", "/") for f in report.files)
    assert displays == ["src/alpha.py", "tests/alpha.py"]
    assert report.extraction.ambiguous == {}


def _fake_report(**ambiguous: list[Path]) -> object:
    class FakeReport:
        root = Path("/repo")

        class extraction:  # noqa: N801 — a stand-in, not a class the product defines
            pass

    FakeReport.extraction.ambiguous = ambiguous  # type: ignore[attr-defined]
    return FakeReport()


def test_picker_loop_handles_a_number_all_and_a_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the picker's own loop with an injected reader, not a patched builtin."""
    from rich.console import Console

    from pharos.preflight.cli import _pick_interactively

    # Under pytest stdin is captured and reports isatty() False, which would send the picker
    # down its "no terminal to ask on" path and test nothing at all.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    report = _fake_report(**{
        "a.py": [Path("/repo/src/a.py"), Path("/repo/tests/a.py")],
        "b.py": [Path("/repo/src/b.py"), Path("/repo/tests/b.py")],
        "c.py": [Path("/repo/src/c.py"), Path("/repo/tests/c.py")],
    })
    answers = iter(["2", "a", ""])
    chosen = _pick_interactively(
        Console(quiet=True),
        report,  # type: ignore[arg-type]
        from_stdin=False,
        ask=lambda _prompt: next(answers),
    )
    assert Path(chosen["a.py"]).parts[-2:] == ("tests", "a.py")
    assert chosen["b.py"] == "*"
    assert "c.py" not in chosen  # a blank answer skips rather than guessing


def test_picker_stops_cleanly_on_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-D mid-picker must abandon the rest, not raise into the user's terminal."""
    from rich.console import Console

    from pharos.preflight.cli import _pick_interactively

    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    report = _fake_report(**{"a.py": [Path("/repo/src/a.py"), Path("/repo/tests/a.py")]})

    def refuse(_prompt: str) -> str:
        raise EOFError

    assert _pick_interactively(
        Console(quiet=True), report, from_stdin=False, ask=refuse  # type: ignore[arg-type]
    ) == {}


def test_bad_resolve_flag_is_rejected(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cd(monkeypatch, project)
    assert check_main(["fix `alpha.py`", "--resolve", "no-equals-sign"]) == 2


def test_pick_refuses_rather_than_prompting_a_pipe(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--pick must never block on input() when there is no terminal to answer it."""
    _cd(monkeypatch, project)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    check_main(["fix `alpha.py`", "--pick"])
    assert "needs an interactive terminal" in capsys.readouterr().out


def test_json_keeps_stdout_clean_when_the_picker_talks(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Everything human goes to stderr under --json, or a caller piping to jq gets garbage."""
    _cd(monkeypatch, project)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    check_main(["fix `alpha.py`", "--json", "--pick"])
    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout is the payload and nothing else
    assert "matches 2 files" in captured.err


def test_pick_resolves_the_choice_it_is_given(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cd(monkeypatch, project)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    check_main(["fix `alpha.py`", "--json", "--pick"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["files"]) == 1
    assert payload["uncounted"]["ambiguous"] == {}
