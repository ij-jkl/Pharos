"""Agent tests: workspace containment, scope enforcement, and the budget ceiling.

No network and no model. The session is driven against a scripted fake backend, because what
is under test is Pharos's arithmetic and refusals — whether a 9B model writes good Python is
not something a test suite can assert, and pretending otherwise would make these tests lie.

Counting is a stand-in ``len(text) // 4`` throughout, so every budget assertion is exact and
reproducible on any machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from pharos.agent.runner import _edit_target, _load_payload, _with_handoff
from pharos.agent.session import (
    AgentSession,
    PartResult,
    recover_tool_calls,
    system_prompt_for,
)
from pharos.agent.tools import (
    ScopeEntry,
    ToolBox,
    ToolResult,
    agent_overhead_tokens,
    catalogue_text,
    scope_from_part_files,
)
from pharos.agent.workspace import (
    GitGuardError,
    NotARepository,
    Undo,
    Workspace,
    WorkspaceError,
    git_guard,
)
from pharos.config import PharosConfig
from pharos.preflight.split import PartFile, build_plan


def _count(text: str) -> int:
    return len(text) // 4


# Built from chr(92) rather than written as an escape: these three tests are ABOUT the
# separator, and a source-level escape is exactly the kind of thing that silently became
# something else once already.
POSIX_STYLE = "src/alpha.py"
WINDOWS_STYLE = "src" + chr(92) + "alpha.py"
WINDOWS_BETA = "src" + chr(92) + "beta.py"


def _part_file(display: str) -> PartFile:
    return PartFile(display=display, tokens=10)


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
    (tmp_path / "src" / "beta.py").write_text("beta = 2\n", encoding="utf-8")
    return Workspace(tmp_path)


# --- containment ------------------------------------------------------------------------


def test_workspace_resolves_a_path_inside_the_root(workspace: Workspace) -> None:
    assert workspace.resolve("src/alpha.py").is_file()


@pytest.mark.parametrize(
    "raw",
    [
        "../outside.py",
        "src/../../outside.py",  # resolves out, even though every component looks tame
        "src/../src/../../escape.txt",
    ],
)
def test_workspace_refuses_to_escape_the_root(workspace: Workspace, raw: str) -> None:
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        workspace.resolve(raw)


def test_workspace_refuses_an_absolute_path_elsewhere(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        workspace.resolve(str(Path.home() / "secrets.txt"))


@pytest.mark.parametrize("raw", ["../.env", ".env", ".git/config", "nested/.ssh/id_rsa"])
def test_workspace_refuses_sensitive_names_even_inside_the_root(
    workspace: Workspace, raw: str
) -> None:
    with pytest.raises(WorkspaceError):
        workspace.resolve(raw)


# --- scope enforcement ------------------------------------------------------------------


def _box(workspace: Workspace, scope: dict[str, ScopeEntry] | None) -> ToolBox:
    return ToolBox(workspace=workspace, scope=scope)


def test_in_scope_file_reads(workspace: Workspace) -> None:
    box = _box(workspace, {"src/alpha.py": ScopeEntry("src/alpha.py")})
    result = box.dispatch("read_file", {"path": "src/alpha.py"}, room=1000, count=_count)
    assert result.ok and "alpha = 1" in result.text
    assert "1| alpha = 1" in result.text  # numbered, so replace_lines can address it


def test_out_of_scope_read_is_refused_not_raised(workspace: Workspace) -> None:
    """The refusal has to come back as a tool result, or the part loses the work it has done."""
    box = _box(workspace, {"src/alpha.py": ScopeEntry("src/alpha.py")})
    result = box.dispatch("read_file", {"path": "src/beta.py"}, room=1000, count=_count)

    assert not result.ok
    assert result.refused_scope == "src/beta.py"
    assert "src/beta.py" not in result.text.split("may open only")[1]  # names what it MAY open
    assert box.scope_refusals == ["src/beta.py"]


def test_out_of_scope_write_never_touches_the_file(workspace: Workspace) -> None:
    box = _box(workspace, {"src/alpha.py": ScopeEntry("src/alpha.py")})
    result = box.dispatch(
        "write_file", {"path": "src/beta.py", "content": "wrecked\n"}, room=1000, count=_count
    )
    assert not result.ok
    assert (workspace.root / "src" / "beta.py").read_text(encoding="utf-8") == "beta = 2\n"
    assert box.files_written == []


def test_unrestricted_scope_allows_any_workspace_file(workspace: Workspace) -> None:
    box = _box(workspace, None)
    assert box.dispatch("read_file", {"path": "src/beta.py"}, room=1000, count=_count).ok


def test_a_file_too_large_for_the_remaining_window_is_refused_not_truncated(
    workspace: Workspace,
) -> None:
    """Truncation is the silent loss Pharos exists to expose; it must not happen to stay alive."""
    big = "x = 1\n" * 2000
    (workspace.root / "src" / "big.py").write_text(big, encoding="utf-8")
    box = _box(workspace, {"src/big.py": ScopeEntry("src/big.py")})

    result = box.dispatch("read_file", {"path": "src/big.py"}, room=100, count=_count)

    assert not result.ok
    assert "not truncated" in result.text
    assert "x = 1" not in result.text  # no partial content leaked in alongside the refusal


# --- slices -----------------------------------------------------------------------------


def test_a_sliced_file_reads_only_its_own_lines(workspace: Workspace) -> None:
    (workspace.root / "src" / "big.py").write_text(
        "".join(f"line{i}\n" for i in range(1, 101)), encoding="utf-8"
    )
    box = _box(workspace, {"src/big.py": ScopeEntry("src/big.py", 10, 12)})

    result = box.dispatch("read_file", {"path": "src/big.py"}, room=1000, count=_count)

    assert result.ok
    # Numbered from the file's own position: replace_lines with these numbers lands where the
    # model looked, not 9 lines earlier.
    assert "10| line10" in result.text
    assert "12| line12" in result.text
    assert "line9" not in result.text and "line13" not in result.text


def test_whole_file_write_is_refused_for_a_file_the_part_only_partly_owns(
    workspace: Workspace,
) -> None:
    original = "".join(f"line{i}\n" for i in range(1, 101))
    (workspace.root / "src" / "big.py").write_text(original, encoding="utf-8")
    box = _box(workspace, {"src/big.py": ScopeEntry("src/big.py", 10, 12)})

    result = box.dispatch(
        "write_file", {"path": "src/big.py", "content": "only what I saw\n"}, room=1000,
        count=_count,
    )

    assert not result.ok and "replace_lines" in result.text
    assert (workspace.root / "src" / "big.py").read_text(encoding="utf-8") == original


def test_replace_lines_edits_only_the_owned_range(workspace: Workspace) -> None:
    (workspace.root / "src" / "big.py").write_text(
        "".join(f"line{i}\n" for i in range(1, 21)), encoding="utf-8"
    )
    box = _box(workspace, {"src/big.py": ScopeEntry("src/big.py", 10, 12)})

    result = box.dispatch(
        "replace_lines",
        {"path": "src/big.py", "line_start": 10, "line_end": 12, "content": "REPLACED"},
        room=1000,
        count=_count,
    )

    assert result.ok
    written = (workspace.root / "src" / "big.py").read_text(encoding="utf-8")
    assert "line9\nREPLACED\nline13\n" in written
    assert "line1\n" in written and "line20\n" in written  # nothing outside the range moved


def test_replace_lines_refuses_to_reach_outside_the_owned_range(workspace: Workspace) -> None:
    original = "".join(f"line{i}\n" for i in range(1, 21))
    (workspace.root / "src" / "big.py").write_text(original, encoding="utf-8")
    box = _box(workspace, {"src/big.py": ScopeEntry("src/big.py", 10, 12)})

    result = box.dispatch(
        "replace_lines",
        {"path": "src/big.py", "line_start": 1, "line_end": 20, "content": "all mine"},
        room=1000,
        count=_count,
    )

    assert not result.ok
    assert (workspace.root / "src" / "big.py").read_text(encoding="utf-8") == original


# --- the git guard ----------------------------------------------------------------------


def test_git_guard_refuses_outside_a_repository(tmp_path: Path) -> None:
    with pytest.raises(GitGuardError, match="not a git repository"):
        git_guard(tmp_path)


# --- the session loop and its ceiling -----------------------------------------------------


class FakeBackend:
    """Replays scripted assistant turns as NDJSON, and records every request it was sent.

    Streamed, because the session streams: Ollama answers /api/chat with one JSON object per
    line and a final object carrying done + prompt_eval_count. A fake that replied in one shot
    would leave the streaming reader — the part that keeps a long generation observable —
    untested.
    """

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = turns
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        turn = self.turns[min(len(self.requests) - 1, len(self.turns) - 1)]
        # Content arrives in pieces, exactly as a real stream delivers it.
        text = str(turn.get("content") or "")
        lines = [
            json.dumps({"message": {"role": "assistant", "content": piece}, "done": False})
            for piece in (text[i : i + 8] for i in range(0, len(text), 8))
        ]
        if turn.get("tool_calls"):
            lines.append(
                json.dumps(
                    {
                        "message": {"role": "assistant", "tool_calls": turn["tool_calls"]},
                        "done": False,
                    }
                )
            )
        lines.append(json.dumps({"done": True, "done_reason": "stop", "prompt_eval_count": 123}))
        ndjson = "\n".join(lines) + "\n"
        return httpx.Response(200, content=ndjson.encode())


def _session(backend: FakeBackend, box: ToolBox, *, budget: int) -> AgentSession:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(backend.handler), base_url="http://backend"
    )
    return AgentSession(
        client=client,
        model="test-model",
        toolbox=box,
        count=_count,
        usable_budget=budget,
        handoff_reserve=100,
    )


async def test_session_executes_a_tool_call_then_finishes(workspace: Workspace) -> None:
    backend = FakeBackend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "write_file",
                                  "arguments": {"path": "src/alpha.py", "content": "alpha = 9\n"}}}
                ],
            },
            {"role": "assistant", "content": "Changed alpha.", "tool_calls": []},
        ]
    )
    box = _box(workspace, {"src/alpha.py": ScopeEntry("src/alpha.py")})
    result = await _session(backend, box, budget=100_000).run("Set alpha to 9.")

    assert result.error is None
    assert result.files_written == ["src/alpha.py"]
    assert result.text == "Changed alpha."
    assert result.reported_tokens == 123
    assert (workspace.root / "src" / "alpha.py").read_text(encoding="utf-8") == "alpha = 9\n"


async def test_a_part_too_big_for_the_window_never_reaches_the_backend(
    workspace: Workspace,
) -> None:
    backend = FakeBackend([{"role": "assistant", "content": "should never be asked"}])
    box = _box(workspace, None)
    result = await _session(backend, box, budget=200).run("x" * 40_000)

    assert result.error is not None and "no room to work in" in result.error
    assert backend.requests == []  # the guarantee: nothing unproven goes on the wire


async def test_the_ceiling_stops_the_loop_and_still_leaves_room_to_hand_off(
    workspace: Workspace,
) -> None:
    """A run that cannot afford to report that it is out of window has lost the work."""
    filler = "y = 2\n" * 400
    (workspace.root / "src" / "filler.py").write_text(filler, encoding="utf-8")
    read_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "src/filler.py"}}}],
    }
    # The model never stops asking; only the ceiling can end this part.
    backend = FakeBackend([read_call])
    box = _box(workspace, {"src/filler.py": ScopeEntry("src/filler.py")})

    result = await _session(backend, box, budget=4000).run("Read it repeatedly.")

    assert result.stopped_early
    assert result.error is None
    # The final request is the hand-off ask, and it carries no tool catalogue, so the model
    # cannot start working again with a window it no longer has.
    assert "tools" not in backend.requests[-1]
    assert backend.requests[-1]["messages"][-1]["role"] == "user"


async def test_every_request_sent_stayed_under_the_ceiling(workspace: Workspace) -> None:
    filler = "y = 2\n" * 400
    (workspace.root / "src" / "filler.py").write_text(filler, encoding="utf-8")
    read_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "src/filler.py"}}}],
    }
    backend = FakeBackend([read_call])
    box = _box(workspace, {"src/filler.py": ScopeEntry("src/filler.py")})
    session = _session(backend, box, budget=4000)

    result: PartResult = await session.run("Read it repeatedly.")

    assert result.steps >= 1
    for sent in backend.requests:
        body = json.dumps(sent["messages"], separators=(",", ":"))
        assert _count(body) <= session.ceiling + session.catalogue_tokens


# --- path spellings ---------------------------------------------------------------------


def test_scope_matches_whatever_separator_the_splitter_used(workspace: Workspace) -> None:
    """The splitter hands back the platform separator; the workspace and the model use posix.

    A real 9-part run refused every part access to its own files because of this, and the
    symptom was indistinguishable from a model ignoring its scope.
    """
    box = ToolBox(
        workspace=workspace, scope=scope_from_part_files([_part_file(WINDOWS_STYLE)])
    )
    for asked in (POSIX_STYLE, WINDOWS_STYLE):
        result = box.dispatch("read_file", {"path": asked}, room=1000, count=_count)
        assert result.ok, f"{asked} was refused"
    assert box.scope_refusals == []


def test_a_posix_scope_still_refuses_a_file_outside_it(workspace: Workspace) -> None:
    box = ToolBox(workspace=workspace, scope=scope_from_part_files([_part_file("src/alpha.py")]))
    result = box.dispatch("read_file", {"path": WINDOWS_BETA}, room=1000, count=_count)
    assert not result.ok and result.refused_scope is not None


# --- tool calls a model wrote as text ------------------------------------------------------


KNOWN = {"read_file", "write_file", "list_dir"}


def test_recovers_a_fenced_json_tool_call() -> None:
    """qwen2.5-coder:14b replies like this and leaves tool_calls null; without recovery a run
    against it changes nothing while reporting success."""
    content = '```json\n{\n  "name": "list_dir",\n  "arguments": {"path": "src/"}\n}\n```'
    calls = recover_tool_calls(content, KNOWN)
    assert calls == [{"function": {"name": "list_dir", "arguments": {"path": "src/"}}}]


def test_recovers_a_bare_json_tool_call() -> None:
    content = '{"name": "read_file", "arguments": {"path": "a.py"}}'
    assert recover_tool_calls(content, KNOWN)[0]["function"]["name"] == "read_file"


def test_recovers_a_call_nested_under_function() -> None:
    content = '```json\n{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}\n```'
    assert recover_tool_calls(content, KNOWN)[0]["function"]["arguments"] == {"path": "a.py"}


def test_recovers_several_calls_from_one_reply() -> None:
    content = (
        '```json\n[{"name":"read_file","arguments":{"path":"a.py"}},'
        '{"name":"read_file","arguments":{"path":"b.py"}}]\n```'
    )
    assert len(recover_tool_calls(content, KNOWN)) == 2


@pytest.mark.parametrize(
    "content",
    [
        "I would call read_file on a.py, then rewrite it.",  # prose about tools
        '```json\n{"name": "rm_rf", "arguments": {"path": "/"}}\n```',  # not in the catalogue
        '```json\n{"name": "read_file"}\n```',  # no arguments mapping
        '```json\n{"path": "a.py"}\n```',  # no name
        "Here is the config: ```json\n{\"debug\": true}\n```",  # unrelated JSON
        "",
    ],
)
def test_does_not_invent_a_call_from_text_that_is_not_one(content: str) -> None:
    """A false positive executes something the model never asked for, so the bar is strict."""
    assert recover_tool_calls(content, KNOWN) == []


# --- dead ends the dispatcher declines to create --------------------------------------------


def test_read_file_on_a_directory_returns_the_listing(workspace: Workspace) -> None:
    """A model that gets "not found" for a folder it can see in the task tends to give up.

    A real run died exactly here: list_dir on Controllers/, then read_file on Services/, then
    nothing. Answering the question being asked costs one tool result and saves the run.
    """
    box = ToolBox(workspace=workspace, scope=None)
    result = box.dispatch("read_file", {"path": "src"}, room=1000, count=_count)

    assert result.ok
    assert "is a directory" in result.text
    assert "alpha.py" in result.text and "beta.py" in result.text


def test_a_missing_file_points_at_list_dir(workspace: Workspace) -> None:
    box = ToolBox(workspace=workspace, scope=None)
    result = box.dispatch("read_file", {"path": "src/nope.py"}, room=1000, count=_count)
    assert not result.ok and "list_dir" in result.text


# --- failures that report themselves --------------------------------------------------------


async def test_a_backend_failure_names_its_exception_type(workspace: Workspace) -> None:
    """httpx timeout classes stringify to "", so a naive f"{exc}" produced a report that said
    only "backend call failed:" — a run died and refused to say how."""
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("")

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom), base_url="http://backend")
    session = AgentSession(
        client=client,
        model="test-model",
        toolbox=_box(workspace, None),
        count=_count,
        usable_budget=100_000,
        handoff_reserve=100,
    )
    result = await session.run("do something")

    assert result.error is not None
    assert "ReadTimeout" in result.error
    assert not result.error.rstrip().endswith(":")


# --- streaming: the reply cap and the truncation report -------------------------------------


async def test_the_reply_is_capped_by_the_room_left_under_the_ceiling(
    workspace: Workspace,
) -> None:
    """A model cannot generate its way out of the window if num_predict says it cannot."""
    backend = FakeBackend([{"role": "assistant", "content": "done", "tool_calls": []}])
    session = _session(backend, _box(workspace, None), budget=5_000)
    await session.run("small task")

    options = backend.requests[0]["options"]
    assert options["num_predict"] > 0
    assert options["num_predict"] <= session.ceiling
    assert backend.requests[0]["stream"] is True


class TruncatingBackend(FakeBackend):
    """Ends the stream with done_reason=length, the way Ollama reports hitting num_predict."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "half a file"}, "done": False}),
            json.dumps({"done": True, "done_reason": "length", "prompt_eval_count": 99}),
        ]
        return httpx.Response(200, content=(("\n".join(lines)) + "\n").encode())


async def test_a_reply_cut_off_by_the_cap_says_so(workspace: Workspace) -> None:
    """Handing back a half-written file as though it were finished is the silent damage."""
    backend = TruncatingBackend([])
    notes: list[str] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(backend.handler), base_url="http://backend"
    )
    session = AgentSession(
        client=client,
        model="test-model",
        toolbox=_box(workspace, None),
        count=_count,
        usable_budget=5_000,
        handoff_reserve=100,
        on_event=notes.append,
    )
    result = await session.run("rewrite something enormous")

    assert "cut off" in result.text
    assert any("cut off" in note for note in notes)


# --- the undo for a folder that is not a repository -----------------------------------------


def test_undo_snapshots_the_original_before_the_first_write(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    (tmp_path / "a.py").write_text("original\n", encoding="utf-8")
    undo = Undo(tmp_path, tmp_path / ".pharos" / "undo-test")
    box = ToolBox(workspace=ws, undo=undo)

    box.dispatch("write_file", {"path": "a.py", "content": "first\n"}, room=999, count=_count)
    box.dispatch("write_file", {"path": "a.py", "content": "second\n"}, room=999, count=_count)

    # The snapshot is what was there when the run STARTED, not the previous tool call.
    saved = (undo.directory / "a.py").read_text(encoding="utf-8")
    assert saved == "original\n"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "second\n"


def test_undo_records_a_created_file_without_snapshotting_it(tmp_path: Path) -> None:
    """Restoring a file that did not exist means deleting it, not writing an empty one."""
    ws = Workspace(tmp_path)
    undo = Undo(tmp_path, tmp_path / ".pharos" / "undo-test")
    box = ToolBox(workspace=ws, undo=undo)

    box.dispatch("write_file", {"path": "new.py", "content": "fresh\n"}, room=999, count=_count)

    assert undo.saved == {"new.py": False}
    assert not (undo.directory / "new.py").exists()
    assert "delete to undo" in undo.restore_hint()


def test_not_a_repository_is_its_own_error(tmp_path: Path) -> None:
    """The runner distinguishes it from a dirty repo: one falls back, the other must stop."""
    with pytest.raises(NotARepository):
        git_guard(tmp_path)


# --- line endings ---------------------------------------------------------------------------


def test_a_crlf_file_stays_crlf_after_a_rewrite(workspace: Workspace) -> None:
    """A real C# run produced a diff where every line was changed, purely from LF writes.

    That makes the result unreviewable, and reviewing the diff is the only check a user has.
    """
    target = workspace.root / "win.cs"
    target.write_bytes(b"one\r\ntwo\r\n")
    box = ToolBox(workspace=workspace)

    box.dispatch(
        "write_file", {"path": "win.cs", "content": "one\nmiddle\ntwo\n"},
        room=999, count=_count,
    )

    written = target.read_bytes()
    assert written.count(b"\r\n") == 3
    assert written == b"one\r\nmiddle\r\ntwo\r\n"


def test_an_lf_file_stays_lf(workspace: Workspace) -> None:
    target = workspace.root / "unix.py"
    target.write_bytes(b"a\nb\n")
    box = ToolBox(workspace=workspace)

    box.dispatch("write_file", {"path": "unix.py", "content": "a\nc\n"}, room=999, count=_count)

    assert target.read_bytes() == b"a\nc\n"
    assert b"\r" not in target.read_bytes()


def test_a_new_file_is_written_lf(workspace: Workspace) -> None:
    box = ToolBox(workspace=workspace)
    box.dispatch("write_file", {"path": "fresh.py", "content": "x\ny\n"}, room=999, count=_count)
    assert (workspace.root / "fresh.py").read_bytes() == b"x\ny\n"


# --- the agent's own overhead, and the window it asks for -----------------------------------


def test_agent_overhead_is_counted_not_guessed(workspace: Workspace) -> None:
    """The proxy has to learn this figure for a third-party agent; here it is simply counted.

    The planner and the session both subtract it, from this one function, so they cannot drift
    apart and hand each other parts that do not fit.
    """
    overhead = agent_overhead_tokens(workspace, _count)

    catalogue = ToolBox(workspace=workspace).catalogue()
    expected = _count(catalogue_text(catalogue)) + _count(system_prompt_for(workspace.root))
    assert overhead == expected > 0


def test_the_plan_target_leaves_room_for_the_agents_own_overhead() -> None:
    """Before this, the planner offered parts the session had already spent the room on."""
    config = PharosConfig(model="m", handoff_reserve=500)
    generous = _edit_target(config, 16_000, 0)
    realistic = _edit_target(config, 16_000, 600)
    assert realistic < generous
    assert generous - realistic == 300  # halved, like every other content token


def test_num_ctx_is_only_requested_when_configured() -> None:
    """Pharos must not silently change a window it is also reporting on."""
    assert "options" not in _load_payload(PharosConfig(model="m"))
    assert _load_payload(PharosConfig(model="m", num_ctx=16384))["options"] == {"num_ctx": 16384}


# --- dividing by work, not only by tokens ---------------------------------------------------


def _report_with_files(tmp_path: Path, count: int) -> tuple[PharosConfig, str, Any]:
    """A prompt naming `count` small files that all fit one window comfortably."""
    import asyncio

    from pharos.accountant import Accountant
    from pharos.preflight.check import run_check
    from pharos.profiler.types import BackendInfo, EnvironmentProfile, GpuInfo

    (tmp_path / "src").mkdir()
    names = [f"mod{i}.py" for i in range(count)]
    for name in names:
        (tmp_path / "src" / name).write_text("x = 1\n" * 20, encoding="utf-8")
    config = PharosConfig(
        model="test-model-not-installed",
        target_folder=str(tmp_path),
        observations_file=str(tmp_path / "obs.json"),
        response_reserve=1024,
    )
    prompt = "Document " + ", ".join(f"`src/{n}`" for n in names) + "."
    gpu = GpuInfo(available=True, name="g", total_mib=12288, used_mib=1, free_mib=12000)
    profile = EnvironmentProfile(
        gpu=gpu,
        backend=BackendInfo(
            reachable=True, base_url="u", model="test-model-not-installed",
            advertised_max_ctx=262144, loaded_ctx=32768,
        ),
        budget=Accountant(config).report(loaded_ctx=32768, gpu=gpu),
        ctx_mismatch=False,
        ctx_mismatch_ratio=None,
    )
    report = asyncio.run(run_check(config, prompt, profile=profile))
    return config, prompt, report


def test_a_part_is_capped_by_file_count_even_when_the_tokens_fit(tmp_path: Path) -> None:
    """The failure this exists for: thirteen files fit the window, and the model wrote none."""
    config, prompt, report = _report_with_files(tmp_path, 12)

    plan = build_plan(config, prompt, report, target=100_000,
                      max_files=config.max_files_per_part)

    assert plan.parts, "a plan should still be produced"
    assert max(len(p.files) for p in plan.parts) <= config.max_files_per_part
    assert len(plan.parts) >= 3
    # Every part is still the splitter's own, so every projection still holds.
    assert all(p.fits for p in plan.parts)


def test_the_file_cap_does_not_split_a_task_that_is_already_small(tmp_path: Path) -> None:
    config, prompt, report = _report_with_files(tmp_path, 2)
    plan = build_plan(config, prompt, report, target=100_000,
                      max_files=config.max_files_per_part)
    assert plan.already_fits or len(plan.parts) <= 1


def test_without_a_cap_the_splitter_behaves_exactly_as_before(tmp_path: Path) -> None:
    """The cap is opt-in: `pharos check --split` and `pharos split` must be unaffected."""
    config, prompt, report = _report_with_files(tmp_path, 12)
    assert build_plan(config, prompt, report, target=100_000).already_fits


def test_the_projection_counts_content_not_its_json_envelope(workspace: Workspace) -> None:
    """JSON escaping roughly doubles code; a real run over-estimated 3,164 tokens as 7,252."""
    backend = FakeBackend([{"role": "assistant", "content": "ok", "tool_calls": []}])
    session = _session(backend, _box(workspace, None), budget=100_000)
    code = 'var x = "a";\nvar y = "b";\n' * 50

    session._messages = [{"role": "user", "content": code}]
    # Compare like with like: the catalogue is in both the old and new figure.
    conversation = session._projected() - session.catalogue_tokens

    envelope = _count(json.dumps(session._messages, separators=(",", ":")))
    assert conversation < envelope  # the envelope is what we used to charge the user for
    assert conversation >= _count(code)  # still a floor on the content itself


def test_the_undo_snapshot_is_invisible_and_unreachable(tmp_path: Path) -> None:
    """A stale pre-run copy of the file being edited is worse than no copy at all."""
    ws = Workspace(tmp_path)
    (tmp_path / "a.py").write_text("original\n", encoding="utf-8")
    undo = Undo(tmp_path, tmp_path / ".pharos" / "undo-test")
    box = ToolBox(workspace=ws, undo=undo)
    box.dispatch("write_file", {"path": "a.py", "content": "changed\n"}, room=999, count=_count)

    assert (undo.directory / "a.py").is_file()  # the snapshot exists
    listing = box.dispatch("list_dir", {"path": "."}, room=999, count=_count)
    assert ".pharos" not in listing.text
    reach = box.dispatch(
        "read_file", {"path": ".pharos/undo-test/a.py"}, room=999, count=_count
    )
    assert not reach.ok and "off limits" in reach.text


def test_replace_lines_is_always_offered(workspace: Workspace) -> None:
    """A model that must JSON-escape a whole C# file to make a three-line change gives up."""
    names = {t["function"]["name"] for t in ToolBox(workspace=workspace).catalogue()}
    assert "replace_lines" in names


def test_read_marks_the_line_numbers_as_not_part_of_the_file(workspace: Workspace) -> None:
    """A model that writes the NNN| prefixes back has corrupted the file, so say so in-band."""
    result = ToolBox(workspace=workspace).dispatch(
        "read_file", {"path": "src/alpha.py"}, room=999, count=_count
    )
    assert "line numbers" in result.text
    assert "never be written back" in result.text


# --- a bad tool call must not end the run ---------------------------------------------------


def test_content_sent_as_a_list_of_lines_is_joined(workspace: Workspace) -> None:
    """A real run crashed here: the schema says string, the model sent a list."""
    box = ToolBox(workspace=workspace)
    result = box.dispatch(
        "write_file", {"path": "src/alpha.py", "content": ["a = 1", "b = 2"]},
        room=999, count=_count,
    )
    assert result.ok
    assert (workspace.root / "src" / "alpha.py").read_text(encoding="utf-8") == "a = 1\nb = 2"


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "src/alpha.py", "line_start": "not a number", "line_end": 2, "content": "x"},
        {"path": "src/alpha.py", "content": {"nested": "object"}},
        {"path": ["a", "list"], "content": "x"},
        {},
    ],
)
def test_a_malformed_tool_call_is_reported_not_raised(
    workspace: Workspace, arguments: dict[str, Any]
) -> None:
    """One bad argument shape must not destroy a run that has already done work."""
    box = ToolBox(workspace=workspace)
    for name in ("write_file", "replace_lines", "read_file"):
        # The property is that it comes back at all. Whether a given shape is salvageable
        # (a list of lines) or nonsense (a path that is a list) is the dispatcher's business;
        # what must never happen is an exception escaping into the run.
        result = box.dispatch(name, arguments, room=999, count=_count)
        assert isinstance(result, ToolResult)
        assert result.text


def test_replace_lines_reports_the_shift_it_caused(workspace: Workspace) -> None:
    """The tool call succeeds either way, so a stale second edit fails silently and wrongly."""
    target = workspace.root / "src" / "big.py"
    target.write_text("".join(f"line{i}\n" for i in range(1, 21)), encoding="utf-8")
    box = ToolBox(workspace=workspace)

    grew = box.dispatch(
        "replace_lines",
        {"path": "src/big.py", "line_start": 5, "line_end": 5, "content": "a\nb\nc"},
        room=999, count=_count,
    )
    assert grew.ok and "+2" in grew.text and "stale" in grew.text

    same = box.dispatch(
        "replace_lines",
        {"path": "src/big.py", "line_start": 1, "line_end": 1, "content": "one"},
        room=999, count=_count,
    )
    assert same.ok and "shifted" not in same.text  # no move, no warning to ignore


def test_content_still_carrying_line_numbers_is_refused(workspace: Workspace) -> None:
    """Numbering reads created this failure mode, so the write path has to check for it."""
    box = ToolBox(workspace=workspace)
    echoed = "1| public class Foo\n2| {\n3| }\n"

    written = box.dispatch(
        "write_file", {"path": "src/alpha.py", "content": echoed}, room=999, count=_count
    )
    assert not written.ok and "line-number prefixes" in written.text
    assert (workspace.root / "src" / "alpha.py").read_text(encoding="utf-8") == "alpha = 1\n"

    ranged = box.dispatch(
        "replace_lines",
        {"path": "src/alpha.py", "line_start": 1, "line_end": 1, "content": echoed},
        room=999, count=_count,
    )
    assert not ranged.ok and "line-number prefixes" in ranged.text


def test_ordinary_code_is_not_mistaken_for_a_numbered_view(workspace: Workspace) -> None:
    box = ToolBox(workspace=workspace)
    result = box.dispatch(
        "write_file",
        {"path": "src/alpha.py", "content": "x = 1\ny = 2\nz = 3\nprint(x | y)\n"},
        room=999, count=_count,
    )
    assert result.ok


def test_replace_lines_keeps_a_crlf_file_crlf(workspace: Workspace) -> None:
    """The write_file path was fixed and this one was not; a real run converted 150 lines."""
    target = workspace.root / "win.cs"
    target.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    box = ToolBox(workspace=workspace)

    result = box.dispatch(
        "replace_lines",
        {"path": "win.cs", "line_start": 2, "line_end": 2, "content": "TWO"},
        room=999, count=_count,
    )

    assert result.ok
    written = target.read_bytes()
    assert written == b"one\r\nTWO\r\nthree\r\n"
    assert written.count(b"\r\n") == 3


def test_a_part_that_only_fails_is_abandoned(workspace: Workspace) -> None:
    """A real run made eleven straight refused writes to a folder the project does not have.

    Every refusal was correct and the repo was never at risk; the waste was grinding on to the
    ceiling afterwards and handing off nothing.
    """
    doomed = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "write_file",
                          "arguments": {"path": "src/nope.py", "content": "x = 1\n"}}}
        ],
    }
    backend = FakeBackend([doomed])
    box = _box(workspace, {"src/alpha.py": ScopeEntry("src/alpha.py")})
    notes: list[str] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(backend.handler), base_url="http://backend"
    )
    session = AgentSession(
        client=client, model="m", toolbox=box, count=_count,
        usable_budget=100_000, handoff_reserve=100, on_event=notes.append,
    )

    result = await_run(session, "do the impossible")

    assert result.stopped_early
    assert any("in a row failed" in note for note in notes)
    # Abandoned early, not at _MAX_STEPS: the point is not burning forty rounds first.
    assert result.steps < 12
    assert result.files_written == []


def await_run(session: AgentSession, body: str) -> PartResult:
    import asyncio

    return asyncio.run(session.run(body))


# --- the nudge, when a part is only partly done ---------------------------------------------


async def test_a_part_that_wrote_some_of_its_files_is_still_asked_for_the_rest(
    workspace: Workspace,
) -> None:
    """Partial coverage is the failure that actually happens; real runs landed at 31-70%.

    An all-or-nothing check sees two writes and calls it done. The reminder has to name what
    is outstanding, because "you have not called write_file" tells a model that just called
    write_file nothing it can act on.
    """
    write_one = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "write_file",
                                     "arguments": {"path": "src/alpha.py", "content": "a=1\n"}}}],
    }
    done = {"role": "assistant", "content": "All finished.", "tool_calls": []}
    backend = FakeBackend([write_one, done])
    box = _box(workspace, scope_from_part_files(
        [_part_file("src/alpha.py"), _part_file("src/beta.py")]
    ))

    await _session(backend, box, budget=100_000).run("edit both")

    asked = [m for r in backend.requests for m in r["messages"] if m.get("role") == "user"]
    reminder = [m["content"] for m in asked if "still unchanged" in m.get("content", "")]
    assert reminder, "the part stopped half-done and was never asked about the rest"
    assert "src/beta.py" in reminder[0]
    assert "src/alpha.py" not in reminder[0]  # already written; naming it invites a rewrite


async def test_a_fully_covered_part_is_not_nagged(workspace: Workspace) -> None:
    """The reminder must mean something, so it cannot fire on a part that did its job."""
    write_both = {
        "role": "assistant", "content": "",
        "tool_calls": [
            {"function": {"name": "write_file",
                          "arguments": {"path": "src/alpha.py", "content": "a=1\n"}}},
            {"function": {"name": "write_file",
                          "arguments": {"path": "src/beta.py", "content": "b=2\n"}}},
        ],
    }
    done = {"role": "assistant", "content": "Done both.", "tool_calls": []}
    backend = FakeBackend([write_both, done])
    box = _box(workspace, scope_from_part_files(
        [_part_file("src/alpha.py"), _part_file("src/beta.py")]
    ))

    result = await _session(backend, box, budget=100_000).run("edit both")

    assert sorted(result.files_written) == ["src/alpha.py", "src/beta.py"]
    assert not result.nudged


def test_a_part_may_not_churn_forever_on_one_file(workspace: Workspace) -> None:
    """Observed live: a dozen replace_lines on the same file, every one succeeding.

    The consecutive-failure breaker never saw it because nothing failed, and each edit moved
    the lines under the model's map so it kept fixing what it had just changed — while its
    other files went unopened. Churn on one file is the coverage failure in slow motion.
    """
    target = workspace.root / "src" / "alpha.py"
    target.write_text("a = 1\n", encoding="utf-8")
    box = ToolBox(workspace=workspace, scope=scope_from_part_files(
        [_part_file("src/alpha.py"), _part_file("src/beta.py")]
    ))

    outcomes = [
        box.dispatch("write_file", {"path": "src/alpha.py", "content": f"a = {i}\n"},
                     room=999, count=_count)
        for i in range(8)
    ]

    assert all(r.ok for r in outcomes[:5]), "the first few edits are legitimate"
    assert not outcomes[5].ok and "already been rewritten" in outcomes[5].text
    assert "src/beta.py" in outcomes[5].text  # points at the work being starved
    assert target.read_text(encoding="utf-8") == "a = 4\n"  # the 6th write never landed


def test_the_churn_guard_counts_per_file_not_per_part(workspace: Workspace) -> None:
    """Editing several files a few times each is normal work, not churn."""
    box = ToolBox(workspace=workspace)
    for _ in range(4):
        for name in ("alpha", "beta"):
            result = box.dispatch(
                "write_file", {"path": f"src/{name}.py", "content": "x = 1\n"},
                room=999, count=_count,
            )
            assert result.ok


def test_the_system_prompt_is_not_charged_twice(workspace: Workspace) -> None:
    """It lives in messages[0] and was ALSO folded into the per-request constant.

    Measured on a real workspace: 353 phantom tokens on every projection, which was most of
    the gap between what Pharos projected and what the backend reported.
    """
    backend = FakeBackend([{"role": "assistant", "content": "ok", "tool_calls": []}])
    session = _session(backend, _box(workspace, None), budget=100_000)
    box = ToolBox(workspace=workspace)

    assert session.catalogue_tokens == _count(catalogue_text(box.catalogue()))

    system = system_prompt_for(workspace.root)
    session._messages = [{"role": "system", "content": system}]
    charged = session._projected() - session.catalogue_tokens
    assert charged == _count(system) + 8  # once, plus the per-message constant


# --- what a part is actually told ------------------------------------------------------------


def test_a_part_is_told_how_many_files_it_must_change(tmp_path: Path) -> None:
    """A list of files reads as material, not a checklist; runs settled at half the scope.

    The nudge that catches this only fires after the model has decided it is finished. Saying
    the count up front is the cheap half of the same idea.
    """
    prompt = _with_handoff("SCOPE BLOCK", None, 1, [_part_file("a.py"), _part_file("b.py")])

    assert "YOUR TARGET: 2 files" in prompt
    assert "not finished until every one of the 2" in prompt
    assert prompt.endswith("SCOPE BLOCK")  # the target comes first, the scope block last
    assert "a.py" not in prompt  # names live in the scope block; repeating them costs tokens


def test_a_single_file_part_is_told_in_the_singular() -> None:
    prompt = _with_handoff("SCOPE", None, 1, [_part_file("a.py")])
    assert "YOUR TARGET: 1 file." in prompt and "1 files" not in prompt


def test_an_unscoped_part_gets_no_target_it_cannot_meet(tmp_path: Path) -> None:
    """One unrestricted part has no list to complete, so a count would be a fiction."""
    assert _with_handoff("SCOPE", None, 1, None) == "SCOPE"


def test_a_handoff_is_framed_as_context_about_other_files() -> None:
    """Part 1 reported the task complete and parts 2-4 then did nothing, correctly by their
    reading. The framing is repeated after the hand-off because the last thing read wins."""
    prompt = _with_handoff("SCOPE", "I finished the documentation task.", 2, [_part_file("b.py")])

    assert "context only" in prompt
    assert "DIFFERENT files" in prompt
    assert prompt.index("have NOT been done yet") > prompt.index("I finished")
