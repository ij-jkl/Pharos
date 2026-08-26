"""Agent tests: workspace containment, scope enforcement, and the budget ceiling.

No network and no model. The session is driven against a scripted fake backend, because what
is under test is Pharos's arithmetic and refusals — whether a 9B model writes good Python is
not something a test suite can assert, and pretending otherwise would make these tests lie.

Counting is a stand-in ``len(text) // 4`` throughout, so every budget assertion is exact and
reproducible on any machine.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

from pharos.agent.cli import _render_footer
from pharos.agent.runner import (
    RunOutcome,
    _edit_target,
    _load_payload,
    _with_handoff,
)
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
    current_branch,
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

    Both spellings have to work on both platforms. On POSIX a backslash is a legal character
    in a file name rather than a separator, so this passed on Windows and failed on Linux CI
    until the workspace learned to fall back to the separator reading.
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


async def test_a_part_still_making_progress_is_asked_again(workspace: Workspace) -> None:
    """One reminder was not enough: a part that wrote one of four files, was reminded, wrote a
    second and stopped had already spent the only ask available."""
    def writes(path: str) -> dict[str, Any]:
        return {
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "write_file",
                                         "arguments": {"path": path, "content": "x = 1\n"}}}],
        }
    stop = {"role": "assistant", "content": "Done.", "tool_calls": []}
    # Writes one, stops; reminded, writes another, stops; reminded again, writes the third.
    backend = FakeBackend([
        writes("src/a.py"), stop, writes("src/b.py"), stop, writes("src/c.py"), stop,
    ])
    box = _box(workspace, scope_from_part_files(
        [_part_file("src/a.py"), _part_file("src/b.py"), _part_file("src/c.py")]
    ))

    result = await _session(backend, box, budget=100_000).run("edit all three")

    assert sorted(result.files_written) == ["src/a.py", "src/b.py", "src/c.py"]
    assert result.nudges >= 2  # the second ask is what got the third file


async def test_a_part_ignoring_the_reminder_is_not_asked_forever(workspace: Workspace) -> None:
    """Each further ask has to be earned by the last one producing a write, or a part that
    will not finish costs the same refusal over and over."""
    stop = {"role": "assistant", "content": "I have finished.", "tool_calls": []}
    backend = FakeBackend([stop])
    box = _box(workspace, scope_from_part_files(
        [_part_file("src/alpha.py"), _part_file("src/beta.py")]
    ))

    result = await _session(backend, box, budget=100_000).run("edit both")

    assert result.files_written == []
    assert result.nudges == 1  # asked once, ignored, not asked again


def test_a_refused_call_is_logged_with_its_reason(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """Six consecutive failures abandoned a real part and the log said only which tool.

    The one question worth asking — why — needed the whole run reproducing.
    """
    box = _box(workspace, {"src/alpha.py": ScopeEntry("src/alpha.py")})
    session = _session(FakeBackend([]), box, budget=100_000)

    with caplog.at_level(logging.WARNING, logger="pharos.agent"):
        session._execute({"function": {"name": "read_file",
                                       "arguments": {"path": "src/beta.py"}}})

    assert "REFUSED" in caplog.text
    assert "not in this part's scope" in caplog.text


def test_file_content_is_not_copied_into_the_log(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """A write_file call carries a whole source file; logging it verbatim buries the path."""
    box = _box(workspace, None)
    session = _session(FakeBackend([]), box, budget=100_000)
    big = "x = 1\n" * 500

    with caplog.at_level(logging.INFO, logger="pharos.agent"):
        session._execute({"function": {"name": "write_file",
                                       "arguments": {"path": "src/alpha.py", "content": big}}})

    assert "src/alpha.py" in caplog.text
    assert "x = 1" not in caplog.text
    assert "chars>" in caplog.text


# --- the window changing underneath a run ----------------------------------------------------


def test_num_ctx_is_repeated_on_every_request_not_just_the_load() -> None:
    """Ollama reloads the model to serve a differing options set. A run that loaded at 16,384
    and then asked for num_predict alone got it reloaded at the backend's default 4,096, while
    the ceiling went on being enforced against 16,384 and the backend truncated silently."""
    from pharos.agent.session import _request_options

    assert _request_options(500, 16384) == {"num_predict": 500, "num_ctx": 16384}
    assert _request_options(500, None) == {"num_predict": 500}


class ShrinkingBackend(FakeBackend):
    """Reports a falling prompt_eval_count: the shape of a backend dropping context."""

    counts = [1000, 2000, 1400]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        n = self.counts[min(len(self.requests) - 1, len(self.counts) - 1)]
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": False}),
            json.dumps({"done": True, "done_reason": "stop", "prompt_eval_count": n}),
        ]
        return httpx.Response(200, content=(("\n".join(lines)) + "\n").encode())


async def test_a_falling_backend_count_is_reported_as_truncation(workspace: Workspace) -> None:
    """A conversation only grows, so the backend's count for it can only grow. When it falls,
    the backend has stopped evaluating everything it was sent — silent context loss, which is
    the one thing this project must never be the last to notice in its own client."""
    backend = ShrinkingBackend([])
    notes: list[str] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(backend.handler), base_url="http://backend"
    )
    session = AgentSession(
        client=client, model="m", toolbox=_box(workspace, None), count=_count,
        usable_budget=100_000, handoff_reserve=100, on_event=notes.append,
    )

    tools = ToolBox(workspace=workspace).catalogue()
    session._messages = [{"role": "user", "content": "one"}]
    await session._chat(tools)
    session._messages.append({"role": "user", "content": "two"})
    await session._chat(tools)
    session._messages.append({"role": "user", "content": "three"})
    await session._chat(tools)

    assert session.truncated_by_backend
    assert any("BACKEND TRUNCATED" in note for note in notes)


async def test_a_growing_backend_count_is_not_an_alarm(workspace: Workspace) -> None:
    backend = FakeBackend([{"role": "assistant", "content": "ok", "tool_calls": []}])
    session = _session(backend, _box(workspace, None), budget=100_000)
    await session.run("do a thing")
    assert not session.truncated_by_backend


async def test_the_toolless_handoff_does_not_look_like_truncation(workspace: Workspace) -> None:
    """The hand-off goes out with no tool catalogue, so its prompt is legitimately a few
    hundred tokens smaller than the request before it.

    Read naively that is indistinguishable from the backend dropping context — and it would
    have fired on every part that ends by handing off, which is most of them. A warning that
    cries wolf on the common path is worse than no warning.
    """
    backend = ShrinkingBackend([])  # 1000, then 2000, then 1400
    notes: list[str] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(backend.handler), base_url="http://backend"
    )
    session = AgentSession(
        client=client, model="m", toolbox=_box(workspace, None), count=_count,
        usable_budget=100_000, handoff_reserve=100, on_event=notes.append,
    )
    tools = ToolBox(workspace=workspace).catalogue()

    session._messages = [{"role": "user", "content": "one"}]
    await session._chat(tools)          # 1000, with tools
    session._messages.append({"role": "user", "content": "two"})
    await session._chat(tools)          # 2000, with tools
    session._messages.append({"role": "user", "content": "three"})
    await session._chat(None)           # 1400, WITHOUT tools — the hand-off

    assert not session.truncated_by_backend
    assert not any("BACKEND TRUNCATED" in note for note in notes)


def test_a_toolless_request_is_not_charged_for_the_catalogue(workspace: Workspace) -> None:
    """Counting a catalogue that was never sent makes that sample incomparable and inflates
    the drift figure for every part that ends with a hand-off."""
    backend = FakeBackend([{"role": "assistant", "content": "ok", "tool_calls": []}])
    session = _session(backend, _box(workspace, None), budget=100_000)
    session._messages = [{"role": "user", "content": "hello"}]

    with_tools = session._projected(with_tools=True)
    without = session._projected(with_tools=False)

    assert with_tools - without == session.catalogue_tokens > 0


# --- a hand-off must fit the room reserved for it --------------------------------------------


def test_an_oversized_handoff_is_cut_to_the_reserve() -> None:
    """Every part's ceiling was computed with handoff_reserve subtracted for exactly this text.

    A real run produced 1,804 tokens against a 500-token reserve and forwarded it whole, so the
    next part ran under a projection wrong by 1,300 tokens before it read anything.
    """
    from pharos.agent.runner import _cap_handoff

    huge = "Documented the repository classes in detail. " * 200
    kept, cut = _cap_handoff(huge, 100, _count)

    assert cut
    # The marker used to be appended AFTER the prose had been trimmed to exactly the reserve,
    # so every cut hand-off overran by the size of its own explanation. This assertion read
    # `<= 100 + 60  # the marker itself costs a little` and waved it through.
    assert _count(kept) <= 100
    assert kept.startswith("Documented the repository")  # the opening survives
    assert "was cut" in kept  # and the loss is stated in the text, not just the report


def test_a_reserve_too_small_to_explain_the_cut_still_says_there_was_one() -> None:
    """Below the marker's own size the choice is a shorter truth or a silent drop."""
    from pharos.agent.runner import _cap_handoff

    huge = "Documented the repository classes in detail. " * 200
    kept, cut = _cap_handoff(huge, 30, _count)

    assert cut
    assert _count(kept) <= 30
    assert "dropped" in kept


def test_a_handoff_inside_the_reserve_is_passed_through_untouched() -> None:
    from pharos.agent.runner import _cap_handoff

    normal = "Added doc comments to NoteRepository.cs and INoteRepository.cs."
    kept, cut = _cap_handoff(normal, 500, _count)
    assert kept == normal and not cut


async def test_a_part_that_reads_but_does_not_write_is_asked_again(workspace: Workspace) -> None:
    """It cost a real run two files. Part 1 was reminded, went and READ the second of its two
    files, wrote nothing, and was never asked again because opening a file it had not seen did
    not count as movement. It plainly is."""
    read_a = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "read_file",
                                     "arguments": {"path": "src/alpha.py"}}}],
    }
    read_b = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "src/beta.py"}}}],
    }
    stop = {"role": "assistant", "content": "Looks fine.", "tool_calls": []}
    backend = FakeBackend([read_a, stop, read_b, stop, stop])
    box = _box(workspace, scope_from_part_files(
        [_part_file("src/alpha.py"), _part_file("src/beta.py")]
    ))

    result = await _session(backend, box, budget=100_000).run("document both")

    assert result.nudges >= 2, "reading the second file should have earned another ask"


def test_the_handoff_request_forbids_the_decline_phrase() -> None:
    """Parts that WROTE files opened their hand-off with "NO CHANGES NEEDED", sometimes then
    contradicting themselves with a correct summary underneath.

    The system prompt defines that phrase as the way to decline work, and it bled into the
    hand-off, where it tells the next part the opposite of what happened.
    """
    from pharos.agent.session import _HANDOFF_REQUEST

    assert "NAME each file you changed" in _HANDOFF_REQUEST
    assert "Do NOT write 'NO CHANGES NEEDED' here" in _HANDOFF_REQUEST


# --- the undo instruction names the branch the run actually started from -------------------------


def _repo_on_branch(root: Path, branch: str) -> None:
    """A git repository whose default branch is `branch` — the case `main` was assumed for."""
    subprocess.run(["git", "init", "-q", "-b", branch, str(root)], check=True, capture_output=True)
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    for args in (
        ["config", "user.email", "t@t"],
        ["config", "user.name", "t"],
        ["add", "-A"],
        ["commit", "-qm", "init"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_current_branch_reads_a_non_main_default(tmp_path: Path) -> None:
    _repo_on_branch(tmp_path, "master")
    assert current_branch(tmp_path) == "master"


def test_current_branch_falls_back_to_a_sha_when_head_is_detached(tmp_path: Path) -> None:
    _repo_on_branch(tmp_path, "trunk")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "checkout", "-q", "--detach"], cwd=tmp_path, check=True)
    # `git checkout <sha>` is as valid as a branch name, so the undo line still works.
    assert current_branch(tmp_path) == head


def test_current_branch_is_none_outside_a_repository(tmp_path: Path) -> None:
    assert current_branch(tmp_path) is None


def test_undo_instruction_names_the_real_base_branch(tmp_path: Path) -> None:
    """The bug: `main` was hardcoded, so on a `master` repository the advice printed at the
    end of a run does not work — and it is the only route back to the pre-run tree."""
    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    outcome = RunOutcome(
        branch="pharos-run/2026-01-01-000000",
        base_branch="master",
        files_changed=["a.py"],
    )
    _render_footer(console, outcome)
    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert "git diff master" in text
    assert "git checkout master" in text
    assert "main" not in text


def test_undo_instruction_falls_back_to_previous_ref_when_base_unknown(tmp_path: Path) -> None:
    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    outcome = RunOutcome(branch="pharos-run/x", base_branch=None, files_changed=["a.py"])
    _render_footer(console, outcome)
    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert "git checkout -" in text


# --- the ceiling tightens to what the backend is really counting ---------------------------------


def _session_with_drift(
    workspace: Workspace, samples: list[tuple[int, int]], *, budget: int = 4000
) -> AgentSession:
    session = _session(FakeBackend([]), ToolBox(workspace, scope=set()), budget=budget)
    session._drift = list(samples)
    return session


def test_template_offset_is_zero_before_any_response(workspace: Workspace) -> None:
    """Nothing has been counted yet and nothing was remembered, so there is nothing to correct
    by — SAFETY_MARGIN alone covers the first request. v0.9's template memory is what closes
    this for a model that has been run before."""
    assert _session_with_drift(workspace, [])._template_offset() == 0


def test_template_offset_takes_the_worst_gap_seen(workspace: Workspace) -> None:
    """A ceiling holds at the worst case or it does not hold."""
    session = _session_with_drift(workspace, [(1000, 1100), (1000, 1250), (1000, 1050)])
    assert session._template_offset() == 250


def test_template_offset_never_loosens_the_ceiling(workspace: Workspace) -> None:
    """Over-counting wastes room; it is not permission to fit more in."""
    session = _session_with_drift(workspace, [(1000, 800), (1000, 900)])
    assert session._template_offset() == 0


def test_ceiling_is_enforced_against_the_corrected_projection(workspace: Workspace) -> None:
    """The regression: on qwen3.5-9b the backend counted ~275 tokens above our estimate on
    every request, so a part that looked 200 tokens clear of the ceiling was already over it."""
    session = _session_with_drift(workspace, [(1000, 1275)])
    session._messages = [{"role": "user", "content": "x" * 4000}]
    raw = session._projected()
    corrected = session._projected_for_ceiling()
    assert corrected > raw
    assert corrected == raw + 275


def test_the_correction_does_not_grow_with_the_conversation(workspace: Workspace) -> None:
    """Measured over a 3x range of conversation sizes: the gap is flat at 263-314 tokens. A
    ratio fitted to the smallest reserves three times what the largest needs."""
    session = _session_with_drift(workspace, [(1000, 1275)])
    session._messages = [{"role": "user", "content": "x" * 40_000}]
    assert session._projected_for_ceiling() - session._projected() == 275


def test_a_tool_is_offered_only_the_room_that_really_remains(workspace: Workspace) -> None:
    """`room` decides whether a file is refused as too large. Handing a tool room computed
    from an estimate the backend has already exceeded is how a part overflows its window."""
    session = _session_with_drift(workspace, [(1000, 1200)])
    session._messages = [{"role": "user", "content": "x" * 400}]
    assert session._ceiling - session._projected_for_ceiling() < (
        session._ceiling - session._projected()
    )


def test_no_git_reports_what_it_wrote_rather_than_nothing() -> None:
    """--no-git has no diff to ask and no snapshot to list, and used to answer "nothing".

    That was not "we do not know" but a positive claim, printed as "Nothing was written" under
    a run that had just correctly edited both its files. It also went to the verifier as the
    set of written files, so --no-git quietly turned the syntax check off: a run could break
    every file in the tree and be told there was nothing in a format it could parse.
    """
    from pharos.agent.runner import written_by_parts

    def part(*written: str) -> PartResult:
        return PartResult(
            text="",
            steps=1,
            files_written=list(written),
            peak_tokens=0,
            reported_tokens=None,
            stopped_early=False,
        )

    assert written_by_parts([]) == []
    assert written_by_parts([part()]) == []
    # Deduplicated across parts, and one spelling: a repair part rewrites what a plan part did.
    got = written_by_parts([part(r"src\a.py", "src/b.py"), part("src/a.py")])
    assert got == ["src/a.py", "src/b.py"]
