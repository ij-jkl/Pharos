"""The tool catalogue a run exposes, and the enforcement that makes its projection a bound.

The splitter's promise is that a part fits. That promise survives contact with a real model
only if the part physically cannot read outside its scope — an instruction not to is a
request, and a 9B model under a long prompt will not always honour it. So scope lives here,
in the dispatcher, and a read of a deferred file comes back as a refusal the model can act on
rather than as content that quietly blows the window.

Two refusals matter, and both are deliberate non-features:

* **Out of scope.** Returned as a tool result, not an exception: the model is told which part
  owns the file and asked to note it and continue. A crash would lose the work already done.
* **Too large for what is left.** A file that does not fit the remaining window is NOT
  truncated. Truncation is the silent context loss Pharos exists to make visible, and doing
  it here to keep a run alive would be the exact behaviour the project refuses in the proxy.
  The model is told the size and the remaining room and left to decide.

Two ways to write, and which one a model reaches for decides whether a run does anything at
all. ``write_file`` takes the complete contents — correct for a new file, and hopeless for a
small change to a large one, because it makes the model JSON-escape an entire source file
inside a single string argument. A 14B asked to do that for a hundred-line C# file routinely
gives up and describes the edit instead, producing a run that reads everything and writes
nothing. ``replace_lines`` takes only the lines being changed, so reads are returned with
line numbers to address, and it is the tool the prompt steers towards.

Diff and patch formats would be more economical still, but small models produce malformed
hunks often enough that the failure becomes "the patch did not apply" rather than "the change
was wrong" — and the second is the one a user can act on.

Where a part owns only a line RANGE of a file, whole-file writes are refused outright: a part
that can see 400 of 4,000 lines must not rewrite the other 3,600 from content it never read.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pharos.agent.ledger import SAMPLE_LINES, FileChange, added_lines
from pharos.agent.workspace import Undo, Workspace, WorkspaceError
from pharos.paths import normalise_display
from pharos.preflight.content import BinaryFile, read_countable
from pharos.preflight.split import PartFile

# Entries a directory listing never shows: noise that costs tokens and teaches the model
# nothing about the task.
# How much of a just-edited file is echoed back with its new numbers, so the next edit on
# the same file does not need a whole re-read. Context either side, and a cap past which
# echoing costs more than the read it saves -- see ToolBox._renumbered.
_ECHO_CONTEXT = 3
_ECHO_MAX_LINES = 40

_HIDDEN = frozenset({".git", "__pycache__", ".venv", "node_modules", ".mypy_cache",
                     ".pytest_cache", ".ruff_cache", ".pharos"})

_MAX_LIST_ENTRIES = 200

# Writes to one file before a part is told to leave it alone. Observed live: a part called
# replace_lines on the same file a dozen times in a row. Every call succeeded, so the
# consecutive-failure breaker never saw it, and every edit moved the lines underneath the
# model's map so it kept "fixing" what it had just changed — while the other files it owned
# were never opened. Churn on one file is the coverage failure in slow motion.
_MAX_WRITES_PER_FILE = 5

_logger = logging.getLogger("pharos.agent")

# How a caller counts tokens; injected so the dispatcher never reaches for a tokenizer itself.
Counter = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class ScopeEntry:
    """One file a part may touch, whole or as the line range the splitter assigned it."""

    display: str
    line_start: int | None = None
    line_end: int | None = None

    @property
    def is_slice(self) -> bool:
        return self.line_start is not None


@dataclass
class ToolResult:
    """What a tool call produced: the text handed back to the model, plus what it did."""

    text: str
    ok: bool = True
    wrote: str | None = None  # display path, set only on a successful write
    refused_scope: str | None = None  # the out-of-scope path, when that is why it failed
    # Tokens the result needed, when the refusal was purely about room. Set so the session can
    # tell "there is no space for this" apart from every other reason a call fails: the first
    # is answerable by compacting what the part has finished with, and the rest are not.
    needed_room: int | None = None


@dataclass
class ToolBox:
    """Dispatch for one part: a workspace, the part's scope, and what it actually touched.

    ``scope=None`` means unrestricted — the run fit in one window, so there is no division to
    enforce and the agent may read anything in the workspace.
    """

    workspace: Workspace
    scope: dict[str, ScopeEntry] | None = None
    # Set when the workspace is not a git repository: the only way back for those runs, so
    # every write goes through it first.
    undo: Undo | None = None
    files_written: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    scope_refusals: list[str] = field(default_factory=list)
    write_counts: dict[str, int] = field(default_factory=dict)
    # The lines each write ADDED, kept as the write happens. Both write paths hold the old
    # text and the new one at that moment, so this is exact and costs one diff of text
    # already in memory. It is what `pharos.agent.ledger` carries to the next part, and it
    # is Pharos's record rather than the model's account of itself.
    additions: dict[str, list[str]] = field(default_factory=dict)
    addition_totals: dict[str, int] = field(default_factory=dict)

    # -- catalogue -----------------------------------------------------------------------

    def catalogue(self) -> list[dict[str, Any]]:
        """The tools to advertise — the same four every part, so the overhead is a constant."""
        tools = [
            _fn(
                "read_file",
                "Read a UTF-8 text file from the workspace. Returns its full contents.",
                {"path": ("string", "Path relative to the workspace root.")},
            ),
            _fn(
                "write_file",
                "Write a file, replacing it entirely. Supply the COMPLETE new contents, not "
                "a diff or a fragment — whatever you send becomes the whole file.",
                {
                    "path": ("string", "Path relative to the workspace root."),
                    "content": ("string", "The complete new contents of the file."),
                },
            ),
            _fn(
                "list_dir",
                "List the files and folders in a workspace directory.",
                {"path": ("string", "Directory relative to the workspace root; '.' for root.")},
            ),
        ]
        tools.append(
            _fn(
                "replace_lines",
                "Replace an inclusive range of lines in a file. PREFER THIS over write_file "
                "for a small change to an existing file — you only send the lines you are "
                "changing, not the whole file. Line numbers are the ones shown by read_file, "
                "or by the renumbered window this tool returns after each edit — that window is "
                "already current, so a further edit inside it needs no re-read.",
                {
                    "path": ("string", "Path relative to the workspace root."),
                    "line_start": ("integer", "First line to replace, 1-based inclusive."),
                    "line_end": ("integer", "Last line to replace, inclusive."),
                    "content": ("string", "The replacement text for those lines."),
                },
            )
        )
        return tools

    # -- dispatch ------------------------------------------------------------------------

    def dispatch(
        self, name: str, arguments: dict[str, Any], *, room: int, count: Counter
    ) -> ToolResult:
        """Run one tool call. ``room`` is the token budget its result must fit inside."""
        try:
            if name == "read_file":
                return self._read(str(arguments.get("path", "")), room=room, count=count)
            if name == "write_file":
                return self._write(
                    str(arguments.get("path", "")), _as_text(arguments.get("content"))
                )
            if name == "list_dir":
                return self._list(str(arguments.get("path", ".")))
            if name == "replace_lines":
                return self._replace_lines(arguments)
        except WorkspaceError as exc:
            return ToolResult(f"Refused: {exc}", ok=False)
        except OSError as exc:
            return ToolResult(f"Failed: {exc}", ok=False)
        except Exception as exc:  # noqa: BLE001 - see below; this must not be narrowed
            # Arguments come from a model and can be any shape JSON allows. One that does not
            # match the schema is a bad tool call, not a reason to end a run that has already
            # done work: a real run died here with a traceback because a model sent `content`
            # as a list of lines. Hand the error back and let it try again.
            _logger.warning("tool %s failed on %r", name, arguments, exc_info=True)
            return ToolResult(
                f"Failed: {type(exc).__name__}: {exc}. Check the argument types against the "
                f"tool description and try again.",
                ok=False,
            )
        return ToolResult(
            f"Unknown tool {name!r}. Available: read_file, write_file, list_dir.", ok=False
        )

    def _note_write(self, display: str) -> ToolResult | None:
        """Count a write, and cut a part off once it is plainly churning on one file."""
        count = self.write_counts.get(display, 0) + 1
        self.write_counts[display] = count
        if count > _MAX_WRITES_PER_FILE:
            others = [p for p in (self.scope or {}) if p != display] if self.scope else []
            move_on = f" Files still yours: {', '.join(others)}." if others else ""
            return ToolResult(
                f"Refused: {display} has already been rewritten {count - 1} times in this "
                f"part. Further edits to it are being declined so the rest of the work is not "
                f"starved of the window.{move_on} Leave it as it is.",
                ok=False,
            )
        if display not in self.files_written:
            self.files_written.append(display)
        return None

    def _note_added(self, display: str, before: str, after: str) -> None:
        """Record the lines a successful write added to a file.

        Only a sample is kept -- a new 400-line file added 400 lines and the next part needs
        enough to match a convention, not the file. The total counts the whole diff, so the
        record can say it is showing a sample instead of implying it is showing everything.

        Blank lines are counted and not shown. They carry no convention a later part could
        match, and on the first live run of the record they were worse than useless: an edit
        that inserted a constant followed by two blank lines put those blanks in the record,
        the next part read them as part of the pattern to follow, and reproduced them -- two
        of that run's lint failures were blank-line churn copied faithfully from one file to
        the next. Showing what a later part should imitate means showing only lines worth
        imitating.
        """
        lines = added_lines(before, after)
        if not lines:
            return
        kept = self.additions.setdefault(display, [])
        room = SAMPLE_LINES - len(kept)
        if room > 0:
            kept.extend([line for line in lines if line.strip()][:room])
        self.addition_totals[display] = self.addition_totals.get(display, 0) + len(lines)

    def changes(self) -> list[FileChange]:
        """What this part changed, in the order it first touched each file."""
        return [
            FileChange(
                display=display,
                added=tuple(self.additions.get(display, ())),
                added_total=self.addition_totals.get(display, 0),
            )
            for display in self.files_written
        ]

    def _entry(self, display: str) -> ScopeEntry | None:
        """The scope entry for a path, or None when scope does not cover it."""
        if self.scope is None:
            return None
        return self.scope.get(normalise(display))

    def _check_scope(self, display: str) -> ToolResult | None:
        if self.scope is None or normalise(display) in self.scope:
            return None
        self.scope_refusals.append(display)
        owned = ", ".join(sorted(self.scope)) or "nothing"
        return ToolResult(
            f"Refused: {display} is not in this part's scope, and another part owns it. "
            f"This part may open only: {owned}. Do not try to reach it another way — say in "
            f"your hand-off that you needed it, and carry on with what is in scope.",
            ok=False,
            refused_scope=display,
        )

    def _read(self, raw: str, *, room: int, count: Counter) -> ToolResult:
        path = self.workspace.resolve(raw)
        display = self.workspace.display(path)
        refusal = self._check_scope(display)
        if refusal is not None:
            return refusal
        if path.is_dir():
            # Answer the question actually being asked instead of bouncing it. A model that
            # gets "not found" for a directory it can see in the task tends to stop rather
            # than retry with the right tool, and a whole run dies on a naming quibble.
            listing = self._list(raw)
            return ToolResult(
                f"{display} is a directory, so here is its listing instead. "
                f"Call read_file on one of these files.\n{listing.text}",
                ok=listing.ok,
            )
        if not path.is_file():
            return ToolResult(
                f"Not found: {display}. Call list_dir on the folder above it to see what "
                f"is actually there.",
                ok=False,
            )
        try:
            text = read_countable(path).text
        except BinaryFile:
            return ToolResult(f"{display} is not text; it has no contents to read.", ok=False)

        entry = self._entry(display)
        if entry is not None and entry.is_slice:
            lines = text.splitlines(keepends=True)
            text = "".join(lines[(entry.line_start or 1) - 1 : entry.line_end])
            display = f"{display} (lines {entry.line_start}-{entry.line_end})"

        if display not in self.files_read:
            self.files_read.append(display)
        # Numbered from the file's own first line, not from 1. A slice numbered 1,2,3 would
        # send replace_lines at the wrong place in the file — an edit landing hundreds of
        # lines from where the model looked.
        numbered = _with_line_numbers(text, start=(entry.line_start or 1) if entry else 1)
        size = count(numbered)
        if size > room:
            # Deliberately not truncated: see the module docstring.
            return ToolResult(
                f"Refused: {display} is {size:,} tokens and only {room:,} remain in this "
                f"part's window. It is not truncated, because a partial file you were not "
                f"told was partial is how a wrong answer gets written confidently. Work with "
                f"what you have already read, and note this file in your hand-off.",
                ok=False,
                needed_room=size,
            )
        total = len(text.splitlines())
        return ToolResult(
            f"--- {display} ({total} lines) ---\n{numbered}\n"
            f"--- end of {display}. The leading numbers are line numbers for replace_lines; "
            f"they are not part of the file and must never be written back. ---"
        )

    def _write(self, raw: str, content: str) -> ToolResult:
        path = self.workspace.resolve(raw)
        display = self.workspace.display(path)
        refusal = self._check_scope(display)
        if refusal is not None:
            return refusal

        entry = self._entry(display)
        if entry is not None and entry.is_slice:
            first, last = entry.line_start or 1, entry.line_end
            return ToolResult(
                f"Refused: this part owns only lines {first}-{last} of {display}, so a "
                f"whole-file write would destroy the {first - 1} lines above it and "
                f"everything below line {last} that you have not read. Use replace_lines.",
                ok=False,
            )
        if looks_numbered(content):
            return ToolResult(
                f"Refused: that content still has the NNN| line-number prefixes from the "
                f"read view. Those are not part of {display} — send the source lines "
                f"alone, without the numbers.",
                ok=False,
            )
        if not content.strip():
            return ToolResult(
                f"Refused: an empty write to {display} is almost never intended. Send the "
                f"complete file contents.",
                ok=False,
            )
        churn = self._note_write(display)
        if churn is not None:
            return churn
        before = path.read_text(encoding="utf-8") if path.is_file() else ""
        if self.undo is not None:
            self.undo.before_write(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline=_existing_newline(path))
        self._note_added(display, before, content)
        return ToolResult(f"Wrote {display} ({len(content.splitlines()):,} lines).", wrote=display)

    def _replace_lines(self, arguments: dict[str, Any]) -> ToolResult:
        path = self.workspace.resolve(str(arguments.get("path", "")))
        display = self.workspace.display(path)
        refusal = self._check_scope(display)
        if refusal is not None:
            return refusal
        entry = self._entry(display)
        try:
            start = int(arguments["line_start"])
            end = int(arguments["line_end"])
        except (KeyError, TypeError, ValueError):
            return ToolResult("Refused: line_start and line_end must be integers.", ok=False)
        if not path.is_file():
            return ToolResult(f"Not found: {display}", ok=False)

        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not 1 <= start <= end <= len(lines):
            return ToolResult(
                f"Refused: lines {start}-{end} are outside {display}, which has "
                f"{len(lines):,} lines.",
                ok=False,
            )
        if entry is not None and entry.is_slice:
            owned_start, owned_end = entry.line_start or 1, entry.line_end or len(lines)
            if start < owned_start or end > owned_end:
                return ToolResult(
                    f"Refused: this part owns lines {owned_start}-{owned_end} of {display}; "
                    f"{start}-{end} reaches outside that.",
                    ok=False,
                )
        replacement = _as_text(arguments.get("content"))
        if replacement and not replacement.endswith("\n"):
            replacement += "\n"
        if looks_numbered(replacement):
            return ToolResult(
                f"Refused: that content still has the NNN| line-number prefixes from the "
                f"read view. Those are not part of {display} — send the source lines "
                f"alone, without the numbers.",
                ok=False,
            )
        # Read the file's own ending BEFORE the snapshot copy and the write: read_text above
        # normalised CRLF to LF in memory, so writing back without this converts the whole file
        # and a three-line edit shows as 150 changed lines.
        churn = self._note_write(display)
        if churn is not None:
            return churn
        ending = _existing_newline(path)
        if self.undo is not None:
            self.undo.before_write(path)
        updated = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
        path.write_text(updated, encoding="utf-8", newline=ending)
        self._note_added(display, "".join(lines), updated)
        # Every line below the edit has just moved, and the model is still holding numbers from
        # a read taken before it. Editing top-to-bottom off a stale map lands the second change
        # in the wrong place — silently, because the tool call itself succeeds. Say the shift
        # out loud, with the direction, rather than trusting the model to track it.
        delta = len(replacement.splitlines()) - (end - start + 1)
        window, shown_to = _renumbered(display, updated, start, len(replacement.splitlines()))
        if delta == 0:
            moved = ""
        elif window:
            moved = (
                f" Lines after {end} have shifted by {delta:+d}. The window below is current; "
                f"below line {shown_to} the numbers from your earlier read are stale."
            )
        else:
            moved = (
                f" Lines after {end} have shifted by {delta:+d}; the numbers from your earlier "
                f"read of this file are stale below that point — read it again before editing "
                f"further down."
            )
        return ToolResult(
            f"Replaced lines {start}-{end} of {display}.{moved}{window}", wrote=display
        )


    def _list(self, raw: str) -> ToolResult:
        path = self.workspace.resolve(raw or ".")
        if not path.is_dir():
            return ToolResult(f"Not a directory: {self.workspace.display(path)}", ok=False)
        entries = sorted(
            (p for p in path.iterdir() if p.name not in _HIDDEN),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        shown = entries[:_MAX_LIST_ENTRIES]
        lines = [f"{'  ' if p.is_file() else '/ '}{p.name}" for p in shown]
        if len(entries) > len(shown):
            lines.append(f"... {len(entries) - len(shown):,} more not shown")
        body = "\n".join(lines) or "(empty)"
        return ToolResult(f"--- {self.workspace.display(path)}/ ---\n{body}")


def _fn(name: str, description: str, params: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """One OpenAI/Ollama-shaped function definition. Every parameter is required."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": kind, "description": text} for key, (kind, text) in params.items()
                },
                "required": list(params),
            },
        },
    }


def catalogue_text(tools: list[dict[str, Any]]) -> str:
    """The catalogue as the string to count.

    The backend serialises tools into the prompt in a template-specific way Pharos cannot see,
    so this is the JSON as sent — close to what the model receives, and stable enough to plan
    against. It is a floor on the catalogue's cost, in the same sense as every other Pharos
    number, and the run's own budget check leaves headroom above it.
    """
    return json.dumps(tools, separators=(",", ":"))


_LF = chr(10)
_CRLF = chr(13) + chr(10)


_NUMBERED_LINE = re.compile(r"^\s*\d+\| ")


def _renumbered(display: str, updated: str, start: int, written: int) -> tuple[str, int]:
    """The region just written, with its NEW numbers. Returns the text and its last line.

    This is the cheapest fix available for the thing that actually fills a part's window.
    Every replace_lines shifts the numbering below it, so a model holding numbers from an
    earlier read has to get fresh ones before touching the same file again -- and the only
    way to get them was to read the whole file back. Pharos was even telling it to.

    Measured on the six-part run in DESKTOP_VALIDATION §28: one part read a 344-token file
    SEVEN times and another six times, and the run spent about 15,541 tokens re-reading files
    already sitting in its window -- more than a single part's entire ceiling. Two parts hit
    that ceiling and handed off early.

    A few lines of context either side, so an edit next to the last one needs nothing further.
    Capped, because past a certain size echoing the region back costs more than the re-read it
    saves, and then the honest answer is the old advice: go and read it.
    """
    lines = updated.splitlines(keepends=True)
    last = start - 1 + written
    if written > _ECHO_MAX_LINES or last < start or not lines:
        return "", 0
    first = max(1, start - _ECHO_CONTEXT)
    last = min(len(lines), last + _ECHO_CONTEXT)
    if last < first:
        return "", 0
    body = _with_line_numbers("".join(lines[first - 1 : last]), start=first)
    return (
        f"\n--- {display} lines {first}-{last}, renumbered after the edit ---\n"
        f"{body}\n"
        f"--- end of window. These numbers are current, and like every read view they are "
        f"not part of the file and must never be written back. ---"
    ), last


def looks_numbered(text: str) -> bool:
    """True when content appears to carry the read view's line-number prefixes.

    Numbering reads is what makes replace_lines usable, and it creates exactly one new way to
    ruin a file: a model that pastes the display back writes "12| public void Foo()" as source.
    The prompt says not to, which is a request; this is the check. A clear majority of prefixed
    lines is the signal — a file that genuinely begins several lines with "3| " does not exist
    in any language, while a stray match in prose would be one line in twenty.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    return sum(bool(_NUMBERED_LINE.match(line)) for line in lines) > len(lines) * 0.7


def _as_text(value: Any) -> str:
    """Coerce a model-supplied content argument to text.

    The schema says string; models send a list of lines about as often, and one did exactly
    that in a real run. Joining is unambiguous and is what was meant — rejecting it would be
    technically correct and would waste a round trip to make the same edit.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return chr(10).join(str(item) for item in value)
    return "" if value is None else str(value)


def _with_line_numbers(text: str, start: int = 1) -> str:
    """Prefix each line with its number, so the model can address a range it wants to change.

    Without numbers the only edit a model can express is "here is the whole file again", which
    for anything sizeable it declines to attempt. The prefix costs a few percent in tokens and
    buys the small-edit path; the read result says plainly that the numbers are not part of
    the file, because a model that pastes them back has corrupted it.
    """
    lines = text.splitlines()
    width = len(str(start + len(lines))) if lines else 1
    return chr(10).join(f"{i:>{width}}| {line}" for i, line in enumerate(lines, start=start))


def _existing_newline(path: Path) -> str:
    """The line ending a file already uses, so rewriting it does not restyle every line.

    A model hands back newline-separated text whatever the file looked like on disk. Writing
    that as LF into a CRLF file turns a three-line addition into a diff where every single
    line shows as changed, which makes the result unreviewable — and reviewing the diff is the
    one thing a user must be able to do after an agent run. A real run against a C# project
    produced exactly that before this existed.

    New files get LF: the sane default, and what git normalises to anyway.

    Decided by the first terminator in the file. A mixed file is pathological either way, and
    the first line is a cheap, stable proxy for the majority.
    """
    try:
        head = path.read_bytes()[:8192]
    except OSError:
        return _LF
    index = head.find(b"\n")
    if index <= 0:
        return _LF
    return _CRLF if head[index - 1 : index] == b"\r" else _LF


def normalise(display: str) -> str:
    """One spelling for a path, so scope lookups cannot miss on separators alone.

    Kept as a name here because the scope machinery reads better for it; the rule itself lives
    in ``pharos.paths``, which is also where the semantic grouper gets it. It had been
    reimplemented there, and promptly grew the same bug this function exists to fix.
    """
    return normalise_display(display)


def scope_from_part_files(files: list[PartFile]) -> dict[str, ScopeEntry]:
    """Build the dispatcher's scope from a split part's files, keyed on one spelling."""
    return {
        normalise(f.display): ScopeEntry(
            display=normalise(f.display), line_start=f.line_start, line_end=f.line_end
        )
        for f in files
    }


def agent_overhead_tokens(workspace: Workspace, count: Counter) -> int:
    """What `pharos run` costs before any task text: its system prompt plus its catalogue.

    Measured, not estimated. The proxy has to LEARN this figure for Continue or Cursor by
    watching their traffic; Pharos wrote this client, so it can simply count what it is about
    to send. Both the planner and the session subtract it, from the same function, so they
    cannot drift apart.
    """
    from pharos.agent.session import system_prompt_for

    catalogue = ToolBox(workspace=workspace).catalogue()
    return count(catalogue_text(catalogue)) + count(system_prompt_for(workspace.root))


def workspace_root(target_folder: str | None) -> Path:
    return Path(target_folder).resolve() if target_folder else Path.cwd().resolve()
