"""Ask a model what it thinks of the diff — and keep the answer away from every measurement.

Pharos proves a task fit, ran, and still builds — it does not judge the code. Every other
number a run produces is a measurement — coverage is counted, damage is parsed, drift is
compared against ``prompt_eval_count`` — and an opinion filed next to them borrows their
authority without having earned any of it.

So the opinion is admitted, and quarantined. Three rules, and they are the whole design:

1. **It is off unless asked.** ``pharos run --review``, never by default.
2. **It cannot change the verdict.** The scorecard's ``complete``, the exit code, coverage,
   damage and verification are all computed before the review runs, and none of them is shown
   it. A run that built and covered its files is a passing run whatever the review says.
3. **Every finding is checked in code before it is shown.** The same discipline ``--semantic``
   works under: the model proposes, and the proposal is validated against facts it did not
   supply. A finding must name a file this run actually changed, and a line inside a hunk the
   model was actually shown. One that names a file nobody touched, or a line the diff does
   not contain, is discarded and counted — a model asked to review code will invent a
   plausible line number, and a plausible line number is exactly what a reader trusts.

What is left after that is genuinely limited, and worth saying plainly: it is one local
model's reaction to a diff, with no repository context, no test run behind it, and no memory
of why the code is the way it is. It goes in its own panel, under its own heading, with the
run's verdict printed above it and already decided.

Sending: ``--review`` sends the diff of the files this run changed to the backend configured
in ``pharos.toml`` — the same backend that just wrote them. Nothing else leaves, and without
the flag nothing does.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from pharos.config import PharosConfig
from pharos.paths import normalise_display

_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)
_CONTEXT_LINES = 3
_REPLY_TOKENS = 900
# A diff bigger than this is not one a 9B model reviews usefully in a single pass; it is
# reported as unreviewed rather than truncated, for the same reason read_file refuses to cut
# a file down to whatever is left of the window.
_MAX_FILE_TOKENS = 6_000
_NULL = "/dev/null"

SEVERITIES = ("bug", "risk", "note")

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "note": {"type": "string"},
                },
                "required": ["file", "line", "severity", "note"],
            },
        },
    },
    "required": ["findings"],
}

_INSTRUCTION = """\
You are reviewing a diff. The code below was just written by a model working from a task \
description, and you are being asked whether it looks correct.

Report only things you can point at IN THE DIFF. For each one give:
- "file": one of the file names listed, spelled exactly as it is given
- "line": a line number that appears in that file's diff, on a line the diff added or changed
- "severity": "bug" for something that is wrong, "risk" for something that may be, "note" \
for anything else worth a reader's attention
- "note": one sentence, at most 30 words

Do not report style preferences, missing tests, or anything you cannot point at a line for. \
An empty list is a valid and useful answer. Do not invent a file name or a line number: a \
finding that does not match the diff is discarded.

Reply with JSON only, in exactly this shape:
{"findings": [{"file": "...", "line": 1, "severity": "note", "note": "..."}]}\
"""


@dataclass(frozen=True, slots=True)
class Hunk:
    """One changed region, numbered in the NEW file — the only numbering a finding may use."""

    start: int
    end: int

    def holds(self, line: int) -> bool:
        return self.start <= line <= self.end


@dataclass(frozen=True, slots=True)
class FileDiff:
    display: str
    text: str
    hunks: tuple[Hunk, ...]

    def holds(self, line: int) -> bool:
        return any(hunk.holds(line) for hunk in self.hunks)


@dataclass(frozen=True, slots=True)
class Finding:
    file: str
    line: int
    severity: str
    note: str


@dataclass(frozen=True, slots=True)
class Review:
    """One model's opinion of one run's diff, and everything thrown away getting to it."""

    findings: list[Finding] = field(default_factory=list)
    reviewed: list[str] = field(default_factory=list)  # files actually shown to the model
    unreviewed: list[str] = field(default_factory=list)  # too large, or nothing to show
    discarded: int = 0  # findings that failed the check — see the module docstring
    note: str | None = None  # why the review is thin, or absent

    @property
    def ran(self) -> bool:
        return bool(self.reviewed)

    @property
    def by_severity(self) -> dict[str, int]:
        return {name: sum(1 for f in self.findings if f.severity == name) for name in SEVERITIES}


# --- collecting the diff ----------------------------------------------------------------------


def collect(
    root: Path, paths: list[str], *, originals: dict[str, str] | None = None
) -> list[FileDiff]:
    """The diff of each changed file, the new content on the right.

    ``originals`` is the pre-run text of each file, for a workspace that is not a repository —
    the snapshots the undo directory already holds. With none, git is asked instead. A file
    neither can describe is simply absent from the result, and reported as unreviewed.
    """
    out: list[FileDiff] = []
    for display in paths:
        text = (
            _from_originals(root, display, originals)
            if originals is not None
            else _from_git(root, display)
        )
        if text and text.strip():
            out.append(FileDiff(display=display, text=text, hunks=hunks_of(text)))
    return out


def _from_git(root: Path, display: str) -> str | None:
    tracked = _git(root, "diff", f"--unified={_CONTEXT_LINES}", "--", display)
    if tracked and tracked.strip():
        return tracked
    # A file the run CREATED is untracked, and `git diff` says nothing about it at all.
    # Comparing against the null device is how git itself renders that case as a diff.
    return _git(root, "diff", "--no-index", f"--unified={_CONTEXT_LINES}", "--", _NULL, display)


def _from_originals(root: Path, display: str, originals: dict[str, str]) -> str | None:
    try:
        after = (root / display).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    before = originals.get(normalise_display(display), "")
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{display}",
            tofile=f"b/{display}",
            n=_CONTEXT_LINES,
        )
    )


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def hunks_of(diff_text: str) -> tuple[Hunk, ...]:
    """The line ranges of the NEW file that this diff actually shows.

    A finding is only allowed to point inside one of these, and that is the whole check. A
    model that says "line 412" about a file whose diff stops at 88 has not read the diff, and
    a reader who follows that number lands somewhere unrelated and believes what they read.
    """
    found: list[Hunk] = []
    for match in _HUNK.finditer(diff_text):
        start = int(match.group(1))
        # An absent count means one line (`@@ -1 +1 @@`); an explicit `,0` means the hunk adds
        # nothing to the new file at all. Those are different, and `max(length, 1)` read them
        # the same way -- so a pure deletion, which git and difflib both emit as `+N,0` when a
        # run empties a file, produced a one-line hunk at N and a finding pointing there passed
        # the one check that exists to catch exactly that. A hunk showing no new lines can hold
        # no line number, so it contributes none.
        length = int(match.group(2)) if match.group(2) is not None else 1
        if length == 0:
            continue
        found.append(Hunk(start=start, end=start + length - 1))
    return tuple(found)


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None
    # --no-index exits 1 when the two files differ, which is the successful case here.
    return done.stdout if done.returncode in (0, 1) else None


# --- asking ------------------------------------------------------------------------------------


def batches(
    diffs: list[FileDiff], budget: int, count: Callable[[str], int]
) -> tuple[list[list[FileDiff]], list[str]]:
    """Pack the diffs into calls that fit ``budget``. Returns (batches, too large to review).

    A file whose own diff will not fit is never cut down into one that does. Half a diff
    reviewed as though it were whole is the same failure as a half-read file presented as a
    file, and this project has already decided how it feels about that.
    """
    packed: list[list[FileDiff]] = []
    oversized: list[str] = []
    current: list[FileDiff] = []
    spent = 0
    for diff in diffs:
        size = count(diff.text)
        if size > min(budget, _MAX_FILE_TOKENS):
            oversized.append(diff.display)
            continue
        if current and spent + size > budget:
            packed.append(current)
            current, spent = [], 0
        current.append(diff)
        spent += size
    if current:
        packed.append(current)
    return packed, oversized


def request_payload(config: PharosConfig, diffs: list[FileDiff], model: str) -> dict[str, Any]:
    """The exact body sent to ``/api/chat``. Separated out so a test can read it."""
    listing = "\n\n".join(f"--- {d.display} ---\n{d.text}" for d in diffs)
    names = ", ".join(d.display for d in diffs)
    content = f"{_INSTRUCTION}\n\nFiles in this diff: {names}\n\n{listing}\n"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "format": _SCHEMA,
        # Same reasoning as the semantic grouper: the answer IS the reasoning here, and a
        # thinking budget spent before it starts is budget the findings never get.
        "think": False,
        "options": {"temperature": 0.0, "seed": 0, "num_predict": _REPLY_TOKENS},
    }
    if config.num_ctx is not None:
        payload["options"]["num_ctx"] = config.num_ctx
    return payload


def ask(config: PharosConfig, diffs: list[FileDiff], model: str) -> tuple[str | None, str | None]:
    """Put one batch to the backend. Returns ``(reply, error)``, and never raises."""
    try:
        with httpx.Client(base_url=config.backend_url, timeout=_TIMEOUT) as client:
            response = client.post("/api/chat", json=request_payload(config, diffs, model))
            if response.status_code >= 400:
                return None, f"backend returned HTTP {response.status_code}"
            data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        detail = str(exc).strip()
        return None, f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    if isinstance(data, dict) and data.get("error"):
        return None, str(data["error"])
    if isinstance(data, dict) and data.get("done_reason") == "length":
        return None, "the reply was cut off at its length limit"
    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return None, "the backend returned an empty reply"
    return content, None


# --- checking what came back --------------------------------------------------------------------


def parse(text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Read a reply into raw finding dicts. Shape only — the checking happens in ``validate``."""
    raw = _json_object(text)
    if raw is None:
        return [], "the reply was not JSON"
    try:
        data = json.loads(raw)
    except ValueError:
        return [], "the reply was not valid JSON"
    if not isinstance(data, dict):
        return [], "the reply was not a JSON object"
    findings = data.get("findings")
    if not isinstance(findings, list):
        return [], "the reply carried no findings array"
    return [item for item in findings if isinstance(item, dict)], None


def validate(raw: list[dict[str, Any]], diffs: list[FileDiff]) -> tuple[list[Finding], int]:
    """Keep only findings that point at a real file and a line the model was actually shown.

    Returns ``(kept, discarded)``. Nothing here trusts the model for anything checkable: the
    file must be one of the ones sent, the severity one of the three asked for, and the line
    must fall inside a hunk of that file's own diff.
    """
    by_name = {d.display: d for d in diffs}
    by_normal = {normalise_display(d.display): d for d in diffs}
    kept: list[Finding] = []
    discarded = 0
    for item in raw:
        name = str(item.get("file") or "")
        target = by_name.get(name) or by_normal.get(normalise_display(name))
        severity = str(item.get("severity") or "").strip().lower()
        note = " ".join(str(item.get("note") or "").split())
        line = item.get("line")
        if (
            target is None
            or severity not in SEVERITIES
            or not note
            # bool is an int in Python, and `"line": true` is not a line number.
            or not isinstance(line, int)
            or isinstance(line, bool)
            or not target.holds(line)
        ):
            discarded += 1
            continue
        kept.append(Finding(file=target.display, line=line, severity=severity, note=note[:300]))
    return kept, discarded


def _json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else None
