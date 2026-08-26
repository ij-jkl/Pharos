"""Did the run break anything? Questions with exact answers, none of which ask a model.

The scorecard says a run covered its scope and stayed in its window. It has never said the
code still works, and the gap is not theoretical: a measured run wrote all six of its files,
scored 100% coverage, and left `sorted(total.items(), ...)` where the variable is `totals`.
COMPLETE, and a NameError. That defect is not subtle to a linter -- ruff calls it F821 in
milliseconds -- so the honest fix is to run the checks the project already has rather than to
start reasoning about code.

Nothing here interprets anything. It runs the project's own tools and reports their exit
codes, which keeps this on the same footing as coverage and drift: measured, reproducible,
and falsifiable by rerunning it yourself.

Three rules keep the number honest.

**A baseline first.** Every check runs BEFORE the first part as well as after. A suite already
red when the run started is reported as such and never counted against the run -- otherwise
Pharos would blame the model for a repository it walked into. This is the whole reason
verification can gate an exit code without being a nuisance.

**A check that cannot run is skipped out loud.** No tool on PATH, no configuration, a timeout:
each is reported by name as skipped. Silently passing a check that never executed would be the
one failure mode worse than not checking at all.

**Only the project's own tools.** Auto-detection fires solely when the repository configures a
tool AND that tool resolves on PATH -- Pharos never invents a build command for a project that
did not ask for one. `verify_commands` in pharos.toml overrides the lot, which is how any
ecosystem beyond the Python ones detected here gets checked.
"""

from __future__ import annotations

import ast
import json
import os
import shlex
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Parsers for the formats a syntax check can answer exactly. Anything else is reported as
# unchecked rather than assumed fine -- Pharos edits C#, TypeScript and Markdown too, and a
# green tick over a language nothing parsed would be a lie.
_PARSERS: dict[str, str] = {
    ".py": "python",
    ".json": "json",
    ".ipynb": "json",
    ".toml": "toml",
}

SYNTAX_CHECK = "syntax"

# Lines of a failing tool kept verbatim before the report excerpts it from both ends.
_EXCERPT_LINES = 6


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """One check, after the run, carrying whether it was already failing before it."""

    name: str
    ok: bool
    detail: str = ""
    skipped: str | None = None  # why it could not run; None when it ran
    pre_existing: bool = False  # it failed before the run too, so the run did not cause it
    # The baseline for this check never completed -- it timed out, or its tool was missing then
    # -- so there is no "before" to compare against. Failing now might be the run's doing or
    # might not, and blaming it on the strength of one measurement would be a guess.
    unattributable: bool = False
    # It failed before and after, but not with the same output. A pass/fail baseline cannot
    # tell "unchanged" from "made worse" on a repository that arrived red, and the honest
    # answer is to say the output moved rather than to claim either.
    changed_while_failing: bool = False

    @property
    def newly_broken(self) -> bool:
        """Failing now, passing before, and both measurements actually happened."""
        return (
            not self.ok
            and self.skipped is None
            and not self.pre_existing
            and not self.unattributable
        )


@dataclass(frozen=True, slots=True)
class Verification:
    """What the project's own checks said, once the run had finished."""

    checks: tuple[CheckOutcome, ...] = ()
    unchecked_files: tuple[str, ...] = ()  # written, but in no language we can parse

    @property
    def newly_broken(self) -> list[CheckOutcome]:
        return [check for check in self.checks if check.newly_broken]

    @property
    def already_failing(self) -> list[CheckOutcome]:
        return [c for c in self.checks if not c.ok and c.pre_existing and c.skipped is None]

    @property
    def worsened(self) -> list[CheckOutcome]:
        """Failing before and after, but not identically. Not charged to the run -- a coarse
        pass/fail baseline cannot prove the run caused the difference -- but not hidden."""
        return [c for c in self.checks if c.pre_existing and c.changed_while_failing]

    @property
    def compared(self) -> bool:
        """Every check that ran had a baseline to be judged against."""
        return not self.unattributable

    @property
    def unattributable(self) -> list[CheckOutcome]:
        """Failing, with no usable baseline to say whether this run caused it."""
        return [c for c in self.checks if not c.ok and c.unattributable and c.skipped is None]

    @property
    def skipped(self) -> list[CheckOutcome]:
        return [check for check in self.checks if check.skipped is not None]

    @property
    def passing(self) -> list[CheckOutcome]:
        return [check for check in self.checks if check.ok and check.skipped is None]

    @property
    def ran(self) -> bool:
        """At least one check actually executed, so the verdict below means something."""
        return any(check.skipped is None for check in self.checks)

    @property
    def ok(self) -> bool:
        """Nothing that was working before the run is broken after it."""
        return not self.newly_broken


def _parse(kind: str, text: str) -> str | None:
    """None when it parses, else the error, in the one form every parser can answer."""
    try:
        if kind == "python":
            ast.parse(text)
        elif kind == "json":
            json.loads(text)
        elif kind == "toml":
            tomllib.loads(text)
    except (SyntaxError, ValueError) as exc:
        return str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return None


def syntax_state(root: Path, paths: list[str]) -> dict[str, str | None]:
    """Whether each file parses right now: None for fine, a message for broken.

    A path that is absent or unreadable is left out entirely rather than recorded as broken.
    Before a run, a file the plan will create does not exist yet; calling that a failure would
    make every new file look like damage the run did.
    """
    state: dict[str, str | None] = {}
    for path in paths:
        kind = _PARSERS.get(Path(path).suffix.lower())
        if kind is None:
            continue
        target = root / path
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        state[path] = _parse(kind, text)
    return state


def syntax_check(
    root: Path, written: list[str], baseline: dict[str, str | None]
) -> tuple[CheckOutcome, tuple[str, ...]]:
    """Parse everything the run wrote, discounting whatever was already unparseable."""
    after = syntax_state(root, written)
    unchecked = tuple(
        path for path in written if _PARSERS.get(Path(path).suffix.lower()) is None
    )
    broken = {path: err for path, err in after.items() if err is not None}
    if not broken:
        if not after:
            return (
                CheckOutcome(
                    name=SYNTAX_CHECK,
                    ok=True,
                    skipped="no file written is in a format this can parse",
                ),
                unchecked,
            )
        return CheckOutcome(name=SYNTAX_CHECK, ok=True), unchecked

    # A file that did not parse before the run does not become the run's fault by being
    # written to; only a file that was fine and is not any more.
    caused = {path: err for path, err in broken.items() if baseline.get(path) is None}
    detail = "; ".join(f"{path}: {err}" for path, err in sorted((caused or broken).items()))
    return (
        CheckOutcome(
            name=SYNTAX_CHECK,
            ok=False,
            detail=detail,
            pre_existing=not caused,
        ),
        unchecked,
    )


def _configures(root: Path, *, tool: str) -> bool:
    """Does this repository configure `tool` itself? Detection never guesses on its behalf."""
    candidates = [root / f"{tool}.toml", root / f".{tool}.toml"]
    if any(path.is_file() for path in candidates):
        return True
    for name in ("pyproject.toml", "setup.cfg", "tox.ini", f"{tool}.ini"):
        target = root / name
        if not target.is_file():
            continue
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if f"[tool.{tool}" in text or f"[{tool}]" in text:
            return True
    return False


def detect_commands(root: Path) -> list[str]:
    """The project's own checks, when it configures them and they are installed.

    Deliberately short, and deliberately Python-only: these are the ones that could be tested
    against a real repository here. Inventing a `dotnet build` or an `npm test` for a project
    that never asked would be Pharos deciding what your build is. Any other ecosystem is one
    line of `verify_commands` away, and that path is exercised by the same code.
    """
    found: list[str] = []
    if _configures(root, tool="ruff") and shutil.which("ruff"):
        found.append("ruff check .")
    if (_configures(root, tool="pytest") or (root / "tests").is_dir()) and shutil.which("pytest"):
        found.append("pytest -q")
    return found


def tokenise(command: str) -> list[str]:
    r"""Split a command line without eating Windows path separators.

    ``shlex`` in POSIX mode treats a backslash as an escape, which quietly turns
    ``C:	ools\lint.exe`` into ``C:toolslint.exe`` and then reports the check as "not on
    PATH". Non-POSIX mode keeps the backslashes but leaves the quotes attached to the token,
    so they come off here -- subprocess re-quotes the arguments itself.
    """
    tokens = shlex.split(command, posix=os.name != "nt")
    if os.name != "nt":
        return tokens
    return [
        token[1:-1] if len(token) > 1 and token[0] == token[-1] and token[0] in "\"'" else token
        for token in tokens
    ]


def run_command(root: Path, command: str, *, timeout: float) -> CheckOutcome:
    """Run one check and report its exit code. Never raises; a check that dies is a skip."""
    try:
        parts = tokenise(command)
    except ValueError as exc:
        return CheckOutcome(name=command, ok=True, skipped=f"could not parse command: {exc}")
    if not parts:
        return CheckOutcome(name=command, ok=True, skipped="empty command")
    if shutil.which(parts[0]) is None:
        return CheckOutcome(name=command, ok=True, skipped=f"{parts[0]} is not on PATH")
    try:
        done = subprocess.run(
            parts,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return CheckOutcome(name=command, ok=True, skipped=f"timed out after {timeout:.0f}s")
    except OSError as exc:
        return CheckOutcome(name=command, ok=True, skipped=f"could not run: {exc}")
    if done.returncode == 0:
        return CheckOutcome(name=command, ok=True)
    stream = (done.stdout or "") + (done.stderr or "")
    return CheckOutcome(name=command, ok=False, detail=_excerpt(stream))


def _excerpt(stream: str) -> str:
    """The useful part of a failing tool's output, taken from both ends.

    Which end carries the answer depends on the tool: ruff leads with the diagnostic, pytest
    trails with the summary. Taking only the tail turned ruff's first error into three lines
    of source context with the rule code scrolled off, which is what the first run of this
    reported. So both ends, and a marker saying what was dropped rather than a silent cut.
    """
    lines = [line.rstrip() for line in stream.splitlines() if line.strip()]
    if len(lines) <= _EXCERPT_LINES:
        return chr(10).join(lines)
    head, tail = lines[:4], lines[-2:]
    hidden = len(lines) - len(head) - len(tail)
    return chr(10).join([*head, f"... {hidden} more line(s) ...", *tail])


def baseline(root: Path, commands: list[str], *, timeout: float) -> dict[str, CheckOutcome]:
    """How each check stood BEFORE the run. A red suite on arrival is not the run's doing.

    The whole outcome is kept, not just a pass/fail: a baseline that was itself skipped -- its
    tool missing, or timed out -- reports ``ok=True`` because a check that did not run must
    never be treated as a failure. Reducing that to a bool here would let a skipped baseline
    masquerade as a passing one, and the run would then be blamed for a regression nobody ever
    measured.
    """
    return {command: run_command(root, command, timeout=timeout) for command in commands}


def verify(
    root: Path,
    *,
    written: list[str],
    commands: list[str],
    command_baseline: dict[str, CheckOutcome],
    syntax_baseline: dict[str, str | None],
    timeout: float,
) -> Verification:
    """Run every check and say which of them the run itself broke.

    ``command_baseline`` and ``syntax_baseline`` are the same checks taken before the first
    part executed. Without them this could only report that a repository is red, which says
    nothing about the run and would make the exit code useless on any project with a failing
    test already in it.
    """
    checks: list[CheckOutcome] = []
    syntax, unchecked = syntax_check(root, written, syntax_baseline)
    checks.append(syntax)
    if not written:
        # Nothing was written, so nothing can have been broken. Running the suite again would
        # only re-measure the baseline at the cost of running it twice.
        for command in commands:
            checks.append(
                CheckOutcome(name=command, ok=True, skipped="the run wrote no files")
            )
        return Verification(checks=tuple(checks), unchecked_files=unchecked)

    for command in commands:
        outcome = run_command(root, command, timeout=timeout)
        if outcome.ok or outcome.skipped is not None:
            checks.append(outcome)
            continue
        before = command_baseline.get(command)
        was_failing = before is not None and not before.ok and before.skipped is None
        checks.append(
            CheckOutcome(
                name=outcome.name,
                ok=False,
                detail=outcome.detail,
                pre_existing=was_failing,
                unattributable=before is None or before.skipped is not None,
                changed_while_failing=(
                    was_failing and before is not None and before.detail != outcome.detail
                ),
            )
        )
    return Verification(checks=tuple(checks), unchecked_files=unchecked)


@dataclass(frozen=True, slots=True)
class Damage:
    """One file a part left unparseable that parsed before that part ran."""

    label: str  # the part that did it
    path: str
    error: str
    repaired_by: str | None = None  # a later part that made it parse again

    @property
    def outstanding(self) -> bool:
        return self.repaired_by is None


class SyntaxWatch:
    """Parse what each part wrote as it finishes, and remember which part broke what.

    The end-of-run check answers "is the project broken", which is the question that gates the
    exit code. It cannot answer "by whom", and on a divided run that is the more useful half:
    a five-part run that ends red tells you to read five diffs.

    Cheap enough to do every time. Only the files the part itself wrote are parsed, only in
    formats there is a parser for, and parsing a handful of files costs milliseconds against
    the minute a part takes -- so unlike re-running the project's whole suite between parts,
    this needs no budget, no timeout and no configuration.

    Attribution is against the state immediately BEFORE the part, not the run's baseline. A
    file part 2 broke and part 4 repaired is recorded as both, because charging part 4 for
    arriving at a file part 2 had already ruined would name the wrong part, and hiding the
    break because it did not survive to the end would hide a real thing that happened.
    """

    def __init__(self, root: Path, baseline: dict[str, str | None]) -> None:
        self._root = root
        self._state = dict(baseline)
        self._damage: list[Damage] = []

    def after_part(self, label: str, written: list[str]) -> list[Damage]:
        """Record what this part did to the files it wrote; return what it broke."""
        found: list[Damage] = []
        for path, error in syntax_state(self._root, written).items():
            was = self._state.get(path)
            if error is not None and was is None:
                # Broken now, and either fine before or newly created by this part. Both are
                # this part's doing; a file that arrived broken is not.
                found.append(Damage(label=label, path=path, error=error))
            elif error is None and was is not None:
                self._repair(label, path)
            self._state[path] = error
        self._damage.extend(found)
        return found

    def _repair(self, label: str, path: str) -> None:
        """Credit a later part with fixing an earlier one's damage, most recent first."""
        for index in reversed(range(len(self._damage))):
            entry = self._damage[index]
            if entry.path == path and entry.repaired_by is None:
                self._damage[index] = Damage(
                    label=entry.label,
                    path=entry.path,
                    error=entry.error,
                    repaired_by=label,
                )
                return

    @property
    def damage(self) -> list[Damage]:
        """Everything any part broke, in the order it happened, repairs noted."""
        return list(self._damage)

    @property
    def outstanding(self) -> list[Damage]:
        """What is still broken: nobody came back for it."""
        return [entry for entry in self._damage if entry.outstanding]
