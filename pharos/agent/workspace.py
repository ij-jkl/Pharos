"""Where a run is allowed to read and write, and the git safety net around it.

Two independent guards, because they fail differently:

* ``Workspace`` confines every path a model asks for to the folder the run was started in.
  A model that emits ``../../.ssh/id_rsa`` or ``C:\\Windows\\System32\\drivers\\etc\\hosts``
  gets an error string back, not a file. This is resolved-path containment, not string
  matching, so ``a/../../b`` cannot walk out either.
* ``git_guard`` refuses to start on a dirty tree and puts the run on its own branch. The
  agent writes real files; the only reason that is safe is that ``git diff`` can undo all of
  it, and that is only true if the tree was clean when it started.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Collection
from datetime import datetime
from pathlib import Path

from pharos.paths import normalise_display

# Never readable by a run, whatever the scope says: a model that asks for these is either
# confused or being steered, and neither case has a good outcome. Matched on any path
# component so `foo/.env` is caught as well as `.env`.
# ".pharos" holds the undo snapshot, which sits INSIDE the workspace and therefore inside
# everything the agent can see. A model that reads the pre-run copy of the file it is halfway
# through editing gets stale content that looks authoritative, and writing there would destroy
# the only way back. The snapshot writer copies with shutil and never comes through here, so
# denying the name costs nothing.
_DENIED_NAMES = frozenset(
    {".env", ".git", ".ssh", ".aws", "id_rsa", "credentials", ".pharos"}
)

# Windows resolves these names to hardware devices in EVERY directory, and with any extension:
# `src/con.py` is the console, not a file. Writing to one is not an error -- it succeeds,
# `exists()` returns True afterwards, and the directory is empty, because the bytes went to the
# device. Measured exactly that way: a 49-character write to `<root>/NUL` returned normally and
# read back as "".
#
# So a model asked to create `aux.py` or `con.py` -- ordinary names for "auxiliary" and
# "configuration", legal on Linux and in any repository written there -- would have its write
# reported as landing, and vanish. The audit catches it as a write the disk does not show, which
# is the audit working, but coverage still counts the file and `--no-audit` turns the only
# witness off. `COM1` and `LPT1` are worse than silent: they open a serial or printer port.
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(1, 10)}
    | {f"lpt{n}" for n in range(1, 10)}
)

_BRANCH_PREFIX = "pharos-run"


def reserved_device(path: Path) -> str | None:
    """The Windows device a path would open, or None. Always None off Windows.

    Matched on the stem, because the reservation ignores the extension -- `con`, `con.py` and
    `con.tar.gz` are all the console. Checked on every component, since the name means the
    device wherever it appears in a path.

    Refused only on Windows: on Linux `aux.py` is a legal file that exists and works, and a
    repository written there is entitled to contain one. The same run of the same task is
    therefore allowed to differ between platforms here, which is the truth about the platforms
    rather than an inconsistency in Pharos.
    """
    if os.name != "nt":
        return None
    for part in path.parts:
        stem = part.split(".", 1)[0].strip().lower()
        if stem in _WINDOWS_DEVICE_NAMES:
            return stem.upper()
    return None


class WorkspaceError(Exception):
    """A path is outside the workspace, or otherwise not something a run may touch."""


class GitGuardError(Exception):
    """The repository is not in a state where an agent may safely write to it."""


class NotARepository(GitGuardError):
    """There is no git here at all — distinct from a repository that is merely dirty.

    The two want opposite treatment. A dirty repository must stop the run: git is the better
    undo and the user's own uncommitted work must not become indistinguishable from the
    agent's. No repository at all is just a folder, and refusing to work in one would make
    the tool useless for exactly the throwaway directories people try it on first — so that
    case falls back to a file snapshot instead.
    """


class Undo:
    """Copies a file aside the first time a run overwrites it, so there is always a way back.

    The git branch is the good undo. This is the one for a folder that is not a repository:
    originals are copied under ``.pharos/undo-<timestamp>/`` preserving their relative path,
    once each, before the first write. Later writes to the same file do not re-snapshot — the
    point is to restore what was there when the run STARTED, not the previous tool call.

    A file the run creates is recorded with no snapshot: restoring means deleting it, and a
    zero-byte placeholder would restore it as an empty file instead of removing it.
    """

    def __init__(self, root: Path, directory: Path) -> None:
        self.root = root
        self.directory = directory
        self.saved: dict[str, bool] = {}  # relative path -> did it exist before the run

    def before_write(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return
        key = relative.as_posix()
        if key in self.saved:
            return
        existed = path.is_file()
        self.saved[key] = existed
        if not existed:
            return
        target = self.directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    def original(self, key: str) -> str:
        """What this file held before the run, from the snapshot. Empty when it did not exist.

        Empty is the right answer for a file the run CREATED: diffed against nothing, it
        reads as wholly added, which is exactly what happened to it.
        """
        if not self.saved.get(key, False):
            return ""
        try:
            return (self.directory / key).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def restore_hint(self) -> str:
        created = [name for name, existed in self.saved.items() if not existed]
        lines = [f"Originals saved in {self.directory}"]
        if self.saved:
            lines.append(f"  restore: copy that folder back over {self.root}")
        if created:
            shown = ", ".join(sorted(created)[:4])
            more = f" (+{len(created) - 4} more)" if len(created) > 4 else ""
            lines.append(f"  files the run CREATED, delete to undo: {shown}{more}")
        return "\n".join(lines)


class Workspace:
    """The single folder a run may read and write, with every path resolved against it."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, raw: str) -> Path:
        """Resolve a model-supplied path inside the workspace, or raise WorkspaceError.

        ``strict=False``: a write to a file that does not exist yet is legitimate, and the
        containment check is about where the path POINTS, not whether it is there already.
        """
        candidate = self._locate(raw)
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(
                f"{raw!r} is outside the workspace ({self.root}). A run may only touch files "
                f"under the folder it was started in."
            )
        lowered = {part.lower() for part in candidate.parts}
        denied = lowered & _DENIED_NAMES
        if denied:
            raise WorkspaceError(f"{raw!r} is off limits ({', '.join(sorted(denied))}).")
        device = reserved_device(candidate)
        if device is not None:
            raise WorkspaceError(
                f"{raw!r} names {device}, which Windows resolves to a device in every folder. "
                f"A write there reports success and leaves nothing on disk. Pick another name."
            )
        return candidate

    def _locate(self, raw: str) -> Path:
        r"""Resolve a path, treating a backslash as a separator only when that finds something.

        On Windows a backslash IS the separator. On POSIX it is a legal character in a file
        name, so "src\alpha.py" is a different file that does not exist — and a model that has
        seen Windows-style paths in a prompt will emit them anywhere. CI found this: the same
        call succeeded on one runner and came back "not found" on the other.

        Rewriting every backslash unconditionally would be wrong on POSIX, where a file really
        can be called that. So the literal path is tried first and the separator reading is
        only used when it lands on something that actually exists — a fallback that cannot
        shadow a real file, only rescue a request that would otherwise dead-end.
        """
        first = self._as_path(raw)
        if chr(92) not in raw or first.exists():
            return first
        alternative = self._as_path(raw.replace(chr(92), "/"))
        return alternative if alternative.exists() else first

    def _as_path(self, raw: str) -> Path:
        asked = Path(raw)
        return (asked if asked.is_absolute() else self.root / asked).resolve()

    def display(self, path: Path) -> str:
        """The path as the report should show it: relative to the root, forward slashes."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False, encoding="utf-8"
    )


def current_branch(root: Path) -> str | None:
    """The ref a run started from, so the undo instruction names it instead of assuming.

    "main" was hardcoded into that advice, which is wrong on every repository that calls its
    default branch something else -- following it fails and leaves you stranded on the run
    branch, holding the changes but not the way back. Returns the short commit sha when HEAD
    is detached, which `git checkout` accepts just the same, and None when there is nothing
    to read.
    """
    name = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if name.returncode != 0:
        return None
    ref = name.stdout.strip()
    if ref and ref != "HEAD":
        return ref
    sha = _git(root, "rev-parse", "--short", "HEAD")
    return sha.stdout.strip() or None if sha.returncode == 0 else None


def _is_ours(key: str, ignore: Collection[str]) -> bool:
    """Exact match, or sitting under an ignored directory -- the audit's rule, verbatim."""
    return key in ignore or any(key.startswith(f"{prefix}/") for prefix in ignore)


def _porcelain_path(line: str) -> str:
    """The path out of one ``git status --porcelain`` line.

    A rename reads ``R  old -> new``; the new name is the one on disk. Git quotes a path
    holding unusual characters, and a quoted form simply will not match an ignore key --
    which fails towards refusing the run rather than towards writing over somebody's work.
    """
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return normalise_display(path.strip('"'))


def git_guard(
    root: Path, *, create_branch: bool = True, ignore: Collection[str] = ()
) -> str | None:
    """Verify the tree is clean and move the run onto its own branch. Returns the branch name.

    Raises GitGuardError rather than proceeding, in every ambiguous case. The agent is about
    to overwrite files with model output; "you can always git diff it" is the entire safety
    argument, and it is false the moment uncommitted work is already in the tree — a bad run
    would then be indistinguishable from the user's own changes.

    ``ignore`` names paths that are Pharos's own -- see ``pharos.agent.audit.own_paths``,
    which computes them from configuration rather than guessing at what a Pharos file looks
    like. Without it the run's own log, written into the workspace by the pre-flight, counts
    as the user's uncommitted work: `--dry-run` then leaves a tree that the very next
    `pharos run` refuses to start on, blaming the user for a file Pharos wrote. The audit
    has always excluded these; the guard asking the same question deserves the same answer.

    ``create_branch=False`` keeps the cleanliness check and skips the branch, for a caller
    that has already isolated the work (a throwaway clone, a worktree, CI).
    """
    probe = _git(root, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0:
        raise NotARepository(f"{root} is not a git repository")

    # `-uall` lists untracked files one by one instead of collapsing a directory to `dir/`.
    # Two reasons: an ignore key naming a path inside one of ours cannot match a collapsed
    # parent, and a whole untracked folder reported as "1 uncommitted change" understates
    # what the user is being asked to deal with. Ignored files are still ignored either way.
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0:
        raise GitGuardError(f"git status failed: {status.stderr.strip()}")
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    if ignore:
        dirty = [line for line in dirty if not _is_ours(_porcelain_path(line), ignore)]
    if dirty:
        raise GitGuardError(
            f"the working tree has {len(dirty)} uncommitted change(s). Commit or stash them "
            f"first: a run's edits have to be separable from yours, or reviewing the diff "
            f"afterwards tells you nothing."
        )

    if not create_branch:
        return None

    branch = f"{_BRANCH_PREFIX}/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    made = _git(root, "checkout", "-b", branch)
    if made.returncode != 0:
        raise GitGuardError(f"could not create branch {branch}: {made.stderr.strip()}")
    return branch


def changed_files(root: Path, *, ignore: Collection[str] = ()) -> list[str]:
    """Paths the run has modified, for the closing report. Empty when git is unavailable.

    ``ignore`` excludes Pharos's own files, for the same reason ``git_guard`` does: the run's
    log lives in the workspace, and counting it here reported "3 file(s) changed on disk" for
    a run that changed two -- and handed the .log to the syntax verifier, which then reported
    a written file in a format it could not parse.
    """
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0:
        return []
    paths = [_porcelain_path(line) for line in status.stdout.splitlines() if line.strip()]
    return [path for path in paths if not _is_ours(path, ignore)]
