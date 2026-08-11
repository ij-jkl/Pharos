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

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

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

_BRANCH_PREFIX = "pharos-run"


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
        return candidate

    def _locate(self, raw: str) -> Path:
        """Resolve a path, treating a backslash as a separator only when that finds something.

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


def git_guard(root: Path, *, create_branch: bool = True) -> str | None:
    """Verify the tree is clean and move the run onto its own branch. Returns the branch name.

    Raises GitGuardError rather than proceeding, in every ambiguous case. The agent is about
    to overwrite files with model output; "you can always git diff it" is the entire safety
    argument, and it is false the moment uncommitted work is already in the tree — a bad run
    would then be indistinguishable from the user's own changes.

    ``create_branch=False`` keeps the cleanliness check and skips the branch, for a caller
    that has already isolated the work (a throwaway clone, a worktree, CI).
    """
    probe = _git(root, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0:
        raise NotARepository(f"{root} is not a git repository")

    status = _git(root, "status", "--porcelain")
    if status.returncode != 0:
        raise GitGuardError(f"git status failed: {status.stderr.strip()}")
    if status.stdout.strip():
        changed = len(status.stdout.strip().splitlines())
        raise GitGuardError(
            f"the working tree has {changed} uncommitted change(s). Commit or stash them "
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


def changed_files(root: Path) -> list[str]:
    """Paths the run has modified, for the closing report. Empty when git is unavailable."""
    status = _git(root, "status", "--porcelain")
    if status.returncode != 0:
        return []
    return [line[3:].strip() for line in status.stdout.splitlines() if line.strip()]
