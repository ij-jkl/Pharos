"""What the filesystem says happened, next to what Pharos believes it did.

Every other record a run keeps is a record of Pharos's own actions: the ledger holds what the
dispatcher saw land, the scorecard counts what the parts reported, and the damage list names
the files that stopped parsing. All of it is testimony from the same witness. If a write is
reported and never lands, or a file changes that no tool of ours touched, none of those
records can say so -- they are not looking at the disk, they are looking at us.

So the run indexes the tree: before it starts, after each part's tools have finished, and
again after the project's own checks have run. Three snapshots, two questions:

* what changed while the part worked, and did the part claim all of it? A change nobody
  claimed is UNATTRIBUTED -- an editor left open on the workspace, a git hook, a file
  generated as a side effect. It is not necessarily wrong. It is necessarily worth knowing,
  because every coverage figure in the scorecard is computed from claims.
* what changed while the CHECKS ran? A formatter wired into a test command, a snapshot test
  writing its snapshots, a build step touching generated sources. Those writes are real edits
  to the user's tree made during a Pharos run, and until v1.0 nothing recorded them at all.

Identity is (size, mtime_ns), not a content hash. The question here is "did this change",
which mtime answers for a few thousand files in the time hashing answers it for a few dozen;
a run pays this three times per part, and an audit that makes the run noticeably slower is
one people turn off.

What that gives up, stated exactly: a write is invisible to this if it leaves the file the
same LENGTH and lands on the same timestamp as the snapshot it is compared against. NTFS
stores 100 ns ticks but Windows advances its clock roughly every 15 ms, so the window is that
wide there rather than a nanosecond. Between two of these snapshots sits a whole model round
trip, which is four orders of magnitude longer, so the case is a fast tool call landing in
the same tick as the snapshot before it. Real, small, and worth knowing about rather than
worth hashing every file of every part three times to close.

Nothing here reads file CONTENT, and nothing here is sent anywhere. The audit is paths and
two integers per path, computed locally, reported locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from pharos.paths import normalise_display
from pharos.preflight.extract import IGNORED_DIRS

# A tree bigger than this is not indexed to the end. The number is generous -- a repository
# with more source files than this has other problems -- and being over it is reported rather
# than absorbed, because an audit that quietly stopped looking would be worse than no audit.
FILE_CAP = 20_000


@dataclass(frozen=True, slots=True)
class TreeIndex:
    """Every file under a root, by relative path, as (size, mtime_ns)."""

    entries: dict[str, tuple[int, int]] = field(default_factory=dict)
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class TreeDiff:
    created: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(sorted({*self.created, *self.modified, *self.deleted}))

    def __bool__(self) -> bool:
        return bool(self.paths)


@dataclass(frozen=True, slots=True)
class PartAudit:
    """One part, measured against the disk rather than against its own account of itself."""

    part: str
    changed: TreeDiff  # what moved on disk while the part's tools ran
    claimed: tuple[str, ...]  # what the dispatcher said it wrote
    unattributed: tuple[str, ...]  # changed, and no tool of this part claimed it
    absent: tuple[str, ...]  # claimed, and the disk does not show it
    out_of_scope: tuple[str, ...]  # changed, and this part was told to leave it alone
    by_checks: TreeDiff = field(default_factory=TreeDiff)  # moved while the checks ran
    truncated: bool = False

    @property
    def clean(self) -> bool:
        """The disk and the run agree, and nothing else wrote while the part was working."""
        return not (self.unattributed or self.absent or self.out_of_scope or self.by_checks)


def index_tree(root: Path, *, cap: int = FILE_CAP) -> TreeIndex:
    """Index every file under ``root``, pruned of the same vcs/venv/cache dirs as the rest.

    An unreadable entry is skipped rather than raised on: a file that vanished between the
    walk and the stat is a race, not a finding, and the audit must never be able to end a run.
    """
    entries: dict[str, tuple[int, int]] = {}
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            if len(entries) >= cap:
                truncated = True
                return TreeIndex(entries=entries, truncated=truncated)
            path = Path(dirpath) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            try:
                key = normalise_display(str(path.relative_to(root)))
            except ValueError:
                continue
            entries[key] = (stat.st_size, stat.st_mtime_ns)
    return TreeIndex(entries=entries, truncated=truncated)


def diff_trees(before: TreeIndex, after: TreeIndex) -> TreeDiff:
    """What moved between two indexes of the same tree."""
    return TreeDiff(
        created=tuple(sorted(set(after.entries) - set(before.entries))),
        modified=tuple(
            sorted(
                path
                for path, stamp in after.entries.items()
                if path in before.entries and before.entries[path] != stamp
            )
        ),
        deleted=tuple(sorted(set(before.entries) - set(after.entries))),
    )


def audit_part(
    part: str,
    *,
    changed: TreeDiff,
    claimed: list[str],
    scoped: list[str] | None,
    by_checks: TreeDiff | None = None,
    truncated: bool = False,
) -> PartAudit:
    """Reconcile one part's claims against the disk.

    ``scoped`` is the files the part was allowed to touch, or None for an unrestricted part.
    An unrestricted part cannot be out of scope -- there is no list to be outside of -- and
    saying so beats inventing a denominator, which is the same rule the coverage figure keeps.
    """
    said = {normalise_display(path) for path in claimed}
    moved = set(changed.paths)
    allowed = {normalise_display(path) for path in scoped} if scoped else None
    return PartAudit(
        part=part,
        changed=changed,
        claimed=tuple(sorted(said)),
        unattributed=tuple(sorted(moved - said)),
        absent=tuple(sorted(said - moved)),
        out_of_scope=(
            () if allowed is None else tuple(sorted(path for path in moved if path not in allowed))
        ),
        by_checks=by_checks or TreeDiff(),
        truncated=truncated,
    )
