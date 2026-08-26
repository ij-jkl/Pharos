"""The run's own record of what landed on disk, carried between parts by Pharos.

Every part starts from an empty conversation, so the only thing joining part 4 to part 1 used
to be the hand-off the model wrote. Measured across sixteen live runs (DESKTOP_VALIDATION §14)
that thread is unreliable in a specific, repeatable way: parts change files and then report
*"NO CHANGES NEEDED"*. `kept_the_thread` was false in every two-file run, and the next part was
handed a summary that was not merely thin but wrong.

Pharos does not have to ask. The dispatcher records each write as it succeeds, and both write
paths hold the old text and the new one at the moment they run, so the lines a part ADDED are
available exactly — no model, no diff of the working tree, no guess. That record is what this
module carries forward.

Two things it deliberately does not do.

It does not replace the model's hand-off. The prose carries intent — *why* a thing was done,
what was deferred, what looked wrong — and no mechanical record reconstructs that. The ledger
goes above it, labelled as Pharos's, and the model's own words follow unchanged.

It does not touch how continuity is SCORED. `thin_handoffs` still asks whether the model's
prose named a file its part changed, and `kept_the_thread` still fails when it did not. Folding
the ledger into either would turn a measurement of the model into a measurement of Pharos and
make both green by construction. The ledger changes what the next part is told; the scorecard
goes on reporting how well the model told it.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass, field

# How many of a file's added lines are kept for the record. A part that writes a new 400-line
# file added 400 lines and nobody downstream needs them: the next part needs enough to match a
# convention, not the file. Sampling is stated wherever it happens, never silent.
#
# Six to begin with, lowered to three after two live runs. The first added line is the change;
# what follows it is whatever else the write disturbed, and on a whole-file rewrite that was
# five lines of reformatted class body carried to the next part for nothing. The tail dilutes
# the signal and costs reserve, and a part that imitates what it is shown should be shown less
# rather than more.
SAMPLE_LINES = 3

# Added lines longer than this are cut. A minified bundle or a long data literal would eat the
# whole reserve on one line and tell the next part nothing it could use.
_LINE_CHARS = 120

# The largest share of `handoff_reserve` the ledger may take, so the model's own hand-off is
# never starved by Pharos's record of it. Halves cleanly and has no measurement behind it —
# it is a stated preference, like `semantic_max_extra_parts`.
LEDGER_SHARE = 0.5


def added_lines(before: str, after: str) -> list[str]:
    """The lines present in ``after`` and not in ``before``, in order.

    A real diff rather than a set difference: a line duplicated by an edit is an addition, and
    a line that merely moved is not. `difflib` is stdlib and exact, and this runs once per
    successful write on text already in memory.
    """
    old = before.splitlines()
    new = after.splitlines()
    out: list[str] = []
    for line in difflib.unified_diff(old, new, n=0, lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return out


@dataclass(frozen=True, slots=True)
class FileChange:
    """What one part did to one file, as Pharos recorded it while it happened."""

    display: str
    added: tuple[str, ...] = ()  # a sample, oldest first
    added_total: int = 0  # how many lines were added in all, sampled or not

    @property
    def sampled(self) -> bool:
        return self.added_total > len(self.added)


@dataclass(frozen=True, slots=True)
class Done:
    """One part's entry in the record."""

    label: str
    changes: tuple[FileChange, ...]


def names_its_work(text: str, files: list[str]) -> bool:
    """Does ``text`` mention the basename of any file in ``files``?

    The one implementation of this question. It was written twice before — once in the
    scorecard and once here — and this project has already paid for keeping two copies of a
    path comparison (see `pharos.paths`), so the scorecard imports it rather than repeating it.

    Length was the original measure and was wrong: real useful hand-offs run about fifteen
    tokens ("Added XML doc comments to `INoteRepository.cs` and `NoteRepository.cs`") while the
    useless ones are parts that wrote files and reported "NO CHANGES NEEDED". A threshold flags
    the first and waves the second through; naming separates them exactly.

    A part that changed nothing has nothing to name and is not held to it.
    """
    if not files:
        return True
    lowered = text.lower()
    return any(
        path.replace(chr(92), "/").rsplit("/", 1)[-1].lower() in lowered for path in files
    )


@dataclass
class Ledger:
    """The running record for one run. Append-only, and never written by a model."""

    entries: list[Done] = field(default_factory=list)

    def record(self, label: str, changes: list[FileChange]) -> None:
        """Add what one part did. A part that changed nothing adds nothing."""
        if changes:
            self.entries.append(Done(label=label, changes=tuple(changes)))

    @property
    def files(self) -> list[str]:
        """Every file the run has changed so far, first-touched order, deduplicated."""
        out: list[str] = []
        for entry in self.entries:
            for change in entry.changes:
                if change.display not in out:
                    out.append(change.display)
        return out

    def render(self, *, count: Callable[[str], int], budget: int) -> str:
        """The record as text for the next part, inside ``budget`` tokens.

        Degrades rather than truncating: the sample of added lines shrinks first, then it
        drops to filenames, then to a count. Every rung says what it is showing, so a next
        part is never handed a partial record it believes is complete. Below the last rung it
        returns nothing at all — an empty ledger is honest, a silently clipped one is not.
        """
        if not self.entries or budget <= 0:
            return ""
        for sample in range(SAMPLE_LINES, -1, -1):
            text = self._render(sample)
            if count(text) <= budget:
                return text
        names = self.files
        while names:
            text = self._names_only(names, len(self.files))
            if count(text) <= budget:
                return text
            names = names[:-1]
        return ""

    # -- rendering ---------------------------------------------------------------------

    def _render(self, sample: int) -> str:
        """Full form: every file, with up to ``sample`` of the lines added to it."""
        lines = [_HEADER]
        for change in self._merged():
            shown = change.added[:sample]
            note = _plural(change.added_total)
            if shown and change.added_total > len(shown):
                # "shown" rather than "first N shown": blank lines are counted in the total
                # and skipped in the sample, so these are not necessarily the first N.
                note += f", {len(shown)} shown"
            lines.append(f"{change.display}  ({note})")
            lines.extend(f"    {_clip(line)}" for line in shown)
        lines.append(_FOOTER)
        return chr(10).join(lines)

    def _names_only(self, names: list[str], total: int) -> str:
        """Last rung: names alone, in framing short enough to leave room for them.

        The long footer tells the next part to match the lines above it, which says nothing
        once there are no lines above it -- and it costs more tokens than the names it would
        be introducing. Both ends shrink here, so a small reserve gets a shorter record rather
        than no record at the exact point room is hardest to find.
        """
        more = f", and {total - len(names)} more" if total > len(names) else ""
        return chr(10).join([_TERSE_HEADER, ", ".join(names) + more, _TERSE_FOOTER])

    def _merged(self) -> list[FileChange]:
        """One entry per file across the whole run, in first-touched order.

        A file two parts both edited is one thing that happened to it, not two, and the next
        part cares about its state rather than its history.
        """
        order: list[str] = []
        added: dict[str, list[str]] = {}
        totals: dict[str, int] = {}
        for entry in self.entries:
            for change in entry.changes:
                if change.display not in added:
                    order.append(change.display)
                    added[change.display] = []
                    totals[change.display] = 0
                added[change.display].extend(change.added)
                totals[change.display] += change.added_total
        return [
            FileChange(
                display=name,
                added=tuple(added[name][:SAMPLE_LINES]),
                added_total=totals[name],
            )
            for name in order
        ]


def _plural(total: int) -> str:
    return f"+{total} line" if total == 1 else f"+{total} lines"


def _clip(line: str) -> str:
    stripped = line.rstrip()
    if len(stripped) <= _LINE_CHARS:
        return stripped
    return stripped[: _LINE_CHARS - 1] + chr(8230)


_HEADER = (
    "--- ALREADY DONE IN THIS RUN (recorded by Pharos as each write landed, not summarised "
    "by a model) ---"
)

_FOOTER = (
    "Those files are finished and belong to other parts: do not open or change them. Where "
    "your own work is the same kind of change, match what is above - the naming, the wording "
    "and the placement of this run are already decided.\n"
    "--- END OF RECORD ---"
)

_TERSE_HEADER = "--- ALREADY DONE IN THIS RUN (Pharos's record) ---"

_TERSE_FOOTER = (
    "Those belong to other parts: do not open or change them.\n--- END OF RECORD ---"
)
