"""Turn a prompt that EXCEEDS the budget into an ordered set of sub-prompts that fit.

The split is mechanical, and by default entirely local. Two shapes, chosen by what actually
blows the budget:

* **scope** — the task text fits in a part, but the files it names do not. Every part repeats
  the task verbatim and narrows the *scope* to a subset of the files (large files are cut into
  line ranges). The parts are independent slices of the same instruction.
* **text** — the pasted text alone is too big (a log, a spec, a transcript). Parts are ordered
  segments of that text, cut on paragraph then line boundaries, to be pasted in sequence; only
  the last part asks for the work to be done.

Honesty rules, same as the rest of Pharos:

* Every part carries a *projected* cost measured with the same tokenizer as the verdict —
  overhead + the rendered part text + the files that part scopes. It is still a FLOOR: it
  holds only while the agent respects the scope block, and a part that does not fit even
  alone is reported as such rather than quietly shipped.
* Nothing is dropped silently. A file too large for even an empty part, a segment that cannot
  be cut small enough — both surface in the plan.

One thing here is neither mechanical nor local, and is opt-in for both reasons.
``semantic=True`` asks the configured backend which files belong together — sending the task,
the filenames, their counts and the first five lines of each — and uses the answer *only* to
decide which part each file lands in. It cannot change a budget, a projection, a refusal or a
part's text, and a proposal that fails any check in ``pharos.preflight.semantic`` is thrown
away for the packer's own answer. The plan always states which grouping produced it and why.

Position packing remains the default, and not only for privacy: a plan you can reproduce on a
machine with no GPU, and get the same parts from twice, is worth more than a tidy one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from pharos.config import PharosConfig
from pharos.paths import normalise_display, resolve_display
from pharos.preflight.check import CheckReport, CountedFile, build_counter
from pharos.preflight.content import BinaryFile, read_countable
from pharos.preflight.semantic import FileBrief, GroupRequest, ask, brief, parse, validate

# Coarse room for the per-part scaffold, used only for the "can any split work at all?"
# pre-check. Each mode then MEASURES its own scaffold before packing (see _scaffold_cost):
# a guessed reserve that undershoots produces parts that miss the target by a few tokens,
# which is exactly the kind of quietly-wrong number Pharos exists to not produce.
_SCAFFOLD_RESERVE = 220
# Slack over the measured scaffold: rendering swaps labels between the in/out-of-scope lists
# and the part numbers gain digits, so the real body can differ from the probe by a little.
_SCAFFOLD_SLACK = 24

_MIN_PART_TOKENS = 200  # below this a "part" is scaffold with no room for content
# A scope part asks the agent for a hand-off of at most 10 lines and tells the user to paste it
# above the next part — so every part after the first carries input the part itself does not
# contain. Room for it is held back, and it is counted into those parts' projections: asking
# for text and then not counting it is precisely the accounting error this tool is against.
#
# How much room is `handoff_reserve` in pharos.toml, not a constant here. It started as a round
# 200; a real agent run then produced a 328-token hand-off from the same "at most 10 lines"
# instruction, which is how a guessed constant becomes a broken promise. See the config field
# for the measurements behind the default.


class SplitMode(Enum):
    SCOPE = "scope"  # parts narrow which files are in scope; the task text repeats
    TEXT = "text"  # parts are ordered segments of the pasted text
    NONE = "none"  # no plan: unnecessary, or impossible


class Grouping(Enum):
    """How the files ended up in the parts they did — always reported, never inferred.

    POSITION is the default and the fallback: first-fit in the order the prompt named things.
    SEMANTIC means a model proposed the grouping and that proposal passed every mechanical
    check in ``pharos.preflight.semantic``. The distinction is published because the two are
    not equally trustworthy, and a reader deserves to know which one they are looking at.
    """

    POSITION = "position"
    SEMANTIC = "semantic"


@dataclass(frozen=True, slots=True)
class PartFile:
    """One file in a part's scope, whole or as a line range."""

    display: str
    tokens: int
    line_start: int | None = None  # 1-based inclusive; None for a whole file
    line_end: int | None = None
    total_lines: int | None = None
    source_dir: str | None = None  # the named directory this file came from, if any

    @property
    def is_slice(self) -> bool:
        return self.line_start is not None

    def label(self) -> str:
        span = (
            "whole file"
            if not self.is_slice
            else f"lines {self.line_start}-{self.line_end} of {self.total_lines}"
        )
        origin = f" · from {self.source_dir}" if self.source_dir else ""
        return f"{self.display} ({span}{origin})"


@dataclass(frozen=True, slots=True)
class Part:
    index: int  # 1-based
    total: int
    body: str  # the sub-prompt, ready to paste
    files: list[PartFile]
    projected_tokens: int  # overhead + body + scoped file content, same ruler as the verdict
    fits: bool
    over_by: int  # 0 when it fits
    title: str = ""  # a model's name for this group, empty under position grouping


@dataclass(frozen=True, slots=True)
class SplitPlan:
    mode: SplitMode
    target_per_part: int  # the per-part ceiling the packer aimed at
    target_label: str  # where that ceiling came from
    parts: list[Part] = field(default_factory=list)
    reason: str | None = None  # why there is no plan, when mode is NONE
    notes: list[str] = field(default_factory=list)  # anything the plan could not honour
    handoff_reserve: int = 0  # tokens held back per part for the previous part's hand-off
    already_fits: bool = False  # no plan because none was needed — not a failure
    indeterminate: bool = False  # no plan because there was no budget to plan against
    grouping: Grouping = Grouping.POSITION  # which one produced these parts
    grouping_note: str | None = None  # why, whenever a model was asked — accepted or not
    reads_reserved: int = 0  # room held back for what the agent opens unprompted (--reserve-reads)

    @property
    def ok(self) -> bool:
        if self.already_fits:
            return True
        return bool(self.parts) and all(p.fits for p in self.parts)


def build_plan(
    config: PharosConfig,
    prompt: str,
    report: CheckReport,
    *,
    target: int | None = None,
    max_files: int | None = None,
    semantic: bool = False,
    reserve_reads: bool = False,
) -> SplitPlan:
    """Plan a split of ``prompt`` given its pre-flight ``report``.

    ``target`` overrides the per-part ceiling — the escape hatch for planning offline, when
    no backend was reachable to state a budget.

    ``max_files`` caps how many files land in one part regardless of how well they fit. Token
    budgets are the reason this module exists, but they are not the only limit a plan meets:
    an agent handed thirteen files that fit its window comfortably will read all thirteen and
    then describe the work instead of doing it. That ceiling belongs to the model, not the
    hardware, so nothing here can measure it — the cap is a stated preference, and callers
    that only want the token split leave it None and get exactly the previous behaviour.

    ``semantic`` asks the backend to propose which files belong together instead of packing
    them in the order they were named. It changes the grouping and nothing else: the budget,
    the projections and the refusals are identical either way, and a proposal that fails any
    check in ``pharos.preflight.semantic`` is discarded for the mechanical one. Turning it on
    can therefore change how a plan reads, never whether it is honest.

    ``reserve_reads`` takes what agents on this model historically opened unprompted (see
    ``pharos.calibration.estimate_agent_reads``) off the per-part ceiling, so a part still has
    room once the agent starts pulling in files nobody named. Off by default, and deliberately:
    every other number a part carries is a measurement of the part itself, and this one is a
    prediction about a client Pharos has not met yet. It buys smaller parts that survive
    contact, at the cost of more of them.
    """
    ceiling, label = _ceiling(report, target)
    if ceiling is None:
        return SplitPlan(
            mode=SplitMode.NONE,
            target_per_part=0,
            target_label="unknown",
            reason=(
                "no usable budget to split against — the backend is unreachable or no model "
                "is resident. Load the model, or pass --target N to plan against N tokens."
            ),
            indeterminate=True,
        )

    reads_reserved = 0
    reads_note: str | None = None
    if reserve_reads and report.reads is not None and report.reads.tokens > 0:
        reads_reserved = report.reads.tokens
        ceiling -= reads_reserved
        label = f"{label} less what the agent opens on its own"
        reads_note = (
            f"{reads_reserved:,} tokens held back from every part for what the agent opens "
            f"unprompted — {report.reads.provenance}. A prediction, not a measurement of these "
            f"parts: a run that stays inside its scope block simply finishes with room spare."
        )
    elif reserve_reads:
        reads_note = (
            "--reserve-reads asked for room the store cannot size yet: what an agent opens on "
            "its own is learned from whole conversations through the proxy, and there are not "
            "three to learn from. Parts are packed against the full ceiling, as usual."
        )

    # A prompt whose FLOOR fits can still need splitting: name a directory and the request is
    # the floor plus whatever of that directory the agent reads. Split against the ceiling —
    # the number that has to fit for the work to actually go through.
    #
    # A file cap overrides "nothing to split" outright: the whole point of it is the task that
    # fits and still will not get done.
    too_many = max_files is not None and len(_scope_entries(report)) > max_files
    if report.ceiling <= ceiling and not too_many:
        detail = (
            f"the floor ({report.floor:,}) already sits inside the {label} ({ceiling:,})"
            if not report.directory_tokens
            else (
                f"even reading every named directory in full ({report.ceiling:,}) stays inside "
                f"the {label} ({ceiling:,})"
            )
        )
        return SplitPlan(
            mode=SplitMode.NONE,
            target_per_part=ceiling,
            target_label=label,
            reason=f"nothing to split: {detail}",
            already_fits=True,
        )

    count, _ = build_counter(config, report.model)
    overhead = report.overhead_tokens
    fixed = overhead + _SCAFFOLD_RESERVE

    # Room a part has for content once the overhead and the scaffold are paid for.
    if ceiling - fixed < _MIN_PART_TOKENS:
        return SplitPlan(
            mode=SplitMode.NONE,
            target_per_part=ceiling,
            target_label=label,
            reason=(
                f"the fixed cost alone (client overhead {overhead:,} + part scaffold) leaves "
                f"under {_MIN_PART_TOKENS} tokens of the {label} ({ceiling:,}) for content — "
                f"no split can help. Raise num_ctx, or use a client with less overhead."
            ),
        )

    if report.prompt_tokens + fixed <= ceiling:
        return _reserving(
            _plan_scope(
                prompt,
                report,
                count,
                ceiling,
                label,
                config.handoff_reserve,
                max_files,
                config=config if semantic else None,
            ),
            reads_reserved,
            reads_note,
        )
    if semantic:
        # A text split cuts a pasted blob on paragraph boundaries. There is no set of files to
        # group, so there is nothing for a proposal to be about — said plainly rather than
        # letting --semantic look like it did something.
        return _reserving(
            _with_note(
                _plan_text(prompt, report, count, ceiling, label),
                "semantic grouping does not apply to a text split: the parts are ordered "
                "segments of one oversized input, and their order is the input's own",
            ),
            reads_reserved,
            reads_note,
        )
    return _reserving(
        _plan_text(prompt, report, count, ceiling, label), reads_reserved, reads_note
    )


def _with_note(plan: SplitPlan, note: str) -> SplitPlan:
    return replace(plan, grouping_note=note)


def _reserving(plan: SplitPlan, reserved: int, note: str | None) -> SplitPlan:
    """Record that the ceiling these parts were packed against had room taken out of it."""
    if note is None:
        return plan
    return replace(plan, reads_reserved=reserved, notes=[*plan.notes, note])


# --------------------------------------------------------------------------- scope mode


def _plan_scope(
    prompt: str,
    report: CheckReport,
    count: Callable[[str], int],
    ceiling: int,
    label: str,
    handoff_reserve: int,
    max_files: int | None = None,
    config: PharosConfig | None = None,
) -> SplitPlan:
    """Repeat the task in every part; bin-pack the named files across the parts.

    ``config`` is passed only when a semantic grouping was asked for; it carries the backend
    to ask and the ceilings to hold the answer to.
    """
    overhead = report.overhead_tokens
    notes: list[str] = []

    # The scaffold carries one line per file, so its cost depends on how the files were cut —
    # which depends on the room the scaffold leaves. Measure, cut, re-measure; it settles in
    # one or two rounds, and the loop is bounded either way.
    labels = [
        PartFile(display=entry.display, tokens=entry.tokens, source_dir=source).label()
        for entry, source in _scope_entries(report)
    ]
    scaffold = _scope_scaffold(prompt, labels, count)
    units: list[PartFile] = []
    per_part_content = 0
    for _ in range(3):
        # The hand-off is held back from every part, not just the ones that will carry it: a
        # part's room must not depend on where the packer happens to place it.
        per_part_content = ceiling - overhead - scaffold - handoff_reserve
        if per_part_content < _MIN_PART_TOKENS:
            return SplitPlan(
                mode=SplitMode.NONE,
                target_per_part=ceiling,
                target_label=label,
                reason=(
                    f"the task text plus the part scaffold and the client overhead "
                    f"({overhead:,}) already fill the {label} ({ceiling:,}) — there is no room "
                    f"left to put a file in. Shorten the prompt, or raise num_ctx."
                ),
            )
        units, notes = _scope_units(report, per_part_content, count)
        grown = _scope_scaffold(prompt, [u.label() for u in units], count)
        if grown <= scaffold:
            break
        scaffold = grown

    if not units:
        # Two very different situations, and telling the user the wrong one wastes their time:
        # nothing to narrow at all, versus files that exist but could not be cut (a notebook
        # bigger than a part, an unreadable file). The notes hold the specifics either way.
        reason = (
            "the prompt names no countable files, so there is no scope to narrow — the excess "
            "must come from the client overhead or from files the agent reads on its own, "
            "neither of which a split can control"
            if not notes
            else "every file named is too large for a part and none could be cut down; see below"
        )
        return SplitPlan(
            mode=SplitMode.NONE,
            target_per_part=ceiling,
            target_label=label,
            reason=reason,
            notes=notes,
        )

    bins = _position_bins(units, per_part_content, max_files)
    grouping = Grouping.POSITION
    titles = [""] * len(bins)
    grouping_note: str | None = None
    if config is not None:
        proposal = _semantic_bins(config, prompt, report, units, bins, per_part_content, max_files)
        grouping_note = proposal.note
        if proposal.bins is not None:
            bins, titles, grouping = proposal.bins, proposal.titles, Grouping.SEMANTIC

    all_labels = [u.label() for u in units]
    parts: list[Part] = []
    for i, group in enumerate(bins, start=1):
        scoped = {u.label() for u in group}
        deferred = [lbl for lbl in all_labels if lbl not in scoped]
        body = _render_scope_part(prompt, i, len(bins), group, deferred, title=titles[i - 1])
        parts.append(
            _finalise(
                i,
                len(bins),
                body,
                group,
                count,
                ceiling,
                report,
                # Part 1 has nothing pasted above it; every later part does.
                carried=handoff_reserve if i > 1 else 0,
                title=titles[i - 1],
            ),
        )
    sliced = {u.display: u for u in units if u.is_slice}
    if sliced:
        # A projection for a sliced part assumes the agent reads only those lines. Most
        # read_file tools take a path and nothing else, and hand back the whole file — so on
        # such a client the part costs the full file, not the slice. Said out loud, with the
        # number, because it is the difference between a plan and a plan that works.
        whole = sum(entry.tokens for entry, _ in _scope_entries(report) if entry.display in sliced)
        notes.append(
            f"{len(sliced)} file(s) are scoped as line ranges. That assumes your agent can "
            f"read a range; a tool that only reads whole files would pull {whole:,} tokens "
            f"for them instead of the slice, and the parts holding them would run over"
        )
    return SplitPlan(
        mode=SplitMode.SCOPE,
        target_per_part=ceiling,
        target_label=label,
        parts=parts,
        notes=notes,
        handoff_reserve=handoff_reserve,
        grouping=grouping,
        grouping_note=grouping_note,
    )


def _position_bins(
    units: list[PartFile], per_part_content: int, max_files: int | None
) -> list[list[PartFile]]:
    """First-fit in the order the prompt named things.

    Locality is a feature: a bin-packed optimum that scatters related files across parts is
    worse for the human reading it.

    One constraint on top: the slices of a single file may only move FORWARD through the
    parts. Plain first-fit will happily backfill lines 3879-4000 into part 1 next to lines
    1-1939, which packs marginally tighter and asks a reader to hold two disjoint windows of
    one file at once. Reading order is worth more than the odd saved part.
    """
    bins: list[list[PartFile]] = []
    room: list[int] = []
    last_bin: dict[str, int] = {}  # display -> the bin its previous slice landed in
    for unit in units:
        floor_bin = last_bin.get(unit.display, -1) + 1 if unit.is_slice else 0
        for i in range(floor_bin, len(room)):
            if max_files is not None and len(bins[i]) >= max_files:
                continue
            if unit.tokens <= room[i]:
                bins[i].append(unit)
                room[i] -= unit.tokens
                break
        else:
            i = len(bins)
            bins.append([unit])
            room.append(per_part_content - unit.tokens)
        if unit.is_slice:
            last_bin[unit.display] = i
    return bins


@dataclass(frozen=True, slots=True)
class _Grouped:
    """The outcome of asking. ``bins`` is None whenever the mechanical grouping should stand.

    ``note`` is never None and never empty: asking a model and not saying so is the one thing
    this feature is not allowed to do, so there is no way to construct a silent outcome.
    """

    note: str
    bins: list[list[PartFile]] | None = None
    titles: list[str] = field(default_factory=list)


def _declined(note: str) -> _Grouped:
    """A refusal, stated as the reason alone.

    The note does not repeat which grouping won. Every renderer prints that as a label beside
    it and the JSON carries it as its own field, so appending it here produced "grouped by
    position - the proposal was rejected ...; grouped by position" on every fallback.
    """
    return _Grouped(note=note)


def _semantic_bins(
    config: PharosConfig,
    prompt: str,
    report: CheckReport,
    units: list[PartFile],
    position_bins: list[list[PartFile]],
    per_part_content: int,
    max_files: int | None,
) -> _Grouped:
    """Ask the backend to group ``units``, then refuse the answer unless it survives everything.

    Most of the interesting cases end in a refusal, and each of them names itself in the note.

    One thing here is NOT checked, and saying which is the point: the model also chooses the
    ORDER of the parts, and nothing verifies that order is a real dependency order. It is asked
    for one — definitions before their users — but no static analysis backs that up, and a
    grouping whose part 2 needs something part 3 defines would be accepted. Every *quantity* is
    re-measured; the sequencing is taken on trust, exactly as the hand-off between parts always
    has been.
    """
    # Not necessarily the model doing the work: grouping is a different job and wants a
    # different model. See PharosConfig.semantic_model for what was measured.
    model = config.semantic_model or config.model
    if model is None:
        return _declined("semantic grouping needs a model in pharos.toml")
    sliced = [u for u in units if u.is_slice]
    if sliced:
        # A group of line ranges is not a group of ideas. "lines 1940-3878 of forward.py"
        # belongs where the previous slice left off and nowhere else, so the ordering is
        # already determined and a model has nothing to add but risk.
        return _declined(
            f"{len(sliced)} file(s) had to be cut into line ranges, whose order is fixed by "
            f"the file itself"
        )

    request = GroupRequest(
        task=prompt,
        files=tuple(_briefs(report, units)),
        target_parts=len(position_bins),
        max_parts=len(position_bins) + config.semantic_max_extra_parts,
        max_files=max_files,
        per_part_budget=per_part_content,
    )
    reply, error = ask(config, request, model)
    if reply is None:
        return _declined(f"the backend could not be asked ({error})")
    proposal, error = parse(reply)
    if proposal is None:
        return _declined(f"the proposal was rejected — {error}")
    rejection = validate(proposal, request)
    if rejection is not None:
        return _declined(f"the proposal was rejected — {rejection}")

    # Keyed on the canonical spelling for the same reason validate() compares on it: the model
    # answers in posix whatever it was shown, and a raw dict lookup would KeyError on Windows.
    by_name = {normalise_display(u.display): u for u in units}
    known = {key: key for key in by_name}
    proposed = [
        [by_name[key] for name in group.files if (key := resolve_display(name, known)) is not None]
        for group in proposal.groups
    ]
    bins, titles, split_count = _repair(
        proposed, [g.title for g in proposal.groups], per_part_content, max_files
    )
    # Re-checked after the repair, not before: splitting an oversized concern is what buys the
    # grouping, and it is also the only thing here that can grow the part count.
    if len(bins) > request.max_parts:
        return _declined(
            f"the proposal was rejected — keeping its groups intact needs {len(bins)} parts "
            f"against a ceiling of {request.max_parts}"
        )
    still_over = [i for i, group in enumerate(bins, start=1) if _content(group) > per_part_content]
    if still_over:
        # A single file larger than a part. Nothing to split it against here — the slicer
        # already declined this whole path — so the grouping goes with it.
        return _declined(
            f"the proposal was rejected — part(s) {', '.join(map(str, still_over))} would not "
            f"fit in {per_part_content:,} tokens even alone"
        )
    shape = (
        f"{len(bins)} parts, the same count position packing gave"
        if len(bins) == len(position_bins)
        else f"{len(bins)} parts where position packing gave {len(position_bins)}"
    )
    if split_count == 1:
        repaired = " 1 group was too big for one part and was split in order."
    elif split_count:
        repaired = f" {split_count} groups were too big for one part and were split in order."
    else:
        repaired = ""
    # A proposal can be accepted and still have decided nothing. Asked to group six files, a
    # model may return them as one group; repair then cuts that group in order, and the result
    # is position packing wearing the model's title. The provenance is still semantic — this
    # IS what it proposed — but calling it "grouped by meaning" and leaving it there would
    # credit a decision that was not made. Cheap to detect, so it is said.
    same_as_position = _partition(bins) == _partition(position_bins)
    identical = (
        " The partition is identical to position packing, so the model changed nothing."
        if same_as_position
        else ""
    )
    # Not "grouped by X": every renderer prints that as a label beside this, and the JSON has
    # it as its own field, so leading with it gave "grouped by meaning - grouped by qwen..."
    # on every accepted plan.
    return _Grouped(
        note=(
            f"{model} chose {shape}.{repaired}{identical} The grouping and the part order are "
            f"its own; every projection below is not"
        ),
        bins=bins,
        titles=titles,
    )


def _partition(bins: list[list[PartFile]]) -> list[list[str]]:
    """A grouping reduced to what it actually decided: which labels sit together, in order."""
    return [[unit.label() for unit in group] for group in bins]


def _repair(
    groups: list[list[PartFile]],
    titles: list[str],
    per_part_content: int,
    max_files: int | None,
) -> tuple[list[list[PartFile]], list[str], int]:
    """Cut any group too big for one part into consecutive parts, keeping the model's order.

    This is what makes the feature fire more than half the time. Measured, the models are
    specifically bad at the *packing* constraint: they group by concern, correctly, and then
    ignore the token ceiling they were handed. Rejecting the whole proposal for that threw away
    a right answer over arithmetic — a clean render/physics/audio split lost because physics
    happened to need two parts, replaced by three parts each spanning all three subsystems.

    Splitting is mechanical and order-preserving, so nothing new is trusted: the group's files
    stay in the sequence the model put them in and fill parts in that sequence. An oversized
    concern becomes two *consecutive parts of that concern*, which is the outcome anyone would
    have chosen by hand. Titles are carried onto the continuation parts and numbered, because a
    reader looking at two parts called "Physics Subsystem" would fairly wonder which is which.
    """
    out: list[list[PartFile]] = []
    out_titles: list[str] = []
    split_count = 0
    for group, title in zip(groups, titles, strict=True):
        chunks = _chunk(group, per_part_content, max_files)
        if len(chunks) > 1:
            split_count += 1
        for n, chunk in enumerate(chunks, start=1):
            out.append(chunk)
            out_titles.append(_numbered(title, n, len(chunks)))
    return out, out_titles, split_count


def _numbered(title: str, n: int, total: int) -> str:
    """``Physics Subsystem (2 of 3)`` — and nothing at all when there was no title to number.

    A model may return an empty title, or one that was not a string; both arrive here as "".
    Numbering that produces " (1 of 2)" with a leading space, which the renderer then treats as
    a real heading because it is truthy.
    """
    if total == 1:
        return title
    return f"{title} ({n} of {total})" if title else ""


def _chunk(
    group: list[PartFile], per_part_content: int, max_files: int | None
) -> list[list[PartFile]]:
    """One group as the fewest consecutive parts that each respect the budget and the cap."""
    chunks: list[list[PartFile]] = []
    current: list[PartFile] = []
    used = 0
    for unit in group:
        too_heavy = current and used + unit.tokens > per_part_content
        too_many = max_files is not None and len(current) >= max_files
        if too_heavy or too_many:
            chunks.append(current)
            current, used = [], 0
        current.append(unit)
        used += unit.tokens
    if current:
        chunks.append(current)
    return chunks


def _content(group: list[PartFile]) -> int:
    return sum(f.tokens for f in group)


def _briefs(report: CheckReport, units: list[PartFile]) -> list[FileBrief]:
    """Each unit as a name, a cost and its opening lines — for the ones that can be re-read.

    A file whose head cannot be recovered still goes in the question, name only. Dropping it
    would be a coverage failure of our own making, and the checks would then blame the model
    for a file it was never shown.
    """
    paths = {entry.display: entry.path for entry, _ in _scope_entries(report)}
    out: list[FileBrief] = []
    for unit in units:
        text: str | None = None
        path = paths.get(unit.display)
        if path is not None:
            try:
                text = read_countable(path).text
            except (OSError, BinaryFile):
                text = None
        out.append(brief(unit.display, unit.tokens, text))
    return out


def _scope_scaffold(prompt: str, labels: list[str], count: Callable[[str], int]) -> int:
    """Cost of everything a scope part carries besides file content: task text and scope block."""
    probe = _render_scope_part(prompt, 1, max(len(labels), 1), [], labels)
    return count(probe) + _SCAFFOLD_SLACK


def _scope_units(
    report: CheckReport, per_part_content: int, count: Callable[[str], int]
) -> tuple[list[PartFile], list[str]]:
    """Every named file as one unit, or as several line ranges when it is too big for a part.

    Files pulled in by a named DIRECTORY are units too: "refactor everything in src/" is the
    case a scope split exists for, and a plan that scoped nothing because the user wrote a
    directory instead of forty filenames would be a plan in name only.
    """
    units: list[PartFile] = []
    notes: list[str] = []
    for entry, source_dir in _scope_entries(report):
        if entry.tokens <= per_part_content:
            units.append(
                PartFile(display=entry.display, tokens=entry.tokens, source_dir=source_dir)
            )
            continue
        if entry.path is None:
            notes.append(f"{entry.display} is too large for one part and could not be re-read")
            continue
        slices, note = _slice_file(
            entry.display, entry.path, entry.tokens, per_part_content, count, source_dir
        )
        units.extend(slices)
        if note:
            notes.append(note)
    return units, notes


def _scope_entries(report: CheckReport) -> list[tuple[CountedFile, str | None]]:
    """Named files first, then each named directory's contents, each tagged with its origin."""
    entries: list[tuple[CountedFile, str | None]] = [(f, None) for f in report.files]
    for directory in report.directories:
        entries.extend((f, directory.display) for f in directory.files)
    return entries


def _slice_file(
    display: str,
    path: Path,
    total_tokens: int,
    budget: int,
    count: Callable[[str], int],
    source_dir: str | None = None,
) -> tuple[list[PartFile], str | None]:
    """Cut one oversized file into line ranges that each fit ``budget``.

    Lines are grown by a characters-per-token ratio (cheap) and then counted exactly (honest);
    an over-long span is shrunk proportionally until it fits or is a single line.
    """
    try:
        countable = read_countable(path)
    except (OSError, BinaryFile):
        return [], f"{display} is too large for one part and could not be re-read to slice"
    if countable.note is not None:
        # The counted text is not the file's own lines (a notebook is counted by cell source),
        # so "lines 40-80 of this file" would name a range that does not exist on disk. Left
        # whole and reported: a scope instruction nobody can follow is worse than no plan.
        return [], (
            f"{display} needs {total_tokens:,} tokens — more than a part holds — and cannot be "
            f"cut by line range ({countable.note}). Split it yourself, or name fewer files."
        )
    lines = countable.text.splitlines(keepends=True)
    if not lines:
        return [], None

    chars = sum(len(line) for line in lines)
    ratio = total_tokens / chars if chars else 1.0
    out: list[PartFile] = []
    note: str | None = None

    start = 0
    while start < len(lines):
        end, tokens = _fit_span(lines, start, ratio, budget, count)

        if tokens > budget:
            note = (
                f"{display} lines {start + 1}-{end}: a single line of {tokens:,} tokens exceeds "
                f"the per-part room ({budget:,}) and is kept whole"
            )
        out.append(
            PartFile(
                display=display,
                tokens=tokens,
                line_start=start + 1,
                line_end=end,
                total_lines=len(lines),
                source_dir=source_dir,
            )
        )
        start = end
    return out, note


def _fit_span(
    items: list[str], start: int, ratio: float, budget: int, count: Callable[[str], int]
) -> tuple[int, int]:
    """How far past ``start`` fits inside ``budget``, and what that span actually costs.

    The one packing loop in the project. It was written twice -- once to cut a file into
    line ranges and once to cut free text into segments -- and the two copies were identical
    but for what they did with the answer. This is the arithmetic the whole promise rests on,
    so it is the last thing that should exist in two versions capable of drifting apart.

    Two passes, because a character ratio is a guess and a token count is not. The first grows
    the span while the PREDICTED cost fits, which is cheap and usually close. The second
    measures what was actually chosen and shrinks it proportionally if the guess was
    optimistic -- bounded at eight rounds, though it converges in two or three, and each round
    strictly decreases the span so it terminates regardless.

    An item wider than the whole budget still has to go somewhere, so a span is never empty.
    The caller is the one that notices ``tokens > budget`` and says so.
    """
    end = start
    predicted = 0.0
    while end < len(items) and predicted + ratio * len(items[end]) <= budget:
        predicted += ratio * len(items[end])
        end += 1
    if end == start:
        end = start + 1
    tokens = count("".join(items[start:end]))
    for _ in range(8):
        if tokens <= budget or end - start <= 1:
            break
        end = start + max(1, int((end - start) * budget / tokens))
        tokens = count("".join(items[start:end]))
    return end, tokens


def _render_scope_part(
    prompt: str,
    index: int,
    total: int,
    files: list[PartFile],
    deferred: list[str],
    *,
    title: str = "",
) -> str:
    heading = f"Part {index} of {total}"
    if title:
        heading += f" — {title}"
    lines = [
        f"[Pharos] {heading} — this task was split to fit the context window.",
        "",
        "IN SCOPE for this part — read and change only these:",
    ]
    lines += [f"  - {f.label()}" for f in files]
    if deferred:
        lines += [
            "",
            "OUT OF SCOPE — do not open these in this part; other parts cover them:",
        ]
        lines += [f"  - {label}" for label in deferred]
    lines += [
        "",
        (
            "RULE: open ONLY the in-scope files. Opening anything else is what overflowed the "
            "window in the first place, and it will overflow again. If you believe you need a "
            "deferred file to proceed, do NOT open it — say which one and why, and stop."
        ),
    ]
    if index < total:
        lines += [
            (
                f"When you finish, end with a hand-off of at most 10 lines: what you changed "
                f"and what part {index + 1} needs to know. Paste that hand-off above the "
                f"next part."
            ),
        ]
    elif total > 1:
        # The last part has nobody to hand off to; asking it to write one anyway wastes output
        # tokens and reads as a mistake to whoever is following the instructions.
        lines += ["This is the final part — no hand-off is needed after it."]
    lines += [
        "",
        f"--- TASK (identical in all {total} parts) ---",
        prompt.strip(),
    ]
    if deferred:
        # Repeated after the task, deliberately. A real run (qwen3.5-9b) read two deferred
        # files when the rule appeared only above a long task block; attention falls off in
        # the middle, and the last thing read is the thing obeyed. The scope line is the one
        # instruction the whole projection rests on, so it gets the last word.
        # label(), not display: for a sliced file the range IS the permission. "only
        # forward.py" where the scope is lines 1-427 reads as leave for the whole file.
        scoped_names = ", ".join(f.label() for f in files)
        lines += [
            "",
            f"--- REMINDER --- In this part you may open ONLY: {scoped_names}. "
            f"The other {len(deferred)} file(s) listed above belong to other parts.",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- text mode


def _plan_text(
    prompt: str,
    report: CheckReport,
    count: Callable[[str], int],
    ceiling: int,
    label: str,
) -> SplitPlan:
    """The pasted text itself is the problem: cut it into ordered segments."""
    overhead = report.overhead_tokens
    # The text scaffold is fixed-size, so one probe render measures it exactly.
    scaffold = count(_render_text_part("", 1, 9)) + _SCAFFOLD_SLACK
    per_part_content = ceiling - overhead - scaffold
    notes: list[str] = []
    if per_part_content < _MIN_PART_TOKENS:
        return SplitPlan(
            mode=SplitMode.NONE,
            target_per_part=ceiling,
            target_label=label,
            reason=(
                f"the client overhead ({overhead:,}) and the part scaffold leave under "
                f"{_MIN_PART_TOKENS} tokens of the {label} ({ceiling:,}) for text"
            ),
        )
    if report.files:
        notes.append(
            f"the text is what overflows, so the {len(report.files)} named file(s) are left "
            f"as-is; a segment that names one still costs its content when the agent reads it"
        )

    segments, seg_note = _segment_text(prompt, per_part_content, count)
    if seg_note:
        notes.append(seg_note)
    if len(segments) <= 1:
        return SplitPlan(
            mode=SplitMode.NONE,
            target_per_part=ceiling,
            target_label=label,
            reason="the text could not be cut into more than one part",
            notes=notes,
        )

    parts: list[Part] = []
    for i, segment in enumerate(segments, start=1):
        body = _render_text_part(segment, i, len(segments))
        parts.append(_finalise(i, len(segments), body, [], count, ceiling, report))
    return SplitPlan(
        mode=SplitMode.TEXT,
        target_per_part=ceiling,
        target_label=label,
        parts=parts,
        notes=notes,
    )


def _segment_text(
    text: str, budget: int, count: Callable[[str], int]
) -> tuple[list[str], str | None]:
    """Cut ``text`` into ordered segments of at most ``budget`` tokens.

    Paragraph boundaries first (a paragraph is a unit of meaning); a paragraph too big for a
    segment falls back to its lines. Spans are grown by a characters-per-token ratio and then
    COUNTED — prose and fenced code do not tokenize at the same density, so a ratio alone
    produces segments that miss the budget by a little, which is the one thing this tool may
    not do. An atom that is over budget by itself is kept whole and reported.
    """
    atoms = _atoms(text, budget, count)
    chars = sum(len(a) for a in atoms) or 1
    ratio = count(text) / chars

    segments: list[str] = []
    note: str | None = None
    start = 0
    while start < len(atoms):
        end, tokens = _fit_span(atoms, start, ratio, budget, count)
        segment = "".join(atoms[start:end])

        if tokens > budget:
            note = (
                f"one indivisible line of {tokens:,} tokens exceeds the per-part room "
                f"({budget:,}) and is kept whole — that part will not fit"
            )
        segments.append(segment.strip("\n"))
        start = end
    return [s for s in segments if s.strip()], note


def _atoms(text: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """The indivisible units a segment is built from: paragraphs, or lines of a big paragraph."""
    out: list[str] = []
    for block in _paragraphs(text):
        if count(block) <= budget:
            out.append(block)
        else:
            out.extend(block.splitlines(keepends=True) or [block])
    return out


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, keeping the separators so the text round-trips."""
    out: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines(keepends=True):
        buffer.append(line)
        if line.strip() == "":
            out.append("".join(buffer))
            buffer = []
    if buffer:
        out.append("".join(buffer))
    return out or [text]


def _render_text_part(segment: str, index: int, total: int) -> str:
    if index < total:
        instruction = (
            f"This is segment {index} of {total} of one oversized input. Do NOT act on it yet. "
            f"Acknowledge in one line what this segment contains and wait; the remaining "
            f"{total - index} segment(s) follow."
        )
    else:
        instruction = (
            f"This is the final segment ({index} of {total}). All segments are now in front of "
            f"you: carry out the task described across them."
        )
    return (
        f"[Pharos] Part {index} of {total} — the input was too large for one request "
        f"and was cut into ordered segments.\n"
        f"{instruction}\n"
        f"\n--- SEGMENT {index}/{total} ---\n"
        f"{segment}\n"
    )


# ------------------------------------------------------------------------------ shared


def _finalise(
    index: int,
    total: int,
    body: str,
    files: list[PartFile],
    count: Callable[[str], int],
    ceiling: int,
    report: CheckReport,
    *,
    carried: int = 0,
    title: str = "",
) -> Part:
    """Cost the rendered part for real: overhead + the text as written + the scoped content.

    ``carried`` is input the part does not contain but will arrive with — the previous part's
    hand-off, pasted above it.
    """
    overhead = report.overhead_tokens
    projected = overhead + count(body) + sum(f.tokens for f in files) + carried
    over = max(0, projected - ceiling)
    return Part(
        index=index,
        total=total,
        body=body,
        files=files,
        projected_tokens=projected,
        fits=over == 0,
        over_by=over,
        title=title,
    )


def _ceiling(report: CheckReport, target: int | None) -> tuple[int | None, str]:
    if target is not None:
        return target, "requested target"
    budget = report.profile.budget if report.profile is not None else None
    if budget is None or budget.usable_budget is None:
        return None, "unknown"
    if budget.warn_tokens is not None and budget.warn_tokens > 0:
        return budget.warn_tokens, "warn threshold"
    return budget.usable_budget, "usable budget"
