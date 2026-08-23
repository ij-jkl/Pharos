"""Did the run actually work? Six questions, all answerable without asking a model.

"Every part completed" is not success. A part completes by replying without calling a tool,
which is exactly what a model does when it has read its files and described the change instead
of making it. The per-part lines say what happened; this says whether it added up.

Nothing here interprets the code that was written, and no model is asked to. Whether an edit
is *right* remains outside what Pharos claims — `git diff` is still the reviewer. What can be
measured exactly is whether the run covered its scope, kept its thread, stayed inside its
window, whether Pharos's own arithmetic held up against the backend's, and whether the
project's own checks still pass. That last one is an exit code from a tool the repository
already had, which is a fact rather than an opinion; a scorecard that offered an opinion would
be the overclaiming this project refuses everywhere else.

The six:

* **Coverage** — of the files the plan assigned, how many were written. The headline. A run
  that touches three of twenty did not succeed, whatever its parts reported.
* **Continuity** — the thread between parts is the hand-off and nothing else, since every part
  starts from an empty conversation. So: was one produced wherever there was a next part, and
  did it fit the reserve held back for it. A hand-off that overran its reserve was planned
  against a budget that was too small, which is a config finding, not a model failure.
  A hand-off can also be present and carry nothing, so each one is checked for whether it
  NAMES any file its part changed. That started as a length threshold and length turned out to
  be the wrong measure: real useful hand-offs run about fifteen tokens ("Added XML doc comments
  to `INoteRepository.cs` and `NoteRepository.cs`"), while the useless ones are parts that
  wrote files and then reported "NO CHANGES NEEDED". A threshold flags the first and waves the
  second through; naming separates them exactly. A part that changed nothing is not held to it.
* **Revisits** — the fingerprint of a part that lost the thread. A part reaching for a file
  ANOTHER PART OWNED is redoing work already done; the scope layer refuses it, so it is
  recorded rather than damaging. Distinguished from a path the model simply invented, which is
  confusion about the project and not about what has been done — one real run tried to write
  to a `Data/` folder that has never existed, and counting that as lost continuity would have
  been wrong.
* **Headroom** — the highest fraction of any part's ceiling actually used. Near 100% means the
  next slightly larger file breaks the run; low means the division has room.
* **Verification** — the project's own checks, re-run afterwards. Coverage says every
  assigned file was written; it cannot say the result still parses. A measured run scored 100%
  coverage and shipped `sorted(total.items(), ...)` where the variable is `totals` - COMPLETE,
  and a NameError. Every check also runs BEFORE the first part, so a repository that arrived
  red is reported as such and never charged to the run, and only a check that PASSED before
  and fails after counts against it.
* **Drift** — Pharos's own projection against the backend's ``prompt_eval_count``, paired per
  REQUEST. The number this project is least entitled to hide, and the one it has been most
  wrong about.

  On conversations constructed by hand and put to a live backend, the projection lands at
  0.87x-0.98x — tens of tokens BELOW what the backend reports, which is the chat template's
  own scaffolding and cannot be seen from here. Across a real four-part run the per-request
  ratio ranged 0.96x to 2.57x. The high end is NOT explained: the two shapes that looked like
  candidates (a large file in a tool result, a large file inside tool_call arguments) both
  measured under 1.0 when tested directly. It is recorded as unexplained rather than given a
  plausible story, because three plausible stories about this number have already been wrong.

  Only the low side threatens anything. Over-counting wastes room; under-counting means a
  ceiling was enforced against a number below the real prompt. ``SAFETY_MARGIN`` is subtracted
  from every ceiling to absorb the residue, so ``under_counted`` asks whether the shortfall
  outgrew that margin rather than whether it exists — it always does. The shortfall is
  reported in tokens for the same reason: 0.96x is 40 tokens on a small prompt and 800 on a
  large one, and only one of those matters. Measured worst shortfall on a real run: 60 tokens,
  against a 256-token margin.

  Two earlier versions of this number were wrong in ways worth remembering. The first divided
  a part's PEAK projection by whichever count came back last — two different requests — and
  read 2.56x. The second was honest arithmetic over a double-counted projection, and read
  1.09-2.20x "conservative". Both went into a docstring and a README before being checked
  against the backend directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pharos.agent.session import SAFETY_MARGIN, PartResult
from pharos.agent.tools import normalise
from pharos.agent.verify import Verification


def _carried_the_work(part: PartResult) -> bool:
    """Does a part's hand-off mention any of the files it changed?

    This was a length check, and length is a poor proxy. Measured against real hand-offs: the
    useful ones read "Added XML doc comments to `INoteRepository.cs` and `NoteRepository.cs`"
    - about fifteen tokens, and everything the next part needs. The useless ones read "NO
    CHANGES NEEDED" from parts that demonstrably wrote files. A token threshold flags the
    first and, at the wrong setting, waves the second through; naming separates them exactly.

    A part that changed nothing has nothing to name, and is not held to this.
    """
    if not part.files_written:
        return True
    lowered = part.text.lower()
    return any(path.replace(chr(92), "/").rsplit("/", 1)[-1].lower() in lowered
               for path in part.files_written)


@dataclass(frozen=True, slots=True)
class Scorecard:
    """What a run achieved, in numbers a script can assert on."""

    parts: int
    parts_that_wrote: int
    scoped_files: int
    written_files: int
    untouched: list[str] = field(default_factory=list)

    # Parts that existed only because the plan's own parts left work undone. Coverage counts
    # what they wrote, because the file did get changed — but a run needing many of them is a
    # plan that is not sized for this model, and that belongs on the report rather than
    # smoothed into a single percentage.
    repair_parts: int = 0
    # Files written by the planned parts alone, before any repair. Without it the two numbers
    # that matter are indistinguishable: a run where the plan did everything and a run where
    # the plan did three quarters and the sweep rescued the rest both read 92%. Measured on
    # consecutive runs of one task, those were exactly the two cases.
    written_before_repair: int = 0

    handoffs_expected: int = 0
    handoffs_produced: int = 0
    handoff_reserve: int = 0
    largest_handoff: int = 0
    handoff_overruns: int = 0

    thin_handoffs: int = 0  # parts that wrote files and then reported almost nothing

    revisits: list[str] = field(default_factory=list)  # files another part already owned
    invented: list[str] = field(default_factory=list)  # paths in no part's scope at all

    peak_fraction: float = 0.0  # highest part peak / its ceiling
    # Our projection over the backend's count, per REQUEST. A range, because one number hides
    # which direction it went: high is waste, and anything below 1.0 is the failure mode.
    drift_high: float | None = None
    drift_low: float | None = None
    # The ratio for the BIGGEST request of the run. The over-count is a fixed offset — the
    # tool catalogue, counted as wire JSON against the backend's compact rendering — so the
    # ratio is worst on the smallest conversation, where it could not matter less. The one
    # that decides whether a ceiling holds is the ratio where the window is nearly full.
    drift_at_peak: float | None = None
    drift_samples: int = 0
    # Worst shortfall in TOKENS: how far the projection sat below the backend's count on the
    # request where it was furthest under. The ratio alone cannot say whether that matters —
    # 0.96x is 40 tokens on a small prompt and 800 on a large one — and tokens are what the
    # safety margin is denominated in.
    worst_shortfall: int = 0

    # What the project's own checks said afterwards, discounting whatever was already failing
    # when the run started. None when verification was switched off or never ran.
    verification: Verification | None = None
    # Parts where the backend's own count FELL mid-conversation: it stopped evaluating
    # everything it was sent. Not a drift statistic — proof that the window Pharos measured is
    # not the window in force, and that context was dropped without anybody being told.
    truncated_parts: int = 0

    nudged_parts: int = 0
    abandoned_parts: int = 0
    failed_parts: int = 0

    @property
    def plan_coverage(self) -> float | None:
        """What the plan achieved on its own, before the repair pass went back over it."""
        if not self.scoped_files:
            return None
        return self.written_before_repair / self.scoped_files

    @property
    def rescued(self) -> int:
        """Files the repair pass changed that the plan had left alone."""
        return max(self.written_files - self.written_before_repair, 0)

    @property
    def coverage(self) -> float | None:
        """Written over scoped. None when the run was one unrestricted part: nothing to
        compare against, and inventing a denominator would be worse than admitting it."""
        if not self.scoped_files:
            return None
        return self.written_files / self.scoped_files

    @property
    def kept_the_thread(self) -> bool:
        """Every hand-off that was needed was produced, fitted, and nothing was redone."""
        return (
            self.handoffs_produced == self.handoffs_expected
            and not self.handoff_overruns
            and not self.thin_handoffs
            and not self.revisits
        )

    @property
    def under_counted(self) -> bool:
        """Under by more than the margin that exists to absorb it.

        Being slightly under is the normal, measured state: the chat template adds scaffolding
        Pharos cannot see, so the projection sits about 40-50 tokens below the backend's count
        regardless of conversation size. ``SAFETY_MARGIN`` is subtracted from every ceiling for
        exactly that. An alarm that fired on every run would be noise, so this asks the
        question that matters instead — did the shortfall outgrow the margin?

        It is no longer the emergency it was. The session corrects its ceiling by the ratio
        measured here once the first response has been counted, so a shortfall past the margin
        is exposed for one request per part rather than for the whole run. Still worth the
        alarm: it says the constant did not fit this template, and the first request is the one
        the correction cannot cover.
        """
        return self.worst_shortfall > SAFETY_MARGIN

    @property
    def complete(self) -> bool:
        """The work the plan assigned was done, and nothing failed.

        A part stopping early does NOT make a run incomplete. It used to, and produced a
        report reading "INCOMPLETE - 0 of 13 files were never changed" beside a full coverage
        bar, which is not a stern verdict but a contradiction. A part that reached its ceiling
        after writing everything it owned did its job; that it ran out of room on the way is
        worth reporting, and `convergence` reports it.
        """
        if self.failed_parts:
            return False
        if self.verification is not None and not self.verification.ok:
            # Writing every assigned file is not success if the project stopped building on
            # the way. Only checks that PASSED before the run count here, so this cannot be
            # tripped by a repository that arrived red.
            return False
        if self.coverage is None:
            # No scope to measure against, so the only honest test is whether anything at all
            # was written — a run that touched nothing is not a complete run.
            return self.written_files > 0
        return self.coverage == 1.0


def score(
    parts: list[PartResult],
    *,
    handoff_reserve: int,
    repair_parts: int = 0,
    verification: Verification | None = None,
) -> Scorecard:
    """Reduce a finished run to the five questions above."""
    scoped: list[str] = []
    for part in parts:
        for path in part.scoped:
            if path not in scoped:
                scoped.append(path)
    scoped_set = {normalise(path) for path in scoped}

    written: set[str] = set()
    for part in parts:
        written.update(normalise(path) for path in part.files_written)

    planned_parts = parts[: len(parts) - repair_parts] if repair_parts else parts
    written_by_plan: set[str] = set()
    for part in planned_parts:
        written_by_plan.update(normalise(path) for path in part.files_written)

    # A refusal for a file some part owns is a revisit; anything else the model made up.
    revisits: list[str] = []
    invented: list[str] = []
    for part in parts:
        for raw in part.scope_refusals:
            path = normalise(raw)
            bucket = revisits if path in scoped_set else invented
            if path not in bucket:
                bucket.append(path)

    # The last part hands off to nobody, so it is not expected to produce one. Repair parts
    # run AFTER the plan and hand off to nobody either — they are a sweep, not a continuation,
    # so holding them to the thread would penalise a run for the mechanism that rescued it.
    planned = parts[: len(parts) - repair_parts] if repair_parts else parts
    expects_handoff = planned[:-1] if len(planned) > 1 else []
    produced = [p for p in expects_handoff if p.text.strip()]

    # A hand-off from a part that CHANGED something has to describe it. From a part that did
    # nothing, brevity is the honest answer, so silence there is not held against continuity.
    thin = sum(1 for p in expects_handoff if not _carried_the_work(p))

    peak_fraction = max(
        (p.peak_tokens / p.ceiling for p in parts if p.ceiling > 0),
        default=0.0,
    )
    # Each ratio pairs one request's projection with that same request's prompt_eval_count.
    # Dividing a part's PEAK by whichever count arrived last compares two different requests:
    # it read 2.56x on a run whose honest worst case was 1.53x.
    pairs = [
        (ours, theirs)
        for part in parts
        for ours, theirs in part.drift_samples
        if theirs > 0
    ]
    ratios = [ours / theirs for ours, theirs in pairs]
    biggest = max(pairs, key=lambda pair: pair[0], default=None)
    shortfall = max((theirs - ours for ours, theirs in pairs), default=0)

    return Scorecard(
        parts=len(parts),
        parts_that_wrote=sum(1 for p in parts if p.files_written),
        repair_parts=repair_parts,
        written_before_repair=(
            len(written_by_plan & scoped_set) if scoped_set else len(written_by_plan)
        ),
        scoped_files=len(scoped_set),
        written_files=len(written & scoped_set) if scoped_set else len(written),
        untouched=sorted(scoped_set - written),
        handoffs_expected=len(expects_handoff),
        handoffs_produced=len(produced),
        handoff_reserve=handoff_reserve,
        largest_handoff=max((p.handoff_tokens for p in parts), default=0),
        handoff_overruns=sum(
            1 for p in expects_handoff if handoff_reserve and p.handoff_tokens > handoff_reserve
        ),
        thin_handoffs=thin,
        revisits=revisits,
        invented=invented,
        peak_fraction=peak_fraction,
        drift_high=max(ratios) if ratios else None,
        drift_low=min(ratios) if ratios else None,
        drift_at_peak=(biggest[0] / biggest[1]) if biggest else None,
        drift_samples=len(ratios),
        worst_shortfall=max(shortfall, 0),
        truncated_parts=sum(1 for p in parts if p.truncated),
        nudged_parts=sum(1 for p in parts if p.nudged),
        abandoned_parts=sum(1 for p in parts if p.stopped_early),
        failed_parts=sum(1 for p in parts if p.error),
        verification=verification,
    )


def to_dict(card: Scorecard) -> dict[str, object]:
    """The scorecard as plain data, so a wrapper or CI step can assert on a run."""
    return {
        "complete": card.complete,
        "kept_the_thread": card.kept_the_thread,
        "coverage": card.coverage,
        "parts": card.parts,
        "parts_that_wrote": card.parts_that_wrote,
        "repair_parts": card.repair_parts,
        "plan_coverage": card.plan_coverage,
        "rescued_by_repair": card.rescued,
        "scoped_files": card.scoped_files,
        "written_files": card.written_files,
        "untouched": card.untouched,
        "handoffs": {
            "expected": card.handoffs_expected,
            "produced": card.handoffs_produced,
            "reserve": card.handoff_reserve,
            "largest": card.largest_handoff,
            "overruns": card.handoff_overruns,
        },
        "thin_handoffs": card.thin_handoffs,
        "revisits": card.revisits,
        "invented_paths": card.invented,
        "peak_fraction": round(card.peak_fraction, 3),
        "estimate_over_backend": {
            "high": None if card.drift_high is None else round(card.drift_high, 3),
            "low": None if card.drift_low is None else round(card.drift_low, 3),
            "at_largest_request": (
                None if card.drift_at_peak is None else round(card.drift_at_peak, 3)
            ),
            "requests": card.drift_samples,
        },
        "worst_shortfall_tokens": card.worst_shortfall,
        "truncated_parts": card.truncated_parts,
        "under_counted": card.under_counted,
        "nudged_parts": card.nudged_parts,
        "abandoned_parts": card.abandoned_parts,
        "failed_parts": card.failed_parts,
        "verification": _verification_dict(card.verification),
    }


def _verification_dict(verification: Verification | None) -> dict[str, object] | None:
    """The checks as plain data. `ran` matters as much as `ok`: a CI step must be able to tell
    "nothing was broken" from "nothing was checked"."""
    if verification is None:
        return None
    return {
        "ok": verification.ok,
        "ran": verification.ran,
        "newly_broken": [
            {"name": c.name, "detail": c.detail} for c in verification.newly_broken
        ],
        "already_failing": [c.name for c in verification.already_failing],
        "unattributable": [c.name for c in verification.unattributable],
        "already_failing_but_changed": [c.name for c in verification.worsened],
        "passed": [c.name for c in verification.passing],
        "skipped": [
            {"name": c.name, "reason": c.skipped} for c in verification.skipped
        ],
        "unchecked_files": list(verification.unchecked_files),
    }
