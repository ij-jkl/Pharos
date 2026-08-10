"""Did the run actually work? Five questions, all answerable without asking a model.

"Every part completed" is not success. A part completes by replying without calling a tool,
which is exactly what a model does when it has read its files and described the change instead
of making it. The per-part lines say what happened; this says whether it added up.

Nothing here interprets the code that was written. Whether the edit is correct is outside what
Pharos claims — `git diff` is the reviewer, and a scorecard that implied otherwise would be
the same overclaiming this project refuses everywhere else. What can be measured exactly is
whether the run covered its scope, kept its thread, stayed inside its window, and whether
Pharos's own arithmetic held up against the backend's.

The five:

* **Coverage** — of the files the plan assigned, how many were written. The headline. A run
  that touches three of twenty did not succeed, whatever its parts reported.
* **Continuity** — the thread between parts is the hand-off and nothing else, since every part
  starts from an empty conversation. So: was one produced wherever there was a next part, and
  did it fit the reserve held back for it. A hand-off that overran its reserve was planned
  against a budget that was too small, which is a config finding, not a model failure.
  A hand-off can also be present and still carry nothing. A real run produced three of three
  hand-offs whose largest was SIX tokens against a 500-token reserve, and the metric called
  that continuity while coverage sat at 31%. Emptiness is only honest when the part had
  nothing to report; a part that wrote files and then said six tokens about it has dropped the
  thread just as surely as one that said nothing at all, so those are counted separately.
* **Revisits** — the fingerprint of a part that lost the thread. A part reaching for a file
  ANOTHER PART OWNED is redoing work already done; the scope layer refuses it, so it is
  recorded rather than damaging. Distinguished from a path the model simply invented, which is
  confusion about the project and not about what has been done — one real run tried to write
  to a `Data/` folder that has never existed, and counting that as lost continuity would have
  been wrong.
* **Headroom** — the highest fraction of any part's ceiling actually used. Near 100% means the
  next slightly larger file breaks the run; low means the division has room.
* **Drift** — Pharos's own projection divided by the backend's ``prompt_eval_count``, paired
  per REQUEST and reported as a range. The number this project is least entitled to hide.
  Above 1.0 means Pharos counted more than the backend saw: safe, and wasteful, because parts
  come out smaller than they needed to be. Below 1.0 means a ceiling was enforced against an
  estimate that sat under the real prompt, which is not a ceiling at all — that is the alarm.
  Measured across two real runs: 1.12x to 1.53x, never under.

  The pairing is the whole point. The first version divided a part's PEAK projection by
  whichever count came back last, which are two different requests: it read 2.56x on a run
  whose honest worst case was 1.53x, and that fabricated number went into a docstring and a
  README before the arithmetic was checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pharos.agent.session import PartResult
from pharos.agent.tools import normalise

# Below this, a hand-off from a part that changed files is not a summary of anything. Chosen
# against measured hand-offs: real ones ran 64-328 tokens (see PharosConfig.handoff_reserve),
# and the degenerate ones observed were single digits. Nothing useful lives in between.
_THIN_HANDOFF_TOKENS = 20


@dataclass(frozen=True, slots=True)
class Scorecard:
    """What a run achieved, in numbers a script can assert on."""

    parts: int
    parts_that_wrote: int
    scoped_files: int
    written_files: int
    untouched: list[str] = field(default_factory=list)

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

    nudged_parts: int = 0
    abandoned_parts: int = 0
    failed_parts: int = 0

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
        """Any request where Pharos projected FEWER tokens than the backend then reported.

        The only direction that matters for safety: a ceiling enforced against an estimate
        that sits under the real prompt is not a ceiling. Over-counting merely wastes room.
        """
        return self.drift_low is not None and self.drift_low < 1.0

    @property
    def complete(self) -> bool:
        """The whole scope was written and no part failed or was abandoned."""
        return (
            self.coverage == 1.0
            and not self.failed_parts
            and not self.abandoned_parts
        )


def score(parts: list[PartResult], *, handoff_reserve: int) -> Scorecard:
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

    # A refusal for a file some part owns is a revisit; anything else the model made up.
    revisits: list[str] = []
    invented: list[str] = []
    for part in parts:
        for raw in part.scope_refusals:
            path = normalise(raw)
            bucket = revisits if path in scoped_set else invented
            if path not in bucket:
                bucket.append(path)

    # A hand-off from a part that CHANGED something has to describe it. From a part that did
    # nothing, brevity is the honest answer, so silence there is not held against continuity.
    thin = sum(
        1
        for p in parts[:-1]
        if p.files_written and p.handoff_tokens < _THIN_HANDOFF_TOKENS
    )

    # The last part hands off to nobody, so it is not expected to produce one.
    expects_handoff = parts[:-1] if len(parts) > 1 else []
    produced = [p for p in expects_handoff if p.text.strip()]

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

    return Scorecard(
        parts=len(parts),
        parts_that_wrote=sum(1 for p in parts if p.files_written),
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
        nudged_parts=sum(1 for p in parts if p.nudged),
        abandoned_parts=sum(1 for p in parts if p.stopped_early),
        failed_parts=sum(1 for p in parts if p.error),
    )


def to_dict(card: Scorecard) -> dict[str, object]:
    """The scorecard as plain data, so a wrapper or CI step can assert on a run."""
    return {
        "complete": card.complete,
        "kept_the_thread": card.kept_the_thread,
        "coverage": card.coverage,
        "parts": card.parts,
        "parts_that_wrote": card.parts_that_wrote,
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
        "under_counted": card.under_counted,
        "nudged_parts": card.nudged_parts,
        "abandoned_parts": card.abandoned_parts,
        "failed_parts": card.failed_parts,
    }
