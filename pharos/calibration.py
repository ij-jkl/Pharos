"""Observed-traffic calibration: Observe teaches Warn what requests really cost.

The proxy records one compact record per successfully completed counted request — token
counts, message count, output size. COUNTS ONLY, NEVER TEXT: no prompt content, no file
names, nothing reconstructable ever touches disk. The pre-flight check reads these records
to estimate what a coding agent adds on top of a pasted prompt.

Estimator design (the part that has to be right):

* ``client_added = input_total - user_content`` per request captures everything the client
  injected — system prompt, tool catalogue, file context, prior turns — without itemizing it.
* The fixed overhead is estimated as the MINIMUM of ``client_added``, taken over the records
  with the FEWEST messages. Conversation history only ever inflates the number, so the
  minimum converges on first-turn cost — and a minimum keeps the pre-flight verdict a true
  lower bound, which is the one promise its output makes. Restricting to fewest-message
  records first makes it a real first-turn estimate rather than a hopeful one: in a long
  session with no fresh starts, a global minimum would still carry history.
* AGENT-SHAPED records (tools or a system prompt) are preferred over bare ones. A minimum is
  only as representative as the pool it minimises over, and a proxy sees more than one client:
  four `curl` pokes at the endpoint are enough to drag the estimate from ~1,800 tokens to ~10
  and keep it there. Found live, on real traffic — the bare records stay as a fallback for a
  store that has nothing better in it.
* Exact records (backend-reconciled) are preferred over estimated ones whenever any exist.

Writes go through ``ObservationRecorder``, which batches in memory and flushes off the event
loop (``asyncio.to_thread``) — the proxy's request path stays non-blocking.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pharos.naming import same_model
from pharos.store import read_json, write_json

_logger = logging.getLogger("pharos.calibration")

_MAX_RECORDS = 200
_FLUSH_EVERY_N = 5
_FLUSH_EVERY_S = 10.0


@dataclass(frozen=True, slots=True)
class Observation:
    """Token accounting for one completed request. Counts only — never text."""

    ts: float
    endpoint: str  # "ollama-chat" | "ollama-generate" | "openai-chat" | "openai-completions"
    model: str | None
    input_tokens: int  # reconciled prompt_eval_count when reported, else the labeled estimate
    input_exact: bool  # True only when input_tokens came from backend reconciliation
    user_tokens: int  # tokens of user-authored content alone, counted by the bound tokenizer
    messages: int  # message count of the request (1 for completion-style endpoints)
    output_tokens: int | None  # eval_count when the backend reported it
    # True when the request carried tools or a system prompt: the shape of a coding agent, as
    # opposed to a bare curl at the endpoint. Still counts only — a boolean, never text.
    agent_shaped: bool = False
    # True when it carried a TOOL CATALOGUE specifically, which is the stronger signal of the
    # two ``agent_shaped`` folds together. A `curl` with a system prompt is agent-shaped and
    # costs about nineteen tokens; measured live, two such pokes were enough to hold the
    # learned overhead at 19 across 116 real requests, understating a check's floor by more
    # than a thousand tokens. A tool catalogue is the thing a chat client does not send.
    #
    # None means "written before this was tracked", tri-state for the same reason
    # ``user_exact`` is: a record that never knew is not one that checked and found no tools,
    # and preferring the known-tooled pool must not silently discard an older store.
    has_tools: bool | None = None
    # Whether ``user_tokens`` was counted with the vocabulary the request actually named.
    #
    # ``input_exact`` is about the MINUEND: it is True when input_tokens came from the
    # backend's own prompt_eval_count. Overhead is learned by SUBTRACTING user_tokens from it,
    # and that subtrahend comes from the one tokenizer the proxy has bound. Point a second
    # model at the proxy and the two numbers stop being commensurable — an exact total minus a
    # count taken in another model's vocabulary — with nothing in the record admitting it.
    #
    # None means "written before this was tracked". Deliberately not True: a record that never
    # knew is not the same as one that checked and passed, and this project does not launder
    # unknowns into confidence. The learner accepts None and rejects False, so existing stores
    # keep working while the case now known to be wrong is dropped.
    user_exact: bool | None = None


def _optional_bool(value: Any) -> bool | None:
    """Tri-state read: absent stays absent rather than collapsing into False."""
    return None if value is None else bool(value)


def load_observations(path: Path) -> list[Observation]:
    """Read the store; an absent, malformed or partially valid file degrades to what parses."""
    raw = read_json(path, [])
    if not isinstance(raw, list):
        return []
    out: list[Observation] = []
    for item in raw:
        parsed = _parse_record(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _parse_record(item: object) -> Observation | None:
    if not isinstance(item, dict):
        return None
    try:
        return Observation(
            ts=float(item["ts"]),
            endpoint=str(item["endpoint"]),
            model=item["model"] if isinstance(item.get("model"), str) else None,
            input_tokens=int(item["input_tokens"]),
            input_exact=bool(item["input_exact"]),
            user_tokens=int(item["user_tokens"]),
            messages=int(item["messages"]),
            output_tokens=int(item["output_tokens"])
            if item.get("output_tokens") is not None
            else None,
            # Absent in records written before this field existed: they read as not
            # agent-shaped, which only ever makes them a fallback, never a wrong answer.
            agent_shaped=bool(item.get("agent_shaped", False)),
            has_tools=_optional_bool(item.get("has_tools")),
            user_exact=_optional_bool(item.get("user_exact")),
        )
    except (KeyError, TypeError, ValueError):
        return None


class ObservationRecorder:
    """Buffers observations in memory and flushes them to disk off the event loop.

    ``add()`` is synchronous and cheap (append under a lock). A flush merges the buffer into
    the on-disk store — load, append, trim to the newest ``_MAX_RECORDS``, atomic replace —
    inside ``asyncio.to_thread``, debounced to every ``_FLUSH_EVERY_N`` records or
    ``_FLUSH_EVERY_S`` seconds. Failures are logged and dropped: calibration is an optional
    luxury and must never affect the proxy.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._pending: list[Observation] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._flushing = False

    def add(self, observation: Observation) -> bool:
        """Buffer one record; True when the caller should schedule ``flush()``."""
        with self._lock:
            self._pending.append(observation)
            due = (
                len(self._pending) >= _FLUSH_EVERY_N
                or time.monotonic() - self._last_flush >= _FLUSH_EVERY_S
            )
            if due and not self._flushing:
                self._flushing = True
                return True
            return False

    async def flush(self) -> None:
        """Write buffered records to the store, off the event loop."""
        try:
            await asyncio.to_thread(self._flush_sync)
        except Exception as exc:  # noqa: BLE001 — calibration must never hurt the proxy
            _logger.warning("observation flush failed (%s); records dropped", exc)
        finally:
            with self._lock:
                self._flushing = False
                self._last_flush = time.monotonic()

    async def aclose(self) -> None:
        """Final flush on shutdown so short sessions still leave records behind."""
        with self._lock:
            if not self._pending:
                return
            self._flushing = True
        await self.flush()

    def _flush_sync(self) -> None:
        with self._lock:
            batch = self._pending
            self._pending = []
        if not batch:
            return
        existing = load_observations(self._path)
        merged = (existing + batch)[-_MAX_RECORDS:]
        payload: list[dict[str, Any]] = [asdict(obs) for obs in merged]
        write_json(self._path, payload)


@dataclass(frozen=True, slots=True)
class OverheadEstimate:
    tokens: int
    provenance: str  # human-readable origin, shown verbatim in the verdict


def estimate_client_overhead(
    observations: list[Observation], model: str | None
) -> OverheadEstimate | None:
    """Estimate the fixed per-request overhead a client adds on top of the user's prompt."""
    pool = _matching(observations, model)
    if not pool:
        return None
    # Agent-shaped records first. One `curl` at the proxy carries no system prompt and no tool
    # catalogue, so its client_added is ~10 tokens — and being a MINIMUM over the fewest-message
    # records, that single poke would otherwise define the estimate for every later pre-flight,
    # collapsing 1,800 tokens of real agent overhead to nothing. Mixing clients is normal; the
    # estimate has to survive it.
    # Three tiers, strongest first. A tool catalogue is what a coding agent sends and a chat
    # client does not; a system prompt alone is sent by anything, `curl` included. Preferring
    # the tooled pool is the fix for a measured failure, not a refinement: on a store of 116
    # real agent requests, two `curl` pokes carrying a system prompt held the estimate at 19
    # tokens, because a MINIMUM is only as good as the pool it minimises over and both pokes
    # were first-turn records.
    tooled = [o for o in pool if o.has_tools]
    agent = [o for o in pool if o.agent_shaped]
    if tooled:
        shaped, shape_note = tooled, "tool-carrying"
    elif agent:
        shaped, shape_note = agent, "agent-shaped"
    else:
        shaped, shape_note = pool, "all"
    # Drop records whose user_tokens came from the wrong vocabulary: subtracting one of those
    # from an exact total yields an overhead that is confidently wrong, which is worse than a
    # smaller sample. Records predating the flag (None) are kept — see Observation.user_exact.
    commensurable = [o for o in shaped if o.user_exact is not False]
    # Falling back to them rather than refusing: an overhead the user can see and distrust
    # beats none at all, and someone whose pharos.toml names one model while another is in use
    # should still get a number. But it gets said out loud in the provenance — every figure in
    # this project carries where it came from, and "quietly slightly wrong" is the one outcome
    # that is not allowed.
    mismatched = not commensurable and any(o.user_exact is False for o in shaped)
    shaped = commensurable or shaped
    exact = [o for o in shaped if o.input_exact]
    used, kind = (exact, "exact") if exact else (shaped, "estimated")
    fewest = min(o.messages for o in used)
    first_turn_like = [o for o in used if o.messages == fewest]
    tokens = min(max(o.input_tokens - o.user_tokens, 0) for o in first_turn_like)
    caveat = (
        " · counted against a different model's vocabulary, so the subtraction is approximate"
        if mismatched
        else ""
    )
    provenance = (
        f"learned · min over {len(first_turn_like)} of {len(pool)} observed requests "
        f"({shape_note}, {kind} inputs, fewest-message = {fewest}){caveat}"
    )
    return OverheadEstimate(tokens=tokens, provenance=provenance)


def estimate_typical_output(observations: list[Observation], model: str | None) -> int | None:
    """Median observed output size, for sanity-checking response_reserve."""
    sizes = [
        o.output_tokens
        for o in _matching(observations, model)
        if o.output_tokens is not None and o.output_tokens > 0
    ]
    if not sizes:
        return None
    return int(statistics.median(sizes))


def _matching(observations: list[Observation], model: str | None) -> list[Observation]:
    """Records relevant to ``model``; with no model to filter by, every record counts."""
    if model is None:
        return list(observations)
    return [o for o in observations if same_model(o.model, model)]



# --- what the chat template costs -------------------------------------------------------------
#
# The second thing this module remembers, and it is learned by `pharos run` rather than by the
# proxy. Pharos counts a conversation from the messages it holds; the backend counts it after
# applying a chat template that runs server-side and cannot be seen from here. The gap is real,
# one-directional and specific to a model.
#
# A session already corrects for it within itself: `prompt_eval_count` comes back with every
# response, so from the second request onwards the ceiling is enforced against a measured gap
# rather than a constant. The first request of every part is the one that correction cannot
# cover, because a part starts with no responses to learn from. With three parts that is three
# requests a run sent against a ceiling with nothing behind it but `SAFETY_MARGIN`, and nine
# live runs measured that constant 288-297 tokens short.
#
# So the gap is remembered between runs. Nothing else changes: it seeds the same correction the
# session was already making for itself, one request earlier.
#
# It is stored as TOKENS, not as a ratio, and that is a measurement rather than a preference.
# The first design here multiplied: remember `backend / ours` and scale the projection by it.
# Seventeen paired requests over conversations from 1,235 to 3,604 tokens said otherwise --
# the gap was 263 to 314 tokens across the whole range, flat, while the ratio it implied fell
# from 1.22x to 1.09x as the conversation grew. It is fixed scaffolding, so a fixed number is
# what describes it. A 1.22x factor on a 3,604-token conversation reserves 793 tokens to cover
# 314, and the waste grows with the window: on a 60K conversation it would throw away more
# than 13,000 tokens of every part.
#
# The assumption that buys, stated plainly: the template's cost does not grow with the
# conversation. That is what was measured over a 3x range on one model. A template whose
# scaffolding scaled with length would be under-corrected above the largest conversation yet
# observed -- and the live in-session measurement tracks it upward within a run, `SAFETY_MARGIN`
# sits underneath, and the drift line goes on reporting the raw estimator so the shortfall
# stays visible either way.

_TEMPLATE_SAMPLES = 20  # per-run worst offsets kept per model

# Refused above this. A remembered offset comes straight off every part's ceiling, so one that
# says the template costs thousands of tokens would shrink every part for good on the strength
# of a stored number. Real chat-template scaffolding is a few hundred tokens; past this the
# projection is wrong in a way a constant should not be papering over, and the run should say
# so rather than quietly lose the room.
TEMPLATE_OFFSET_CAP = 4096


@dataclass(frozen=True, slots=True)
class TemplateCost:
    """How many tokens the backend's count runs above ours, for one model."""

    model: str
    offsets: tuple[int, ...] = ()  # per-RUN worst offsets, oldest first
    updated: float = 0.0

    @property
    def offset(self) -> int:
        """The tokens to seed a session with: the median of per-run worst offsets.

        A median of maxima, and both halves are deliberate. The MAXIMUM within a run, because
        a ceiling holds at the worst case or it does not hold. The MEDIAN across runs, because
        the maximum across runs ratchets and never comes back down: one anomalous request --
        and this project has recorded a 2.57x it could not explain -- would shrink every future
        part permanently on the strength of one measurement. A median absorbs that and still
        moves if the model or its template really changes.

        Never below zero. A backend counting fewer tokens than we did is a sign we were being
        cautious, not licence to fit more in.
        """
        if not self.offsets:
            return 0
        return min(max(0, round(statistics.median(self.offsets))), TEMPLATE_OFFSET_CAP)

    @property
    def runs(self) -> int:
        return len(self.offsets)

    @property
    def capped(self) -> bool:
        """The measurement wanted more than the cap allows, and was refused it."""
        return bool(self.offsets) and statistics.median(self.offsets) > TEMPLATE_OFFSET_CAP


def load_template_costs(path: Path) -> dict[str, TemplateCost]:
    """Read the store. A missing or unreadable file is an empty memory, never an error.

    Same rule as the observation store: this makes runs better when it is there and must never
    be able to stop one happening. A corrupt file is worth a log line and nothing more.
    """
    raw = read_json(path, {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, TemplateCost] = {}
    for name, item in (raw.get("models") or {}).items():
        if not isinstance(item, dict) or not isinstance(name, str):
            continue
        offsets = tuple(
            int(value)
            for value in (item.get("offsets") or [])
            if isinstance(value, int | float) and value > 0
        )
        out[name] = TemplateCost(
            model=name,
            offsets=offsets[-_TEMPLATE_SAMPLES:],
            updated=float(item.get("updated") or 0.0),
        )
    return out


def remembered_cost(path: Path, model: str | None) -> TemplateCost | None:
    """What is known about this model's template, matching names the way the profiler does."""
    if model is None:
        return None
    for name, cost in load_template_costs(path).items():
        if same_model(name, model) and cost.offsets:
            return cost
    return None


def remember_run(path: Path, model: str | None, offsets: list[int]) -> TemplateCost | None:
    """Fold one run's observations into the memory and write it back.

    ``offsets`` is every (backend - ours) gap the run measured. The run contributes ONE number
    -- its worst -- so a long run cannot outvote a short one, and the median across runs stays
    a median across runs rather than across requests.
    """
    usable = [value for value in offsets if value > 0]
    if model is None or not usable:
        return None
    store = load_template_costs(path)
    key = next((name for name in store if same_model(name, model)), model)
    previous = store.get(key)
    kept = ((previous.offsets if previous else ()) + (max(usable),))[-_TEMPLATE_SAMPLES:]
    updated = TemplateCost(model=key, offsets=kept, updated=time.time())
    store[key] = updated

    payload = {
        "models": {
            name: {"offsets": list(cost.offsets), "updated": cost.updated}
            for name, cost in store.items()
        }
    }
    write_json(path, payload, indent=2)
    return updated


# --- what the agent opens on its own ----------------------------------------------------------
#
# The third number. `pharos check` counts what you NAMED: files exactly, into the floor;
# directories in full, into a separate ceiling. What the agent decides to open once it starts
# working is in neither, and a floor that clears the budget by 2,000 tokens looks like a pass
# right up until the agent opens four files nobody mentioned.
#
# It turns out to be derivable from the store exactly as it already stands, which is the only
# reason it is allowed to exist. Between two consecutive requests of one conversation the input
# grows by three things and no others: what you typed, what the model last said, and whatever
# the client injected on its own -- tool results, file reads, retrieved context. The first two
# are already in the record, so the third is the remainder:
#
#     injected = (input_n - input_prev) - output_prev - (user_n - user_prev)
#
# That residue is what the agent opened without being told to. It is a COUNT derived from
# counts, and it names nothing: the store's promise -- counts only, never text, no file names,
# nothing reconstructable -- is untouched. A version of this that logged which files an agent
# read would have been easier and is not on the table.
#
# What the number is NOT: a bound. The floor stays the guarantee and the ceiling stays the
# worst case; this sits between them and says what usually happened. It is reported with the
# range it was drawn from precisely because one session is not the next.

_SESSION_GAP_S = 1800.0  # 30 minutes of silence ends a conversation

# Below this the median is a coincidence rather than a measurement, and the check says it has
# nothing to tell you instead of standing a guess in the gap. Same rule as everywhere else
# here: a missing number is honest, a made-up one is not.
_MIN_READ_SESSIONS = 3


@dataclass(frozen=True, slots=True)
class ReadEstimate:
    """How many tokens a client pulled in on its own over one task, learned from traffic."""

    tokens: int  # the number to plan against: median of per-session totals
    sessions: int  # how many conversations it was measured over
    low: int  # smallest session total
    high: int  # largest session total
    provenance: str

    @property
    def learned(self) -> bool:
        """True when this was measured from traffic, False when it was pinned in config.

        The distinction has to survive into the prose: a configured number has no
        conversations behind it and no range, and describing it as what agents "historically"
        did across "0 conversations" would be a sentence this project has no business writing.
        """
        return self.sessions > 0


def estimate_agent_reads(
    observations: list[Observation], model: str | None
) -> tuple[ReadEstimate | None, str | None]:
    """What a client historically added on its own, beyond the prompt and the model's replies.

    Returns ``(estimate, why_not)`` -- exactly one of the two, in the shape the rest of this
    project uses for an answer that may not exist.

    **The reason is not decoration.** The first live run of this estimator was made with the
    proxy still bound to another model's GGUF, so every pair was dropped as incommensurable and
    the check reported "there are not yet three conversations to learn from" -- when there were
    seven, and every one of them had been refused. The overhead estimator handles the same
    condition by falling back AND saying so in its provenance; this one had no way to say
    anything, so it said the only sentence it had. A missing number is honest here; a missing
    number with the wrong explanation attached is not.

    Agent-shaped records only. A bare `curl` at the endpoint opens no files, and the overhead
    estimator already learned once -- live, on real traffic -- what happens when a handful of
    pokes are allowed to vote on a number about coding agents.
    """
    pool = _matching(observations, model)
    if not pool:
        return None, (
            "no traffic has been seen for this model yet -- it is learned from whole agent "
            "conversations that go past the Pharos proxy"
        )
    shaped = [o for o in pool if o.agent_shaped]
    if not shaped:
        return None, (
            f"none of the {len(pool)} observed request(s) carried tools or a system prompt, so "
            f"none of them was a coding agent opening files on its own"
        )

    totals: list[int] = []
    pairs = 0
    refused: Counter[str] = Counter()
    for session in _sessions(shaped):
        injected = 0
        usable = 0
        for previous, current in zip(session, session[1:], strict=False):
            measured, why = _injected(previous, current)
            if measured is None:
                refused[why or "it could not be accounted for"] += 1
                continue
            injected += measured
            usable += 1
        if usable and injected > 0:
            totals.append(injected)
            pairs += usable

    if len(totals) < _MIN_READ_SESSIONS:
        return None, _why_not(totals, pairs, refused, len(shaped))
    return (
        ReadEstimate(
            tokens=int(statistics.median(totals)),
            sessions=len(totals),
            low=min(totals),
            high=max(totals),
            provenance=(
                f"observed · median of {len(totals)} conversations "
                f"({min(totals):,}-{max(totals):,} tokens), {pairs} turn-to-turn measurements"
            ),
        ),
        None,
    )


def _why_not(totals: list[int], kept: int, refused: Counter[str], records: int) -> str:
    """Say which of the several possible nothings this is.

    Dropped measurements outnumbering the kept ones is the interesting case and gets named
    with its own cause, because each cause has a different thing the reader could do about it:
    point the proxy at the model actually in use, or wait for a backend that reports its
    counts, or simply carry on working.
    """
    dropped = sum(refused.values())
    if dropped and dropped > kept:
        cause, count = refused.most_common(1)[0]
        return (
            f"{dropped} of {dropped + kept} turn-to-turn measurement(s) over {records} "
            f"observed request(s) could not be used -- {count} of them because {cause}. "
            f"Nothing is guessed in their place"
        )
    if not totals:
        return (
            f"{records} agent-shaped request(s) seen, but none of them formed a conversation "
            f"that grew -- this is learned from whole tasks, and at least "
            f"{_MIN_READ_SESSIONS} of them"
        )
    return (
        f"only {len(totals)} whole conversation(s) so far; {_MIN_READ_SESSIONS} is the minimum "
        f"below which a median is a coincidence rather than a measurement"
    )


def _sessions(records: list[Observation]) -> list[list[Observation]]:
    """Cut a flat record list into conversations, by the only signals counts can carry.

    A request continues the one before it when it went to the same endpoint, arrived inside
    ``_SESSION_GAP_S``, and GREW in all three ways a continuing conversation must: more
    messages, more input, and no less user-authored content than before. A fresh conversation
    restarts at a low message count and breaks the chain on the first test.

    The assumption, stated: one conversation is one task. Two agents talking to the proxy at
    once would interleave into nonsense here — the guards would mostly break them apart rather
    than merge them, but nothing in a counts-only record can prove which client sent what.
    """
    out: list[list[Observation]] = []
    current: list[Observation] = []
    for record in sorted(records, key=lambda o: o.ts):
        if current and _continues(current[-1], record):
            current.append(record)
            continue
        if len(current) > 1:
            out.append(current)
        current = [record]
    if len(current) > 1:
        out.append(current)
    return out


def _continues(previous: Observation, current: Observation) -> bool:
    return (
        current.endpoint == previous.endpoint
        and 0 <= current.ts - previous.ts <= _SESSION_GAP_S
        and current.messages > previous.messages
        and current.input_tokens > previous.input_tokens
        and current.user_tokens >= previous.user_tokens
    )


def _injected(previous: Observation, current: Observation) -> tuple[int | None, str | None]:
    """Tokens the client added between two turns that were neither typed nor generated.

    Returns ``(tokens, why_dropped)``. None for a pair that cannot be accounted for, and three
    cases qualify:

    * the previous response reported no ``eval_count`` — the model's own reply is then an
      unknown quantity sitting inside the growth, and subtracting nothing would bill it to the
      agent as a file it read;
    * one input is backend-exact and the other an estimate — the difference of those two
      carries the chat-template offset instead of cancelling it (see the template section
      below, where that same offset is the thing being measured);
    * either side counted its user content in another model's vocabulary, which makes the
      ``user`` term of the subtraction incommensurable with the rest.

    Clamped at zero. A reply that is not replayed verbatim into the next prompt — a thinking
    model whose reasoning block is dropped from the history — over-subtracts, and the honest
    floor for "tokens the agent read" is none rather than a negative number. That direction
    makes this estimate read LOW on such a model, which is the direction to be wrong in.
    """
    if previous.output_tokens is None:
        return None, "the backend reported no eval_count, so the reply itself is an unknown"
    if previous.input_exact != current.input_exact:
        return None, "one input was backend-exact and the other an estimate"
    if previous.user_exact is False or current.user_exact is False:
        return None, "it was counted in another model's vocabulary"
    growth = current.input_tokens - previous.input_tokens
    typed = current.user_tokens - previous.user_tokens
    return max(growth - previous.output_tokens - typed, 0), None
