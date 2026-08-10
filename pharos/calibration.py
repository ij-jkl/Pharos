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
import json
import logging
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pharos.naming import same_model

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
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
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
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self._path)


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
    agent = [o for o in pool if o.agent_shaped]
    shaped, shape_note = (agent, "agent-shaped") if agent else (pool, "all")
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
