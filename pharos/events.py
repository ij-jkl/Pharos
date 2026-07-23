"""Internal async pub/sub event bus and event types.

The proxy *emits* events; the TUI and accountant *consume* them. This decouples the proxy from
the UI so neither blocks the other. All cross-component communication flows through this bus.

``publish()`` never blocks and never raises: every subscriber has its own bounded queue, and
under backpressure the OLDEST event for that subscriber is dropped to make room. For a
monitoring bus, losing a stale event to a stalled consumer beats ever stalling the proxy's
request path.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestStarted:
    """A counted request arrived at the proxy and is about to be forwarded upstream."""

    request_id: int
    method: str
    path: str
    endpoint: str  # "ollama-chat" | "ollama-generate" | "openai-chat" | "openai-completions"
    model: str | None
    stream: bool


@dataclass(frozen=True, slots=True)
class InputCounted:
    """Out-of-band input token count for a request.

    ``exact`` is True only when the backend will tokenize exactly the string Pharos saw
    (e.g. /api/generate with raw=true). Templated chat input makes this an estimate; the
    reconciled truth arrives later in ``RequestCompleted.prompt_eval_count``.
    """

    request_id: int
    tokens: int
    exact: bool
    source: str  # "gguf" (real tokenizer) | "request" (client token array) | "heuristic"


@dataclass(frozen=True, slots=True)
class RequestCompleted:
    """The upstream response finished streaming (any status code).

    ``prompt_eval_count`` is the backend-reported ground truth for input tokens. It is None
    whenever the backend did not report it — the normal case for /v1 streaming without
    ``stream_options.include_usage`` — and then the labeled ``InputCounted`` estimate stands.
    """

    request_id: int
    status_code: int
    prompt_eval_count: int | None
    eval_count: int | None
    eval_duration_ns: int | None
    tokens_per_second: float | None
    duration_s: float


@dataclass(frozen=True, slots=True)
class RequestAborted:
    """The response stream ended before completion — client abort or upstream drop.

    Distinct from ``RequestCompleted`` so a dead request is never rendered as a normal
    completion; whatever metrics the tail held are discarded as unreliable.
    """

    request_id: int
    status_code: int  # the status that had already been relayed when the stream died
    duration_s: float


@dataclass(frozen=True, slots=True)
class RequestFailed:
    """The request could not be forwarded (e.g. the backend is unreachable)."""

    request_id: int
    error: str


PharosEvent = RequestStarted | InputCounted | RequestCompleted | RequestAborted | RequestFailed

_DEFAULT_QUEUE_SIZE = 1024


class EventBus:
    """Fan-out pub/sub over per-subscriber ``asyncio.Queue``s. Publishing never blocks."""

    def __init__(self, queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._queue_size = queue_size
        self._subscribers: list[asyncio.Queue[PharosEvent]] = []
        self._dropped: dict[asyncio.Queue[PharosEvent], int] = {}

    def subscribe(self) -> asyncio.Queue[PharosEvent]:
        """Register a new subscriber and return its private event queue."""
        queue: asyncio.Queue[PharosEvent] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(queue)
        self._dropped[queue] = 0
        return queue

    def unsubscribe(self, queue: asyncio.Queue[PharosEvent]) -> None:
        """Remove a subscriber; unknown queues are ignored."""
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)
        self._dropped.pop(queue, None)

    def dropped_count(self, queue: asyncio.Queue[PharosEvent]) -> int:
        """Total events dropped for this subscriber, so a UI can surface the gap honestly."""
        return self._dropped.get(queue, 0)

    def publish(self, event: PharosEvent) -> None:
        """Deliver ``event`` to every subscriber without ever blocking or raising.

        A full subscriber queue drops its oldest event to admit the new one (drop-oldest
        policy) — ``put_nowait`` on a full bounded queue would otherwise raise QueueFull.
        """
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._dropped[queue] = self._dropped.get(queue, 0) + 1
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)
