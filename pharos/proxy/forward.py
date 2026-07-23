"""Byte-for-byte streaming passthrough over httpx, plus an out-of-band token-counting tee.

v0.1 is a pure passthrough: it never mutates a request. The counted routes parse a COPY of
the request body for token estimates and forward the original bytes verbatim — never
re-serialize (hard rule). Counting must never delay or alter the stream: input counting runs
as a background task, and response metrics are parsed from a bounded tail buffer only after
the stream has closed.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from pharos.events import (
    EventBus,
    InputCounted,
    RequestAborted,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
)
from pharos.tokenizer.cache import TokenCountCache
from pharos.tokenizer.gguf import TokenCounter

# Hop-by-hop headers (RFC 9110 §7.6.1) never travel across a proxy hop. On responses,
# Content-Length and Transfer-Encoding are dropped as well: the body is re-streamed, so the
# ASGI server must be the one to frame it — forwarding upstream framing headers alongside a
# re-streamed body yields truncated or hung responses.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_DROP_REQUEST_HEADERS = _HOP_BY_HOP | {"host", "content-length"}
_DROP_RESPONSE_HEADERS = _HOP_BY_HOP | {"content-length"}

# The final metrics chunk sits at the very end of the body, so a bounded tail is enough; the
# limit only needs to comfortably exceed one NDJSON/SSE chunk plus any trailing done object.
_TAIL_LIMIT = 64 * 1024

_HEURISTIC_CHARS_PER_TOKEN = 4

# Full per-request detail goes to the log file; the TUI event log is terse and lossy by design.
_logger = logging.getLogger("pharos.proxy")


class ProxyState:
    """Shared per-app state threaded through the routes; created once in ``create_app()``."""

    def __init__(
        self,
        *,
        bus: EventBus,
        client: httpx.AsyncClient,
        tokenizer: TokenCounter | None,
        cache: TokenCountCache | None = None,
    ) -> None:
        self.bus = bus
        self.client = client
        self.tokenizer = tokenizer
        self.cache = cache if cache is not None else TokenCountCache()
        self._ids = itertools.count(1)
        self._tasks: set[asyncio.Task[None]] = set()

    def next_request_id(self) -> int:
        return next(self._ids)

    def spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run ``coro`` in the background, holding a strong reference until it finishes."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.client.aclose()


@dataclass(frozen=True, slots=True)
class InputSpec:
    """What a counted route extracted from a parsed copy of the request body."""

    text: str
    exact: bool  # True only when the backend tokenizes exactly this string (e.g. raw=true)
    model: str | None
    stream: bool
    tokens: int | None = None  # pre-known count (e.g. a token-array prompt); skips tokenizing


async def counted_forward(
    state: ProxyState,
    request: Request,
    *,
    endpoint: str,
    extract: Callable[[dict[str, Any]], InputSpec],
) -> Response:
    """Forward a named route: emit events and count input, streaming the body untouched."""
    body = await request.body()
    payload = parse_json_object(body)
    spec = extract(payload) if payload is not None else None

    request_id = state.next_request_id()
    state.bus.publish(
        RequestStarted(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            endpoint=endpoint,
            model=spec.model if spec is not None else None,
            stream=spec.stream if spec is not None else False,
        )
    )
    if spec is not None:
        state.spawn(_count_input(state, request_id, spec))
    _logger.info(
        "#%d %s %s endpoint=%s model=%s stream=%s body=%dB",
        request_id,
        request.method,
        request.url.path,
        endpoint,
        spec.model if spec is not None else None,
        spec.stream if spec is not None else False,
        len(body),
    )

    started = time.monotonic()
    try:
        upstream = await _open_upstream(state.client, request, content=body)
    except httpx.HTTPError as exc:
        detail = f"upstream request failed: {type(exc).__name__}: {exc}"
        _logger.warning("#%d %s", request_id, detail)
        state.bus.publish(RequestFailed(request_id=request_id, error=detail))
        return JSONResponse({"error": f"pharos: {detail}"}, status_code=502)

    tee = TailBuffer()

    def on_close(aborted: bool) -> None:
        duration = time.monotonic() - started
        if aborted:
            # The stream died early (client abort or upstream drop): the tail is unreliable,
            # so no metrics are reconciled and the event says plainly what happened.
            _logger.warning(
                "#%d aborted after %.2fs (status %d was already sent)",
                request_id,
                duration,
                upstream.status_code,
            )
            state.bus.publish(
                RequestAborted(
                    request_id=request_id,
                    status_code=upstream.status_code,
                    duration_s=duration,
                )
            )
            return
        metrics = extract_metrics(tee.tail())
        _logger.info(
            "#%d completed status=%d prompt_eval=%s eval=%s eval_duration_ns=%s duration=%.2fs",
            request_id,
            upstream.status_code,
            metrics.prompt_eval_count,
            metrics.eval_count,
            metrics.eval_duration_ns,
            duration,
        )
        state.bus.publish(
            RequestCompleted(
                request_id=request_id,
                status_code=upstream.status_code,
                prompt_eval_count=metrics.prompt_eval_count,
                eval_count=metrics.eval_count,
                eval_duration_ns=metrics.eval_duration_ns,
                tokens_per_second=metrics.tokens_per_second,
                duration_s=duration,
            )
        )

    return _relay(upstream, tee=tee, on_close=on_close)


async def passthrough_forward(state: ProxyState, request: Request) -> Response:
    """Forward any other /api/* or /v1/* path untouched: no counting, no buffering."""
    try:
        upstream = await _open_upstream(state.client, request, content=_request_content(request))
    except httpx.HTTPError as exc:
        detail = f"upstream request failed: {type(exc).__name__}: {exc}"
        _logger.warning("passthrough %s %s: %s", request.method, request.url.path, detail)
        return JSONResponse({"error": f"pharos: {detail}"}, status_code=502)
    return _relay(upstream, tee=None, on_close=None)


async def _open_upstream(
    client: httpx.AsyncClient,
    request: Request,
    *,
    content: bytes | AsyncIterator[bytes] | None,
) -> httpx.Response:
    headers = [
        (name.decode("latin-1"), value.decode("latin-1"))
        for name, value in request.headers.raw  # ASGI header names are already lowercase
        if name.decode("latin-1") not in _DROP_REQUEST_HEADERS
    ]
    # Transparency: if the client did not negotiate compression, forbid it upstream too —
    # httpx would otherwise advertise gzip itself and the client would receive encoded bytes
    # it never asked to decode.
    if "accept-encoding" not in request.headers:
        headers.append(("accept-encoding", "identity"))
    # The ASGI raw_path keeps percent-encoding intact (request.url.path is decoded, which
    # would corrupt e.g. %2F); some servers include the query string in it, so split it off.
    raw_path = request.scope.get("raw_path")
    path = bytes(raw_path).split(b"?", 1)[0].decode("latin-1") if raw_path else request.url.path
    target = path + (f"?{request.url.query}" if request.url.query else "")
    upstream_request = client.build_request(
        request.method, target, headers=headers, content=content
    )
    return await client.send(upstream_request, stream=True)


def _request_content(request: Request) -> AsyncIterator[bytes] | None:
    """The request body as a stream, or None for bodyless requests (e.g. plain GETs)."""
    if (
        request.headers.get("content-length") in (None, "0")
        and "transfer-encoding" not in request.headers
    ):
        return None
    return request.stream()


def _relay(
    upstream: httpx.Response,
    *,
    tee: TailBuffer | None,
    on_close: Callable[[bool], None] | None,
) -> StreamingResponse:
    """Stream the upstream body through unchanged, with cleanup that cannot be skipped.

    The generator's ``finally`` covers the normal path, but a client that disconnects before
    the ASGI server ever iterates the body leaves a never-started generator whose ``finally``
    never runs — so the same idempotent cleanup is also attached as the response's background
    task, which Starlette runs on every exit path (including disconnect). Without it, an
    aborted request would hold its upstream connection forever, since the read timeout is
    disabled for long generations.
    """
    cleanup = _RelayCleanup(upstream, on_close)

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                if tee is not None:
                    tee.feed(chunk)
                yield chunk
            cleanup.mark_streamed()  # the loop ran to the end: the body was fully relayed
        finally:
            await cleanup.run()

    headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() not in _DROP_RESPONSE_HEADERS
    }
    return StreamingResponse(
        body(),
        status_code=upstream.status_code,
        headers=headers,
        background=BackgroundTask(cleanup.run),
    )


class _RelayCleanup:
    """Close the upstream response and fire ``on_close`` exactly once, on any exit path.

    ``on_close`` receives ``aborted=True`` unless the relay generator streamed the body to
    its natural end — a never-started generator and a mid-stream disconnect both count as
    aborted.
    """

    def __init__(self, upstream: httpx.Response, on_close: Callable[[bool], None] | None) -> None:
        self._upstream = upstream
        self._on_close = on_close
        self._done = False
        self._streamed = False

    def mark_streamed(self) -> None:
        self._streamed = True

    async def run(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            # Teardown noise must never mask the stream; the completion event still fires.
            with contextlib.suppress(Exception):
                await self._upstream.aclose()
        finally:
            if self._on_close is not None:
                self._on_close(not self._streamed)


class TailBuffer:
    """Accumulates only the last ``limit`` bytes fed to it — bounded memory for any stream."""

    def __init__(self, limit: int = _TAIL_LIMIT) -> None:
        self._limit = limit
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._buf.extend(chunk)
        overflow = len(self._buf) - self._limit
        if overflow > 0:
            del self._buf[:overflow]

    def tail(self) -> bytes:
        return bytes(self._buf)


@dataclass(frozen=True, slots=True)
class StreamMetrics:
    """Final token metrics reported by the backend, when it reported any."""

    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if self.eval_count is None or not self.eval_duration_ns:
            return None
        return self.eval_count / (self.eval_duration_ns / 1e9)


def extract_metrics(tail: bytes) -> StreamMetrics:
    """Pull final token metrics out of a response tail, defensively.

    Handles the shapes Ollama produces — NDJSON streaming (/api/*: the final ``done`` object
    carries prompt_eval_count / eval_count / eval_duration), SSE streaming (/v1/*: ``usage``
    is null on every chunk unless the client sent ``stream_options.include_usage`` — missing
    usage is the normal case and yields empty metrics, so the labeled estimate stands) and
    plain JSON (non-streaming). A regex sweep is the last resort for a tail whose JSON head
    was truncated by the buffer limit.
    """
    text = tail.decode("utf-8", errors="replace")
    best = StreamMetrics()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if not line or line == "[DONE]":
            continue
        payload = parse_json_object(line)
        if payload is None:
            continue
        found = _metrics_from_payload(payload)
        if found is not None:
            best = found
    if best == StreamMetrics():
        payload = parse_json_object(text.strip())
        found = _metrics_from_payload(payload) if payload is not None else None
        if found is not None:
            best = found
    if best == StreamMetrics():
        best = _metrics_from_regex(text)
    return best


def _metrics_from_payload(payload: dict[str, Any]) -> StreamMetrics | None:
    prompt = _as_int(payload.get("prompt_eval_count"))
    output = _as_int(payload.get("eval_count"))
    duration = _as_int(payload.get("eval_duration"))
    if prompt is None and output is None and duration is None:
        usage = payload.get("usage")
        if isinstance(usage, dict):
            prompt = _as_int(usage.get("prompt_tokens"))
            output = _as_int(usage.get("completion_tokens"))
    if prompt is None and output is None and duration is None:
        return None
    return StreamMetrics(prompt_eval_count=prompt, eval_count=output, eval_duration_ns=duration)


def _metrics_from_regex(text: str) -> StreamMetrics:
    def last_int(name: str) -> int | None:
        matches = re.findall(rf'"{name}"\s*:\s*(\d+)', text)
        return int(matches[-1]) if matches else None

    prompt = last_int("prompt_eval_count")
    if prompt is None:
        prompt = last_int("prompt_tokens")
    output = last_int("eval_count")
    if output is None:
        output = last_int("completion_tokens")
    return StreamMetrics(
        prompt_eval_count=prompt,
        eval_count=output,
        eval_duration_ns=last_int("eval_duration"),
    )


async def _count_input(state: ProxyState, request_id: int, spec: InputSpec) -> None:
    """Count input tokens out of band and publish the labeled result."""
    if spec.tokens is not None:
        # Pre-counted input (a token-array prompt): the request itself carried the count.
        state.bus.publish(
            InputCounted(
                request_id=request_id, tokens=spec.tokens, exact=spec.exact, source="request"
            )
        )
        return
    tokens: int | None = None
    source = "heuristic"
    exact = False
    counter = state.tokenizer
    if counter is not None:
        try:
            tokens = await asyncio.to_thread(state.cache.get_or_count, spec.text, counter.count)
            source = "gguf"
            exact = spec.exact
        except Exception as exc:
            # Counting must never break a request — degrade to the heuristic, but loudly.
            _logger.warning("#%d input counting failed (%s); using heuristic", request_id, exc)
            tokens = None
    if tokens is None:
        tokens = _heuristic_count(spec.text)
    state.bus.publish(
        InputCounted(request_id=request_id, tokens=tokens, exact=exact, source=source)
    )


def _heuristic_count(text: str) -> int:
    """Ceiling of chars/4 — the crude fallback when no GGUF tokenizer is available."""
    per = _HEURISTIC_CHARS_PER_TOKEN
    return (len(text) + per - 1) // per


# --- shared payload helpers for the route extractors --------------------------------------------


def parse_json_object(raw: bytes | str) -> dict[str, Any] | None:
    """Parse ``raw`` as a JSON object; None for invalid JSON or a non-object top level."""
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def payload_model(payload: dict[str, Any]) -> str | None:
    model = payload.get("model")
    return model if isinstance(model, str) else None


def payload_stream(payload: dict[str, Any], *, default: bool) -> bool:
    stream = payload.get("stream")
    return stream if isinstance(stream, bool) else default


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
