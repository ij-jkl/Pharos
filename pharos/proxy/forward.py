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
import functools
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

from pharos.calibration import Observation, ObservationRecorder
from pharos.events import (
    EventBus,
    InputCounted,
    RequestAborted,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
)
from pharos.naming import same_model
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
        tokenizer_model: str | None = None,
        recorder: ObservationRecorder | None = None,
    ) -> None:
        self.bus = bus
        self.client = client
        self.tokenizer = tokenizer
        # The model whose vocabulary ``tokenizer`` actually holds. v0.1 resolves exactly one
        # tokenizer at startup, so a request for any other model is counted with the wrong
        # vocabulary; that count is labelled untrusted rather than exact. See _count_input.
        self.tokenizer_model = tokenizer_model
        # Optional calibration sink (counts only, never text); None disables recording.
        self.recorder = recorder
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
        if self.recorder is not None:
            # Final flush AFTER task cancellation so buffered observations from cancelled
            # record tasks are not lost; short sessions still leave calibration data behind.
            await self.recorder.aclose()
        await self.client.aclose()


@dataclass(frozen=True, slots=True)
class InputSpec:
    """What a counted route extracted from a parsed copy of the request body."""

    text: str
    exact: bool  # True only when the backend tokenizes exactly this string (e.g. raw=true)
    model: str | None
    stream: bool
    tokens: int | None = None  # pre-known count (e.g. a token-array prompt); skips tokenizing
    extra_tokens: int = 0  # already-tokenized input alongside the text (e.g. `context`)
    opaque: str | None = None  # names content Pharos cannot tokenize at all (e.g. "images")
    # For calibration: the user-authored portion alone, and how many messages the request
    # carried. client overhead = total input - user content (see pharos.calibration).
    user_text: str = ""
    message_count: int = 0
    # True when the request carried a tool catalogue or a system prompt — the signature of a
    # coding agent rather than a bare poke at the endpoint. The overhead estimator needs it to
    # compare like with like; see pharos.calibration.
    agent_shaped: bool = False


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
        if state.recorder is not None and spec is not None and upstream.status_code < 400:
            # Calibration: record what this request really cost (counts only, never text).
            # Rejected requests are excluded — they are not representative traffic.
            state.spawn(_record_observation(state, endpoint, spec, metrics))

    return _relay(upstream, tee=tee, on_close=on_close)


async def _record_observation(
    state: ProxyState, endpoint: str, spec: InputSpec, metrics: StreamMetrics
) -> None:
    """Assemble and buffer a calibration record, entirely off the request path.

    ``spec.text`` was already counted for the InputCounted event, so recounting it here is a
    cache hit; ``spec.user_text`` costs one extra tokenize in a worker thread.
    """
    recorder = state.recorder
    if recorder is None:
        return
    try:
        if metrics.prompt_eval_count is not None:
            input_tokens, input_exact = metrics.prompt_eval_count, True
        elif spec.tokens is not None:
            input_tokens, input_exact = spec.tokens + spec.extra_tokens, False
        else:
            input_tokens = await _count_text(state, spec.text) + spec.extra_tokens
            input_exact = False
        user_tokens = await _count_text(state, spec.user_text) if spec.user_text else 0
        observation = Observation(
            ts=time.time(),
            endpoint=endpoint,
            model=spec.model,
            input_tokens=input_tokens,
            input_exact=input_exact,
            user_tokens=user_tokens,
            messages=spec.message_count,
            output_tokens=metrics.eval_count,
            agent_shaped=spec.agent_shaped,
        )
        if recorder.add(observation):
            await recorder.flush()
    except Exception as exc:  # noqa: BLE001 — calibration must never affect proxying
        _logger.warning("observation recording failed (%s)", exc)


async def _count_text(state: ProxyState, text: str) -> int:
    """Count with the loaded tokenizer (through the cache) or the labeled heuristic."""
    counter = state.tokenizer
    if counter is None:
        return _heuristic_count(text)
    try:
        return await asyncio.to_thread(
            functools.partial(state.cache.get_or_count, identity=_tokenizer_identity(counter)),
            text,
            counter.count,
        )
    except Exception:  # noqa: BLE001
        return _heuristic_count(text)


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
    """Count input tokens out of band and publish the labeled result.

    THE COUNTING CONTRACT — what the input estimate does and does not include.

    Counted: every part of the request body that the backend renders into the prompt and that
    Pharos can tokenize. Message content (plain strings and typed text parts), the system
    prompt, tool DEFINITIONS and assistant tool_calls history (serialized to compact JSON, an
    approximation of the template's own rendering), a /v1/completions suffix, and any
    already-tokenized ``context`` array (added by length).

    NOT counted, deliberately:

    * The chat template's own scaffolding — role markers, and the fixed prelude a model emits
      when tools are present (~200 tokens on qwen3.5-9b-heretic, ~10 for a bare chat). It is
      applied server-side, so Pharos never sees the string, and its size is a property of the
      model rather than the request. Hardcoding a fitted constant would be confidently wrong
      on the next model; reconciliation against ``prompt_eval_count`` closes it instead.
    * A ``template`` override on /api/generate, for the same reason.
    * Images. Vision models price them by a separate mechanism entirely, so no text-based
      count applies. Their presence downgrades the label rather than guessing a number.

    Fields verified NOT to reach the prompt on Ollama 0.31.1, and correctly ignored:
    ``format`` (string or JSON schema), ``response_format``, ``tool_choice``, ``think`` —
    these constrain decoding rather than being injected as text.

    The result is always an estimate unless the backend tokenizes exactly what Pharos saw
    (``raw=true``), and reconciliation later replaces it with ground truth where reported.
    """
    if spec.tokens is not None:
        # Pre-counted input (a token-array prompt): the request itself carried the count.
        state.bus.publish(
            InputCounted(
                request_id=request_id,
                tokens=spec.tokens + spec.extra_tokens,
                exact=spec.exact,
                source="request",
            )
        )
        return
    tokens: int | None = None
    source = "heuristic"
    exact = False
    counter = state.tokenizer
    if counter is not None:
        # v0.1 holds ONE vocabulary, resolved from config.model. Counting a request aimed at a
        # different model with it produces a plausible but wrong number, so the result is
        # labelled "gguf:other-model" and never exact — an admitted unknown beats a confident
        # error. Per-request tokenizer resolution is a v0.2 item.
        trusted = _tokenizer_matches(state, spec)
        identity = _tokenizer_identity(counter)
        try:
            tokens = await asyncio.to_thread(
                functools.partial(state.cache.get_or_count, identity=identity),
                spec.text,
                counter.count,
            )
            source = "gguf" if trusted else "gguf:other-model"
            exact = spec.exact and trusted
        except Exception as exc:
            # Counting must never break a request — degrade to the heuristic, but loudly.
            _logger.warning("#%d input counting failed (%s); using heuristic", request_id, exc)
            tokens = None
    if tokens is None:
        tokens = _heuristic_count(spec.text)
    tokens += spec.extra_tokens
    if spec.opaque is not None:
        # Content Pharos cannot tokenize at all (images). Whatever the text count says, the
        # total is unknowable — say so rather than publish a number that looks authoritative.
        source = f"gguf:{spec.opaque}" if source == "gguf" else source
        exact = False
    state.bus.publish(
        InputCounted(request_id=request_id, tokens=tokens, exact=exact, source=source)
    )


def _tokenizer_matches(state: ProxyState, spec: InputSpec) -> bool:
    """Whether the loaded vocabulary is actually the one this request's model uses.

    When the tokenizer's model is unknown (a test double, or no model configured) the count is
    taken at face value — there is nothing to contradict it. A request that names a *different*
    model is the case worth flagging.
    """
    if state.tokenizer_model is None or spec.model is None:
        return True
    return same_model(spec.model, state.tokenizer_model)


def _tokenizer_identity(counter: TokenCounter) -> str:
    """Namespace for cached counts: the GGUF path when there is one, else the class name."""
    path = getattr(counter, "path", None)
    return str(path) if path is not None else type(counter).__name__


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


_IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})


def content_text(content: object) -> str:
    """Message content as text: a plain string, or a list of typed parts with ``text`` fields."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "".join(parts)
    return ""


def content_has_image(content: object) -> bool:
    """Whether a content value carries an image part (OpenAI-style multimodal array)."""
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in _IMAGE_PART_TYPES:
            return True
        if any(key in part for key in _IMAGE_PART_TYPES):
            return True
    return False


def message_has_image(message: dict[str, Any]) -> bool:
    """Ollama puts images in a sibling ``images`` array; OpenAI nests them in content parts."""
    images = message.get("images")
    if isinstance(images, list) and images:
        return True
    return content_has_image(message.get("content"))


def json_text(value: object) -> str:
    """Compact JSON for a structure the backend renders into the prompt (tools, tool_calls).

    An approximation on purpose: the chat template re-renders these into its own wire format,
    so the serialized size tracks the real cost closely but never matches it exactly. Measured
    on qwen3.5-9b-heretic: ~89 tokens of JSON per tool against ~88 tokens of actual prompt
    growth. Good enough for a labeled estimate, and vastly better than counting zero.

    Default separators on purpose — NOT ``separators=(",", ":")``. Compacting strips whitespace
    that the rendered form actually contains, which measurably undercounts (66 tokens per tool
    against the real 88, versus 89 with default spacing). This is a serialization choice, not a
    fudge factor: no constant is fitted to any observation.
    """
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return ""


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
