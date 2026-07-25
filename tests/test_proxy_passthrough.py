"""Proxy tests: byte-for-byte streaming passthrough plus the out-of-band counting tee, with a
respx-mocked upstream. Every counted route must forward the exact original bytes, relay the
exact upstream bytes back, and emit correctly labeled events without touching the stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from starlette.requests import Request as StarletteRequest

from pharos.events import (
    InputCounted,
    PharosEvent,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
)
from pharos.proxy.forward import (
    StreamMetrics,
    TailBuffer,
    _open_upstream,
    _relay,
    extract_metrics,
)

BASE = "http://localhost:11434"


class ChunkStream(httpx.AsyncByteStream):
    """An upstream body served in explicit chunks, so streaming is actually exercised."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


async def drain(queue: asyncio.Queue[PharosEvent], *kinds: type) -> dict[type, PharosEvent]:
    """Consume events until one of each requested kind has been seen."""
    found: dict[type, PharosEvent] = {}
    while any(kind not in found for kind in kinds):
        event = await asyncio.wait_for(queue.get(), timeout=2.0)
        found.setdefault(type(event), event)
    return found


NDJSON_CHUNKS = [
    b'{"model":"m","message":{"role":"assistant","content":"He"},"done":false}\n',
    b'{"model":"m","message":{"role":"assistant","content":"llo"},"done":false}\n',
    b'{"model":"m","done":true,"prompt_eval_count":7,"eval_count":4,"eval_duration":2000000000}\n',
]

SSE_CHUNKS_NO_USAGE = [
    b'data: {"id":"1","choices":[{"delta":{"content":"He"}}],"usage":null}\n\n',
    b'data: {"id":"1","choices":[{"delta":{"content":"llo"}}],"usage":null}\n\n',
    b"data: [DONE]\n\n",
]

SSE_CHUNKS_WITH_USAGE = [
    b'data: {"id":"1","choices":[{"delta":{"content":"Hi"}}],"usage":null}\n\n',
    b'data: {"id":"1","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":5}}\n\n',
    b"data: [DONE]\n\n",
]


async def test_ollama_chat_stream_byte_identical_and_counted(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    route = respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(NDJSON_CHUNKS),
            headers={"content-type": "application/x-ndjson"},
        )
    )
    # Deliberately odd spacing: the proxy must forward these bytes VERBATIM, not re-serialize.
    body = b'{"model": "m",  "messages": [{"role":"user","content":"alpha beta gamma"}] }'

    client = make_proxy(tokenizer=fake_tokenizer)
    resp = await client.post(
        "/api/chat", content=body, headers={"content-type": "application/json"}
    )

    assert resp.status_code == 200
    assert resp.content == b"".join(NDJSON_CHUNKS)
    assert resp.headers["content-type"] == "application/x-ndjson"
    assert "content-length" not in resp.headers
    assert "transfer-encoding" not in resp.headers
    assert route.calls.last.request.content == body

    found = await drain(events, RequestStarted, InputCounted, RequestCompleted)
    started = found[RequestStarted]
    assert (started.endpoint, started.model, started.stream) == ("ollama-chat", "m", True)
    counted = found[InputCounted]
    assert (counted.tokens, counted.exact, counted.source) == (3, False, "gguf")
    assert fake_tokenizer.calls == ["alpha beta gamma"]
    completed = found[RequestCompleted]
    assert completed.status_code == 200
    assert completed.prompt_eval_count == 7  # reconciled ground truth from the done chunk
    assert completed.eval_count == 4
    assert completed.tokens_per_second == pytest.approx(2.0)  # 4 / (2e9 ns / 1e9)
    assert completed.duration_s >= 0


async def test_generate_raw_true_is_exact(respx_mock, make_proxy, events, fake_tokenizer) -> None:
    upstream_body = json.dumps(
        {
            "response": "ok",
            "done": True,
            "prompt_eval_count": 3,
            "eval_count": 2,
            "eval_duration": 1000000000,
        }
    ).encode()
    respx_mock.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(
            200, content=upstream_body, headers={"content-type": "application/json"}
        )
    )
    body = b'{"model":"m","prompt":"a b c","raw":true,"stream":false}'

    client = make_proxy(tokenizer=fake_tokenizer)
    resp = await client.post("/api/generate", content=body)

    assert resp.content == upstream_body
    found = await drain(events, RequestStarted, InputCounted, RequestCompleted)
    assert found[RequestStarted].stream is False
    counted = found[InputCounted]
    assert (counted.tokens, counted.exact, counted.source) == (3, True, "gguf")
    completed = found[RequestCompleted]
    assert completed.prompt_eval_count == 3
    assert completed.tokens_per_second == pytest.approx(2.0)


async def test_generate_templated_counts_system_plus_prompt_as_estimate(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    respx_mock.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, content=b'{"done":true}')
    )
    body = b'{"model":"m","prompt":"tail words","system":"sys here"}'

    client = make_proxy(tokenizer=fake_tokenizer)
    await client.post("/api/generate", content=body)

    found = await drain(events, InputCounted)
    counted = found[InputCounted]
    assert (counted.tokens, counted.exact) == (4, False)  # "sys here\ntail words" -> 4 words
    assert fake_tokenizer.calls == ["sys here\ntail words"]


async def test_openai_sse_without_usage_keeps_estimate(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    route = respx_mock.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(SSE_CHUNKS_NO_USAGE),
            headers={"content-type": "text/event-stream"},
        )
    )
    body = b'{"model":"m","messages":[{"role":"user","content":"alpha beta"}],"stream":true}'

    client = make_proxy(tokenizer=fake_tokenizer)
    resp = await client.post("/v1/chat/completions", content=body)

    assert resp.content == b"".join(SSE_CHUNKS_NO_USAGE)
    assert resp.headers["content-type"] == "text/event-stream"
    assert route.calls.last.request.content == body  # verbatim — no include_usage injection
    found = await drain(events, RequestStarted, InputCounted, RequestCompleted)
    assert found[RequestStarted].endpoint == "openai-chat"
    assert found[RequestStarted].stream is True
    counted = found[InputCounted]
    assert (counted.exact, counted.source) == (False, "gguf")  # /v1 chat is always an estimate
    completed = found[RequestCompleted]
    # usage was never reported (client did not send include_usage): no reconciliation —
    # the labeled estimate from InputCounted stands.
    assert completed.prompt_eval_count is None
    assert completed.eval_count is None
    assert completed.tokens_per_second is None


async def test_openai_sse_with_include_usage_reconciles(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    route = respx_mock.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(SSE_CHUNKS_WITH_USAGE),
            headers={"content-type": "text/event-stream"},
        )
    )
    body = (
        b'{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,'
        b'"stream_options":{"include_usage":true}}'
    )

    client = make_proxy(tokenizer=fake_tokenizer)
    resp = await client.post("/v1/chat/completions", content=body)

    assert resp.content == b"".join(SSE_CHUNKS_WITH_USAGE)
    # The client asked for usage itself; Pharos must forward that request untouched.
    assert route.calls.last.request.content == body
    completed = (await drain(events, RequestCompleted))[RequestCompleted]
    assert completed.prompt_eval_count == 9
    assert completed.eval_count == 5
    assert completed.tokens_per_second is None  # OpenAI usage carries no duration


async def test_openai_non_streaming_reconciles_usage(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    upstream_body = json.dumps(
        {
            "id": "1",
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 6},
        }
    ).encode()
    route = respx_mock.post(f"{BASE}/v1/completions").mock(
        return_value=httpx.Response(
            200, content=upstream_body, headers={"content-type": "application/json"}
        )
    )
    body = b'{"model":"m","prompt":"one two three"}'

    client = make_proxy(tokenizer=fake_tokenizer)
    resp = await client.post("/v1/completions", content=body)

    assert resp.content == upstream_body
    assert route.calls.last.request.content == body
    found = await drain(events, RequestStarted, InputCounted, RequestCompleted)
    assert found[RequestStarted].endpoint == "openai-completions"
    counted = found[InputCounted]
    assert (counted.tokens, counted.exact, counted.source) == (3, False, "gguf")
    completed = found[RequestCompleted]
    assert completed.prompt_eval_count == 11
    assert completed.eval_count == 6


async def test_catch_all_forwards_untouched_without_events(respx_mock, make_proxy, events) -> None:
    upstream_body = b'{"models": []}'
    route = respx_mock.get(f"{BASE}/api/tags", params={"verbose": "true"}).mock(
        return_value=httpx.Response(
            200, content=upstream_body, headers={"content-type": "application/json"}
        )
    )

    client = make_proxy()
    resp = await client.get("/api/tags?verbose=true")

    assert resp.status_code == 200
    assert resp.content == upstream_body
    assert route.called
    await asyncio.sleep(0)
    assert events.qsize() == 0  # catch-all traffic is not counted and not evented


async def test_unreachable_upstream_returns_502_and_emits_failure(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    respx_mock.post(f"{BASE}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    client = make_proxy(tokenizer=fake_tokenizer)

    resp = await client.post("/api/chat", content=b'{"messages":[]}')

    assert resp.status_code == 502
    assert "pharos" in resp.json()["error"]
    found = await drain(events, RequestStarted, RequestFailed)
    assert found[RequestStarted].request_id == found[RequestFailed].request_id


async def test_catch_all_unreachable_returns_502(respx_mock, make_proxy) -> None:
    respx_mock.get(f"{BASE}/api/ps").mock(side_effect=httpx.ConnectError("refused"))
    client = make_proxy()
    resp = await client.get("/api/ps")
    assert resp.status_code == 502
    assert "pharos" in resp.json()["error"]


async def test_streamed_response_drops_upstream_framing_headers(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    chunks = [b'{"done":true}\n']
    respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(
            200,
            stream=ChunkStream(chunks),
            headers={
                "content-type": "application/x-ndjson",
                "content-length": str(len(chunks[0])),
                "transfer-encoding": "chunked",
                "x-upstream": "yes",
            },
        )
    )
    client = make_proxy(tokenizer=fake_tokenizer)
    resp = await client.post("/api/chat", content=b"{}")

    assert resp.content == chunks[0]
    assert "content-length" not in resp.headers  # the ASGI server frames the re-streamed body
    assert "transfer-encoding" not in resp.headers
    assert resp.headers["x-upstream"] == "yes"
    await drain(events, RequestCompleted)


async def test_request_headers_forwarded_and_hop_by_hop_dropped(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    route = respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(200, content=b'{"done":true}')
    )
    body = b'{"messages":[{"role":"user","content":"x"}]}'
    client = make_proxy(tokenizer=fake_tokenizer)

    await client.post(
        "/api/chat",
        content=body,
        headers={
            "authorization": "Bearer secret",
            "x-custom": "1",
            "proxy-authorization": "hop-by-hop",
        },
    )

    sent = route.calls.last.request
    assert sent.headers["host"] == "localhost:11434"  # rewritten for the upstream hop
    assert sent.headers["authorization"] == "Bearer secret"
    assert sent.headers["x-custom"] == "1"
    assert "proxy-authorization" not in sent.headers
    assert sent.headers["content-length"] == str(len(body))
    await drain(events, RequestCompleted)


async def test_missing_accept_encoding_becomes_identity(respx_mock) -> None:
    """White-box: without a client Accept-Encoding, the upstream hop must not invent gzip."""
    route = respx_mock.post("http://upstream.test/api/chat").mock(
        return_value=httpx.Response(200, content=b"{}")
    )
    client = httpx.AsyncClient(base_url="http://upstream.test")
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "server": ("pharos.test", 80),
        "path": "/api/chat",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    request = StarletteRequest(scope)

    upstream = await _open_upstream(client, request, content=b"{}")
    await upstream.aclose()
    await client.aclose()

    assert route.calls.last.request.headers["accept-encoding"] == "identity"


async def test_invalid_json_body_is_still_forwarded_verbatim(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    route = respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(400, content=b'{"error":"invalid request"}')
    )
    body = b"not json {{"
    client = make_proxy(tokenizer=fake_tokenizer)

    resp = await client.post("/api/chat", content=body)

    assert resp.status_code == 400  # upstream's error status passes through untouched
    assert route.calls.last.request.content == body
    found = await drain(events, RequestStarted, RequestCompleted)
    assert found[RequestStarted].model is None
    assert found[RequestCompleted].status_code == 400
    await asyncio.sleep(0)
    assert events.qsize() == 0  # unparseable body -> nothing to count, no InputCounted
    assert fake_tokenizer.calls == []


async def test_heuristic_when_no_tokenizer(respx_mock, make_proxy, events) -> None:
    respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(200, content=b'{"done":true}')
    )
    body = b'{"messages":[{"role":"user","content":"alpha beta gamma"}]}'
    client = make_proxy(tokenizer=None)

    await client.post("/api/chat", content=body)

    counted = (await drain(events, InputCounted))[InputCounted]
    # "alpha beta gamma" is 16 chars -> ceil(16/4) = 4, distinct from its 3 words.
    assert (counted.tokens, counted.exact, counted.source) == (4, False, "heuristic")


async def test_openai_completions_token_array_prompt_is_exact(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    respx_mock.post(f"{BASE}/v1/completions").mock(return_value=httpx.Response(200, content=b"{}"))
    body = b'{"model":"m","prompt":[11,12,13,14,15]}'
    client = make_proxy(tokenizer=fake_tokenizer)

    await client.post("/v1/completions", content=body)

    counted = (await drain(events, InputCounted))[InputCounted]
    # A pre-tokenized prompt IS its own count: len(array), exact, no tokenizer involved.
    assert (counted.tokens, counted.exact, counted.source) == (5, True, "request")
    assert fake_tokenizer.calls == []


async def test_catch_all_forwards_request_bodies(respx_mock, make_proxy, events) -> None:
    route = respx_mock.post(f"{BASE}/api/show").mock(
        return_value=httpx.Response(200, content=b'{"details":{}}')
    )
    body = b'{"model": "qwen3.5:9b" }'
    client = make_proxy()

    resp = await client.post("/api/show", content=body)

    assert resp.status_code == 200
    assert resp.content == b'{"details":{}}'
    assert route.calls.last.request.content == body
    await asyncio.sleep(0)
    assert events.qsize() == 0


async def test_catch_all_preserves_percent_encoded_paths(respx_mock, make_proxy) -> None:
    route = respx_mock.route(method="GET", url__regex=r".*/api/blobs/.*").mock(
        return_value=httpx.Response(200, content=b"{}")
    )
    client = make_proxy()

    resp = await client.get("/api/blobs/foo%2Fbar")

    assert resp.status_code == 200
    # %2F must reach the upstream intact — a decoded path would collapse it to a real slash.
    assert route.calls.last.request.url.raw_path == b"/api/blobs/foo%2Fbar"


class ExplodingTokenizer:
    def count(self, text: str) -> int:
        raise RuntimeError("vocab load failed")


async def test_tokenizer_failure_degrades_to_heuristic(respx_mock, make_proxy, events) -> None:
    respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(200, content=b'{"done":true}')
    )
    body = b'{"messages":[{"role":"user","content":"alpha beta gamma"}]}'
    client = make_proxy(tokenizer=ExplodingTokenizer())

    await client.post("/api/chat", content=body)

    counted = (await drain(events, InputCounted))[InputCounted]
    assert (counted.tokens, counted.exact, counted.source) == (4, False, "heuristic")


# --- forward.py units: tail buffer + metrics extraction -----------------------------------------


def test_tail_buffer_keeps_only_the_tail() -> None:
    buffer = TailBuffer(limit=8)
    buffer.feed(b"abcdef")
    buffer.feed(b"ghijkl")
    assert buffer.tail() == b"efghijkl"


def test_extract_metrics_ndjson_final_done() -> None:
    metrics = extract_metrics(b"".join(NDJSON_CHUNKS))
    assert metrics == StreamMetrics(prompt_eval_count=7, eval_count=4, eval_duration_ns=2000000000)
    assert metrics.tokens_per_second == pytest.approx(2.0)


def test_extract_metrics_sse_usage_present() -> None:
    metrics = extract_metrics(b"".join(SSE_CHUNKS_WITH_USAGE))
    assert metrics.prompt_eval_count == 9
    assert metrics.eval_count == 5
    assert metrics.tokens_per_second is None


def test_extract_metrics_sse_usage_absent_is_empty() -> None:
    assert extract_metrics(b"".join(SSE_CHUNKS_NO_USAGE)) == StreamMetrics()


def test_extract_metrics_whole_body_json() -> None:
    body = json.dumps({"done": True, "prompt_eval_count": 12, "eval_count": 8}).encode()
    metrics = extract_metrics(body)
    assert metrics.prompt_eval_count == 12
    assert metrics.eval_count == 8
    assert metrics.tokens_per_second is None  # no duration reported


def test_extract_metrics_truncated_json_falls_back_to_regex() -> None:
    line = (
        b'{"response":"' + b"x" * 100 + b'",'
        b'"prompt_eval_count":7,"eval_count":4,"eval_duration":2000000000,"done":true}'
    )
    buffer = TailBuffer(limit=80)  # cuts off the JSON head -> structured parsing fails
    buffer.feed(line)
    metrics = extract_metrics(buffer.tail())
    assert metrics == StreamMetrics(prompt_eval_count=7, eval_count=4, eval_duration_ns=2000000000)


def test_extract_metrics_garbage_is_empty() -> None:
    assert extract_metrics(b"\x00\xffnot metrics at all") == StreamMetrics()


# --- forward.py units: relay cleanup (upstream must never leak) ----------------------------------


class ExplodingCloseStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"x"

    async def aclose(self) -> None:
        raise RuntimeError("boom on close")


async def test_relay_closes_upstream_even_if_body_never_iterated() -> None:
    upstream = httpx.Response(200, stream=ChunkStream([b"x"]))
    fired: list[bool] = []
    resp = _relay(upstream, tee=None, on_close=fired.append)

    # A client that vanishes before ASGI iterates the body leaves a never-started generator;
    # Starlette still runs the background task, which must release the upstream connection.
    await resp.background()

    assert upstream.is_closed
    assert fired == [True]  # the stream never ran to its end -> reported as aborted


async def test_relay_cleanup_fires_exactly_once_after_full_stream() -> None:
    upstream = httpx.Response(200, stream=ChunkStream([b"a", b"b"]))
    fired: list[bool] = []
    resp = _relay(upstream, tee=None, on_close=fired.append)

    chunks = [chunk async for chunk in resp.body_iterator]
    await resp.background()  # the second cleanup invocation must be a no-op

    assert chunks == [b"a", b"b"]
    assert fired == [False]  # fully relayed -> not aborted


async def test_on_close_fires_even_when_upstream_close_raises() -> None:
    upstream = httpx.Response(200, stream=ExplodingCloseStream())
    fired: list[bool] = []
    resp = _relay(upstream, tee=None, on_close=fired.append)

    await resp.background()  # aclose raises inside -> swallowed; the event still fires

    assert fired == [True]


# --- counting with the wrong model's vocabulary -------------------------------------------------
#
# v0.1 loads ONE tokenizer, from config.model. A request naming a different model gets counted
# with it anyway; confirmed on real hardware to be wrong (llama3.2 counted 5 by Qwen's vocab vs
# Ollama's 6). The count must therefore never be presented as exact.

_GEN_DONE = json.dumps(
    {"model": "m", "done": True, "prompt_eval_count": 3, "eval_count": 4,
     "eval_duration": 2_000_000_000}
).encode()


async def test_other_model_count_is_downgraded_to_untrusted(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    """raw=true would normally be exact — but not when the vocabulary belongs to another model."""
    respx_mock.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(
            200, content=_GEN_DONE, headers={"content-type": "application/json"}
        )
    )
    client = make_proxy(tokenizer=fake_tokenizer, tokenizer_model="qwen3.5-9b-heretic")
    await client.post(
        "/api/generate",
        content=b'{"model":"llama3.2:1b","prompt":"a b c","raw":true,"stream":false}',
    )

    counted = (await drain(events, InputCounted))[InputCounted]
    assert counted.source == "gguf:other-model"
    assert counted.exact is False  # raw=true alone must NOT buy exactness here
    assert counted.tokens == 3  # still reported — labelled honestly, not withheld


async def test_matching_model_keeps_exact_label(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    """The configured model still counts as exact, and an absent tag must not break the match."""
    respx_mock.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(
            200, content=_GEN_DONE, headers={"content-type": "application/json"}
        )
    )
    # config says untagged, request says ":latest" — the same model, so exactness survives.
    client = make_proxy(tokenizer=fake_tokenizer, tokenizer_model="qwen3.5-9b-heretic")
    await client.post(
        "/api/generate",
        content=b'{"model":"qwen3.5-9b-heretic:latest","prompt":"a b c","raw":true,"stream":false}',
    )

    counted = (await drain(events, InputCounted))[InputCounted]
    assert (counted.source, counted.exact) == ("gguf", True)


# --- prompt-contributing fields beyond message content ------------------------------------------
#
# Audited against Ollama 0.31.1: tools, assistant tool_calls history and /api/generate `context`
# all reach the prompt and were counted as zero. `format`, `response_format`, `tool_choice` and
# `think` were measured NOT to reach it, so they stay uncounted on purpose.

_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


async def _counted_for(respx_mock, make_proxy, events, tokenizer, path, payload):
    respx_mock.post(f"{BASE}{path}").mock(
        return_value=httpx.Response(
            200, content=_GEN_DONE, headers={"content-type": "application/json"}
        )
    )
    client = make_proxy(tokenizer=tokenizer)
    await client.post(path, content=json.dumps(payload).encode())
    return (await drain(events, InputCounted))[InputCounted]


@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_tool_definitions_are_counted(
    respx_mock, make_proxy, events, fake_tokenizer, path
) -> None:
    """Previously zero: a coding agent ships its whole tool catalogue on every request."""
    msg = {"role": "user", "content": "alpha beta"}
    without = await _counted_for(
        respx_mock, make_proxy, events, fake_tokenizer, path,
        {"model": "m", "messages": [msg], "stream": False},
    )
    with_tools = await _counted_for(
        respx_mock, make_proxy, events, fake_tokenizer, path,
        {"model": "m", "messages": [msg], "tools": [_TOOL], "stream": False},
    )
    assert without.tokens == 2
    assert with_tools.tokens > without.tokens
    # The serialized schema is what gets counted, so the growth tracks the tool's size.
    serialized = json.dumps([_TOOL], ensure_ascii=False)
    assert with_tools.tokens == without.tokens + len(serialized.split())
    assert with_tools.exact is False  # still an estimate, never promoted


@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_assistant_tool_calls_history_is_counted(
    respx_mock, make_proxy, events, fake_tokenizer, path
) -> None:
    calls = [{"function": {"name": "get_weather", "arguments": {"city": "Lisbon"}}}]
    counted = await _counted_for(
        respx_mock, make_proxy, events, fake_tokenizer, path,
        {"model": "m", "stream": False, "messages": [
            {"role": "user", "content": "alpha"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "content": "sunny"}]},
    )
    # "alpha" + "sunny" alone would be 2; the serialized tool_calls add the rest.
    assert counted.tokens > 2


async def test_generate_context_array_adds_its_length(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    """`context` is an already-tokenized prefix: it contributes length, not text."""
    counted = await _counted_for(
        respx_mock, make_proxy, events, fake_tokenizer, "/api/generate",
        {"model": "m", "prompt": "a b c", "context": list(range(50)),
         "raw": True, "stream": False},
    )
    assert counted.tokens == 3 + 50
    # Length is close but not exact (measured +49 for 50 entries), so raw=true loses exactness.
    assert counted.exact is False


async def test_generate_suffix_is_counted(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    counted = await _counted_for(
        respx_mock, make_proxy, events, fake_tokenizer, "/api/generate",
        {"model": "m", "prompt": "a b", "suffix": "c d e", "stream": False},
    )
    assert counted.tokens == 5


async def test_completions_suffix_is_counted(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    counted = await _counted_for(
        respx_mock, make_proxy, events, fake_tokenizer, "/v1/completions",
        {"model": "m", "prompt": "a b", "suffix": "c d e", "stream": False},
    )
    assert counted.tokens == 5


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/chat", {"model": "m", "stream": False, "messages": [
            {"role": "user", "content": "look", "images": ["BASE64DATA"]}]}),
        ("/v1/chat/completions", {"model": "m", "stream": False, "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}),
    ],
)
async def test_images_downgrade_the_label(
    respx_mock, make_proxy, events, fake_tokenizer, path, payload
) -> None:
    """Vision tokens are priced by a different mechanism; no text count can stand in."""
    counted = await _counted_for(respx_mock, make_proxy, events, fake_tokenizer, path, payload)
    assert counted.source == "gguf:images"
    assert counted.exact is False


@pytest.mark.parametrize("field", ["format", "response_format", "tool_choice", "think"])
async def test_decoding_only_fields_are_not_counted(
    respx_mock, make_proxy, events, fake_tokenizer, field
) -> None:
    """Measured on Ollama 0.31.1 not to reach the prompt — counting them would inflate."""
    base = {"model": "m", "stream": False, "messages": [{"role": "user", "content": "alpha beta"}]}
    counted = await _counted_for(
        respx_mock, make_proxy, events, fake_tokenizer, "/api/chat",
        {**base, field: {"type": "object", "properties": {"x": {"type": "string"}}}},
    )
    assert counted.tokens == 2


async def test_ollama_chat_reads_list_shaped_content(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    """The native route used to read only plain strings, silently counting parts as zero."""
    counted = await _counted_for(
        respx_mock, make_proxy, events, fake_tokenizer, "/api/chat",
        {"model": "m", "stream": False, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "alpha beta gamma"}]}]},
    )
    assert counted.tokens == 3


async def test_unknown_tokenizer_model_is_taken_at_face_value(
    respx_mock, make_proxy, events, fake_tokenizer
) -> None:
    """With no declared tokenizer model there is nothing to contradict the count."""
    respx_mock.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(
            200, content=_GEN_DONE, headers={"content-type": "application/json"}
        )
    )
    client = make_proxy(tokenizer=fake_tokenizer)  # tokenizer_model left None
    await client.post(
        "/api/generate",
        content=b'{"model":"anything-at-all","prompt":"a b c","raw":true,"stream":false}',
    )

    counted = (await drain(events, InputCounted))[InputCounted]
    assert (counted.source, counted.exact) == ("gguf", True)
