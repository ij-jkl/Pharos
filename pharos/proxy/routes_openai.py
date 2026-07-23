"""OpenAI-compatible routes: /v1/chat/completions and /v1/completions (counted + evented).

Handlers parse a COPY of the JSON body for input estimates and forward the original bytes
verbatim — never re-serialize. Both endpoints are always estimates: the backend applies the
model's template server-side. Reconciliation against ``usage`` happens only when the backend
actually reports it (non-streaming responses, or streaming with
``stream_options.include_usage`` — which Pharos never injects; passthrough is sacred).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response

from pharos.proxy.forward import (
    InputSpec,
    ProxyState,
    counted_forward,
    payload_model,
    payload_stream,
)


def build_router(state: ProxyState) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await counted_forward(state, request, endpoint="openai-chat", extract=_extract_chat)

    @router.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await counted_forward(
            state, request, endpoint="openai-completions", extract=_extract_completions
        )

    return router


def _extract_chat(payload: dict[str, Any]) -> InputSpec:
    texts: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                text = _content_text(message.get("content"))
                if text:
                    texts.append(text)
    return InputSpec(
        text="\n".join(texts),
        exact=False,
        model=payload_model(payload),
        stream=payload_stream(payload, default=False),
    )


def _extract_completions(payload: dict[str, Any]) -> InputSpec:
    prompt = payload.get("prompt")
    model = payload_model(payload)
    stream = payload_stream(payload, default=False)
    if isinstance(prompt, list) and prompt and _all_ints(prompt):
        # A pre-tokenized prompt IS its own count: len(array) is exact by definition —
        # no tokenizer involved and no template shift to estimate around.
        return InputSpec(text="", exact=True, model=model, stream=stream, tokens=len(prompt))
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, list):
        text = "\n".join(item for item in prompt if isinstance(item, str))
    else:
        text = ""
    return InputSpec(text=text, exact=False, model=model, stream=stream)


def _all_ints(items: list[Any]) -> bool:
    return all(isinstance(item, int) and not isinstance(item, bool) for item in items)


def _content_text(content: object) -> str:
    """Chat message content: a plain string, or a list of typed parts with ``text`` fields."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""
