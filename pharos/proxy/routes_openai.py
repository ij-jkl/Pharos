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
    content_text,
    counted_forward,
    json_text,
    message_has_image,
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
    user_texts: list[str] = []
    message_count = 0
    has_image = False
    has_system = False
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            message_count += 1
            if message.get("role") == "system":
                has_system = True
            text = content_text(message.get("content"))
            if text:
                texts.append(text)
                if message.get("role") == "user":
                    user_texts.append(text)
            # Assistant tool calls are replayed into the prompt as part of the history.
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                texts.append(json_text(calls))
            if message_has_image(message):
                has_image = True
    # Tool definitions dominate a coding agent's input; previously counted as zero.
    # ``functions`` is the deprecated spelling and lands in the prompt the same way.
    has_tools = False
    for key in ("tools", "functions"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            texts.append(json_text(value))
            has_tools = True
    return InputSpec(
        text="\n".join(texts),
        exact=False,
        model=payload_model(payload),
        stream=payload_stream(payload, default=False),
        opaque="images" if has_image else None,
        user_text="\n".join(user_texts),
        message_count=message_count,
        agent_shaped=has_tools or has_system,
    )


def _extract_completions(payload: dict[str, Any]) -> InputSpec:
    prompt = payload.get("prompt")
    model = payload_model(payload)
    stream = payload_stream(payload, default=False)
    if isinstance(prompt, list) and prompt and _all_ints(prompt):
        # A pre-tokenized prompt IS its own count: len(array) is exact by definition —
        # no tokenizer involved and no template shift to estimate around.
        return InputSpec(
            text="", exact=True, model=model, stream=stream, tokens=len(prompt), message_count=1
        )
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, list):
        text = "\n".join(item for item in prompt if isinstance(item, str))
    else:
        text = ""
    user_text = text
    # An infill suffix is prompt content like any other.
    suffix = payload.get("suffix")
    if isinstance(suffix, str) and suffix:
        text = f"{text}\n{suffix}"
    return InputSpec(
        text=text, exact=False, model=model, stream=stream, user_text=user_text, message_count=1
    )


def _all_ints(items: list[Any]) -> bool:
    return all(isinstance(item, int) and not isinstance(item, bool) for item in items)
