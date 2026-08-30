"""Ollama-native routes: /api/chat and /api/generate (counted + evented).

Handlers parse a COPY of the JSON body for input estimates and forward the original bytes
verbatim — never re-serialize. Chat input is always an estimate: Ollama applies the model's
chat template server-side, so Pharos never sees the true prompt string. /api/generate is
exact only with ``raw=true`` (the template is bypassed and the prompt IS the tokenized text).

Everything in the body that reaches the prompt is counted, not just message content — see the
counting contract on ``pharos.proxy.forward._count_input``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response

from pharos.proxy.forward import (
    InputSpec,
    ProxyState,
    all_ints,
    content_text,
    counted_forward,
    json_text,
    message_has_image,
    payload_model,
    payload_stream,
)


def build_router(state: ProxyState) -> APIRouter:
    router = APIRouter()

    @router.post("/api/chat")
    async def api_chat(request: Request) -> Response:
        return await counted_forward(state, request, endpoint="ollama-chat", extract=_extract_chat)

    @router.post("/api/generate")
    async def api_generate(request: Request) -> Response:
        return await counted_forward(
            state, request, endpoint="ollama-generate", extract=_extract_generate
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
    # Tool definitions dominate a coding agent's input and were previously counted as zero:
    # measured 268 uncounted tokens for a single trivial tool, 382 for three.
    tools = payload.get("tools")
    has_tools = isinstance(tools, list) and bool(tools)
    if has_tools:
        texts.append(json_text(tools))
    return InputSpec(
        text="\n".join(texts),
        exact=False,  # the chat template is applied server-side; we never see the real string
        model=payload_model(payload),
        stream=payload_stream(payload, default=True),
        opaque="images" if has_image else None,
        user_text="\n".join(user_texts),
        message_count=message_count,
        agent_shaped=has_tools or has_system,
        has_tools=has_tools,
    )


def _extract_generate(payload: dict[str, Any]) -> InputSpec:
    raw = payload.get("raw") is True
    prompt = payload.get("prompt")
    text = prompt if isinstance(prompt, str) else ""
    if not raw:
        system = payload.get("system")
        if isinstance(system, str) and system:
            text = f"{system}\n{text}"
    suffix = payload.get("suffix")
    if isinstance(suffix, str) and suffix:
        text = f"{text}\n{suffix}"
    # `context` is an already-tokenized conversation prefix: it contributes its own length,
    # with no text for the tokenizer to see. Measured +49 prompt tokens for 50 entries, so the
    # length is a close estimate but not exact — enough to drop the raw=true exactness claim.
    context = payload.get("context")
    extra = len(context) if isinstance(context, list) and all_ints(context) else 0
    images = payload.get("images")
    has_image = isinstance(images, list) and bool(images)
    return InputSpec(
        text=text,
        exact=raw and extra == 0,
        model=payload_model(payload),
        stream=payload_stream(payload, default=True),
        extra_tokens=extra,
        opaque="images" if has_image else None,
        # The prompt is the user-authored portion; system/suffix/context are client-added.
        user_text=prompt if isinstance(prompt, str) else "",
        message_count=1,
        # /api/generate has no tool catalogue, so a system prompt is the only agent tell.
        agent_shaped=isinstance(payload.get("system"), str) and bool(payload.get("system")),
    )
