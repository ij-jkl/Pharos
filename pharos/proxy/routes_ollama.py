"""Ollama-native routes: /api/chat and /api/generate (counted + evented).

Handlers parse a COPY of the JSON body for input estimates and forward the original bytes
verbatim — never re-serialize. Chat input is always an estimate: Ollama applies the model's
chat template server-side, so Pharos never sees the true prompt string. /api/generate is
exact only with ``raw=true`` (the template is bypassed and the prompt IS the tokenized text).
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
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    texts.append(content)
    return InputSpec(
        text="\n".join(texts),
        exact=False,  # the chat template is applied server-side; we never see the real string
        model=payload_model(payload),
        stream=payload_stream(payload, default=True),
    )


def _extract_generate(payload: dict[str, Any]) -> InputSpec:
    raw = payload.get("raw") is True
    prompt = payload.get("prompt")
    text = prompt if isinstance(prompt, str) else ""
    if not raw:
        system = payload.get("system")
        if isinstance(system, str) and system:
            text = f"{system}\n{text}"
    return InputSpec(
        text=text,
        exact=raw,
        model=payload_model(payload),
        stream=payload_stream(payload, default=True),
    )
