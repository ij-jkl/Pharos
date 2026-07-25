"""FastAPI application factory for the proxy.

``create_app()`` wires the counted routes, the generic catch-all passthrough, the shared
upstream httpx client and the event bus. Route order matters: the four named routes register
first, then the catch-all picks up every other /api/* and /v1/* path so a client can point at
Pharos as its only endpoint. The proxy never mutates a request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response

from pharos.calibration import ObservationRecorder
from pharos.config import PharosConfig
from pharos.events import EventBus
from pharos.proxy import routes_ollama, routes_openai
from pharos.proxy.forward import ProxyState, passthrough_forward
from pharos.tokenizer.gguf import GgufTokenizer, TokenCounter
from pharos.tokenizer.resolver import resolve_gguf_path

# Generation can legitimately stall for a long time between chunks (e.g. prompt eval on a big
# context), so the read timeout is disabled; connect stays short so an unreachable backend
# fails fast instead of hanging the client.
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=None, pool=5.0)

_PASSTHROUGH_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


def create_app(
    config: PharosConfig,
    bus: EventBus,
    *,
    tokenizer: TokenCounter | None = None,
    resolve_tokenizer: bool = True,
    tokenizer_model: str | None = None,
    record_observations: bool = True,
) -> FastAPI:
    """Build the proxy app. ``tokenizer=None`` + ``resolve_tokenizer=False`` forces the
    heuristic counting path (used by tests); by default the GGUF is resolved from config.

    ``tokenizer_model`` names the model whose vocabulary ``tokenizer`` holds. Leave it None to
    accept every count at face value (the default for an injected test double); set it and any
    request naming a different model has its count marked untrusted rather than exact.

    ``record_observations`` controls the calibration sink (counts only, never text) that the
    pre-flight check learns client overhead from; tests disable it to avoid writing files."""
    if tokenizer is None and resolve_tokenizer:
        gguf = resolve_gguf_path(config)
        if gguf is not None:
            tokenizer = GgufTokenizer(gguf)
            # Remember WHICH model this vocabulary belongs to. v0.1 loads exactly one, so a
            # request naming a different model must not have its count labelled exact.
            tokenizer_model = config.model

    recorder = ObservationRecorder(Path(config.observations_file)) if record_observations else None
    client = httpx.AsyncClient(base_url=config.backend_url, timeout=_UPSTREAM_TIMEOUT)
    state = ProxyState(
        bus=bus,
        client=client,
        tokenizer=tokenizer,
        tokenizer_model=tokenizer_model,
        recorder=recorder,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await state.aclose()

    app = FastAPI(
        title="pharos-proxy",
        lifespan=lifespan,
        openapi_url=None,  # the proxy serves the backend's API surface, not its own docs
        docs_url=None,
        redoc_url=None,
    )
    app.state.pharos = state

    app.include_router(routes_openai.build_router(state))
    app.include_router(routes_ollama.build_router(state))

    @app.api_route("/api/{rest:path}", methods=_PASSTHROUGH_METHODS)
    @app.api_route("/v1/{rest:path}", methods=_PASSTHROUGH_METHODS)
    async def passthrough(request: Request) -> Response:
        return await passthrough_forward(state, request)

    return app
