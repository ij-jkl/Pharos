"""Shared pytest fixtures: event bus, fake tokenizer, and a proxy-app client factory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from pharos.config import PharosConfig
from pharos.events import EventBus, PharosEvent
from pharos.proxy.app import create_app
from pharos.tokenizer.gguf import TokenCounter


class FakeTokenizer:
    """Deterministic TokenCounter: one token per whitespace-separated word; records calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def count(self, text: str) -> int:
        self.calls.append(text)
        return len(text.split())


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def events(bus: EventBus) -> asyncio.Queue[PharosEvent]:
    return bus.subscribe()


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


MakeProxy = Callable[..., httpx.AsyncClient]


@pytest.fixture
async def make_proxy(bus: EventBus) -> AsyncIterator[MakeProxy]:
    """Factory building a proxy app plus an ASGI test client; everything closes on teardown.

    The upstream client inside the app uses the real (respx-patched) HTTP transport; the
    returned client talks to the proxy app in-process via ASGITransport.
    """
    created = []

    def _make(
        *,
        tokenizer: TokenCounter | None = None,
        config: PharosConfig | None = None,
        tokenizer_model: str | None = None,
        record_observations: bool = False,  # tests opt in; never write files by default
    ) -> httpx.AsyncClient:
        app = create_app(
            config if config is not None else PharosConfig(),
            bus,
            tokenizer=tokenizer,
            resolve_tokenizer=False,  # tests must never depend on the host's Ollama store
            tokenizer_model=tokenizer_model,
            record_observations=record_observations,
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://pharos.test"
        )
        created.append((app, client))
        return client

    yield _make

    for app, client in created:
        await client.aclose()
        await app.state.pharos.aclose()
