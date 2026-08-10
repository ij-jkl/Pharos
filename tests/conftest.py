"""Shared pytest fixtures: LF-pinned file writes, event bus, fake tokenizer, proxy client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest

from pharos.config import PharosConfig
from pharos.events import EventBus, PharosEvent
from pharos.proxy.app import create_app
from pharos.tokenizer.gguf import TokenCounter


@pytest.fixture(autouse=True)
def no_forced_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip colour-forcing variables so Rich renders plain text under capture.

    Several tests assert on what the user reads — "matches 2 files", a verdict word, a warning.
    Rich honours FORCE_COLOR/CLICOLOR_FORCE even when its output is a pytest capture buffer, so
    on a machine that exports one (this shell exports FORCE_COLOR=3) those assertions fail
    against strings full of escape sequences, and the failure looks like a broken report rather
    than a broken environment. The suite should test the words, not the terminal it ran in.
    """
    for name in ("FORCE_COLOR", "CLICOLOR_FORCE", "PY_COLORS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def lf_only_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``Path.write_text`` emit LF on every platform, for every test.

    Fixture sizes ARE assertions here: "three files of ~1,500 heuristic tokens each" only
    holds if each file is 6,000 bytes. ``write_text`` translates "\\n" to ``os.linesep``, so
    the identical fixture is 6,000 bytes on Linux and 7,000 on Windows — and the counter reads
    what is on disk, which is right, because CRLF is what a model would receive.

    The consequence was that the suite tested a different thing on each platform, and the
    difference stayed invisible until one assertion sat near a budget threshold: the scope
    split test passed on Windows and failed on Linux CI purely on padding. Pinning the
    fixtures removes the platform from the arithmetic. Production code is untouched and still
    counts CRLF honestly wherever it finds it.
    """
    original = Path.write_text

    def write_lf(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        return original(self, data, encoding=encoding, errors=errors, newline=newline or "\n")

    monkeypatch.setattr(Path, "write_text", write_lf)


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
