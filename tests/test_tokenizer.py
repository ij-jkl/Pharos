"""Tokenizer tests: the count cache, the GGUF path resolver, and — when a real GGUF is
present under tests/models/ (gitignored) — exact vocab_only counting against it.

The real-GGUF tests auto-skip when no model file is available so the suite stays green on any
machine; pull a small GGUF (e.g. Qwen3-0.6B) into tests/models/ to enable them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharos.config import PharosConfig
from pharos.tokenizer.cache import TokenCountCache
from pharos.tokenizer.gguf import GgufTokenizer
from pharos.tokenizer.resolver import resolve_from_store, resolve_gguf_path

GGUF_DIR = Path(__file__).parent / "models"
GGUFS = sorted(GGUF_DIR.glob("*.gguf"))

requires_gguf = pytest.mark.skipif(
    not GGUFS, reason="no GGUF under tests/models/ — pull a small model to enable"
)


# --- cache --------------------------------------------------------------------------------------


class CountingSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> int:
        self.calls.append(text)
        return len(text)


def test_cache_counts_once_per_content() -> None:
    cache = TokenCountCache()
    spy = CountingSpy()
    assert cache.get_or_count("hello", spy) == 5
    assert cache.get_or_count("hello", spy) == 5
    assert spy.calls == ["hello"]


def test_cache_distinguishes_content() -> None:
    cache = TokenCountCache()
    spy = CountingSpy()
    assert cache.get_or_count("a", spy) == 1
    assert cache.get_or_count("bb", spy) == 2
    assert len(cache) == 2


def test_cache_evicts_least_recently_used() -> None:
    cache = TokenCountCache(max_entries=2)
    spy = CountingSpy()
    cache.get_or_count("first", spy)
    cache.get_or_count("second", spy)
    cache.get_or_count("first", spy)  # touch: "second" is now the least recently used
    cache.get_or_count("third", spy)  # evicts "second"
    assert len(cache) == 2
    cache.get_or_count("first", spy)  # still cached — no recount
    assert spy.calls.count("first") == 1
    cache.get_or_count("second", spy)  # evicted — recounted
    assert spy.calls.count("second") == 2


def test_cache_rejects_zero_capacity() -> None:
    with pytest.raises(ValueError):
        TokenCountCache(max_entries=0)


# --- resolver -----------------------------------------------------------------------------------


def _write_store(root: Path, *, name: str, tag: str, digest: str, with_blob: bool) -> None:
    manifest_dir = root / "manifests" / "registry.ollama.ai" / "library" / name
    manifest_dir.mkdir(parents=True)
    manifest = {
        "layers": [
            {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:tmpl"},
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest},
        ]
    }
    (manifest_dir / tag).write_text(json.dumps(manifest), encoding="utf-8")
    if with_blob:
        blobs = root / "blobs"
        blobs.mkdir()
        (blobs / digest.replace(":", "-")).write_bytes(b"GGUF")


def test_resolver_prefers_explicit_gguf_path(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    config = PharosConfig(gguf_path=str(gguf))
    assert resolve_gguf_path(config) == gguf


def test_resolver_missing_explicit_path_degrades_to_none(tmp_path: Path) -> None:
    config = PharosConfig(gguf_path=str(tmp_path / "absent.gguf"))
    assert resolve_gguf_path(config) is None


def test_resolver_no_path_and_no_model_is_none() -> None:
    assert resolve_gguf_path(PharosConfig()) is None


def test_store_resolves_model_with_tag(tmp_path: Path) -> None:
    _write_store(tmp_path, name="qwen3", tag="0.6b", digest="sha256:abc123", with_blob=True)
    resolved = resolve_from_store("qwen3:0.6b", root=tmp_path)
    assert resolved == tmp_path / "blobs" / "sha256-abc123"


def test_store_defaults_to_latest_tag(tmp_path: Path) -> None:
    _write_store(tmp_path, name="qwen3", tag="latest", digest="sha256:abc123", with_blob=True)
    assert resolve_from_store("qwen3", root=tmp_path) is not None


def test_store_missing_blob_is_none(tmp_path: Path) -> None:
    _write_store(tmp_path, name="qwen3", tag="latest", digest="sha256:abc123", with_blob=False)
    assert resolve_from_store("qwen3", root=tmp_path) is None


def test_store_malformed_manifest_is_none(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "qwen3"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "latest").write_text("not json", encoding="utf-8")
    assert resolve_from_store("qwen3", root=tmp_path) is None


def test_store_unknown_model_is_none(tmp_path: Path) -> None:
    assert resolve_from_store("missing", root=tmp_path) is None


def test_resolver_honors_ollama_models_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_store(tmp_path, name="qwen3", tag="latest", digest="sha256:abc123", with_blob=True)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    config = PharosConfig(model="qwen3")
    assert resolve_gguf_path(config) == tmp_path / "blobs" / "sha256-abc123"


# --- real GGUF (auto-skips when tests/models/ is empty) -----------------------------------------


@requires_gguf
def test_gguf_counts_real_text() -> None:
    tokenizer = GgufTokenizer(GGUFS[0])
    count = tokenizer.count("Hello, world! This is Pharos counting tokens.")
    assert count > 0
    assert count == len(tokenizer.tokenize("Hello, world! This is Pharos counting tokens."))


@requires_gguf
def test_gguf_count_is_deterministic() -> None:
    tokenizer = GgufTokenizer(GGUFS[0])
    assert tokenizer.count("the same text twice") == tokenizer.count("the same text twice")


@requires_gguf
def test_gguf_empty_text_counts_zero() -> None:
    assert GgufTokenizer(GGUFS[0]).count("") == 0


@requires_gguf
def test_gguf_tokenize_returns_ints() -> None:
    tokens = GgufTokenizer(GGUFS[0]).tokenize("hi")
    assert tokens
    assert all(isinstance(token, int) for token in tokens)


@requires_gguf
def test_gguf_with_cache_tokenizes_once() -> None:
    tokenizer = GgufTokenizer(GGUFS[0])
    cache = TokenCountCache()
    calls: list[str] = []

    def counting(text: str) -> int:
        calls.append(text)
        return tokenizer.count(text)

    first = cache.get_or_count("cache me", counting)
    second = cache.get_or_count("cache me", counting)
    assert first == second > 0
    assert calls == ["cache me"]
