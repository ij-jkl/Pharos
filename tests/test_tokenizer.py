"""Tokenizer tests: the count cache, the GGUF path resolver, and — when a real GGUF is
present under tests/models/ (gitignored) — exact vocab_only counting against it.

The real-GGUF tests auto-skip when no model file is available so the suite stays green on any
machine; pull a small GGUF (e.g. Qwen3-0.6B) into tests/models/ to enable them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

import pharos.tokenizer.gguf as gguf
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


def test_cache_does_not_leak_counts_across_tokenizers() -> None:
    """A count is only meaningful for the vocabulary that produced it.

    Keying on the text alone would hand one model's count to another the moment more than one
    tokenizer exists — a wrong number, delivered with a cache hit's confidence.
    """
    cache = TokenCountCache()
    qwen = CountingSpy()
    llama = CountingSpy()

    assert cache.get_or_count("hello world", qwen, identity="/blobs/qwen") == 11
    # Same text, different vocabulary: must miss and recount, not reuse the entry above.
    assert cache.get_or_count("hello world", llama, identity="/blobs/llama") == 11
    assert qwen.calls == ["hello world"]
    assert llama.calls == ["hello world"]

    # ...and each identity still caches independently on a repeat.
    cache.get_or_count("hello world", qwen, identity="/blobs/qwen")
    assert qwen.calls == ["hello world"]


def test_cache_identity_defaults_to_a_shared_namespace() -> None:
    cache = TokenCountCache()
    spy = CountingSpy()
    assert cache.get_or_count("abc", spy) == 3
    assert cache.get_or_count("abc", spy, identity="") == 3
    assert spy.calls == ["abc"]


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


# --- llama.cpp log sink ---------------------------------------------------------------------------
#
# Driven with a stand-in for the llama_cpp module rather than the real one: the behaviour under
# test is ours, and the native library offers no way to ask which callback is registered or to
# make it emit on demand. The stand-in also keeps these tests running on a machine with no GGUF.


class FakeLlamaCpp:
    """Just enough llama_cpp: the CFUNCTYPE decorator and the setter, both recording."""

    def __init__(self) -> None:
        self.registered: list[object] = []

    @staticmethod
    def llama_log_callback(func: object) -> object:
        return func

    def llama_log_set(self, callback: object, user_data: object) -> None:
        self.registered.append(callback)


@pytest.fixture
def fresh_sink() -> Iterator[None]:
    """Clear the process-wide sink around a test and put the original back."""
    saved = gguf._log_sink
    gguf._log_sink = None
    try:
        yield
    finally:
        gguf._log_sink = saved


def test_log_sink_registers_with_llama_cpp(fresh_sink: None) -> None:
    fake = FakeLlamaCpp()
    gguf._install_log_sink(fake)
    assert len(fake.registered) == 1
    # The reference has to survive the call: a collected ctypes callback is a crash, not a
    # traceback, and it would happen inside llama.cpp with no Python frame to blame.
    assert gguf._log_sink is fake.registered[0]


def test_log_sink_installs_only_once(fresh_sink: None) -> None:
    fake = FakeLlamaCpp()
    gguf._install_log_sink(fake)
    gguf._install_log_sink(fake)
    gguf._install_log_sink(FakeLlamaCpp())
    assert len(fake.registered) == 1


def test_log_sink_diverts_llama_output_to_the_log(
    fresh_sink: None, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeLlamaCpp()
    gguf._install_log_sink(fake)
    noisy = b"llama_context: n_ctx_seq (512) > n_ctx_train (0) -- possible training context\n"

    with caplog.at_level(logging.DEBUG, logger="pharos.tokenizer"):
        fake.registered[0](2, noisy, None)

    assert "n_ctx_seq (512) > n_ctx_train (0)" in caplog.text
    # The whole point: the loader note is kept, but it never reaches the terminal, where it
    # would land under the `pharos check` report or on top of the TUI.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_log_sink_swallows_its_own_failures(fresh_sink: None) -> None:
    """A raising sink surfaces as "Exception ignored on calling ctypes callback function".

    It is raised into C, so no caller can catch it and no `try` around the model load helps.
    Teardown is the real case — logging is half dismantled there — so the body must be total.
    """
    fake = FakeLlamaCpp()
    gguf._install_log_sink(fake)
    sink = fake.registered[0]

    class Exploding:
        def decode(self, *args: object, **kwargs: object) -> str:
            raise RuntimeError("interpreter is going down")

    sink(2, Exploding(), None)  # must not raise


# --- the store lookup stays inside the store -----------------------------------------------------


def _store(root: Path) -> Path:
    """A minimal Ollama blob store holding one resolvable model."""
    manifest = {
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:deadbeef"}
        ]
    }
    entry = root / "store" / "manifests" / "registry" / "library" / "qwen"
    entry.mkdir(parents=True)
    (entry / "latest").write_text(json.dumps(manifest), encoding="utf-8")
    blobs = root / "store" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "sha256-deadbeef").write_text("weights", encoding="utf-8")
    # A file of the same shape, outside the store, for a lookup to try to reach.
    outside = root / "elsewhere"
    outside.mkdir()
    (outside / "latest").write_text(json.dumps(manifest), encoding="utf-8")
    return root / "store"


def test_a_model_name_cannot_walk_out_of_the_blob_store(tmp_path: Path) -> None:
    """The name goes straight into a glob pattern, and `..` in a glob is resolved by the OS.

    It arrives from pharos.toml or from whatever the backend reported at /api/ps, so it is not
    hostile in any ordinary setup — but a lookup documented as "in an Ollama blob store" should
    be one, and the digest check in resolve_from_store already holds that line for the blob
    half. Measured before the fix: a name of `../../elsewhere` returned a manifest two
    directories above the one being searched.
    """
    store = _store(tmp_path)

    for name in ("../../elsewhere:latest", "registry/../../../elsewhere:latest"):
        assert resolve_from_store(name, root=store) is None, name


def test_an_ordinary_name_still_resolves(tmp_path: Path) -> None:
    """The positive control: containment must not cost a lookup that was always legitimate."""
    store = _store(tmp_path)

    assert resolve_from_store("qwen:latest", root=store) == store / "blobs" / "sha256-deadbeef"
