"""Load the GGUF vocabulary once and count tokens exactly.

Uses ``Llama(model_path=..., vocab_only=True)``: loads only the vocabulary — light, fast,
CPU-only, no GPU required — and an exact match to the tokenizer the model actually uses.

Loading is lazy (deferred to the first count) and serialized by a lock, as is tokenize()
itself: llama-cpp contexts are not documented thread-safe and callers run in worker threads
via ``asyncio.to_thread``.

Special-token text is parsed (``special=True``), and ``add_bos=True`` maps to llama.cpp's
``add_special``: BOS is prepended only when the model's own GGUF metadata asks for it — the
same rule the backend applies to a raw prompt — so raw counts stay exact for both BOS-adding
(Llama-family) and non-BOS (Qwen) models. ``prompt_eval_count`` from the response remains
the ground truth.

CONFIRMED 2026-07-24 (llama-cpp-python 0.3.34), both directions:

* Negative — Qwen3 declares add_bos_token=false: ``add_bos=True`` and ``add_bos=False`` both
  return 2 tokens for b"hello world". No BOS is forced.
* Positive — llama3.2:1b (add_bos_token absent, llama arch default true): ``add_bos=True``
  returns 3 tokens vs 2, and ``tokens[0] == token_bos()`` (128000).

So raw-exact counts are not off by one on BOS-adding models.

This module answers only "how many tokens is this string". WHICH strings get counted for a
request — and what is deliberately left out of the estimate — is the counting contract
documented on ``pharos.proxy.forward._count_input``. Read that before trusting a total.

Caveat, unrelated to BOS: the proxy binds ONE tokenizer from ``config.model`` at startup, so a
request targeting a different model is counted with the wrong vocabulary. Such counts are
labelled ``gguf:other-model`` and never exact. See item 6 in DESKTOP_VALIDATION.md.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

_logger = logging.getLogger("pharos.tokenizer")


class TokenCounter(Protocol):
    """Anything that can count tokens for a text; the proxy depends only on this."""

    def count(self, text: str) -> int: ...


class GgufTokenizer:
    """Exact token counting backed by a lazily loaded ``vocab_only`` GGUF vocabulary."""

    def __init__(self, gguf_path: str | Path) -> None:
        self._path = Path(gguf_path)
        self._lock = threading.Lock()
        self._llama: Any = None

    @property
    def path(self) -> Path:
        return self._path

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

    def tokenize(self, text: str) -> list[int]:
        with self._lock:
            llama = self._load_locked()
            # add_bos maps to llama.cpp's add_special: BOS is added only when the model's
            # metadata declares it, matching how the backend tokenizes a raw prompt.
            tokens = llama.tokenize(text.encode("utf-8"), add_bos=True, special=True)
        return [int(token) for token in tokens]

    def _load_locked(self) -> Any:
        if self._llama is None:
            # Deferred import: importing pharos must never require llama-cpp's native DLLs.
            from llama_cpp import Llama

            _logger.info("loading tokenizer vocab (vocab_only) from %s", self._path)
            with _silenced_stderr():
                self._llama = Llama(model_path=str(self._path), vocab_only=True, verbose=False)
        return self._llama


@contextlib.contextmanager
def _silenced_stderr() -> Iterator[None]:
    """Mute fd 2 for the duration of the block.

    ``verbose=False`` only quiets llama-cpp's Python layer; the native library still writes
    loader notes straight to fd 2 (e.g. "n_ctx_seq (512) > n_ctx_train (0)"). The TUI owns the
    terminal, so that output lands on top of the dashboard and Python-level redirection cannot
    intercept it. Redirect the file descriptor itself, and restore it whatever happens.

    Degrades to a no-op if the platform will not give us a usable fd 2 (e.g. a pythonw-style
    host with no stderr at all) — never let logging cosmetics break token counting.
    """
    try:
        saved = os.dup(2)
    except (OSError, ValueError):
        yield
        return
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
