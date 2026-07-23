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

UNVERIFIED (confirm on the desktop): only the negative case is proven — ``add_special`` does
not force a BOS onto a model declaring add_bos_token=false (Qwen3). That it actually adds one
for a model declaring true (Llama family) has not been tested; until it is, raw-exact counts
on BOS-adding models may be off by one.
"""

from __future__ import annotations

import logging
import threading
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
            self._llama = Llama(model_path=str(self._path), vocab_only=True, verbose=False)
        return self._llama
