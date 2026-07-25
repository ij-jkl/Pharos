"""Bounded LRU cache of token counts, keyed by tokenizer identity + content hash.

Keys are SHA-256 digests of the text rather than the text itself, so the cache never pins
large prompt strings in memory. Unchanged content is never re-tokenized. Dictionary access is
lock-guarded because counting runs in worker threads; the count callable itself runs outside
the lock (two threads racing on the same new text both count — benign — rather than
serializing all tokenization behind the cache).

The key is namespaced by an ``identity`` string naming the vocabulary that produced the count.
A token count is only meaningful for the tokenizer that produced it, so caching on text alone
would hand one model's count to another the moment more than one tokenizer is in play.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable

_DEFAULT_MAX_ENTRIES = 1024


class TokenCountCache:
    """LRU ``text -> token count`` cache with a fixed entry budget."""

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max = max_entries
        self._lock = threading.Lock()
        self._counts: OrderedDict[str, int] = OrderedDict()

    def __len__(self) -> int:
        return len(self._counts)

    def get_or_count(
        self, text: str, count: Callable[[str], int], *, identity: str = ""
    ) -> int:
        """Return the cached count for ``text`` under ``identity``, computing it on a miss.

        ``identity`` names the vocabulary doing the counting (in practice the GGUF path), so
        entries produced by different tokenizers can never satisfy each other's lookups.
        """
        key = f"{identity}\x00{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        with self._lock:
            cached = self._counts.get(key)
            if cached is not None:
                self._counts.move_to_end(key)
                return cached
        value = count(text)
        with self._lock:
            self._counts[key] = value
            self._counts.move_to_end(key)
            while len(self._counts) > self._max:
                self._counts.popitem(last=False)
        return value
