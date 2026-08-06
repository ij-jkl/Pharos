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
import ctypes
import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

_logger = logging.getLogger("pharos.tokenizer")

# Strong reference to the installed ctypes callback. A CFUNCTYPE object that gets garbage
# collected while llama.cpp still holds its address is a hard crash, not an exception, so this
# must outlive every tokenizer. Doubles as the "already installed" flag.
_log_sink: Any = None
_log_sink_lock = threading.Lock()


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
            import llama_cpp
            from llama_cpp import Llama

            _install_log_sink(llama_cpp)
            _logger.info("loading tokenizer vocab (vocab_only) from %s", self._path)
            with _silenced_stderr():
                self._llama = Llama(model_path=str(self._path), vocab_only=True, verbose=False)
        return self._llama


def _install_log_sink(llama_cpp: Any) -> None:
    """Take over llama.cpp's log callback so its notes go to our log file, not the terminal.

    llama.cpp emits through a C callback, and importing ``llama_cpp`` installs one that does
    ``print(text, file=sys.stderr)``. Two problems follow, both of which users saw:

    * The line lands on the terminal wherever it is emitted — under the `pharos check` report,
      or on top of the TUI. ``_silenced_stderr`` does not stop it, and the reason is why this
      only ever showed up for people running interactively: when stderr is a real Windows
      console, ``sys.stderr`` is a ``_WindowsConsoleIO`` that writes to a console HANDLE it
      cached at startup, not to whatever file descriptor 2 currently points at. Redirecting
      the descriptor therefore misses it. Pipe the same command into a file and the message
      disappears — the write follows the descriptor there, which is exactly what made this
      look unreproducible.
    * Anything that print raises escapes into C, where no caller can catch it. That is the
      "Exception ignored on calling ctypes callback function" pair — the callback also fires
      while the interpreter is tearing down and ``sys.stderr`` is already gone, and the
      unraisable hook then fails for the same reason and reports itself.

    Replacing the callback fixes both: nothing reaches ``sys.stderr``, and the body cannot
    raise. The lines are not discarded — they go to the rotating log at DEBUG, which is where
    a loader note belongs. Installed once per process, before the first model load.

    The message itself, "n_ctx_seq (512) > n_ctx_train (0) -- possible training context
    overflow", is harmless: ``Llama.__init__`` builds a context unconditionally, even under
    ``vocab_only=True`` (llama.py, ``internals.LlamaContext(...)``), and then compares its
    default 512 against training metadata that a vocab-only load never read, so n_ctx_train is
    0. Nothing is generated through that context — tokenize() only needs the vocabulary — so
    there is no overflow to have. The warning is an artefact of the load mode.
    """
    global _log_sink
    with _log_sink_lock:
        if _log_sink is not None:
            return

        def sink(level: int, text: bytes, user_data: Any) -> None:
            # try/except rather than contextlib.suppress: this body has to work while the
            # interpreter is tearing modules down, and it should reach for as little as
            # possible on the way. Broad on purpose — see the docstring.
            try:  # noqa: SIM105
                _logger.debug("llama.cpp: %s", text.decode("utf-8", "replace").rstrip())
            except Exception:  # noqa: BLE001 - raising here is unrecoverable, not reportable
                pass

        callback = llama_cpp.llama_log_callback(sink)
        # Assign before handing the address to C, so the reference exists the moment it can
        # be called, and so a failing llama_log_set does not leave a half-installed sink.
        _log_sink = callback
        llama_cpp.llama_log_set(callback, ctypes.c_void_p(0))


@contextlib.contextmanager
def _silenced_stderr() -> Iterator[None]:
    """Mute fd 2 for the duration of the block.

    ``verbose=False`` only quiets llama-cpp's Python layer; parts of the native stack (the ggml
    backend registry among them) write straight to fd 2, which Python-level redirection cannot
    intercept. The TUI owns the terminal, so that output would land on top of the dashboard.
    Redirect the file descriptor itself, and restore it whatever happens.

    This covers only writes made while the constructor runs. Everything routed through
    llama.cpp's log callback — which includes the loader notes, and which fires outside this
    window as well — is handled by ``_install_log_sink``. Both are needed.

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
