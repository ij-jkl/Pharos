"""File-based logging: the TUI owns the terminal, so everything else goes to a file.

``configure_logging()`` routes the root logger — pharos, uvicorn and httpx included — into a
rotating file and removes any stdout/stderr handlers, so nothing ever fights the TUI for the
screen. The proxy's per-request detail lands in the file, which means the on-screen event log
can stay terse and lossy (drop-oldest) without information being lost.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_MAX_BYTES = 5 * 1024 * 1024  # an always-on proxy must not grow an unbounded log
_BACKUPS = 2


def configure_logging(log_file: str | Path) -> Path:
    """Route all logging into ``log_file`` (rotating); returns the resolved path.

    Replaces every existing root handler: after this call nothing logs to stdout/stderr,
    including the stdlib's last-resort stderr handler (a root handler now exists).
    """
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return path.resolve()
