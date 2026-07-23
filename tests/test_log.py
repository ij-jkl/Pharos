"""Logging tests: everything routes to the file, nothing attaches to stdout/stderr."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from pharos.log import configure_logging


@pytest.fixture
def restore_root_handlers() -> Iterator[None]:
    """configure_logging replaces root handlers; put pytest's own back afterwards."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_logging_goes_to_file_not_terminal(tmp_path: Path, restore_root_handlers: None) -> None:
    log_file = tmp_path / "logs" / "pharos.log"  # parent dir is created on demand
    resolved = configure_logging(log_file)
    assert resolved == log_file.resolve()

    logging.getLogger("pharos.proxy").info("#1 completed status=200")
    logging.getLogger("uvicorn.error").warning("server-side detail")

    root = logging.getLogger()
    # The TUI owns the terminal: no stream handlers may remain on the root logger.
    assert not any(type(handler) is logging.StreamHandler for handler in root.handlers)
    for handler in root.handlers:
        handler.flush()
    content = log_file.read_text(encoding="utf-8")
    assert "#1 completed status=200" in content
    assert "server-side detail" in content
    assert "pharos.proxy" in content


def test_configure_logging_replaces_previous_handlers(
    tmp_path: Path, restore_root_handlers: None
) -> None:
    first = configure_logging(tmp_path / "a.log")
    second = configure_logging(tmp_path / "b.log")
    assert first != second
    root = logging.getLogger()
    assert len(root.handlers) == 1  # reconfiguring must not stack handlers
