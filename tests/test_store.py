"""A half-written store must not be possible.

All three stores are written while other things are happening, and two of them can be written by
two processes at once -- a `pharos run` in one terminal and a dashboard in another. Every reader
treats a corrupt store as an empty one, so the failure mode is a silent loss of what was learned:
exactly the kind that is never noticed and never fixed.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pharos.store import write_json


def test_a_store_is_replaced_whole(tmp_path: Path) -> None:
    store = tmp_path / "s.json"
    assert write_json(store, {"a": 1}) is True
    assert write_json(store, {"a": 2, "b": 3}) is True
    assert json.loads(store.read_text(encoding="utf-8")) == {"a": 2, "b": 3}


def test_no_temporary_is_left_behind(tmp_path: Path) -> None:
    """One per interrupted write would accumulate forever in the user's project."""
    store = tmp_path / "s.json"
    write_json(store, {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["s.json"]


def test_a_reader_never_sees_a_truncation(tmp_path: Path) -> None:
    """The point of the rename. A reader racing a writer gets the old file or the new one.

    `write_text` truncates first, so the same test against it sees `{` or `` and fails to parse.
    """
    store = tmp_path / "s.json"
    write_json(store, {"n": 0})
    seen: list[object] = []
    errors: list[Exception] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                seen.append(json.loads(store.read_text(encoding="utf-8")))
            except (FileNotFoundError, PermissionError):
                # Windows denies a read while a replace is in flight. Transient, and what
                # `read_json` retries; never a partial parse, which is the failure under test.
                continue
            except Exception as exc:  # noqa: BLE001 - a parse error is the failure under test
                errors.append(exc)
                return

    watcher = threading.Thread(target=reader)
    watcher.start()
    try:
        for n in range(200):
            write_json(store, {"n": n, "padding": "x" * 4000})
    finally:
        stop.set()
        watcher.join(timeout=5)

    assert not errors, f"a reader saw a partial file: {errors[0]!r}"
    assert seen, "the reader never got a look in"
    assert all(isinstance(item, dict) and "n" in item for item in seen)


def test_two_writers_do_not_corrupt_the_file(tmp_path: Path) -> None:
    """Distinct temporary names per process, so neither writer clobbers the other's scratch."""
    store = tmp_path / "s.json"
    done = threading.Barrier(3)

    def writer(tag: str) -> None:
        for n in range(100):
            write_json(store, {"who": tag, "n": n})
        done.wait(timeout=10)

    for tag in ("a", "b"):
        threading.Thread(target=writer, args=(tag,)).start()
    done.wait(timeout=10)

    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["who"] in {"a", "b"}
    assert [p.name for p in tmp_path.iterdir()] == ["s.json"]


def test_an_unwritable_path_is_reported_not_raised(tmp_path: Path) -> None:
    """None of these stores is worth failing a run over."""
    target = tmp_path / "dir"
    target.mkdir()
    assert write_json(target, {"a": 1}) is False  # a directory, not a file


def test_unserialisable_content_is_reported_not_raised(tmp_path: Path) -> None:
    store = tmp_path / "s.json"
    assert write_json(store, {"bad": object()}) is False
    assert not store.exists()
    assert list(tmp_path.iterdir()) == []
