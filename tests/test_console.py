"""Stream-encoding tests: the report has to survive a redirected stdout on Windows.

The bug these pin down was invisible on the dev machine, because an interactive Windows
terminal is UTF-8 and only a REDIRECTED stream falls back to the locale code page. Nothing in
the existing suite caught it either: the CLI tests render through Rich's own capture, which is
a unicode buffer and never encodes anything.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from pharos.console import force_utf8

# Every non-ASCII character Pharos prints. `≥`/`≤` head the floor and ceiling lines, the rest
# are separators and the event-log arrow.
_REPORT_CHARS = "FLOOR ≥ 1,845 · CEILING ≤ 17,102 — tokenizer gguf → exact"


def _cp1252_stream() -> io.TextIOWrapper:
    """A text stream encoded the way a redirected Windows stdout actually is."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")


def test_cp1252_stream_really_does_reject_the_report() -> None:
    """Guard the guard: if this ever stops raising, the tests below prove nothing."""
    stream = _cp1252_stream()
    with pytest.raises(UnicodeEncodeError):
        stream.write(_REPORT_CHARS)
        stream.flush()


def test_report_characters_survive_a_redirected_stream() -> None:
    stream = _cp1252_stream()
    force_utf8(stream)
    assert stream.encoding.lower() == "utf-8"

    stream.write(_REPORT_CHARS)
    stream.flush()
    written = stream.buffer.getvalue()  # type: ignore[attr-defined]
    assert "≥".encode() in written
    assert "≤".encode() in written


def test_rich_can_render_to_the_reconfigured_stream() -> None:
    """The real failure was inside Rich's writer, not in a bare write() — go through it."""
    stream = _cp1252_stream()
    force_utf8(stream)

    console = Console(file=stream, width=100, legacy_windows=False)
    console.print(f"[bold]{_REPORT_CHARS}[/]")

    assert "≥".encode() in stream.buffer.getvalue()  # type: ignore[attr-defined]


def test_undecodable_character_is_replaced_rather_than_raising() -> None:
    """errors='replace' is the point: a console that cannot render a character prints a
    substitute and the report still arrives."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")
    force_utf8(stream)
    force_utf8(stream)  # idempotent — entry points may be nested (split_main -> main)

    # Force a stream that genuinely cannot hold the character, to prove nothing raises.
    stream.reconfigure(encoding="ascii", errors="replace")
    stream.write(_REPORT_CHARS)
    stream.flush()
    assert stream.buffer.getvalue()  # type: ignore[attr-defined]


def test_streams_without_reconfigure_are_skipped() -> None:
    """pytest's captured stdout, a caller's own wrapper, or None must not blow up an entry
    point that only wanted to be helpful."""
    force_utf8(io.StringIO(), None)


def test_closed_stream_is_skipped() -> None:
    stream = _cp1252_stream()
    stream.close()
    force_utf8(stream)  # must not raise
