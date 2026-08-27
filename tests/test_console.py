"""Stream-encoding tests: the report has to survive a redirected stdout on Windows.

The bug these pin down was invisible on the dev machine, because an interactive Windows
terminal is UTF-8 and only a REDIRECTED stream falls back to the locale code page. Nothing in
the existing suite caught it either: the CLI tests render through Rich's own capture, which is
a unicode buffer and never encodes anything.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from pharos.console import force_utf8
from pharos.paths import shorten_path

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


# --- long paths stay readable and copyable ------------------------------------------------------


def test_shorten_path_keeps_the_end_that_identifies_the_run() -> None:
    """The bug: Rich hard-wrapped a long workspace path at the console edge, splitting it
    through the middle of a uuid across three lines. It could not be copied out."""
    deep = (
        r"C:\Users\<you>\AppData\Local\Temp\pharos"
        r"\pharos-workspace\a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d\scratchpad\proj"
    )
    out = shorten_path(deep, 76)

    assert len(out) <= 76
    assert out.endswith(r"scratchpad\proj"), "the leaf is what a reader is looking for"
    assert "\u2026" in out, "something has to say the middle was dropped"
    assert "\n" not in out


def test_shorten_path_leaves_a_path_that_already_fits_alone() -> None:
    assert shorten_path("/home/u/proj", 76) == "/home/u/proj"


def test_shorten_path_uses_the_home_shorthand() -> None:
    """Shorter and clearer than the absolute form, and it is what a shell would print."""
    inside = Path.home() / "Desktop" / "Pharos"
    assert shorten_path(inside, 76).startswith("~")


def test_shorten_path_truncates_a_single_huge_component() -> None:
    """No separator to cut on, so the only honest answer is a marked truncation."""
    out = shorten_path("C:\\a\\" + "d" * 200, 40)
    assert len(out) == 40
    assert out.endswith("\u2026")


def test_shorten_path_never_returns_more_than_asked_for() -> None:
    deep = "/" + "/".join(f"component{i}" for i in range(40))
    for width in (10, 20, 40, 80, 160):
        assert len(shorten_path(deep, width)) <= max(width, 8)
