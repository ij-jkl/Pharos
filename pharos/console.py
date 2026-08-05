"""Make the standard streams safe for the characters Pharos actually prints.

Every rendered report carries non-ASCII: `≥` and `≤` on the floor and ceiling lines, `·` and
`—` as separators, `→` in the event log. On Windows an interactive terminal is usually UTF-8,
so this never shows up while developing — but a redirected or piped stream falls back to the
locale code page (cp1252 here), and Rich then dies mid-report:

    uv run pharos check "..." > report.txt
    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2265'

which cost the two things that make the CLI scriptable at all: `--json | jq` took the process
down (the JSON goes to stdout, but the human report still goes to stderr), and the documented
exit codes stopped meaning anything, because the crash replaced the verdict with 1.
"""

from __future__ import annotations

from typing import IO, Any


def force_utf8(*streams: IO[Any] | None) -> None:
    """Re-encode the given text streams as UTF-8 in place, best effort.

    ``errors="replace"`` rather than ``"strict"`` on purpose: a console that genuinely cannot
    represent a character should print a substitute and keep going. Losing `≥` off the floor
    line is a blemish; losing the report is a broken tool.

    Streams that cannot be reconfigured are skipped rather than raised on — a stream replaced
    by a test harness or a caller's own wrapper may not offer ``reconfigure`` at all, and none
    of them are worth failing a run over.
    """
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, AttributeError):
            # Detached, already closed, or a wrapper that only pretends to be a TextIOWrapper.
            continue
