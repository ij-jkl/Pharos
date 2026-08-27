"""One way to put a small JSON store on disk, so a half-written one cannot exist.

Pharos keeps three of these -- observed token counts, what each model's chat template costs,
and what a context token costs in VRAM here. All three are written while other things are
happening: the proxy flushes observations mid-conversation, a run rewrites the template cost as
it finishes, and the profiler files a VRAM reading on a timer the dashboard drives. So all three
can be interrupted, and two of them can be written by two processes at once -- a `pharos run`
in one terminal and a dashboard in another are the ordinary case, not a stretch.

`path.write_text` truncates before it writes. A crash, a full disk or a second writer arriving
mid-call leaves a file that is valid UTF-8 and invalid JSON, and on Windows the second writer
can simply fail to open a file the first still holds. Every reader here already treats a corrupt
store as an empty one, so the cost is a silent loss of what was learned rather than a crash --
which is exactly the kind of failure that is never noticed and never fixed.

Writing a sibling temporary file and renaming it over the target makes the swap atomic on both
platforms: a reader sees the old file or the new one, never a truncation. The observation store
has always done this. This module is that code, in one place, for all three -- the same reason
`pharos.paths` exists.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

# Windows holds a brief lock on a file being read, and a reader here holds it for as long as
# it takes to parse a few KB. Five attempts over ~150ms clears anything that short; longer
# than that is a real lock, and losing one write to it beats blocking a run.
_REPLACE_ATTEMPTS = 5
_RETRY_INITIAL_DELAY_S = 0.01

_logger = logging.getLogger("pharos.store")


def write_json(path: Path, payload: Any, *, indent: int | None = None) -> bool:
    """Write ``payload`` to ``path`` atomically. Returns False instead of raising.

    None of these stores is worth failing a run over: each one makes Pharos better when it is
    there and none is needed for anything to happen. A caller that cannot write gets a log line
    and carries on, which is the rule every one of them already followed individually.

    The temporary file is a sibling rather than somewhere under /tmp, because a rename is only
    atomic within one filesystem and a store can be configured onto any drive. Its name carries
    the pid, so two processes writing the same store never share a scratch file.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=indent)
        tmp.write_text(text if indent is None else text + "\n", encoding="utf-8")
        _replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        _logger.warning("could not write %s: %s", path, exc)
        # A temporary left behind would accumulate one file per interrupted write.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        return False


def read_json(path: Path, default: Any = None) -> Any:
    """Read a store, returning ``default`` for absent, unreadable or corrupt content.

    The Windows lock cuts both ways: while a writer is mid-replace, a reader opening the
    destination is denied too. Caught as a plain OSError that becomes "the store is empty",
    which is how a run would quietly start uncorrected while another process happened to be
    writing. So a read gets the same short backoff a write does.

    A genuinely corrupt file is NOT retried -- it will parse the same way five times -- and is
    reported once, because the readers here are all allowed to carry on without their store.
    """
    delay = _RETRY_INITIAL_DELAY_S
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                _logger.warning("could not read %s: locked by another process", path)
                return default
            time.sleep(delay)
            delay *= 2
        except (OSError, ValueError) as exc:
            _logger.warning("could not read %s: %s", path, exc)
            return default
    return default


def _replace(tmp: Path, path: Path) -> None:
    """``os.replace``, retried while Windows says the destination is in use.

    POSIX renames over an open file happily. Windows refuses with WinError 5 (or 32) while any
    other handle is open on the destination -- including a reader that has the store open for
    the microsecond it takes to parse it. Measured under a reader thread looping over a store
    being rewritten, the replace failed on most attempts, and since a failed write here is only
    a log line the effect was a store that silently stopped learning whenever anything was
    reading it. The dashboard reads these on a timer while a run writes them, so that is the
    ordinary case rather than a stretch.

    A reader's handle lives for microseconds, so a short backoff clears it. Atomicity is not
    traded away for that: every attempt is still a whole-file swap, and giving up leaves the
    previous store intact rather than a damaged one.
    """
    delay = _RETRY_INITIAL_DELAY_S
    for _ in range(_REPLACE_ATTEMPTS - 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay)
            delay *= 2
    os.replace(tmp, path)  # the last attempt reports rather than swallows
