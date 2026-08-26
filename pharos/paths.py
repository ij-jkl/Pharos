"""One spelling for a display path, so a comparison cannot miss on punctuation alone.

The same path arrives here in three spellings in a single run. The splitter builds
``PartFile.display`` with the platform separator (a backslash on Windows), ``Workspace.display``
returns posix, and a model writes whichever it likes — usually posix, because that is what code
looks like, whatever it was shown.

Comparing those raw has now caused the same bug twice, in two layers, with two different
symptoms. In the tool dispatcher it refused every part access to its own files, and looked
exactly like a model ignoring its scope. In the semantic grouper it reported every file as
simultaneously *dropped* and *invented*, so a proposal could never pass on Windows outside a
flat directory. Both were one `replace` away.

So there is one function, here, and everything that compares displayed paths goes through it.
"""

from __future__ import annotations

from pathlib import Path


def normalise_display(display: str) -> str:
    """Canonical form for comparing a displayed path: posix separators, no leading ``./``."""
    out = display.strip().replace("\\", "/")
    while out.startswith("./"):
        out = out[2:]
    return out


def resolve_display(name: str, known: dict[str, str]) -> str | None:
    """Map a path as somebody else spelled it onto the one spelling we already hold.

    ``known`` maps ``normalise_display(display) -> display``. Exact match first; a
    case-insensitive match is accepted only when exactly one candidate matches, because
    Windows will hand back ``Src/Store_Models.py`` for a file recorded as ``src/store_models.py``
    and a case-sensitive filesystem can legitimately hold both.
    """
    key = normalise_display(name)
    if key in known:
        return known[key]
    folded = key.casefold()
    hits = [value for candidate, value in known.items() if candidate.casefold() == folded]
    return hits[0] if len(hits) == 1 else None


def display_path(path: Path, root: Path) -> str:
    """A path as it should be shown: relative to the workspace when it is inside it.

    Written twice, in check.py and in the preflight CLI, and the two copies had already
    drifted -- one resolved the root before comparing and the other did not, so the same file
    could be reported by name in the verdict and by absolute path in the plan beside it. This
    module exists because this project has paid for two copies of a path comparison once
    already.

    Both are tried, unresolved first: a resolved root can differ from the one the user typed
    (a symlink, a junction, an 8.3 name on Windows), and a relative answer is never worse than
    an absolute one for something only a human reads.
    """
    for base in (root, root.resolve()):
        try:
            return str(path.relative_to(base))
        except (OSError, ValueError):
            continue
    return str(path)
