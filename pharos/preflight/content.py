"""Read a file as the text that actually reaches a model, not as the bytes on disk.

For nearly every file those are the same thing. Jupyter notebooks are the exception that
matters: a `.ipynb` is JSON whose real content is the `source` of each cell, wrapped in
metadata, execution counts, and outputs that can include base64 images. Counting the raw
JSON of a notebook with two plots reports tens of thousands of tokens for a file an agent
sees as forty lines of Python — and a "floor" that overshoots by 10x is not a floor.

So notebooks are counted as their cell sources, outputs excluded. That direction is the safe
one: it can only under-count, which is what a lower bound is allowed to do, and the report
says which treatment produced the number. A notebook that will not parse falls back to raw
text and says so rather than guessing.

One reader, used by both the checker and the splitter — a file counted one way and then cut
up another way would put the two out of agreement about the same bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Countable:
    """The text a file contributes, plus how it was derived when that is not obvious."""

    text: str
    note: str | None = None  # shown verbatim in the report; None means "read as plain text"


class BinaryFile(Exception):
    """The file is not decodable text; it has no token count and must not be given one."""


def read_countable(path: Path) -> Countable:
    """Read ``path`` as countable text. Raises BinaryFile / OSError for the caller to report."""
    try:
        raw = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BinaryFile(str(path)) from exc
    if path.suffix.lower() == ".ipynb":
        return _notebook(raw)
    return Countable(text=raw)


def _notebook(raw: str) -> Countable:
    try:
        data = json.loads(raw)
        cells = data["cells"]
        if not isinstance(cells, list):
            raise TypeError("cells is not a list")
    except (ValueError, KeyError, TypeError):
        return Countable(
            text=raw, note="notebook — unparseable, counted as raw JSON (an over-count)"
        )

    sources: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        source = cell.get("source")
        if isinstance(source, list):
            sources.append("".join(str(line) for line in source))
        elif isinstance(source, str):
            sources.append(source)
    return Countable(
        text="\n".join(sources),
        note=f"notebook — {len(cells)} cells, outputs excluded",
    )
