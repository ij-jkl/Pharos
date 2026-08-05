"""Extract explicitly-referenced file paths from free-form prompt text and resolve them.

Extraction is deliberately conservative — v0.2 only handles files the user NAMES; predicting
what an agent will read on its own is v1.0. The bias is against false positives: a stray
"v0.2" or "e.g." must not surface as a missing file. Three candidate tiers:

1. backtick-quoted spans,
2. quoted spans (double quotes may contain spaces — paths with spaces usually arrive quoted;
   single-quoted spans must be a single token, or every apostrophe would spawn candidates),
3. bare whitespace-separated tokens, accepted only when they LOOK like a path: contain a
   separator, carry a known code/text extension, or are a dot-leading name like `.gitignore`.

Resolution against the root (``target_folder``, else cwd): absolute or direct relative hit
first, then a basename search of the tree (pruned of vcs/venv/cache dirs). One match resolves
(flagged "found by search"); several are AMBIGUOUS — counted as a guess would poison an exact
floor; none is MISSING. Nothing path-like is ever silently dropped: ambiguous and missing
references are surfaced in the verdict. Candidates that never looked path-like (a quoted
word like 'hello') are dropped silently on failure instead — they were probably prose.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Extensions that make a bare token path-like. Modest on purpose: every entry here widens the
# false-positive surface ("3.14" must never be a candidate, so no bare numeric suffixes).
_TEXT_EXTENSIONS = frozenset({
    "bash", "bat", "c", "cc", "cfg", "cjs", "cmd", "conf", "cpp", "cs", "css", "csv",
    "dockerfile", "env", "go", "h", "hpp", "html", "ini", "ipynb", "java", "js", "json",
    "jsonl", "jsx", "kt", "lock", "log", "md", "mjs", "php", "ps1", "py", "rb", "rs", "rst",
    "scala", "scss", "sh", "sql", "svelte", "swift", "tcss", "toml", "ts", "tsx", "txt",
    "vue", "xml", "yaml", "yml", "zig",
})
# Binary formats are still REFERENCES — they must resolve and then be surfaced as
# "skipped, binary" by the checker. Silently ignoring a named file breaks the contract
# that nothing path-like is ever dropped without a word. Directory expansion, which nobody
# named file-by-file, uses the text set instead: it may skip quietly, and says how many.
_BINARY_EXTENSIONS = frozenset({
    "7z", "bin", "bmp", "dll", "exe", "gguf", "gif", "gz", "ico", "jpeg", "jpg", "mp3",
    "mp4", "onnx", "pdf", "png", "pt", "pth", "safetensors", "so", "tar", "ttf", "wav",
    "webp", "woff", "woff2", "zip",
})
_KNOWN_EXTENSIONS = _TEXT_EXTENSIONS | _BINARY_EXTENSIONS

# A directory reference can name a tree of any size, and expanding it means reading and
# tokenizing every file in it. Past this many files the expansion stops and says so — a
# truncated, labeled answer beats a five-minute pre-flight.
DIRECTORY_FILE_CAP = 300

# The `--resolve REF=*` answer: count every candidate rather than choose between them.
ALL_CANDIDATES = "*"

_IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build", ".idea", ".vscode",
})

_BACKTICK = re.compile(r"`([^`\n]+)`")
_DOUBLE_QUOTED = re.compile(r'"([^"\n]+)"')
_SINGLE_QUOTED = re.compile(r"'(\S+)'")
# Trailing may include "." (a filename ending a sentence); leading must NOT — stripping the
# dot off ".gitignore" would demote a dotfile to prose.
_STRIP_TRAILING = ".,;:!?()[]{}<>*\"'"
_STRIP_LEADING = ",;:!?()[]{}<>*\"'"


@dataclass(frozen=True, slots=True)
class Candidate:
    """A potential file reference: the raw text plus whether it inherently looks like a path.

    ``path_like`` decides failure behaviour: path-like candidates that do not resolve are
    REPORTED (the user plainly meant a file); non-path-like ones (quoted prose) vanish quietly.
    """

    raw: str
    path_like: bool


@dataclass(frozen=True, slots=True)
class ResolvedFile:
    raw: str  # as written in the prompt
    path: Path  # resolved absolute path
    found_by_search: bool  # True when located by tree search rather than direct resolution


@dataclass(frozen=True, slots=True)
class Extraction:
    """Everything the prompt referenced, sorted by fate. Nothing path-like is dropped."""

    files: list[ResolvedFile] = field(default_factory=list)
    directories: list[ResolvedFile] = field(default_factory=list)  # named but never expanded
    ambiguous: dict[str, list[Path]] = field(default_factory=dict)  # raw -> the candidates
    missing: list[str] = field(default_factory=list)  # path-like but nowhere to be found


@dataclass(frozen=True, slots=True)
class DirectoryContents:
    """What a named directory holds, as far as a token count is concerned."""

    files: list[Path]  # text files, sorted, capped at DIRECTORY_FILE_CAP
    truncated: bool  # the cap was hit: there are more files than these
    skipped_non_text: int  # files passed over for having no text extension


def extract_references(
    prompt: str, root: Path, overrides: dict[str, str] | None = None
) -> Extraction:
    """Extract candidates from ``prompt`` and resolve them against ``root``.

    ``overrides`` maps a raw reference to the path the user chose for it — the answer to an
    AMBIGUOUS reference, which Pharos will never guess at on its own.
    """
    return _resolve(_candidates(prompt), root, overrides or {})


def expand_directory(directory: Path, *, cap: int = DIRECTORY_FILE_CAP) -> DirectoryContents:
    """List the text files under ``directory``, pruned of vcs/venv/cache dirs.

    Sorted, so a plan built from a directory is the same plan tomorrow. Extension-whitelisted
    rather than decode-and-see: a pre-flight must not read a 4 GB weights file to discover it
    is not source.
    """
    found: list[Path] = []
    skipped = 0
    truncated = False
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIRS)
        for name in sorted(filenames):
            if not _has_text_extension(name):
                skipped += 1
                continue
            if len(found) >= cap:
                truncated = True
                break
            found.append(Path(dirpath) / name)
        if truncated:
            break
    return DirectoryContents(files=found, truncated=truncated, skipped_non_text=skipped)


def _has_text_extension(name: str) -> bool:
    _, dot, ext = name.rpartition(".")
    return bool(dot) and ext.lower() in _TEXT_EXTENSIONS


def _candidates(prompt: str) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []

    def add(raw: str, *, quoted: bool) -> None:
        raw = raw.strip().lstrip(_STRIP_LEADING).rstrip(_STRIP_TRAILING)
        if not raw or "\n" in raw or raw in seen:
            return
        path_like = _is_path_like(raw)
        # Quoted multi-word spans that don't look like paths are prose, not references.
        if not path_like and (not quoted or " " in raw):
            return
        seen.add(raw)
        out.append(Candidate(raw=raw, path_like=path_like))

    stripped = prompt
    for pattern in (_BACKTICK, _DOUBLE_QUOTED, _SINGLE_QUOTED):
        for match in pattern.finditer(stripped):
            add(match.group(1), quoted=True)
        stripped = pattern.sub(" ", stripped)  # bare pass must not re-see quoted spans
    for token in stripped.split():
        add(token, quoted=False)
    return out


def _is_path_like(raw: str) -> bool:
    if " " in raw:
        # A path with spaces is only recognizable via its separator or extension.
        return ("/" in raw or "\\" in raw) and _has_known_extension(raw)
    if "/" in raw or "\\" in raw:
        return True
    if raw.startswith(".") and len(raw) > 1 and not raw.startswith(".."):
        return True  # dotfiles: .gitignore, .env
    return _has_known_extension(raw)


def _has_known_extension(raw: str) -> bool:
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    _, dot, ext = name.rpartition(".")
    return bool(dot) and ext.lower() in _KNOWN_EXTENSIONS


def _resolve(candidates: list[Candidate], root: Path, overrides: dict[str, str]) -> Extraction:
    result = Extraction()
    claimed: set[Path] = set()  # real paths already counted — dedupe across spellings
    index: dict[str, list[Path]] | None = None  # basename -> paths, built lazily once
    unused = dict(overrides)

    for candidate in candidates:
        chosen = unused.pop(candidate.raw, None)
        # "*" means "I meant all of them" — the other honest answer to an ambiguous reference,
        # and the only one that does not require typing out forty paths.
        take_all = chosen == ALL_CANDIDATES
        if chosen is not None and not take_all:
            picked = _resolve_direct(chosen, root)
            if picked is not None:
                _classify(result, claimed, candidate.raw, picked, found_by_search=False)
            else:
                result.missing.append(f"{candidate.raw} (resolved to {chosen}, which is absent)")
            continue
        direct = _resolve_direct(candidate.raw, root)
        if direct is not None:
            _classify(result, claimed, candidate.raw, direct, found_by_search=False)
            continue
        if index is None:
            index = _index_tree(root)
        matches = _search(index, candidate.raw)
        if len(matches) == 1:
            _classify(result, claimed, candidate.raw, matches[0], found_by_search=True)
        elif len(matches) > 1:
            if take_all:
                for match in sorted(matches):
                    _classify(result, claimed, candidate.raw, match, found_by_search=True)
            else:
                result.ambiguous[candidate.raw] = sorted(matches)
        elif candidate.path_like:
            result.missing.append(candidate.raw)
        # Non-path-like candidates that resolve nowhere were prose after all: dropped quietly.

    # An override for a reference the prompt does not contain is a typo in the flag, not a
    # silent no-op: the user believes they resolved something, and they have not.
    for raw in unused:
        result.missing.append(f"{raw} (given to --resolve, but the prompt never names it)")
    return result


def _classify(
    result: Extraction, claimed: set[Path], raw: str, path: Path, *, found_by_search: bool
) -> None:
    try:
        real = path.resolve()
    except OSError:
        real = path
    if real in claimed:
        return
    claimed.add(real)
    entry = ResolvedFile(raw=raw, path=real, found_by_search=found_by_search)
    if path.is_dir():
        result.directories.append(entry)
    else:
        result.files.append(entry)


def _resolve_direct(raw: str, root: Path) -> Path | None:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    joined = root / candidate
    return joined if joined.exists() else None


def _index_tree(root: Path) -> dict[str, list[Path]]:
    """Basename -> paths for every file under ``root``, pruned of vcs/venv/cache dirs.

    Case-insensitive keys: Windows filesystems are, and a prompt that says ``readme.md`` for
    ``README.md`` plainly means that file.
    """
    index: dict[str, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for name in filenames:
            index.setdefault(name.casefold(), []).append(Path(dirpath) / name)
    return index


def _search(index: dict[str, list[Path]], raw: str) -> list[Path]:
    normalized = raw.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    matches = index.get(basename.casefold(), [])
    if "/" in normalized:
        # The candidate carried directories: only paths whose tail matches all of them count,
        # so "auth/login.py" disambiguates between src/auth/login.py and tests/auth/login.py.
        suffix = tuple(part.casefold() for part in normalized.split("/") if part)
        matches = [p for p in matches if _tail_matches(p, suffix)]
    return matches


def _tail_matches(path: Path, suffix: tuple[str, ...]) -> bool:
    parts = tuple(part.casefold() for part in path.parts)
    return len(parts) >= len(suffix) and parts[-len(suffix) :] == suffix
