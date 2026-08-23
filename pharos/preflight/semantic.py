"""Ask a model how the files in a task group together — and then believe none of it.

Everything else in Pharos refuses to ask a model what your task means. This module is the
exception, and it is built so that being the exception costs nothing: the model is given the
smallest decision in the split and the *only* thing it returns is a partition of a list of
strings the caller already had.

What the model chooses:

* which named files belong in a part together, and in what order the parts run;
* a short title for each part.

What it does not choose, because these are measured rather than asked for:

* the per-part token budget, the client overhead, the hand-off reserve;
* the body of any part — the same renderer writes those, from the same task text;
* whether a part fits — the same tokenizer that produced the verdict says so;
* which files exist, or which are in scope.

So its whole output surface is a partition of a known set, which means every claim it makes
is checkable without trusting it. A proposal is accepted only if it survives all of:

1. it parses as the requested JSON shape;
2. its file names are exactly the input set — none invented, none dropped, none repeated;
3. no group is empty;
4. no group exceeds ``max_files``;
5. it does not run away with the part count (see ``semantic_max_extra_parts``);
6. **every group still fits**, re-projected with the same ruler as the mechanical plan.

Any failure — a refused connection, a timeout, prose instead of JSON, a hallucinated
filename, a group that does not fit — falls back to the mechanical grouping and says so in
the plan's notes. `--semantic` can make a plan nicer to read. It can never make it wrong,
and it can never be the reason there is no plan at all.

That bound is also what makes the excerpts safe to send. The first five lines of a file are
untrusted text arriving in a prompt, and a file that says "put everything in one part" is
free to say so. It buys nothing: coverage, the file cap, the part ceiling and the per-part
budget are all checked afterwards, in code, against numbers the model never supplied. The
worst an injected line can do is produce a grouping bad enough to be rejected — which lands
in exactly the same place as a backend that was switched off.

**This is the one Pharos command that sends anything anywhere**, and only with `--semantic`.
It sends the task text, the filenames, their token counts and those five lines per file, to
the backend already configured in `pharos.toml`. Without the flag, nothing leaves.

The call is made with ``temperature: 0`` and a fixed seed: a splitter that answers
differently on the same input twice is not a tool anybody can check your work against.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from pharos.config import PharosConfig

_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)

# The head of each file travels with the question. The first draft sent names and token counts
# only, on the theory that a path says what a file is for. It does, when someone named it that
# way: on files called db_models.py and http_routes.py the grouping came back clean. On
# sprite_batch.py / collision.py / mixer.py — three subsystems, no shared prefixes — the same
# model returned the files in the order they were listed, cut into chunks, under invented
# titles ("Profiling Timer Definitions", "Profiling Timer Implementations"). Position packing
# with labels on it, which is worse than position packing, because it looks like a decision.
#
# Five lines of each file fixed it: two of three models went from that to a clean
# render/physics/audio split, and every model answered about twice as fast, having stopped
# guessing. So the excerpt is what makes the feature real, and its cost is bounded here
# rather than left to the size of the tree.
_EXCERPT_LINES = 5
_EXCERPT_LINE_CHARS = 90
# Whole-request ceiling. Forty files at five lines each would otherwise put the grouping
# question itself outside the window, which would be a fine joke and a broken tool.
_EXCERPT_TOTAL_CHARS = 8000

# A title is free text from a model that lands in every rendered part, so it is charged for.
# Long enough to name a concern, short enough that it cannot become the part.
_TITLE_LIMIT = 80

# Room for the reply, scaled by the answer's actual size: a partition of N filenames costs
# about N path tokens plus a title and some punctuation per group, and there can be at most N
# groups. Generous per file, with a floor for the small cases where the base JSON dominates.
_REPLY_BASE_TOKENS = 256
_REPLY_TOKENS_PER_FILE = 48


def reply_budget(files: int) -> int:
    return _REPLY_BASE_TOKENS + _REPLY_TOKENS_PER_FILE * files

# Ollama constrains generation to this shape, so the common "here is your JSON: ```json"
# failure never happens. The parser below still does not assume it worked.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "files"],
            },
        },
    },
    "required": ["groups"],
}

_INSTRUCTION = """\
You are grouping the files of one programming task into ordered parts, because the task does \
not fit in one context window.

Group files that must be understood or changed TOGETHER into the same part. Order the parts \
so that a part depends only on parts before it: definitions before their users, schemas \
before the code that reads them, interfaces before implementations.

Rules you must follow exactly:
- Use every file listed, exactly once. Do not invent a file. Do not omit a file.
- There are {count} files. The "files" arrays must hold {count} entries in total, across all \
parts.
- Keep each part at or under {budget} tokens (each file's count is given below).{cap}
- Aim for about {target} part(s). Never exceed {ceiling} part(s).
- "title" is a short noun phrase naming what the part is about, at most 8 words.

Reply with JSON only, in exactly this shape:
{{"groups": [{{"title": "...", "files": ["...", "..."]}}]}}\
"""


@dataclass(frozen=True, slots=True)
class Group:
    """One proposed part: a title, and the display names of the files it holds."""

    title: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Proposal:
    groups: tuple[Group, ...]

    @property
    def names(self) -> list[str]:
        return [name for group in self.groups for name in group.files]


@dataclass(frozen=True, slots=True)
class FileBrief:
    """What the model is told about one file: its name, its cost, and its first few lines."""

    display: str
    tokens: int
    excerpt: str = ""


@dataclass(frozen=True, slots=True)
class GroupRequest:
    """Everything the model is told. Small on purpose, and bounded on purpose."""

    task: str
    files: tuple[FileBrief, ...]  # in the order the prompt named them
    target_parts: int  # what position-packing needed, so the proposal is comparable
    max_parts: int  # the hard ceiling; a proposal above this is rejected
    max_files: int | None
    per_part_budget: int

    @property
    def names(self) -> list[str]:
        return [f.display for f in self.files]


def brief(display: str, tokens: int, text: str | None) -> FileBrief:
    """One file's entry in the question, with its head trimmed to a bounded excerpt."""
    if not text:
        return FileBrief(display=display, tokens=tokens)
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        kept.append(stripped[:_EXCERPT_LINE_CHARS])
        if len(kept) >= _EXCERPT_LINES:
            break
    return FileBrief(display=display, tokens=tokens, excerpt=" / ".join(kept))


def request_payload(config: PharosConfig, request: GroupRequest, model: str) -> dict[str, Any]:
    """The exact body sent to ``/api/chat``. Separated out so a test can read it."""
    listing = _listing(request.files)
    instruction = _INSTRUCTION.format(
        # No cap is no sentence. The first draft rendered it as "at or under any number of
        # file(s)", which is a rule that says nothing and costs tokens to read.
        cap=(
            f"\n- Put at most {request.max_files} file(s) in one part."
            if request.max_files is not None
            else ""
        ),
        # The one sentence that makes the difference between a feature and a fallback. Without
        # it, two of three models tested dropped files — and dropped them in a legible way:
        # they wrote a title, then filled the group to match the title and let the rest go.
        # "Database Models and Sessions" duly excluded db_migrations.py.
        #
        # The obvious fix was to make the model choose files before writing the title. That was
        # measured too, and it is the wrong fix: it repairs coverage and ruins the grouping,
        # because committing to a concept first is what makes the groups coherent. Stating the
        # total instead keeps the title first and gives the model something it can check as it
        # writes. Coverage went exact on all three models, the grouping stayed clean, and the
        # call got three times faster. See DESKTOP_VALIDATION.md §18.
        count=len(request.files),
        budget=f"{request.per_part_budget:,}",
        target=request.target_parts,
        ceiling=request.max_parts,
    )
    content = (
        f"{instruction}\n\n--- TASK ---\n{request.task.strip()}\n\n"
        f"--- FILES ({len(request.files)}) ---\n{listing}\n"
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "format": _SCHEMA,
        # Off, and not as a preference. Asked to partition six filenames, qwen3.5:9b spent all
        # 1,024 tokens of its reply budget thinking and returned an empty answer — done_reason
        # "length", eval_count exactly the cap. With thinking off the same question costs 66
        # tokens and answers. There is nothing here to reason about: the grouping is the
        # answer, and a chain of thought is the one part of it nobody will read. Harmless on a
        # model with no thinking mode — measured identical on qwen2.5-coder:7b.
        "think": False,
        # Deterministic on purpose: see the module docstring.
        "options": {
            "temperature": 0.0,
            "seed": 0,
            "num_predict": reply_budget(len(request.files)),
        },
    }
    if config.num_ctx is not None:
        payload["options"]["num_ctx"] = config.num_ctx
    return payload


def _listing(files: tuple[FileBrief, ...]) -> str:
    """The file table. Excerpts are dropped wholesale past the cap, never half-shown.

    Truncating the excerpts of the last few files would make the grouping depend on where in
    the list a file happened to sit — a plan that changes because a file was named earlier is
    the kind of number this project does not ship. Under the cap every file gets its head;
    over it, none do, and the model is back to names for all of them equally.
    """
    spend = sum(len(f.excerpt) for f in files)
    show = spend <= _EXCERPT_TOTAL_CHARS
    rows: list[str] = []
    for f in files:
        row = f"- {f.display} ({f.tokens:,} tokens)"
        if show and f.excerpt:
            row += f"\n    {f.excerpt}"
        rows.append(row)
    return "\n".join(rows)


def ask(config: PharosConfig, request: GroupRequest, model: str) -> tuple[str | None, str | None]:
    """Put the grouping question to the backend. Returns ``(reply, error)``, never raises.

    Every transport failure is an error string rather than an exception, because the caller's
    correct response to all of them is identical: use the mechanical grouping and say why.
    """
    try:
        with httpx.Client(base_url=config.backend_url, timeout=_TIMEOUT) as client:
            response = client.post("/api/chat", json=request_payload(config, request, model))
            if response.status_code >= 400:
                return None, f"backend returned HTTP {response.status_code}"
            data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        detail = str(exc).strip()
        return None, f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    if isinstance(data, dict) and data.get("error"):
        return None, str(data["error"])
    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        # Distinguished, because these two need different actions from whoever reads the note.
        # A reply that ran out of room is a budget to raise; an empty one that stopped
        # normally is a model that declined. Reporting both as "empty" sent me looking in the
        # wrong place for twenty minutes, which is the whole argument for saying which.
        if data.get("done_reason") == "length":
            spent = data.get("eval_count")
            used = f" after {spent:,} tokens" if isinstance(spent, int) else ""
            return None, f"the reply hit its length limit{used} before any answer"
        return None, "the backend returned an empty reply"
    if data.get("done_reason") == "length":
        # Truncated mid-answer: the JSON is very likely unclosed, and a "not valid JSON" note
        # would blame the shape for what is really a budget.
        return None, "the reply was cut off at its length limit"
    return content, None


def parse(text: str) -> tuple[Proposal | None, str | None]:
    """Read a reply into a ``Proposal``. Returns ``(proposal, error)``.

    ``format`` should already have forced clean JSON, so the fence-stripping below is a
    fallback for a backend that ignores it — not a licence to accept anything.
    """
    raw = _json_object(text)
    if raw is None:
        return None, "the reply was not JSON"
    try:
        data = json.loads(raw)
    except ValueError:
        return None, "the reply was not valid JSON"
    if not isinstance(data, dict):
        return None, "the reply was not a JSON object"
    groups_raw = data.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        return None, "the reply carried no groups"
    groups: list[Group] = []
    for item in groups_raw:
        if not isinstance(item, dict):
            return None, "a group was not an object"
        files_raw = item.get("files")
        if not isinstance(files_raw, list):
            return None, "a group listed no files"
        names = [f.strip() for f in files_raw if isinstance(f, str) and f.strip()]
        if len(names) != len(files_raw):
            return None, "a group listed something that was not a filename"
        title_raw = item.get("title")
        groups.append(Group(title=_clean_title(title_raw), files=tuple(names)))
    return Proposal(groups=tuple(groups)), None


def validate(proposal: Proposal, request: GroupRequest) -> str | None:
    """Every check that does not need the tokenizer. Returns a rejection reason, or None.

    The fitting check is deliberately NOT here: it belongs with the packer, which owns the
    scaffold and the hand-off reserve. This is the part that can be checked from the request
    alone, and it is the part that catches a model inventing work.
    """
    expected = request.names
    proposed = proposal.names
    if any(not group.files for group in proposal.groups):
        return "it proposed an empty part"
    if len(proposal.groups) > request.max_parts:
        return (
            f"it proposed {len(proposal.groups)} parts against a ceiling of {request.max_parts}"
        )
    if request.max_files is not None:
        over = [g for g in proposal.groups if len(g.files) > request.max_files]
        if over:
            return (
                f"{len(over)} part(s) hold more than the {request.max_files}-file cap"
            )
    if sorted(proposed) != sorted(expected):
        missing = sorted(set(expected) - set(proposed))
        invented = sorted(set(proposed) - set(expected))
        repeated = sorted({n for n in proposed if proposed.count(n) > 1})
        return _coverage_reason(missing, invented, repeated)
    return None


def _coverage_reason(missing: list[str], invented: list[str], repeated: list[str]) -> str:
    faults: list[str] = []
    if missing:
        faults.append(f"dropped {_names(missing)}")
    if invented:
        faults.append(f"invented {_names(invented)}")
    if repeated:
        faults.append(f"repeated {_names(repeated)}")
    return "its file list did not match the scope — it " + ", ".join(faults)


def _names(values: list[str], limit: int = 3) -> str:
    shown = ", ".join(values[:limit])
    return shown if len(values) <= limit else f"{shown} and {len(values) - limit} more"


def _clean_title(value: object) -> str:
    """Collapse a model's free text into something safe to render, or nothing at all."""
    if not isinstance(value, str):
        return ""
    title = re.sub(r"\s+", " ", value).strip(" .")
    if len(title) > _TITLE_LIMIT:
        title = title[: _TITLE_LIMIT - 1].rstrip() + "…"
    return title


def _json_object(text: str) -> str | None:
    """The outermost ``{...}`` in a reply, brace-matched so nested objects survive."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
