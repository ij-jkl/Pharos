# Pharos

[![CI](https://github.com/ij-jkl/Pharos/actions/workflows/ci.yml/badge.svg)](https://github.com/ij-jkl/Pharos/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A transparent, context-aware proxy and live terminal dashboard that sits between a local
coding agent and a local LLM backend (Ollama first) — so you can *see* your context and VRAM
budget in real time and never hit a silent context overflow or OOM again.

**Status: v0.4 "Do"** — the proxy stays observe-only and a pure passthrough (Pharos never
mutates a request or a response). Alongside it, three tools outside the request path: what a
prompt will cost before you paste it (`pharos check`), how to cut it up when it will not fit
(`pharos split`), and carrying those parts out against your local model (`pharos run`).

<!-- Capture docs/pharos-dashboard.png (see docs/CAPTURE.md), then uncomment:
![The Pharos dashboard: context mismatch banner, context and VRAM gauges, and the request event log](docs/pharos-dashboard.png)
-->

## The problem it solves

Local models advertise big context windows, but the window that is *actually loaded* is
whatever `num_ctx` your backend picked — often a fraction of the advertised maximum. Nothing
tells you when your agent's conversation silently overflows that window, and nothing tells
you how close the KV cache is to eating your remaining VRAM. Pharos surfaces both, loudly:

- **⚠ CONTEXT MISMATCH** — the headline feature. If the model advertises 262,144 tokens but
  only 32,768 are loaded, a persistent banner says so the whole time you run.
- **Context gauge** — how many tokens the last request actually used, against the usable
  budget (loaded ctx minus a response reserve), with 80%/90% warning thresholds.
- **VRAM gauge** — measured free VRAM plus an *estimate* of how many more context tokens fit
  before OOM.
- **Event log** — one line per request: input tokens, output tokens, tok/s, status.

### Honest counting

A number rendered without its provenance is a confident wrong number, so every count carries
its label:

- `1,720 (exact)` — reconciled against the backend's own `prompt_eval_count`, or a raw
  prompt / token-array the tokenizer saw verbatim.
- `~1,500 (estimate · gguf)` — counted with the model's real GGUF tokenizer, but the backend
  applies its chat template server-side, so the true prompt is slightly larger.
- `~5 (heuristic chars/4)` — no GGUF tokenizer available; a rough character heuristic.

Estimates are reconciled against `prompt_eval_count` from the response whenever the backend
reports it. On `/v1` streaming, OpenAI-compatible responses only carry `usage` if *your
client* asked for it (`stream_options.include_usage`); Pharos never injects that — the
estimate simply stands.

## Install and run

Clone it, then run one script — the same one, every time:

```bash
git clone https://github.com/ij-jkl/Pharos.git
cd Pharos

./start.sh        # macOS / Linux
.\start.ps1       # Windows
```

The first run installs [uv](https://docs.astral.sh/uv/) if it is missing, fetches Python 3.12,
installs the dependencies, creates `pharos.toml`, and checks for a backend. Every run after
that detects all of it is already there, skips straight past, and opens the prompt:

```
pharos> Refactor everything in `pharos/proxy/` following `README.md`

FLOOR   ≥ 5,007 tokens
CEILING ≤ 14,615 tokens if every named directory is read in full
```

Type a prompt to pre-flight it, `s <prompt>` to cut one that does not fit into parts that do,
`d` for the live dashboard, `q` to quit.

Setup is re-detected rather than remembered, so a half-finished first run is repaired by a
second one, and a `git pull` that moves dependencies re-syncs on its own. `--reinstall` forces
the setup steps, `--setup-only` stops before the prompt (`-Reinstall` / `-SetupOnly` on
PowerShell).

Nothing is installed system-wide except uv, and nothing outside the clone is touched. Pharos
runs fine on a machine with no NVIDIA GPU and no backend — everything degrades to N/A /
UNREACHABLE, and the pre-flight plans against a stand-in window instead of a real one. An
NVIDIA GPU and a local [Ollama](https://ollama.com) make it useful.

## Configure

```bash
cp pharos.toml.example pharos.toml
```

`pharos.toml` is machine-specific and gitignored. The keys that matter:

| Key | Default | What it does |
|---|---|---|
| `backend_url` | `http://localhost:11434` | The Ollama instance Pharos forwards to. |
| `model` | *(auto)* | Preferred model; otherwise the loaded model from `/api/ps`. |
| `gguf_path` | *(auto)* | Explicit path to the model's GGUF for exact token counting. If unset, Pharos tries Ollama's blob store; if that fails, counting degrades to a labeled heuristic. Setting it explicitly is the robust option. |
| `response_reserve` | `1024` | Tokens reserved for the reply when computing the usable input budget. |
| `warn_threshold` / `alert_threshold` | `0.80` / `0.90` | Context-usage fractions that turn the gauge yellow / red. |
| `kv_mib_per_1k` | `32` | KV-cache VRAM estimate, MiB per 1K context tokens. Drives the *estimated* token headroom; measured free VRAM always wins. |
| `proxy_host` / `proxy_port` | `127.0.0.1` / `11435` | Where Pharos listens. |
| `log_file` | `pharos.log` | Rotating file for request/error detail (the TUI owns the terminal). |

## Run

```bash
uv run pharos
```

Starts the proxy and the dashboard in one process (`q` quits). For a one-shot environment
report — GPU, model, advertised vs loaded context, budget — without the TUI:

```bash
uv run pharos-profile
```

## Pre-flight a prompt (v0.2 "Warn")

Before pasting a big prompt into your coding agent, ask whether it can possibly fit:

```bash
uv run pharos check "Refactor `src/auth/login.py` and `src/auth/session.py` per docs/plan.md"
uv run pharos check --file prompt.txt
```

The prompt can also arrive on stdin — pipe it, or run `pharos check` with no argument, paste,
and end with Ctrl-Z Enter (Windows) / Ctrl-D (Unix). The same is true of `pharos split`.

`pharos check` extracts the files you explicitly named, tokenizes them exactly, adds the
client overhead learned from traffic previously observed through the proxy (system prompt,
tool catalogue — the part that made your 50-token prompt a 20K request), and compares the
total against the live usable budget. The number is a **floor**: exact for what you named,
silent about whatever the agent decides to read on its own. Ambiguous, missing, binary and
directory references are listed rather than silently dropped. Exit codes: `0` fits, `1`
exceeds, `2` no verdict (backend unreachable or no model loaded) — scriptable.

File references are resolved against `target_folder` from `pharos.toml`. The check runs
entirely locally and sends nothing to the backend.

### Floor and ceiling

Name a **directory** and you get a second number. The floor stays what the prompt guarantees;
the directory's text files are counted separately and reported as a ceiling:

```
pharos/preflight (directory, if fully read)   +15,257   estimate — 5 text files

FLOOR   ≥ 1,845 tokens
CEILING ≤ 17,102 tokens if every named directory is read in full
```

Adding those tokens to the floor would turn a lower bound into a guess, so they stay out of
it — but a floor that fits while the ceiling does not is called out explicitly, because the
floor is not the number that has to survive contact with the agent. Expansion prunes
vcs/venv/cache directories, whitelists text extensions (it will not read a `.safetensors` to
find out it is not source) and stops at 300 files, saying so when it does.

## Split a prompt that does not fit (v0.3 "Divide")

When the verdict is EXCEEDS, the answer is not "write a smaller prompt" — it is to cut the
work into pieces that each fit:

```bash
uv run pharos split "Refactor `src/auth/login.py` and `src/auth/session.py` per docs/plan.md"
uv run pharos split --file prompt.txt --out parts/     # writes parts/part-01.txt …
uv run pharos check "…" --split                        # same plan, after the report
```

The split is mechanical and local — no model is asked what your task *means*, nothing leaves
the machine — and it takes one of two shapes depending on what actually overflows:

- **scope split** — the task text fits but the files it names do not. Every part repeats your
  task verbatim and narrows the scope to a subset of the files; files too large for one part
  are cut into line ranges (`forward.py (lines 428-684 of 684)`), and the slices of one file
  only ever move forward through the parts — no part asks you to hold two disjoint windows of
  the same file. A named directory contributes its files here too, each labelled with where it
  came from, which is what makes "refactor everything in `src/`" splittable at all. Each part
  names the files it defers, so the agent knows what it is *not* to open, and asks for a
  ≤10-line hand-off to paste above the next part; room for that hand-off is held back from
  every part and counted into the ones that will carry it.
- **text split** — the pasted text itself is too big (a log, a spec, a transcript). Parts are
  ordered segments cut on paragraph, then line boundaries; the first parts ask only for an
  acknowledgement and the last one asks for the work.

Every part carries a projected cost measured with the same tokenizer that produced the
verdict — client overhead + the part as written + the content it scopes — and each is packed
against the warn threshold, not the hard limit. Two honesty rules stay in force: the
projection is a floor that holds only while the agent stays inside the part's scope, and a
plan that cannot work is refused rather than faked. If the client overhead alone fills the
window, or a single line is wider than a part, `pharos split` says so instead of shipping
parts that will fail.

`--target N` plans against N tokens per part without probing the backend — useful offline, or
to plan for a window you have not loaded yet. Exit codes: `0` a plan whose every part fits (or
nothing to split), `1` no plan or a part still over, `2` no budget to plan against.

## Carry the task out (v0.4 "Do")

`pharos run` executes the plan instead of printing it. It pre-flights the task, divides it
with the same splitter, then runs each part as its own conversation against your local model
— fresh context each time, seeded only with the previous part's hand-off.

```bash
pharos run "Refactor the controllers in `backend/Controllers/` to a consistent response shape"
pharos run --dry-run "..."     # plan only; changes nothing
pharos run --no-split "..."    # one undivided conversation, for comparison
```

This is the one part of Pharos that **writes files**, so it will not start without a way back:
in a git repository it requires a clean tree and puts the run on its own `pharos-run/…` branch;
in a plain folder it copies every original into `.pharos/undo-<timestamp>/` before the first
write. Review with `git diff`, or copy the snapshot back.

Three properties make this compatible with everything above:

- **The proxy is untouched.** The runner is a *client* of it, like Continue or Cursor. Its
  traffic is observed on the way past, never rewritten.
- **Overhead is counted, not learned.** Pharos wrote this client, so it counts its own system
  prompt and tool catalogue exactly instead of estimating them from observed traffic.
- **Nothing is trimmed.** A file too large for the remaining window is *refused*, not
  truncated, and the model is told the size and the room left. A part that reaches its ceiling
  stops and hands off rather than silently dropping its earliest context.

Scope is enforced in the tool layer, not merely requested in the prompt: a part told to open
three files physically cannot open a fourth. That is what turns the splitter's projection from
a hope into a bound.

Two limits are in play and only one is measurable. The window is exact. How many files a model
will work through in one sitting before it stops calling tools and starts describing them is a
property of the model, not the hardware — `max_files_per_part` (default 2) bounds it.

That default is measured, not guessed. On the same 13-file task against `qwen2.5-coder:14b`, a
part completes about **1.5–2.0 files** and then believes itself finished, whatever it was
given. Window pressure is never the cause: peak usage sat at 42–51% of the ceiling throughout.
Nor is persuasion the fix — stating the target up front, naming the outstanding files, and
asking again all failed to push a part past roughly two. So the fix is arithmetic. At four
files per part that task covered **46%, 62%, 46%**; at two, **92%**. Raise it for a model that
finishes more, and watch the scorecard's continuity line, since more parts means more
hand-offs.

**It does not make the model good.** Pharos proves the task fits and that every part ran; it
does not check that the code is right. Small local models still invent types, miss files, and
occasionally answer in prose without editing anything — the report says `wrote nothing` when
that happens, per part, rather than letting the totals absorb it. `git diff` is your reviewer.

### Did it work? The scorecard

"Every part completed" is not success. A part completes by replying without calling a tool —
exactly what a model does when it has read its files and *described* the change instead of
making it. So every run ends with five numbers, none of which require asking a model anything:

```
Scorecard
  coverage     14 of 20 scoped files written (70%)
                 untouched: Repositories/NoteRepository.cs
  continuity   3 of 3 hand-offs produced; largest 214 of 500 reserved
  headroom     peak used 58% of a part's ceiling
  drift        our estimate ran 1.27x the backend's count  (above the real prompt: conservative)
  convergence  1 part(s) needed a nudge, 0 stopped early
```

- **Coverage** is the headline: of the files the plan assigned, how many were actually
  written. A run that touches three of twenty did not succeed, whatever its parts reported.
- **Continuity** measures the thread between parts. Every part starts from an empty
  conversation, so the hand-off is the *only* thing carrying context forward — this checks one
  was produced wherever there was a next part, and that it fitted the reserve held back for
  it. A hand-off that overran means the next part began with a truncated thread, which is a
  `handoff_reserve` problem rather than a model failure. A hand-off can also be present and
  carry nothing: a real run produced three of three whose largest was **six tokens** against a
  500-token reserve while coverage sat at 31%, and the first version of this metric called
  that continuity. A part that *changed files* and then reported six tokens has dropped the
  thread; a part that changed nothing is entitled to be brief. They are counted separately.
- **Revisits** are the fingerprint of a part that lost the thread: reaching for a file another
  part already owned. The scope layer refuses it, so it is recorded rather than damaging.
  Deliberately *not* the same as a path the model invented — one real run tried to write to a
  `Data/` folder that has never existed, which is confusion about the repository, not about
  what has already been done. Those are counted separately as `wandering`.
- **Headroom** is the highest fraction of any part's ceiling actually used. Near 100% means
  the next slightly larger file breaks the run.
- **Drift** is Pharos's own projection against the backend's `prompt_eval_count`, paired per
  request — the number this project is least entitled to hide, and the one it has been most
  wrong about. Put to a live backend on hand-built conversations it lands at **0.87–0.98x**:
  tens of tokens *below* what the backend reports, which is the chat template's own
  scaffolding and is invisible from here. Across a real run the per-request ratio ranged
  0.96–2.57x, and the high end is **not explained** — two plausible causes were tested
  directly and both measured *under* 1.0. It is left recorded as unexplained rather than given
  a story. Only the low side threatens anything: over-counting wastes room, under-counting
  means a ceiling was enforced against a number below the real prompt. So the scorecard
  reports the worst shortfall in **tokens** against the safety margin that absorbs it —
  measured at 60 tokens against a 256-token margin.

`--json` emits the same thing for a script or a CI step, and **the exit code follows coverage,
not survival**: 0 only when the run wrote everything it was given and no part failed. A run
whose parts all said "done" while three files were never touched exits 1, because exiting 0
would flatter precisely the failure this tool exists to expose.

None of this says the code is *correct*. That is outside what Pharos claims; `git diff` is the
reviewer.

## Ambiguity, notebooks, and JSON

Three smaller things that decide whether the number is trustworthy:

**Ambiguous references are never guessed.** `utils.py` matching both `src/` and `tests/` is
reported, not resolved by coin-flip. Answer it and re-run:

```bash
uv run pharos check "fix utils.py" --resolve utils.py=tests/utils.py
uv run pharos check "fix utils.py" --resolve utils.py=*   # I meant all of them
uv run pharos check "fix utils.py" --pick     # choose from a list, interactively
```

`--pick` needs a terminal to ask on; when the prompt itself came from stdin, or output is
piped, it says so and defers to `--resolve` rather than blocking on a prompt nobody can answer.
A `--resolve` for a reference the prompt never makes is reported too — it is a typo in the
flag, not a silent no-op.

**Notebooks are counted as cell sources, outputs excluded.** A `.ipynb` is JSON, and a
notebook with two plots carries tens of thousands of tokens of base64 that an agent never
sees; counting the raw file overshoots by an order of magnitude, and an over-count is not a
floor. The treatment is printed next to the number. The same reader feeds the splitter, so a
notebook that is too big for one part is refused rather than cut — a line range into a
notebook names a document that does not exist on disk.

**`--json` makes it scriptable.** Both commands take `--json` and emit the full report — every
count with its provenance, the budget, the verdict, and (for `split`) the plan with each part's
projection and body. Human output moves to stderr, so stdout is the payload alone:

```bash
uv run pharos split --file prompt.txt --json | jq '.plan.parts[] | {index, projected_tokens}'
```

## Point your client at it

Use Pharos as your only endpoint; it forwards everything and watches the four inference
routes (`/v1/chat/completions`, `/v1/completions`, `/api/chat`, `/api/generate`). All other
`/api/*` and `/v1/*` paths pass through untouched.

- OpenAI-compatible clients (Copilot, Cursor, most agent frameworks):
  `base_url = http://127.0.0.1:11435/v1`
- Ollama-native clients: `http://127.0.0.1:11435` (e.g. `/api/chat`)

Passthrough is byte-for-byte in both directions: request bodies are forwarded verbatim
(Pharos parses only a copy for counting) and responses are re-streamed raw, without
re-encoding or re-chunking.

## What Pharos does NOT do (yet)

- **No request mutation, ever.** No fields added or removed, no prompt rewriting, no
  `stream_options` injection. The proxy remains a pure observer. `pharos check` and
  `pharos split` are advisory; `pharos run` writes files but does so as an ordinary client of
  the proxy, outside the request path — the constraint applies to what Pharos does to *other
  people's* traffic, and it has not moved.
- **No prediction of which files an agent will read** — `pharos check` counts what you named:
  files exactly, into the floor; directories in full, into a separate ceiling. What the agent
  decides to open on its own is in neither number. A split part is a floor on the same terms:
  it holds while the agent respects the scope block it was given.
- **No history compaction** and no automatic trimming when you approach the budget — it
  warns; it does not intervene.
- **No semantic decomposition.** `pharos split` cuts by scope and by position, never by
  meaning, and no model is ever asked what your task means. `pharos run` executes those same
  mechanically-derived parts — it divides by files and by position, never by intent.
- **No verification of the work.** `pharos run` proves a task fit and ran; it does not build,
  test or review what the model wrote.
- **No change auditing / filesystem watching.**

Those belong to later tiers. The contract is simple: what your agent sends is what the
backend receives, and every number you see is honestly labeled — exact, estimate, heuristic
or floor. The one file Pharos writes from observed traffic (`pharos_observations.json`)
contains token counts only, never text.

## Development

```bash
uv run ruff check .
uv run mypy pharos
uv run pytest
```

Tokenizer tests against a real GGUF auto-skip unless a model file is present under
`tests/models/` (gitignored). See `DESKTOP_VALIDATION.md` for the checklist of assumptions
to confirm against a live GPU + Ollama machine.

`tests/test_end_to_end.py` runs the whole loop against a mocked backend — a coding-agent-shaped
request through the proxy, the observation it records, the overhead the pre-flight learns from
it, the split that overhead forces, and then each generated part fed back through the checker.
That last step is the one that matters: the splitter's projection and the checker's floor come
from different code, and a plan whose parts do not re-check as fitting is fiction. It also
pins down the scope contract by measuring it — a part read literally, deferred filenames and
all, costs more than its projection, which is exactly why the part says "do not open these".

## Author

Built by **Isaac Jordan** — [LinkedIn](https://www.linkedin.com/in/isaac-jordan-464563215/)

Licensed under the [MIT License](LICENSE).
