# Pharos

A transparent, context-aware proxy and live terminal dashboard that sits between a local
coding agent and a local LLM backend (Ollama first) — so you can *see* your context and VRAM
budget in real time and never hit a silent context overflow or OOM again.

**Status: v0.1 "Observe"** — observe-only and a pure passthrough. Pharos never mutates a
request or a response; it watches the traffic and tells you the truth about it.

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

## Install

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is pinned and fetched automatically.

```bash
git clone https://github.com/ij-jkl/Pharos.git
cd Pharos
uv sync
```

Runs fine on a machine with no NVIDIA GPU and no backend (everything degrades to N/A /
UNREACHABLE); an NVIDIA GPU and a local [Ollama](https://ollama.com) make it useful.

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

## What v0.1 does NOT do

- **No request mutation, ever.** No fields added or removed, no prompt rewriting, no
  `stream_options` injection.
- **No history compaction** and no automatic trimming when you approach the budget — it
  warns; it does not intervene.
- **No prompt decomposition / task splitting.**
- **No change auditing / filesystem watching.**

Those belong to later tiers, where Pharos stops being a pure observer. v0.1's contract is
simple: what your agent sends is what the backend receives, and what you see on the
dashboard is honestly labeled.

## Development

```bash
uv run ruff check .
uv run mypy pharos
uv run pytest
```

Tokenizer tests against a real GGUF auto-skip unless a model file is present under
`tests/models/` (gitignored). See `DESKTOP_VALIDATION.md` for the checklist of assumptions
to confirm against a live GPU + Ollama machine.
