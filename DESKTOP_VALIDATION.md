# Desktop validation checklist

Every assumption that could not be verified on the dev laptop (no NVIDIA GPU, no Ollama),
in the order to work through them on the desktop (RTX 3060 12GB, Ollama, qwen3.5-9b-heretic).
Each item: what to run, what to expect, what a failure means. Commands are for Git Bash;
they work in PowerShell too unless noted.

Legend: ☐ open · ☑ pass · ☒ fail (note what you saw)

**Run completed 2026-07-24** on RTX 3060 12GB (driver 580.88), Ollama 0.31.1,
`qwen3.5-9b-heretic:latest` (arch qwen35, 9.0B, Q8_0). Confirmed values are recorded inline
below. Two code fixes came out of it (items 1–2); one open finding remains (item 6).

---

## ☑ 0. Fresh-machine setup

**Run:**
```bash
git clone https://github.com/ij-jkl/Pharos.git && cd Pharos
uv sync
cp pharos.toml.example pharos.toml     # then set model = "qwen3.5-9b-heretic"
uv run ruff check . && uv run mypy pharos && uv run pytest
```

**Expect:** `uv sync` resolves without needing a C++ toolchain (the llama-cpp CPU wheel comes
from its dedicated index). Tests: everything passes, with exactly **5 skips** — the
real-GGUF tokenizer tests skip because `tests/models/` is empty on a fresh clone. Nothing
else should assume laptop state.

**CONFIRMED 2026-07-24:** `uv sync` pulled CPython 3.12.13 and 50 packages including a
prebuilt `llama-cpp-python==0.3.34` wheel — no compile step. Suite: **78 passed, 5 skipped**
(all five `tests/models/` tokenizer skips), ruff and mypy --strict clean. Note `uv` itself is
not bundled: install it first (`irm https://astral.sh/uv/install.ps1 | iex` on Windows).

**If it fails:** a compile attempt during sync means the CPU wheel index was not used
(check `[tool.uv.sources]` / `[[tool.uv.index]]` in `pyproject.toml`); more than 5 skips or
any error means a hidden laptop assumption — note which test.

---

## ☑ 1. `/api/ps` loaded-context field name

The defensive key chain in `pharos/profiler/backend.py` tries
`context_length → context → num_ctx → details.context_length`. It has only ever run against
mocked payloads.

**CONFIRMED 2026-07-24 (Ollama 0.31.1):** the loaded window is a **top-level
`context_length`** on the /api/ps model entry — the FIRST key in the chain. Raw entry:

```json
{"name": "qwen3.5-9b-heretic:latest", "model": "qwen3.5-9b-heretic:latest",
 "size": 9688129207, "size_vram": 9688129207,
 "details": {"family": "qwen35", "parameter_size": "9.0B", "quantization_level": "Q8_0"},
 "expires_at": "2026-07-24T21:44:34.1257096-03:00",
 "context_length": 32768}
```

`details` carries **no** context key here. Two changes followed:

* `details.context_length` was **removed** from the fallback chain — in the sibling /api/tags
  payload that same key holds the ADVERTISED max (262144), so honouring it would report
  advertised-as-loaded, give ratio 1.0 and silently suppress the banner.
* **Model names are tag-normalized** when matching. /api/ps reports `<name>:latest`, but the
  natural thing to put in `pharos.toml` is the untagged name. Exact string matching made that
  common case fall through to `loaded_ctx = None` — profile fully populated from /api/show,
  but `loaded N/A` and no banner. See `_normalize_model_name` in `backend.py`.

**Run:**
```bash
ollama run qwen3.5-9b-heretic "hi"        # ensure the model is loaded
curl -s http://localhost:11434/api/ps | python -m json.tool
uv run pharos-profile
```

**Expect:** the raw `/api/ps` JSON shows the loaded context under one of the four keys, and
`pharos-profile` prints `loaded <N>` (non-N/A) in the Context row.

**If it fails:** profile shows `loaded N/A` → the real key is outside the chain. Read the
raw JSON, add the actual key name to `_LOADED_CTX_KEYS` (or the nested path) in
`backend.py`, and drop a note in the docstring that it is now confirmed.

---

## ☑ 2. `/api/show` advertised max context

**Run:**
```bash
curl -s http://localhost:11434/api/show -d '{"model": "qwen3.5-9b-heretic"}' \
  | python -c "import json,sys; d=json.load(sys.stdin); print({k:v for k,v in d['model_info'].items() if 'context' in k or k=='general.architecture'})"
```

**Expect:** `general.architecture` (e.g. `qwen3`) plus `<arch>.context_length` with the
advertised maximum (Qwen3.5-9B should advertise 262144 or similar). `pharos-profile` shows
the same number as `advertised`.

**CONFIRMED 2026-07-24:** `general.architecture` is **`"qwen35"`** (not `qwen3`), and the only
`*.context_length` key in `model_info` (33 keys total) is **`qwen35.context_length` = 262144**.
The arch-prefixed lookup in `_extract_advertised_ctx` resolves it on the first try. /api/show
accepts the bare name and the `:latest` name identically, which is why a tag mismatch leaves
`advertised` populated while `loaded` goes None — the failure looks like success.

Real mismatch on this machine: **32,768 loaded vs 262,144 advertised = 12.5%**.

**If it fails:** no `*.context_length` key → adjust `_extract_advertised_ctx` in
`backend.py` to the real key layout.

---

## ☑ 3. The mismatch banner firing for real

Only ever exercised against mocked payloads and stubbed profiles.

**Run:** `uv run pharos` with the model loaded at Ollama's default `num_ctx` (much smaller
than the advertised max).

**Expect:** the dark-red persistent banner:
`⚠ CONTEXT MISMATCH — advertised <adv> · loaded <loaded> (<pct>% of capacity)`, where
`pct == loaded/adv`, staying visible across profile refreshes (every 5s), not just at
startup. Then reload the model with a larger window, e.g.
```bash
curl -s http://localhost:11434/api/chat -d '{"model":"qwen3.5-9b-heretic","messages":[{"role":"user","content":"hi"}],"options":{"num_ctx":65536},"stream":false}' > /dev/null
```
and confirm the banner updates to the new loaded value within a refresh.

**CONFIRMED 2026-07-24** (real `PharosApp` driven headlessly via Textual's test pilot against
the live backend, with the untagged `model = "qwen3.5-9b-heretic"` in `pharos.toml`):

```
[t+  0.2s] ⚠ CONTEXT MISMATCH — advertised 262,144 · loaded 32,768 (12.5% of capacity) — raise num_ctx to use the full window
   ... unchanged across 14s, spanning three 5s refresh cycles ...
[t+ 41.0s] reloaded backend with num_ctx=65536
[t+ 42.0s] ⚠ CONTEXT MISMATCH — advertised 262,144 · loaded 65,536 (25.0% of capacity) — raise num_ctx to use the full window
```

Banner persisted rather than only rendering at startup, and picked up the new window within
one refresh. `ctx_mismatch_ratio` 0.125 → 0.25.

**If it fails:** banner absent while profile shows both numbers → `detect_ctx_mismatch` or
the TUI banner logic; banner shows stale numbers → the 5s profile refresh isn't updating.

---

## ☑ 4. NVML on the RTX 3060 + headroom sanity

**Run:** `uv run pharos-profile`, then cross-check `nvidia-smi`.

**Expect:** GPU row `NVIDIA GeForce RTX 3060 · free X / 12,288 MiB` matching nvidia-smi
(±100 MiB), source nvml (no fallback). VRAM headroom row shows
`≈<free/32*1000> more ctx tokens (estimate)` — e.g. 8,000 MiB free → ≈250,000. Sane means:
headroom scales with free VRAM and drops visibly after the model loads.

**CONFIRMED 2026-07-24:** `source='nvml'` (no fallback). Free VRAM tracked nvidia-smi to
within **8 MiB** loaded and **1 MiB** idle — well inside the ±100 tolerance.

Measured in both load states, because the honesty of the headroom number depends on it:

| state | free VRAM | `vram_headroom_tokens` | honest? |
|---|---|---|---|
| model resident @ 65,536 ctx | 532 MiB | 16,625 | yes — genuine remaining capacity |
| no model resident | 10,743 MiB | 335,718 | **no — fantasy** |

Unloaded, the figure is computed against VRAM that must first hold ~10,091 MiB of weights
before a single context token exists; it overstates by ~20× and exceeds the model's own
advertised max (262,144). It prints directly beneath an honest `Usable budget N/A — no loaded
context detected`, so the row is confidently wrong exactly where the row above it admits
ignorance. Fix belongs in a later tier: suppress or caveat headroom when `loaded_ctx is None`.

**Measured `kv_mib_per_1k` for this model:** `size_vram` was 9,688,129,207 B at 32,768 ctx and
10,580,981,185 B at 65,536 ctx → 892,851,978 B for 32,768 extra tokens = **≈26 MiB per 1K
ctx**, against the configured estimate of 32. The config value is conservative by ~23%.

**If it fails:** GPU N/A → NVML binding vs driver problem (check `nvidia-smi` works at all;
then the `_probe_nvml` fallback order). Absurd headroom → revisit `kv_mib_per_1k` for this
model in `pharos.toml` (it is a config estimate, not code).

---

## ☑ 5. GGUF resolution for the real model (both paths)

**Run (autodetect):** leave `gguf_path` unset in `pharos.toml`, keep `model` set. Start
`uv run pharos`, send any request through the proxy, then check:
- the TUI input line says `(estimate · gguf)` — **not** `heuristic`;
- `pharos.log` contains `loading tokenizer vocab (vocab_only) from ...ollama...blobs...`.

**Run (override):** set `gguf_path` to the blob path the log printed (or find it:
`ls ~/.ollama/models/blobs` and match the manifest digest), restart, same expectations.

**Expect:** both paths produce gguf-labeled counts.

**CONFIRMED 2026-07-24 — both paths.** Autodetect resolved
`C:\Users\<you>\.ollama\models\blobs\sha256-0a132b761289e2205a8c73bde3a328f285ba6ba5e96a55c3841daf6d8809c1b0`
(9,086 MiB), and the log carried the expected line:

```
INFO pharos.tokenizer: loading tokenizer vocab (vocab_only) from C:\Users\<you>\.ollama\models\blobs\sha256-0a132b...
```

Counts came back `source='gguf'` (not heuristic) on both the autodetect and the explicit
`gguf_path` override. Ollama's manifest layout matched `_find_manifest`'s assumption.

Cosmetic note: `llama.cpp` writes `n_ctx_seq (512) > n_ctx_train (0)` to stderr on vocab_only
load. Harmless here, but it is native-level output the TUI cannot capture, so it can scribble
on the dashboard on first count.

**If it fails:** autodetect falls to heuristic → Ollama's manifest layout differs from
`manifests/<registry>/<ns>/<name>/<tag>`; inspect `~/.ollama/models/manifests` and fix
`_find_manifest` / `_model_layer_digest` in `pharos/tokenizer/resolver.py`. Override path
failing means the file check or vocab load — see the log for the exception.

---

## ☑☒ 6. C8 positive case: BOS on a Llama-family model

Locally verified: `add_bos=True` does **not** force a BOS on Qwen3 (declares
`add_bos_token: false`). Unverified: that it **does** add one when the model declares true.
`pharos/tokenizer/gguf.py` carries an UNVERIFIED note until this passes.

**Run:**
```bash
ollama pull llama3.2:1b
uv run python - <<'EOF'
from pharos.tokenizer.resolver import resolve_from_store
from llama_cpp import Llama
path = resolve_from_store("llama3.2:1b")
print("blob:", path)
l = Llama(model_path=str(path), vocab_only=True, verbose=False)
meta = {k: v for k, v in (l.metadata or {}).items() if "bos" in k.lower()}
print("metadata:", meta)
a = l.tokenize(b"hello world", add_bos=True, special=True)
b = l.tokenize(b"hello world", add_bos=False, special=True)
print("add_bos=True :", len(a), a[:3])
print("add_bos=False:", len(b), b[:3])
print("token_bos id :", l.token_bos())
EOF
```

**Expect:** metadata shows `add_bos_token: true` (or absent with a llama arch default of
true), `len(a) == len(b) + 1`, and `a[0] == token_bos()`.

**Cross-check against Ollama:** send `/api/generate` with `"raw": true` and the same prompt
through Pharos; the TUI's `(exact · gguf)` input count must equal the response's
`prompt_eval_count` exactly.

**BOS behaviour CONFIRMED 2026-07-24 — option (a) is correct for this wheel.**
`llama-cpp-python==0.3.34`, `llama3.2:1b`:

```
metadata      : {'tokenizer.ggml.bos_token_id': '128000'}   # add_bos_token absent -> llama default true
add_bos=True  : 3  [128000, 15339, 1917]
add_bos=False : 2  [15339, 1917]
token_bos id  : 128000
len(a) == len(b) + 1 : True      a[0] == token_bos() : True
```

Qwen control unchanged: `add_bos=True` and `add_bos=False` both give 2 tokens — no BOS forced.
So the UNVERIFIED paragraph in `gguf.py` is now settled on its own terms and can be deleted.

**Cross-check FAILED, for an unrelated reason — see finding below.** Sending the same
`raw:true` prompt through Pharos:

| target model | pharos count | ollama `prompt_eval_count` | |
|---|---|---|---|
| `llama3.2:1b` | 5 `(exact · gguf)` | 6 | **off by −1** |
| `qwen3.5-9b-heretic` | 5 `(exact · gguf)` | 5 | match |

The cause is not BOS. `pharos/proxy/app.py` resolves **one** tokenizer at startup from
`config.model` (`resolve_gguf_path(config)`) and never consults the per-request
`spec.model`. The `llama3.2:1b` request was counted with the **Qwen** vocabulary and still
labelled `exact`. `TokenCountCache` compounds it: keys are a SHA-256 of the text alone, with
no tokenizer identity, so counts leak between models.

This is a silent-failure path of the same family as item 1 — a confidently wrong number
presented as ground truth — and it is OPEN. Sensible responses, in rough order of cost:
count per request model with a small per-model tokenizer cache; or degrade to the labelled
heuristic when `spec.model != config.model`; or at minimum drop the `exact` label and add the
tokenizer identity to the cache key. Any of these is a v0.2 scope decision.

---

## ☑☐ 7. End-to-end through a real client

**Run:** point Copilot/Cursor (or any OpenAI-compatible agent) at
`base_url = http://127.0.0.1:11435/v1` with `uv run pharos` running, and use it normally
for a few tasks.

**Expect:**
- The client behaves *identically* to talking to Ollama directly — completions arrive,
  streaming is smooth, tool calls work. Any client-side error that vanishes when pointing
  back at 11434 is a passthrough bug: capture `pharos.log` and the failing request.
- Every request appears in the TUI: `→`, `✎ input ~N (estimate · gguf)`, `✓ status`.
- Where the backend reports usage (non-streaming, or the client asks for it), the completed
  line shows `in <N> (exact)` and the gauge flips from `~` to exact.
- tok/s on the RATE line is plausible for a 9B Q4 on a 3060 (roughly 25–60 tok/s) and
  matches item 11's cross-check.

**Passthrough fidelity CONFIRMED 2026-07-24 (automated half).** With `seed:42, temperature:0`,
run direct→proxy→direct→proxy: direct runs identical to each other, proxy runs identical to
each other, and **direct == proxy byte for byte**. `options` survives the hop
(`num_predict:5` yielded `eval_count=5` both ways). Response headers through the proxy were
`content-type, date, server` — hop-by-hop and framing headers correctly dropped.

Beware a false alarm here: an A/B run straight after a *different* model was loaded shows
differing text, because the first generation after Ollama swaps models does not reproduce.
Re-run once the target model is warm before concluding anything.

**Still open:** the human half — actually driving Copilot/Cursor against
`http://127.0.0.1:11435/v1` for real work, including tool calls. Not automatable here.
tok/s measured **33.9** on this box (9B **Q8_0**, not Q4) — below the 25–60 band's midpoint
but plausible for the heavier quant. **Runbook in §13.**

---

## ☑ 8. `/v1` streaming with and without `include_usage`

**Run (without):**
```bash
curl -sN http://127.0.0.1:11435/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"qwen3.5-9b-heretic","messages":[{"role":"user","content":"count to 5"}],"stream":true}'
```
**Expect:** normal SSE deltas ending in `data: [DONE]`; TUI completed line reads
`in — not reported; estimate stands` and the gauge stays `~estimate`. This proves Pharos did
NOT inject `stream_options` (if usage appears, something is mutating the request).

**Run (with):** same command plus `"stream_options":{"include_usage":true}` in the JSON.
**Expect:** a final usage chunk arrives; TUI reconciles: `in <prompt_tokens> (exact)`.

**CONFIRMED 2026-07-24.** Without `include_usage`: 502 delta chunks, `data: [DONE]` seen, **no
usage chunk**, `RequestCompleted.prompt_eval_count = None`, TUI usage stayed
`ContextUsage(tokens=4, exact=False, source='gguf')` — the estimate stands, proving Pharos did
not inject `stream_options`. With `include_usage`: 707 delta chunks plus a final
`{'prompt_tokens': 14, 'completion_tokens': 708, 'total_tokens': 722}`; the TUI reconciled to
`ContextUsage(tokens=722, exact=True, source='reconciled')`.

---

## ☑ 9. Long generation survives (read timeout disabled)

**Run:** through the proxy, request something genuinely long, e.g.
`"write a 3000-word story"` with `"stream": true`, and let it run to completion (several
minutes on a 3060).

**Expect:** the stream never stalls or dies at a fixed interval — the proxy's upstream read
timeout is disabled (`read=None`); only connect is bounded (5s). Completion line appears at
the end with sane totals.

**CONFIRMED 2026-07-24:** `"write a 3000-word story"` streamed **5,686 chunks / 624,352 chars
over 170.3s** to completion. Largest inter-chunk gap **0.38s** — no stall, and nothing
resembling a fixed-interval cut. Completion line reported `eval_count=5686`, 33.47 tok/s,
duration 170.2s.

**If it fails** (cut at a suspiciously round time): something reintroduced a read/pool
timeout — check `_UPSTREAM_TIMEOUT` in `pharos/proxy/app.py`.

---

## ☑ 10. Client abort mid-stream → ABORTED, no leaked connection

**Run:** start the item-9 curl, Ctrl+C it after ~5 seconds. Repeat ~10 times. Then:
```bash
netstat -ano | grep 11434 | grep ESTABLISHED | wc -l
```
(PowerShell: `(netstat -ano | Select-String 11434 | Select-String ESTABLISHED).Count`)

**Expect:** each abort produces `✕ ABORTED after Xs` (red) in the TUI — never `✓ 200` — and
a matching `aborted after` line in `pharos.log`. The ESTABLISHED count to 11434 stays flat
(a few keep-alive pool connections are normal; monotonic growth across aborts is the leak
this guards against).

**CONFIRMED 2026-07-24, 10/10 aborts.** Every run produced exactly one `RequestAborted` and
zero `RequestCompleted`, with a matching log line each time:

```
WARNING pharos.proxy: #2 aborted after 5.53s (status 200 was already sent)
... ten of these, #2 through #11 ...
```

ESTABLISHED connections to 11434 went `2` at baseline then `0` after every single abort —
sequence `[0,0,0,0,0,0,0,0,0,0]`. No monotonic growth: the upstream response is being
released.

**If it fails:** growth per abort → the relay cleanup (background task) isn't releasing the
upstream response; `✓` instead of ABORTED → the streamed/aborted flag in
`pharos/proxy/forward.py`.

---

## ☑ 11. tok/s cross-check

**Run:** one non-streaming request through Pharos, then compute by hand from the response:
```bash
curl -s http://127.0.0.1:11435/api/generate \
  -d '{"model":"qwen3.5-9b-heretic","prompt":"explain dns briefly","stream":false}' \
  | python -c "import json,sys; d=json.load(sys.stdin); print('tok/s =', d['eval_count']/(d['eval_duration']/1e9))"
```

**Expect:** the printed number matches the TUI's RATE line for that request (same request,
same math: `eval_count / (eval_duration / 1e9)`).

**CONFIRMED 2026-07-24:** `eval_count=1160`, `eval_duration=34,225,036,000 ns` →
**33.893317161156524 tok/s** by hand, and the identical float on the RATE line and in
`RequestCompleted.tokens_per_second`. Exact match to full float precision.

---

## Afterwards

- Update the confirmed-vs-assumed comments: `backend.py` (key names, item 1–2),
  `gguf.py` (BOS note, item 6).
- Record any `pharos.toml` values that differed from the example (e.g. a better
  `kv_mib_per_1k` for this model — item 4 gives you the data: KV growth per 1K ctx).

### Outcome of the 2026-07-24 run

Done:

- `backend.py` docstring and `_LOADED_CTX_KEYS` now record confirmed ground truth (items 1–2).
- Tag normalization added to `_select_model`, with tests for untagged→`:latest`, tagged→tagged,
  a distinct tag that must NOT match, and a genuinely absent model.
- `details.context_length` removed from the loaded-context fallback chain.

### Fix round (same session)

- **Tokenizer trust (item 6).** `TokenCountCache` keys are now namespaced by tokenizer identity
  (the GGUF path) instead of text alone. When a request names a model other than the one the
  loaded vocabulary came from, the count is labelled `gguf:other-model` and forced non-exact —
  the TUI renders it red as `(untrusted · other model's tokenizer)`. Tag-insensitive throughout,
  so untagged config vs `:latest` request still counts as a match. Multi-model tokenizer
  support was deliberately NOT built.
- **Headroom (item 4).** `vram_headroom_tokens` is None unless a model is resident; both the CLI
  and the TUI say `headroom N/A — no model resident`, mirroring the existing usable-budget row.
  Measured VRAM is still reported — only the derived projection is withheld.
- **stderr leak (item 5).** Vocab load now runs with fd 2 redirected to devnull, so llama.cpp's
  native `n_ctx_seq` note cannot scribble on the TUI. No-ops safely if fd 2 is unavailable.
- **Naming.** Tag normalization moved to `pharos/naming.py` (`normalize_model_name`,
  `same_model`) so the profiler and the proxy share one implementation.
- **Config.** Local `pharos.toml` set to the measured `kv_mib_per_1k = 26`. The shipped example
  keeps the conservative 32 but now documents that the value is model-specific and how to
  measure it from the `size_vram` delta between two `num_ctx` values.

### v0.2 items

1. **Per-request tokenizer resolution.** v0.1 loads one vocabulary from `config.model`; requests
   for other models are counted with it and marked untrusted. Resolving a tokenizer per request
   model (with a small per-model cache) would make those counts correct rather than merely
   honest. The cache is already keyed for this.
2. **The banner's "raise num_ctx" advice is not achievable on this hardware.** At ~26 MiB/1K
   measured, reaching the advertised 262,144 needs ≈6,816 MiB of KV on top of ≈8,387 MiB of
   Q8_0 weights — ≈15.2 GB against a 12 GB card. Realistic ceiling here is roughly 96K ctx.
   The banner should say how far `num_ctx` can actually go rather than pointing at a window
   this box cannot reach. Deliberately not built.

### Input-counting audit (all four counted routes)

Every request-body field was varied against a fixed baseline while watching the backend's
reported prompt size, rather than reasoned about from the schema. Ollama 0.31.1,
qwen3.5-9b-heretic.

**Reaches the prompt — now counted:**

| field | route(s) | evidence | treatment |
|---|---|---|---|
| `tools` / `functions` | both chat | 1 tool = +258 tokens, 3 = +465, 8 = +905 | serialize to JSON, tokenize |
| `messages[].tool_calls` | both chat | +48 over content alone | serialize to JSON, tokenize |
| `context` (token array) | /api/generate | 50 entries = +49 tokens | add its length; drops `raw` exactness |
| `suffix` | /api/generate, /v1/completions | text content by definition | appended to the counted text |
| list-shaped `content` | /api/chat | native route read plain strings only | shared `content_text` helper |

**Reaches the prompt — deliberately NOT counted:** the chat template's own scaffolding (~10
tokens for a bare chat, ~200 more when tools are present) and a `template` override on
/api/generate. Both are applied server-side and are properties of the model, not the request.
No fitted constant is baked in; reconciliation closes the remainder.

**Verified NOT to reach the prompt — correctly ignored:** `format` (string *and* JSON schema),
`response_format` with a json_schema, `tool_choice`, `think`. These constrain decoding rather
than being injected as text; counting them would have inflated the estimate.

**Cannot be counted — label downgraded instead:** images (`messages[].images` on the native
route, `image_url` content parts on /v1). Vision models price these by a separate mechanism, so
no text-based count applies. Source becomes `gguf:images`, never exact, rendered red as
`(untrusted · images not counted)`. Not measurable on this machine — no vision model available.

**Result.** The residual error is now a constant rather than a function of request size:

| tools | pharos before | pharos after | backend | uncounted before | uncounted after |
|---|---|---|---|---|---|
| 0 | 1 | 1 | 11 | 10 | 10 |
| 1 | 1 | 92 | 300 | 299 | 208 |
| 3 | 1 | 270 | 476 | 475 | 206 |
| 8 | 1 | 715 | 916 | **915** | **201** |

The variable term is fully captured (~89 counted per tool against ~88 actual); what remains is
the fixed tool-mode prelude. Serialization uses default `json.dumps` spacing on purpose —
compacting with `separators=(",", ":")` strips whitespace the rendered form contains and
measurably undercounts (66 per tool vs 88), which would have left the error growing with the
request again.

The counting contract is documented on `pharos.proxy.forward._count_input`.

- `gguf.py`'s UNVERIFIED paragraph replaced with the confirmed BOS result (both directions),
  plus a pointer to the open tokenizer-binding caveat, which is unrelated to BOS.

Deliberately NOT changed: findings 1 and 2 above (tokenizer binding, headroom-when-unloaded)
are documented only, pending a v0.2 scope decision. `kv_mib_per_1k` left at 32.

`pharos.toml` values that differ from the example: `model = "qwen3.5-9b-heretic"`.
`kv_mib_per_1k` left at 32; measured ≈26 for this model.

## ☑ 12. `pharos split` — do the parts actually fit?

Everything the splitter promises is arithmetic done offline; the one thing it cannot prove on
its own is that a part it called 5,960 tokens really lands under the loaded window when a real
client sends it. Confirm with the proxy running and the dashboard open:

1. Pick a prompt that `pharos check` calls EXCEEDS against the live budget, then
   `uv run pharos split "<same prompt>" --out parts/`.
2. Paste `parts/part-01.txt` into the coding agent pointed at the proxy. Watch the event log:
   the request's counted input should land at or below the part's projection, and the context
   gauge should stay under the warn threshold.
3. Repeat for the remaining parts, carrying each hand-off forward. The hand-off is extra input
   the projection does not include — confirm it stays small enough that the last part, with the
   most accumulated hand-off text, still fits.
4. Note whether the agent honoured the OUT OF SCOPE block. If it reads deferred files anyway,
   the projection is not wrong — the scope contract is — and the wording is what needs work.

Open question this settles: whether one packed part per request is the right granularity, or
whether the hand-off accumulation means later parts need a smaller target than the warn
threshold.

### Outcome of the 2026-08-05 run (§12, live)

RTX 3060, Ollama, qwen3.5-9b-heretic Q8_0, `num_ctx=32768` -> usable 31,744 (warn 25,395).
Method: build a real plan against the live budget, then for each part assemble the request an
OBEDIENT agent would send — the part body plus the contents of exactly the files it scopes,
honouring line ranges — and compare Ollama's own `prompt_eval_count` against Pharos's
projection minus the terms the harness cannot reproduce (learned client overhead, hand-off
allowance).

**Scope mode**, prompt `Refactor everything in \`pharos/\`…`, 3 parts:

| part | projected | expected | backend | delta | fits |
|---|---|---|---|---|---|
| 1 | 25,099 | 23,273 | 23,282 | +0.04% | yes |
| 2 | 23,417 | 21,391 | 21,400 | +0.04% | yes |
| 3 | 7,237 | 5,211 | 5,220 | +0.17% | yes |

**Text mode**, a 72,730-token pasted document, 4 parts: +0.04% / +0.04% / +0.04% / +0.48%,
every part inside the budget, and all 120 section markers present and in order across the
segments (nothing lost at a cut).

The deviation is a CONSTANT +9 tokens on every part, not a percentage: it is the chat
template's own scaffolding, which §"Input-counting audit" already documents as deliberately
uncounted. Pharos under-counts by that fixed amount, which is the safe direction for a floor.
Slicing is confirmed sound: a file cut at line 427/684 costs what the parts claim.

Streaming through the proxy, same prompt direct vs. proxied (seed 7, temperature 0): 25 chunks
each, byte-identical assembled content, identical final-chunk keys.

### Finding: one `curl` erased the learned client overhead (fixed)

Live traffic found what no unit test did. After four bare `curl`-shaped requests through the
proxy, `pharos check` reported `client overhead ~10` where the real coding-agent overhead was
~1,826. The estimator minimises `input - user` over the FEWEST-message records, and a
hand-written poke has one message, no system prompt and no tools — so it wins that minimum and
then defines the overhead for every later pre-flight. The floor stayed technically a lower
bound, and completely lost its point.

Fix: observations now record `agent_shaped` (the request carried tools or a system prompt —
a boolean, so the counts-only contract holds), and the estimator prefers agent-shaped records,
falling back to bare ones only when there is nothing better. Re-validated live: with one
agent-shaped record and four pokes in the same store, the estimate is 1,232, not 10. Records
written before the field exists parse as not-agent-shaped, so old stores degrade rather than
crash.

Left as-is: `pharos_observations.json.bak` holds the pre-fix store from this session (records
without the new field), kept rather than deleted.

### §12 continued: the scope contract, measured against a real agent

The open question — *does an agent handed a part actually leave the deferred files alone?* —
was answered by running every part through the live model as a coding agent: a `read_file`
tool it could call, answered for real, so the conversation grew exactly as it would in a
session. Three parts, ~33 planned files, qwen3.5-9b at `num_ctx=32768`.

**Run 1 (original wording): 2 violations.** Part 2 opened `pharos/naming.py` and
`pharos/__init__.py`, neither in its scope, and the conversation peaked at 25,451 tokens
against a 23,990 projection. Still inside the 31,744 budget, so nothing broke — but the
exposure is real and now has a number on it.

Two changes followed, both from reading that run:

* The rule was rewritten as a RULE with a consequence and a stated alternative ("do NOT open
  it — say which one and why, and stop"), and a one-line REMINDER naming the in-scope files
  was added *after* the task block. Attention falls off in the middle of a long prompt; the
  last thing read is the thing obeyed, and the scope line is what the whole projection rests
  on, so it gets the last word.
* That reminder names files by `label()`, not `display`. Naming "forward.py" where the scope
  is lines 1-427 reads as leave for the whole file.

**Run 2 (greedy) and Run 3 (temperature 0.8, an independently sampled trajectory): 0
violations each.** Two runs is not a proof, and the mechanism is a prompt, not an
enforcement — but the change is measured, not hoped for.

Also confirmed in these runs: no part overflowed the loaded window; VRAM peaked at 10,781 MiB
of 12,288 with no OOM; and the projection tracked the real conversation closely once the agent
had read its scope (part 2: 23,569 actual vs 23,005 projected, the difference being the
harness's own tool catalogue).

### Finding: "at most 10 lines" is not a constant (fixed)

Hand-off sizes across those runs, for the identical instruction: 64, 74, 83, 107, 160, 161,
162, 168 and **328** tokens. The reserve had just been raised 200 -> 320 on the strength of the
first three measurements; the 328-token hand-off arrived on the very next trajectory and blew
through it. A guessed constant becomes a broken promise the moment a model gets wordy.

Fixed by making it `handoff_reserve` in `pharos.toml`, defaulting to 500 (clears every
hand-off observed), documented with the measurements rather than a round number.

### Finding: a sliced part assumes a range-capable reader (documented, not fixable here)

`read_file(path)` on a typical agent takes a path and nothing else, and returns the whole
file. A part scoped to `forward.py (lines 1-427 of 684)` therefore costs the FULL file on such
a client, not the slice. Pharos cannot fix another agent's tool signature, so the plan now
says it out loud with the number: what those files would cost if read whole. Silence here
would have been a plan that quietly does not work.

## ☐ 13. The human half — a real coding agent, doing real work

The last unverified claim in the project. Everything in §7 that a script could check is
confirmed: passthrough is byte-identical, `options` survive, headers are clean. What no script
here can produce is a real agent's traffic — a system prompt, a tool catalogue, a conversation
that grows over a dozen turns, and tool calls that actually round-trip. Two things rest on it
that nothing else can settle:

- **The passthrough claim is only as strong as the clients that have exercised it.** So far
  that is `curl` and a purpose-built harness, both of which send what Pharos expects.
- **`agent_shaped` calibration was fixed against one synthetic record.** The whole point of
  §12's finding is that the learned overhead should come from agent-shaped traffic. Until a
  real agent has written observations, the pre-flight floor for a real agent is still an
  extrapolation from a harness.

### Setup

```bash
ollama run qwen3.5-9b-heretic "hi"     # model resident at a known num_ctx
uv run pharos                           # dashboard up, proxy on 11435
```

Point the client at the proxy and nothing else — the value of the run comes from it being the
only endpoint for a whole session:

| client | where |
|---|---|
| Continue / Cline | `apiBase: http://127.0.0.1:11435/v1` on an `openai`-type provider |
| Cursor | Models → OpenAI → Override base URL → `http://127.0.0.1:11435/v1` |
| Copilot (BYOK) | OpenAI-compatible endpoint → same URL |

Any dummy string works as the API key; Pharos forwards headers untouched and Ollama ignores it.

Before starting, snapshot the store so the session's contribution is separable:

```bash
cp pharos_observations.json observations-before-13.json
```

### Then just work for 20–30 minutes

Real tasks, not prompts written to be validated: have it read files, edit something, run a
test, iterate on a failure. Tool calls are the part that matters — they are the request shape
(`tools` in the body, `tool_calls` in the messages) that §"Input-counting audit" measured but
that no real client has yet sent through the proxy.

### Pass criteria

1. **Indistinguishable from talking to Ollama directly.** No client-side error, no stall, no
   truncated stream. The test for any oddity: point back at `11434` and see if it survives. If
   it vanishes, it is a passthrough bug — keep `pharos.log` and the failing request.
2. **Every request appears in the TUI** — `→`, `✎ input ~N`, `✓ status`. A request the agent
   made that the log does not show is a counting gap, and worth more than a clean run.
3. **Tool calls round-trip.** The agent calls a tool, gets a result, continues. Note the input
   count on the turns that carry a tool catalogue: they should be dramatically larger than the
   bare turns (the audit measured ~+258 tokens for one tool, ~+905 for eight).
4. **The label is `gguf`, never `heuristic`**, and never the red
   `(untrusted · other model's tokenizer)` — if the client sends a model name that is not the
   one in `pharos.toml`, that red label is correct behaviour, but it means the session's counts
   are not the ones you want to draw conclusions from. Fix the client's model name and re-run.
5. **The context gauge tracks a growing conversation.** Over a dozen turns it should climb
   toward the warn threshold rather than sitting flat — a flat gauge across a long session
   means the conversation is not being counted as it accumulates.

### What to record afterwards

```bash
# Did the session write agent-shaped observations?
python -c "import json; r=json.load(open('pharos_observations.json'))['records']; print(len(r), 'records,', sum(bool(x.get('agent_shaped')) for x in r), 'agent-shaped')"

# What does the pre-flight now think a real agent costs?
uv run pharos check "add a docstring to pharos/config.py" --json | python -c "import json,sys; print(json.load(sys.stdin)['overhead'])"
```

**Expect:** agent-shaped records in the store, and an `overhead.tokens` in the ~1,000–2,000
band with `provenance` naming the observed traffic — the ~1,826 figure §12 quotes came from a
real agent, so a number in that neighbourhood is the calibration fix confirming itself against
the traffic it was designed for. A figure near 10 means bare pokes are still winning the
minimum, and the fix did not hold outside its test.

**If a request breaks:** the two most likely shapes are a body field no route parses (the
counting path takes a COPY, so a parse failure should degrade the count, never the request —
if it killed the request, that is the bug) and a streaming client that disconnects differently
from `curl` (§10 covers abort handling for `curl`'s shape only).

Record the outcome here the way §12 records its runs: what was measured, not what was hoped.
