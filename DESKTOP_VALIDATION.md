# Desktop validation checklist

Every assumption that could not be verified on the dev laptop (no NVIDIA GPU, no Ollama),
in the order to work through them on the desktop (RTX 3060 12GB, Ollama, qwen3.5-9b-heretic).
Each item: what to run, what to expect, what a failure means. Commands are for Git Bash;
they work in PowerShell too unless noted.

Legend: ☐ open · ☑ pass · ☒ fail (note what you saw)

---

## ☐ 0. Fresh-machine setup

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

**If it fails:** a compile attempt during sync means the CPU wheel index was not used
(check `[tool.uv.sources]` / `[[tool.uv.index]]` in `pyproject.toml`); more than 5 skips or
any error means a hidden laptop assumption — note which test.

---

## ☐ 1. `/api/ps` loaded-context field name

The defensive key chain in `pharos/profiler/backend.py` tries
`context_length → context → num_ctx → details.context_length`. It has only ever run against
mocked payloads.

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

## ☐ 2. `/api/show` advertised max context

**Run:**
```bash
curl -s http://localhost:11434/api/show -d '{"model": "qwen3.5-9b-heretic"}' \
  | python -c "import json,sys; d=json.load(sys.stdin); print({k:v for k,v in d['model_info'].items() if 'context' in k or k=='general.architecture'})"
```

**Expect:** `general.architecture` (e.g. `qwen3`) plus `<arch>.context_length` with the
advertised maximum (Qwen3.5-9B should advertise 262144 or similar). `pharos-profile` shows
the same number as `advertised`.

**If it fails:** no `*.context_length` key → adjust `_extract_advertised_ctx` in
`backend.py` to the real key layout.

---

## ☐ 3. The mismatch banner firing for real

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

**If it fails:** banner absent while profile shows both numbers → `detect_ctx_mismatch` or
the TUI banner logic; banner shows stale numbers → the 5s profile refresh isn't updating.

---

## ☐ 4. NVML on the RTX 3060 + headroom sanity

**Run:** `uv run pharos-profile`, then cross-check `nvidia-smi`.

**Expect:** GPU row `NVIDIA GeForce RTX 3060 · free X / 12,288 MiB` matching nvidia-smi
(±100 MiB), source nvml (no fallback). VRAM headroom row shows
`≈<free/32*1000> more ctx tokens (estimate)` — e.g. 8,000 MiB free → ≈250,000. Sane means:
headroom scales with free VRAM and drops visibly after the model loads.

**If it fails:** GPU N/A → NVML binding vs driver problem (check `nvidia-smi` works at all;
then the `_probe_nvml` fallback order). Absurd headroom → revisit `kv_mib_per_1k` for this
model in `pharos.toml` (it is a config estimate, not code).

---

## ☐ 5. GGUF resolution for the real model (both paths)

**Run (autodetect):** leave `gguf_path` unset in `pharos.toml`, keep `model` set. Start
`uv run pharos`, send any request through the proxy, then check:
- the TUI input line says `(estimate · gguf)` — **not** `heuristic`;
- `pharos.log` contains `loading tokenizer vocab (vocab_only) from ...ollama...blobs...`.

**Run (override):** set `gguf_path` to the blob path the log printed (or find it:
`ls ~/.ollama/models/blobs` and match the manifest digest), restart, same expectations.

**Expect:** both paths produce gguf-labeled counts.

**If it fails:** autodetect falls to heuristic → Ollama's manifest layout differs from
`manifests/<registry>/<ns>/<name>/<tag>`; inspect `~/.ollama/models/manifests` and fix
`_find_manifest` / `_model_layer_digest` in `pharos/tokenizer/resolver.py`. Override path
failing means the file check or vocab load — see the log for the exception.

---

## ☐ 6. C8 positive case: BOS on a Llama-family model

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

**If it fails** (no BOS added, or cross-check off by one): option (a) is wrong for this
wheel — downgrade `raw=true` labeling to `exact=False` with an explanatory source label
(option (b) from the design discussion) and update the `gguf.py` docstring. If it passes:
delete the UNVERIFIED paragraph in `gguf.py`.

---

## ☐ 7. End-to-end through a real client

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

---

## ☐ 8. `/v1` streaming with and without `include_usage`

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

---

## ☐ 9. Long generation survives (read timeout disabled)

**Run:** through the proxy, request something genuinely long, e.g.
`"write a 3000-word story"` with `"stream": true`, and let it run to completion (several
minutes on a 3060).

**Expect:** the stream never stalls or dies at a fixed interval — the proxy's upstream read
timeout is disabled (`read=None`); only connect is bounded (5s). Completion line appears at
the end with sane totals.

**If it fails** (cut at a suspiciously round time): something reintroduced a read/pool
timeout — check `_UPSTREAM_TIMEOUT` in `pharos/proxy/app.py`.

---

## ☐ 10. Client abort mid-stream → ABORTED, no leaked connection

**Run:** start the item-9 curl, Ctrl+C it after ~5 seconds. Repeat ~10 times. Then:
```bash
netstat -ano | grep 11434 | grep ESTABLISHED | wc -l
```
(PowerShell: `(netstat -ano | Select-String 11434 | Select-String ESTABLISHED).Count`)

**Expect:** each abort produces `✕ ABORTED after Xs` (red) in the TUI — never `✓ 200` — and
a matching `aborted after` line in `pharos.log`. The ESTABLISHED count to 11434 stays flat
(a few keep-alive pool connections are normal; monotonic growth across aborts is the leak
this guards against).

**If it fails:** growth per abort → the relay cleanup (background task) isn't releasing the
upstream response; `✓` instead of ABORTED → the streamed/aborted flag in
`pharos/proxy/forward.py`.

---

## ☐ 11. tok/s cross-check

**Run:** one non-streaming request through Pharos, then compute by hand from the response:
```bash
curl -s http://127.0.0.1:11435/api/generate \
  -d '{"model":"qwen3.5-9b-heretic","prompt":"explain dns briefly","stream":false}' \
  | python -c "import json,sys; d=json.load(sys.stdin); print('tok/s =', d['eval_count']/(d['eval_duration']/1e9))"
```

**Expect:** the printed number matches the TUI's RATE line for that request (same request,
same math: `eval_count / (eval_duration / 1e9)`).

---

## Afterwards

- Update the confirmed-vs-assumed comments: `backend.py` (key names, item 1–2),
  `gguf.py` (BOS note, item 6).
- Record any `pharos.toml` values that differed from the example (e.g. a better
  `kv_mib_per_1k` for this model — item 4 gives you the data: KV growth per 1K ctx).
