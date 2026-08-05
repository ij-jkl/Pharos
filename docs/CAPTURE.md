# Capturing the dashboard shot

The README describes a dashboard nobody reading it can see. One image fixes that, and it has
to show the thing Pharos exists for — not an idle screen.

**Target:** `docs/pharos-dashboard.png`, then uncomment the image line at the top of
`README.md`.

## What has to be in frame

The shot is worth taking only if all four are visible at once:

1. The **⚠ CONTEXT MISMATCH** banner. Load the model at a `num_ctx` well below its advertised
   maximum and the banner is there by itself — the measured 32,768-of-262,144 (12.5%) state
   from `DESKTOP_VALIDATION.md` §3 is the whole argument for the project in one line.
2. The **context gauge** carrying a real number with its provenance label — `(exact)` or
   `(estimate · gguf)`, not `heuristic`. A heuristic label in the hero image undersells the
   part that took the most work.
3. The **VRAM gauge** with a model resident, so headroom is a real figure rather than the
   honest `N/A — no model resident`.
4. **Several event-log lines**, not one. A log with a single request looks like a demo; four
   or five with varying input sizes and tok/s looks like a session.

## Getting there

```bash
ollama run qwen3.5-9b-heretic "hi"      # load the model at a small num_ctx
uv run pharos                            # dashboard up
```

Then drive a few requests through `http://127.0.0.1:11435` — the natural way is to point your
coding agent at it and work for a minute, which fills the log with genuinely varied traffic.

Widen the terminal until nothing wraps (the gauges and the event log both lose their shape in a
narrow window), then capture the terminal only, not the whole desktop.

## If you want a GIF instead

A still is enough, and a still always renders. A GIF earns its size only if it shows something
a still cannot: the banner *updating* when the loaded window changes (reload at a larger
`num_ctx` and watch it move within one 5s refresh — the §3 run has the exact command), or the
context gauge flipping from `~estimate` to `exact` when the backend reports usage. Keep it
under ~5 MB and put it at `docs/pharos-dashboard.gif`.

## Also worth a shot, further down the README

A terminal capture of `pharos check` returning EXCEEDS, and `pharos split` cutting the same
prompt into parts — the floor/ceiling block and the per-part projections are the output that
makes the two commands legible without reading the prose. Those are plain text, so a fenced
code block pasted into the README works as well as an image and stays searchable.
