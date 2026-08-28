# Capturing the README's shots

The README describes a tool nobody reading it can see. Four images fix that, and every one of
them has to be a real session — a staged screenshot of a measurement tool is a contradiction.

Two scripts produce all of them, and both can be re-run whenever the output changes:

```bash
uv run python docs/capture_dashboard.py     # docs/pharos-dashboard.svg
uv run python docs/capture_cli.py           # shot-check, shot-split, shot-run, shot-review
```

Each capture is written twice: the SVG (or two, for a run) the README embeds, and
`docs/shot-*.txt` beside it holding the **whole** capture — the report, then the live progress
that was on stderr while it was produced. A screenshot that quietly drops the part where it went
wrong is the one thing this must not be.

## The dashboard — `capture_dashboard.py`

It starts the real proxy, runs the real dashboard headless via Textual's test harness, sends
five `/api/chat` requests of deliberately varying size straight through the proxy, and exports
an SVG. The numbers in the image are the numbers those requests produced.

The shot is worth taking only if all four are visible at once:

1. The **⚠ CONTEXT MISMATCH** banner. Load the model at a `num_ctx` well below its advertised
   maximum and the banner is there by itself — the measured 32,768-of-262,144 (12.5%) state from
   `DESKTOP_VALIDATION.md` §3 is the whole argument for the project in one line. `SHOT_NUM_CTX`
   in the script pins the window every request is sent under, so the image cannot drift away
   from the caption beside it.
2. The **context gauge** carrying a real number with its provenance label — `(exact)` or
   `(estimate · gguf)`, not `heuristic`. A heuristic label in the hero image undersells the part
   that took the most work.
3. The **VRAM gauge** with a model resident, so headroom is a real figure rather than the honest
   `N/A — no model resident`.
4. **Several event-log lines**, not one. A log with a single request looks like a demo; four or
   five with varying input sizes and tok/s looks like a session.

## The CLI — `capture_cli.py`

```bash
uv run python docs/capture_cli.py --only check,split   # fast; no generation
uv run python docs/capture_cli.py --only run           # 20-40 minutes on a 3060
uv run python docs/capture_cli.py --keep               # leave the fixture on disk to inspect
uv run python docs/capture_cli.py --project ../thing   # shoot an existing project instead
```

It builds its own fixture: a 22-module `shop/` package that formats with `%` throughout, a
13-test suite, a clean `ruff` baseline, and a git repository holding all of it as one commit —
then writes a `pharos.toml` beside it inheriting this machine's backend, model and tokenizer, at
`SHOT_NUM_CTX = 8192`. That window is the point of the exercise: the prompt's floor lands just
above the usable budget, exactly the situation the tool exists for.

Then it runs the ordinary CLI against that fixture as a subprocess and photographs whatever came
back, exit code included. The `run` shot writes files, which is why it works on a throwaway
fixture in a temp directory and not on this repository.

**Three things it needs.** A backend on the address in `pharos.toml` (the script loads the model
at the shot's window first, or every shot reports `NO VERDICT` — correct, and a poor
photograph); a model that can actually call tools for the `run` shot, which is a smaller set than
"coding models" (`DESKTOP_VALIDATION.md` §16); and `git`, for the run's own undo.

**One thing it fakes, and only one.** Rich writes ANSI escapes to a terminal and not to a pipe,
and on Windows a redirected stream is detected as a legacy console and gets no colour at all — so
a piped capture comes out flat grey. The child process therefore starts with
`detect_legacy_windows` pinned to False and `FORCE_COLOR` set. That is a lie told to Rich about
where its output is going. The words, the numbers, the verdict and the exit code are the CLI's.

### What each shot has to show

- **check** — the per-file table with `exact` beside every count, the floor, the budget it is
  judged against, and a verdict that is genuinely `EXCEEDS`. If the fixture grows and the floor
  clears the budget, the shot is worthless: grow `PROMPT`'s scope or drop `SHOT_NUM_CTX`.
- **split** — the plan region only (`FLOOR` through *"Each projection is a floor"*), because the
  part bodies below it run to hundreds of lines. It should carry the per-part projections and
  the `--semantic` note saying whether the model's grouping was used or rejected, and why.
- **run** — two images out of one capture, because ninety lines is not a picture: `shot-run`
  from the header through the end of the scorecard (the part list, coverage, continuity, damage,
  compaction, the audit and verification), and `shot-review` for the opinion printed underneath
  it. The report is on stdout and the live progress is on stderr, so the capture keeps the two
  streams apart — merged, the image photographs the spinner's last frame instead of the
  scorecard, which is exactly what the first attempt produced.

A bad run is still a good shot. Coverage below 100%, a part named for breaking the build, a
`complete: false` verdict — those are the tool working. The README says as much beside the image.
