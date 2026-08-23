# Changelog

Every version is a tier, and each one is allowed to do strictly more than the last. The proxy
itself has not changed since v0.1 and is not going to: it observes, it never mutates.

Numbers quoted here were measured on the machine described in `DESKTOP_VALIDATION.md` — an
RTX 3060 12 GB running Ollama — and the record of how is in that file rather than this one.

## v0.5 — "Divide by meaning"

`pharos split --semantic` and `pharos run --semantic` let a model choose which files belong in
a part together. A run has always talked to a model to get the work done; this is the first
time one is allowed to decide something *about the plan*, and it is opt-in for that reason.

- **The model's job is one partition, and nothing else.** It never writes a part, picks a
  budget, or decides whether one fits — the same renderer, tokenizer and thresholds produce all
  of that either way. Its answer is checked in code for coverage (nothing dropped, invented or
  repeated), for empty parts, and for the part-count ceiling. Anything to do with size is
  repaired rather than rejected; see below.
- **Any failure falls back to position packing and names the check it failed.** A refused
  connection, prose instead of JSON, a hallucinated filename, a group over budget — all land in
  the same place, and the plan says so in one line whichever way it went.
- **A group that is right and too big is cut, not rejected.** Size is the one thing a proposal
  can be wrong about that is mechanically repairable, so an oversized group becomes consecutive
  parts of the same concern, in the model's own order. Rejecting on size had been discarding
  correct groupings over arithmetic — including the clean three-subsystem split, at a budget
  one part tighter. Coverage failures are still fatal; those repair cannot fix.
- **It works, and it is honest about how often.** On a task across three subsystems whose
  filenames carry no hint of them, `qwen3.5:9b` returned the ideal render/physics/audio split in
  the same three parts position packing used — a free improvement. On a tighter budget, two of
  three models were rejected and the plan was identical to not passing the flag.
- **Thinking is disabled for the call, and that is a measurement.** Asked to partition six
  filenames, one model spent 4,096 tokens and 81 seconds reasoning and returned nothing. With
  thinking off the same question costs 66 tokens.
- **Five lines of each file travel with the question.** On names alone a model returns the files
  in listed order under invented titles, which is position packing wearing a hat. Sending heads
  made two of three models produce the clean grouping and every one of them twice as fast. This
  is the one Pharos command that sends anything anywhere; without the flag, nothing leaves.
- Config: `semantic_model`, `semantic_max_extra_parts`; CLI: `--semantic` on both commands, and
  `s!` in the launcher.

### Fixed, by an end-to-end pass on a project shaped like a project

Every fixture used to build the above had its files in one flat directory. Driving the whole
CLI against a small but ordinary project — `src/` package, `tests/`, a `pyproject.toml`
configuring ruff and pytest — turned up four bugs that no test above could see. Three of them
predate v0.5.

- **Semantic grouping never worked on a nested project.** The report spells paths with the
  platform separator, a model answers in posix, and the comparison missed on punctuation — so
  every file came back reported as *dropped and invented at once* and the grouping fell back on
  all three groupers. It had never worked outside a flat directory. The same bug, with a
  different symptom, had already been fixed once in the tool dispatcher; there were two
  implementations of one rule and the second grew the same defect. There is one now
  (`pharos/paths.py`), and nought of three groupers became three of three.
- **`--no-git` reported that a run which wrote files wrote nothing.** `files_changed` was an
  empty list rather than "unknown", printed as `Nothing was written.` under a run that had just
  done the work. That same list is handed to the verifier as the set of written files, so
  `--no-git` also silently turned the syntax check off: a run could break every file in the tree
  and be told there was nothing in a format it could parse. The flag costs you the undo, which
  is documented; it was also costing the change report and the verification, which is not.
- **`pharos run --json` never said how the parts were grouped.** The human output says it on
  every run — that note is the whole safety story for `--semantic` — and the machine-readable
  form was silent, so a CI step could not tell a model-arranged plan from a position-packed one.
  It now carries the mode, the grouping, the note, the per-part budget and each part's title,
  projection and files.
- **`pharos check --target` was accepted and ignored.** A 500-token budget and a million-token
  budget returned the same verdict. It works now, and skips the backend probe entirely when
  given, so a verdict is available with nothing running.
- **A proposal can be accepted and have decided nothing.** One group containing every file
  passes every check, and the oversized-group repair then cuts it in the order it arrived —
  which is position packing wearing the model's title. Detected and said outright.
- Smaller: `--semantic` without `--split` is an error rather than a silent no-op; a group with
  no title is no longer numbered into `" (1 of 2)"`; the grouping note stopped saying which
  grouping won twice; and 31 lines of unreachable code came out of the run renderer.

### Measured, not claimed

Seven real agentic runs of one task, tree reset between each, recorded in
`DESKTOP_VALIDATION.md` §19. **Coverage read 100% every single time and four of the seven
shipped a broken build** — every one caught by verification, with the reported syntax errors
matching the damage line for line. The passthrough guarantee was checked against a live backend
rather than a mock: same request direct and proxied gives identical content, an identical
response key set, and an identical `prompt_eval_count` — the backend's own count of what it
received, which would move if the proxy had injected anything.

## v0.4.2 — verification

`pharos run` now finishes by re-running the checks your project already has, and the exit code
follows them.

- **Verification, as a sixth scorecard line.** Coverage says every assigned file was written;
  it cannot say the result still works. A measured run scored 100% coverage and shipped
  `sorted(total.items(), ...)` where the variable is `totals` — COMPLETE, and a `NameError`.
  Mechanical throughout: exit codes from the project's own tools, never a model's opinion of
  the code.
- **A baseline before the first part.** A check that was already failing is reported as such
  and never charged to the run. Only a check that passed before and fails after can fail it.
- **Skips are never passes.** No tool on PATH, an unparseable command, a timeout — each is
  reported by name. A failure whose baseline never completed is reported as unattributable
  rather than blamed on the run.
- **A third verdict.** Full coverage with a broken build reads `BROKEN` and names the check,
  instead of the contradictory `INCOMPLETE — 0 of 6 files were never changed`.
- Config: `verify`, `verify_commands`, `verify_timeout_seconds`; CLI: `--no-verify`.

## v0.4.1 — corrections

Three numbers that were wrong, and the dashboard nobody could see.

- **The KV rate is derived from the model, not guessed.** `kv_mib_per_1k` was a hand-tuned
  constant and it divides the VRAM headroom estimate. With 4 GB free it claimed 153,846 further
  context tokens where 28,050 was the truth — advice pointing at the OOM this project exists to
  warn about. Every term needed to compute it already arrived from `/api/show`. Hybrid
  SSM stacks and sliding-window attention are handled or declined explicitly; the report says
  `derived` or `configured`.
- **The ceiling is enforced against the backend's own counts.** `SAFETY_MARGIN` was calibrated
  where drift ran 0.87–0.98x; on another model it ran 0.81–0.92x and overshot by 275 tokens,
  meaning a ceiling was enforced against a number below the real prompt. `prompt_eval_count`
  was already arriving and being discarded.
- **The undo instruction names the branch the run started from.** `git checkout main` was
  hardcoded, so on a `master` repository the printed way back failed — at the one moment it is
  needed.
- **The README shows the dashboard**, captured from a real session by
  `docs/capture_dashboard.py` rather than composed.

## v0.4 — "Do"

`pharos run` carries a task out instead of printing a plan: it pre-flights, divides with the
same splitter, and runs each part as its own conversation seeded only by the previous part's
hand-off. This is the one part of Pharos that writes files, and it will not start without a way
back — a clean tree and its own branch, or a snapshot of every original.

- The runner is a **client** of the proxy, exactly like Continue or Cursor. Its traffic is
  observed on the way past, never rewritten.
- Scope is enforced in the tool layer, not requested in the prompt: a part told to open three
  files physically cannot open a fourth.
- Nothing is trimmed. A file too large for the remaining window is refused, not truncated.
- The **scorecard**: coverage, continuity, repair, headroom, drift, convergence — none of which
  require asking a model anything. The exit code follows coverage, not survival.
- `max_files_per_part` defaults to 2 because that is what was measured: a part completes about
  1.5–2.0 files and then believes itself finished, whatever it was given.

## v0.3 — "Divide"

`pharos split` cuts a prompt that does not fit into parts that do — mechanically, by scope and
by position, never by meaning. No model is asked what your task means.

- **Scope split** when the files overflow: each part repeats the task verbatim and narrows the
  files, slicing large ones into line ranges that only ever move forward.
- **Text split** when the pasted text itself overflows: ordered segments on paragraph, then
  line boundaries.
- Every part carries a projection from the same tokenizer that produced the verdict, packed
  against the warn threshold. A plan that cannot work is refused rather than faked.

## v0.2 — "Warn"

`pharos check` answers whether a prompt can possibly fit, before you paste it.

- Named files are tokenized exactly into a **floor**; named directories are counted separately
  into a **ceiling**, because adding them would turn a lower bound into a guess.
- Client overhead is learned from traffic previously observed through the proxy.
- Ambiguous, missing, binary and directory references are reported, never silently dropped.
- Exit codes: `0` fits, `1` exceeds, `2` no verdict.

## v0.1 — "Observe"

The proxy and the live dashboard. A transparent passthrough between a local coding agent and a
local LLM backend, byte-for-byte in both directions.

- **⚠ CONTEXT MISMATCH** — the headline: what the model advertises against what is actually
  loaded.
- Context and VRAM gauges, and a per-request event log.
- **Honest counting**: every number carries its provenance — `(exact)`, `(estimate · gguf)`, or
  `(heuristic chars/4)`.
