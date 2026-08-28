# Changelog

Every version is a tier, and each one is allowed to do strictly more than the last. The proxy
itself has not changed since v0.1 and is not going to: it observes, it never mutates.

Numbers quoted here were measured on the machine described in `DESKTOP_VALIDATION.md` — an
RTX 3060 12 GB running Ollama — and the record of how is in that file rather than this one.

## v1.1.6 — "A fraction of a limit that cannot be exceeded"

- **`headroom` reported 104% of a part's ceiling.** It is peak-over-ceiling, so above 100% it
  describes a limit being exceeded — and the limit is the one thing in a run that by
  construction cannot be: nothing goes on the wire until it has been proven to fit.

  Nothing was wrong with the ceiling. The **peak** was taken at the top of the loop, which is
  the single moment a conversation is legitimately over it: a tool result has just been
  appended, and the two checks that either compact it back or end the part have not run yet.
  So the high-water mark included a state that was never sent, and then reported it as a
  fraction of what may be sent.

  The peak is now taken past both checks, immediately before the request goes out: the largest
  conversation this part *actually sent*, which is what the line always claimed to be. The
  excursion is not lost — it is exactly what the compaction and hand-off lines report, by name
  and in tokens, at the moment it happens. A test pins the invariant, since the number is one a
  reader uses to decide whether the next slightly larger file breaks the run.

## v1.1.5 — "One bad part is not a bad run"

Both of these came out of the run photographed for the front page, which is the argument for
photographing a run that went badly.

- **A part that failed ended the run.** Part 11 of 13 died because the model emitted a tool
  call the backend could not parse — `XML syntax error on line 2: element <function> closed by
  </parameter>` — and parts 12 and 13 were never sent. Each part is its own conversation
  against its own files, seeded only by the hand-off and the record, so a call that came back
  wrong says nothing about the next part.

  A failed part is now recorded, said, and stepped over. What ends a run is **two failures
  back to back**: that is the shape of a backend that has gone away rather than a reply that
  came back malformed, and grinding through eleven more parts to discover it is not a knowable
  cost. The verdict is unchanged — any failed part still makes the run `FAILED` — and the
  repair sweep now runs after a single failure, because the files that part never wrote are
  exactly the leftovers it exists for. It still does not run after an abort or a
  `--stop-on-break`: there is either nothing left to ask, or a tree that no longer parses.

  The failed part's **writes go into the record either way**. They landed on disk; dropping
  them because the call after them failed would hand the next part a record missing files it
  can see. Its hand-off does not: a part that failed produced no closing summary, and passing
  the previous part's on would describe work this part never did.

- **The part table renamed the plan underneath itself.** The header read *"divided into 13
  parts"* and the rows below it were numbered *"part 1/11"* through *"part 11/11"* — the
  denominator was how many parts had executed, so a run that ended early silently restated how
  many there had been. One screen disagreeing with itself about the only number on it a reader
  can check. The rows now count against the plan, and a line under the table says how many
  parts never ran and that their files still count against coverage — which they always did
  (`planned_files`), invisibly.

## v1.1.4 — "Show it, then say it"

No behaviour changed. The front page did.

- **The README was 798 lines of reasoning and one picture.** Everything in it was true and
  almost none of it was the first thing a reader needs, which is: paste a prompt too big for
  your card at your own project and the work still gets done. It is now a third of the length,
  built around that one path — install, point it at a project, paste, watch it iterate — with
  the design arguments compressed to the sentence each one earns and the measurements moved to
  a benchmarks section at the end, where a reader who wants numbers can find all of them
  together.

- **`docs/capture_cli.py` takes the CLI shots the README now embeds**, the same way
  `capture_dashboard.py` takes the dashboard: by running the real thing. It builds a 22-module
  fixture project that formats with `%`, gives it a 13-test suite, a clean `ruff` baseline and
  a git repository, writes a `pharos.toml` beside it at an 8,192-token window, and then runs
  `check`, `split` and `run` against it as ordinary subprocesses. What the commands printed is
  what the images show, exit codes included, and the untrimmed transcript of each is written to
  `docs/shot-*.txt` beside the image it came from.

  One thing is forced and it is not a number: Rich writes ANSI escapes to a terminal and not to
  a pipe, and on Windows a redirected stream is detected as a legacy console and gets no colour
  at all, so the child process starts with `detect_legacy_windows` pinned to False. That is a
  lie told to Rich about where its output is going, and it is the only one.

  The run shot is of a run that went badly, which is the point: 68% coverage, three files left
  unparseable with the parts that broke them named, one part dead on a tool call the backend
  could not parse, a clean audit, `FAILED`, exit 1 — and 8,856 tokens reclaimed by compaction
  across nine parts that would otherwise have stopped at their ceiling. A tool whose front page
  only shows it succeeding is advertising, not documentation.

- **A test now fails if the README embeds an image the repository does not hold.** Generated
  files in `docs/` are exactly what a cleanup sweep removes without anyone noticing until the
  front page is a row of broken frames.

- `pharos.toml.example` described `target_folder` as the root `pharos check` resolves against.
  It is also the tree `run` writes to and audits, which is the whole reason the README can tell
  you to point Pharos at a project and leave it where it is.

## v1.1.3 — "Names that are not files"

I threw twenty-six escape shapes at the path containment `pharos run` relies on — parent
traversal, laundered traversal, absolute paths, UNC paths, backslash separators, a directory
junction pointing out of the workspace, and a twelve-deep climb at `Windows/win.ini`. **Nothing
resolved outside the workspace.** The junction is the one worth naming: it is the classic way
past a containment check that compares strings, and this one resolves before it compares.

What it did turn up is on the other side of the fence:

- **Windows resolves `NUL`, `CON`, `AUX`, `PRN`, `COM1-9` and `LPT1-9` to hardware devices in
  every directory, with any extension.** `src/con.py` is the console. Writing to one is not an
  error — it succeeds, `exists()` returns True afterwards, and the directory is empty, because
  the bytes went to the device. Measured: a 49-character write to `<root>/NUL` returned
  normally and read back as `""`.

  A model asked to create `aux.py` or `con.py` — ordinary names for "auxiliary" and
  "configuration", legal on Linux and present in repositories written there — would have its
  write reported as landing, and lose it. The audit catches that as a write the disk does not
  show, which is the audit doing its job, but coverage still counts the file and `--no-audit`
  turns the only witness off. `COM1` and `LPT1` are worse than silent: they open a serial or
  printer port.

  Refused now, with a message that says why. Matched on the stem, since the reservation ignores
  the extension, and only on Windows — on Linux `aux.py` is a file that exists and works, and a
  repository is entitled to contain one. The same task is allowed to differ between the two
  platforms here, because the platforms differ.

## v1.1.2 — "The way back, after a Ctrl-C"

- **An interrupted run could not tell you how to undo it.** `pharos run` makes a branch (or a
  snapshot) before it writes anything, and on Ctrl-C it printed *"Files already written are on
  the run branch"* and stopped there. It could not do better: the record holding the branch
  name was only bound once `run_task` returned, and an interrupted run never returns one. So
  the user was left checked out on a `pharos-run/...` branch, told that in the abstract, with
  no name for it and no base to go back to — at the one moment they most need it, having
  stopped the run because it looked wrong.

  `run_task` now accepts the record to write into, so the CLI holds it throughout and prints
  the same two lines the normal footer does: what to diff, and what to check out and delete.
  Interrupted before a branch or snapshot existed, it says nothing has been changed, which is
  a stronger statement than naming a way back that does not exist.

Also checked and left alone: every backend call is already bounded — connect, read, write and
pool timeouts on the chat, semantic and review paths — so a hung backend cannot hang a run.

## v1.1.1 — "Survives being installed"

Nothing users see. Everything here is something that only shows up on somebody else's machine.

- **Two of the three stores could be left half-written.** The observation store has always
  written a temporary file and renamed it over the target; the template-cost store and the new
  VRAM store called `write_text`, which truncates first. A crash, a full disk or a second
  process arriving mid-call left valid UTF-8 that is not valid JSON — and since every reader
  here treats a corrupt store as an empty one, the cost was a silent loss of everything that
  had been learned. All three now go through one `pharos.store.write_json`.

- **The atomic rename was failing on Windows, and had been all along.** Writing the test for
  the above turned it up: `os.replace` refuses with WinError 5 while any other handle is open
  on the destination, including a reader holding it for the microsecond it takes to parse.
  Under a reader thread looping over a store being rewritten, most replaces failed — and a
  failed write here is only a log line, so the effect was a store that quietly stopped learning
  whenever anything was reading it. The dashboard reads these on a timer while a run writes
  them, so that is the ordinary case, not a stretch. Both directions now retry over ~150ms:
  a reader is denied just as transiently while a replace is in flight, and swallowing that as
  "the store is empty" is how a run would start uncorrected for no reason. Atomicity is not
  traded for it — every attempt is still a whole-file swap, and giving up leaves the previous
  store intact.

- **The package shipped no `py.typed`.** Every module is `mypy --strict` clean and fully
  annotated, and none of it was visible to anything that imported Pharos. Added, declared as a
  wheel artifact, and pinned by a test — along with one that imports every console script's
  entry point, since a typo there is only ever discovered by a user on a fresh install.

- **Python 3.13 is supported.** `requires-python` was `<3.13`, which on a current machine means
  no install at all. The whole suite passes on 3.13 unchanged, so the pin is `<3.14` and CI runs
  four legs: 3.12 and 3.13 on Linux and Windows. Each leg names its interpreter explicitly,
  because `.python-version` pins 3.12 and a bare `uv run` in the 3.13 leg would silently test
  3.12 and report green — which is worse than not testing 3.13 at all. A CI step asserts the
  interpreter is the one asked for.

- **CI now tests the artifact, not just the source tree.** Every leg builds the wheel, installs
  it into a clean environment and asserts that `py.typed` and `styles.tcss` actually shipped,
  that the proxy and the TUI both construct, and that `pharos check` reaches a verdict with no
  config file and no backend — the state a new user is in. This is the step that would have
  caught `py.typed` shipping nowhere, and the only one that would notice the stylesheet falling
  out of the wheel and the TUI coming up unstyled for everyone except the developer.

- Packaging metadata for a public release: classifiers, keywords, and Homepage / Repository /
  Issues / Changelog URLs. The launchers no longer promise to fetch "Python 3.12" specifically.

## v1.1 — "Measured here"

The VRAM figures stop guessing on the models they could not derive a rate for.

- **A context token's VRAM cost is now measured on your machine, not assumed.** The KV rate
  drives the headroom estimate and the "raise num_ctx toward N" advice, and it had two sources:
  derived from the model's own GGUF metadata, or the single configured constant. Two
  architecture families are refused by the derivation — hybrid SSM stacks and sliding-window
  attention — and those fell back to one number for every model, which measured 4-7x wrong.

  Nothing extra had to be run to fix it. `/api/ps` reports `size_vram` and the loaded window on
  every probe, VRAM is linear in context, and a profile happens on every `check`, every `run`
  and every few seconds of the dashboard. So each probe now files what the model occupied at
  that window, and once three windows above 8K have been seen the rate is the slope of a line
  fitted through them. That is exactly how the derivation's own validation table was built,
  by hand, with curl; `pharos/vram.py` does it for free.

  It reproduces the hand measurement it replaces. Three ordinary profiles at 8K, 16K and 32K
  against `qwen3.5-9b-heretic` fitted **32.23 MiB/1K** — the same figure to two decimal places
  as the validation table arrived at with curl, and 0.01% off it. On `qwen3.5:9b`, a hybrid
  stack that publishes too little to derive from and was therefore stuck on the constant, the
  same three windows gave 33.20. Both are labelled `measured here` and print the windows
  behind them.

- **A measurement outranks a derivation**, which is not the obvious order and is the one the
  numbers argue for. Derivation is exact arithmetic over published metadata, but it counts the
  cache and nothing else, and on all three architectures it was validated against it read 1-4%
  *under* what the card actually gave up — llama.cpp allocates per-token scratch outside the
  cache proper. Reading low overstates headroom, which is the one direction this figure must
  not be wrong in. Where both exist, both are printed: agreeing is worth seeing and disagreeing
  is worth more.

- **Four things stop a measurement from existing at all**, and each one is a refusal rather than
  a smaller number: fewer than three windows, no 8K of span between the lowest and highest, a
  model `/api/ps` does not report as fully resident (`size_vram != size` — a partly offloaded
  model's VRAM does not move with context in a way worth fitting), and a changed digest, since
  a model re-pulled under one tag can be a different file with a different cache. A slope that
  comes out negative or outside 1-1000 MiB/1K is discarded too: something other than the cache
  moved between those readings.

- **The fit is honest about its own shape.** The first live run turned up something worth
  recording: `qwen3.5:9b` measured 33.20 MiB/1K from 8K to 32K — exactly, to the byte, across
  two intervals — and then 28.56 from 32K to 64K. The curve is piecewise linear, not linear,
  and one straight line describes neither piece exactly. It is still one straight line: the
  bend is downward, because a fixed allocation is amortising over more tokens, and a rate
  fitted slightly high spends the headroom estimate too fast, which is the direction this
  number is allowed to be wrong in. Fitting only the top segment would answer the marginal
  question more exactly and would do it from two points, which is the shortcut the three-point
  rule exists to refuse. The table is in `pharos/vram.py` and pinned by a test.

`kv_rate_derived: bool` becomes `kv_rate_source: "measured" | "derived" | "configured"`, since
there are three answers now and a bool cannot name three things. `vram_memory_file`
(`pharos_vram.json`) is the store: a window size and a byte total per model, gitignored, and
safe to delete — ordinary use re-measures it.

## v1.0.7 — "What the run itself left behind"

An end-to-end pass against a live backend — profiler, proxy, dashboard, `check`, `split` and
two real runs that wrote files. Six defects, four of them in the first five minutes of using
it the way the README says to.

- **`--dry-run` made the very next `pharos run` refuse to start.** The dry run writes
  `pharos.log` into the workspace, `git_guard` counted it as the user's uncommitted work, and
  the run stopped with *"Commit or stash them first"* — blaming the user for a file Pharos had
  just written. Reproduced from a clean repository on the first attempt.

  `own_paths` has always computed which files are ours, from configuration rather than by
  guessing at what a Pharos file looks like; the guard simply never asked. It does now, and so
  does `changed_files`, which was reporting *"3 file(s) changed on disk"* for a run that
  changed two and handing the `.log` to the syntax verifier, which duly reported a written file
  in a format it could not parse. The guard also reads `--untracked-files=all`: git collapses
  an untracked folder to `dir/`, which no per-file exclusion can match and which understates a
  hundred stray files as one change.

- **A task that fit in one window had no coverage denominator, and printed a bold green DONE.**
  An undivided run carries no per-part file list, so `planned_files` was empty and the
  scorecard said *"one unrestricted part; nothing to measure coverage against"* — for the
  commonest case there is. Observed live: a run naming two files touched one and reported DONE.

  The denominator was never missing. The prompt named the files and the pre-flight resolved
  them, so an unscoped run is now scored against `scoped_display_names(report)` — deliberately
  the same list the splitter packs from, so a divided run and an undivided one cannot drift
  into being scored against two different things. It also gives the syntax baseline something
  to record, which an unscoped run previously did without. A text split names no files, finds
  no denominator, and is scored exactly as before.

- **`pharos run --dry-run --json` wrote nothing at all and exited 0.** The human table goes to
  stderr under `--json`, and the dry-run branch returned before reaching any payload, so a CI
  step piping into `jq` got a parse error from a command that had succeeded. It now emits the
  plan, which is the answer to *"what would this run do?"* without running it.

- **Long paths wrapped through the middle of a directory name.** A workspace under
  `AppData\Local\Temp\pharos\<uuid>\…` came out split across three lines through the uuid and
  could not be copied out of the terminal. `shorten_path` drops the middle rather than the end,
  because the leaf is the part a reader is looking for, and uses `~` where it applies.

- **The drift line contradicted itself.** It shouted *"short by 276 tokens, past the 256-token
  margin"* in yellow while noting, dimly, *"every request corrected"* — both true, and together
  useless. When the remembered template cost covers the shortfall and nothing went out
  uncorrected, no ceiling was ever enforced against a number below the real prompt, so it now
  says that instead of raising an alarm about a risk that was already handled.

- **`docs/capture_dashboard.py` was not reproducible, and the README told you to run it.** It
  set `num_predict` and never `num_ctx`, so the backend loaded whatever it liked — 4,096 here —
  and the regenerated shot read *"loaded 4,096 (1.6%)"* under a caption, alt text and body
  promising 32,768 (12.5%). The window is pinned now.

Two smaller things: the plan table says that parts after the first include room for the
hand-off they arrive with, so two parts holding near-identical files no longer show projections
that differ by half without explanation; and a check excerpt no longer ends on ruff's bare `|`
diagram opener when the cut has already removed the diagram under it.

Nothing above changes what the proxy does, and the four measurements that were checked against
the hardware all held: the KV fallback for a hybrid SSM stack measured 28.6 MiB/1K against a
configured 33 — conservative, in the safe direction; the estimate ran a flat ~10 tokens under
the backend at every conversation size, which is the fixed-offset claim the template memory is
built on; and the template memory took `exposed_requests` from 2 to 0 between two runs.

## v1.0.6 — "One version"

No behaviour changed. The front page stopped being a history.

- **The README describes what Pharos is, not the order its tiers arrived in.** Four
  release-numbered headings (`(v0.2 "Warn")`, `(v0.3 "Divide")`), a `**Status:**` line and 24
  version mentions in the prose (*"From v0.6…"*, *"new in v1.0"*, *"Until v1.0 the README
  said…"*) meant a first-time reader had to reconstruct the product from its release order.
  All of it is gone; `CHANGELOG.md` was always the release history and now it is the only one.
  Same features, same measurements, 898 lines down to 766, and in one voice.

- **Two things it never documented.** `--exclude` has been on both `pharos check` and
  `pharos run` and appeared on neither page, and `max_files_per_part` — the knob that decides
  how a run is divided — was in `PharosConfig` but not in `pharos.toml.example`, so the one
  value most worth tuning was invisible to anyone editing their config. Both are documented,
  and a test now fails if a `pharos run` flag exists that the README does not mention.

- **The test that enforced the old shape is replaced by one that enforces the new.** It used
  to assert the README carried a status line matching `__version__`, and it would have failed
  this release for the right reason under the wrong rule. It now asserts the opposite: no
  status line, no release-numbered headings, no prose dating a feature to a version.

- **Comments stopped narrating the release they were written in.** Thirty-five version
  references across eighteen modules said when a line arrived rather than what it does. One was
  actively wrong: `pharos/agent/__init__.py` still promised that "history compaction is out of
  scope here", which `--compact` has contradicted since v1.0. The one version string left in
  the source is the false positive `preflight/extract.py` tests itself against.

- **`config.py` was 57% comment, and most of it was `pharos.toml.example` retyped.** The
  measurements behind every default live in the example file, which is the one a user edits.
  The module now carries the invariant a code reader needs — that `handoff_ledger` *shares*
  `handoff_reserve` rather than adding to it, that an unset `num_ctx` means measure what is
  loaded rather than change it — and points at the example for the reasoning. One copy of each
  explanation instead of two drifting ones.

- **`run_task` was 443 lines.** Five self-contained phases lift out of it as named steps —
  taking the baseline, choosing the route, seeding the template memory, resolving what changed
  on disk, folding this run's drift back in. The execution loop is untouched. 374 lines, and
  each phase now readable without holding the other four.

- **Removed `workspace_for`**, a one-line wrapper nothing has called.

## v1.0.5 — "An edit that says where it landed"

Why two parts ran out of window, from the run in §27. It was not the size of the task.

- **`replace_lines` now hands back the region it wrote, renumbered.** Every edit shifts the
  line numbers below it, so a model holding numbers from an earlier read has to get fresh ones
  before touching the same file again — and the only way to get them was to read the whole file
  back. Pharos was explicitly telling it to: *"read it again before editing further down"*.

  The log makes the loop plain: `replace_lines X`, `read_file X`, `replace_lines X`,
  `read_file X`, over and over. One part read a 344-token file **seven times** and a second one
  six times. Across the run, **~15,541 tokens went on re-reading files already sitting in the
  window** — more than a single part's entire ceiling of 14,604. Two parts hit that ceiling.

  The result of an edit now carries the changed region with its new numbers and three lines of
  context either side, so an adjacent edit needs nothing further. Capped: past forty written
  lines, echoing the region back costs more than the read it saves, and the answer goes back to
  the old advice. The shift warning stays, and now says exactly where the model's own numbers
  go stale rather than condemning the whole file.

  **What five live runs do not establish is whether the model takes the offer.** The rate of
  edits chased by a re-read went 52.6% / 64.3% / 68.8% before, and 57.6% / 53.3% after — both
  after-figures land inside the before-range, and the run that re-read the most is the best run
  by every other measure. The mechanism is pinned by tests and visible in the log (three
  consecutive edits with no read between them, which no earlier run does). The claim stops
  there: Pharos no longer asks for a whole file when it can hand back the ten lines that answer
  the question. See `DESKTOP_VALIDATION.md` §28.

- **"Stopped early" was doing the work of two different facts.** Both parts that hit the ceiling
  had already written every file they owned; they ran out of window checking their work over,
  not with files untouched. The scorecard says how many of them that was, beside the count and
  folded into nothing.

## v1.0.4 — "The room that was set aside"

The one thin hand-off left after v1.0.3, chased down. It was not the model either.

- **The hand-off was never allowed into the reserve kept for it.** `handoff_reserve` is
  subtracted from a part's ceiling up front, precisely so the closing summary has somewhere to
  go. The reply room was then measured against that same ceiling — which hands the hand-off
  everything *except* the space set aside for it. It was also charged for a tool catalogue it
  does not carry, the request going out with no tools at all.

  Both together: a part that ended at 99% of its ceiling was asked for its hand-off with
  **443 tokens against a 500-token reserve**, and answered with nothing. On the same part the
  fix gives it about **1,379** — the 443, plus the 500 reserved, plus the ~436 the catalogue
  was costing it. `SAFETY_MARGIN` still stands behind the whole thing untouched.

- **Pharos's own words were being counted as the model's hand-off.** A reply cut off at its
  token limit gets a note appended saying so. A reply cut off before it produced a single
  character is then made *entirely* of that note — and it reads as a hand-off: non-empty, in
  the part's own text field, counted as produced. It is Pharos talking to itself. `produced`
  now asks what the model said, so the count got stricter rather than kinder, and a hand-off
  that is only a note is asked for again.

## v1.0.3 — "Both halves of a hand-off"

The two things §25 left open, diagnosed. Neither was a property of the model, which is what
both had been recorded as.

- **The hand-off was only ever asked for on the exit that went badly.** A part ends two ways:
  it stops calling tools, or the step limit cuts it off. Only the second was sent the
  instruction that says *NAME each file you changed*; the ordinary exit was handed whatever
  the model happened to say alongside its last tool call. The six-part run in §25 abandoned no
  part, so every one of them took the unasked path — **5 hand-offs expected, 3 produced, 2 of
  those 3 thin**, with two parts answering their own last tool result with nothing at all.
  A parting message that names none of the part's own work is now asked for one properly,
  once, with no tools offered.

  It does not make continuity green by construction, and it is not meant to. Being asked is
  not answering: `names_its_work` still judges the reply, a part that answers with prose about
  nothing is still thin, and the count of hand-offs that **had to be asked for** is reported
  beside the others rather than folded into any of them. A part that volunteered a real
  hand-off keeps its own words and costs no extra turn. A part that wrote nothing and said
  `NO CHANGES NEEDED` is left alone — the request forbids that phrase, and asking would only
  talk it out of a true answer.

- **Compaction was reported as the backend dropping context.** Both features shipped in v1.0
  and met for the first time on a real run. The truncation detector rests on *"a conversation
  only grows, so the backend's count for it can only grow"*, and `--compact` is Pharos
  deliberately making the conversation smaller. In §25 one line reclaiming 1,955 tokens
  produced **two BACKEND TRUNCATED warnings and a `truncated_parts: 1`** — blaming the user's
  backend for context Pharos had just dropped itself. This is the one warning that must never
  cry wolf, so it now stops crying at its own footsteps: compacting drops the high-water mark,
  which rearms on the next request. Dropped rather than adjusted by what was reclaimed,
  because that figure is in our vocabulary and the mark is in the backend's, and subtracting
  one from the other only cries wolf more quietly.

- **Written once, where it had been written twice.** No behaviour change, and the plan for a
  real nineteen-file prompt comes out identical part for part. Each of these was two copies
  that could drift apart, and two of them already had:
  - The **packing loop** — grow a span while the predicted cost fits, then shrink it against
    the measured one — existed once for cutting a file into line ranges and once for cutting
    free text into segments. This is the arithmetic the whole promise rests on.
  - **A finished part** was described by hand at three of its four exits, twenty-odd identical
    fields each. That is how `files_read` and `handoff_requested` each reached some exits and
    not others in a single afternoon.
  - **The relative display path** had drifted: one copy resolved the workspace root before
    comparing and the other did not, so the same file could be named in the verdict and shown
    absolute in the plan beside it. It now lives in `pharos/paths.py`, which exists because
    this project has paid for two copies of a path comparison once already.
  - The one-token **model load** request, the first-line **truncator**, and the four-times
    unwrapping of a client overhead that may not have been measured yet.

- **`repair_pass` was missing from `pharos.toml.example`.** It is the one config key the
  example never listed.

- **A test asserted that a binary was on your PATH.** `detect_commands` is configuration AND
  installation, and the test for it failed on this machine purely for being invoked from a
  shell where the virtualenv was not activated. It skips when the tool is not installed.

## v1.0.2 — "A prompt that does not fit"

A giant prompt, run end to end against a nineteen-module project on an RTX 3060, found
five things. `DESKTOP_VALIDATION.md` §25 is the pass; this is what came out of it.

- **A refusal with no reason.** `pharos run` exited with *"no plan could be built — unknown
  reason"*. `SplitPlan.reason` is set only when no plan can be built at all; a scope plan
  whose parts come out over budget explains itself through the parts, and the runner had no
  handler for that case. It now names them: which parts, how many, and the worst overrun.
- **`--exclude PATH`** (repeatable, on `check`, `split` and `run`). A long prompt names a
  file in order to FORBID it — *"do not touch `templates.py`"*, *"leave the tests alone"* —
  and extraction cannot tell that apart from naming it as work, because the difference is in
  the meaning of the sentence. So it is a flag, on the same bargain `--resolve` strikes over
  an ambiguous reference. Excluded files are listed in the report, never silently dropped,
  and an exclusion reaches inside an expanded directory.
- **The overhead estimator prefers records that carried a tool catalogue.** `agent_shaped`
  is "tools OR a system prompt", so a `curl` probe with a system prompt satisfies it — and
  two of them held the learned overhead at **19 tokens across 116 real agent requests**,
  understating a check's floor by more than a thousand. Tri-state, so an existing store keeps
  working and falls back exactly as before.
- **A format spec is not a dotfile.** A prompt about string formatting spells out
  `` `%.2f` becomes `:.2f` ``, and `.2f` was surfacing in the verdict as a file that could
  not be found. No dotfile convention starts a name with a digit.
- **Coverage was punishing a run for correctly leaving a file alone.** Three of a run's
  three misses were one-line `__init__.py` modules a part had opened, found nothing to do
  in, and left — indistinguishable, inside an 83.3%, from three files nobody looked at.
  Coverage stays written-over-scoped; a line beside it now says how many misses were opened
  and left alone, measured from the dispatcher's record of the reads rather than from
  anything the model claimed about them.
- **661 tests**, `ruff` and `mypy --strict` clean.

## v1.0.1 — what validating v1.0 found

`DESKTOP_VALIDATION.md` §24 was the first live pass over the v1.0 features, and it closed
with two defects of Pharos's own. Both are fixed here, along with the diagnostic that pass
spent twenty minutes wishing existed.

### The audit was reporting Pharos's own bookkeeping

Every live run carried the same two entries, in three categories at once:

```
unattributed: ['pharos.log', 'pharos_observations.json']
out_of_scope: ['pharos.log', 'pharos_observations.json']
by_checks:    ['pharos.log', 'pharos_observations.json']
```

Pharos writes its log and its observation store into the folder it is auditing, and then
reported them as changes nobody claimed. Six false findings a run, beside the real ones. A
check whose output is mostly noise is one people learn to skip, which would have cost the
feature everything it is for.

- **Excluded by configuration, not by pattern.** `log_file`, `observations_file` and
  `template_memory_file` are config keys, so `own_paths()` resolves those exact three against
  the workspace root and the walk skips them. Nothing guesses at what a Pharos file looks
  like; a store the user has configured somewhere else entirely resolves outside the root and
  is simply absent, because the walk was never going to see it.
- **The undo directory goes with them, and is not walked at all.** In a folder that is not a
  repository, `.pharos/undo-<timestamp>/` holds a copy of every original the run is about to
  overwrite — the single largest false finding available, and indexing it would have doubled
  the audit's cost to produce it.
- Confirmed on a live run afterwards: `clean: true`, all four categories empty, every part's
  changed files exactly its claimed files.

### "Not enough conversations" could be the wrong reason

`estimate_agent_reads` returned `ReadEstimate | None`, so a caller with no number had only one
sentence to print. On the pass's first run the proxy was still bound to another model's GGUF;
every pair was refused as incommensurable, and the check reported *"there are not yet three
conversations to learn from"* — when there were **seven**, and all of them had been refused.

- **It now returns `(estimate, why_not)`**, the shape the rest of this project uses for an
  answer that may not exist, and the reason travels through `CheckReport.reads_note` to the
  renderer instead of being replaced there by a stock line. It is in `--json` as
  `reads_unknown_because`.
- **Five different nothings, told apart**: no traffic for this model at all; traffic that was
  never agent-shaped; measurements refused, with the dominant cause named and counted;
  conversations that never grew; and genuinely too few conversations, with the count it does
  have. Each one has a different thing the reader could do about it, which is the whole
  argument for distinguishing them.
- Replayed against the real store with every record tainted, it now reads: *"60 of 60
  turn-to-turn measurement(s) over 76 observed request(s) could not be used — 59 of them
  because it was counted in another model's vocabulary. Nothing is guessed in their place."*

### And the thing a run could not say: the model cannot call tools

The pass opened with a run that covered **0%** — every part `steps=0`, seventeen seconds for
the whole thing. Not the window, not the task, and not a Pharos fault: asked with a tool
catalogue, `qwen2.5-coder` at both 7b and 14b writes the call into `content` as text on Ollama
0.31.1. The qwen3.5 family returns a real `tool_calls`. Finding that out took a hand-written
probe against the backend, which is not something this tool should make anybody do.

- **The scorecard counts native calls and recovered ones separately**, and says so when a run
  ends having never received a structured call: *"none — the model never returned a structured
  tool call. Every request carried the catalogue … try one that does before reading anything
  else on this card."*
- **Recovery is not evidence of support.** `recover_tool_calls` picks up a bare JSON call
  written as prose and gets the work done; it is counted apart, and a run that leaned on it is
  told so in its own line. A model that instead answers *"I have read the file and it needs no
  changes"* gives recovery nothing to work with, which is exactly the run that covered 0%.
- It is a fact rather than an inference: the requests carried the catalogue, and nothing came
  back with a call in it.

### Also

- `pharos check --json` gains `reads_unknown_because`; the run's JSON gains a `tool_calls`
  block (`native`, `recovered_from_text`, `unsupported`).
- **648 tests**, `ruff` and `mypy --strict` clean.

## v1.0 — "Everything it would not say"

Nine versions carried a section headed *What Pharos does NOT do (yet)*. Five entries. This
release closes four of them and keeps the fifth forever.

> The compaction figures below come from the suite's fixtures. The live pass is
> `DESKTOP_VALIDATION.md` §24, run afterwards on an RTX 3060 against `qwen3.5:9b` and a
> project of real classes: same task, same window, `--compact` took parts abandoned at the
> ceiling from **2 to 0** and reclaimed **1,459 tokens**; the read prediction was corroborated
> against a file read twice, to within ~1%; `--review` named the right file, line and cause
> of a real syntax error with **0 findings discarded**. That section also lists two defects
> the pass found.

The four were not closed by relaxing anything. Each one had a reason it was excluded, and each
is admitted here only in the shape that survives that reason: a prediction that is labelled a
prediction and cannot move an exit code, a compaction that cannot touch what a part was asked
to do, an audit that reports facts and not intent, and an opinion that is checked in code
before anyone reads it and cannot change the verdict printed above it.

### What the agent opens on its own — the third number

`pharos check` counted what you NAMED: files exactly, into the floor; directories in full, into
a separate ceiling. What the agent decides to open once it starts working was in neither, and
`pharos/preflight/extract.py` has carried a note since v0.2 saying that predicting it "is v1.0".

- **It needed no new record.** Between two consecutive requests of one conversation the input
  grows by three things and no others: what you typed, what the model last said, and whatever
  the client injected on its own. The first two are already in the observation store, so the
  third is the remainder — `injected = Δinput − output_prev − Δuser`. That residue is tool
  results and file reads. It is a count derived from counts, and it names nothing: the store's
  promise (counts only, never text, no file names, nothing reconstructable) is untouched. A
  version of this that logged which files an agent read would have been easier and was never
  on the table.
- **A median across conversations, with its range printed beside it.** Not a maximum: this is
  a prediction about a client Pharos has not met yet, and the honest centre is the middle with
  the spread shown, not the worst case dressed as a budget.
- **Four kinds of pair are dropped rather than guessed at.** A response that reported no
  `eval_count` hides the model's own reply inside the growth, and subtracting nothing would
  bill it to the agent as a file it read. A pair mixing a backend-exact input with an estimated
  one carries the chat-template offset instead of cancelling it. A pair counted in another
  model's vocabulary is not commensurable. And only agent-shaped requests count at all — the
  overhead estimator already learned live what happens when four `curl` pokes are allowed to
  vote on a number about coding agents.
- **Clamped at zero, which makes it read LOW on a thinking model.** A reply whose reasoning
  block is not replayed into the next prompt over-subtracts. The honest floor for "tokens the
  agent read" is none rather than a negative number, and that is the direction to be wrong in.
- **Below three usable conversations it says nothing.** Same rule as everywhere else here: a
  missing number is honest, a made-up one is not. Config: `agent_read_tokens` pins it.
- **It never changes an exit code.** Those still judge the floor. What is new is that a floor
  which fits while the expected total does not is now called out — the case every version
  before this was silent about. `--reserve-reads` on `split` and `run` goes further and takes
  that room off every part's ceiling before packing.

### Compaction — `--compact`

What fills a part's window is tool results, and most of them are files the model finished with
long before anything stopped it. Until now the ceiling stopped the part there and asked for the
hand-off: correct, and expensive.

- **Only tool results are ever touched**, and they are stubbed in place rather than removed. A
  result deleted out from under the assistant turn that called for it leaves a tool call with
  no answer — a malformed conversation, not a smaller one. The stub names what was dropped and
  how big it was, which is the whole difference between compaction and a context silently
  truncated underneath the model.
- **The newest results survive, and so do the small ones.** A refusal, a write confirmation and
  a failed call are all tool results and all tiny. Counting them into the protected window
  meant three refusals in a row could push the one real file a part had read out of protection
  and stub it, while carefully preserving three messages saying "there was no room". Measured
  exactly that way on the first run of this code, and fixed by protecting substance rather than
  recency alone.
- **It runs at both moments the window runs out**: when the ceiling is reached before a
  request, and when a read is refused for space — the commoner shape, and one the ceiling check
  never sees, because the part is comfortably inside its window and still cannot open the next
  file. `ToolResult.needed_room` is what tells the two apart from every other reason a call
  fails.
- **The ceiling still decides.** Whatever compaction gives back, the projection is measured
  again against the same number, and a part that still does not fit stops where it would have.
  Bounded at three rounds, so a part that re-reads what it just dropped cannot grind.
- **A run WITHOUT the flag now reports what it would have bought.** On the test fixture, five
  files of 1,834 tokens against an 8,344-token ceiling: without it, three reads land and two
  are refused for space; with it, two rounds of compaction give back 3,668 tokens and all five
  land in the same window. Only a run that did NOT use the flag can produce that number, which
  is why it is reported at all.
- **The part that runs out of room does not stop.** It is refused a read and carries on, so
  `stopped_early` is False and the scorecard reported nothing for exactly the case worth
  reporting. `room_refusals` -- counted after any compaction retry, so it means the window
  stopped the call rather than that it was briefly tight -- is what the reclaimable figure is
  now asked about.

### The audit — what the disk says

Every record a run kept was testimony from one witness. The ledger holds what the dispatcher
saw land, the scorecard counts what the parts reported, the damage list names what stopped
parsing — all of it Pharos describing its own actions. If a write was reported and never
landed, or a file changed that no tool of ours touched, none of them could say so.

- **Three snapshots per part**: before it starts, after its tools finish, and after your own
  checks run. Four findings come out — *unattributed* (changed, nobody claimed it),
  *absent* (claimed, the disk does not show it), *out of scope* (moved during a part told to
  leave it alone), and *by your checks* (a formatter in a test command, a snapshot test writing
  its snapshots — real edits to your tree during a run that nothing recorded before).
- **Facts, not judgements.** An unattributed change may be perfectly fine. The point is that it
  stops being invisible, because every coverage figure in the scorecard is computed from
  claims.
- **`(size, mtime_ns)`, not a content hash**, and what that gives up is stated exactly rather
  than glossed: a write is invisible to this only if it leaves the file the same length AND
  lands on the same timestamp as the snapshot it is compared against. Windows advances its
  clock about every 15 ms, so that window is milliseconds wide there rather than nanoseconds —
  and between two of these snapshots sits a whole model round trip.
- On by default; `--no-audit` turns it off. Prunes the same vcs/venv/cache directories as every
  other walk in the project, which is why `IGNORED_DIRS` is now public.

### The review — `--review`

The exclusion this project resisted hardest, because an opinion filed next to a column of
measurements borrows their authority without earning it. So it is admitted, and quarantined.

- **Off unless asked, and it cannot change the verdict.** Coverage, damage, verification, the
  scorecard and the exit code are all computed before the review runs, and none of them is
  shown it. It is the last thing computed and the last thing printed, under its own heading,
  which says what it is every single time.
- **Every finding is checked in code before you see it** — the same discipline `--semantic`
  works under. A finding must name a file this run actually changed and point at a line inside
  a hunk the model was actually shown; the severity must be one of the three asked for; a
  boolean is not a line number. Anything else is discarded, and **the count of what was
  discarded is printed**. A model asked to review code will invent a plausible line number, and
  a plausible line number is exactly what a reader trusts.
- **A diff too large for one call is left out and named**, never shown in half. Half a diff
  reviewed as though it were whole is the failure `read_file` already refuses to commit.
- Works without git too: a workspace that is not a repository already has every original under
  `.pharos/undo-<timestamp>/`, so the diff is reconstructed from the snapshots.
- A review that could not happen reads as a review that could not happen, never as a clean bill
  of health. The two are opposite conclusions and would otherwise print almost identically.

### And the one that stays

**No request mutation, ever.** Everything above writes, asks or decides as an ordinary client
of the proxy, outside the request path, exactly like Continue or Cursor. `--compact` reclaims
the window inside a conversation Pharos owns; your agent's history belongs to your agent, and
trimming it would mean rewriting a request. The proxy has not changed since v0.1 and is not
going to.

### Also

- `pharos check --json` schema version is **2**: the payload gained `reads` and `expected`, and
  a plan gained `reads_reserved`. Additive, and the number still moved — a version that never
  changes tells a consumer nothing it can act on.
- The launcher gains `r? <prompt>` (run, then review) and names the three flags it has no
  letter for. It also had a duplicated menu line since v0.7, which is gone.
- Config: `agent_read_tokens`.
- **648 tests**, `ruff` and `mypy --strict` clean.

## v0.9 — "Remember what the template costs"

Every run in `DESKTOP_VALIDATION.md` §20-§22 reported the same line, nine times: *short by
288-297 tokens, past the 256-token margin*. Pharos counts a conversation from the messages it
holds; the backend counts it after applying a chat template applied server-side and invisible
from here, so the projection is a floor by construction and `SAFETY_MARGIN` is a constant
guessing how short it falls. On this model it guessed low, reproducibly, all day.

A session already fixed most of this for itself: `prompt_eval_count` comes back with every
response, so from the second request onwards the ceiling is scaled by a measured ratio rather
than a constant. The `under_counted` docstring named what was left: *"the first request is the
one the correction cannot cover."* A part starts with no responses to learn from, so with three
parts that is three requests a run enforced against a bare constant.

- **The ratio is remembered between runs, per model.** Nothing new is measured and nothing new
  is asked of the backend — it seeds the correction a session was already making for itself,
  one request earlier. A model's first run learns it; every run after starts corrected.
- **It is stored in TOKENS, not as a ratio, and that is a measurement.** The first design
  here multiplied — remember `backend / ours` and scale the projection by it. Seventeen paired
  requests over conversations from 1,235 to 3,604 tokens said otherwise: the gap was **263-314
  tokens across the whole range**, flat, while the ratio it implied fell from 1.22x to 1.09x as
  the conversation grew. It is fixed scaffolding, so a fixed number describes it. A 1.22x
  factor on a 3,604-token conversation reserves 793 tokens to cover 314, and the waste grows
  with the window — on a 60K conversation it would throw away more than 13,000 tokens of every
  part. The multiplicative version was built, measured, and replaced before it shipped.
- **A median of per-run maxima.** The maximum WITHIN a run, because a ceiling holds at the
  worst case or it does not hold. The median ACROSS runs, because a maximum across runs
  ratchets and never comes back down — this project has recorded a 2.57x it could not explain,
  and storing it would have shrunk every future part for good on the strength of one request.
  One number per run, so a long run cannot outvote a short one.
- **Capped at 4,096 tokens, and it says when it hits the cap.** A remembered gap comes straight
  off every part's ceiling; real chat-template scaffolding is a few hundred tokens, and past
  that the projection is wrong in a way a constant should not be papering over.
- **Never below zero.** A backend counting fewer tokens than we did is a sign we were being
  cautious, not licence to fit more in.
- **The assumption, stated:** the template's cost does not grow with the conversation. That is
  what was measured over a 3x range on one model. A template whose scaffolding scaled with
  length would be under-corrected above the largest conversation yet seen — the live in-session
  measurement tracks it upward within a run, `SAFETY_MARGIN` sits underneath, and the drift line
  reports the raw estimator either way.
- **The new number to watch is `exposed_requests`** — requests sent with nothing correcting
  their ceiling. It is a different question from `under_counted`, which measures the ESTIMATOR
  and is expected to read short whatever happens; this asks whether any request was ever
  actually enforced against a bare constant. Before this it was the first request of every part
  on every run.
- **The store is counts only, like the observation store**, and it makes runs slightly better
  rather than being needed for one: delete it, corrupt it, or point at a read-only path and the
  next run measures it from scratch and says nothing about it.
- Config: `template_memory_file` (`pharos_templates.json`, gitignored).

## v0.8 — "Cheap checks, every part"

v0.7 named the part that broke a file. It watched only the parser, and the limit it shipped
with was measured immediately: two runs in four broke `ruff` without breaking any file's
syntax, so half the observed damage had no part's name on it. A constant above a module's
`import` is valid Python; `ast.parse` has no opinion about it and `E402` does.

- **The project's own checks now run after every part, when they are cheap enough.** Same
  before/after comparison as the parser: a check that passed before a part and fails after it
  was broken by that part, one that arrived red is nobody's, and a later part that makes it
  pass again is credited.
- **Cheap enough is measured, not configured.** The baseline already runs every check before
  the first part, so how long each one costs on this machine and this repository is known and
  free to read. `per_part_check_seconds` (3.0) is the ceiling; on the measured fixture `ruff
  check .` comes in under it and `pytest -q` does not, and neither number had to be guessed by
  anybody. Set it to 0 for v0.7 behaviour, where only the parser ran per part.
- **A check with no baseline never runs per part**, whatever it costs. Without a "before"
  there is nothing to compare against, and reporting a failure with no baseline as damage
  would name a part for a state it may have inherited.
- **A part that wrote nothing is not re-checked.** There is nothing it could have broken, and
  the checks would only re-measure the previous part's answer at the price of running them.
- **`--stop-on-break` covers both now.** A part that leaves the linter red ends the run on the
  same terms as one that leaves a file unparseable.
- The scorecard's `damage` gains `check`, so a row reads `part 2 broke ruff check .` or
  `part 2 broke src/a.py` from one record. Both are in `--json`.
- Config: `per_part_check_seconds`.

Not demonstrated live: a run **halted** by a linter break. Runs 4 and 5 showed the halt on a
syntax break and the branch is shared, but the model behaved on the two `--stop-on-break` runs
made after this landed, so that specific path rests on unit tests and an argument rather than
an observation. Recorded in `DESKTOP_VALIDATION.md` §22 as such.

## v0.7 — "Name the part that broke it"

v0.4.2 answered *is the project broken*. On a divided run the more useful question is *by
whom*, and the checks at the end cannot answer it: they see one tree, after five parts have
all written to it, and a red suite at that point sends you to read five diffs.

- **Every part is parsed the moment it finishes.** Only the files that part wrote, only in
  formats there is a parser for — Python, JSON, TOML — which costs milliseconds against the
  minute a part takes. No timeout, no configuration, and nothing new asked of the project.
- **Attribution is against the state immediately before the part, not the run's baseline.** A
  file part 2 broke does not become part 4's fault for touching it afterwards, and a file that
  arrived broken is nobody's.
- **A break that a later part repairs is recorded as both.** It is not charged against the run
  — it did not survive to the end — and it is not hidden either, because a part that spends
  its window undoing an earlier part's damage is a division that is not working.
- **`--stop-on-break` ends the run at the first part that leaves a file unparseable.** Off by
  default, which is a judgement rather than a measurement: every coverage figure this project
  has published was measured on runs that ran to the end. What argues for it is measured
  though — from v0.6 the record carries a broken part's conventions to every part after it, so
  damage propagates instead of staying where it happened. A stopped run reads `STOPPED`, not
  `FAILED` (nothing errored) and not `INCOMPLETE` (the remaining files were never attempted),
  and the repair sweep does not run over a tree that no longer parses.
- **The scorecard says who.** `damage` lists part, file, error and who repaired it;
  `broke_the_build` reduces it to the parts still responsible at the end. Both are in `--json`.
- `--stop-on-break` with `--no-verify` is refused rather than accepted and ignored — the
  failure `pharos check --target` shipped with in v0.4.
- Config: `stop_on_break`. CLI: `--stop-on-break`.

### Fixed

- **A run that stopped early reported 100% coverage.** Coverage was measured against the scope
  of the parts that executed, so halting after part 1 of three removed four files from the
  numerator and the denominator together and the bar stayed full beside the verdict saying the
  run had been halted. The denominator is now the plan, which is what coverage has always
  claimed to measure. **Predates v0.7 and is not about `--stop-on-break`:** the loop has broken
  early on a failed part since v0.4, and every such run reported the same flattered figure.
  Nobody noticed because a failed part is rare and its report is read for the failure.

### Known: it catches what does not parse

A run can break a linter without breaking the parser. One measured run put a constant above a
module's `import` — valid Python, `E402` to ruff — and got no damage row, correctly. The
end-of-run checks caught it without naming a part. Extending the watch to `ruff` would work;
extending it to `pytest` would not, and a check that attributes some failures and not others
needs designing rather than assuming.

## v0.6 — "Carry the thread"

Every part starts from an empty conversation, so whatever crosses the gap between them is the
whole of what part 4 knows about part 1. Until now that was the hand-off the model wrote, and
`DESKTOP_VALIDATION.md` §14 has carried an open finding about it since v0.4: parts change files
and then report *"NO CHANGES NEEDED"*, and `kept_the_thread` was false in every two-file run.
Pharos does not have to take a model's word for what a model just did.

- **The run carries its own record of what landed.** The dispatcher already sees every write
  succeed, and both write paths hold the old text and the new one at that moment, so the lines
  a part ADDED are known exactly — no model, no diff of the working tree, no guess. That record
  goes to every later part, under the model's hand-off rather than instead of it.
- **It carries the lines, not just the filenames.** A list of files says what was touched; the
  first few added lines of each say what was decided. On a live run part 2 read the record and
  reported back *"Pattern matched the existing files: LAYER assigned at line 3"* — the
  convention crossed the gap mechanically, which is the thing prose was failing to do.
- **It is not allowed to flatter the scorecard.** `thin_handoffs` still asks whether the MODEL's
  summary named a file its part changed, and `kept_the_thread` still fails when it did not.
  Answering either with Pharos's record would make both true by construction and stop them
  measuring anything. A run where a part reported nothing now reads *"1 part(s) changed files
  and reported almost nothing · Pharos carried 4 changed file(s) forward regardless"* — two
  facts, neither excusing the other.
- **One reserve, not two.** The record and the hand-off share `handoff_reserve`, because that
  is the number every part's ceiling was computed against; adding the record on top of it would
  make each part quietly smaller than the plan promised. The record goes first and takes at
  most half, so the prose is never squeezed out by a long list of filenames.
- **It degrades instead of truncating.** Too tight a budget drops the sample of added lines
  first, then falls back to filenames in shorter framing, then to a count. Every rung says what
  it is showing, and below the last rung it carries nothing rather than something misleading.
- **Repair parts get it too.** Their scope is by definition the files NOT in the record, so
  nothing is withheld — and a sweep that knows the conventions the run settled on stops starting
  blind.
- Config: `handoff_ledger` (on; turn it off to measure what the prose alone achieves). CLI:
  `--no-ledger`.

### Fixed

- **A cut hand-off overran the reserve by the size of its own explanation.** `_cap_handoff`
  trimmed the prose to exactly `handoff_reserve` and then appended forty tokens of marker
  saying it had done so — the precise overrun the function exists to prevent, committed by the
  fix for it. The test covering it asserted `<= 100 + 60  # the marker itself costs a little`,
  so the bug was documented and waved through. Predates v0.6; found because the record is
  measured against the same reserve and the total is now asserted.
- **Blank lines in the record were copied into the code.** Found on the first live run and not
  by a test: an edit that inserted a constant followed by two blank lines put those blanks in
  the record, the next part read them as part of the pattern to match, and reproduced them.
  Two of that run's lint failures were blank-line churn propagated faithfully from one file to
  the next. Blank lines are now counted in the total and kept out of the sample.
- **The sample was too long.** Six added lines per file became three after two live runs: the
  first added line is the change, and what follows it is whatever else the write disturbed —
  on a whole-file rewrite, five lines of reformatted class body carried forward for nothing.

### Known: it propagates conventions, including wrong ones

Measured, first run. Part 1 put `LAYER` above the module's `import`, the record showed it, and
parts 2 and 3 matched — E402 in two files instead of one. That is the feature working: it buys
consistency across parts, not correctness, and the checks are what catch the difference. A run
that is wrong the same way in six files is also easier to fix than one that is wrong six ways,
but nothing here should be read as a claim that the record improves the code.

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
