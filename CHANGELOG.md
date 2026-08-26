# Changelog

Every version is a tier, and each one is allowed to do strictly more than the last. The proxy
itself has not changed since v0.1 and is not going to: it observes, it never mutates.

Numbers quoted here were measured on the machine described in `DESKTOP_VALIDATION.md` — an
RTX 3060 12 GB running Ollama — and the record of how is in that file rather than this one.

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
- **635 tests**, `ruff` and `mypy --strict` clean.

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
