# REVIEW-RESULT — P124k DS4 keepread (engine part), round 2, diff 856af24ab..d2a96ae46

**MERGE-WITH-NITS** — the round-1 HIGH (finding 1) is fixed and verified: the keep point is taken by
progress (`keep_at`), clear of picture spans, and the whole range builds and tests green here. The
keep's landing logic, the OFF path, multi, locks/stdout and the file discipline all check out.
What is left: two redundant-snapshot nits, two wording nits in the docs, and the accepted infos.

## Findings by severity

### 1. NIT — the landing's `checkpoint(0)` re-snapshots a point that already stands there.
`src/ds4/ds4_main.cpp:1870` takes `m.checkpoint(0)` unconditionally when the branch keeps. Between
runs the live landing can be the keep point the last run just left (`ckpt` already at `keep`), and a
mid-run landing onto `ck[0]` itself re-snaps the same state after `restore()` reloaded it
(`src/ds4/ds4_model.cu:11275-11305` keeps the point it restored from). One extra `snap_into`
(`check_bytes()` MB of D2H + stream syncs, `src/ds4/ds4_model.cu:11161-11183`) per stopped read —
bounded, stop-path-only, no correctness impact. A cheap guard would be `m.checkpoint(0)` only when
the restored/live point is not already `ck[0]`'s position; not worth API churn.

### 2. NIT — at a cut position ≥ `keep_at` the same state is snapshotted twice.
`m.mark(sh)` at `src/ds4/ds4_main.cpp:1819` stores the point in `mk`, and the keep block at
`:1832-1836` then stores the identical state again in `ck[0]` (`mark_room` dedups only inside `mk`,
`src/ds4/ds4_model.cu:11206-11213`). One extra snapshot per requested cut that crosses a step
boundary. Cost-only, and it buys the `restorable()` shortcut for spans-dodged cases; noted only.

### 3. NIT — REPORT §4 says the resumed check reads the DONE; the script reads the stderr summary.
`REPORT-p124k-ds4-keep-read.md:114-116`: "must show the engine's own `prompt N tokens = K reused`
with 0 < K < N **on its DONE**". DONE goes to the engine's stdout (server log); the script greps the
engine's stderr request-summary line `prompt %zu tokens = %zu reused + ...`
(`src/ds4/ds4_main.cpp:2076`, `:2404`; greps at `scratch/card-test-ds4-keep-read.sh:129,166`). Same
number (`from` of the resumed request), right decision — the sentence just names the wrong line,
the class of error finding 3 was about. One-word fix in a follow-up doc touch.

### 4. NIT — REPORT §4:118-119 says `REPORT-*.md` "is ignored in this worktree": `.gitignore` has
no such pattern (only `scratch/` at line 20, `*.log`); the REPORT was added plainly. The
force-add reality applies to `REVIEW-RESULT.md` (main checkout's `.git/info/exclude:42`), not to it.

### 5. INFO — round-1 finding 5 (steps 2+3 in one commit, `70464b721`) stands unfixed by design
(`FIX-REPORT.md` §Finding 5): rewriting reviewed history. Accepted as recorded; future steps commit
per numbered item.

### 6. INFO — default-on before the gate: the run cap and the lost marks-inside shortcut cost every
*uninterrupted* read too (`:1754-1757`, `:1768`), while nothing can cut a DS4 read until the gate
task (`serve/server.py:5740-5743` with `keeps_stopped_read=False`, `:1464`). Disclosed as open q.3;
the cap is load-bearing (a mid-run stop falls back at most one step, `:1754-1755` comment) and the
card script measures it (`SAGUARO_DS4_KEEP_READ=0` A/B, `scratch/card-test-ds4-keep-read.sh:17-18`).
If the card A/B shows a chunk-path cost that matters, flipping the default until the gate lands is
the one-line answer — the READY word already follows the setting (`:757-766`).

## Round-1 findings — fix verification (the fix commits, one per finding)

- **Finding 1 (HIGH) — FIXED (73e5666cf), verified.** The checkpoint is by progress, not run end:
  `Prompt::keep_at` (`src/ds4/ds4_main.cpp:1476-1478`), started at `from + keep_step` (`:1662`), and
  `:1832-1836` takes `checkpoint(0)` where the read first stands at or past it, then steps it on —
  the `took == n` condition is gone, so a windows-path run of 16 no longer pays a snapshot
  (`feed_unit()`'s 16: the path a Stop can never fall inside anyway — `feed_windows` polls no cancel,
  only `forward_chunks` does, `src/ds4/ds4_model.cu:22174-22183`). A run that *yielded* is a whole
  state (`set_yield` keeps its chunks read, `src/ds4/ds4_model.hpp:272-275`), so checkpointing at a
  yield's end is valid. The run cap stays (`:1756-1757`) — that is what bounds the mid-run fallback
  to one step behind `run_start`.
- **Finding 2 (LOW) — FIXED (94568bc49), verified.** `splits_span` is shared
  (`src/ds4/ds4_marks.hpp:120-126`), used by `read_stop_landing` itself (`:138-144`) and by the keep
  block (`src/ds4/ds4_main.cpp:1832-1833`); the spans are built once before the loop from the read's
  pictures and the state's held ones (`:1740-1744`) — the same list the landing weighs (`:1858`), so
  a point skipped for splitting a span does not hide the deeper ones. `keep_at` not advanced on a
  skip (`:1835` only runs with the checkpoint), so the first clear run end takes the point.
  Boundary tests at `tests/ds4_marks_test.cpp:190-193`.
- **Finding 3 (NIT) — FIXED (3ba3a058b), verified.** The READY grep is gone; the script now reports
  `SAGUARO_DS4_KEEP_READ` as the engine really gets it — config copy's `env` first, the exported one
  as fallback (`scratch/card-test-ds4-keep-read.sh:77-85`), which matches `child_env` precedence
  (`serve/server.py:2954-2966`: `dict(os.environ)` then config wins). PASS/FAIL rests on the
  behaviour greps, all engine-stderr lines (`:153,157,170` — `stopped while reading the prompt`
  `src/ds4/ds4_main.cpp:1887`, `kept live at N tokens` `:1871-1873`).
- **Finding 4 (NIT) — FIXED (74f24e568), verified.** `tests/ds4_marks_test.cpp:218` asks
  `cv.restorable(16384)` at the run's own start, as the engine does
  (`m.restorable(mid ? P.run_start : P.done)`, `src/ds4/ds4_main.cpp:1857`); REPORT §4 names the real
  `FRACS` default (`.50 .65 .80`, script `:26`).

## Checked and good (with evidence)

- **Landing correctness.** Mid-run stops never ask for the stopped run's consumed count
  (`src/ds4/ds4_marks.hpp:140`, `mid ? -1 : done`) — right per `set_cancel`'s contract
  (`src/ds4/ds4_model.hpp:263-265`) and `fed()` standing at the full n on a cancelled feed
  (`src/ds4/ds4_model.cu:24352` vs the `fed_n = took` only on success, `:22234`); a cancelled run
  commits no tokens and no `inner` marks (`:22224-22230` returns before `:22392-22395`). The live
  shortcut is taken only between runs and verified against `m.pos()` (`src/ds4/ds4_main.cpp:1859-1860`);
  a mid-run landing always goes through `restore()`, which reloads, rewinds `pos` and drops the
  points past it (`src/ds4/ds4_model.cu:11295-11303`), so a chunk-halt's dirty rows cannot survive. Every candidate is a stored point ≤ `pos` holding the
  conversation's own tokens (`latest_point`'s `≤ limit && ≤ pos`, `src/ds4/ds4_marks.hpp:24-29`;
  invariant `src/ds4/ds4_model.cu:1705-1708`), and after `prompt_begin` all stored points agree with
  this prompt's prefix (reset/restore/slot paths: `src/ds4/ds4_main.cpp:1540-1592`, restore clears
  points past pos). No landing → fall-through to the pre-change branch, byte-for-byte the old
  behaviour (`:1882-1894`). `held` re-hang mirrors `prompt_end`'s at `keep` (`:1862-1869` vs
  `:1919-1925`); held spans are all ≤ `from`, so `≤ keep` matches what the kept state really holds.
- **Feature OFF unchanged.** With `SAGUARO_DS4_KEEP_READ=0`: READY strings identical to base
  (`:767-770` vs `856af24ab:750-752`), the run-cap min gains `(size_t)-1` (`:1756-1757`), the
  marks-inside condition reduces to the old one (`:1768`), the keep block and the stopped-keep
  branch are `keep_read`-guarded; `keep_at`/`run_start` are inert writes. With keep ON and
  `SAGUARO_DS4_KEEP_READ_STEP=0` the read shape is the old one too (every new condition keys on
  `keep_step > 0`) while the branch keeps off marks/base — consistent with READY saying `keepread`.
  The OFF path emits no new line anywhere.
- **--parallel / solo / reader thread.** One change in `prompt_piece` covers all
  (`:1931` solo, `:2439` multi with the per-rid tag), checkpoints/marks travel with their
  conversation on bind (`ck`/`mk` moved in park/unpark, `src/ds4/ds4_model.cu:11087-11102`); the new
  model calls (`checkpoint(0)` between runs, `restorable`/`restore` in the branch) sit exactly where
  `mark()`/`feed()` already sat — inside the reader's exclusive sections between a run's start and
  end (`src/ds4/ds4_main.cpp:2262`), the serve loop gated out (DS4_READER gate, `:2153`).
- **Locks/threads/stdout.** No new lock, no nested acquire: `stop_now` → `Lines::stopped` takes only
  `Lines::mu` (`src/ds4/ds4_main.cpp:175-179`), held nowhere across a model call; the branch prints
  one DONE with the per-rid tag, `end_note()` before it, `fflush` after — the other DONE paths'
  pattern (`:1876-1878` vs `:1889-1892`); field layout unchanged (`reused` stays f[8] of the 11-field
  DONE, parsed at `serve/server.py:1852`).
- **Generated file discipline.** The whole range touches only `src/ds4/ds4_main.cpp`,
  `src/ds4/ds4_marks.hpp`, `tests/ds4_marks_test.cpp`, the REPORT, the FIX-REPORT and the scratch
  script; `serve/`, `engines/qfn/serve/server.py` and `serve/qfn_engine.py` are untouched
  (`git diff --stat`), so no `tools/saguaro/extract_qfn_engine.py` run was owed, as the reports say.
  `keepread` is inert to today's server (READY words pass through, `serve/server.py:1515-1523`) and
  the DS4 gate stays shut until the follow-up task.
- **Tests really exercise the change.** `read_stop_landing` covers live/stored/base/mid/span cases
  (`tests/ds4_marks_test.cpp:197-210`) and `splits_span` its boundaries (`:190-193`); the
  stopped-branch case replays the engine's own calls on the Conv mock, whose `restorable` mirrors
  `best_check` (`:67-71`) — kept point, continue in place, read only the rest (`:211-226`).
- **Hygiene.** Script writes only under `scratch/ds4-keep-read/`, strips `api_key`, owns its
  session/slot dirs and engine log (`scratch/card-test-ds4-keep-read.sh:43-60`); the only external
  path is the sanctioned `/opt/llm/saguaro/.venv/bin/python` (same as the base's card scripts); no
  secrets in the diff; the ten commits carry no `Co-Authored-By`/`Claude-Session`/`Assisted-by`
  trailers; `SAGUARO_DS4_KEEP_READ` parses like the existing `*_AUTO` settings (garbage reads as off
  — the conservative direction, READY follows).

## Commands run for this round (one at a time, as the plan names)

- `g++ -std=c++17 -Wall -Wextra -O1 tests/ds4_marks_test.cpp` (output into `scratch/`) + run →
  `ds4_marks_test: ok`, no warnings.
- `ninja -C scratch/build-p124k maya-ds4 ds4_marks_test` → EXIT 0 (the build regenerated its build-id
  and recompiled `ds4_main.cpp` warning-free, relinked `saguaro-ds4`; cache confirms
  `/usr/local/cuda-13.1/bin/nvcc`, arch 75); `./scratch/build-p124k/ds4_marks_test` → ok; `strings`
  on the engine shows `READY %lld stop%s` / ` multi=%d%s`, `keepread`, both env names.
- `bash -n scratch/card-test-ds4-keep-read.sh` → clean. No GPU run; no serve test module beyond the
  plan's named one was run (none is affected: `serve/` untouched).

Only REVIEW-RESULT.md was written (force-added: it sits in the main checkout's info/exclude);
nothing else edited or committed; the tree was clean before and is clean after.
