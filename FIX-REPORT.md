# FIX-REPORT — P124k keepread: answers to REVIEW-RESULT (diff 856af24ab..07f0fe467)

One commit per finding, in review order (73e5666cf, 94568bc49, 3ba3a058b, 74f24e568; this file is the fifth).
Nothing under `serve/` needed changing for any finding, so `engines/qfn/serve/server.py` was not touched and
`serve/qfn_engine.py` / `tools/saguaro/extract_qfn_engine.py` were not used. The DS4 gate stays the follow-up task.

## Finding 1 — HIGH — FIXED (73e5666cf)

The keep checkpoint now fires **by progress, not by run end**, as the review's second option suggests.

* `src/ds4/ds4_main.cpp`: `Prompt` grew `keep_at` (:1476-1478); `prompt_begin` starts it at `from + keep_step`
  (:1662); at the end of each whole run, `checkpoint(0)` is taken where the read first stands at or past `keep_at`,
  which then moves another step on (:1827-1836). The old `took == n` condition is gone: a run that *yielded* leaves
  a whole state at its end too (that is what the next piece continues from), and a run cut short by `feed_unit()` -
  the windows path's 16, `--prefill-chunk 0`, the buffers-do-not-fit and few-experts-in-VRAM fallbacks - no longer
  pays a rings+partials snapshot of its own. A 30K read on a windows card goes from ~1,900 `snap_into` syncs to ~4;
  on the chunk path runs are exactly `keep_step`, so the points land where they did before.
* The run stays capped at `keep_step` (:1754-1757): that is what keeps the deepest keep point at most one step
  behind `run_start`, which is what `restorable(run_start)` weighs on a mid-run stop. Nothing in the stopped
  branch or `read_stop_landing` changed for this fix.
* Why this option and not "skip the checkpoint when `n == feed_unit()`": between-run stops on a windows read land
  on the live `done` - but only while it splits no picture's span (finding 2's shape); when it does, `stored` is
  consulted, and one point every `keep_step` rows beats falling to the base. The cost is the same one the chunk
  path already pays.
* REPORT §2's `prompt_piece` bullet and open q.3 reworded, and their line refs re-pinned, to the new mechanism.

## Finding 2 — LOW — FIXED (94568bc49)

A keep point never stands inside a picture's span, so it cannot be the latest stored point that `read_stop_landing`
rejects while a deeper valid point lies behind it.

* `src/ds4/ds4_marks.hpp`: the landing's span test became the shared **`splits_span(x, spans)`** (:120-126), used by
  `read_stop_landing` itself (:128-144, behaviour identical).
* `src/ds4/ds4_main.cpp`: the read's spans (its pictures + the state's held ones) are built once before
  `prompt_piece`'s loop (:1740-1744) - the stopped branch's own list, which no longer rebuilds it - and the keep
  checkpoint adds `!splits_span(P.done, spans)` to its condition (:1832-1833). `keep_at` is **not** advanced when a
  position is skipped, so the first run end clear of the span takes the point in the step's place.
* `tests/ds4_marks_test.cpp`: `splits_span` boundary checks (:188-193) - inside the rows yes; start, end and an
  empty list no.

## Finding 3 — NIT — FIXED (3ba3a058b)

`scratch/card-test-ds4-keep-read.sh:73-74` (old): `READY` goes to the engine's stdout, which `serve/server.py:1480`
pipes to itself while the engine log carries only stderr (`:1481`) - the grep could never match, so the NOTE printed
even with keepread on. Replaced (:73-85) with what the script can actually see: `SAGUARO_DS4_KEEP_READ` as the engine
gets it - the config copy's `env` first, then the exported one, `child_env`'s order (`serve/server.py:2954-2965`) -
and the NOTE now prints only when that setting says 0. PASS/FAIL stays with the behaviour greps, which are all
engine-stderr lines and did match. `bash -n` clean; the snippet's exit code checked both ways with the sanctioned
venv python.

## Finding 4 — NIT — FIXED (74f24e568)

* `tests/ds4_marks_test.cpp:218`: the stopped-branch case now asks `cv.restorable(16384)` - the run's own start,
  which is the limit the engine passes (`ds4_main.cpp` `m.restorable(mid ? P.run_start : P.done)`) - under its
  "began at 16384" comment; `restorable(20000)` had only coincided because 16384 is that mock's only point.
* `REPORT-p124k-ds4-keep-read.md` §4: a missed cut is retried at the script's real default, `FRACS=".50 .65 .80"`
  (`scratch/card-test-ds4-keep-read.sh:26`), not ".65/.80".

## Finding 5 — NIT (process) — NOT FIXED, on purpose

`70464b721` (steps 2+3 in one commit) is part of the reviewed history: splitting it now means rewriting commits the
review is pinned to (856af24ab..07f0fe467), which the brief's no-history-surgery posture rules out. The fix round
itself commits one finding at a time; the step-merge fact stays recorded in REPORT §4 as it was.

## Files and verification

* Touched: `src/ds4/ds4_main.cpp`, `src/ds4/ds4_marks.hpp`, `tests/ds4_marks_test.cpp`,
  `scratch/card-test-ds4-keep-read.sh`, `REPORT-p124k-ds4-keep-read.md`, this file.
* Run after each C++/test change (one module at a time, as the plan names):
  `g++ -std=c++17 -Wall -Wextra -O1 tests/ds4_marks_test.cpp` → `ds4_marks_test: ok`;
  `ninja -C scratch/build-p124k ds4_marks_test` + run → ok; `ninja -C scratch/build-p124k maya-ds4` → clean link,
  binary still carries `keepread`. `bash -n scratch/card-test-ds4-keep-read.sh` → clean. No GPU run.
* Behaviour OFF stays byte-identical: every new condition is inside `keep_read && keep_step > 0` guards or the
  `keep_read` branch; READY's strings, the OFF-path min and the marks-inside condition are untouched.
