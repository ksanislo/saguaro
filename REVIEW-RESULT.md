# REVIEW-RESULT — P124k DS4 keepread, round 4 (fix verification), diff 856af24ab..db0a2b4c4

**MERGE-WITH-NITS** — round 3's finding 1 is fixed correctly on both halves: the engine says its read's base
(`RESUME%s %zu`, src/ds4/ds4_main.cpp:1664) and the StrataEngine driver parses it into the attribute the continue
line reads (serve/server.py:1848-1865, :2131-2133, :2368-2370, per-call resets :2054/:2282), with tests on the real
driver (serve/test_priority.py:756-785); the honesty rule holds (no line → None → the numberless wording,
serve/server.py:5731-5739; a lost read says `RESUME 0` → logs its own 0). Finding 2 is verified out of this tree
(`tools/card_tests/lost_read.sh`: no `git ls-files tools/` hit, `git merge-base --is-ancestor e5f683c24 HEAD` fails).
The keep itself, re-checked at HEAD, is sound (below). What stands: the diff now contradicts its own plan/report on
`serve/` (finding 1), and the engine's RESUME line is not gated by the keep setting (finding 2).

## Findings by severity

### 1. MEDIUM — the diff changed `serve/` while its plan and its REPORT say it did not; REPORT contradicts HEAD.

PLAN-p124k-ds4-keep-read.md:7 "This task is the ENGINE part only (the server gate comes later, in its own task)";
REPORT-p124k-ds4-keep-read.md:3 "no `serve/` change" and :98 "Not touched: `serve/`" — yet 1a1444ac9 adds 30 lines
to serve/server.py and 30 to serve/test_priority.py. Round 3 (this file at f0629fa17) explicitly routed the fix the
other way ("both fixes belong to the P124u/candidate line ... Fixes 1+2 therefore go to that line/branch"), and
FIX-REPORT.md:39-41 overrides that with the review's *fix constraint*, not its *routing*. What is verified: the
change itself is correct and bounded — the gate stays absent here (`keeps_stopped_read` still the False default,
serve/server.py:1464; the READY parse at :1516-1523 still knows no `keepread`), the `reused` property cannot collide
with QFN (`class QfnEngine(QE.StrataEngine)`, serve/server.py:9477, inherits the generated driver's own class and
its plain attribute, serve/qfn_engine.py:687, :2504 — a different class from serve/server.py:1453's), and it is not
DS4-only: GLM's solo and multi continue lines become figured too (GLM already says RESUME —
serve/fake_solo_parity_engine.py:33, serve/fake_multi_engine.py:62 — and GLM solo is cuttable,
serve/server.py:1523), which matches P124b's intent but was not this task's stated scope. For the manager: the
candidate line owns the same hunk (round 3 read `engine.reused` at saguaro/rc124-cand:serve/server.py:5813), so
this serve/server.py half must be taken as-is or rebased there before the lines merge, or it forks; and the two
serve modules the fix names (`test_lost_read`, `test_priority`) are not plan modules — they were reported green
(3/58 ok, FIX-REPORT.md:90-92) and here only static-read and `py_compile`d (see Commands). Fix the REPORT lines:
drop "no serve/ change"/"Not touched: serve/" or restate them as "the gate, not the RESUME wiring".

### 2. INFO — `RESUME` is said by every read, including with the feature OFF; engine stdout is not byte-identical to base under `SAGUARO_DS4_KEEP_READ=0`.

src/ds4/ds4_main.cpp:1664 sits outside the `keep_read` gate (the settings at :757-765 do not reach it), so the one
stdout line every DS4 read gains is unconditional — the keep's own paths all obey the switch (READY word :766-770,
run cap :1762-1763, marks-inside gate :1774, keep point :1838, stopped branch :1861 — OFF leaves each byte-identical
to 856af24, and the OFF DONE/stderr paths :1888-1899 are base verbatim). No observable server behaviour changes
with the keep off: `resume_note` is only set by the preemption gate (:5935, :5769-5770), which DS4 cannot pass
before the follow-up task, so the parsed figure is inert — and the line is honest off-keep (`from` is that read's
own base, the GLM convention per FIX-REPORT.md:12). Kept as INFO because it is deliberate, but the follow-up gate
must key on the `keepread` word (READY, only while the keep is real), never on the presence of RESUME, and this
finding is the OFF-path exception the brief's "byte-identical" line deserves in the record.

### 3. NIT — stale file:line in the REPORT's own verification lines (post-3da343739 drift).

REPORT-p124k-ds4-keep-read.md:116 names `ds4_main.cpp :2076, :2404` for the stderr request-summary line the card
script greps — at HEAD those lines are :2082 and :2410 (the RESUME block's 6 lines shifted them, and a7132a2d8
quoted the pre-shift numbers). §2's engine refs predate the same commit the same way: ":1754-1757" → :1762-1763,
":1761-1775" → :1767-1781, ":1827-1836" → :1833-1842, ":1846-1894" → :1854-1900 (the ":750-766"/":1474" ones still
land). FIX-REPORT.md's own refs are right at HEAD (:1660-1665, :1883, and serve/server.py :1848-1865/:2054/
:2131-2133/:2368-2370/:5732 all check out).

### 4. NIT — the card script's header still credits the DONE with the PASS figure.

scratch/card-test-ds4-keep-read.sh:4 ("its DONE shows reused > 0") and :13 ("the engine's own DONE must say
`= K reused`") — the DONE goes to stdout, which the server pipes to itself (this was 3ba3a058b's whole point); the
PASS greps are the stderr request-summary line at :166 and the engine-stderr stop lines at :152-157. The wording
a7132a2d8 fixed in the REPORT was not carried into the script's header; the behaviour (grep patterns against
:2082/:2410 and the stopped lines at :1877/:1893) is correct — `bash -n` clean.

### 5. INFO — round 2's standing items, unchanged.

The redundant `checkpoint(0)` at a landing that already stands there (src/ds4/ds4_main.cpp:1876, the `live` case)
and the double snapshot at a cut ≥ `keep_at` stay as round 2 left them ("bounded, cost-only"). `atoi`/`atoll`
garbage reads as off (:757-765, safe and consistent). The first `keep_step` of a fresh read stays unprotected
(reused=0, retried by the script's FRACS), the read's shape changed for ordinary reads while the keep is on
(runs ≤ step, marks-inside shortcut out while it splits — A/B via the script's `SAGUARO_DS4_KEEP_READ=0` phase),
pin-after-resume and reader-path parity remain the report's open questions (:123-142).

## The keep at HEAD, re-verified (file:line)

* **Audit (§1) holds at 856af24ab** — sampled every §1 ref: the between-pieces STOP and loop top
  (src/ds4/ds4_main.cpp:1717-1721), per-token/per-rid STOPs (:1929-1933, :2704-2707, `Lines::stopped` :175-179,
  `STOP <rid>` :215-218), the discard to checkpoint 1/reset with `reused = back ? from : 0` (:1805-1818),
  checkpoint 1 (:1637), READY's `stop`-only (:750-753); the `feed_marks`/`marks_inside` contract
  (src/ds4/ds4_model.hpp:181-188), the set_cancel contract "the state past the prompt's start is then not usable"
  (:263-267) and set_yield keeping its chunks (:272-275); `Check`/`snap_into` (src/ds4/ds4_model.cu:1694-1700,
  :11158), the 50 ms cancel poll and the halted return (≈:22174-22183, :22224-22230), `fed_n = n` on a cancelled
  feed (:24352), `restorable` = `restore`'s own `best_check` (:11240-11243 vs :11275-11277 — so a `restore(land)`
  can only miss `pos == land` through the guard's own check, src/ds4/ds4_main.cpp:1866).
* **Landing** (src/ds4/ds4_marks.hpp:122-143 + branch src/ds4/ds4_main.cpp:1854-1900): `mid = !ok` is exact
  (`stopped` implies `ok`: the loop-top check is the only setter inside `while (ok …)`); `stored` is asked at
  `run_start` mid-run (:1863, so nothing a cancelled run consumed is ever trusted) and every keep point lies at or
  before the next run's start (:1838-1841 runs after `P.done += took`); a between-runs stop keeps the live end
  (:1865, guarded by `pos == land`); spans (built :1748-1750, the read's plus `held`'s — `held`'s matter for a
  landing below `from`) are rejected at the keep point and at the landing on the one `splits_span`; `restore()`
  drops the points past the landing (src/ds4/ds4_model.cu:11297-11300), so a cancelled run's committed inner marks
  cannot outlive the cut; the `held` re-hang mirrors `prompt_end`'s (:1869-1874 vs :1925-1931, `pic_same`'s
  exclusion safe since `held` entries end at or before `from ≤ keep`); checkpoint 0 is the prompt-end slot, and
  `prompt_begin` has already parked the base in slot 1 (:1659), so the overwrite is consistent; a rejected or
  failed landing falls through to the base path (:1888-1899) — fresh conversation cut inside its first run keeps
  nothing, exactly as before; DONE's format is the base stop-line's (:1883 vs base :1816) and it returns false on
  the same contract (:1885).
* **Server wiring** — the property is thread-local with `multi` (serve/server.py:1856-1865) and `_sliced` reads it
  on the request's own thread (:5732); the multi route registers before its GEN (:2275-2289, so a routed
  `RESUME <rid> <n>` cannot miss, :1693-1706 requires `len(f) >= 3`); a non-parsing figure is None (:2133, :2369,
  `f = line.split()` already precedes the multi chain at :2338); resets per call (:2054 solo, :2282 after `_tls` is
  in use at :2278-2279); the new branches yield nothing and change no queue/routing flow.
* **Locks/threads/stdout** — no new locks: `_routes_lock` usage is pre-existing and unnested, the setter is a plain
  attribute/thread-local store under the GIL, and the engine's RESUME is one `printf`+`fflush` like every tagged
  line it already writes (:1664-1665; solo serialises, multi tags per rid).
* **Generated file** — the range touches neither serve/qfn_engine.py nor engines/qfn/serve/server.py, and the
  extractor runs qfn→qfn_engine only (serve/qfn_engine.py:1-4), so no extraction was owed by the serve/server.py
  change.
* **Tests** — ds4_marks_test:188-225 exercises the helpers (span boundaries, all four landing candidates, mid-run
  never the consumed count, base-inside-span → 0) and the branch shape against the mock `Conv.restorable`
  (:67-70); TheResumeLine drives the real StrataEngine against the parity fakes with a per-call figure change
  (serve/test_priority.py:756-785). Engine glue (keep_at progress, span skip, landing application) is card-script
  territory, as the plan framed it.
* **Hygiene** — no Co-Authored-By/Claude-Session/Assisted-by in any message (`git log --format=%B` grep: empty);
  the added lines carry no in-repo-violating paths (only the sanctioned `/opt/llm/saguaro/.venv/bin/python` at the
  script's :22) and no secrets (the script *drops* `api_key`, scratch/card-test-ds4-keep-read.sh:58); `.gitignore`
  does ignore `scratch/` (:20, the force-add claim) and the common dir's `info/exclude` does hide
  REVIEW-RESULT.md (:42).

## Commands run for this round (one at a time, the plan's module only)

* `g++ -std=c++17 -Wall -Wextra -O1 tests/ds4_marks_test.cpp -o scratch/review/ds4_marks_test` + run →
  `ds4_marks_test: ok`, no warnings.
* `ninja -C scratch/build-p124k maya-ds4 ds4_marks_test` → EXIT 0 (cache reads Release,
  /usr/local/cuda-13.1/bin/nvcc, arch 75 — the plan's step-4 build dir); `scratch/build-p124k/ds4_marks_test` ok;
  `strings scratch/build-p124k/saguaro-ds4` carries `RESUME%s %zu` and ` keepread`. No GPU run.
* `bash -n scratch/card-test-ds4-keep-read.sh` clean; the script itself not run (needs the card).
* `py_compile` of serve/server.py and serve/test_priority.py only — the two serve test modules the fix names are
  not the plan's module, so they were read, not run (the constraint bound this round); `git status` clean.

Only REVIEW-RESULT.md was written; nothing else edited or committed.
