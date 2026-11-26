# P124k — DeepSeek-V4.1-Flash keeps a stopped prompt read (ENGINE PART ONLY; Saguaro 1.2.4, audit item 4)

ref/REPORT-p124c-prio-audit.md §2 and §8.4. This is the engine part only: no `serve/` change, `serve/qfn_engine.py`
untouched (nothing in `engines/qfn/serve/server.py` needed changing, so the extractor was not run). The server gate
(`keeps_stopped_read` for a DS4 that says `keepread`) is the follow-up task, as the plan says.

## 1. The audit verified (file:line at 856af24ab, this worktree's start)

The audit's claims hold. `src/ds4/ds4_main.cpp`:

* **A STOP ends the read between pieces** — `prompt_piece`'s loop top (:1717-1721: `while (ok && P.done < total) {
  if (stop_now()) { stopped = true; break; } ...`), STOP per token in decode solo :1929-1933 and per rid in
  `--parallel` :2704-2707 (`Lines::stopped(rid)` :175-179, `STOP <rid>` parsing :215-218). READY offers `stop` only,
  never `pause` (:750-753).
* **The read is then discarded** — :1805-1818: `if (stopped || (!ok && m.cancelled()))` → `m.restore(from)` back to
  the state **before the prompt** (checkpoint 1, taken at :1637 `if (from > 0 && from < r.ids.size())
  m.checkpoint(1)`), else `m.reset()` + `held.clear()`, and DONE carries `reused = back ? from : 0` (:1816). Nothing
  of the part read survives the cut.
* **The marks/feed machinery stores states at cut points** — GEN's `marks=`/`shared=` keys (`Req` parse :529-533),
  planned per read by `mark_plan` (:1655-1685, `ds4_marks.hpp`); the read runs each stretch to its cut and keeps the
  state there (`upto` :1723-1726, `m.mark(sh)` at the point :1784-1793, or `feed_marks`' inner points :1727-1743 —
  `Model::feed_marks`, `ds4_model.hpp:181-188`). Model side (`ds4_model.cu`): a checkpoint is the window rings + the
  compressors' partial groups at a position, host buffers (`Check`, `snap_into` :1694-1700, :11158-11183); marks are
  the same, LRU under `SAGUARO_DS4_RESTORE_POINTS` (`mark_room` :11206-11230); `restore(limit)` reloads the latest
  point at or before it and truncates tokens/points past it (:11275-11305); `restorable(limit)` is the latest point
  restore would take (:11235-11243). Every stored point is at or before `pos` and holds the conversation's own tokens
  (:1705-1708) - that invariant is what makes a mid-read landing safe.
* **A run stopped inside is only trustworthy at stored points** — `Model::set_cancel`'s contract (`ds4_model.hpp`
  :263-267): "one GPU [stops] after the layer it is on, several at a chunk all of them reach - ... The state past the
  prompt's start is then not usable: restore() a checkpoint kept before the feed, or reset()". `forward_chunks` polls
  the cancel every 50 ms (:22174-22183) and a halted run returns false at :22224-22230 without committing `fed_n`,
  positions or tokens; only a *yield* (`set_yield`, hpp :272-275) keeps its chunks read. A cancelled feed also leaves
  `Model::fed()` at the full n (:24352), which is why the kept position must never be the stopped run's consumed
  count.
* **The server never cuts a DS4 read today** — the read-preemption gate `serve/server.py:5732-5742` takes the cut
  only for `can_pause` (READY `pause` - GLM) with `multi`, or `keeps_stopped_read` (QFN solo); DS4's StrataEngine
  default is `keeps_stopped_read = False` (:1464) and READY has no `pause`, so its read runs to its end. The drain
  comment at :2125-2137 is the audit's: STOP is seen between chunks, "a 32K chunk deep in a long context takes over a
  minute". The READY parser passes unknown words (`"stop" in f[2:]`, :1516-1523), so a new READY token is safe.

## 2. The keep (what changed)

On a stopped read the conversation stays **live at the deepest state the read already stores**, and the DONE says it
as its `reused`, so the next request of the same prompt shares it and reads only the rest - exactly as a finished
turn's conversation is shared (`prompt_begin`'s in-place continuation at the state's end, :1518-1521 pre-change).
One change, in `prompt_piece`, so solo and `--parallel` both get it; the DONE keeps the per-rid tag in multi.

* `src/ds4/ds4_marks.hpp` - **`read_stop_landing(mid, done, stored, from, spans)`** (pure, :120-142): first of the
  live piece end `done` (only asked for between runs - `mid` false; a Stop inside a run leaves everything past that
  run's start unusable, and a cancelled feed's `fed()` is the full n, never what it consumed), the deepest restore
  point `stored` at or before the stopped run's own start (the read's last mark, or the keep checkpoint below it),
  and `from` (checkpoint 1 of a continued prompt); 0 for nothing of its own. A landing inside a picture's span is
  not kept - a span is stored whole or not at all (the model's `Check` holds no partial rows) - so a landing that
  splits one of `spans` gives way to the next candidate.
* `src/ds4/ds4_main.cpp`:
  - `serve()` :750-766 - the keep's two settings: `SAGUARO_DS4_KEEP_READ` (default on; 0: no keep, and READY says no
    `keepread`) and `SAGUARO_DS4_KEEP_READ_STEP` (default 8192 tokens; 0: the read takes no keep points of its own
    and keeps only the request's marks and its base).
  - `prompt_piece` - a run is capped at `keep_step` tokens (:1745-1747; `feed`'s `after` arg still tells the model of
    the prompt up to the next cut), each whole run's end stands as checkpoint 0 (:1817-1819 - the conversation's own
    slot, no mark-LRU pressure; the old prompt's end it replaces is held as checkpoint 1 by `prompt_begin` :1637),
    and the marks-inside one-run shortcut (:1727-1743) is not taken while the keep splits (:1758) - a run stopped
    inside keeps nothing of the points it was carrying. `Prompt::run_start` (:1474) records where each run began.
  - the stopped branch (:1831-1880): `mid = !ok` (stopped inside a run) → `stored = m.restorable(run_start)`, else
    the live end at `done`; land via `read_stop_landing` (restore only where the state does not already stand there),
    re-hang the pictures the kept state really holds (`held` as `prompt_end` re-hangs it, at `keep`), `checkpoint(0)`
    at the kept position, and DONE with `reused = keep`. Where there is no landing (nothing stored - a fresh
    conversation cut inside its very first run, or a prompt whose base lies inside a picture's span) the branch
    falls through to the old behaviour: back to before the prompt, or reset.
  - **READY** (:766-770): `READY <ctx> stop [multi=n] keepread` - only when `keep_read` is on, so the server can gate
    on truth; the server-side reading of the word is the follow-up task's.

Design notes: the keep checkpoint reuses checkpoint slot 0 (nothing new allocated, slots unchanged, a parked slot
carries the keep point automatically since a slot saves every checkpoint); the cancelled-run case always calls
`restore()` even when `pos()` already stands at the point, because a halt can end mid-chunk on a single GPU and may
leave dirty rows and compressor partials past the run's start; a mid-run STOP is seen by `forward_chunks` within its
50 ms poll and one chunk's tail, so the drain stays bounded as the audit expects (§8.4 "Drain per chunk fits the
~1 s goal"). The read's shape changes for ordinary reads too (runs ≤ 8192 rows instead of one run to the first
requested cut, no marks-inside shortcut): the multi/reader paths already read in pieces and `DS4_CHECK_READER`
compares piecewise against whole reads bit for bit.

## 3. Files

* `src/ds4/ds4_marks.hpp` - `read_stop_landing` (pure helper).
* `src/ds4/ds4_main.cpp` - the keep: settings + READY word, run cap, keep checkpoint, `run_start`, stopped branch.
* `tests/ds4_marks_test.cpp` - the helper's checks + the stopped-branch shape as a case; `Conv` grew a
  `restorable()` beside its `restore()`.
* `scratch/card-test-ds4-keep-read.sh` - the manager's card check (force-added; `scratch/` is ignored, as P124b's
  card script was).
* Not touched: `serve/` (gate comes in its own task), `serve/qfn_engine.py` (no server change → no extractor run),
  `ds4_model.cu/.hpp` (the keep builds entirely on the existing checkpoint/mark/restore API).

## 4. Tests run and results

* `ds4_marks_test` (the module the plan names), standalone:
  `g++ -std=c++17 -Wall -Wextra -O1 tests/ds4_marks_test.cpp && run` → `ds4_marks_test: ok` (no warnings).
* The same module as a build target of the new CUDA build:
  `cmake -B scratch/build-p124k -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
  -DCMAKE_CUDA_ARCHITECTURES=75 -DMAYA_ENABLE_CUDA=ON -DMAYA_BUILD_TESTS=ON` + `ninja maya-ds4 ds4_marks_test` →
  build `EXIT 0` (107/107, nvcc 13.1.115, `scratch/build-p124k/saguaro-ds4` linked; nothing run on a GPU);
  `scratch/build-p124k/ds4_marks_test` → `ds4_marks_test: ok`. The engine binary carries the new READY
  (`READY %lld stop%s` + `keepread`) and the keep's lines (`strings`).
* `scratch/card-test-ds4-keep-read.sh` - written, `bash -n` clean, **not run here** (no GPU runs and no card model
  on this box): a test copy of the card's DeepSeek config on :8092 with its own session/slot dirs, its own engine
  log and no api key; a fresh ~30K control read sets the cut timing; a second fresh ~30K read is cut mid-way by a
  client disconnect (the server drains to the engine's "stopped while reading" line); the same request asked again
  must show the engine's own `prompt N tokens = K reused` with 0 < K < N on its DONE → PASS (a cut that fell before
  the first keep point keeps nothing - honest reused=0 - and is retried at .65/.80 of the control read).
* Commits: steps 2+3 (keep + READY), a step-2 wording nit, step 4 (tests + card script), and this REPORT
  (force-added: `REPORT-*.md` is ignored in this worktree, as P124h did). Step 1 changed no files.

## 5. Open questions

1. **The server gate (the follow-up task)**: `keeps_stopped_read` for the DS4 when READY says `keepread` - both solo
   and `multi=n` - and whether the gate keys on the word or the exe name (P124h's open question, same shape).
2. **A cut before the first keep point keeps nothing** (reused=0, as today): the first `keep_step` tokens of a fresh
   conversation are unprotected, and `SAGUARO_DS4_KEEP_READ_STEP` is one size for every card. Should a low-priority
   read start with a shorter first step (or should the server only cut past the first step)? A card run of the script
   with default settings will show where the cuts actually land.
3. **`keep_step` = 8192 default**: each whole run's end now costs one `checkpoint(0)` copy (rings + compressor
   partials, ~`check_bytes()` MB of host traffic). On the 30K control reads that is noise, but the read shape also
   left the marks-inside shortcut (§8.4's cheap-points optimization) whenever the keep splits - measurable against
   `SAGUARO_DS4_KEEP_READ=0` with the card script's control phase.
4. **Anchor/pin after a resumed read**: a prompt that began at 0, was kept mid-read and finished on the resumed
   request ends `continued=true`, so `prompt_end` never pins the anchor - before P124k the discarded-rerun path
   pinned on the completed rerun. Graceful (the anchor is one checkpoint among the points `restore()` weighs),
   but a re-rendered-history conversation loses that point; worth a small follow-up if the logs show it mattering.
5. **DS4_READER background reads** take the same keep (shared `prompt_piece`); `DS4_CHECK_READER`'s bit-parity
   between an interrupted-then-resumed read and a whole one is not covered by any test here - a card run with
   `--parallel` + `DS4_READER=1` would close it.
