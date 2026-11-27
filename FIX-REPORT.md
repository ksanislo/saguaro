# FIX-REPORT - P125j layer pause (fixes for REVIEW-RESULT.md, redo #2)

Every finding fixed, one commit per finding; the tree builds (`strata_prefill` and `strata` exit 0 in
`scratch/build-p125j-cuda75`), `ctest -R layer_pause_test` passes, and all 68 tests of
`serve/test_priority.py` pass. Commit range: `1c79c06e..HEAD` (11 commits, oldest first below).

## Opening complaint - "steps 2-4 sit uncommitted in the worktree"

Fixed: `56755db44` commits the steps 2-4 worktree as found (message says plainly it does not build yet),
then each finding gets its own commit as the plan asks. The step-3/step-4 deliverables that simply did not
exist are written here, not folded into the checkpoint.

## BLOCKER 1 - the tree does not compile (`lp_resume` used before declared) - FIXED (`8248c2a81`)

The resume decision (flag, `l_from`, and the rows' H2D copy) moved to the chunk's top, right above the
C-4 gather that consumes the flag, in `engines/qfn/src/prefill/prefill.cpp`; the old block at the former
:3572 deleted. While moving it I also changed the copy: it goes up with `cudaMemcpyAsync` on the compute
stream followed by a stream sync, so at a `c0 > 0` resume a previous chunk's still-queued layer kernels
finish before the rows overwrite the `m.R` they read (the old synchronous default-stream copy did not
serialize with `m.cs`'s non-blocking queue). Reproduced green on this box:
`cmake --build scratch/build-p125j-cuda75 --target strata_prefill` exit 0, and `--target strata` (the
downstream `generate.cpp` the review says was never reached) exit 0 (`scratch/build-strata-fixed.log`,
final re-check `scratch/build-final.log`). The REPORT's "build exit 0" claim stays false in history;
this report and the build logs are the correction.

## MAJOR 2 - step 4 deliverables missing - FIXED (two commits)

- `05b3b5ee3`: `engines/qfn/src/program/layer_pause_test.cpp` + its `add_executable`/`add_test` entry in
  `engines/qfn/CMakeLists.txt` (inside the existing `STRATA_BUILD_TESTS` block, beside `fast_lane_test`).
  41 hand-computed checks over the three pure helpers: `pause_point` (first/last pass bounds, short last
  chunk keeps `n - c0`, last pass and one-layer stage never pause, `[LB, LE)` stage bounds, degenerate
  runs), `resumes` (whole-chunk check at the run's own `c0`, the cross-run continuation at `c0 = 0` whose
  saved label is stale, a relayout's different `C` refusing, a restored pause's missing label taking no
  part), `state_bytes` (320 MiB for an 8192-token chunk, zero/negative, saturation exactly one over 2^64).
  Run here: the scratch build was reconfigured `-DSTRATA_BUILD_TESTS=ON`, ONLY the `layer_pause_test`
  target built, `ctest -R '^layer_pause_test$'` passes.
- `64ee73f9e`: `tools/card_tests/layer_pause.sh` (port 8092, own `--out` dirs, api_key dropped, engine log
  inside the out folder - same shape as `lost_read.sh`). A ~60K low read; `CUTS` (default 2) short normal
  requests at random read positions; the four PASS conditions: (1) the cutting request's first token within
  `FIRST_TOKEN_S` (1 s) for every cut that landed between layers, (2) the read resumes with the engine's own
  `reused` figure, M > 0, and the engine logged the between-layers pause (the server line naming chunk and
  layer is also required), (3) the engine's per-leg `reused + read` figures sum to the prompt - nothing read
  twice - and the last leg ends at `reused + read = prompt`, (4) after restarting the server on an empty
  session dir, the SAME prompt uninterrupted answers identically (temp 0, 64 tokens). `bash -n` clean,
  executable. NOT run: the plan excludes GPU runs here and this box has no card for one.

## MAJOR 3 - step 3 incomplete, REPORT claims false - FIXED (`4baa3570c`)

- Grace: `serve/server.py` `priority_idle_seconds()` default 5.0 -> 0.0 with the docstring and the inline
  history comment updated. The review's caution ("grace -> 0 only with a working pause") is met the way the
  MANAGER framed it: the BLOCKER is fixed and compiled, the helpers are unit-tested on the repo side, and
  `SAGUARO_PRIORITY_IDLE_S` still overrides. The card proof remains `layer_pause.sh`; the grader/card run
  should run it before trusting the new default.
- Pause line: `_sliced` (serve/server.py ~5875) now prints, when this leg's DONE carried the new keys
  (`fresh` taken after `gen.close()` drained to DONE, guarded against a stale `last0`),
  `prompt read paused at N of M prompt tokens, inside chunk c at layer l (the chunk from token P)`. With an
  engine that reports no keys the line is byte-identical to the old one - the 12+ `prompt read paused`
  assertions in `test_priority.py` and `test_lost_read.py`'s un-anchored regex still match.
- `docs/SETTINGS.md:238` and `:250` now say 0 and why; the engine's `SAGUARO_LAYER_PAUSE` switch (introduced
  by step 2, previously undocumented) got its row beside `SAGUARO_SLOT_KEEP`.
- Its test: `serve/test_priority.py` `IdleGraceDefault` now asserts `0.0` (renamed accordingly). Run: all
  68 tests pass (`python -m unittest serve.test_priority`, 86 s).
- Generated file: `serve/qfn_engine.py` regenerated with `tools/saguaro/extract_qfn_engine.py` (never
  hand-edited); the diff is exactly the new DONE-key tuple at :1195, so the running server parses
  `paused_layer/paused_chunk/paused_at` too. The `self.last["paused_*"]` consumer is the pause line above.

## MAJOR 4 - a parked paused read could corrupt a prefix borrower - FIXED (`0b1434d01`)

Two guards in `engines/qfn/src/program/generate.cpp`, plus one hole of the same family I found while
implementing it:

- Borrow decision (`:19704`): if the parked best match is a paused image and `parked.tokens` (the borrow
  depth) is past its `pause_pos`, the borrow is refused with a log line and the image goes the whole-adopt
  way; borrows up to `pause_pos` stay allowed - below the pause every layer wrote, and the donor keeps its
  pause record and rows untouched. (This is the review's "cap a donor's borrow depth to its pause_pos",
  taken as refuse-deeper rather than clamp-depth, because a clamped borrow would mount no checkpoint and
  the whole-adopt fallback keeps the borrower on a correct path.)
- Whole-adopt (`incoming` restore, ~:20100): the adopted paused image keeps its pause only when the
  adoption lands exactly at the paused chunk's END - the continuation and the diverge-after-chunk cases,
  which `go_on` then drives. A prompt branching INSIDE the paused chunk resumes from `pause_pos` instead
  (`resume = pause_pos`, so the chunk is re-read over all layers) and the pause is not armed for it.
- Same family: on that clamp I also cap `live`/`live_imgs`/`checkpoints` to `pause_pos`. Without the cap,
  a branch ending before the paused chunk's end left `live` (which is what a later park snapshots and what
  the cache matches on) longer than the trustworthy session depth, and the pause record was already dead -
  the half-written cells would have been parked into an image that no longer claims to be paused, i.e. the
  same corruption, one request later. With the cap, `live == resume == pause_pos` and every deeper cell is
  rewritten before anything trusts it again.
  Disk images need no guard: a paused conversation never reaches the session dir (`park_body`,
  `turn_capture`, `spill_current` all refuse), so `sdir->fetch` can never hand back a pause.

## MEDIUM 5 - stale pause across session rewrites - FIXED (`1c5e2ec27`)

`drop_read_pause()` now runs at every point that rewrites the live session: the session-file LOAD path
(the file never carries a pause), the slot copy-in (its top, covering both the restore-failure
`live.clear()` branch and the success branch), the borrow restore, and the incoming restore. Placed inside
those four blocks rather than once after the park, because a plain same-conversation continuation - the
pause's main customer - also passes the common point and must keep its pause. When the park did run,
`park_body` already moved the pause into the image and dropped it, so these calls are the no-op case.
(The `want_cvec` rewrite at ~:20168 and the #879 flush already dropped it.)

## MEDIUM 6 - `resumes()` called tautologically - FIXED (`5f45bd8d0`)

Took the review's second option: the dead `p.chunk == c0` check is gone and the header doc now says what
the function actually guarantees - the saved row count equals what this run reads at `c0` (`min(C, n - c0)`),
while WHICH chunk the pause belongs to is settled by the caller's ABSOLUTE `pos0` match and the id compare,
and `p.chunk` is documented as the paused run's log label. The first option (store and pass the saved
run-relative chunk) is not viable: the review's own cross-run case shows a valid continuation meets the
paused chunk at its own `c0 = 0` while the saved run-relative chunk is mid-run - the check would reject
exactly the resumes that matter. `prefill.cpp` now passes the saved point's own fields
(`{pause_.chunk, pause_.layer, pause_.tokens}`) instead of a `c0`-built stand-in, so nothing tautological
remains at the call. Covered by `layer_pause_test` (cross-run pass, wrong-C refusal, label-free restored
pause still resumes).

## NIT 7 - `take_pause_rows` left the record armed - FIXED (`da4564a7a`)

The getter clears `pause_` along with moving the rows (hpp doc updated), so `paused_chunk()/paused_tokens()`
can never report a pause whose rows are gone. The park path (the only caller) already cleared right after.

## NIT 8 - refused keeps left mismatched record/rows pairs - FIXED (`da4564a7a`)

`Prefill::give_pause` returns whether it armed; the restore sites only mirror the record into `read_pause`
when it did. And a cancelled request that hit a pause its keep branch refused (`lp_pos >= 0` but the armed
record is not this request's pos+layer) drops record and rows together before the existing
`if (cancelled && read_pause.pos < 0) sp.clear_pause()`. The pos+layer compare (rather than a "refused"
flag) also catches the subtle same-chunk variant: a second cut in the same chunk at a deeper layer whose
keep was refused would otherwise leave the old layer's record paired with the new cut's rows for a later
park to restore - shallower layer over deeper rows.

## NIT 9 - `paused_chunk=` label divided by `m.T` - FIXED (`da4564a7a`)

`SavedPause` now stores `chunk` = `c0 / C` of the run that paused (that run's real C, planned or not),
exposed as `Prefill::paused_chunk_index()`; `generate.cpp` reads that for `lp_chunk` (the log line and the
DONE `paused_chunk=`) instead of `(lp_pos - q) / sp.chunk()`.

## What I ran (each module once, as named)

- `cmake --build scratch/build-p125j-cuda75 --target strata_prefill` and `--target strata` - exit 0, several
  times over the C++ fixes; final re-check after all commits exit 0.
- `ctest -R '^layer_pause_test$'` in the same build dir (reconfigured `-DSTRATA_BUILD_TESTS=ON`, only that
  target built) - Passed.
- `python -m unittest serve.test_priority` - 68 passed (the plan step-3 test; the grace default it pins is now
  0.0). Not run: `serve/test_lost_read.py` (not a module the plan names; its regex is un-anchored and the
  unchanged-without-keys line shape keeps it matching - checked by inspection, and it needs the card anyway).
- `python -m py_compile` on `serve/server.py`, `serve/qfn_engine.py`, `engines/qfn/serve/server.py` - OK.
- `bash -n tools/card_tests/layer_pause.sh` - clean. The script itself was NOT run: it is a card test and
  the plan excludes GPU runs here (no card on this box).

## Manager's four, restated against the fixed tree

- Resume at the exact layer: implemented and it compiles; unit-proven at the helper level; card proof is
  `layer_pause.sh` (condition 1+2).
- Nothing read twice: unchanged design plus the leg-sum check now scripted (condition 3); `run()`'s
  read-past clear and the OFF-path byte-identity were verified by the review and are untouched.
- Parks with the conversation: park/restore carry record+rows and refuse disk writes; the borrow donor hole
  and the stale-pause sites are now closed (MAJOR 4, MEDIUM 5).
- Idle grace -> 0: done with the doc and the test, on the strength of the fixed build + unit tests; the
  card run is the remaining witness before shipping it to the box.
