# REPORT - U5b: the read heartbeat's done= moves forward only on a layer split (QFN engine)

## What changed

**Engine (commit 2683276b2).** The `RHB` line's `done=` is now ONE progress figure per reading request,
shared by every host point of that read, instead of the printing host point's own piece count. On a layer
split each card's host points report their own chunk position and the first cards are several pieces ahead
of the last (ref/hb-4xT4.txt: 14592, 19456, 14592; 34048, 29184); the shared figure counts the new tokens
whose reading is complete as **tokens x (layers done / all layers)**: for every card, its fully-passed
positions weighted by its share of the layers, plus the part of its current chunk that runs up to the next
chunk boundary revealed by another card's report (or by the last stage's chunk end). A token the first card
has passed and the last has not therefore counts in part. The figure never decreases (`best_` latch) and
ends at `new` when the read ends (every chunk end at the last stage - the solo `on_chunk` / the batch `BRP`
points - reports the fully-read position). One card: the only positions known are its own chunk starts and
a current chunk's end is never its own report, so the figure is exactly the position it reported before -
**a single card prints what it printed today**. Per-read state (solo: one instance; batch: one per
`AdmJob`, separate for two reads at once), guarded by a mutex (the stages' pipeline threads report
concurrently); the PP/RHB/BRA line fields and format are unchanged, `pass` stays the printing point's own.
Card identity: the heartbeat callback gained the stage's first layer (`lb`) - a queued job's
`PromptJob::on_layer_pass` runs on every stage's thread and carries no stage index, and `lb` is unique
across a split's stages. Updates happen only at the heartbeat host points that already run the one-second
rate-limit check (and the chunk-end `report_full` at the existing PP/BRP points); nothing was added to the
per-token path; no compute, kernel, launch, ordering or read change.

**Driver (commit 6e54f3afa).** `StrataEngine._rhb`: `done` is a belt - `max(previous done of the same read,
new done)`; a backwards line changes neither `done` nor `rate`; a new read (other `new` or `rid`) starts
the belt over. `serve/qfn_engine.py` regenerated with `tools/saguaro/extract_qfn_engine.py` (never edited
by hand).

**Tests (commit 7c98aac0e).** A C++ unit test of the progress counter without a GPU (the counter is a
header-only class) and three new Python tests driving the real driver/adapter with the 4 x T4 trace shape:
recorded beats never go back, a backwards line leaves done and rate untouched (no dip, no spike), a new
read starts over, the row's `read_done` only moves forward.

## Files

- new `engines/qfn/include/strata/program/read_progress.hpp` - `strata::program::ReadProgress`
- new `engines/qfn/src/program/read_progress_test.cpp`
- `engines/qfn/CMakeLists.txt` - `read_progress_test` target (beside `slot_grid_test`)
- `engines/qfn/include/strata/prefill/prefill.hpp` - `on_layer_pass` (both PromptJob and Prefill): `lb` added
- `engines/qfn/src/prefill/prefill.cpp` - the two host points pass `LB`
- `engines/qfn/src/program/generate.cpp` - solo path (`solo_prog` reset per request at `pp_total/pp_from`,
  fed by `sp.on_layer_pass` with the card's `lb`, floored by the last stage's `on_chunk`, the between-pieces
  heartbeat reports the piece end as the fully-read position), the batch slot path (`AdmJob::prog` reset at
  the read's starts incl. the pipelined feeder, fed by both `on_layer_pass` installs, floored by both BRP
  points), and the protocol note
- `engines/qfn/serve/server.py` - `_rhb` belt
- `serve/qfn_engine.py` - regenerated (10/4 lines, the `_rhb` change)
- `serve/test_read_heartbeat.py` - `test_a_split_read_solo_beats_are_a_belt`,
  `test_a_split_read_slot_beats_are_a_belt`, `test_a_split_read_row_never_goes_back`

## Build check and tests run

- `QFN_GGML_DIR=<../strata-main/third_party/llama.cpp> tools/qfn/build.sh ./build-u5b` ->
  **QFN BUILD OK** (only pre-existing warnings; build logs `build-u5b.cmake.log` / `build-u5b.build.log`)
- `ninja -C build-u5b read_progress_test && ./build-u5b/read_progress_test` -> ok: 18 checks, 0 bad;
  `ctest --test-dir build-u5b -R read_progress_test` -> 1/1 Passed
- `python -m unittest serve.test_read_heartbeat` -> **Ran 16 tests, OK** (13 existing + 3 new)
- `python -m unittest serve.test_server` -> **Ran 178 tests, OK**
- `python -m unittest serve.test_dashboard` -> **Ran 78 tests, OK**

## Open questions

- The figure deliberately under-counts the leading card's current chunk (its end is unknown until another
  card or the read's end reveals it): a beat can show one chunk less than the true partial progress, and
  the last printed beat may sit slightly below `new` (the state itself ends at `new`). The 4 x T4 trace's
  last `done=38903 / new=38910` is this shape; RDONE/BRA carry the exact figures anyway.
- Card identity by `lb` assumes the stages' layer ranges do not overlap (`set_stage`'s contract); a helper
  GPU (`set_stage_helper`) is not a pipeline card and reports nothing extra.
- Transiently, a lagging card's part-credit can run to the next *reported* boundary rather than its true
  chunk end when an intermediate card has not reported yet (bounded by one chunk, corrected at the next
  beat, locked in by the never-decreasing max).
- No GPU here, so the split behaviour is exercised by the unit test and the fake-line driver tests, not by
  a 4 x T4 run.
