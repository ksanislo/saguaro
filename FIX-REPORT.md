# P124w FIX-REPORT - REVIEW-RESULT.md findings -> what changed (or why not)

Branch `saguaro/q1-p124w-verify-crash`, base `f4b4dac08`. Three commits now carry the work the round-2 review
found missing: `ae4391f3e` (the step-3 code), `98f8c6094` (its unit test), `a0bb977e4` (the four REPORT nits,
force-added past the checkout's `REPORT-*.md` exclude as P124a2's nit commit did).

## BLOCKER 1 - "step 3 has no artifacts" -> implemented exactly as REPORT:157-162 promised (commit ae4391f3e)

The review's "To re-run (coder)" list, item by item:

1. **A `slot_cells` bound at the batch staging.** `Verifier::stage_batch` (src/core/verify.cpp:3887-3900) now
   refuses a window whose row reads at or past its slot session's backed cells: it gathers each row's
   `SessionState::slot_cells` (include/strata/core/session.hpp:64) and calls the pure helper
   `first_row_past_backed` (include/strata/core/batch_rows.hpp:36), then fails with
   `ERR verify: slot <N> row at cell <P> reads past the <B> cells backed for it` instead of an illegal access -
   the plan's "assertion that names the bad index". `slot_cells < 0` (not an elastic slot session) skips the
   check, and it sits right behind the existing `pos[t] < 0 || pos[t] + 1 > max_cells` check (:3882-3886),
   the site candidate 1 (REPORT:122-135) names. Behind no switch; hot-path cost is one `int64_t[8]` gather and
   an 8-iteration compare per stage_batch, and the message build only on a refusal.
2. **A cell-range / layer-range / slot / backed-window log line at the failing launch, and at the
   launch-failure / never-rang sites.** `Verifier::bwindow_note` (src/core/verify.cpp:4003-4018) formats one
   line: `batch window on CUDA<d> layers [lb,le): slot <s> cells <lo>..<hi> backed <slot_cells>/<max_cells>`
   per slot group (`whole(<max_cells>)` for a non-elastic session). It is printed, `saguaro verify:`-prefixed
   like the existing engine log lines (:3624), at exactly the sites the review named:
   - the caught-past-backed refusal at the staging (:3896-3897);
   - `verify: batch launch:` in `run_slot_rows` (:4045-4051) and in `batch_launch_rows` (:4235-4241);
   - `verify batch: layer K never rang` via `note_never_rang` (:4023-4028) from the `run_slot_rows` spin
     (:4098), `batch_poll`'s external-serve check (:4458) and its in-loop check (:4482).
   So a crash 4 says, on the same stderr stream the `dequant launch:` line reaches:
   `saguaro verify: batch window on CUDA0 layers [0,11): slot 2 cells 156582..156586 backed 156590/1048576 slot 4 cells 249827..249827 backed whole(1048576): layer 3 never rang (an illegal memory access was encountered)`
   - the gap REPORT:113-116 names. The existing `err` strings are untouched byte for byte (FATAL_PREFIXES in
   engines/qfn/serve/server.py matches their prefixes; the note goes to stderr, not into them).
3. **A pure-helper unit test.** The bound factored out, so `batch_rows_test` (tests/core/batch_rows_test.cpp)
   gained six checks for `first_row_past_backed`: rows inside, the last backed cell still readable, a row at
   the first unbacked cell caught, the first offender (not a later one) reported, `-1` backed-whole sessions
   never past, and an unfitted elastic slot (0 cells) catching its first row. Run one module at a time, per
   the plan's step 3: standalone (`g++ -std=c++20 -Wall -Wextra`) -> `batch_rows_test: ok`; and as the sm_75
   build's own target -> `ctest -R ^batch_rows_test$`: `100% tests passed, 0 tests failed out of 1` (test
   #128). The plan names no other test module, so none was run.
4. **Build QFN for CUDA sm_75 in a NEW build dir under scratch/, no GPU runs.**
   `scratch/build-p124w-sm75` (CUDA 13.1, `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`): target
   `strata` built and linked `saguaro-qfn` ([100%], log `scratch/build-p124w-sm75/build-strata.log`); verify.cpp
   compiled with no warning from the new code (the log's verify.cpp warnings are the pre-existing ones at
   :3345). Nothing was executed on a GPU.
5. **Commit each part.** `ae4391f3e` then `98f8c6094` then `a0bb977e4`; each `git add` named only its own
   files, no `git add -A`, no Co-Authored-By / Claude-Session / Assisted-by lines.

Deliberately NOT taken: the single-cause fix. The manager rule (REVIEW-BRIEF.md:8) and the review agree none
of candidates 1-3 is provably *the* cause from these logs, so this is the plan's second branch - cheap checked
bounds plus the naming line. The `verify batch:` stream-sync failures and the 20 s `timed out` lines were left
byte-identical too: the review named the launch-failure and never-rang sites, and those are what the three
crashes printed.

## NITs (all cheap, all fixed in a0bb977e4)

1. **"the elastic fit/trim calls above never touch it" (REPORT:139)** - restated truthfully: `session_slot_fit`
   grows/trims the drafter's *host* copy through `host_follow` (src/core/session.cpp:172, 180); what never
   follows is its VRAM rows - the claim the review calls accurate and alarming.
2. **bound quoted as only `pos[t] + 1 > max_cells` (REPORT:123-124)** - now quoted whole:
   `pos[t] < 0 || pos[t] + 1 > max_cells`.
3. **":14517-14525 with `--spec 3` drafts" (REPORT:137-138)** - now named as the `batch_mtp` path, reached
   because SAGUARO_PIPED_MTP forces `batch_mtp = true` (generate.cpp:5837, config :69), not because of
   `--spec` itself; both citations checked on this tree before editing.
4. **list scoped to "engines/qfn/src" while citing include/ files (REPORT:118)** - the heading now declares
   paths relative to `engines/qfn/` and the bare `core/…`, `program/…`, `kernels/…` citations carry their
   `src/` prefix once.

## Generated-file rule

The step-3 work is entirely engine C++; `engines/qfn/serve/server.py` was not touched, so `serve/qfn_engine.py`
owes no re-extraction through tools/saguaro/extract_qfn_engine.py - both files remain byte-identical to base.
No lock, thread or stdout-protocol change: the new lines are single `fprintf(stderr, ...)` calls on paths that
already return, the same way the existing `saguaro verify:` lines print.

## Constraints kept

No sub-agents; relative paths only (the one toolchain exception used: /opt/llm/saguaro/.venv/bin/python was
not even needed - the test is C++); no cd, no stash; scratch files only under scratch/ (the build dir, its
log, and the two pre-existing leftovers); engine sources edited directly but serve/qfn_engine.py never by hand.
