# REVIEW-RESULT - P124w verify-batch crash instrumentation (diff f4b4dac08..HEAD)

MERGE-WITH-NITS

Reviewed `git diff f4b4dac08e8a7a673d733956b593a7bf6be72208..HEAD` (4 commits: ae4391f3e the code, 98f8c6094 its
test, a0bb977e4 the REPORT nits, 414654c70 the FIX-REPORT) against PLAN-p124w-verify-crash.md and
REPORT-p124w-verify-crash.md. No blocking finding: the change is the plan's second branch, correctly placed,
and everything it claims re-verified on this tree.

## What was verified (evidence, file:line)

- Step 1 quotes are real: every sampled REPORT quote exists in the ref/ logs (crash-around-line-351460.log:
  "captured the batch window over slots 2,2,2,2,2,4" x4, "elastic slots: CUDA1's expert cache takes 42 MiB
  back" x1, "slot 2 admitted beside the decoding slots: 156650" x1; 356061.log: the "no progress for 60 s ...
  token 17083" stall line x1; 775696.log: "the loan was lent again under held reads", "ERR verify batch:
  layer 3 never rang", "slot 1 admitted ... 149903" x1 each).
- The bound matches the driver's own semantics: the driver treats a row as backed iff `need <= slot_cells`
  (generate.cpp:14253) and admits with `slot_fit(b, upto + 1, ...)` (generate.cpp:14202); the new helper
  refuses exactly `pos[t] + 1 > backed[t]` (batch_rows.hpp:36-40), i.e. it fires only where the driver's
  guard was missed - no legitimate window is refused by the comparison itself.
- Rows are validated before indexing: `batch_rows_error` bounds `rows[t]` to `slots_.size()`
  (batch_rows.hpp:21, called verify.cpp:3875) and the pre-existing `slots_[(size_t) rows[t]]` access at
  verify.cpp:3883 precedes the new one at :3892; `backed[kVerifyMaxT]` (verify.cpp:3891) fits because
  `S <= max_t <= kVerifyMaxT = 8` (batch_rows.hpp:18, verify.cpp:730, verify_kernels.hpp:21).
- Feature OFF is unchanged: `slot_cells = -1` by default (session.hpp:64) and only an elastic slot session
  gets >= 0 (session.cpp:121-124, reachable only through the slot pool created when SAGUARO_ELASTIC_SLOTS
  provides spool, layer.cpp:1800-1803); with -1 `first_row_past_backed` skips every row, prints nothing, and
  the cost is one 8-entry gather per stage_batch. The `err` strings at every touched site are byte-identical
  (verify.cpp:4049, :4099, :4239, :4459, :4483 vs base) so server.py's FATAL_PREFIXES (server.py:899-900) and
  the driver's `vram_short` classification (generate.cpp:2750-2758 - matches "out of memory"/"do not fit",
  neither in the new refusal) behave as before.
- No locks: the diff adds no lock acquisition anywhere; `bwindow_note` (verify.cpp:4003-4018) touches no mutex
  and guards its own slot-range read (:4009-4010).
- Threads: `note_never_rang` reads `last_batch_/last_t_/last_rows_/last_pos_b_`, written only by `stage_batch`
  (verify.cpp:3990-3994); while a batch window runs, `batch_launch_rows` refuses any new stage on that
  Verifier ("batch_launch while this stage is busy", verify.cpp:4221), so the note names the running window;
  the caught-past-backed refusal at :3893-3899 fires before :3990, so a refused window cannot pollute the
  next never-rang's note. `q` is passed in (:4024-4027) rather than re-queried - no sticky-error consumption.
- Stdout protocol: every new line is one `fprintf(stderr, ...)` with the pre-built string (verify.cpp:3896,
  :4025, :4047, :4237), matching the existing `saguaro verify:` stderr style (verify.cpp:3624, :3858);
  the engine's stdout protocol ("ERR ...", server.py:1352) is untouched, and the engine's stderr goes to the
  log file line-by-line (server.py:1173-1176), so no interleaving or parse hazard beyond the existing one.
- Generated file: `serve/qfn_engine.py` and `engines/qfn/serve/server.py` are byte-identical to base (empty
  `git diff` on both), and tools/saguaro/extract_qfn_engine.py extracts only server.py (its docstring and
  line 10) - so no re-extraction was owed. No path outside the repo, no secrets in the diff; the only
  absolute-path mention is the allowed `/opt/llm/saguaro/.venv/bin/python` inside FIX-REPORT.md:77 prose.
- Tests really exercise the change: batch_rows_test.cpp:76-95 pins `first_row_past_backed` (inside, last
  backed cell readable, first unbacked row caught, first offender reported, -1 never past, 0-cell slot
  catches its first row). Run here: standalone `g++ -std=c++20 -Wall -Wextra` -> `batch_rows_test: ok`
  (no warnings); `ctest -R ^batch_rows_test$` in scratch/build-p124w-sm75 -> Passed, test #128 (matching the
  FIX-REPORT claim). That build is sm_75/CUDA ON (CMakeCache.txt:58,940) and its verify.cpp.o (22:44) is
  newer than verify.cpp (22:30), so it compiled this exact code; the only verify.cpp warning in
  build-strata.log:1865 is the pre-existing unused `f` at :3345.
- Citations re-checked on this tree: generate.cpp:5837 (SAGUARO_PIPED_MTP forces batch_mtp), 14156
  (elastic_idle), 14191 (slot_admit_fit), 14250 (slot_grow_or_end), 14564 (run_slot_rows), 15620
  (batch_launch_rows); session.hpp:32/63-64; session.cpp:161-183 with host_follow at :172/:180;
  ref/mangonel-q4-config.json:69 = `"SAGUARO_PIPED_MTP": "1"`. Commits carry no Co-Authored-By /
  Claude-Session / Assisted-by lines.

## Findings by severity

LOW 1 - non-atomic cross-thread read of `slot_cells`. `stage_batch` reads `SessionState::slot_cells`
(verify.cpp:3892) while the driver's `session_slot_fit` writes it (session.cpp:173/:181) from another thread;
no atomic or lock on either side - formally a data race. The failure directions are the safe ones (a stale
larger value misses the bound, i.e. today's behaviour; a fresh smaller value names the slot instead of
crashing), and the codebase already reads this field plainly across threads (e.g. generate.cpp:5958,
:14253), so this matches existing practice rather than degrading it. A `std::memory_order_relaxed` load (or
a one-line comment saying the value may be stale either way) would make the intent explicit.

LOW 2 - the note can name a different layer index than the err string in one narrow site. verify.cpp:4482
loads `b_k_` for `note_never_rang` while :4483 loads it again for `err`; under concurrent progress the two
could differ by a layer. Cosmetic (the err string, which is what server.py sees, is unchanged byte-for-byte).

NIT 3 - stale line numbers in the REPORT's candidate 4: "the row-indexed buffers are sized kVerifyMaxT
(verify.cpp:3911-3914, 3945)" (REPORT:156) quotes base-tree positions - `rr[kVerifyMaxT]` is now at
verify.cpp:3925 and the ple rows at :3958 after ae4391f3e's +13 shift; the neighbouring citations in the same
paragraph (batch_rows.hpp:16-27, verify.cpp:3875) still land correctly. Documentation-only.

NIT 4 - what the bound is not, stated for the record: it compares a row's own position, so a kernel that
reads page-rounded beyond the last backed cell (e.g. the `idx_block`-rounded `b0` staging at
verify.cpp:3941-3945) is outside its guarantee - it is the cheap lower bound the plan asked for, not a proof
of safety; and crash 2's paging stall (no illegal access at all) is by design untouched. The chosen shape
(checked bound + naming line behind no switch, single-cause fix deliberately not taken, FIX-REPORT:48-52)
matches the manager rule and the logs' actual resolution - none of candidates 1-3 is provably *the* cause.

## To re-run (manager)

- `g++ -std=c++20 -Wall -Wextra -I engines/qfn/include engines/qfn/tests/core/batch_rows_test.cpp -o
  scratch/review/batch_rows_test && scratch/review/batch_rows_test`
- `ctest --test-dir scratch/build-p124w-sm75 -R '^batch_rows_test$' --output-on-failure`
