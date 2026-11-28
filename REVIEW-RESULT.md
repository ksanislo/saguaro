# Code review (local agent) - P124v2 read-rate-true - reviewed 2026-11-28 at 854866cd8

FIX-FIRST

Scope: `git diff b7ed679cbc75cf1959bb1d793e8442534ac4a5f3..HEAD` (12 files, 32 commits) against
`PLAN-p124v2-read-rate-true.md` / `REPORT-p124v2-read-rate-true.md` / `FIX-REPORT.md`. Repro probes in
`scratch/` (gitignored): `review_r7_restart_hook_probe.py`, `review_r7_qfn_engine.committed`. Nothing else edited.

## Findings by severity

1. FIX - the round-6 numerator fix (`stat.read_count_reset`) is released by the next admission hook, and its
   release reason does not hold after a restart: the same spike class returns on the restart -> `_read_start`
   path. `serve/server.py:4597` (`_read_start`) zeroes `read_count_reset` on the stated reasoning "this hook
   does NOT zero read_leg_ms - the closed legs' wall time stands in the span - so their counted tokens stand
   in the numerator too". That premise fails when the restart branch (`serve/server.py:4767`) has JUST zeroed
   `read_leg_ms`: the pre-restart legs' TIME is gone from the span (`_read_span_s`, `serve/server.py:4512`)
   while the hook hands their TOKENS back to the numerator (`_read_counted`, `serve/server.py:4577-4583`),
   breaking the docstring's own promise that the abandoned attempt's tokens "never divide the new read's
   span" (`serve/server.py:4537-4539`). Verified on the committed code through the real Service and the
   committed harness (scratch/review_r7_restart_hook_probe.py): a 9,600-token / 24 s read at 400 tok/s
   restarts and retries at 300 - the committed round-6 gate path reads a correct 300.0 - then ONE fresh
   admission hook (`_read_start` fires on every admission: `serve/server.py:6326`, `:9907-9909`, e.g. a slice
   re-admit or a requeue after the retried failure) puts the live readout at **1,500 tok/s where the engine
   says 300** (12,000 history tokens over the retry's 8 s pair floor), until the pre-restart interval prunes
   (~10 s, `_read_prune` `serve/server.py:4496-4499`). The pair floor cannot hold it - the round-6 commit
   message itself notes the floor fails with `n_own` above the pair's own `n`. `peak_prefill` dodges the big
   spike only by the same accident as round 6 (`_note_peak_read`'s internal `tokens/tok_s` < PEAK_MIN_SPAN_S
   check, `serve/server.py:4821-4823`; the open-read peak feed at `:4420-4421` gates on the PAIR's span) - a
   MILD inflation (rate <= pair tokens / 5 s) passes that check and records an inflated peak; the chat row and
   `live.prefill_tok_s` show the value either way. No committed test gates restart -> `_read_start`: the
   round-6 gate (`serve/test_read_rate_true.py:578-614`) stops at the retry's own steps, and no
   `serve/test_read_rate.py` `_read_start` call follows a restart. Fix direction: do NOT clear
   `read_count_reset` in `_read_start` - the `min(excl, closed)` at `serve/server.py:4540-4545` already tracks
   pruning correctly, the exclusion is about the SPAN that only the restart branch zeroes, a later restart
   recomputes it, and with no restart it is 0 so the clear is a no-op anyway; the comment's pairing argument
   is true only for the no-restart case it does not discriminate. Gate the post-hook readout in the restart
   test.

2. Carried, documented (no action per round 6): the four unconditional changes - the `_note_hb` belt
   (`serve/server.py:4671-4683`), the cumulative readout (`:4517+`), the `engine_t0` re-baseline
   (`:4778-4782`), the `read_hb` clear (`:4592`, `:4773`) - are not behind the `pos0=`/`chunk=` fields, so the
   strict byte-identical-OFF check formally fails; the in-flight SHARE itself is properly field-gated
   (`serve/server.py:4645-4648`; parse `engines/qfn/serve/server.py:1777-1784`, absent/malformed ->
   `chunk=0`/`pos0=None`), the old-engine restart path still fires (the fall gate at `:4750-4758`), and the
   report states all of this plainly (REPORT...md:147-159). Accepted as recorded.

3. Documented deviation (accepted design, carried): the cumulative readout's sample average lags a warm-up
   read (~1,097 vs 1,580 on the card replay; report lines 130-136) - the plan's 5% clause binds the steady
   mangonel replay, where it holds (below); the card's gates are end-value, cap and last-5-s window.

4. Nit: `_note_hb` reads `stat.read_pos` for its belt (`serve/server.py:4672`) without holding `status_lock`
   (it is mutated under the lock in `_note_read`). Benign under the GIL (atomic int read; worst case a
   one-beat-stale belt bound, and the beat and step paths already share this pattern for `read_hb`) - noting
   for the lock review, not a defect.

5. Environment note (affects the report's claim, not the code): the round-5 "both changed C++ files pass
   `g++ -std=c++20 -fsyntax-only -Wall -Wextra` on the host path" is NOT reproducible in this environment -
   neither CUDA nor ROCm headers exist (`cuda_runtime.h`, `hip/hip_runtime.h` both absent; tried with and
   without the `hip_compat` shim). The C++ changes were instead verified statically here (below); the
   device/nvcc build and card run stay owed exactly as open question 1 records.

## Manager gate - satisfied (with the recorded-file caveat)

`serve.test_read_rate_true` 17 green (281.8 s, run alone): `MangonelTrueReadout`
(`serve/test_read_rate_true.py:277-357`) asserts the readout non-blank and > 0 after EVERY RHB of the read
(:336-339), the sample average within 5% of the engine's own 1,567.8 (:348 - green), the max and
`peak_prefill` <= 1.25x (:351-354) and the close at exactly 27,254 counted (:357). Caveat as recorded (open
question 2): `ref/mangonel-engine-after-p124v.txt` holds only the summary line, so the replay reconstructs
that read's uniform shape rather than replaying recorded beats.

## What checks out (verified here)

- Plan modules, one at a time: `serve.test_read_rate_true` 17 green, `serve.test_read_rate` 34 green (module
  byte-identical to b7ed679 - the P124v pins still hold), `serve.test_read_heartbeat` 19 green.
- Generated file: `tools/saguaro/extract_qfn_engine.py` re-run reproduces `serve/qfn_engine.py`
  byte-identically (diff clean against a saved copy, `git status` clean after); its diff mirrors
  `engines/qfn/serve/server.py`'s `_rhb` parse exactly; no hand edit.
- Engine side, statically: both RHB printfs are 11 specifiers / 11 args (`generate.cpp:11332-11334`,
  `:16673-16676`), each line ONE printf + fflush under `rhb_mu` (`:11315`) - no stdout line can interleave;
  `ReadProgress::mu_` (`read_progress.hpp:42/:55/:64`) is taken and released BEFORE `rhb_mu` at both sites -
  sequential, never nested; no 5-arg `on_layer_pass` site left (`prefill.hpp:70`/`:341`,
  `prefill.cpp:5439-5440`, `generate.cpp:11708`, the swap-aside vector `:16947` and its move sites
  `:16973`/`:16994`, callbacks `:16995-16997`/`:17766-17768`); the piece-boundary line prints `pos0=-1
  chunk=0` (`:11352-11355`) and the share guard needs `chunk > 0`, so it books `done=` only (pinned by
  `test_read_heartbeat.py:383-400`).
- The server's guard assumptions match `read_progress.hpp:89-101`: with no boundary above the reporting
  unit revealed, `b == c.p0` gives every card of a split no part credit and `done == pos0 - reused` holds
  for all of them (the flag's domain); once a card crosses a boundary the cross-card figure carries its
  share and the guard fails for it - no double-count; the fraction-falling-back flag (`serve/server.py:4661-
  4668`) cannot fire for a single card (its passes only rise on one unit) and the piece line never touches
  the tracker.
- `status_lock` is never nested: `_note_read` holds it once (`serve/server.py:4712`) and the whole fall/restart
  branch runs inside that `with`; `_read_close`/`_read_open`/`_read_counted`/`_read_span_s`/`_own_read_rate`
  take no lock themselves; `_note_hb` books through `_note_read` without holding it; `_read_start` takes it
  once (`:4589`).
- py_compile clean on every touched Python file.
- Commits: 32 in range, plan steps each committed (2b5e1daaa, 63c028574, 59d618841); no Co-Authored-By,
  Claude-Session or Assisted-by lines anywhere in the range.
- No secrets, no paths outside the repo: the only absolute path added in the diff is the documented
  `/opt/llm/saguaro/.venv/bin/python` in the report; the timeline fixture holds only /metrics fields; probes
  live in gitignored `scratch/`.
