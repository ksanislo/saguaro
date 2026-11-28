# Code review (local agent) - P124v2 read-rate-true - reviewed 2026-11-28 at c75f80547

FIX-FIRST

Scope: `git diff b7ed679cbc75cf1959bb1d793e8442534ac4a5f3..HEAD` (12 files, 28 commits) against
`PLAN-p124v2-read-rate-true.md` / `REPORT-p124v2-read-rate-true.md` / `FIX-REPORT.md`. Repro probes in
`scratch/` (gitignored): `r6_restart_spike.py`, `r6b.py`, `qfn_engine.pre-extract`. Nothing else edited.

## Findings by severity

1. FIX (blocker) - after a mid-read restart the cumulative readout divides the OLD attempt's tokens by the
   NEW read's span, reading grossly high - exactly the failure class this plan exists to kill.
   `serve/server.py:4750-4764` (the restart branch) zeroes `read_leg_ms`, clears `read_hb` and re-baselines
   `engine_t0`, but `_read_counted` (`serve/server.py:4566-4572`) still sums the CLOSED intervals - the failed
   attempt's tokens stay in the numerator `n_own`, while `_own_read_rate`'s denominators (`:4545`, `:4556`,
   `:4559-4563`) restart with the new read. The docstring's floor promise - "the engine's own read_ms pair as
   the floor ... never raise the rate above the engine's own figure" (`serve/server.py:4519`, report step 2) -
   fails because the numerator holds tokens the pair (`read_eng_n`) never had. Measured on the committed code
   through the real Service + driver hooks (scratch/r6_restart_spike.py, scratch/r6b.py): a 15 s / 400 tok/s
   read (24,000 tokens, readout a true 1,600) then the retry's step `new_read=500` - the live readout shows
   **49,000 tok/s** on the step (the engine's own figure: 333), and **3,363 tok/s six seconds later** (the
   engine's own: 300.0); it converges only once the closed interval prunes (~10-15 s). All realistic retry
   shapes land in the branch: fresh beats fail its "one whole unit closed" gate (`:4741-4744`, the very shape
   of P1's committed test), and the hook-first retry clears `read_hb` at `_read_start` (`:4581`).
   `peak_prefill` survives only by accident - `_note_peak_read`'s internal `span = tokens/tok_s` check
   (`serve/server.py:4799-4804`) rejects the inflated rate - but the chat row and `live.prefill_tok_s` show it.
   No committed test gates the readout's VALUE after a restart: `CounterFallsRestartOrCorrectDown`
   (`serve/test_read_rate_true.py:517-576`) asserts re-open/re-baseline only, and the replay/split gates never
   restart. Fix direction: on restart also baseline the rate's numerator (subtract the closed intervals'
   tokens from `n_own`, e.g. a counter set with the reset, keeping `_read_counted`'s history figure), or keep
   the span consistent with the counted - the latter collides with the two committed assertions (`:532`
   engine_t0 re-baseline, `LateStepAfterClose`), so the baseline is likely the way; add a post-restart
   readout gate to the restart test.

2. Carried, documented (no action beyond the report's open question 1): the four unconditional changes - the
   `_note_hb` belt, the cumulative readout, the `engine_t0` re-baseline, the `read_hb` clear - are not behind
   the missing fields, so a P124v engine (RHB without `pos0=`/`chunk=`) does not get byte-identical OFF
   behaviour. The in-flight SHARE is properly field-gated (`serve/server.py:4633-4634`; parse at
   `engines/qfn/serve/server.py:1777-1784`: absent or malformed -> `chunk=0`/`pos0=None`), and the
   old-engine restart path still fires (the fall gate at `:4741-4744` needs `chunk>0` to suppress), but the
   readout's denominator/clock semantics change for the old engine regardless. The report states this plainly
   (REPORT...md:147-159); the brief's byte-identical check formally fails on these paths.

3. Documented deviation: the CUMULATIVE readout time-averages ~31% under the client's figure on the card's
   warm-up read (~1,097 vs 1,580; report round-5 addendum, lines 130-136). The plan's 5% clause binds the
   steady mangonel replay, where it holds (below); the card's gates are end-value, cap and last-5-s window.

4. Nit: report line references drifted - `_note_hb` is serve/server.py:4607 (not 4582), `_own_read_rate`
   :4514 (not 4508), `_read_span_s` :4498 (not 4495), `EngineSideContract` serve/test_read_heartbeat.py:421
   (not :382).

5. Observation (accepted design): until a lagging card's lower fraction lands, the split read's counter runs
   on the leading card's share - up to a unit ahead - and is corrected DOWN by the next engine step
   (`serve/server.py:4644-4656`); the gate is the fraction-falling-back flag, sound for a single card (whose
   passes never go down on one unit; piece lines print `pos0=-1 chunk=0` and take no share).

## Manager gate - satisfied (with the recorded-file caveat)

`serve.test_read_rate_true` 16 green (211.7 s, run alone): `MangonelTrueReadout`
(`serve/test_read_rate_true.py:277-357`) asserts the readout non-blank and > 0 after EVERY RHB of the read
(:336-339), the sample average within 5% of the engine's own 1,567.8 (:348; the run is green), the max and
`peak_prefill` <= 1.25x (:351-354) and the close at exactly 27,254 counted (:357). Caveat the report states
(open question 2): `ref/mangonel-engine-after-p124v.txt` holds only the summary line, so the replay is a
reconstruction of that read's shape, not its recorded beats.

## What checks out (verified here)

- Plan modules, one at a time: `serve.test_read_rate_true` 16 green; `serve.test_read_rate` 34 green (module
  unchanged from b7ed679 - the P124v pins still true). `serve.test_read_heartbeat` was NOT run (the plan names
  the read-rate modules only); its new contract was verified statically instead - the pinned two-part printf
  literals are in `engines/qfn/src/program/generate.cpp:11332-11333` and `:16673-16674` exactly as asserted.
- Generated file: `tools/saguaro/extract_qfn_engine.py` re-run twice reproduces `serve/qfn_engine.py`
  byte-identically (cmp clean, `git status` clean); its diff mirrors `engines/qfn/serve/server.py`'s exactly.
- C++ host path (re-run here): `g++ -std=c++20 -fsyntax-only -Wall -Wextra -DSTRATA_VERSION=...` exits 0 on
  both changed files (pre-existing warnings only). No 5-arg `on_layer_pass` site left; the swap-aside vector
  (generate.cpp:16947) and both put-back sites match the 6-arg type (prefill.hpp:70/:341,
  prefill.cpp:5439-5440; callbacks :11708, :16995-16997, :17766-17768); piece line prints `pos0=-1 chunk=0`
  (:11352-11355), never read by the share guard.
- Locks/threads/stdout: each RHB line is ONE printf (11 specifiers / 11 args at both sites) under `rhb_mu`
  (generate.cpp:11315) - no line can interleave; `ReadProgress::mu_` and `rhb_mu` are taken sequentially,
  never nested (report, then print); server-side `status_lock` is never re-entered (the fall branch runs
  inside the single `with` in `_note_read`; `_read_close`/`_read_open`/`_read_span_s`/`_own_read_rate` take
  no lock themselves; `_note_hb` books through `_note_read` without holding it).
- Commits: 28 in range, step 1/2/3 each committed (2b5e1daaa, 63c028574, 59d618841); no Co-Authored-By,
  Claude-Session or Assisted-by lines anywhere in the range.
- No secrets, no paths outside the repo: the only absolute path added is the documented
  `/opt/llm/saguaro/.venv/bin/python` in the report; the timeline fixture holds only /metrics fields.
