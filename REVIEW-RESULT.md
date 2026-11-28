# P124v2 review round 8 (local agent, read-only) - diff b7ed679cbc75cf1959bb1d793e8442534ac4a5f3..HEAD (HEAD fde18a4a2 = 854866cd8 + round-7 review file, which touches REVIEW-RESULT.md only)

FIX-FIRST

## Findings

1. FIX-FIRST (P1) - the round-6 numerator re-baseline is released by the next admission hook:
   `_read_start` clears `stat.read_count_reset` unconditionally (serve/server.py:4597). Its comment's
   premise - "this hook does NOT zero read_leg_ms, so their counted tokens stand in the numerator too" -
   holds only when the span was not just zeroed by the restart branch, which zeroes `read_leg_ms`
   (serve/server.py:4767) and re-baselines `engine_t0` (:4778-4782). After a mid-read restart, the next
   admission of the same request (`tls.on_read_start` installed at serve/server.py:6326, fired by every
   `_admission_wait` - serve/server.py:9907-9909, so a slice's next leg or a retried read) clears the
   exclusion, and `_own_read_rate` (:4540, excl = 0) puts the abandoned attempt's counted tokens back in
   the numerator while `_read_span_s` (:4512) holds only the post-restart time. Measured through the real
   Service with my own probe (scratch/review_r8_restart_hook.py, the committed gate test's exact shape:
   9,600 tokens / 400 tok/s, restart at the engine's own 300 tok/s retry): the retry's six steps show
   3,500 / 2,700 / 2,220 / 1,900 / 1,671 / 1,500 tok/s where the engine says 300, decaying only as the
   pair grows (~10 s, and a second round until the closed legs prune). It is not display-only: the live
   readout feeds peaks - `_prefill_tok_s` gates on `read_eng_n` >= PEAK_MIN_READ_TOKENS (2,400 >= 512,
   serve/server.py:4417-4421, serve/server.py:327) and the pair span >= 5 s, so `_note_peak_read` records
   the inflated figure as `peak_prefill`. No committed test covers restart -> `_read_start`:
   `CounterFallsRestartOrCorrectDown.test_a_restart_rebaselines_the_rate_numerator_with_the_span`
   (serve/test_read_rate_true.py:578) feeds only steps after the restart. This is the same failure class
   the round-6 fix removed (old tokens over the new span), re-opened by one line; round 7 (commit
   fde18a4a2) records the same finding - I re-derived it independently, same direction, same order of
   magnitude. Fix direction (review's, not prescriptive): do not release the exclusion in `_read_start`
   - it stays coherent because the exclusion snapshot and the closed intervals it counts against
   (serve/server.py:4540-4545, `_read_counted` :4577-4583) both survive that hook, and
   `min(excl, closed)` already tracks pruning; or key the release on the span actually standing.

2. Low - a sliced read's legs after the first never get the in-flight share: the shared adapter shifts
   the routed heartbeat's `done` by the previous legs' count - `hcb({**entry, "done": base +
   entry["done"]})`, serve/server.py:9937 - but `pos0`, `new` and `reused` stay engine-relative, so the
   share guard's `pos0 - reused == n` and `new > n` (serve/server.py:4647-4648) cannot hold once
   base > 0. Those legs fall back to whole-unit booking (P124v behavior): direction-safe, and the same
   conservative class as the report's open question 3, but this instance (the base shift, not the
   cross-card split) is not stated anywhere. The plan's never-0-after-the-first-RHB property therefore
   holds for whole reads and the read's first leg, not for a slice's second leg after its BRA.

3. Info (documented, accepted in earlier rounds) - the OFF-path is not byte-identical: with the feature
   off (a P124v engine printing no pos0=/chunk=) the beat belt (serve/server.py:4671-4683), the
   cumulative readout (`_own_read_rate` :4517-4575), the `engine_t0` re-baseline (:4778-4782) and the
   `read_hb` clearing (:4592, :4773) all apply; a beat-only restart signal now waits for the next
   PP/BRP step. The report states this plainly (REPORT-p124v2-read-rate-true.md:152-159) and every
   deviation is safe in direction (never above the engine's pair); recorded, not blocked.

4. Info - stale causation in a test docstring: `CardTrueReadout` (serve/test_read_rate_true.py:373-377)
   still says its cut-end BRP steps land below the booked share (13,824 under 13,900). With the
   committed synthesis (PASSES=60, beats strictly before the cut, :424) the booked share maxes ~77
   tokens below each cut, so no step ever falls and the monotone / never-re-opened gates pass because
   no fall occurs - the report already corrected the causation claim (REPORT:70, :115-120); the docstring
   did not follow. The fall-correct-down branch is genuinely covered elsewhere
   (`test_a_step_lagging_a_booked_share_correct_down` serve/test_read_rate_true.py:551,
   `SplitReadFollowsEngineFigure` :617).

5. Nit - transient blank at the restart tick: the restart branch sets `read_eng_ms` None (:4766) and
   `engine_t0` = now (:4782), and the no-pair path needs `now > st.engine_t0` (:4570-4574), so a poll
   between the restart and the retry's first step reads 0 (measured in my probe). Drain-shaped and brief;
   noted next to finding 1, same code.

6. Nit - `read_share_unit` / `read_split_share` (RequestStat :3167) are tracked in `_note_hb`
   (:4661-4668) without `status_lock` while `_read_start` (:4595) and the restart branch (:4776) reset
   them under it. Beats and steps for one request run on its consumer thread, so per-stat writers are
   effectively one thread; worst case a lost reset (one unit followed by the leading card before the
   flag, or one stale tracker on the next read). GIL-atomic single ops; benign, carried from round 7.

## Manager gates (the brief's line), measured, not asserted

Replay of the recorded mangonel read through the committed computation
(`serve.test_read_rate_true.MangonelTrueReadout`; the recorded file holds the summary line only, so the
replay is that line's uniform shape - REPORT open question 2). Independent re-measure through the test
harness (scratch/review_gate_probe.py): first RHB row 1,539.1 tok/s and minimum 1,539.1 over all 75
beats - never 0 or blank after the read's first RHB; the readout's time-average 1,561.3 = -0.41% of the
engine's own 1,567.9 (gate: within 5%); max 1,568.9 = 1.001x (gate: <= 1.25x); `peak_prefill` 1,568.9;
the read closes at exactly 27,254 counted. All green.

## Verified clean (each checked myself)

- Modules run ONE at a time on /opt/llm/saguaro/.venv/bin/python: `serve.test_read_rate_true` 17 green
  (~282 s), `serve.test_read_rate` 34 green (module untouched by the diff - the P124v pairings stand),
  `serve.test_read_heartbeat` 19 green. No other modules run.
- Generated file: re-ran tools/saguaro/extract_qfn_engine.py - serve/qfn_engine.py byte-identical
  (clean git status); the driver change lives in engines/qfn/serve/server.py:1767-1784, parse as plain
  `int()` of the string (no float round-trip), malformed value reads as absent, the piece-boundary
  pos0=-1 chunk=0 keeps its defined values; the untested parse paths are now committed tests.
- C++: `on_layer_pass` is the 6-argument type at every site (prefill.hpp:70, :341; prefill.cpp:5439-5440
  passing `sg.n` / `T`; generate.cpp:11708, :16945 swap-aside vector, :16995, :17766) - no 5-arg site
  left. Both RHB printfs are a single std::printf with the two literals compile-time-concatenated:
  11 specifiers / 11 arguments at each site (generate.cpp:11332-11335, :16672-16678), one line per
  ~230 ms under `rhb_mu`, no interleaving; `j.prog.report()` / `solo_prog.report()` (read_progress
  `mu_`) is taken and released BEFORE `rhb_mu` is acquired - sequential, never nested. The piece line
  prints pos0=-1 chunk=0 (generate.cpp:11354). Host syntax: g++ -std=c++20 -fsyntax-only clean on both
  changed files (generate.cpp needs the build's -DSTRATA_VERSION; prefill.cpp carries two pre-existing
  -Wunused warnings at :3141 only). The device/nvcc build and a card run stay owed - REPORT open
  question 1, unchanged.
- Locks/threads: `status_lock` is a plain Lock (serve/server.py:3288) and is never nested:
  `metrics()` runs `_prefill_tok_s()` before taking it (:5073-5075); `_chats_list` / `_convs_list`
  call only the lock-free helpers; `_note_read` / `_note_hb` / `_read_start` acquire internally and are
  called off-lock from the drivers' consumer threads. All `read_leg_ms` writes are under the lock
  (:4482, :4590, :4767). Per-request beat/step callbacks share one consumer thread (routing at
  :9931-9937 / :9859-9864), so the booked-counter belt (:4671-4683) has a single writer.
- OFF-path booking: with no pos0/chunk the share guard (:4647-4648) fails and only the belt + readout
  changes remain (finding 3); the piece line books done only - committed
  (test_a_piece_boundary_line_books_only_its_done).
- No secrets, no paths outside the repo in any changed file; scratch/ is gitignored (:20); the tracked
  fixture serve/fixtures/card-rc2-read-rate-timeline.txt is the manager's own /metrics capture.
- Commits b7ed679..HEAD carry no Co-Authored-By / Claude-Session / Assisted-by trailers.

## Note on this session

HEAD advanced mid-review: commit fde18a4a2 (a parallel round-7 review, REVIEW-RESULT.md only) landed
after my code review had passed its static checks; this round-8 text re-ran every check against the new
HEAD (code unchanged since 2c9b4719b) and re-derived its central finding independently (finding 1).
