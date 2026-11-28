# Code review (local agent) - P124v2 read-rate-true - reviewed 2026-11-28 at 27849ce3b

FIX-FIRST

Scope reviewed: `git diff b7ed679cbc75cf1959bb1d793e8442534ac4a5f3..HEAD` (12 files) against
`PLAN-p124v2-read-rate-true.md`, `REPORT-p124v2-read-rate-true.md`, `FIX-REPORT.md`.
Probes used as evidence are in `scratch/` (gitignored): `review_probe_fall.py`, `review_probe_prefix.py`,
`gen_wall.txt`, `server_base.py`. Nothing in the repo was edited.

## What checks out (verified here, no action)

- Plan modules green, run one at a time: `serve.test_read_rate_true` 11 (141.1 s), `serve.test_read_rate` 34,
  `serve.test_read_heartbeat` 17. `py_compile` clean on every touched Python file.
- Manager gate on the mangonel replay: `MangonelTrueReadout` (serve/test_read_rate_true.py:277-356) asserts
  never blank/0 after the read's first RHB (:335-338), the readout's average within 5% of 1,567.8 (:347, the
  committed run is 1,560.3 = -0.4%), max and `peak_prefill` <= 1.25x (:350-353), and the read closing at
  27,254 counted (:356). Green.
- `serve/qfn_engine.py` is generated, not hand-edited: `tools/saguaro/extract_qfn_engine.py` re-run reproduces
  it byte-identically (diff vs. the pre-run copy clean, `git status` clean afterwards).
- The report's open question 1 (REPORT-p124v2-read-rate-true.md:100-106, "the C++ side is not compiled here")
  is answerable in this env: `g++ -std=c++20 -fsyntax-only -Wall -Wextra` (the build's CXX host path) passes on
  `engines/qfn/src/prefill/prefill.cpp` (exit 0) and `engines/qfn/src/program/generate.cpp` (exit 0, one
  pre-existing unrelated warning at generate.cpp:15825; prefill.cpp:3141 likewise). The 6-arg
  `on_layer_pass` swap (generate.cpp:16947, :16995, :17766) and the two-part RHB printf (:11332-11333,
  :16672-16675) are type-correct and format-correct, 11 specifiers / 11 args in both; a grep finds no 5-arg
  site left (prefill.hpp:70, :341; prefill.cpp:5439-5440; generate.cpp:11708, :16995, :17766). A device/nvcc
  build and a card run are still owed.
- Locks: `rhb_mu` is taken once per line (generate.cpp:11327, :16667) and never nested - the host-point
  helpers are atomics only (:11268-11283, :16652-16660), `ReadProgress` takes and frees its own mutex before
  the printf (read_progress.hpp:42, :55). Python: `status_lock` is not re-entered by `_note_hb` -> `_note_read`
  -> `_read_close`/`_read_open`/`_note_peak_read`, nor by `_own_read_rate` -> `_read_span_s`.
- stdout interleaving: the line is one `printf` + `fflush` (:11332-11335, :16672-16680); the two-part literal
  is concatenated at compile time, so nothing new can interleave. The driver reads with `readline()`
  (engines/qfn/serve/server.py:722, :3494), so the longer line needs no reader change.
- No secrets, no paths outside the repo in the diff; the fixture holds only /metrics fields.

## P1 - the round-4 fall branch is unproven, has no coverage, and can swallow a real restart

- `serve/server.py:4701-4716` (new branch in `_note_read`), justified by `FIX-REPORT.md:24-26` and `:42`.
- The trigger the FIX-REPORT describes cannot be produced by the code it describes. The share is capped at the
  unit's end (`min(chunk, new - done)`, serve/server.py:4620), and the fixture's beats use
  `pos0=<cut start>`, `chunk=<cut> - <cut start>` with `pass <= PASSES`
  (serve/test_read_rate_true.py:423-426), while the cut-end BRP carries exactly that cut
  (serve/test_read_rate_true.py:428). So a beat's booking reaches the whole-cut count and never passes it:
  `n_read < stat.read_pos` never fires. Measured on the committed test: **0 steps below `read_pos`**
  (`scratch/review_probe_fall.py`). Disabling only the fall branch (its `hb` lookup forced to `None`) leaves
  the committed test bit-identical - first non-zero 0.53 s, max 1,679.5, interval never re-opened, 27,299
  counted (`scratch/review_probe_prefix.py`). So the branch that the whole round-4 finding 1 rests on is dead
  code as far as the committed tests go, and "13,900 booked vs. 13,824" (FIX-REPORT.md:25) is not reachable.
- The recorded card numbers still have no demonstrated cause. Driving the fixture's own fields through the
  pre-P124v2 code (b7ed679's `_own_read_rate`, `_note_hb`, `_note_read`, `_read_close`, `_read_start`,
  `_read_counted`, exec'd verbatim) gives first non-zero **6.86 s**, max **1,698.4**, mean **1,218**, 0
  interval re-opens - not the card's 0.78 s / 9,881 / ~2,393, and not FIX-REPORT.md:42's 4,088 / 8,243 /
  13,270. Note also that before this branch the restart branch did *not* re-baseline `engine_t0` (added here
  at serve/server.py:4725-4729), so the "0.466 s since the restart" denominator (FIX-REPORT.md:32-34) cannot
  occur at b7ed679 at all. The capture has no RHB lines, so the 9,881 sample is unattributed; the branch is
  shipping on a guess.
- The tolerance is a live regression risk: `stat.read_pos - n_read <= (hb.get("chunk") or 0)`
  (serve/server.py:4704) is satisfied by an early read whose beats have already booked up to a whole unit
  (serve/server.py:4620) while the engine's whole-cut counter is still 0. A genuine restart of that read - a
  retry, a slice's leg, a resume, with the new read's live beat at `done=0` - now books `read_pos` instead of
  taking the restart branch (serve/server.py:4717-4730): the leg is never closed (`r[3]` unset, so the window
  figure at serve/server.py:4421-4433 loses it), `read_pos` freezes until the engine's counter climbs past the
  stale value, and only BRA's exact figures (serve/server.py:4691-4695) put it right at the end. No committed
  test asserts that a restart through beats still fires the restart branch (the round-2 probe was gitignored).
- Ask (small): either reproduce the fall from a line shape the engine actually prints (a recorded RHB stream,
  or a split read - see P2) and gate the branch so a restart cannot satisfy it (the live beat's `new`/`rid`
  belonging to this read, and at least one whole unit already closed), or drop the branch and keep the
  cumulative readout alone, which by itself already removes the interval spike - plus a committed test that a
  real restart still re-opens the interval.

## P2 - on a split read the in-flight share follows the leading card, and P1 locks the lead in

- `serve/server.py:4616-4620` adds the share whenever `pos0 - reused == done`. On a layer split with the cards
  between the same two boundaries, `ReadProgress::refresh_` gives **no** part credit (`b == p0` when no
  boundary above the reporting card's position has been revealed -
  engines/qfn/include/strata/program/read_progress.hpp:89-101, esp. :93-94), so `done == pos0 - from_`
  exactly and the guard holds for *every* card. Each card's line books its own `chunk x pass_i/pass_n` and the
  belt keeps the maximum (serve/server.py:4622-4633), so the counter tracks the leading card, not the cards'
  layer-weighted mean - up to roughly one unit of over-count (4,608 tokens on mangonel), a few hundred tok/s
  of inflation mid-read on the plan's own 4xT4 box. Then the cut-end whole-cut step falls below that booked
  lead and P1's branch belts it *up* rather than correcting it down, so the lead persists to BRA.
- REPORT-p124v2-read-rate-true.md:53-57 and :109-112 assert the opposite ("a split read's cross-card figure
  already holds the share: `pos0 - reused < done`"). That is true only once some `report_full`/other-card
  boundary above `p0` exists (read_progress.hpp:57-59, :79-83); in lock-step between boundaries it is false.
- Nothing in the three committed modules replays a split beat with `pos0`/`chunk` (only the gitignored
  `probe_split_share`); `serve/test_read_heartbeat.py:135`, `:160` belt split beats but without the fields.
  Ask: one committed split replay (two cards, different `pass` fractions) with its gate on the readout never
  exceeding the engine's own cumulative figure, and correct the report's claim.

## P3 - nits

- OFF-path claim overstates the split. "an engine without `pos0=`/`chunk=` keeps the old, whole-units
  behaviour - the split is fully behind the missing fields" (REPORT-p124v2-read-rate-true.md:104-107) is not
  true of four unconditional changes: the beat belt (serve/server.py:4622-4633), the cumulative readout
  (:4508-4553), `engine_t0` re-baselining in the restart branch (:4725-4729) and clearing `read_hb` at
  `_read_start` (:4570) and on restart (:4722). For an engine that beats but prints no `pos0`/`chunk` (the
  P124v build): a mid-read restart no longer fires from a beat, a post-restart rate is measured from the new
  start instead of carrying the old one, and a fresh read shows no "staged" hint until its first beat. All
  safe in direction, but they are not behind the fields.
- `_read_span_s` (serve/server.py:4495-4506) double-counts a multi-leg read whose leg is re-opened **without**
  a start hook: `read_leg_ms` already holds the closed legs and `now - engine_t0` re-adds their time from the
  first start. Reachable whenever a step arrives after `_read_close` (a late BRP after BRA, or after the first
  token, serve/server.py:4697-4699) - the span inflates and the rate reads low. Measure the open leg from
  `read_cur[1]`, or re-take `engine_t0` in `_read_open`.
- The ended branch of `_own_read_rate` (serve/server.py:4538-4547) is now also reached for an **open** leg that
  has a first token (the guard above it needs `first_token is None`), where `ended = read_t1` and the gap is
  added to the span. Harmless (it drains), but the comment says "the read is over"; the case added for the
  round-1 should-fix 2 is a different shape from what the comment describes.
- Cumulative readout, stated plainly: over the recorded card read the committed replay's samples average
  **1,097** tok/s against the client's 1,580 (-31%), because the read's first cuts run ~711 tok/s and a
  cumulative value lags (FIX-REPORT.md:77 concedes this). The gates that pass are the last sample (1,581.2,
  +0.1%), the cap (max 1,679.5 <= 1,975) and the >= 5 s peak (1,698.4). Plan step 3's "time-average within 5%"
  is met only by the uniform replay of the summary line, because `ref/mangonel-engine-after-p124v.txt` holds
  that one summary line and no RHB lines (verified: 1 line) - so plan step 1/3's "the recorded engine lines"
  are reconstructed, never replayed, and a recorded-shape replay would average low by construction.
- Fixture location: `serve/testdata/card-rc2-read-rate-timeline.txt` (serve/test_read_rate_true.py:383) starts
  a second test-data directory; the repo's convention is `serve/fixtures/` (serve/test_loop_guard.py and five
  others).
- Parse robustness: `int(float(...))` at engines/qfn/serve/server.py:1772-1779 catches `ValueError` only -
  `chunk=1e400` gives `inf` and `OverflowError` escapes `_rhb` on the pump thread. Unreachable from `%lld`;
  `int(kv.get("chunk") or 0)` (the style at :1744) cannot fail.
- Untested paths of the new parser/protocol: a line **without** `pos0`/`chunk` (the P124v engine), a malformed
  field, and the piece-boundary `pos0=-1 chunk=0` line (generate.cpp:11352-11355). `serve/test_read_heartbeat`
  pins the printf (:386-393) and `test_an_engine_before_the_lines_is_untouched` (:111) covers only the case of
  no RHB lines at all.
