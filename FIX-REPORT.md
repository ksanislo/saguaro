# P124v2 FIX-REPORT - manager review round 5 (FIX-FIRST: REVIEW-RESULT.md reviewed at 27849ce3b, 2026-11-28)

Review: REVIEW-RESULT.md - FIX-FIRST with P1 (the round-4 fall branch in `_note_read` is unproven,
uncovered, and can swallow a real restart), P2 (on a split read the in-flight share follows the
leading card and the fall branch locks the lead in), and seven P3 nits. Every FIX-FIRST finding and
every nit taken; each fix committed separately. Modules only the read-rate ones, one at a time:
`serve.test_read_rate_true` 16 + `serve.test_read_rate` 34 + `serve.test_read_heartbeat` 19, green
at every code commit; `tools/saguaro/extract_qfn_engine.py` re-run reproduces `serve/qfn_engine.py`
byte-identically after the driver edit (no second-round diff, `git status` clean). No engine C++
touched. No `serve/qfn_engine.py` hand-edit (the driver change went through
`engines/qfn/serve/server.py` + the extractor).

## P1 (FIX-FIRST): the round-4 fall branch - unproven trigger, no coverage, swallows a real restart

Chosen from the review's two options: KEEP the tolerance but GATE it so a restart cannot satisfy it,
and CORRECT DOWN instead of belting up - not "drop the branch": the fall IS producible by a line
shape the engine really prints (the split read's unit-alignment, see P2: both cards' beats hold the
share guard at the same whole-done while their fractions differ, and the next cross-card step lands
below the leading card's booking - `scratch/probe_split_share.py`'s round-2 shape, now committed as
a test), and with no tolerance at all that fall would fire the restart branch - the doubled leg and
mid-read re-baseline this plan is about.

- serve/server.py `_note_read` (the old 4701-4716): the no-restart path now needs, on top of the
  fresh beat and the beat's agreement (`n_read >=` the beat's `done=`) and the one-unit fall cap,
  that at least one WHOLE unit stands closed in the engine's own counter
  (`n_read >= max(hb["done"], hb["chunk"])`). The review's swallow case - an early read whose beats
  booked only the share of a still-OPEN first unit (`done=0`), its retry's first step - fails that
  gate (`n_read=0 < chunk`) and takes the restart branch. The beat belongs to this read: it is
  routed to this stat by its slot and the driver's `_rhb` belt restarts with `new`/`rid`.
- Belt-UP replaced with the step booking as it came: `read_pos` corrects DOWN to the engine's
  figure (round 4 kept the leading card's lead standing above the engine's cross-card figure until
  BRA's exact - the review's P2 lock-in).
- The restart branch is otherwise untouched (round 2's signal, nit 3's re-baseline) plus resetting
  the new split-share trackers.
- The review's measurements accepted as recorded: the fixture's beats cannot book above the cut end
  (`min(chunk, new - done)`), 0 steps below `read_pos` in the committed replay, disabling the branch
  left it bit-identical - the round-4 "13,900 vs 13,824" is not reachable; the card capture had no
  RHB lines, and driving the fixture's fields through b7ed679 gives first-non-zero 6.86 s / max
  1,698 / mean 1,218, not the card's 0.78 s / 9,881 / ~2,393. The card's exact cause stays
  UNPROVEN and is now stated so where the code claimed it (the `_own_read_rate` docstring, commit
  9a39ddc7f); the cumulative readout of round 4 stands regardless - it is what makes the readout
  independent of which steps fire restarts.
- Committed coverage (serve/test_read_rate_true.py `CounterFallsRestartOrCorrectDown`): a restart
  before any whole unit closed (retry's `new_read=0` step under `done=0` beats) re-opens the
  interval and re-baselines `engine_t0`; a restart below the live beat's own `done=` re-opens; a
  lagging cross-card step (the aligned-unit shape, booked 8,746, step 7,007) does NOT re-open -
  interval and `engine_t0` untouched, `read_pos` corrected down, BRA closes exact.
- Commit 30f74aa48.

## P2 (FIX-FIRST): on a split the share follows the leading card; P1's belt locked it in

- Confirmed against read_progress.hpp:89-101: between revealed boundaries `b == p0`, no part
  credit, `done == pos0 - reused` for EVERY card of the unit in flight, so the share guard holds
  for all of them and the `_note_hb` belt keeps the MAX of their shares - the counter tracked the
  leading card, up to a unit (4,608 tokens) over the engine's own cross-card figure.
- Fix (serve/server.py `_note_hb`): a `pass` fraction coming back DOWN on the SAME unit (one
  card's passes never go back; the RHB line carries no card id) proves two cards report it - flag
  the read (`stat.read_split_share`, unit tracker `stat.read_share_unit`, both cleared at
  `_read_start` and in the restart branch); from the flag on no in-flight share is added and the
  counter follows the engine's own figure, which is never behind its own steps. Before the first
  lagging fraction two cards are indistinguishable from one card: that share stands, and the next
  engine step corrects it down through the P1 branch (not a restart).
- Committed `SplitReadFollowsEngineFigure`: a two-card split (78 passes each of 156 layers), one
  fresh 13,824-token read in 4,608-token units; the RHB `done=` values come from a faithful Python
  replica of `ReadProgress::refresh_` (same card terms, same bounds, same rounding) driven by the
  same host points, with chunk-end BRP steps carrying that figure. Gates: once the engine has
  printed a cumulative figure, `_read_counted` never exceeds it at any sample (card A's 40/78 on
  the aligned unit books 4,608, not 6,971); `read_pos` never steps back; the open interval is never
  re-opened and `engine_t0` never moves; the read closes at exactly 13,824.
- The report's opposite claim ("a split read's cross-card figure already holds the share:
  pos0 - reused < done") corrected in place (REPORT-p124v2-read-rate-true.md, commit 1b148056e)
  and the same overbroad comment fixed in both `_note_hb` (serve/server.py) and the driver's `_rhb`
  (engines/qfn/serve/server.py).
- Commit a459a667c.

## P3 nits

- OFF-path claim overstates the split: REPORT open question 1 rewritten (commit 1b148056e): the
  in-flight SHARE is behind the missing fields, but the beat belt, the cumulative readout, the
  `engine_t0` re-baselining and the `read_hb` clearing are unconditional; the P124v-engine
  behaviour shifts they cause are listed (safe in direction, not behind the fields).
- `_read_span_s` double-counted a leg re-opened after a close without a start hook (late BRP after
  BRA / after the first token): the open leg now measured from `read_cur`'s own start `r[1]`
  (identical to `engine_t0` for a single leg and for restarts, which re-take both). Committed
  `LateStepAfterClose`: span stays 8.0 s after a re-open (old formula 8 + 13 = 21, verified red by
  re-measuring), readout holds ~400 not ~150. Commit d078eb2d6.
- `_own_read_rate` ended-branch comment said "the read is over" while the branch also serves an
  OPEN leg with a first token (round 1 should-fix 2): comment now names both shapes; docstring
  causation softened per P1 above. Commit 9a39ddc7f.
- Cumulative readout stated plainly (REPORT addendum + here): the committed card replay's samples
  average ~1,097 tok/s against the client's 1,580 because the read's first cuts run ~711 and a
  cumulative value lags a warm-up BY CONSTRUCTION; the gates that hold are the read's END value
  (1,581.2, +0.1% of 1,580), the cap (max 1,679.5 <= 1,975) and the last-5 s window (1,600); a
  recorded-shape replay would average low on mangonel too, and both replays are reconstructions
  (the recorded mangonel file holds only its summary line - re-verified, 1 line).
- Fixture location: `git mv serve/testdata/card-rc2-read-rate-timeline.txt
  serve/fixtures/card-rc2-read-rate-timeline.txt` (repo convention; `serve/testdata/`'s pre-existing
  openrouter sample left alone); the three references updated. Commit 2803c9060.
- Parse robustness: `int(float(...))` -> `int(kv.get("chunk") or 0)` / `int(kv["pos0"])` in the
  driver `_rhb` (engines/qfn/serve/server.py; a `chunk=1e400` line used to escape `_rhb` with
  OverflowError on the pump thread; malformed fields read as absent; the piece line's
  `pos0=-1 chunk=0` keeps its defined values) - re-extracted `serve/qfn_engine.py` byte-identically
  reproducible. Commit abe41e2c5.
- Untested parser/protocol paths now committed (serve/test_read_heartbeat): a line without
  `pos0`/`chunk` (the P124v engine), malformed `chunk=1e400` / `chunk=abc` / `pos0=nan`, the
  piece-boundary `pos0=-1 chunk=0` parse, and that line behind the real adapter booking its `done=`
  only (share guard needs `chunk > 0`, no restart signal). Commit 92e3505e0.

## Left as recorded

- The C++ side still un-built here (no nvcc): the review's own host-path `g++ -fsyntax-only` pass
  is recorded in the report; device/nvcc build and a card run stay owed. Nothing this round touched
  src/ or engines/ C++.
- The card run's exact 9,881/2,393 attribution stays unproven (recorded as such in the code and the
  report); a future capture WITH RHB lines is owed before any further causation claim.
- Probes stay gitignored in scratch/ (only the committed tests pin behavior).

## Modules (one at a time, this checkout)

- serve.test_read_rate_true - 16 green (11 prior + 3 P1 + split replay + late-step) ~212 s
- serve.test_read_rate - 34 green
- serve.test_read_heartbeat - 19 green (17 prior + 2 parser/protocol)
- extractor re-run: serve/qfn_engine.py byte-identical (clean `git status` after).
