# Code review (local agent) — P124v2 read-rate-true, round 3

MERGE-WITH-NITS

Reviewed `git diff b7ed679..HEAD` (HEAD 7c31fd4be) against PLAN-p124v2-read-rate-true.md and
REPORT-p124v2-read-rate-true.md. No code edited; the three plan modules run one at a time, green.

## Manager gate — green (re-measured here)

`serve.test_read_rate_true.MangonelTrueReadout` (serve/test_read_rate_true.py:276-355) replays the recorded
engine read (ref/mangonel-engine-after-p124v.txt carries only the summary line — 27,254 tokens, 0 reused,
17,383 ms, 1,567.8 tok/s — verified by reading the file; the raw RHB lines were never recorded, report
open question 2 stands). Independent probe over the same replay: 75 RHB samples, **never None/0 after the
first RHB**, time-average **1,561.3 = −0.41% of 1,567.8** (limit ±5%), peak **1,568.9 = 1.001×** (limit
1.25×), closes at exactly **27,254** counted. The test asserts all four (lines 334-355) and passes.

## Modules (one at a time, this checkout)

- `serve.test_read_rate_true` — 10 green (81.2 s)
- `serve.test_read_rate` — 34 green
- `serve.test_read_heartbeat` — 17 green

## Verified sound

- **Generated file**: re-ran `tools/saguaro/extract_qfn_engine.py` — `serve/qfn_engine.py` reproduces
  **byte-identical** from `engines/qfn/serve/server.py:1767-1779`; the only generated file in the diff.
- **C++ shape** (no nvcc here — build still open, report open question 1): `on_layer_pass` is 6-arg at every
  site — prefill.hpp:70/341 (both declarations), prefill.cpp:5439-5440 (`sg.n` / `T`; `T`,`p0` are the chunk's
  own tokens/position, prefill.cpp:3133), generate.cpp:11708, 16995, 17766; no 5-arg `std::function` or
  5-arg call survives anywhere (grep); the swap-aside vector generate.cpp:16947 matches prefill.hpp:341, so
  the round-1 compile blocker stays fixed. Both RHB printf sites (generate.cpp:11332, 16673) are single
  `printf` calls with adjacent-literal concatenation — 11 format specs, 11 arguments, one `fflush` under
  `rhb_mu` (11327, 16666): no new stdout interleaving, no new or nested locks. Piece line prints
  `pos0=-1 chunk=0` (generate.cpp:11354-11355); the server's guard `chunk > 0` (serve/server.py:4608) never
  reads -1.
- **Share guard**: `pos0 − reused == done` (serve/server.py:4608-4610) admits the share only when `done` stands
  exactly at the reporting unit's start, clamps `min(chunk, new−done)`, skips `stage` beats, and int()-truncates;
  absent fields (an engine before P124v2) leave the share path inert
  (`engines/qfn/serve/server.py:1772-1779` → chunk 0 / pos0 None).
- **Locks**: `read_leg_ms` is only touched under `status_lock` (`_read_close` serve/server.py:4476 called from
  4396/4561/4690; `_note_read` holds it at 4652); `_read_span_s`/`_own_read_rate` are readers; no new lock, no
  nesting. Metrics rows pick hb keys by name (serve/server.py:4801-4803) — the two new entry keys do not leak.
- **Denominator**: `_own_read_rate` (serve/server.py:4520-4546) divides by the read's own wall span with the
  engine's own pair as the floor (`max`), so the readout can only read lower than the engine's figure, never
  higher; peaks stay gated at `PEAK_MIN_SPAN_S = 5.0` s (serve/server.py:296) as the plan requires.
- **No outside paths, no secrets**: the diff writes nothing outside the repo; nothing secret.

## Findings by severity

1. (low) **The belt is not behind the new fields — "OFF unchanged" is not strictly true.**
   serve/server.py:4612-4623 clamps every RHB booking that would step backwards, including lines without
   `pos0=`/`chunk=`. REPORT-p124v2-read-rate-true.md:105-106 ("the split is fully behind the missing fields")
   is true for the *share* but not the *belt*: with an old engine, a decreasing RHB `done=` used to reach
   `_note_read`'s restart branch (serve/server.py:4689-4699); now it is clamped and the restart waits for the
   first unbelted PP/BRP step (serve/server.py:9868/9872 call `cb(..., False, ...)` — the restart branch does
   still fire there, so real restarts survive). On the batch path the next BRP can be a whole chunk later
   (~6.5 s on the mangonel shape): that window shows the pre-restart figure pinned, instead of the old
   (worse) double-counted leg. Deliberate trade of the round-2 fix — accept, but the report's sentence should
   say the belt is unconditional. (For the record: `_note_read`'s docstring at serve/server.py:4636 calls
   prov=True the BRP step, but no caller passes prov=True any more — predates this diff.)
2. (low) **Test gap: the belt itself has no committed test.** `probe_split_share` / `probe_belt_release` live
   only in gitignored `scratch/`; no committed module feeds split-card beats (same `done=`/`pos0=`, differing
   `pass=`) or a going-backwards beat — `pos0=` appears in the tests only in the single-card
   MangonelTrueReadout (serve/test_read_rate_true.py:319-321) and the contract's text pins
   (serve/test_read_heartbeat.py:386-393). Deleting serve/server.py:4623 would keep all 61 tests green. The
   round-2 MUST-FIX deserves a permanent test; lift the probe into `serve/test_read_rate_true`.
3. (low) **Staging subtraction is one-beat-deep** (serve/server.py:4505): `_read_span_s` subtracts only the
   latest beat's `stage_ms` from the summed legs' wall time, so a multi-leg read that staged in several legs
   under-subtracts the closed legs' staging (rate reads low — safe side of the `max()` floor), and `read_hb`
   is not cleared at `_read_start` (serve/server.py:4556-4563) or at the restart reset (4691-4698): until the
   new read's first beat, the *previous* read's `stage_ms` is subtracted from the new leg's span. Bounded
   (floor + `min(wall, stage)`) but an avoidable wart; clear `stat.read_hb` with the pair.
4. (info) **C++ still unbuilt here** (no nvcc): everything checkable without a compiler checks out (items
   above), but the card build must happen before the engine side is believed — report open question 1 stands.
   The replay acceptance rides on the summary line's uniform shape only, as the report states.
5. (nit) FIX-REPORT.md:12 and :87 cite `/opt/llm/saguaro/.venv/bin/python` — a machine path in repo docs
   (documentation only, no code reads it).

## Verdict

MERGE-WITH-NITS — the readout formula, the line-format change, the belt and the tests match the plan; the
manager gate measures green (−0.41% mean, never 0, 1.001× peak, exact 27,254); findings 1-3 are
follow-up-grade, finding 4 (a card build) gates the hardware claim, not the merge.
