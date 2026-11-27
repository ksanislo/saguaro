# P124v2 — the read rate is a true readout

Plan: PLAN-p124v2-read-rate-true.md. Date: 2026-11-27.

## Step 1 — what the recorded mangonel read actually did

`ref/mangonel-engine-after-p124v.txt` holds the engine's own summary:
`serve: prompt 27254 tokens = 0 reused + 27254 read in 17383 ms (1567.8 tok/s)`.
`ref/mangonel-timeline-after-p124v.txt` holds the Monitor's display of the same read.
The raw RHB lines were not recorded, so `scratch/replay_mangonel.py` replays the engine's
own figures (chunk 4,608, 78 beats of 230 ms) through the server's `_note_hb` /
`_own_read_rate` and compares against the timeline.

Three breaks, in order of how they showed up:

1. **0.0 before the first unit closed.** `RHB done=` counts *whole* units only — a chunk,
   or a piece of a pipelined read (`engines/qfn/include/strata/program/read_progress.hpp`,
   U5b: no end sentinel for the chunk in flight). The line's `pass=<i>/<n>` field — the
   current layer pass of the chunk in flight — was parsed but never used. On mangonel
   (27,254 tokens = 5 chunks + a 4,214 tail, first chunk ending at 6,544 ms) the display
   sat at 0.0 for the first ~1.8 s of a 17.4 s read.
2. **The denominator was the wrong clock.** The beat's `ms=` is the engine's *read* clock
   over *settled* legs (`generate.cpp` rhb_slot: `j.read_ms + now - t_read` — closed
   pieces plus the open one, but against `done=` of whole units it pairs a finished chunk
   with a clock that stopped while the rest of the read continued). The server divided
   `done` by `min(ms, window)`: at t=7.11 s the engine's own new/read_ms said 4,608/6,544
   = **704.2 tok/s** (its PP line's `prefill_tok_s_mean` — visible at chunk ends in the
   timeline: 704.2, 1,089.8, 1,328.2, 1,487.6, 1,604.5, 1,678.1, 1,567.8) while the
   Monitor showed **2,099.4**; the session peak read 2,577 against a true 1,567.8 —
   1.6×.
3. **Chunk tokens land whole.** With no in-flight credit the readout is a sawtooth
   (replay of the old code: 1,178 → 801 → 1,214 …, time-mean 1,070 — 32% under the truth
   even before the stale-denominator inflation of individual beats).

## Step 2 — the fix

**The readout = (tokens finished + the unit in flight's tokens × passes done / passes)
over the read's own elapsed span.**

- `engines/qfn/src/program/generate.cpp` — RHB lines carry the unit the passes count:
  `pos0=<unit's first token> chunk=<unit's tokens>`, solo (`rhb_print` printf 11328,
  `rhb_solo` 11333, the `sp.on_layer_pass` lambda 11703) and batch (`rhb_slot` 16658,
  printf 16669, slot lambda 16990, queued-job lambda 17761); the piece-boundary print
  says `chunk=0` (done just moved). Protocol comments 8395, 11302–11308, 16630–16640.
- `engines/qfn/include/strata/prefill/prefill.hpp` — `on_layer_pass` carries the chunk
  length (PromptJob:70, Prefill:341); `engines/qfn/src/prefill/prefill.cpp:5438–5440`
  passes `sg.n` / `T`.
- `engines/qfn/serve/server.py:1727` `_rhb` — parses `pos0=`/`chunk=` into the read_live
  entry (absent fields stay None: an older engine behaves exactly as before P124v2).
- `serve/qfn_engine.py` — re-extracted with `tools/saguaro/extract_qfn_engine.py`, never
  hand-edited.
- `serve/server.py` `_note_hb` (4582) — a beat's figure becomes
  `done + min(chunk, new−done) × pass_i/pass_n` **only when `stage != "stage"` and
  `pos0 − reused == done`**: done stands exactly at the reporting card's unit start, so
  the share is this read's own; a split read's cross-card figure already holds the other
  card's share (`pos0 − reused < done`) and multi-leg base-shifted reads never line up —
  nothing is added and nothing is double-counted. `done=0` beats now book their share —
  the first chunk is on the display from its first layer pass (mangonel's first RHB is at
  ~0.23 s, not 6.5 s).
- `serve/server.py` `_own_read_rate` (4508) + `_read_span_s` (4495) + `read_leg_ms`
  (3159, summed by `_read_close` 4476, zeroed on read restart 4679) — the denominator is
  the read's **own elapsed span**: closed legs' wall time + the open leg's, from
  `engine_t0`, with the engine's `stage_ms` staging subtracted, and the engine's own
  `read_ms − stage_ms` pair as the floor (a slower wall clock can lower but never raise
  the rate above the engine's figure). The beat's `ms=` alone is no longer the
  denominator. The silence gate (`READ_BEAT_STALE_S`) applies only to engines that beat
  (`st.read_hb is not None`): a sparse-step engine's BRP cadence is not silence.

## Step 3 — tests

With `/opt/llm/saguaro/.venv/bin/python -m unittest` (this env has no nvcc, so the C++
changes are not compiled here — see open questions):

- `serve.test_read_rate_true` — **10 green** (~81 s). `MangonelTrueReadout` replays the
  summary line's uniform shape (4,608 × 6 units, 230 ms beats): no 0.0 once the read is
  underway, time-mean **1,560.3** vs the engine's own 1,567.8 (−0.5%, well inside 5%),
  peak **1,567.9** ≤ 1.25 × 1,567.8, closes at exactly 27,254 counted.
- `serve.test_read_rate` — **34 green** (P124v's window/staleness/peak suite intact:
  the wall-span floor by the engine's own pair keeps every old pairing assertion true).
- `serve.test_read_heartbeat` — **17 green** after pinning the new line: the
  `EngineSideContract` (serve/test_read_heartbeat.py:382) now asserts the two-part printf
  with the trailing `" pos0=%lld chunk=%lld…"` literal for solo and slot; the driver and
  live-view assertions unchanged.
- `py_compile` clean on every touched Python file.

Replay evidence (`scratch/replay_mangonel.py`, gitignored): with the *recorded* chunk
timings (first chunk cold at 704.2) the readout now equals the engine's own cumulative
figure at every point — 704 at 6.5 s, 934 at 7.6 s, …, 1,567.8 at 17.4 s (a cold first
chunk drags the cumulative average; its time-mean is 1,091 by construction, and the
engine's own line-by-line figures say the same). The plan's 5% / 1.25× acceptance applies
to the uniform replay of the engine's summary line, which passes.

## Commits

- step 1 `2b5e1daaa` — red `MangonelTrueReadout` + replay analysis
- step 2 `63c028574` — the fix (server + engine line + re-extracted qfn_engine)
- step 3 `59d618841` — heartbeat contract pins the P124v2 line format

## Open questions

1. **The C++ side is not compiled here** (no nvcc in this environment). The
   `on_layer_pass` signature change (prefill.hpp:70/341, prefill.cpp:5438–5440, the four
   generate.cpp lambdas) is mechanical but must be built and the card run before the
   readout change can be believed on real hardware; until then only the server side is
   exercised (an engine without `pos0=`/`chunk=` keeps the old, whole-units behaviour —
   the split is fully behind the missing fields).
2. The recorded mangonel file has only the summary line, so the inflation's exact
   beat-by-beat pairing is reconstructed from the timeline + code, not observed; the
   replay's shape (uniform chunks) follows that summary.
3. The share guard (`pos0 − reused == done`) is deliberately conservative: any engine
   whose `done=` and `pos0=` disagree (a split read's second card reporting mid-chunk)
   gets no in-flight credit — correct but no smoother than before P124v2 for that card.
