# REVIEW - U5b (the read heartbeat's `done=` moves forward only on a layer split)

**MERGE-WITH-NITS**

Scope: `git diff abdf593b28e92f1910d0b13cba4df655691e4f63..HEAD` vs PLAN-u5b-split-progress.md / REPORT-u5b-split-progress.md.
The four brief questions, answered first:

- **Per read and thread-safe?** Yes. Solo: one `ReadProgress solo_prog` (generate.cpp:11087), reset at every read's
  start (generate.cpp:20063) and fed only by that read's host points (the batch path swaps the callbacks out,
  generate.cpp:16712-16718/16690-16700). Batch: one `prog` per `AdmJob` (generate.cpp:16375), and an `AdmJob` is
  created per read (`make_unique` at generate.cpp:17349, 18096), so two reads at once keep separate figures; the
  `prog_begun` latch (generate.cpp:16376, 16706-16709, 17548-17550, 17885-17887) resets once per read and lets a
  paused/held read keep its own figure. All state is guarded by `ReadProgress::mu_` (read_progress.hpp:101); every
  public method locks; no lock is held across a call out.
- **Can done exceed new?** No. `refresh_` clamps to `newn = upto_ - from_` (read_progress.hpp:99), `report`/`report_full`
  clamp positions into the read (read_progress.hpp:74, 52, 60), and the empty-read guard (read_progress.hpp:44, 58)
  keeps a warm run at 0. The driver belt takes `max` only when `new` and `rid` match the previous beat
  (serve/server.py:1688-1691), so the stored `done` stays within the same read's `new`.
- **Does a one-card read print the same as before?** Yes. The single stage covers all layers, so `passes`
  (`LE - LB`, prefill.cpp:5440) equals `total_` (`g.n_layers`), and a lone card's reports only ever mark its own
  chunk starts and completed ends, so `upper_bound` (read_progress.hpp:97) finds no end for its current chunk
  (b == p0, no part-credit) and `v = max(full_, p0) = p0` - exactly the old `pos0 - pp_from`; the old printed
  sequence was already monotone, so the `best_` latch changes nothing. Unit-tested at read_progress_test.cpp:44-55.
  The driver belt is likewise the identity on a monotone stream (covered by the 13 existing tests).
- **Anything added to a per-token or per-layer hot path?** Nothing per token. `report()`/`report_full()` run at the
  existing per-layer-pass heartbeat host points (prefill.cpp:5436-5440, before their 950 ms gate) and at the PP/BRP
  chunk-end points (generate.cpp:11384, 16730, 17461) - a mutex + map/vector update + a ≤ n_stages loop per host
  point, only while a read runs (see nit 3). No compute, kernel or ordering change; PP/BRP/RHB line fields and
  format strings are unchanged (generate.cpp:11095, 11386-11388, 16397-16400).

## Verified (no finding)

- **Generated file**: `serve/qfn_engine.py` is exactly what `tools/saguaro/extract_qfn_engine.py` produces from the
  current `engines/qfn/serve/server.py` (regenerated into ./scratch/qfn_engine.check.py and byte-compared:
  IDENTICAL); its diff touches only `_rhb`.
- **Locks / interleaving**: `rhb_slot` takes `prog.mu_` (report), releases, then takes `rhb_mu` and calls
  `prog.done()` inside the single printf under `rhb_mu` (generate.cpp:16391-16401). The order `rhb_mu -> prog.mu_`
  is the only nesting anywhere (`rhb_solo`, the PP/BRP points and the `Swap` undo never hold `prog.mu_` while
  taking `rhb_mu`, and `ReadProgress` calls out under no lock), so no nested acquire of a non-reentrant lock and no
  inversion. Lines are printed by one `printf`+`fflush` under `rhb_mu`, as before; PP/BRP are not under `rhb_mu`
  (pre-existing U5 behaviour, unchanged).
- **Feature OFF**: the belt lives only in `_rhb`, which runs only on `RHB ` lines (server.py:1406-1407) - old
  engines print none, so the driver is byte-identical for them. Engine-side, only the RHB line's `done=` value
  changes; every other line's output is unchanged, and with one card even that is identical (above).
- **Read identity / belt reset**: a slot's entry is popped by its BRA (server.py:1414-1418) and the solo entry by
  RDONE (server.py:1660-1661), so the belt normally starts over between reads independently of the `new`/`rid` test.
- **Tests**: `ninja -C build-u5b read_progress_test && ./build-u5b/read_progress_test` -> ok: 18 checks, 0 bad;
  `serve.test_read_heartbeat` -> 16 tests OK; `serve.test_server` -> 178 OK; `serve.test_dashboard` -> 78 OK
  (each run once, alone). The new tests exercise the change itself: the counter's out-of-order four-card updates,
  single-card equality, two reads, clamping and the end-at-`new` (read_progress_test.cpp:23-116), and the 4 x T4
  trace's backwards beats through the real driver `_rhb` and the real adapter (test_read_heartbeat.py:114-156,
  321-340). No paths outside the repo, no secrets in any added line.

## Findings by severity

1. LOW - transient over-credit is bounded by the gap to the next *reported* boundary, not by one chunk
   (read_progress.hpp:96-99): with several intermediate cards unreported, a lagging card's part-credit runs past
   its own chunk end toward a boundary only a leading card revealed; the report's open question 3
   (REPORT-u5b-split-progress.md:70-72) says "bounded by one chunk", which understates it. The latch then locks a
   value the true progress only catches up to - a beat can print slightly high and stall (mirror of the known
   under-count, REPORT-u5b-split-progress.md:64-65). Display-only; self-corrects; RDONE/BRA stay exact.
2. LOW - the driver belt can mistake a reused batch slot for the same read (server.py:1686-1691): batch lines all
   carry `rid=0`, so only `new` distinguishes reads; a new read on a slot with the identical `new` (a lost BRA,
   evicted only after 60 s, server.py:1703-1706) inherits the old read's `done` floor and keeps its stale `rate`
   until it passes. In practice BRA pops the entry first; the plan defines read identity as `new`/`rid`, so this is
   the belt's designed limit, worth a note only.
3. LOW - `report()`/`report_full()` run at every heartbeat host point *before* the 950 ms gate
   (generate.cpp:16391, 11101), so the ~n_layers-per-chunk host points now each pay a mutex + map insert +
   O(cards) refresh even on suppressed beats. Plan item 2's "the points that already check the one-second rate
   limit" arguably covers this (the points are those points; nothing was added per token), but it is a small new
   cost on the per-layer-pass host path while reading - acceptable, noted.
4. LOW - a mid-request read restart on the solo path (e.g. the #879 "reads from token 0" recovery, or a window
   retry) does not re-lay `solo_prog` (reset exists only at generate.cpp:20063), so `best_` can stay latched above
   the rewound position until the read passes it. Display-only; RDONE carries the exact figures.
5. NIT - the engine-side integration itself (`lb` threading through the callbacks, `rhb_solo`/`rhb_slot` wiring) is
   exercised only by compilation and the driver-level tests; the split shape is reproduced in the C++ unit test and
   the fake-line driver tests, not by a 4 x T4 run (acknowledged in REPORT-u5b-split-progress.md:73-74). Fine here
   (no GPU); the trace run stays the acceptance step.

## Build/tests run (plan's named modules only, one at a time)

- `ninja -C build-u5b read_progress_test` (up to date) + `./build-u5b/read_progress_test` -> ok: 18 checks, 0 bad
- `python -m unittest serve.test_read_heartbeat` -> Ran 16 tests, OK
- `python -m unittest serve.test_server` -> Ran 178 tests, OK
- `python -m unittest serve.test_dashboard` -> Ran 78 tests, OK

No code was edited; `serve/qfn_engine.py` was only regenerated into ./scratch for the byte-comparison.
