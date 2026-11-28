# P124v2 review round 9 (local agent, read-only) - diff b7ed679cbc75cf1959bb1d793e8442534ac4a5f3..HEAD (HEAD e719da96a = round-8 FIX-REPORT; code since d4c2f245f: the P1 fix, comments/docstrings)

MERGE-WITH-NITS

The round-8 blocker is fixed and gated, the manager gates measure green under my own probe, and I found
no correctness bug this round. One documentation nit (drifted C++ line refs in the REPORT) and the
carried, documented Infos below; none blocks the merge.

## Findings

1. Nit - the REPORT's step-2 C++ line references drifted ~4-5 lines at this HEAD (its serve/server.py and
   test references are all exact; these were not re-grepped as rounds 6/8 did for the Python side):
   `rhb_print` printf 11328 -> now 11332-11333, `rhb_solo` 11333 -> 11337, `sp.on_layer_pass` 11703 ->
   11708, `rhb_slot` 16658 -> 16663, batch printf 16669 -> 16672-16674, slot lambda 16990 -> 16995,
   queued-job lambda 17761 -> 17766 (REPORT-p124v2-read-rate-true.md:41-44). A one-line re-grep like
   round 6's nit 4 closes it; the code itself matches every claim at the new lines (checked at each).

2. Info (new, direction-safe) - `_read_exact` (serve/server.py:4487-4496) rewrites a CLOSED interval's
   `r[1]/r[2]/r[3]` to the engine's exact span and count but does not re-scale `read_leg_ms`, which
   `_read_close` (serve/server.py:4484) summed from the hook-based bounds. The span the readout divides by
   can therefore miss the re-anchoring of one leg. Bounded and safe in direction: the denominator is
   `max(engine pair, span)` so the rate stays under the engine's own figure, and if the exact count drops
   `closed` under the exclusion snapshot, `min(excl, closed)` over-subtracts and the readout reads LOW,
   never high. Stated for the record; not worth code.

3. Info (carried, documented, accepted) - the OFF-path is not byte-identical: with the fields absent (a
   P124v engine) the share books nothing (the guard at serve/server.py:4661-4662 needs `chunk > 0` and
   `pos0 - reused == done`), but four changes apply to every engine: the beat belt (:4686-4697), the
   cumulative readout (`_own_read_rate` :4519-4579), the `engine_t0` re-baselining in the restart branch
   (:4792-4796) and the `read_hb`/share-tracker clearing (:4596-4600, :4787-4791). Every deviation is
   safe in direction (never above the engine's pair) and REPORT:152-159 states it plainly. Recorded,
   not blocked - as graded in rounds 6-8.

4. Info (carried, owed) - the device/nvcc build and a card run stay owed (REPORT open question 1). My
   host check: `g++ -std=c++20 -fsyntax-only -Wall -Wextra` (CUDA include, `-DSTRATA_VERSION`) clean on
   both changed C++ files - the only warnings pre-existing and untouched by the diff (prefill.cpp:3141
   x2 -Wunused, generate.cpp:15825 -Wunused). The 6-argument `on_layer_pass` swap has no 5-argument site
   left (prefill.hpp:70, :341; prefill.cpp:5439 `sg.n`, :5440 `p0, T` where :3133 defines
   `T = min(C, n - c0)`, `p0 = pos0 + c0` - the chunk in flight's length and first position, the same
   alignment the batch segs pass as `job->pos0 + sg.off, sg.n`); the two RHB printfs are 11 specifiers /
   11 arguments at each site (generate.cpp:11332-11335, :16672-16678), the piece-boundary line prints
   `pos0=-1 chunk=0` (:11351-11355), and `EngineSideContract` (serve/test_read_heartbeat.py:421) pins
   both two-part literals.

5. Info (carried, documented) - the in-flight share stops, conservatively, in three shapes, all
   direction-safe (never above the engine's cumulative figure): after a pass fraction falls back on one
   unit (a split read's two cards - the flag at serve/server.py:4675-4683, gated by
   `SplitReadFollowsEngineFigure`), for a slice's legs after their BRA (the adapter shifts only `done`
   by the legs' base, serve/server.py:9951-9962, comment + REPORT open question 3), and for any beat
   whose `done` stands off `pos0 - reused`. The plan's never-0-after-the-first-RHB property therefore
   holds for whole reads and a read's FIRST leg - as the REPORT now states.

6. Info (carried, benign) - `_note_hb` rebinding `stat.read_hb` (:4646) and the share trackers
   (:4675-4683) runs outside `status_lock`. I re-verified the single-writer premise: the hooks are
   installed per request onto the request's own thread-local (serve/server.py:6338-6341) and the beats
   are routed from `_ReadCounted.get` inside the same worker's `_admission_wait` - beats, steps and the
   start hook for one stat are one thread; the pump thread only rebinds `read_live` entries; every write
   is a single GIL-atomic assignment and `status_lock` (a plain Lock, :3290) stays never-nested. Worst
   case one lost reset. Carried as benign, as in rounds 7-8.

## Manager gates (the brief's line), measured by my own probe, not asserted

`MangonelTrueReadout` replays the recorded engine line
(ref/mangonel-engine-after-p124v.txt: "prompt 27254 tokens = 0 reused + 27254 read in 17383 ms
(1567.8 tok/s)") - the file holds that summary only, so the replay is the line's own uniform shape
(REPORT open question 2). Independent re-measure through the real harness (scratch/review_r9_gate_probe.py,
75 RHB samples): 0 None/0 rows after the read's first RHB (row min 1,539.1, max 1,568.9); the readout's
time-average 1,561.3 = **-0.41% of the engine's own 1,567.9** (gate: within 5%); max 1,568.9 = 1.001x
(gate: <= 1.25x); `peak_prefill` 1,568.9; the read closes at exactly 27,254 counted. All five gates PASS.

The round-8 P1 fix, checked to discriminate (scratch/review_r9_red_check.py): on the committed tree the
restart->admission->retry shape reads 300.0 tok/s at every retry step (the engine's own 2,400/8.0 s);
simulating round 6's release (zeroing `read_count_reset` where `_read_start` used to clear it) re-spikes
the same shape to 3,500/2,700/2,220/1,900 - the gate's red-before-green holds at this HEAD without
editing the tree. The exclusion's coherence I re-derived: the snapshot and the closed intervals it
counts against both survive `_read_start` (which only closes and re-opens a leg), `min(excl, closed)`
tracks pruning (aging out reads LOW, never the spike), a second restart re-baselines the snapshot
upward, and the stat is the per-request object from serve/server.py:6290 - no cross-request leak.

## Verified clean (each checked myself)

- Modules run ONE at a time on /opt/llm/saguaro/.venv/bin/python, the three the plan names, nothing else:
  `serve.test_read_rate_true` 18 green (351.8 s), `serve.test_read_rate` 34 green (2.7 s - the module is
  untouched by the diff), `serve.test_read_heartbeat` 19 green (0.5 s).
- Tests exercise the change: the replays drive the real adapter (`_rhb`, `_admission_wait`, the
  thread-local hooks) and the real `Service._note_hb/_note_read/_own_read_rate`, sampling the row and
  `/metrics` figures; the gates assert never-0, the 5% average, the 1.25x cap, `peak_prefill`, counted ==
  27,254, counted <= the engine's cumulative figure at every split sample, monotone `read_pos`,
  never-re-opened intervals, the numerator re-baseline with the span, and the exclusion surviving the
  admission hook; the parse tests walk the absent / malformed / piece-boundary field shapes.
- Generated file: `tools/saguaro/extract_qfn_engine.py` re-run twice - `serve/qfn_engine.py`
  byte-identical, `git status` clean before and after; the driver change lives only in
  engines/qfn/serve/server.py:1727 `_rhb` (plain `int()` of the `%lld` strings - a `chunk=1e400` line
  reads as absent, no OverflowError escapes the pump; `pos0=-1 chunk=0` keeps its defined values and
  books `done` only).
- Threads and stdout: each RHB line is ONE `std::printf` + `fflush` under `rhb_mu` at both sites;
  `prog.report()`/`report_full()` (`prog.mu_`) are taken and released BEFORE `rhb_mu` - sequential, never
  nested; every other engine line type is byte-unchanged.
- Locks: `status_lock` is a plain Lock and never nests - `metrics()` runs `_prefill_tok_s()` OUTSIDE it
  (serve/server.py:5087-5089) and `_prefill_tok_s` calls only lock-free helpers under it; `_note_read`
  takes it internally; `_note_hb` takes none (so `_note_read`'s acquire cannot nest); `_read_span_s` /
  `_read_counted` / `_own_read_rate` never acquire.
- C++/server alignment: the guard's `pos0 - reused == done` matches both producers - solo
  (`p0 = pos0 + c0` vs `done` from `pp_from`) and batch (`job->pos0 + sg.off` vs `j.prog.done()` over
  `j.resume`); `ReadProgress::refresh_` (read_progress.hpp:89-99) gives no part credit until a boundary
  above the reporting unit is revealed (`b == c.p0`), so between boundaries the guard holds for every
  card of a split - the round-5 P2 flag exists exactly for that; staging beats (`stage == "stage"`) book
  no share, and the silence gate applies only to engines that beat (:4553-4559).
- No secrets and no paths outside the repo in the diff (the only `/opt/...` strings are the documented
  venv interpreter path in the reports); `scratch/` gitignored (:20); the tracked
  serve/fixtures/card-rc2-read-rate-timeline.txt is the manager's own /metrics capture; `ref/` inputs and
  `serve/test_read_rate.py` untouched; commits b7ed679..HEAD carry no Co-Authored-By / Claude-Session /
  Assisted-by trailers.
- The restart-tick blank (round 8's nit 5) and the warm-up lag of a cumulative readout (REPORT:130-136)
  stand as previously graded; nothing this round touches them.

## Note on this session

HEAD e719da96a (round-8 FIX-REPORT, comments/reports) sits on d4c2f245f (the P1 fix) - I reviewed the
whole range and re-ran every check against HEAD. My two scratch probes are gitignored; nothing else in
the tree was touched (git status clean apart from those).
