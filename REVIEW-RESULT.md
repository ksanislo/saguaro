# P125a review — REPORT-p125a-shutdown-hang.md (report-only diff 17e6b5bb..58f71f6d)

VERDICT: MERGE-WITH-NITS

Scope: the diff adds only REPORT-p125a-shutdown-hang.md (192 lines, one commit 58f71f6d7) — exactly the plan's
"Change NO code; commit only the report" (PLAN-p125a-shutdown-hang.md:12). No code touched, so the OFF-path,
lock-nesting, thread/stdout-interleaving and generated-file rules are unbroken by construction:
serve/qfn_engine.py (generated — its own header, serve/qfn_engine.py:1-4, naming tools/saguaro/extract_qfn_engine.py,
which exists) and engines/qfn/serve/server.py are not in the diff. The plan names no test modules, so none were run
(REPORT:176 says the same). The report contains no absolute paths and no secrets.

Every load-bearing file:line in the report was opened and verified; the analysis is correct. Findings are
accuracy nits in the report itself — none changes the diagnosis or the candidate ranking.

## Findings (by severity)

1. LOW — REPORT:160 cites "writer/preloader (session_dir.cpp:69-70)" for the unbounded joins; the joins are
   engines/qfn/src/core/session_dir.cpp:67-68 (69 is the destructor's `}`). §2 (REPORT:87) cites the same destructor
   correctly as 60-71.

2. LOW — REPORT:108 cites "src/core/verify.cpp:455-472" for the 5 s/verifier bound; the function is
   engines/qfn/src/core/verify.cpp:463-474 (the 5000 ms cap at :467). The accompanying claim — watchdog/FATAL
   paths only, never the QUIT unwind — holds: the wrapper runs at generate.cpp:12887 (serve watchdog, aborts at
   12889) and 24033 (non-serve watchdog, aborts at 24035), 21652 (request-fatal `die`), pool.cpp:1175/1202; the exit
   block (generate.cpp:23222-23230) does not call it.

3. LOW — REPORT:36 "The terminate step is always a wasted 5 s": `terminate()` is reached only when the engine is
   still alive after ENGINE_QUIT_WAIT_S (serve/qfn_engine.py:2717-2720). True for the incident's shape; "always"
   should read "always when reached" — for an engine that exits within the 20 s QUIT grace the 5 s never runs.

4. LOW — REPORT:50 "worst case ~90 s" is loose for the QFN path. Summed from the cited waits: wait_for_stop ~0.5
   (serve/server.py:9202-9208) + httpd.shutdown ≤~0.5 (serve/server.py:10135, stdlib poll loop) + close() 20+5+20
   (serve/qfn_engine.py:59, 2717-2724) + vision ≤10+10 (serve/qfn_engine.py:3179, 3186) + atexit ≤~15 and in fact a
   no-op here (serve/server.py:9224-9266; the QFN engine is untracked, see verified item below) ≈ 65-70 s. The
   conclusion — the server cannot account for 6 minutes — is unaffected; any bound under two minutes proves it.

5. LOW/INFO — the "one true deadlock" (REPORT:89-95) is the report's least-supported claim, and reads stronger than
   the code it cites: `persist_slot` already refuses its device syncs while pipelined groups are in flight
   (generate.cpp:15639 sets `e = "pipelined groups in flight"`; the sync loop at 15646 runs only `e.empty()`), and an
   issued-but-unserved window is by definition "in flight", so the stated "slot mid-decode at the break" window is
   narrow — the report's own hedge ("possible hard hang", "not the incident's shape") is the right strength. The
   sibling hazard it rates at "seconds" — the pre-break `xcache_verify`/`adapt_tick(true)` (REPORT:69-70,
   generate.cpp:18681-18682) — can in principle be a spin-wait on a host flag only this thread raises (that is
   exactly why `release_live_verifiers` exists, verify.cpp:460-474), and it is equally watchdog-blind there
   (`busy` is set only from 18711-18722, after the 18709 break). §6.4's engine-log tail remains the honest arbiter
   for both.

## Verified (real file:line, all as the report cites)

- Server SIGTERM path is bounded for QFN: the stop line prints at serve/server.py:9199; a second SIGTERM is
  ignored serve/server.py:9280-9283 (only a second Ctrl+C escapes, :9281-9282); `wait_for_stop` is a 0.5 s poll
  :9202-9208; `httpd.shutdown()` :10135; the wrap-up block :10140-10154 is skipped — `hasattr(engine, "wrap")`
  (:10141) is false: `def wrap` exists only on the line protocol `Engine` at :2239 and `QfnEngine` (:9490,
  extending `QE.StrataEngine`, serve/qfn_engine.py:629) defines none; its `stop()` is a no-op (:9754-9755); the
  unbounded unlogged `with svc.fifo:` at :10153-10154 is line-engine-only, as REPORT:26-28 says; then
  `engine.close()` :10156, vision/hub :10157-10160.
- close() escalation as reported: `STOP\nQUIT\n` + flush + stdin EOF at serve/qfn_engine.py:2714-2716, waits
  20/5/20 via the constants at :59, `EngineStuck` at :2726; the driver knows the engine ignores SIGTERM
  (:918 comment). The 5 s terminate is a real no-op against the engine's `SIG_IGN` —
  engines/qfn/src/program/generate.cpp:18501-18502.
- atexit backstop is real but never sees the QFN engine: `track_spawned` is called only at serve/server.py:1481
  (line engine) and :2562 (shared vision encoder); no call exists in serve/qfn_engine.py — REPORT:40-43 holds,
  and it is what makes lifeline's kill-a-10-s orphaning real: the watcher ignores its own signals
  (serve/lifeline.py:99-100), forwards one SIGTERM (:117-118), SIGKILLs only the server pid after 40×0.25 s
  (:121-127), and the engine — which sees stdin EOF (generate.cpp:12546-12548, 18624-18625) and ignores
  SIGTERM — outlives everything.
- Engine teardown candidates all exist as described: STOP sets only `stop_req` on the reader thread
  (generate.cpp:12523), honoured at read-window boundaries :20184 and :20265; BSTOP :18631-18633, sent on client
  disconnect (serve/qfn_engine.py:1551-1560); the non-BSTOP pre-line `pipe_drain()` :18573-18601 (lambda
  :15606-15610) — the drain shape, correctly ruled out for the incident (:6-7: the done line at :6585-6586 prints
  only after DONE, :6509-6512); `xcache_verify("line: ...")` + `adapt_tick(true)` :18681-18682; QUIT break
  :18709 with the BusyScope starting only at :18711-18722 and the watchdog skipping everything while `!busy`
  (:12859) — the blind spot is real; exit save block :23222-23229 quoted verbatim, `park_current` :11028-11038,
  `persist_slot` :15629-15687 with its per-stage `device_sync()` :15646-15648, the :15611-15613 comment and the
  `park_body` copy at :15679; the <1 MB/s paged-state copy speed is the code's own measured note
  (src/core/conversation_copy.hpp:151-155); `SessionDir::flush()` is an unbounded `cv_.wait`
  (src/core/session_dir.cpp:228) over a queue capped at the default 64 GiB (generate.cpp:818, wired as
  `so.max_bytes` at :15725), and `~SessionDir` joins both threads with no timeout (:67-68); `save_profile("exit")`
  :23230 is logged and fast (lambda :12448-12464); the unpin runs at src/core/expert_source.cpp:1530-1537 behind
  the unbounded joins prefetch_end :2292-2295 / ram_stop :4459 / io_stop :1178-1179, reached from
  `~FileExpertSource` :939; the 54 GiB arena is registered by `cudaHostRegister` inside `pin_cache_complement`
  (defined :3459) at :3896/:3939, called at generate.cpp:8988-8996, selected by `--resident-pin` at
  :3219/:3232; the HIP warm join :2601-2603; the early-declared sdir comment :10531-10532.
- Candidate ranking (§3) checked against the code: with the server path bounded and the engine ignoring SIGTERM,
  the multi-minute span must be the engine's post-break teardown, and the only unbounded, unlogged,
  watchdog-blind waits there are the session exit save + flush and the GiB-scale unpin/joins — exactly ranks 1-2;
  §4's proposed timeouts target them at the right lines.

## Nits summary (for the coder, no action required to merge)

- Fix the two line refs (findings 1-2) and the "always" wording (3) if the report is ever touched again.
- Consider softening "the one true deadlock in the teardown" (REPORT:95) to match its own hedge, and noting that
  the pre-break verify/adapt waits (18681-18682) share the same unbounded shape (finding 5).
