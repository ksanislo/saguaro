# P125a — why the QFN server took > 6 min to exit after SIGTERM (report only, no code changed)

Seen 2026-11-26 10:18 MST (9070 XT box): test server, QFN UD-Q4, `--max-context 1048576 --parallel 2`,
resident RAM mode (54 GiB pinned experts). A 76K-token prompt read was cancelled at the client's 900 s
timeout — `[saguaro] done: 0 tokens in 926 s ... (length, cancel=True)` (printed by
serve/server.py:6585-6586; that line only prints once the engine's BDONE/DONE arrived, serve/server.py:6509-6512,
so the engine had finished the read at ~926 s). SIGTERM printed `[saguaro] stopping (SIGTERM): the engine ends`
(serve/server.py:9199) — the last line; the process was still alive 6+ min later and `kill -9` ended it.

## 1. What the SERVER waits for on a SIGTERM (all bounded for the QFN path)

Order, with the file:line of each wait:

1. **Signal handler** — serve/server.py:9272-9294 (`stop_signals`). A *second* SIGTERM while stopping is
   ignored (serve/server.py:9280-9283): the supervisor's repeat signals add nothing. The handler only sets
   the stop flag and prints the log line (serve/server.py:9195-9199).
2. **`wait_for_stop`** — serve/server.py:9202-9208: a 0.5 s poll loop; ends within ~0.5 s. Bounded.
3. **`httpd.shutdown()`** — serve/server.py:10135. `ThreadingHTTPServer` (serve/server.py:8697) whose
   `serve_forever` runs on a daemon thread (serve/server.py:8807/9057, handler swap at 9061). socketserver's
   `shutdown()` itself waits on an Event with *no timeout*, but the loop notices the flag within its 0.5 s
   poll and handler threads are daemons — bounded in practice.
4. **The wrap-up block** — serve/server.py:10140-10154. **Skipped for QFN**: `hasattr(engine, "wrap")`
   (serve/server.py:10141) is false — `wrap()` exists only on the line-protocol `Engine`
   (serve/server.py:2239-2247); the QFN driver (`QfnEngine` extends `StrataEngine`, serve/server.py:9490,
   serve/qfn_engine.py:629) has no `wrap`. Its `stop()` is a no-op (serve/server.py:9754-9755).
   *Latent hang, other engines:* after `engine.stop()` the server does `with svc.fifo:`
   (serve/server.py:10153-10154) — an **unbounded, unlogged** wait for the FIFO the cut request still holds.
   Not reached for QFN, but it is the classic "never ends" shape in this path.
5. **`engine.close()`** — serve/server.py:10156 → serve/qfn_engine.py:2703-2743. Writes `STOP\nQUIT\n`,
   flushes, closes stdin (serve/qfn_engine.py:2714-2716), then:
   `proc.wait(ENGINE_QUIT_WAIT_S)` = **20 s**, `proc.terminate()` + `wait(ENGINE_TERM_WAIT_S)` = **5 s**,
   `proc.kill()` + `wait(ENGINE_KILL_WAIT_S)` = **20 s** (constants: serve/qfn_engine.py:59; raises
   `EngineStuck` at 2726 if it still runs). **The terminate step is always a wasted 5 s**: the QFN engine
   `SIG_IGN`s SIGTERM/SIGINT under `--serve` by design
   (engines/qfn/src/program/generate.cpp:18501-18502) — the driver itself knows
   ("the engine ignores SIGTERM in --serve", serve/qfn_engine.py:918). Bounded: ~45 s worst case.
6. **`vision.close()` / `hub.close()`** — serve/server.py:10157-10160; the QFN vision close waits ≤ 10 s
   twice (serve/qfn_engine.py:3171-3186); MCP close only if configured.
7. **atexit `end_spawned`** — serve/server.py:9224-9269: SIGTERM → 5 s → kill → wait; bounded.
   **But it never sees the QFN engine**: `track_spawned` is called only for the line-protocol engine
   (serve/server.py:1481) and the shared vision encoder (2562); the QFN driver's `Popen`
   (serve/qfn_engine.py, `StrataEngine` start) is never registered — if the server dies without reaching
   `close()`, the engine is simply orphaned.
8. **If started under `saguaro.py`/the test harness (lifeline)**: serve/lifeline.py:99-100 ignores the
   signals, forwards one SIGTERM (serve/lifeline.py:117-118), polls 10 s and then **SIGKILLs only the
   server** (serve/lifeline.py:119-128). The engine is never touched by lifeline.

**Conclusion for the server:** every Python-side wait on the QFN path is bounded — the server itself cannot
stay up 6 minutes after printing the stopping line (worst case ~90 s: 20+5+20 in `close()`, an
`EngineStuck` traceback, then atexit). The process observed "still running 6+ minutes later" is the **QFN
engine (C++) process** — it ignores SIGTERM (generate.cpp:18501-18502), and either the server was killed
before `close()` escalated (lifeline SIGKILL at +10 s → engine sees stdin EOF, generate.cpp:12546-12548,
18624-18625, and starts the same teardown), or the engine was SIGKILLed but its teardown/kernel release
kept it from being reaped. Either way, all the multi-minute waits are on the **engine** side.

## 2. What the ENGINE waits for after STOP/QUIT (the multi-minute candidates)

All line refs: engines/qfn/src/program/generate.cpp unless noted.

- **STOP is cheap**: the stdin reader never queues it, it just sets `stop_req` (12521-12523), checked at
  every prompt-window boundary — `fail("cancelled")` at 20184, 20265. The driver also BSTOPs a slot the
  moment its client disconnects (serve/qfn_engine.py:1551-1560; engine side 18631-18633). A cancelled read
  ends within one window.
- **(a) Drain of an in-flight read before QUIT is obeyed — 18573-18601.** When a QUIT line arrives while a
  pipelined prompt read is in flight, the loop takes the line (18574) but, because QUIT is not `BSTOP`,
  runs `pipe_drain()` first (18591, lambda 15606-15610) — "the groups in flight finish first". With the
  STOP sent just before QUIT (serve/qfn_engine.py:2714) the drain stops at the next window (20184); without
  a prior STOP it is the **full remainder of the read** — the "drain of a cancelled read" shape.
- **(b) `xcache_verify("line: ...")` + `adapt_tick(true)` before the break — 18681-18682.** Both wait for
  expert-swap/verify rounds still in flight against the resident tier; GiB-sized copies, seconds.
- **(c) The QUIT break — 18709 — and the whole teardown after it.** No request "busy" flag covers it
  (BusyScope is entered only *after* the break, 18710-18722), and the watchdog thread ignores everything
  while `!busy` (12855-12859; include/strata/core/progress.hpp:1-7). **Nothing bounds or logs any of the
  following — this is the shutdown blind spot:**
  - **Session save of a huge conversation — 23222-23229** (only when `--session-dir`):
    `sdir->begin_exit(); park_current(0); for each batch slot persist_slot(b, ..., "the engine ends");
    sdir->flush();`. `park_current` (11028-11038) and `persist_slot` (15629-15687) do a **device sync per
    stage (15646-15648) plus a device→host copy of the live conversation's state** (`park_body`, 15679):
    a 76K-token conversation (K/V segments + checkpoints, GiB-scale at this context) copied into *pageable*
    host RAM — which on a box holding 54 GiB pinned experts + 1M-token slot buffers is paged out; the
    documented page-in copy speed is **under 1 MB/s, one memcpy past 60 s** (src/core/conversation_copy.hpp:151-155,
    #2049/#2065/#1971). Minutes for a GiB image, and the exit path prints nothing until it is done.
  - **Unbounded queue drain — `SessionDir::flush()`, src/core/session_dir.cpp:224-230**: a `cv_.wait` with
    **no timeout** for every queued session-file write; queued images are capped at 64 GiB
    (generate.cpp:817-819 defaults `--session-dir-gib 64`); the writer comment says a hung write "still ends
    in the watchdog" (session_dir.cpp:155) — but the watchdog is blind here (see above), so the safety net
    does not exist on the exit path. `~SessionDir` then joins the preloader and writer threads with no
    timeout (session_dir.cpp:60-71; declared early, "destroyed before the cache", generate.cpp:10531-10532).
  - **Possible hard hang — `persist_slot` at exit with a still-active slot**: the exit loop (23225) calls
    it for every slot; its stage `device_sync()`s (15646-15648) would wait on GPU windows that only this
    serve thread steps (its own comment, 15611-15613: "the device syncs would wait on windows only this
    thread serves" — respected for pipelined groups, but a `--parallel` slot mid-decode whose STOP was not
    yet honoured at the next step never gets stepped after the break). If QUIT arrives while a slot is
    decoding, this can block forever. Not the incident's shape (the read had ended at 926 s), but it is the
    one true deadlock in the teardown.
  - **Expert profile save** — `save_profile("exit")` (23230, lambda 12448-12464): small file, fast, logged.
  - **Unpinning tens of GiB — `~FileExpertSource`** (src/core/expert_source.cpp:939 → `close()` 1525-1537,
    runs when the serve scope unwinds): first joins the prefetch/RAM/IO worker threads **with no timeout**
    (`prefetch_end` 2292-2295, `ram_stop` 4459, `io_stop` 1169-1181), then
    `cudaHostUnregister` + `unlock_resident` + `free` of the complement arena — **this box's 54 GiB pinned
    experts** (registered at expert_source.cpp:3896/3939 via `pin_cache_complement`, called at
    generate.cpp:8988-8996 with `--resident-pin`, chosen at 3219/3232). Driver/kernel teardown of tens of
    GiB of page-locked memory: seconds to minutes, **no timeout, no log line**. The server's own code
    acknowledges the same cost after the fact: "the driver frees tens of GB of VRAM and its pinned RAM tier
    after the process is gone" (`wait_gpu_release`, serve/server.py:221-253 — used only for line-engine
    reloads, not shutdown).
  - **GPU release / misc joins**: `release_live_verifiers` is bounded at 5 s per verifier
    (src/core/verify.cpp:455-472) but is only run on the watchdog/FATAL paths, not on the normal QUIT
    unwind; the HIP warm thread is joined at exit (2601-2603); device-reset work happens in the kernel after
    the return — which is also why a SIGKILL can leave the process listed until the driver finishes.

## 3. Most likely cause of the 6+ minutes

Ranked:

1. **The engine's QUIT/EOF teardown, dominated by the session-dir exit save** (23222-23228 +
   session_dir.cpp:224-230): parking the 76K-token live conversation and the two kept batch slots
   (device→host copies into paged-out RAM at <1 MB/s, conversation_copy.hpp:151-155) plus the queued
   session-file writes, all unbounded, unlogged, and watchdog-blind. This is exactly the configuration that
   makes it slow: `--session-dir` writer active, 1M-token slots, 54 GiB pinned experts leaving the parked
   state swapped out.
2. **The 54 GiB pinned-expert release after that** (expert_source.cpp:1530-1537) — minutes of driver work,
   silent, and the reason a SIGKILL's reap can itself be delayed.
3. **The server's escalation being a no-op**: 20 s QUIT grace < (1)+(2); then SIGTERM → *ignored*
   (18501-18502) — wasting ENGINE_TERM_WAIT_S — and only then SIGKILL. The supervisor sees the engine
   process the whole time; "kill -9 ended it" is consistent (the server's own kill, or lifeline's SIGKILL of
   the server at +10 s orphaning the engine entirely — nothing else ever kills it: `_SPAWNED` never contains
   the QFN engine, serve/server.py:1481/9218-9221 vs. its absence in serve/qfn_engine.py).
4. Least likely: the in-flight drain before the break (18591) — the incident's read had already printed
   its done line at 926 s, so nothing was in flight when QUIT arrived.

The server-side code cannot produce a 6-minute wait on this path; the "server took > 6 min to exit"
observation is the engine process (SIGTERM-ignoring child) still tearing down — and that matches every
candidate the plan lists: an unbounded wait (sdir flush), a drain (only in the in-flight case), unpinning
tens of GiB, a session save of a huge conversation, and thread joins with no timeout (writer/preloader/IO).

## 4. Proposed bounded shutdown (each wait a timeout + a log line — not implemented here)

Server (serve/qfn_engine.py / serve/server.py):

- Log each step of `close()` (it is silent end to end): "QUIT sent, waiting 20 s" → "engine ignores
  SIGTERM: terminate is a no-op, killing after 25 s" → "killed, waiting 20 s" → the EngineStuck text as a
  one-line log, not a bare traceback. (serve/qfn_engine.py:2703-2743.)
- Register the QFN engine (and elastic vision) with `track_spawned` so `end_spawned` (serve/server.py:9224)
  is a real backstop when `close()` never runs — today the engine is orphaned by a SIGKILLed server.
- Give the FIFO wait for line engines a timeout and a log (serve/server.py:10153-10154).

Engine (engines/qfn):

- **A shutdown phase the watchdog sees**: after the QUIT/EOF break (18709), set `progress().busy` with
  `where="shutdown"` (or a dedicated deadline), and on `SAGUARO_SHUTDOWN_S` (propose ~15 s) run
  `release_gpu_waits` and `_Exit`, naming the stuck phase — closing the blind spot at 12855-12859.
- **Session save with a budget** (23222-23228): log "saving the live conversation (N MiB) and M slot
  files at exit"; cap the whole save+flush with `--session-dir-exit-s` (propose 10 s): on expiry, skip
  the flush, drop queued jobs, log "session save skipped at exit after Xs" — and make `SessionDir::flush`
  take a deadline (`cv_.wait_for`, session_dir.cpp:228) instead of an unbounded wait.
- **Never device-sync an unserved slot at exit**: in `persist_slot`, when `exiting` and the slot is
  active/reading or windows are in flight, skip with a log instead of syncing (15646-15648).
- **Bounded joins**: writer/preloader (session_dir.cpp:69-70) and prefetch/ram/io workers
  (expert_source.cpp:2292-2295, 4459, 1169-1181): `wait_for`-style deadlines; detach + log on overrun.
- **Log the unpin**: "releasing N GiB of pinned experts" before `cudaHostUnregister`/`unlock_resident`
  and the elapsed ms after (expert_source.cpp:1530-1537) — the driver call itself can't be bounded in
  userspace, but the next box would then know where the minute went, and the server can size
  ENGINE_QUIT_WAIT_S against the logged numbers instead of guessing 20 s.
- **Honour a second SIGTERM once in the serve loop**: after QUIT (or a second SIGTERM while quitting),
  call `release_gpu_waits` and `_Exit` — giving supervisors a lever between SIGTERM and SIGKILL instead of
  only SIGKILL (18501-18502).

## 5. Files examined / changed

- Read only (no code changed): serve/server.py, serve/qfn_engine.py, engines/qfn/serve/server.py,
  engines/qfn/src/program/generate.cpp, engines/qfn/src/core/session_dir.cpp,
  engines/qfn/src/core/expert_source.cpp, engines/qfn/src/core/verify.cpp,
  engines/qfn/src/core/conversation_copy.hpp, engines/qfn/include/strata/core/progress.hpp,
  engines/qfn/include/strata/core/session_dir.hpp, serve/lifeline.py, serve/mcp.py, serve/test_stop.py.
- Tests: the plan names no test modules ("Change NO code"); none were run.

## 6. Open questions

1. Was `--session-dir` actually on for that test server (the config isn't in the report)? Without it the
   23222 block is skipped entirely and candidate 1 shrinks to the park/save of the conversation via the
   conversation cache plus the unpin (candidates 2). Check the engine log (`log` in the config) after the
   stopping line: "saving slot N's conversation", "session dir: ... at exit", "expert profile saved" pin
   exactly where it stopped.
2. Which supervisor sent the SIGTERM (llama-swap, lifeline/saguaro.py, systemd)? The lifeline path
   (serve/lifeline.py:119-128) SIGKILLs the *server* after 10 s and orphans the engine — that alone
   explains "the server stopped logging but a process ran 6 more minutes".
3. The exact build (the quoted done line lacks the current `(N tok/s)` field, serve/server.py:6585) — if
   the box ran an older server, its `close()` timeouts may differ from serve/qfn_engine.py:59.
4. The engine log's tail would confirm whether it was stuck in the session save, the unpin, or a GPU sync
   (candidate (c)'s active-slot sync) — worth adding `--session-dir-preload`-style progress to the exit
   path regardless.
