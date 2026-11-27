# P125d - a bounded shutdown (Saguaro 1.2.5) - report

Implements ref/REPORT-p125a-shutdown-hang.md section 4's bounded-shutdown proposal, the plan's three numbered steps,
one commit each (plus two fixups named as such). Nothing else from section 4 was taken (see Open questions).

## What changed

### Step 1 - every shutdown wait logs where it is (commits 293980bf9, 2ecdeed2b)

- **Server / QFN driver** - `engines/qfn/serve/server.py` `StrataEngine.close()` (and the re-extracted
  `serve/qfn_engine.py`): each of the three waits prints one line when it starts and one when the engine ended or
  the wait ran out - "stopping the engine: STOP + QUIT, waiting up to N s" / "the engine ended on QUIT after X s" /
  "QUIT did not end the engine after X s: terminate, ..." / "... killing, waiting up to N s" / "killed and reaped
  after X s"; the `EngineStuck` outcome is now also a one-line log before the raise (P125a §4's first bullet).
- **Engine** - `engines/qfn/src/program/generate.cpp`: the exit-time session save announces itself before it
  starts ("session dir: saving the live conversation and N slot(s) at exit (up to S s)"); the "M file(s), X MiB on
  disk at exit" line stays as its end line.
- **Session writer flush** - `SessionDir` (`include/strata/core/session_dir.hpp`, `src/core/session_dir.cpp`): new
  `flush_until(deadline, what)` - the unbounded `cv_.wait` of `flush()` replaced (for the exit path) by a
  `cv_.wait_for` to the deadline, with a start line ("flushing N session write(s) at exit") and an end line ("flush
  done after X ms" / "flush TIMED OUT after X s: N write(s) still queued"); `drop_queued()` gives up on the queue
  (clears jobs, marks dropped turn tracks broken, cancels the running write so it stops at its next block). The
  old silent `flush()` is kept for the idle-unpark path (not a shutdown wait).
- **Pinned release** - `engines/qfn/src/core/expert_source.cpp` `FileExpertSource::close()`: logs "releasing
  N GiB of pinned experts (cudaHostUnregister + unlock + free)" before the driver calls and "pinned experts
  released in X ms" after - the driver work itself cannot be bounded in userspace (P125a §4's last bullet).

### Step 2 - SAGUARO_SHUTDOWN_S, escalation without the dead wait (commit b3acbff1c)

- `close()` now ends the engine **QUIT -> wait -> SIGKILL**: the QUIT grace is `SAGUARO_SHUTDOWN_S`
  (`shutdown_budget_s()`, default 15 s, bad values fall back to the default, `0` ends at once), capped by
  `ENGINE_QUIT_WAIT_S`; when it runs out the process is SIGKILLed at once (`ENGINE_KILL_WAIT_S` still bounds the
  reap, `EngineStuck` on overrun). The `terminate()` step and `ENGINE_TERM_WAIT_S` are gone: the QFN engine
  `SIG_IGN`s SIGTERM under `--serve` by design, so that 5 s was always a wait that could not succeed (P125a §1.5,
  §3.3). `serve/qfn_engine.py` re-extracted via `tools/saguaro/extract_qfn_engine.py` (never hand-edited).
- Followed in `engines/qfn/serve/test_lifecycle.py` (the three `close()` tests now assert kill-without-terminate)
  and `engines/qfn/serve/test_lifeline.py` (`HardFallback` no longer patches `ENGINE_TERM_WAIT_S`); the new
  env var is documented in `docs/SETTINGS.md` beside `SAGUARO_ENGINE_WRAP_S`.

### Step 3 - the engine's exit save gets a budget (commits a1329b684, c043dff11)

- New engine flag `--session-dir-exit-s S` (default 10; parsed next to `--session-dir-delay`, help text extended).
  The whole exit save in `generate.cpp` runs against a deadline: `park_current`, then each slot's `persist_slot`
  (no new one starts past the deadline), then `flush_until(deadline)`. On expiry: `drop_queued()` and
  "session save skipped at exit after X s (--session-dir-exit-s): the conversations stay saved from their last
  turn" - the exit is never blocked by the save (P125a §3.1, the incident's dominant cost).
  Fixup c043dff11: `duration_cast` the span (a `time_point + duration<double>` floating time_point does not
  convert to `flush_until`'s - caught by the sm_75 build).

## Files changed

`engines/qfn/serve/server.py`, `serve/qfn_engine.py` (generated), `engines/qfn/include/strata/core/session_dir.hpp`,
`engines/qfn/src/core/session_dir.cpp`, `engines/qfn/src/core/session_dir_test.cpp`, `engines/qfn/src/core/
expert_source.cpp`, `engines/qfn/src/program/generate.cpp`, `engines/qfn/serve/test_lifecycle.py`,
`engines/qfn/serve/test_lifeline.py`, `serve/test_server.py`, `docs/SETTINGS.md`.

## Tests run (each module alone) and results

- `python -m unittest serve.test_server` - **183 OK** (179 before; the new `QfnBoundedShutdown` class: fake
  subprocess engines that ignore QUIT and SIGTERM are killed after the grace with the three log lines and
  `-SIGKILL`, a QUIT-honouring fake ends 0 with "ended on QUIT" and no kill line, `shutdown_budget_s` defaults /
  `0` / bad values, and a mock-proc test that the escalation never calls `terminate()`).
- `engines/qfn/src/core/session_dir_test.cpp` (built two ways: a direct g++ line and the CMake tree) - **81
  checks OK**, including 4 new ones: `flush_until` returns false at the deadline with writes queued and returns in
  well under one block, `drop_queued` + destructor end without publishing partial files.
- `scratch/build-p125d-sm75/turn_save_test` - **97 OK** (the unchanged `flush()` turn paths beside the new API).
- **QFN build for CUDA sm_75** in the new dir `scratch/build-p125d-sm75/` (`QFN_GGML_DIR=/data/llm/strata/
  third_party/llama.cpp bash tools/qfn/build.sh scratch/build-p125d-sm75` - the pinned llama.cpp 3cf03257f, since
  this worktree has no `third_party/llama.cpp`): `saguaro-qfn`, `libstrata_kq.so`, `libstrata_kq_tab.so` link
  clean (logs `scratch/build-p125d-sm75.{cmake,build}.log`); `saguaro-qfn --help` shows `--session-dir-exit-s`.
  No GPU runs.

Not runnable here: `engines/qfn/serve/test_lifecycle.py` / `test_lifeline.py` - the QFN subtree's Python tests
cannot import in this monorepo (`engines/qfn/serve/server.py` imports `serve.turbo_live`, which only the root
`serve/` has), so they fail at import on a clean tree too (verified by reading, pre-existing). They were updated
to the new behavior; their equivalents now run in `serve.test_server.QfnBoundedShutdown` against
`serve/qfn_engine.py`, the same code the shared server drives.

## Open questions

1. Sections 4 items **not** in the plan's three steps were left out, by design: registering the QFN engine with
   `track_spawned` (the orphaned-engine backstop), the timeout on the line engine's `with svc.fifo:` wait, a
   watchdog-visible shutdown phase + `_Exit` on `SAGUARO_SHUTDOWN_S` inside the engine, honouring a second SIGTERM
   in the serve loop, and bounded joins for the session/preloader/IO workers. The budgeted save bounds everything
   on the session path, but a stuck `device_sync` inside `park_current`/`persist_slot` (P125a §2c's one true
   deadlock) is still unbounded from within - only the deadline between steps; the server's SIGKILL at +15 s is
   the bound for it now. Want any of these as P125e?
2. `SAGUARO_SHUTDOWN_S` currently bounds only the engine driver's `close()`. The shared server's wrap phase keeps
   its own `SAGUARO_ENGINE_WRAP_S`; a total stop budget across wrap + close would touch every engine kind -
   confirm the narrower reading ("the whole server-side shutdown **of the engine**") is what was wanted.
3. On expiry the engine skips the flush and drops the queue, so a session-file **delta chain** resumes from the
   last landed commit (dropped turn tracks are marked `broken`, next save goes whole) - matches "saved from their
   last turn", but parked conversations that only the queue held are now simply not on disk; previously the
   destructor would still have written them. Acceptable trade (exit time vs durability) - say otherwise and the
   drop can be made exit-only (it already is: `drop_queued` is only called from the exit path).
4. `ENGINE_QUIT_WAIT_S` (20) now only caps `SAGUARO_SHUTDOWN_S`; the report's suggestion to size it from the
   pinned-release log lines needs a box with 54 GiB pinned experts - first data point wanted from the 9070 XT box
   after this lands.
5. Should `docs/SETTINGS.md`'s `SAGUARO_ENGINE_QUIT_S` (the line-protocol engine's 50 s) be reworded to point at
   `SAGUARO_SHUTDOWN_S` for the batch protocol too? Left as-is for now.
