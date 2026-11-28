# REVIEW-RESULT — P125k park-sync, round 5 (local agent, read-only) — 2026-11-27

VERDICT: MERGE-WITH-NITS. The P125k narrowing re-verifies sound, and the round-4 BLOCKER (the merged P125j at its
FIX-FIRST state) is resolved the way that review's second way out prescribed - the pause is armed only at the read
that captures it, and the idle grace is back at 5 - re-verified below from the sources, not from the commits' words.
What remains is four NITs, none blocking: two carried (the ungated restore tail owes one gated-out card run; the
light path's extra own-copy-stream sync waits more, never less) and two new low ones (a legacy-stream wait in the
no-copy-stream refresh fallback; one stale header line in the card script). The manager's card run (P125k's waits
falling to copy-ms during 2-3 busy conversations, plus P125j's layer_pause.sh) is still the gate that closes this
diff - nothing here ran on a GPU.

## Round-4 BLOCKER - resolved

* **Armed only where captured.** `Prefill::stop_between_layers` (prefill.hpp:351) has exactly one writer in the
  tree: read_part's run lambda (generate.cpp:20920-20924), which arms around its own `sp.run` and reads
  `paused_chunk()`/rows right after (20926-20931); the `PauseArm` guard (20920-20923) disarms on every way out,
  an exception included. The global serve-loop arm is gone (generate.cpp:12836-12843 is the comment that says
  why). The batch slot's reader has its own Prefill and lambda (generate.cpp:17144, never armed), the pipelined
  reader thread calls `sp.run` directly (generate.cpp:21176, the armed lambda is the inline/piecewise path at
  21361/21365), and `lp_on` (prefill.cpp:2978-2979) needs the hook - so `err "layer paused"` (prefill.cpp:5543)
  can now only come from the reader that keeps its rows. No hook data race: the reader-thread rounds never
  overlap a `run` lambda call, so the `std::function` member is written only while nobody else reads it.
* **"layer paused" cannot end the run with ERR.** Both `!sp_ok` error branches (generate.cpp:21949, :21956)
  require `!stop_req.load()`, and `stop_req` is set only by the stdin STOP (12799) and cleared only at the NEXT
  request's start (19107) - so a pause always routes to `cancelled` + the keep chain, never to a spurious
  `ERR layer paused` exit.
* **The keep pairs record with rows.** The pause keep (generate.cpp:23165-23207) requires
  `sp.pause_rows_bytes() > 0`; `give_pause` refuses rows-less records (prefill.cpp:1966-1974) and the adopt caps
  a diverging branch at the pause (20370-20395); `take_pause_rows` clears record+rows together (prefill.cpp:1953-
  1956); every reset/rewrite site pairs `drop_read_pause()` (generate.cpp:10553 and the ten call sites: file
  LOAD, head-alias x2, nan-flush, slot copy-in, borrow, incoming, cvec change, read-done, #1620 keep,
  checkpoint landing). A rowless park-write stays unreachable, so the "never written to the session dir"
  invariant (park gate generate.cpp:11220, `!parked_paused`) holds.
* **Grace back at 5.** serve/server.py:420 (`_env_seconds("SAGUARO_PRIORITY_IDLE_S", 5.0)`), the
  IdleGraceDefault assert at 5.0 (serve/test_priority.py:1612), and docs/SETTINGS.md:238 + the Request-priority
  paragraph (:247-253) + the `SAGUARO_LAYER_PAUSE` row (:2104) all say the same thing: the cut works only on the
  SOLO read that keeps its rows, the batch lane is never cut (`keeps_stopped_read and not batch`,
  serve/server.py:5845-5847), and the batch-lane cut + slot-reader capture stand named as P125j's owed fix
  round. With the default at 5 the batch/piped shapes behave exactly as pre-P125j; `SAGUARO_LAYER_PAUSE=0`
  keeps the whole feature inert (`lp_on` false, prefill.cpp:2978-2979 + `layer_pause_wanted()` prefill.cpp:1946).

## P125k (the plan's subject) - re-verified sound

* **The decision and the wait.** `park::wait_streams` (park_policy.hpp:216) filters the scope to the device the
  wait runs on; `conversation_detail::narrow_sync` (conversation_copy.hpp:312-325) waits this thread's private
  copy stream plus that device's share - no `cudaDeviceSynchronize`, and the exclusive capture lock is never
  taken (override check precedes it in `device_sync()`, capture_lock.hpp:48-52, and `sync()` waits the copy
  stream first, conversation_state.cpp:36-41). No lock nesting anywhere on the narrowed path: the only lock it
  touches is the handle pool's short `std::mutex` inside `private_stream` (conversation_copy.hpp:83-104), never
  held across a sync. The override and its arg are `thread_local` (capture_lock.hpp:39-46): slot/pipelined
  reader threads never see a scope.
* **Event/land-before-reuse (manager note) holds at every replaced wait.** Source cells: the gated scope
  (main_stream + ver/ver_b + mtp.stream(), generate.cpp:15939-15950) is this conversation's whole producer set -
  the main Prefill's compute stream IS the scoped `main_stream` (generate.cpp:4500 `main_cs` -> `sp.init`
  :9741 -> prefill.cpp:1542), so the layer-pause resume's H2D (prefill.cpp:3227-3237) and every K/V write ride
  a waited stream; the verifier side streams fork/join through `ver`/`ver_b`, the shared drafter (a slot's
  batched draft prefill into the main draft pool) is `mtp.stream()`. Restored cells: every restore copy rides
  the private stream and lands per copy (`copy` -> `copy_nonblocking` with `queued=false`,
  conversation_state.cpp:60-67 / conversation_copy.hpp:126-139), the residency refresh queues on that same
  stream behind the pool copies (conversation_snapshot.cpp:339-352, 421, 446), and `copy_done`
  (conversation_state.cpp:47-58, :257) waits it - so bytes land before the restore returns in every shape,
  gated or not. Park copies land before the park returns the same way, so a source cell cannot be reused
  (anything queued after the return) before its copy read it.
* **The arming sites** are RAII on the sequential serve thread - context restore generate.cpp:19416, the
  switch's park :20133, the borrow :20248, the incoming restore :20307, the cut's drain+restore :23237, the
  turn save :23588 - never nested, and nothing needing a whole-device drain runs inside one: `copy_from_slot`
  sits outside the park scope (closed :20165 vs opened :20177) and the slot/admission device syncs stand
  (:14490-:14563). The gate `park_own_ok` (:15933) excludes layer split, pipe, concurrent doorbell serving and
  a read running beside - every shape whose writers this stream list cannot bound; those keep the whole-device
  sync. The FAST_LANE light save reuses the same machinery (:15962).
* **stdout**: no format moved. The park/restore `split: ... sync N ms` still comes from SyncRec ->
  `snapshot_split().sync_ms` (conversation_state.cpp:28-32, `copy_done` records into it); the new engine lines
  (prefill.cpp:5537, generate.cpp:23200) are single `fprintf(stderr)+fflush`; the DONE suffix rides as optional
  `key=value` fields parsed in both engine parsers (engines/qfn/serve/server.py:1666,
  serve/qfn_engine.py:1222) - old lines parse unchanged and unknown keys stay ignored; the server's pause line
  keeps its exact pre-P125j wording unless a fresh DONE carries `paused_layer` (serve/server.py:5974-5988,
  `fresh is not last0` guards against a stale dict).
* **Generated file**: ran `tools/saguaro/extract_qfn_engine.py` - serve/qfn_engine.py regenerates
  BYTE-IDENTICAL, tree clean after. The qfn_engine diff is only the three parse keys, mirroring its source.
* **HIP (manager note)**: every CUDA call the diff adds is shimmed - `cudaMemcpy` :107, `cudaMemcpyAsync` :108,
  `cudaStreamSynchronize` :134, `cudaGetLastError` :82, `cudaGetDevice` :84, and the event pair the stream-side
  ordering leans on (`cudaStreamWaitEvent` :136, `cudaEventRecord` :144 -> `hipStreamWaitEvent`/`hipEventRecord`)
  in engines/qfn/include/strata/platform/hip_compat/strata_hip.h. No unshimmed symbol in the diff.
* **Paths/secrets**: none in the diff's code; the only absolute path added is the overridable
  `PYTHON=${PYTHON:-/opt/llm/saguaro/.venv/bin/python}` in tools/card_tests/layer_pause.sh:32, matching the
  existing card-script convention (lost_read.sh, mortar_pressure.sh). The round-4 NIT 3 build-path prose is
  normalized (REPORT-p125k-park-sync.md:128 now `../maya/third_party/llama.cpp`, the nvcc line descriptive;
  FIX-REPORT.md:109 names the venv descriptively).

## NITs

1. (carried, r4 NIT 1) The restore tail `copy_done` (conversation_state.cpp:257) and the copy-stream residency
   refresh (:421/:446) are UNGATED - the gated-out shapes wait less after their copies than before (the §5
   contract: correct, never under-waited, the kept device syncs still drain). The card owes one gated-out-shape
   run (layer split or multi-card) before this ships.
2. (carried, r4 NIT 2) The FAST_LANE light path now also syncs this thread's private copy stream
   (ConvStreams::run -> narrow_sync) - waits more than the old `LightSync::run`, never less; documented in
   REPORT §2 (:84-89). No change asked.
3. (new, low) The no-copy-stream refresh fallback (conversation_snapshot.cpp:354-358) launches on the LEGACY
   stream and `cudaStreamSynchronize`s it WITHOUT the capture lock; the legacy-stream launch there carried the
   same capture-collision exposure before this branch (the hazard conversation_copy.hpp:76-80 documents is why
   the snapshot copies avoid the legacy stream), and the old tail wait - a device sync - held capture_mutex.
   Reachable only when a thread cannot get a copy stream (#2409, full card) beside a live capture; the failure
   mode is a loud restore error, not corruption. A comment or a card/nightly note would close it; not code.
4. (new, wording) tools/card_tests/layer_pause.sh:13-14 still says "SAGUARO_PRIORITY_IDLE_S=0 (P125j's
   defaults)" - after the round-4 revert 0 is no longer the default; the script's own
   `SAGUARO_PRIORITY_IDLE_S="${IDLE:-0}"` (:104) keeps the run correct, so this is prose only.

## What ran here (one at a time, nothing on a GPU)

* `park_policy_test` (the only test module this plan names), fresh from the tip sources:
  `g++ -std=c++20 -I engines/qfn/include engines/qfn/src/core/park_policy_test.cpp -o scratch/review-r5/ppt_r5`
  -> **65 checks passed**; and as the `scratch/build-p125k-merge2` target (ninja: no work to do at tip) ->
  **65 checks passed**.
* `tools/saguaro/extract_qfn_engine.py` -> serve/qfn_engine.py byte-identical, `git status` clean.
* `py_compile` on serve/server.py, serve/test_priority.py, serve/qfn_engine.py, engines/qfn/serve/server.py;
  `bash -n` on tools/card_tests/layer_pause.sh - all clean.
* `layer_pause_test` and `serve.test_priority` were deliberately NOT run - not named by this plan; the P125j
  modules stay that branch's card's court.
