# REVIEW-RESULT — P125k park-sync, round 4 (local agent, read-only) — 2026-11-27

VERDICT: FIX-FIRST. The P125k narrowing itself is sound - gated shape, wait-set, order guarantees, locks, generated
file, HIP mapping and the plan's test all verified below - but the diff ships the merged P125j at its round-4
FIX-FIRST state: the round-5/6 fixes this file's round-3 instruction named do not exist in any ref this repo can
reach, and the layer-pause default flip rides the diff (pause ON, grace 5 -> 0) into the shape where the pause
never fires.

## BLOCKER

1. **P125j ships unfixed, and cannot be re-merged** (manager note said its round-6 review passed - not from here):
   `saguaro/q1-p125j-layer-pause` tip is d8e25fd95, its OWN round-4 FIX-FIRST review commit, already in HEAD via
   merge 70f67efde; no remote branch carries p125j (`git branch -r` shows fork/* only) and no dangling commit mentions
   it (`git fsck`). The 01aa548cb account is accurate - and the round-4 findings therefore stand in the diff:
   * `serve/server.py:5846-5847` - read preemption requires `keeps_stopped_read and not batch` (and
     `serve/server.py:9600-9602`: "the preemption never cuts a batch read"), so under `--batch` the layer pause
     never fires; the read runs to its end beside a waiting person.
   * `serve/server.py:417-420` - `SAGUARO_PRIORITY_IDLE_S` default is 0.0 (was 5.0; serve/test_priority.py:1606-1610
     asserts 0.0): the follow-up grace is removed exactly in the config nothing compensates for. docs/SETTINGS.md:238
     tells operators to set it back to 5 with `SAGUARO_LAYER_PAUSE=0`, but the DEFAULT ships 0 with the pause ON
     (`docs/SETTINGS.md:2104`) and inert on the batch lane - the round-4 gate was "0 only with a working pause".
   * Engine side: the pause capture is inline-only - generate.cpp:20915 (`sp.paused_chunk()`) sits in
     `read_windows` (generate.cpp:20775); the pipelined reader `read_windows_pl` (generate.cpp:20679) captures
     nothing, so a STOP-caused pause there surfaces as the read error "layer paused"
     (engines/qfn/src/prefill/prefill.cpp:5543) instead of a cut read that keeps its rows.
   To merge: land the real P125j fix round, or revert the grace default to 5 (and arm the pause only where it is
   captured) until one exists. P125k itself does not depend on P125j and could ship without it.

## P125k (the plan's subject) - verified sound

* The decision is pure and tested: `park::wait_streams` (park_policy.hpp:216) keeps only the current device's share
  of the scope; `conversation_detail::narrow_sync` (conversation_copy.hpp:312) waits that set plus this thread's
  private copy stream - no `cudaDeviceSynchronize`, no capture lock (correct: only device syncs invalidate captures,
  capture_lock.hpp:1-3, and `device_sync()` skips the lock whenever the override is set, capture_lock.hpp:48-52).
  No lock nesting: the narrowed path takes nothing but the handle pool's short `std::mutex` inside
  `private_stream` (conversation_copy.hpp:83-104, never held across a sync). `ConvSyncScope`
  (generate.cpp:15913-15923) is RAII on a thread_local override - exception/`return 1` paths clear it; the six
  armed sites are sequential, not nested (generate.cpp:20128 park, 20243 borrow, 20302 incoming restore,
  19411 context restore, 23224 cut, 23575 turn save; gate `park_own_ok` generate.cpp:15929).
* Event/land-before-reuse order (manager note) holds on every replaced wait:
  * source cells: the pre-copy sync covers the conversation's producers - `main_stream`, `ver`/`ver_b`,
    `mtp.stream()` (add_main_streams, generate.cpp:15935-15946); verifier side streams fork/join THROUGH their
    waited stream (verify.cpp:1198-1205, fork :1474/:1486, joins :1544-1549), so waiting it waits them; a slot
    without its own drafter writes the MAIN draft pool on the shared drafter, which is in the set
    (park_draft, generate.cpp:10761-10763); `mtp.kv_restore` syncs its own stream directly (mtp.cpp:1407-1415),
    unaffected; the queued PLE pipeline is self-contained inside the run (upload + event on `m.cs` before its
    readers, prefill.cpp:3316-3346).
  * restored bytes: every restore copy rides the private stream (conversation_state.cpp:60-67), the residency
    refresh launches on the SAME stream, stream-ordered after the pool copies
    (conversation_snapshot.cpp:338-355, 421, 446), and is waited by `copy_done`
    (conversation_state.cpp:257)/`narrow_sync`'s copy-stream leg/`copy_stream_done` - or, on the paths that keep
    their device syncs (slot copy/admission, generate.cpp:13708-13726), by those; the no-copy-stream fallback
    (#2409) launches legacy and waits there immediately (conversation_snapshot.cpp:349-353). Never less drained
    than before at any site.
* stdout: no P125k output changes - the park/restore `sync N ms` still comes from SyncRec ->
  `snapshot_split().sync_ms` (conversation_state.cpp:28-32, and `copy_done` records into it), formats at
  generate.cpp:11522/:23542 untouched; P125j's DONE suffix parses as optional key=value (qfn_engine.py:1220-1226),
  old fixtures unaffected.
* Generated file: ran `tools/saguaro/extract_qfn_engine.py` - serve/qfn_engine.py regenerates BYTE-IDENTICAL
  (git status clean after).
* HIP: every new call is shimmed - cudaStream_t strata_hip.h:33, cudaGetLastError :82, cudaGetDevice :84,
  cudaStreamSynchronize :134; the event pair the order relies on was already mapped (cudaStreamWaitEvent :136,
  cudaEventRecord :144). No new unshimmed symbol in the diff.
* No secrets; no absolute paths in code - the only ones in added lines are prose build commands in the reports
  (see NIT 3) and `PYTHON=${PYTHON:-/opt/llm/saguaro/.venv/bin/python}` in tools/card_tests/layer_pause.sh,
  matching the existing convention (lost_read.sh, mortar_pressure.sh).

## NITs

1. Two P125k changes are UNGATED and so are not byte-identical for the gated-out shapes: the restore tail
   `copy_done` (conversation_state.cpp:257) and the copy-stream residency refresh
   (conversation_snapshot.cpp:421/446). Correct per the contract (the device-wide pre-sync still drains everything,
   and each kept site still device-syncs), and REPORT §2 now says so honestly (fixed at f3d89c6c7) - but the card
   run should cover a gated-out shape (layer split or multi-card) once, since that tail rides those too.
2. The FAST_LANE light path is likewise no longer byte-identical: `ConvStreams::run`
   (generate.cpp:15907-15911) -> `narrow_sync` adds a sync of this thread's private copy stream the old
   `LightSync::run` did not make. Documented; it waits MORE, never less.
3. REPORT-p125k-park-sync.md:128-129 cites out-of-repo build paths (`/data/llm/maya/third_party/llama.cpp`,
   `/usr/local/cuda-13.1/bin/nvcc`) in committed prose (FIX-REPORT.md does the same with the venv). Docs only -
   normalize to a relative/env form if house style wants reports portable.
4. The unit test exercises the pure wait-set decision (`wait_streams`, park_policy_test.cpp:101-133); the real
   wiring (override hook, arming sites) is proven only by the build + the card run. The plan explicitly allows
   this ("a unit test where the code allows it").

## What ran here (one at a time, nothing on a GPU)

* `park_policy_test` (the only test module this plan names), fresh from tip sources:
  `g++ -std=c++20 -I engines/qfn/include engines/qfn/src/core/park_policy_test.cpp` into scratch/review-r4/ ->
  **65 checks passed**; and as the `scratch/build-p125k-merge2` target (ninja: no work to do at tip) ->
  **65 checks passed**.
* `tools/saguaro/extract_qfn_engine.py` -> serve/qfn_engine.py byte-identical, tree clean.
* `layer_pause_test` was deliberately NOT run (not named by this plan).
