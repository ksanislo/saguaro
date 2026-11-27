# REVIEW-RESULT - P125k park-sync (diff b7ed679cb..dc436f2f0, this worktree)

MERGE-WITH-NITS

Scope reviewed: PLAN-p125k-park-sync.md / REPORT-p125k-park-sync.md against
`git diff b7ed679cbc75cf1959bb1d793e8442534ac4a5f3..HEAD`. The range also carries merged
P125j (merge 70f67efde + merge-round FIX-REPORT dc436f2f0); the merge itself is checked
below, P125j's own findings are its branch's.

## What was verified (with the evidence)

- Narrow wait-set correctness. `park::wait_streams` (park_policy.hpp:216) filters the
  (device, stream) scope to the waiting device plus this thread's private copy stream
  (conversation_copy.hpp:312-324). The gated set (generate.cpp:15935-15944: `main_stream`,
  `ver`, `ver_b`, `mtp.stream()`) covers every producer of the parked/restored cells in the
  gated shape: session calls ride `main_stream` (generate.cpp:4495-4497, non-blocking);
  the verifier side streams fork/join through their own stream (verify.cpp:1199-1204,
  1478-1485, 1542-1549 - join is `cudaStreamWaitEvent(cs, join_, 0)`, so syncing `cs` waits
  the branches); the drafter's `side_` likewise forks at mtp.cpp:1146 and joins at
  mtp.cpp:1189; batch-slot windows and doorbell-served windows ride `ver`/`ver_b` (they are
  in the set: generate.cpp:18344-18350 shows the Bells thread serving the MAIN verifier),
  so the scope over-waits (busy slot windows), never under-waits. The gate
  `park_own_ok()` (generate.cpp:15929: `stages.empty() && !pipe && !conc_serving &&
  !in_prompt_read`) excludes every shape with writers outside the set (stage threads
  generate.cpp:10097, doorbell thread 15507/18346, pipelined read 18325/21153/21310/21397).
- Cell-reuse ordering (MANAGER: event before reuse). Every replaced wait lands before the
  reuse: park pre-syncs (:335/:395 equivalents, conversation_state.cpp sync()) drain the
  scoped streams before D2H copies which are per-copy host-ordered (conversation_copy.hpp:
  126-139); restore H2D copies land per copy, and the restore tail `copy_done`
  (conversation_state.cpp:257) returns only after this thread's copy stream is done; the
  residency refresh is launched on that same copy stream right after the pool copies
  (stream-ordered, conversation_snapshot.cpp:421/446) and is therefore covered by the same
  wait before any later compute is enqueued. `order_after_queued`'s event-before-legacy
  (conversation_copy.hpp:276-301) and the direct-copy path are untouched
  (conversation_snapshot.cpp:405). Where refresh stays unwaited inside a function - the
  borrow tail (conversation_state.cpp:615-624) - the base was identically unwaited (legacy
  launch, no wait) and the next whole-device :240 sync (outside every scope there) orders
  it in both; the new placement is strictly better covered (every narrowed sync waits the
  private stream, the base's legacy launch was waited only by device syncs).
- Ungated paths keep their drains: stage/slot restores still device_sync at
  generate.cpp:19426, 20335, 14486/14505/14512, 14533/14548/14557/14563.
- Locks: the narrowed path takes neither `capture_mutex` (capture_lock.hpp:48-52 - correct,
  only device syncs invalidate captures, and stream syncs cannot race a capture under the
  gate: no other thread touches the scoped streams in the gated shape) nor nests any lock
  (`private_stream`'s pool mutex, conversation_copy.hpp:83-104, is released before any
  wait). `ConvSyncScope` is never nested (persist_slot call sites 16864-23831 are outside
  every armed scope).
- Threads/stdout: no new threads; no new stdout fields; the `split: ... sync N ms` and
  `wait N ms` formats are untouched; P125j's `paused_*` keys ride the existing DONE line
  (generate.cpp:23402-23416) and parse in both servers (engines/qfn/serve/server.py:1665,
  serve/qfn_engine.py:1221).
- Generated file: `python tools/saguaro/extract_qfn_engine.py` reproduces
  serve/qfn_engine.py byte-identical at HEAD (cmp clean, tree unchanged); only
  engines/qfn/serve/server.py carries the source hunk.
- Tests: park_policy_test run alone, one at a time - standalone
  (`g++ -std=c++20 -I engines/qfn/include`, scratch/) and as the build-p125k-merge target:
  **65 checks passed** both ways, matching the report. The 9 new checks
  (park_policy_test.cpp:178-215) really exercise the new decision: device filter, empty
  scope = whole-device sync, cross-device stream never waited, unrelated stream not waited.
  `ninja -n -C scratch/build-p125k-merge strata` = no work at tip (build claim holds).
- No secrets, no repo-outside paths in the diff; no Co-Authored-By/Assisted-by/
  Claude-Session trailers in b7ed679..HEAD.
- HIP (MANAGER): every CUDA call the branch adds is mapped by the HIP shim
  (strata_hip.h:84 cudaGetDevice, :134 cudaStreamSynchronize, :136 cudaStreamWaitEvent,
  :144 cudaEventRecord); no new raw `hipEvent*` call site; the event path
  (`order_after_queued`) is unchanged, and the refresh's no-copy-stream fallback
  (conversation_snapshot.cpp:354-357, `cudaStreamSynchronize(nullptr)`) keeps the same
  legacy-stream convention already used tree-wide (e.g. expert_cache.cpp:737). The copy
  stream move also avoids the HIP legacy-vs-capturing-stream refusal noted at
  conversation_copy.hpp:76-80.
- Merge (dc436f2f0/70f67efde): both conflict hunks keep P125k's scope AND P125j's
  drop_read_pause (generate.cpp:20245-20249, 20304-20311); auto-merged sites re-checked -
  file restore 19406-19427 (p125k scope wraps it, stage syncs stay whole-device), park
  20124-20163 + park_body pause rows 11090/11196-11216, cut 23225-23244, keep chain
  23153-23255, DONE suffix 23402-23416, grace + pause wording in serve/server.py -
  nothing from either side lost.

## Findings by severity

1. MEDIUM (report wording, no code change): REPORT-p125k-park-sync.md:109-112 says the
   gated-out shapes "keep the whole-device sync, byte-identical to today", but
   conversation_state.cpp:257 replaces the restore post-copy device sync with `copy_done`
   for EVERY shape, and conversation_snapshot.cpp:421/446 move the residency refresh to
   the copy stream for every shape. Verified correctness-equivalent for the split /
   multi-GPU / slot shapes (each restore ends on its device's copy stream, which its own
   copy_done or the retained explicit device_sync above waits, and the refresh rides that
   stream), and REPORT §5 open question 3 records the contract - but "byte-identical" is
   not the true description of those shapes: they now wait strictly less at :257. The
   copy-stream contract ("a future restore-path copy must not launch elsewhere",
   REPORT:157-160) now binds all shapes and is guarded only by convention.
2. NIT: REPORT:87 claims the FAST_LANE `light` path is "byte-identical".
   `ConvStreams::run` -> `narrow_sync` (conversation_copy.hpp:312-317) adds one
   private-copy-stream sync the old `LightSync::run` did not take; at `sync()` sites it is
   redundant (conversation_state.cpp:36-38 already synced the stream) and under the light
   scope it is this thread's own stream only - save bytes and log format unchanged, but
   one extra driver round-trip; "byte-identical" overstates it.
3. NIT (pre-existing, unchanged, worth a comment someday): conversation_state.cpp:620-622
   - the borrow's restore returns with the residency refresh queued and unwaited (no host
   wait after it inside the function); safety comes from the next whole-device sync (the
   checkpoint mount's :240). True at base too, but this branch is precisely about these
   waits, so the asymmetry (every other restore tail waits its copy stream, this one
   doesn't) invites a future bug.
4. CONTEXT (manager-owned): this tree's merged P125j tip is d8e25fd95, i.e. its round-4
   FIX-FIRST record - the round-4 BLOCKER (the batch lane never cuts a read; the pipelined
   reader has no pause capture) and the MAJOR (grace default 0 at serve/server.py:420 in
   exactly the inert batch config) are NOT fixed in these refs; FIX-REPORT says so. The
   merge introduced nothing new there, but shipping this diff ships those defaults, so the
   P125j fix round should land before this whole diff does.

## Not run

Full engine rebuild (FIX-REPORT's scratch/build-p125k-merge claims verified indirectly:
`ninja -n` no-work at tip, test target present and green); GPU/card runs (plan: manager
after; nothing on a card here). Only the plan's named module, park_policy_test, was run.
