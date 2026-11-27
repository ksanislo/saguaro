# REPORT - P125k: a QFN park must not wait on unrelated device work (Saguaro 1.2.5)

Branch `saguaro/q1-p125k-park-sync`, engine tree `engines/qfn`. Follows P124a2
(ref/REPORT-p124a2-park-overlap.md, ref/REVIEW-p124a2-park-overlap.md), which made the park adopt the turn save's
host copy; what is left is the WAIT itself: `sync 2471 ms` on a 1,206-token park, and a 281-token restore that cost
1,066 ms with a 1 ms split (ref/REPORT-p124a2 §1 lines 66-68). Nothing was run on a GPU here. §1 re-anchors
every sync/wait on these paths as they stood at the merge tip (what each must wait for, what it waits for
today); §2 is what this branch changed.

## 1. Every sync / wait on the park, restore, snapshot and checkpoint paths

### 1.1 The one primitive and its scope

All the waits below funnel through `sync()` (conversation_state.cpp:27-43). What it waits for TODAY:
`strata::core::device_sync()` (capture_lock.hpp:48-52) = the EXCLUSIVE `capture_mutex` + `cudaDeviceSynchronize()` -
every stream on the device: other conversations' queued windows on their verifier and drafter streams, the expert
tier's background fills (several ride the LEGACY stream, e.g. expert_cache.cpp:737 `cudaMemcpyAsync(...,
(cudaStream_t) 0)`), PLE io, peer/remote-expert refills, the adaptive tier. What it should wait for is only the
work of THIS conversation's streams; the wait is timed into the split's `sync_ms` (:31-32), which is where the log's
`sync 2471 ms` comes from.

The one narrow variant already shipped is SAGUARO_FAST_LANE's: when a `device_sync_override` (capture_lock.hpp:39-46)
is set, `sync()` waits for this thread's private copy stream (:36 - only its own queued copies) plus the override's
stream list, which for an idle-slot save is exactly "the streams that can write that slot" (`LightSync` +
`SyncScope`, generate.cpp:15851-15861/:15896-15908, armed in `persist_slot` :15874-15908 with the slot's window
streams + its drafter). The MAIN conversation's park/restore/turn-save paths never arm anything, so every one of
their syncs is device-wide.

### 1.2 The sync() call sites (conversation_state.cpp)

| site | path | must wait for (only this conversation's bytes) | waits today |
|---|---|---|---|
| :335 `conversation_snapshot_save` pre-copy | park capture (switch, `park_body` generate.cpp:11153; slot parks via :15912) | the queued writes into what it reads out: this session's running state + K/V pools + draft K/V pool - i.e. this conversation's own session calls (generate.cpp `main_stream`, :4494-4499), its window verifier streams (`ver`/`ver_b` - with slots on the SHARED drafter a batched draft prefill writes the main draft pool, see `park_draft` generate.cpp:10746-10750), and its drafter (`mtp.stream()`: "the draft layer's final cell may not have been computed when the output cap was reached", :378-381) | whole device |
| :395 `conversation_snapshot_sources` pre-copy | turn save / spill (`turn_capture` generate.cpp:10848/:10859, `spill_from` :10765/:10776) | same producers | whole device |
| :225 `conversation_checkpoint_restore` pre-copy | checkpoint mount (`live_from_checks` generate.cpp:11216, rewind restores, cut restore :22967, copy_from_slot :14488, batch admission :14443) | the queued READS/WRITES of the cells it overwrites - again only this conversation's streams (the slot copy-in's source slot is drained by its own gate; the destination is this thread's session) | whole device |
| :242 `conversation_checkpoint_restore` post-copy | every checkpoint restore (the rows above) | NOTHING but its own H2D copies - which `copy()` (:227-237 -> `copy_nonblocking` conversation_copy.hpp:124-137) already lands per copy - plus the residency refresh kernels `refresh_residency` (conversation_snapshot.cpp:332-345) launches on the LEGACY stream (:333/:338-339 `kv_stream_reset(m, nullptr)`, `kv_ring_restore(..., nullptr)`) | whole device |
| :552 `conversation_snapshot_restore` pre-copy | the switch's restore of the incoming conversation (generate.cpp:20178/:20188), the context restore (:19332-19339) | queued accesses to the destination session's cells: this conversation's own queued work (the outgoing side is parked - or adopted, which queues nothing - before this runs) | whole device |
| :603 `conversation_snapshot_restore_prefix` pre-copy | the borrow restore (generate.cpp:20131/:20137; the donor keeps its place, the target is this session) | same: this session's own queued work (the donor's bytes are host-side) | whole device |

### 1.3 The copy-path waits that are already narrow (keep as they are)

* `conversation_checkpoint_save`'s one wait: `copy_stream_done()` (conversation_state.cpp:204 ->
  conversation_copy.hpp:142-149) - this thread's private non-blocking stream, i.e. only the copies this
  conversation queued. No device-wide sync in `conversation_snapshot_save`'s or `sources`' copy bodies.
* every K/V / state `transfer()` (conversation_snapshot.cpp:195-204 -> `copy_nonblocking`
  conversation_copy.hpp:124-137): `cudaStreamSynchronize` on the same PRIVATE stream - a stream-wide wait, but
  the stream only ever carries this conversation's copies.
* `host_copy_threaded`'s ordering sync (conversation_snapshot.cpp:151-157): the private stream, own bytes.
* `conversation_kv_copy`'s queued mode (conversation_snapshot.cpp:369-375): no host wait at all - `order_after_queued`
  (conversation_copy.hpp:274-299) records an event on the copy stream and makes the legacy stream wait on it.
  This is the pattern PLAN-p125k points at: an event after this conversation's copies, not a device wait.

### 1.4 The device_sync() calls the serve loop adds around these paths (generate.cpp)

| site | what it guards | waits today |
|---|---|---|
| :11158 park_body, per later stage | a layer split's stage capture | that whole (stage) device |
| :20134 borrow, per stage | after `conversation_snapshot_restore_prefix` | whole device |
| :20195 incoming restore, per stage | after a stage's `conversation_snapshot_restore` | whole device |
| :14486/:14501/:14505/:14515 batch slot restore + `copy_from_slot` | the slot checkpoint restores and the slot->main / main->slot copies | whole device |
| :15879-15882 persist_slot (not `light`) | an idle-slot save's pre-capture drain | whole device |
| :19342 context restore | after `conversation_snapshot_restore` | whole device |
| :22966/:22989 the cut path | before/after the cut's `checkpoint_restore` | whole device |
| :20296-20297 `session_zero` + `cudaStreamSynchronize(main_stream)` | (already narrow: this thread's own session stream) | - |

The adopted-park path (P124a2, `conversation_snapshot_from_capture` conversation_state.cpp:426-488) does no device
traffic at all, so what the switch pays is the pre-syncs (:335/:395/:552/:603/:225 - device-wide), the restore
post-sync (:242 - device-wide where a 1 ms copy is all there is to wait for), and the `device_sync()` rows above.
Every one of them waits for the whole device today; what they must wait for is this conversation's own queued
compute (its session/window/drafter streams) and, after a restore, only its own copy stream + the residency
kernels queued behind the copies.

## 2. The fix: a park/restore waits only for its own bytes

The wait narrows on three fronts; the copy contract does not move: **every park copy lands before the park
returns, and every restore's bytes land before it returns** - the guarantees the plan names ("no reuse of the
source cells before the copy; no use of restored cells before their bytes") stay host-ordered, and where the
consumer is a stream the event form already existed (`order_after_queued`, conversation_copy.hpp - kept).

* **The pre-copy waits (:335/:395/:225/:552/:603) narrow to this conversation's own streams.** `sync()` already
  honored the `device_sync_override` hook (its FAST_LANE branch); what was missing was anyone setting it for the
  main conversation. `conversation_detail::narrow_sync` (conversation_copy.hpp) is the hook function: this
  thread's private copy stream plus the scope's (device, stream) pairs on the device the wait runs on
  (`park::wait_streams`, the pure decision, now unit-tested) - no `cudaDeviceSynchronize`, no capture-lock
  acquisition (only device syncs invalidate in-flight captures, capture_lock.hpp:2-3). `LightSync` (the slot
  save's scope) was hoisted into `ConvStreams`/`ConvSyncScope` (generate.cpp) - same semantics, now also
  covering the copy stream; `persist_slot`'s `light` path waits the same narrow set plus one addition -
  this thread's private copy stream (`wait_streams` always includes it) - so it is not byte-identical,
  it gained a sync of its own thread's queued copies and still never device-syncs.
* **The post-copy wait (`conversation_checkpoint_restore` :242) is the copy stream alone.** Every copy above it
  rides the private stream and lands per copy, so `copy_done()` (conversation_state.cpp, timed into `sync_ms`)
  replaces the device sync - this is the 281-token restore that cost 1,066 ms with a 1 ms split. For the same
  reason `refresh_residency` (conversation_snapshot.cpp) launches its map-reset/ring-refill kernels on the copy
  stream on the restore paths (stream-ordered right after the pool copies, waited by the same wait) instead of
  the legacy stream; the direct-copy path (`conversation_kv_copy`, queued mode) keeps its legacy launch +
  ordering event + caller's device sync exactly as before. A restore thread that cannot get a copy stream (the
  #2409 full-card case) falls back to the legacy launch with an immediate wait there - never less drained than before.
* **The serve loop arms the scope** (`park_own_ok() && add_main_streams(...)`) around: the switch's park
  (including `live_from_checks` and the spill), the borrow's prefix restore, the incoming restore, the session
  file's context restore, the cut read's checkpoint restore, and the main turn save. The stream set is this
  conversation's: `main_stream` (every session call, :4494), and with `--mtp` `ver`/`ver_b` (on a single card
  the main verify windows, and a slot's batched draft prefill on the SHARED drafter, writes the draft K/V pool
  the park saves - `park_draft` :10746; the verifiers' side streams fork/join through their own stream,
  verify.cpp:1200-1204/:1478/:1542, so waiting for it waits for them) and `mtp.stream()` (the final draft cell
  may still be queued, conversation_state.cpp:378-381). What stays OUTSIDE the scope is the whole point: other
  conversations' queued windows on streams the parked conversation never touches, the expert tier's background
  fills (legacy stream, e.g. expert_cache.cpp:737), PLE io, the adaptive tier, the session-dir writer (no device).
* **The gate**: `park_own_ok() = stages.empty() && !pipe && !conc_serving && !in_prompt_read` - the whole model
  on one card with no read running beside. A layer split moves the conversation's work onto per-stage streams
  (and `pipe`'s pipelined decode onto `pl_stream`/snap side streams) that this list cannot bound, and concurrent
   doorbell serving has a second thread queueing on this thread's streams: those shapes arm nothing and keep the
   whole-device sync for the scoped waits - not byte-identical overall: the ungated restore tail (`copy_done`
   :257) and the copy-stream residency refresh ride every shape (the §5 contract), so a gated-out restore also
   ends on its own copies instead of the old device sync - it waits less, never less-drained, the device sync
   ahead of each copy still drains everything. `serve/qfn_engine.py` and `engines/qfn/serve/` are untouched -
  no new stdout field, the park/restore lines' `split: ... sync N ms ...` plus `wait N ms` keep their format
  (`sync` is now the narrowed wait; `wait` is still the time outside the clocks).

## 3. Tests run

* `park_policy_test.cpp`: nine fake-stream checks on the narrowed wait-set - the device's share of a scope is
  waited (own queued work: source cells not reused before their copy, restored cells not read before their
  bytes), a stream of the OTHER device is not waited from this one, an unrelated conversation's stream is
  never waited under a scope, and no scope means the whole-device sync, as before. **65 checks passed**
  (56 at the merge tip + 9), standalone (`g++ -std=c++20 -I engines/qfn/include ...` into scratch/) and as
  the `park_policy_test` target of the real build - each run one at a time. This is the only test module the
  plan names; nothing ran on a GPU here.
* Engine build proof: `QFN_GGML_DIR=/data/llm/maya/third_party/llama.cpp tools/qfn/build.sh scratch/build-p125k`
  (CUDA, sm_75, /usr/local/cuda-13.1/bin/nvcc, fresh build dir, llama.cpp at the pinned 3cf03257f):
  **QFN BUILD OK 179/179**, `saguaro-qfn` linked; `ninja -n -C scratch/build-p125k strata` reports *no work to
  do* at tip. The one `generate.cpp` warning in the log (:15814 unused `w`) predates this branch.
* Card test (manager, after): parks/restores during 2-3 busy conversations - the park line's `sync` part must
  fall from the wait-storms (2471 ms) to this conversation's own drain, so the whole park approaches its
  `copy` figure; reads/decode flat. `ref/mortar-park-new-20261126.log`'s shape: single card, conversations
  interleaved through the switch - the gated shape. If the log still shows big `sync` there, the scope is not
  being armed for that shape and the `park_own_ok` terms are the first thing to read.

## 4. Files changed

* `engines/qfn/include/strata/core/park_policy.hpp` - `wait_streams` (the pure wait-set decision).
* `engines/qfn/src/core/conversation_copy.hpp` - `narrow_sync` (the override hook: copy stream + the device's
  scoped streams).
* `engines/qfn/src/core/conversation_state.cpp` - `SyncRec` hoisted, `copy_done`, and
  `conversation_checkpoint_restore` ends on the copy stream instead of the device sync.
* `engines/qfn/src/core/conversation_snapshot.cpp` - `refresh_residency` takes a stream; the restore paths pass
  the copy stream (legacy + immediate wait for the no-copy-stream thread); `conversation_kv_copy` unchanged.
* `engines/qfn/src/program/generate.cpp` - the hoisted `ConvStreams`/`ConvSyncScope` (also used by the
  FAST_LANE slot save), `park_own_ok`/`add_main_streams`, and the six arming sites above.
* `engines/qfn/src/core/park_policy_test.cpp` - 56 -> 65 checks.
* `REPORT-p125k-park-sync.md` - this file.

## 5. Open questions

* **`ver`/`ver_b` in the single-card slot shape**: they are in the wait set because a slot's batched draft
  prefill on the shared drafter writes the draft K/V pool the main park reads - correct, but it means a busy
  slot group's queued window is (partly) waited at the switch. The card run shows whether that matters; if it
  does, the narrowing is a separate drafter-pool idle gate (persist_slot's `mtp.idle` precedent) that would let
  `ver`/`ver_b` leave the set.
  * **The multi-GPU / pipelined shapes stay whole-device** (`park_own_ok`). Bounding them means attributing the
    stage devices' window/pl_stream/snap streams per conversation - the same mechanism, more streams; the gate
    keeps their armed waits whole-device until that is done on the card (the ungated restore-tail change above
    still rides them: those shapes wait less after their copies, never under-wait).
* **`copy_done` replaces the device sync at :242 for every caller**, scope or not: it rests on every restore
  copy riding the private copy stream (true today: `copy()` is always `copy_nonblocking`, and the residency
  refresh now too). A future restore-path copy that launches elsewhere must not bypass it - the same contract
  `copy_stream_done` carries for `conversation_checkpoint_save`.
* Whether the 2471 ms was the expert tier's legacy-stream fills or another conversation's window is not
  provable from the log alone; the narrowed wait excludes both, which is why the card run is the real test.
* The slot `copy_from_slot`/admission syncs (:14486-:14515) keep their device syncs: the source there is
  ANOTHER session's cells, whose producer streams are that slot's - a different scope, not this plan's
  "its own bytes".
* The adopted park (P124a2) already had no device traffic; what the switch still paid was this step's target -
  the restore side and the pre-syncs. The remaining restore-without-copies slowness the P124a2 report points
  at (a `SessionDir::claim` writer wait, host-side) is untouched and still the next thread if the card says so.
