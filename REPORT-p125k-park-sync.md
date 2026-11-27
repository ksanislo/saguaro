# REPORT - P125k: a QFN park must not wait on unrelated device work (Saguaro 1.2.5)

Branch `saguaro/q1-p125k-park-sync`, engine tree `engines/qfn`. Follows P124a2
(ref/REPORT-p124a2-park-overlap.md, ref/REVIEW-p124a2-park-overlap.md), which made the park adopt the turn save's
host copy; what is left is the WAIT itself: `sync 2471 ms` on a 1,206-token park, and a 281-token restore that cost
1,066 ms with a 1 ms split (ref/REPORT-p124a2 §1 lines 66-68). Nothing was run on a GPU here; §1 re-anchors every
sync/wait on these paths at this tip.

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

## 2. The fix

(step 2 - in progress)

## 3. Tests

(step 3 - pending)

## 4. Open questions

(to be written with the final report)
