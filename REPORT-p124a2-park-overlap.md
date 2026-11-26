# REPORT - P124a2: a park's K/V copy must not crawl behind the turn's background session save (Saguaro 1.2.4)

Branch `saguaro/q1-p124a2-park-overlap`, engine tree `engines/qfn`. Card evidence:
`ref/mortar-park-new-20261126.log` (RX 9070 XT, QFN UD-Q4, `--session-dir` with the appending turn saves).
Nothing was run on a GPU here.

## 1. The confirmation: what the turn save and the park each copy, on which thread, into which buffers

The turn save of the main conversation runs on the **serve thread** at the end of every request, after the reply's
last commit: generate.cpp:22972-22984 (`turn_capture`, generate.cpp:10698; a batch slot's own save is the same
capture, generate.cpp:15677, and `turn_tick` generate.cpp:18273 covers idle slots).

* `conversation_snapshot_sources` (generate.cpp:10714 -> conversation_state.cpp:377):
  - `sync(error)` - conversation_state.cpp:380;
  - `conversation_checkpoint_save` - conversation_state.cpp:386, body conversation_state.cpp:177: the running state
    (~237 MB for this 62-QSA-layer model: `gdn`, `ple` and three arrays per layer) device->host on the thread's
    **private non-blocking stream** (conversation_copy.hpp:81/:124), waited for **once** at conversation_state.cpp:204
    (`copy_stream_done`, conversation_copy.hpp:142). Destination: the capture's `ConversationCheckpoint` host vectors
    (`meta.live.gdn/ple/tails/dead/block_pos`).
  - `conversation_kv_source` per QSA layer + the draft - conversation_state.cpp:390-392, body
    conversation_snapshot.cpp:428: no copy, it records the layer's **authoritative pool pointers** - which for
    `kv_mode != 0` are the **host** pool (`host = st.kv_mode != 0 && !resident`, `pools()` conversation_snapshot.cpp:84-95).
* `session_log_capture` (generate.cpp:10769 -> session_log.cpp:512) then reads each part's bytes `[from, size)` from
  those sources into `SessionLogKvTail.bytes[p]` host buffers - session_log.cpp:531-537 - through the source's
  `read` (conversation_snapshot.cpp:439-442), which is the same threaded-memcpy/`copy_nonblocking` choice as the
  park's (`transfer`, conversation_snapshot.cpp:195-204). A stored park `reset()`s the turn track
  (generate.cpp:10978), so every capture after a parked switch is a **whole** one (`from == 0`, `delta()` false,
  session_log.hpp:60): the log's "747 MB appended" *is* the whole conversation, and the next park copies that same
  whole conversation again.
* `sdir->write_turn` (generate.cpp:10784 -> session_dir.cpp:331) only queues; the folder's **writer thread**
  (session_dir.cpp:126, `writer_loop` :244, `turn_write` :340) runs `session_log_save_turn` (:388) -> `save_core`:
  it **hashes the just-captured host bytes and appends them to the log file** (`parallel()` hashing
  session_log.cpp:337-343, `pwrite_all` :381). The writer thread touches **no device and no CUDA stream** - the
  plan's phrase "the session-dir writer copying the K/V off the device on its own streams" is not what the code
  does: the capture copies are the serve thread's; what overlaps the park is the writer hashing 747 MB and pushing
  747 MB into the page cache (plus the kernel's writeback of it, which outlives the "appended" line), on a box
  where the engine runs nice +5.

The park: the next request parks the outgoing conversation before restoring the incoming one - `park_current`
(generate.cpp:11028/11032, called at :19729/:19749/:19755), body `park_body` (generate.cpp:10796):

* `conversation_snapshot_save` (generate.cpp:10953 -> conversation_state.cpp:329): `sync` :334, the checkpoint
  tree's host copies :343, `conversation_checkpoint_save` :354 - the **same running state again, same private
  stream, new host vectors** - and `conversation_kv_save` per layer :360-366 (body conversation_snapshot.cpp:263):
  fresh `ConversationBuffer` segments (`resize_for_overwrite` :293) filled by `transfer_jobs` (:302 -> :206) -
  threaded host memcpy when both ends are host (`host_copy_threaded` :141-191, the log's "26 threaded"), else
  `copy_nonblocking` on the thread's private stream. The source is the **same pool pointers** the capture read a
  second before (`pools(st)`), and since nothing generated between the two, the **same bytes**:
  `parked 33427 ... snapshot_bytes=746645424` right after `saved a 33,427-token conversation ... 747 MB appended`.

The overlap, from the log: the slow park of conversation A starts within ~0.3 s of the same conversation's turn
capture finishing - `saved a 33,427-token ... 747 MB appended (0.4 s, in the background)` at 05:10:37.759 with a
park that began at 05:10:37.33 (1341.8 ms ending 38.676; 05:10:48.459 vs a park at 05:10:48.12, 1248.7 ms). Its
`copy` part - 0.48 GB of threaded H2H memcpy, ~41 ms uncontended (the same parks at 05:10:47.482 and 05:10:58.134:
41-43 ms) - runs at ~0.4 GB/s: 1028-1269 ms. What is alive during that window is the writer's hashing + pwrite of
the very bytes being copied a second time, the kernel writing those 747 MB of dirty pages out afterwards, and the
747 MB of capture buffers plus the 747 MB of park buffers plus the page cache moving through a host that already
holds 68 GiB of pinned experts (P124a: `available` read 8.6 GiB). The retained-K/V fast path is not in play there:
the tight budget refuses `retain()` (conversation_cache.hpp:277-280) and `make_room` drops what is left
(conversation_cache.hpp:432), so every such park reports `reused_kv_bytes=0`.

Verdict: the plan's correlation holds, in a sharper form - **the park re-copies byte-for-byte what the turn save
captured seconds earlier, while that copy's background hash+append saturates the same memory**, and the second
copy is not needed at all. A side confirmation of the same contention: `sync 2471 ms` on the 1206-token park at
05:10:06.4 and a 281-token restore that cost 1066.6 ms with a 1 ms split (copy 1 ms, wait outside it) - waits,
not copies, are what the machine was out of.
