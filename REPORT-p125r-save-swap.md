# REPORT - P125r: saving a huge conversation does not push the box into swap

Branch `saguaro/q1-p125r-save-swap` off 1.2.4-pre (`f2e39c87d`).
Card (mortar, 96 GB, 1.2.4-pre test copy): a 546,065-token conversation answered with free RAM
steady at 7.5-10 GB and swap flat; seconds AFTER the answer swap jumped 3 -> 17 GB. The engine
log's next work is the session save of that conversation; its K/V is ~8 GiB at that length.
This part: what the save of a 546K conversation allocates, file:line, before anything changes.

## 1. Which save runs seconds after an answer

`SAGUARO_TURN_SAVE` is on unless set to 0 (generate.cpp:11094) and `turn_capture`
(generate.cpp:11096) runs at the end of every turn of a conversation that stays where it sits:
it builds the turn's `SavedConversation` meta plus one `SessionKvSource` per K/V part-list
(generate.cpp:11112, :11123) and captures the turn through
`strata::core::session_log_capture` (generate.cpp:11167); the folder's writer thread then appends
it to the conversation's file (`SessionDir::turn_write`, session_dir.cpp:374-452, called from the
writer loop at :282, no write delay). This - not the park, not the SAVE command - is the save of
a conversation that is still sitting in its place right after its answer.

Where its K/V lives: for a streamed session (`kv_mode != 0`, the host-resident K/V) the pools'
HOST ends are the authoritative copy (conversation_snapshot.cpp:84-95 - `pools()` answers
`st.host.*` whenever `!resident`; "a streamed one reads its host copy", :40) and the source's
`read` (conversation_snapshot.cpp:453-456, `transfer` :203-212) is a threaded MEMCPY through it
when the host end is CPU memory (`host_end_is_cpu`, :142). So for the machine in the card the
conversation's ~8 GiB of K/V is ALREADY in host RAM before the save starts. For a card-resident
session (`kv_mode == 0`) the same read is a device-to-host copy; the byte count is the same.

## 2. The allocations on that path, for 546,065 tokens (K/V ~8.0 GiB)

- **The full second copy: session_log.cpp:531.** `session_log_capture`
  (session_log.cpp:512-543) walks every part of every K/V layer of every stage and, for each,
  `t.bytes[p].resize_for_overwrite(n)` at :531 with `n = t.size[p] - t.from[p]` (:530), then
  fills it through `s.read` (:532-534). `from` is the byte count `keep_of` proved unchanged
  (generate.cpp:11150-11161): the chunks under it stay in the file and are not captured. A
  WHOLE capture (`from == 0` everywhere) therefore allocates the conversation's whole K/V
  again - ~8.0 GiB at 546,065 tokens - as fresh `ConversationBuffer` segments
  (conversation_buffer.hpp:114: 16 MiB segments; :200-221 `resize_for_overwrite`: untouched
  pages, first touched by the copies at :532-534). Whole is the FIRST turn at a place
  (`h.track` empty, or `h.dirty == INT64_MAX` - "nothing reported = unknown" - forces
  `h.reset()` and `same = 0`, generate.cpp:11141-11147): the first turn of a conversation
  freshly read from disk, a rewound tail, a broken chain. The card's 546K conversation's
  answer is exactly that shape.
- **The second copy then stands there: generate.cpp:11182-11192.** A whole capture is kept
  beyond the write: `h.adopt_cap` holds the `SessionLogTurn` STRONG (PLAN-p124a2, the field at
  :11083, set at :11186-11192) so the park that follows the turn can adopt it
  (`conversation_snapshot_from_capture`, conversation_state.cpp:426-488) instead of copying the
  same K/V again. The ~8 GiB is released only when a park consumes it, a request begins
  (`begin` -> `adopt_clear`, :11089) or a rewrite site clears it.
- **The meta the capture carries (session_log.cpp:516 `out.meta = std::move(meta)`):** the live
  running state plus the checkpoints the save keeps - `gdn/ple/tails/dead/block_pos` host
  vectors, ~0.2 GiB for this geometry ("a turn rewrites the running state, ~240 MB",
  session_log.hpp:27-28) - plus token lists: `live.ids` 546,065 x 4 B ~ 2.1 MiB in the meta,
  the deepest checkpoint's ids ~2.1 MiB, and `h.ids` again (generate.cpp:11179) ~2.1 MiB.
  Linear in tokens but two orders under the K/V.
- **The folder's writer after it is bounded by design:** the hash tasks
  (session_log.cpp:579-601) and the write loop (:712-736) only VISIT the captured buffers,
  batching at `kBatch = 16 MiB` (:67, allocated :695). The v1/v2 writer (`session_file_write`,
  conversation_file.cpp:1271-1286) streams the sources straight into its 16 MiB aligned block
  (`kBlock` :85, `FileSink` :531-593, `Out::source` :658-677) - that path already never holds a
  host copy of the K/V, and it is the pattern to follow.
- The chunk table and commit record are ~16 B per chunk: at the 256 KiB default chunk
  (session_log.hpp:26) 8 GiB is 32,768 chunks ~ 0.5 MiB of `meta` (session_log.cpp:652).
  Negligible.

## 3. Why the box swapped

~8 GiB of freshly touched pages (the capture at :531, first-written by the copies at :532-534)
on a machine holding steady at only 7.5-10 GB free, plus the dirty page cache of the ~8 GiB the
writer then pushes at the file - the kernel's way out is the 14 GB swap jump the card saw
(3 -> 17 GB), and it happens after the answer because that is when the capture and the writer
run. The file write itself (bounded batches, 16 MiB at a time) is not what allocates.

## 4. The change (step 2)

`session_log_capture` stops being allowed to build the full second copy: a capture gets a RAM
budget - the writer's buffer cap, `SAGUARO_TURN_SAVE_SPOOL_MIB`, default 256 MiB. Parts captured
while the capture's held bytes stay under the cap behave exactly as now (every delta save, every
normal chat turn, every existing test). Past the cap the bytes stream: each further part is
copied from its source through the bounded buffer into a SPOOL file - opened in the conversation
folder and unlinked at once, so what survives a crash is nothing and its blocks account with the
folder's disk - and each `SessionLogKvTail` part names its spool bytes in place of its
`ConversationBuffer`. `save_core` reads spooled parts back through the same `Ref::visit` in
bounded pieces for both the hash pass and the write, so the writer's footprint stays at the cap;
the turn-save log line (session_dir.cpp:434-439) says it. The park's adopt path
(conversation_state.cpp:478-479, `contiguous()` answers null for a spooled part) refuses a
spooled capture and captures normally, as it already refuses every other non-aliasable capture.

Commit (step 1) done.
