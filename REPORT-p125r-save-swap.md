# REPORT - P125r: saving a huge conversation does not push the box into swap

Branch `saguaro/q1-p125r-save-swap` off 1.2.4-pre (`f2e39c87d`).
Card (mortar, 96 GB, 1.2.4-pre test copy): a 546,065-token conversation answered with free RAM
steady at 7.5-10 GB and swap flat; seconds AFTER the answer swap jumped 3 -> 17 GB. The engine
log's next work is the session save of that conversation; its K/V is ~8 GiB at that length.
Sections 1-3 are the step-1 report: what the save of a 546K conversation allocates, file:line.
Sections 4-7: the streaming capture that replaced the second copy, the files, the tests, and
what stays open. Commits: cfbb6b742 (step 1), 40c8862b5 (step 2), 10d27c6b3 (step 3), this
report's commit (step 5). QFN build `scratch/build-p125r`: `QFN BUILD OK 10d27c6b3`. No GPU runs.

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

## 4. The change (step 2, commit 40c8862b5)

`session_log_capture` stops being allowed to build the full second copy: a capture gets a RAM
budget - the writer's buffer cap, `SAGUARO_TURN_SAVE_SPOOL_MIB`, default 256 MiB, 0 restores the
old behaviour. Parts captured while the capture's held bytes stay under the cap behave exactly
as before (every delta save, every normal chat turn, every existing test): they land in
`t.bytes[p]`. A part that would pass the budget streams instead: its bytes are copied from the
source through a buffer no bigger than the budget (one reused vector, `held + piece <= cap` plus
a 1 MiB floor) into a `SessionLogSpool` file - opened in the conversation's folder
(`sdir->dir()`) and unlinked at once, so its blocks are disk the folder already accounts, the fd
is the only way in, and a crash leaves no name and no bytes - and the tail part names its
`[from, size)` at `spool_at` in place of a `ConversationBuffer`.

The writer needed no new machinery: `Ref` grew `spool`/`spool_at` and its `visit()` reads a
spooled part back in pieces of at most 4 MiB, so the same hash pass and the same `kBatch` 16 MiB
write loop work over held and spooled bytes alike and the save's footprint stays at the cap. The
turn-save line says so (session_dir.cpp, `turn_write`):

    turn save (main): 546065 tokens, ... ; 7936 MiB of it streamed to its folder through a 256 MiB writer buffer

`SAGUARO_TURN_SAVE_SPOOL_MIB` sets the cap (default 256; 0 keeps the old whole-capture-in-RAM
behaviour and never opens a spool). The spool is created in the session folder and unlinked at
once, so its fd is the only way in and a crash leaves no name and no bytes; capture failure on a
spool write is a turn-save skip (`h.reset()`, next save whole), the session untouched.

## 5. Files

- `engines/qfn/include/strata/core/session_log.hpp` - `SessionLogSpool` (fd, cap_bytes,
  open/write_at/pread), the per-part `spool`/`spool_at` fields on `SessionLogKvTail`,
  `SessionLogTurn::spool`/`spool_bytes` (and `bytes()` counts spooled bytes), the two defaulted
  capture arguments.
- `engines/qfn/src/core/session_log.cpp` - the spool methods (Linux section, like the format)
  and the streaming capture: parts past the budget are read through one reused buffer into the
  spool; `Ref::visit` grew the spool branch (pread in <= 4 MiB pieces) so the hash pass and the
  write loop are unchanged.
- `engines/qfn/src/program/generate.cpp` - `turn_spool_mib` (beside `turn_save_on`) and
  `sdir->dir(), turn_spool_mib << 20` passed to `session_log_capture`.
- `engines/qfn/include/strata/core/session_dir.hpp` - `SessionDir::dir()`.
- `engines/qfn/src/core/session_log_spool_test.cpp` + `engines/qfn/CMakeLists.txt` - the new
  test module (below).
- Commits: step 1 report `cfbb6b742`, step 2 the streaming capture `40c8862b5`, step 3 the
  test `10d27c6b3`.

## 6. Tests run

- `session_log_spool_test` (new): 22 checks - a fake host copy (two K/V layers read through
  `SessionKvSource`, as a streamed session's pools answer) with a 4 MiB cap holds k0 in RAM and
  spools the rest at the right offsets; cap 0 and a spool-less folder behave exactly as before
  (no spool object, byte-identical saves); an all-spooled whole save lands through
  `SessionDir::write_turn`, its log line says "spooled through the folder", and a request that
  rewrites the host copy MID-save (writer stalled) still reads back as captured, not rewritten;
  a delta after a spooled whole reuses the file's chunks and reads back whole; a folder restart
  fetches the spooled-then-delta turn; no `.session-spool-*` files left behind. Ran plain
  (22/22) and under `-fsanitize=address,undefined` (22/22), and off the QFN build (below).
- One mid-development run of the first (sanitized) binary failed once on the delta check. It
  never reproduced since - 153 consecutive clean runs across plain, `-fsanitize=address,undefined`
  and 20-way parallel loads over every build made after it - and the cause is not pinned
  (flagged in §7).
- QFN-only build (`tools/qfn/build.sh scratch/build-p125r`,
  `QFN_GGML_DIR=<strata-main>/third_party/llama.cpp`): `QFN BUILD OK 10d27c6b3` - engine and
  all conversation-test targets. From it, run one at a time: `turn_save_test` 97/97,
  `session_log_spool_test` 22/22. `serve/qfn_engine.py` untouched (no `engines/qfn/serve`
  change, the extractor had nothing to run). No GPU runs.
- Capture-peak probe (`scratch/p125r_peak.cpp`, same objects, a 2 GiB fake K/V): held capture
  adds 2048 MiB of RSS (bytes, not addresses - measured at 16 MiB segments); capped at 256 MiB
  the capture adds 256 MiB and spools the other 1792; the save itself added 1 MiB either way.
  Scaled to the card: the ~8 GiB whole capture of the 546K conversation stops being an ~8 GiB
  RSS step beside its 7.5-10 GB of free RAM.

## 7. Open questions

- The spool writes the streamed bytes to the folder's disk and the writer reads them back, so a
  spooled save costs one extra write+read of the tail's bytes. Chosen deliberately: RAM-bounded
  without a synchronous serve-thread write; a tmp-folder box (tmpfs) moves the bytes to swap-
  backed pages instead of the folder's disk. Worth watching on mortar.
- PLAN-p124a2's park-adopt path (conversation_state.cpp:478-479, `contiguous()` null) refuses a
  spooled whole capture and captures normally, as it already refuses every non-aliasable part -
  so past the cap the park pays its own capture again. An adopt through the spool (pread into
  the park's buffers) is the obvious follow-up if parks beside big saves start showing up.
- The cap is one env for every turn. Turns under it never touch the spool, so ordinary chat
  costs nothing; but a box that wants a spool for 546K turns and RAM for 600K ones would want
  the cap per-model - not built here.
- The one unreproduced delta-check flake (§6): if it ever shows again, dump the file's bytes at
  the failing offset - the check itself cannot say which part mismatched.
- Unverified on hardware: mortar's exact swap shape (this is CPU-only work; the probe is the
  proof, the card test is not run here).

Report committed on top of the three steps above.
