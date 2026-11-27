# REPORT - P124a2: a park's K/V copy must not crawl behind the turn's background session save (Saguaro 1.2.4)

Branch `saguaro/q1-p124a2-park-overlap`, engine tree `engines/qfn`. Card evidence:
`ref/mortar-park-new-20261126.log` (RX 9070 XT, QFN UD-Q4, `--session-dir` with the appending turn saves).
Nothing was run on a GPU here. Line numbers below are re-anchored to the branch tip (the manager-round fixes of
2026-11-26 moved the later `generate.cpp`/`conversation_state.cpp` lines; the review-N drift this closes was
`turn_tokens_before` cited at :19760 and the card test "flooring the median").

## 1. The confirmation: what the turn save and the park each copy, on which thread, into which buffers

The turn save of the main conversation runs on the **serve thread** at the end of every request, after the reply's
last commit: generate.cpp:23083-23096 (`turn_capture`, generate.cpp:10715; a batch slot's own save is the same
capture, generate.cpp:15761, and `turn_tick` generate.cpp:18360 covers idle slots).

* `conversation_snapshot_sources` (generate.cpp:10731 -> conversation_state.cpp:392):
  - `sync(error)` - conversation_state.cpp:395;
  - `conversation_checkpoint_save` - conversation_state.cpp:401, body conversation_state.cpp:177: the running state
    (~237 MB for this 62-QSA-layer model: `gdn`, `ple` and three arrays per layer) device->host on the thread's
    **private non-blocking stream** (conversation_copy.hpp:81/:124), waited for **once** at conversation_state.cpp:204
    (`copy_stream_done`, conversation_copy.hpp:142). Destination: the capture's `ConversationCheckpoint` host vectors
    (`meta.live.gdn/ple/tails/dead/block_pos`).
  - `conversation_kv_source` per QSA layer + the draft - conversation_state.cpp:405-407, body
    conversation_snapshot.cpp:428: no copy, it records the layer's **authoritative pool pointers** - which for
    `kv_mode != 0` are the **host** pool (`host = st.kv_mode != 0 && !resident`, `pools()` conversation_snapshot.cpp:84-95).
* `session_log_capture` (generate.cpp:10786 -> session_log.cpp:512) then reads each part's bytes `[from, size)` from
  those sources into `SessionLogKvTail.bytes[p]` host buffers - session_log.cpp:531-537 - through the source's
  `read` (conversation_snapshot.cpp:439-442), which is the same threaded-memcpy/`copy_nonblocking` choice as the
   park's (`transfer`, conversation_snapshot.cpp:195-204). A stored park `reset()`s the turn track
   (generate.cpp:11061), so every capture after a parked switch is a **whole** one (`from == 0`, `delta()` false,
  session_log.hpp:60): the log's "747 MB appended" *is* the whole conversation, and the next park copies that same
  whole conversation again.
* `sdir->write_turn` (generate.cpp:10812 -> session_dir.cpp:331) only queues; the folder's **writer thread**
  (session_dir.cpp:126, `writer_loop` :244, `turn_write` :340) runs `session_log_save_turn` (:388) -> `save_core`:
  it **hashes the just-captured host bytes and appends them to the log file** (`parallel()` hashing
  session_log.cpp:337-343, `pwrite_all` :381). The writer thread touches **no device and no CUDA stream** - the
  plan's phrase "the session-dir writer copying the K/V off the device on its own streams" is not what the code
  does: the capture copies are the serve thread's; what overlaps the park is the writer hashing 747 MB and pushing
  747 MB into the page cache (plus the kernel's writeback of it, which outlives the "appended" line), on a box
  where the engine runs nice +5.

The park: the next request parks the outgoing conversation before restoring the incoming one - `park_current`
(generate.cpp:11112/11116, called at :19824/:19844/:19850), body `park_body` (generate.cpp:10824):

* `conversation_snapshot_save` (generate.cpp:11036 -> conversation_state.cpp:329): `sync` :335, the checkpoint
  tree's host copies :344, `conversation_checkpoint_save` :368 - the **same running state again, same private
  stream, new host vectors** - and `conversation_kv_save` per layer :375-381 (body conversation_snapshot.cpp:263):
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

## 2. The fix: the park ADOPTS the turn save's host copy (option a, the no-second-copy one)

The bytes are the same, the capture is seconds old, and its owner - the folder's writer - is the very thing the
park was competing with. So the park takes over that host copy instead of capturing again:

* `ConversationBuffer` grows a **counted alias** (conversation_buffer.hpp:99-108): a view whose `keep_` is another
  owner in RAM rather than an OS-reclaimable mapping, so `bytes()` charges its full length (a file-mapping view
  still charges 0) and the cache's budget, `snapshot_bytes` and the RAM admission keep telling the truth. Aliases
  are read-only like views (the `external()` guard keeps them out of the retained K/V - since review N it samples
  the WHOLE image, every buffer of every layer and stage, at generate.cpp:20029-20041 - and a
  resize copies first, as before). `contiguous()` (conversation_buffer.hpp:109-114) hands out an owned buffer's
  single-block base - the shape every captured part has below 16 MiB.
* `park::adopt_turn_capture` (park_policy.hpp:179-182) is the pure decision, all six terms required: a whole
  capture of this place still held, no layer split, no retained prefix already covering the copy, nothing ran
  since the capture, the same token ids and steering mode, and no image keys on either side. The five of them
  minus `no_retained` are `park::reuse_turn_state` (park_policy.hpp:168-171) on their own: they say the capture's
  **host running state** is byte-identical to what this park would copy off the device, so even where the K/V
  cannot be adopted (a retained prefix, or a part that refuses to alias) the park copies no ~237 MB running state
  off the device - the manager's card run of 2026-11-26: the parks still crawling were exactly those, their
  `checkpoints` at 574-1560 ms beside the turn save's file writeback. Both decisions unit-tested both
  ways (section 3).
* `turn_capture` (generate.cpp:10801-10811) holds the *whole* capture **strongly** (the manager's card run: the
  parks right after the writer finished were the slow ones - the weak reference died with the write, so the
  capture must outlive it; it is released by the next park consuming it (:10905), not by the file). It is cleared
  when the engine adopts, or a request starts (`TurnHold::begin`, generate.cpp:10708), or a site rewrites the
  session out of band (a slot's copy-in generate.cpp:22955, a context restore :19131, a checkpoint mount in
  `live_from_checks` :11098, a slot restart :16638/:17045/:17222/:17755, a same-session restart of the main
  session :19793-19799, and once after the switch's park block at :19862 for when that park itself does not run -
  review L1) - so an adoption is never offered for bytes the session no longer holds.
* `park_body` (generate.cpp:10896-10906) asks the helpers; on full adoption `conversation_snapshot_from_capture`
  (conversation_state.cpp:426+, declared conversation_snapshot.hpp:177-181) builds the park's image with **no
  device traffic at all**: the checkpoint tree is copied from the view and the running state is copied from the
  capture's own host copy (timed into the split's `checkpoints` like the save's), every K/V part is aliased from
  the capture (each refuses, whole-image, on any extent mismatch, a delta, a split stage list, or a part split
  across segments - the park then captures normally and says so on one line), and `conversation_kv_validate` /
  `conversation_checkpoint_validate` gate the result exactly as a captured image is gated. Where the capture
  refuses, or where only the retained K/V kept the whole adoption off (review M1), the fresh capture still takes
  the capture's host running state - `conversation_snapshot_save`'s `live_from` (generate.cpp:11034-11037,
  conversation_state.cpp:356-369), gated by `reuse_turn_state` - and after a **refusal** the RAM admission figure
  is re-read without the K/V subtraction and the evict loop re-runs BEFORE the capture allocates
  (generate.cpp:11016-11028; the post-capture floor check can only spill afterwards). `reused_kv_bytes` on
  the park line reports the adopted bytes; the RAM estimate subtracts them (`conversation_snapshot_kv_bytes`,
  conversation_state.cpp:413-422, subtracted at generate.cpp:10963-10965) because adopting allocates none.
* `reset()` deliberately does **not** clear the hold: the conversation switch goes through it at generate.cpp:19794
  and the switch's park at :19801-19851 is exactly the park that adopts - the same pattern the existing
  `turn_tokens_before` at :19790 already needed. The slot parks (`park_body` via generate.cpp:15763) share the
  mechanism through their own `turn_slot[b]` hold and their own turn captures (:15761, `turn_tick` :18360).

Byte-identity: the aliases point at the bytes the capture itself had just copied out of the same authoritative
pools for the same `upto` (its `read` is the park's `transfer`; the extents are the same `layout()`, checked again
by `conversation_kv_validate`), and the running state at the capture's copy of the same
`conversation_checkpoint_save` output - the park's copy and the capture's copy are produced by the same code over
the same state, which the helper's terms establish has not run since. The capture's
writer reads its buffers without modifying them (`session_log_save_turn`/`save_core` hash and pwrite only), so
aliasing them is safe while it runs, and the cache entry's `keep_` frees them exactly when both are done with
them. Option (b) had nothing to wait for (the capture's copies are the serve thread's own, done before the park;
section 1) and (c) would have kept paying for a copy that is simply unnecessary.

`serve/qfn_engine.py` and `engines/qfn/serve/server.py`: unchanged - no new stdout field; the park line's
`reused_kv_bytes` was already the engine's stderr format. No new environment switch.

## 3. Tests run

* `park_policy_test.cpp`: seven checks that the six-term adoption rule admits only the conjunction (each term
  false in turn refuses), and since the manager's card run seven more for the five-term running-state reuse
  decision. **48 checks passed**, standalone (`g++ -std=c++20`) and as the `park_policy_test` target
  of the real build. This is the only test module the plan named and the only one run; nothing ran on a GPU.
* Engine build proof: `tools/qfn/build.sh scratch/build-p124a2` - CUDA, sm_75, `/usr/local/cuda-13.1/bin/nvcc`,
  fresh build dir: **QFN BUILD OK** (179/179, `saguaro-qfn` linked). The changed headers and the new
  `conversation_snapshot_from_capture` / `ConversationBuffer::alias` / `park::adopt_turn_capture` APIs also
  compile and link in the test targets `conversation_snapshot_test`, `conversation_cache_test`,
  `conversation_copy_streams_test`, `conversation_split_failure_test`, `conversation_validation_test`,
  `conversation_file_test`, `session_log_test` and `session_dir_test` (ninja, same build dir; not executed -
  they need the device the plan forbids here).
* Card test for the manager: `scratch/card-test-park-overlap.sh` (scratch is gitignored, as its P124a/P124b
  siblings are). It starts a test copy of the card config on :8092 with its OWN session dir under
  `scratch/park-overlap/` and **no api key** in the config it writes (keys stripped from args and top level, then
  grep-checked), kills whatever holds the port at exit, then asks two fresh ~30K prompts alternately - SWITCHES
  (default 6) switches - so every switch parks the conversation whose turn save is hashing and appending right
   then. PASS requires, over the parks of the ~30K conversations: no `copy N ms` above 3x their median (the
   limit is `max(3 x median, 3 x FLOOR_MS)`, so at least 3 x `FLOOR_MS` = 120 ms by default, the full-speed
   0.48 GB copy in the log being 41-43 ms, so a
   1028-1269 ms crawler fails and an all-adopted ~0 ms run still judges any fallback park against a full-speed
  copy), and at least half of those parks reporting `reused_kv_bytes > 0` - proof the adoption, not luck, made
  them fast. `CFG=... [TOKENS= SWITCHES= PORT= FLOOR_MS=] scratch/card-test-park-overlap.sh`.

## 4. Open questions

* **A part over 16 MiB refuses adoption** (it is split across `ConversationBuffer` segments and a single-block
  alias would not reach it): a ~30K-token conversation is ~4 MB a part, so the mortar shape adopts fully, but a
  ~150K conversation's parts fall back to the normal capture (logged once per park; the RAM gate is honestly
  re-admitted before that capture, review M1). The card run of 2026-11-26 answered the question that was open
  here - the parks that still crawled were NOT these K/V fallbacks but the RUNNING-STATE copy of ordinary parks
  beside the save's writeback; that is fixed (reuse_turn_state / live_from). A multi-block alias (a shared
  segment directory in `ConversationBuffer`) stays the follow-up for the ~150K K/V itself - a deeper class
  change than this plan's "least change" allowed.
* **The hold is strong since the manager's card run**: the capture buffers are kept from the turn until the
  session's next park consumes them, or a continuing request or a rewrite site clears them - including across
  an idle gap, which a weak reference could not span and where the parks were still crawling. The cost is one
  capture's RAM held over that window, which the card run judged against the crawl and accepted.
* Whether the same aliasing should also cover the **borrow** path's checkpoint copies (generate.cpp:19948+, the
  donor's checkpoints are deep-copied today) is out of scope: those are tens of MB, not 0.48 GB.
* The 1066 ms restore (line 64 of the log, a 281-token restore with a 1 ms split) is a wait outside any copy -
  likely `SessionDir::claim`/`fetch` waiting for a writer block of that conversation's own file. This plan's
  change removes the park-side contention that fed it; if the card still shows restores without copies being
  slow, the claim-wait is the next thread to pull.

## Files changed

* `engines/qfn/include/strata/core/conversation_buffer.hpp` - `alias()` (a counted view), `contiguous()`, `charge_`.
* `engines/qfn/include/strata/core/park_policy.hpp` - `adopt_turn_capture` and `reuse_turn_state` (the pure
  decisions).
* `engines/qfn/include/strata/core/conversation_snapshot.hpp` - `conversation_snapshot_from_capture`,
  `conversation_snapshot_kv_bytes`, `conversation_snapshot_save`'s `live_from` running-state host copy.
* `engines/qfn/src/core/conversation_state.cpp` - both implementations and the `live_from` branch (no device
  traffic in the adopted path; no running-state device copy on the reused one).
* `engines/qfn/src/program/generate.cpp` - the `TurnHold` adopt fields and their clear sites, the stash in
  `turn_capture`, the decision + adoption + estimate subtraction + M1 re-admission in `park_body`, and the
  running-state reuse.
* `engines/qfn/src/core/park_policy_test.cpp` - 34 -> 48 checks.
* `scratch/card-test-park-overlap.sh` - the manager's card test (scratch/, untracked by the plan).
