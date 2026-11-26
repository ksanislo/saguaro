# REPORT - P124a: the QFN conversation park lost seconds and the false "host is short of RAM"

Branch `saguaro/q1-p124a-park`, engine tree `engines/qfn` (Strata 0.1.49.2). Card evidence:
`ref/mortar-park-lines-20261225.log` (RX 9070 XT, 96 GB RAM, QFN UD-Q4, resident RAM mode 68.00 GiB pinned).
Nothing was run on a GPU here; the engine was built and its CPU-side test modules run (section 4).

## 1. The lost seconds: the park path end to end, and what the split's clock covers

Where a park happens: when the next request takes the session, the serve thread parks the outgoing
conversation before restoring the incoming one - `park_current` (generate.cpp:19722 borrow, :19748 switch),
`park_body` (generate.cpp:10796); a slot's park is the same body (generate.cpp:15679).

Steps between "the request ended" and the engine's `parked N tokens in T ms` line (generate.cpp:10987),
with split coverage (the split is a thread-local record, conversation_copy.hpp:186-196, taken at each park/restore line):

1. before the reported clock (`t0`, generate.cpp:10914): shared-pool budget (:10812), drop superseded
   copies (:10832), checkpoint split (:10845), per-stage + main K/V byte estimates (:10864-10901),
   `make_room` (:10908). Not counted in T.
2. inside T, un-timed: `admit()` - one `sample_host_memory()` per loop (generate.cpp:10933;
   conversation_memory.cpp:258) plus `evict_oldest` retries.
3. inside T: `conversation_snapshot_save` (conversation_state.cpp:324) -
   * the `sync()` before the capture (:329) - timed as **sync**;
   * the checkpoint *tree*'s host copies (:338) - timed as **checkpoints**;
   * **`conversation_checkpoint_save` (:349, body :177) - timed by no clock at all.** This is the live
     session's running state - `gdn`, `ple` and three arrays per QSA layer - ~237 MB for this model
     whatever the token count (the log's 62-token park carries 237 MB), copied in roughly three
     copies per layer, each through `copy_nonblocking` (conversation_copy.hpp:124) which **synchronized
     the private stream once per copy**: ~200 driver round-trips, each also a wait for the stream's
     tail, and none of them inside alloc/copy/sync/checkpoints.
   * `conversation_kv_save` per layer (conversation_snapshot.cpp:263): buffer growth timed as **alloc**
     (:292), the jobs as **copy** (:302 → transfer_jobs at :206).
4. inside T, un-timed: the per-stage `device_sync()` loop (generate.cpp:10958 - only with a layer
   split, none on the one-card mortar), the second `sample_host_memory()` (:10966),
   `conversations.put` (:10976), `sdir->write` (:10977, a short lock of the folder's mutex).

The split's own arithmetic proves it also *lied*: it is only taken at park/restore lines, so the other
snapshot work on the serve thread - a turn save's `sync()` (`conversation_snapshot_sources`,
conversation_state.cpp:375) and a borrow's restore copies (generate.cpp:19821+) - piled into the next
park's numbers. Mortar line 22: `parked 36326 tokens in 123.4 ms` with a split that sums to 161 ms.

What the slow parks have in common in the card log: each one starts within ~1 s of the cut request's
background session write of the *same* conversation ("saved ... after its turn (main): 250 MB appended
(in the background)", the folder's writer thread streaming that conversation's K/V and running state
off the same device pools on its own private streams, then appending it to disk), and the engine runs
nice +5 on that box. Every step of the park that talks to the device - above all the ~200
one-synchronized state copies of item 3 - queues behind that traffic, and all of it sat outside the
split's clocks: the 2.0-3.2 s gaps in the log (total vs split) are exactly it. The fast parks of the
same size are the cases where that background write had already finished. Without the card I cannot
attribute the wait further than "the un-timed stretches listed above", so the fix below *times* the
whole park - section 3 - and the card test will say in one number what remains.

## 2. What "pressure: 8.6 GiB free" measures

`park_tick` (generate.cpp:18204) samples `sample_host_memory()` (conversation_memory.cpp:258):
`available` is the **smallest** of MemAvailable, every visible cgroup's `limit - (memory.current -
clean inactive cache credit)`, and the operator's `--memory-limit-mib` headroom. `pressure_need`
(park_policy.hpp) calls it pressure when that figure is under `SAGUARO_PARK_FREE_MIB` (16 GiB), and
the tick trims the park pool and releases parked conversations until it is above it.

On the mortar box the host was not short of RAM (`free -g`: 81 GB) - the reading was pinned *permanently*
under the floor by the engine itself: 68 GiB of page-locked experts plus the host K/V pool are charged
in full against that figure (conversation_memory.hpp:21-26: only clean *inactive file* cache is ever
credited; pinned, locked and shmem stay). So the check counted the engine's own pinned tier as "used by
others": a resident-RAM engine can never clear the floor, `pressure_need` answered >0 every tick, and
every park was dropped seconds after landing - the pool trimmed, the RAM given back, the next turn
reading from disk, the next park paying it again. Both readings (MemAvailable and the cgroup room)
contain our own pinned bytes, so no switch of sources fixes it - the *release rule* has to credit back
what no release can hand over. That is the fix.

## 3. The fix

* **The pressure test judges what a park can actually use.** `Memory::own_pinned`
  (park_policy.hpp) carries the RAM the engine holds pinned by design and no release gives back - the
  expert tier and the live K/V pool - and `pressure_need` credits it before comparing to the floor.
  `severe()` (the hard floor that drops unsaved entries) and the park's physical-RAM admission keep the
  raw reading; PSI keeps running. `park_tick` fills it from the source that is live
  (`src.resident_bytes()` / the arena's pinned bytes, plus `qsa_kv_host_bytes()`, generate.cpp:18210),
  and a release line on such a box now says so: `pressure: X GiB free, short even crediting the
  engine's own Y GiB pinned`. On the mortar figures (68 GiB own, 8.6 GiB "free", 16 GiB floor) the
  need is now 0: parks stay parked; a host that is genuinely full still releases.
* **The park line times the whole park.** The park resets the split when it starts (generate.cpp:10915)
  and the restore section when it starts (generate.cpp:19820), so neither counts another operation's
  waits (the 161 ms split under a 123 ms park is gone). The running state's copies are timed into the
  split's `checkpoints` part (conversation_state.cpp:349), and the park line appends
  **`wait N ms`** = total - (alloc + copy + sync + checkpoints) (`conversation_snapshot_take_split(park_ms)`,
  conversation_snapshot.cpp:248) - the outside time is said, not hidden.
* **The lost seconds, where they are ours:** `conversation_checkpoint_save` queued its ~200 copies
  one-by-one with a stream synchronize after each (conversation_copy.hpp:134) - a driver round-trip
  per copy, none of it measured. All of the state is now queued on the thread's private stream and
  waited for **once** at the end (`copy_stream_done`, conversation_copy.hpp:142; the wait also happens
  on the failure paths, so the caller can still free its buffers safely). Same bytes, same "complete on
  return" contract, one wait instead of two hundred; and since the wait is now inside the split, what
  contention remains will show up in `checkpoints`/`wait` on the card rather than as a gap.
* **CUDA/HIP/P40:** every change is in backend-neutral C++ (`cudaStream*` calls the files already
  used); the CUDA sm_75 build is green (section 4). No new environment switch.
* **Server side:** no change - `serve/session_event_text`'s "the host is short of RAM"
  (engines/qfn/serve/server.py:1033) describes a release that, after the fix, only happens when the
  host really is short. `serve/qfn_engine.py` was therefore not regenerated
  (tools/saguaro/extract_qfn_engine.py untouched).

## 4. Tests run

* `park_policy_test.cpp` extended with the mortar case: 96 GB host, 68 GiB pinned by us, 8 GiB read
  free -> `pressure_need == 0` and `plan()` releases nothing; without the credit the same reading is
  still the old pressure (8 + an eighth asked); a raw 1 GiB is still `severe` (unsaved entries still
  go); the PSI/budget/idle cases keep passing. **32 checks passed**, standalone (g++ -std=c++20) and
  as the target `park_policy_test` of the real build.
* Engine build proof: `tools/qfn/build.sh scratch/build-p124a` - CUDA, sm_75,
  `/usr/local/cuda-13.1/bin/nvcc`, fresh build dir: **QFN BUILD OK** (179/179, `saguaro-qfn` linked).
  The targets `park_release_test`, `conversation_snapshot_test` and `conversation_cache_test` compile
  and link against the changed `conversation_checkpoint_save` / split API. Nothing was run on a GPU.
* Card test for the manager: `scratch/card-test-park.sh` - starts a test copy of the server on its own
  port 8092 with its own `--session-dir` (all files under `scratch/park-card/`), runs conversations of
  1K, 30K and 100K tokens three times each: ask A, switch to B (park A), idle past a pressure tick,
  ask A again (restore), then prints every `parked/restored/dropped` line with its times and PASS/FAIL
  on "released for pressure" and "restored from RAM". `SIZES=... PORT=... SAGUARO_TEST_EXE=...` override.

## 5. Open questions

* The exact identity of the remaining wait on the mortar (the writer thread's device traffic vs the
  nice +5 scheduling vs page contention) is what the new `checkpoints`/`wait` fields will say on the
  first card run - if `checkpoints` now shows the seconds, the follow-up is to stop parking and turn-
  saving the same conversation's running state twice (share one capture), which this change does not do.
* `own_pinned` credits the *tier* and the K/V pool only - the parked cache and the pool of freed park
  segments are deliberately not credited (they are what the rule exists to free). If a future resident
  mode pins something else at startup, it belongs in this figure.
* The release line's `pressure: %.1f GiB free` still prints the raw `available` (now with the credit
  named); `serve/qfn_engine.py` and the dashboard read only the `sess_why=pressure` field, so nothing
  parses the text - it stays human-facing.

## Files changed

* `engines/qfn/include/strata/core/park_policy.hpp` - `Memory::own_pinned`; credited `pressure_need`; doc.
* `engines/qfn/src/program/generate.cpp` - split reset at park/restore start; whole-park timing +
  `wait N ms`; `own_pinned` sampled in `park_tick`; the credit said in the drop line.
* `engines/qfn/src/core/conversation_state.cpp` - running-state copies timed; one batched wait instead
  of ~200 per-copy stream syncs.
* `engines/qfn/src/core/conversation_copy.hpp` - `copy_stream_done`.
* `engines/qfn/include/strata/core/conversation_snapshot.hpp`, `engines/qfn/src/core/conversation_snapshot.cpp`
  - `conversation_snapshot_take_split(total_ms)`, `conversation_snapshot_reset_split()`.
* `engines/qfn/src/core/park_policy_test.cpp` - the mortar case (28 -> 32 checks).
* `scratch/card-test-park.sh` - the manager's card test.
