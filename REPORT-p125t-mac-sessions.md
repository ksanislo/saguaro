# REPORT - P125t: mac/qfn-sessions merged (466962d1b) - CUDA build, the plan's tests, and the shared findings (a)-(d)

Branch `saguaro/q1-p125t-mac-sessions` = dev (bd29db04c..f2e39c87d) + the merge of Ken's Mac-session branch
(bfc6c3532) + the link fix (21fece114). The review round (REVIEW-RESULT.md) and this report's fixes on top:

* `aeb80e5cd` - finding 1: a request read refused for RAM now RETURNS (was a livelock).
* `e0c7193c0` - finding 3: session_dir_test covers the RAM-refused read (81 -> 86 checks).
* `ac87ee8bd` - the three nits (window.cpp / stream.h / session_dir.cpp comments).

## Build

`scratch/build-p125t` (tools/qfn/build.sh config: CUDA sm_75, AVX floor, tests ON, QFN_GGML_DIR=/data/llm/strata-main/
third_party/llama.cpp) - incremental ninja after the fixes: `strata strata_kq strata_kq_tab` exit 0, session_dir.cpp
rebuilt, **zero warnings** in the build output. `serve/qfn_engine.py` and `engines/qfn/serve/server.py` are untouched
by every commit here, so the extractor had nothing to re-run.

## Tests (the plan's modules, run ALONE, one at a time, at ac87ee8bd)

| module | result |
|---|---|
| session_dir_test | **86 checks OK** (81 + the 5 new refusal checks) |
| conversation_cache_test | 4810 OK |
| conversation_file_test | 235 OK |
| conversation_snapshot_test | **FLAKY FAIL - pre-merge, owed card below** |
| conversation_validation_test | 972 OK |
| conversation_memory_test | 45 OK |
| park_policy_test | 56 OK |
| turn_save_test | 97 OK |
| session_save_reclaim_test | 119 OK |
| serve.test_server | 183 OK |
| serve.test_priority | 80 OK |
| serve.test_savings | 36 OK |
| tools.test_maya_metal | 26 OK |

(serve.test_priority/test_savings count 80/36 here against the review's 75/34 at the same HEAD; the extra tests are
not this branch's files - everything named ran and passed.)

## Findings on the shared changes (a)-(d), at this branch's tip

### (a) session_dir.cpp - a RAM-refused read keeps its file - was BROKEN, now correct

The keep itself is right, but the REQUEST side livelocked (review finding 1): the transient branch set the entry back
to `idle` inside `entries_` and fell through to the top of `fetch`'s `for (;;)` - `match()` re-matched the unchanged
entry (session_dir.cpp:617-619), took it again (idle -> loading, :686), called `read_file` again, refused again,
forever: no answer, no error, CPU/disk hammering (each attempt re-mmaps, re-parses and re-checksums the file before
the preflight refuses it), and `fetching_` toggling (:687,:696) kept the preload thread's wait (:528) blocked.
Fixed at session_dir.cpp:702-710: after keeping the entry and logging, `cv_.notify_all(); return std::nullopt;` -
tokens stay 0, so the generate.cpp caller (its `fetch` result check) falls through to the plain prompt path: the
request gets a clean answer, and the kept file waits for a turn with more RAM. The plan's checks now hold:

* **no re-admission loop** - fetch returns on the first refusal (the new test times it);
* **the kept file still counts and is evicted** - it stays an idle entry, so `enforce_caps_locked` counts it toward
  max_files/max_bytes and picks it for age/LRU like any other (session_dir.cpp:822, unchanged logic; the review's
  813-829 before this round's nine inserted lines above it);
* **the refused request gets a clean answer** - nullopt with tokens 0 = the pre-change failure shape;
* **preload side** (session_dir.cpp:557-561): returns after the first refusal and preloading stops - consistent
  with the one-shot design (:535 `if (!pick) return;`, `start_preload` called once); left as it was;
* every other failure still drops the file as unreadable (session_dir.cpp:712, :564), and `transient` is set only
  for `SessionError::memory` (read_file, session_dir.cpp:744-755 - both maps set `st` on every path);
* the yarn-retry branch (session_dir.cpp:749-755) now carries the why-comment (the first refusal there is
  non-memory by construction; `st` describes the yarn attempt, whose memory refusal is equally transient).

The review's finding 3 (zero coverage of the keep/refuse branches) is closed by the new session_dir_test block:
write + flush, reopen the folder with an `Options.limits.admit` that refuses, `fetch` -> nullopt with tokens 0 and
**returns** (timed), the file stays indexed and on disk, and opening the admission on the next turn reads the kept
file back. Without fix 1 that test never returns.

### (b) saguaro.py - open_browser / SAGUARO_NO_BROWSER - TRUE, behaviour-off unchanged

`saguaro.py:5251-5253`: with no `open_browser` key (get -> None, `is False` false) and unset/`0` SAGUARO_NO_BROWSER,
`no_browser` is False and `--open` is appended exactly as before. The key is an existing schema entry
(engines/qfn/serve/runconfig.py:41, honoured at engines/qfn/serve/server.py:9768, docs/SETTINGS.md:2257). Gap, as
the review classed it: no test covers the new branch - noted, not fixed here (the review lists it as the same
non-blocking gap class as finding 3, and the fix round's scope was finding 1's test).

### (c) generate.cpp VRAM lines - TRUE, CUDA/HIP print byte-identically

`metal_vram_parts` (generate.cpp:423-435) off Metal sets only `prompt = 0` and leaves weights/kv/pool untouched: the
`extra` tail stays "" and the trailing `%s` prints nothing (the two VramLive lines, :13308, :13553 areas), `known`
gains +0, `used/vt/vf` are rewritten only under STRATA_GPU_METAL, and the reprint fingerprint sees no-op
`metal_vram_parts` and `pool += 0`. The server parses these lines as generic `k=v` tokens
(serve/gpu_map.py, serve/qfn_engine.py), so a Metal `prompt=` tail is tolerated and CUDA lines change not at all.
Left unfixed on purpose, as the review left it: the fingerprint's `b[]` does not include the reprint path's `mw`
(generate.cpp:13536), so a window-weights-only change would not retrigger a reprint - Metal-only, cosmetic, and
adding it would change Metal reprint behaviour, which is not "what is wrong" on this branch.

### (d) conversation_snapshot.cpp / conversation_state.cpp / conversation_memory.cpp / expert_source.cpp - TRUE

Every new non-Metal-visible line sits inside `#if defined(STRATA_GPU_METAL)` / `#elif defined(__APPLE__)`; the only
shared CUDA-visible edit is `conversation_snapshot_save`'s reference overload forwarding `live_from`
(conversation_state.cpp:592-597, header defaults at conversation_snapshot.hpp:113-117/152-155) - byte-neutral for
8-argument callers, and it is what makes conversation_snapshot_test LINK again (21fece114). mac_memory.hpp is
header-only and answers false off-Apple.

## Owed manager card (review finding 2): conversation_snapshot_test "A/B/A byte-exact K/V and indexer state"

**Not introduced by this branch and not fixed here** - the shared diff changes no CUDA-visible behaviour (section
(d)), and the prebuilt sibling binary at base-ancestor `f8ab034eb`
(../saguaro-q1-p124e-elastic-kv/scratch/build-p124e/conversation_snapshot_test) fails the same check on this host
today. The regression window the review bisected is 85b5a6523 -> f8ab034eb, the P124e elastic-KV work
(d49bedb40 kv_stream_reset bound / kv_ring_fill / SlotArray fill, 8b16a85a2 restore resets only the mapped part,
9d621f1bb prefix-mode host ends, all in conversation_snapshot.cpp / layer.* / kv_stream.cu). Older trees pass
(p124a/p124a2: 4195 checks OK). Since 21fece114 the test links again, so this branch finally SHOWS it; it cannot
ship green until the card lands. New facts measured for the card at this tip (all one at a time, T4 sm_75,
CUDA 13.1; artifacts: scratch/snap_probe.cpp, scratch/probe.err, scratch/snap-run*.out, scratch/snap-nofast*.out):

* **It is INTERMITTENT, not the deterministic fail one run suggests**: the real test 3/6 fails at HEAD, 4/6 fails
  with SAGUARO_PARK_FAST=0 (so the threaded park copy is not the trigger), and the f8ab034eb prebuilt sibling
  binary 3/4 fails today. The instrumented copy (scratch/snap_probe.cpp, prints the failing case) 5/6 fails.
* First failure aborts (conversation_snapshot_test.cpp:24), so the failing (fmt, mode, upto) varies by run:
  observed (0,1,1), (0,1,3), (0,0,65), (2,1,1), (2,1,65) - `index=true` always (modes 0/1 fail, mode 2 is never
  reached), and the first differing bytes are ALWAYS the pooled indexer rows, layer 0: `pooled[0]` reads back
  **0x2d where 0x89 was saved** - by the fixture's fill formula (test :95, byte0 = salt + i*31 with pooled at
  source i=4: fill(13) -> 137 = 0x89, fill(177) -> 301 mod 256 = 45 = 0x2d), so intermittently the A-restore
  leaves **B's** bytes in the pooled rows: the restore sometimes does not (or not yet) land the pooled rows.
* Suggested angle for the card owner: the suspects are exactly the restore-side mapping/landing changes
  (8b16a85a2 "resets only the mapped part of an elastic page table", 9d621f1bb prefix-mode ends, d49bedb40's
  ring/reset) racing or skipping the pooled part of the conversation_kv_restore -> save round trip.

## Nits / gaps left as the review left them

* The fingerprint's `mw` omission (generate.cpp:13536) - Metal-only, cosmetic; see (c) above.
* No test for the saguaro.py NO_BROWSER branch - see (b) above.
* The snapshot fail (2) stays open until the P124e card lands; every other test the plan names passes at this tip.
