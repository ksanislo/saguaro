# REVIEW-RESULT - P125t re-review (round 2): the fix commits (aeb80e5cd..e8beebfae) on top of the merge, verified

**MERGE-WITH-NITS** - no defect found in this diff; both claims to verify ((c) and (a)) hold with the
file:line below; every module the plan names passes except `conversation_snapshot_test`, which fails
deterministically here (6/6 runs) but is proven pre-merge (the f8ab034eb sibling binary fails the same check).
The branch stays red until the owed P124e card lands - that is a separate fix, not of this diff.

All runs were done by this reviewer alone, one test module at a time; nothing but this file was written.

## Verifications the brief asked for, with file:line

### Claim (c) - off Metal the two VRAM lines print byte-for-byte as before: TRUE

* `metal_vram_parts` (engines/qfn/src/program/generate.cpp:423-436): off Metal it is exactly
  `prompt = 0;` (generate.cpp:424) with `weights/kv/pool` explicitly untouched (generate.cpp:434).
* Start-up line: `extra` stays `""` (generate.cpp:13309-13310), so the new trailing `%s` (generate.cpp:13329)
  prints nothing; `known` gains `+ prompt` = +0 (generate.cpp:13312); `used` is the same `vt - vf`
  (generate.cpp:13311) and the `used/vt/vf` rewrites sit inside `#if defined(STRATA_GPU_METAL)`
  (generate.cpp:13313-13321).
* Reprint line: `w` starts as `c.w` and is untouched off Metal (generate.cpp:13552-13553), `known` +0
  (generate.cpp:13557), Metal-only used block (generate.cpp:13558-13563), trailing `%s` empty
  (generate.cpp:13570). The reprint fingerprint sees a no-op: `mw == c.w` and `pool += mprompt` adds 0
  (generate.cpp:13535-13537), so `b[]` (generate.cpp:13538-13542) is unchanged off Metal.
* The parser tolerates a Metal `prompt=` tail because it reads generic `k=v` tokens and `prompt` is already a
  known byte field (serve/gpu_map.py:11-13, :25-33); off Metal nothing changes. serve.test_server/-priority/
  -savings: 183/80/36 OK.

### Claim (a) - the RAM-refused read keeps its entry, answers the request, and does not loop: TRUE

* Keep + clean exit: `fetch`'s transient branch sets the entry back to `idle` inside `entries_`, logs,
  notifies and returns `std::nullopt` (engines/qfn/src/core/session_dir.cpp:702-709). `tokens` is left at the
  caller's 0 (engines/qfn/src/program/generate.cpp:20340) and the caller calls `fetch` once
  (generate.cpp:20346-20347); with no image the restore gate `if (fetched && from_disk > ...)` fails and the
  request proceeds down the plain prompt path (generate.cpp:20361). No re-admission: the return happens on the
  first refusal; the loop top can no longer re-match the kept entry.
* Still capped/evictable: `enforce_caps_locked` counts every entry toward `max_files`/`max_bytes`
  (session_dir.cpp:829-831) and evicts `idle` entries by age (session_dir.cpp:824-826) and LRU
  (session_dir.cpp:832-837) - a kept entry is a plain idle entry.
* Preload side: a memory refusal sets the entry back to `idle` and RETURNS from `preload_loop`
  (session_dir.cpp:557-561); `start_preload` is called once at startup (session_dir.cpp:129-132 ←
  generate.cpp:16447), so nothing re-picks it; the destructor's `preloader_.join()` is safe with the early
  return (session_dir.cpp:67). A request parked on the preload's `loading` state (session_dir.cpp:665-671)
  is woken by the `notify_all` (session_dir.cpp:560) and then tries its own read - once, same exit.
* Only `SessionError::memory` takes this path: `transient` is set exactly there in `read_file`
  (session_dir.cpp:744-747 and the yarn retry at session_dir.cpp:752-754), and both readers set `st` on every
  failure path with `memory` only for admission refusal / mmap-bad_alloc
  (conversation_file.cpp:1292-1293, :1334-1341, :1451-1453, :1497-1500, :1505-1508;
  session_log.cpp:853-854, :907-911, :923-924). Every other failure still drops the file as unreadable
  (session_dir.cpp:710-712 request, :562-564 preload).
* Coverage exists and exercises the branch: session_dir_test.cpp:273-307 writes + flushes a conversation,
  reopens with an `Options.limits.admit` that refuses, and checks `fetch` RETURNS (timed bound - the livelock's
  pin), returns nothing with tokens 0, keeps the file indexed and on disk, and reads it back on the next turn
  with the admission open (session_dir_test.cpp:306). 86 checks OK.

### The rest of the standing checklist

* **Feature OFF byte-identical**: saguaro.py appends `--open` exactly as before when no `open_browser` key
  (None is not False) and `SAGUARO_NO_BROWSER` is unset/`0` (saguaro.py:5251-5253; the key is the pre-existing
  schema entry engines/qfn/serve/runconfig.py:41, honoured at engines/qfn/serve/server.py:9768).
  mac_memory.hpp answers false/0 off-Apple (engines/qfn/include/strata/platform/mac_memory.hpp:44-47, :58-61,
  :72-74, :99-102), so the new `__APPLE__` branches in conversation_memory.cpp (:283-289, :329-333),
  expert_source.cpp (:504-508, :1982-1984) and generate.cpp (:1736-1744, :1787-1791) are dead code on CUDA.
  Every non-Metal-visible line in conversation_snapshot.cpp / conversation_state.cpp sits inside
  `#if defined(STRATA_GPU_METAL)`; the single shared edit is the reference `conversation_snapshot_save`
  forwarding `live_from` (conversation_state.cpp:591-597) whose header default is `nullptr`
  (conversation_snapshot.hpp:113-117, :152-155) and which is only consulted when non-null
  (conversation_state.cpp:372-381) - byte-neutral for 8-argument callers. The CUDA build of all of this
  produced the plan's tests passing.
* **Locks/threads/stdout**: no lock is nested - the new branches act only under the already-held `mu_`
  (`lk`), toggle `cv_` and return (the unique_lock releases on return); `fetching_` is set/cleared under the
  lock (session_dir.cpp:687, :696) so the preload's wait predicate (session_dir.cpp:528) still sees it. `say`
  is a single stderr fprintf (session_dir.cpp:71-73, generate.cpp:16442) and events are one whole-line
  printf+fflush (generate.cpp:16444) - no new interleaving surface.
* **Generated file**: `serve/qfn_engine.py` is untouched by every commit in the diff, and re-running
  `tools/saguaro/extract_qfn_engine.py` on HEAD reproduced it byte-identically (git stayed clean).
* **No paths outside the repo, no secrets**: the only absolute-path string in the diff is the REPORT text
  naming the build's `QFN_GGML_DIR`; no key/secret patterns.

## Findings by severity

1. **(High, pre-existing, NOT of this diff - ship blocker for a green run)**
   `conversation_snapshot_test` "A/B/A byte-exact K/V and indexer state" (the check at
   engines/qfn/src/core/conversation_snapshot_test.cpp:359, abort at :24) fails **6 of 6** fresh runs of the
   reviewer's own build at HEAD. The report's "intermittent, 3/6" reading did NOT reproduce here - today it is
   deterministic on this host; treat the intermittency text in REPORT-p125t-mac-sessions.md:107-108 as
   understated. Pre-merge attribution VERIFIED independently: the sibling prebuilt binary at the base ancestor
   (../saguaro-q1-p124e-elastic-kv/scratch/build-p124e/conversation_snapshot_test, built from f8ab034eb)
   fails the same check on this host. Correctly carded to the P124e elastic-KV restore work; nothing in this
   diff may be blamed for it, and nothing here fixes it - the branch must not be declared green until the card
   lands. No code change is owed on this branch for it.
2. **(Medium, gap, left knowingly)** No test anywhere covers the launcher's browser skip: the only references
   to `open_browser`/`SAGUARO_NO_BROWSER` are saguaro.py:5249-5253 itself (grep over serve/, tools/, tests/).
   The OFF-by-default behaviour was verified by inspection only. A `server_command` unit test (key false → no
   `--open`; env `1` → no `--open`; neither → `--open`) is cheap; the review round left it and so does this one.
3. **(Low, Metal-only, cosmetic - left as the round left it)** the reprint fingerprint omits `mw`: the weight
   figure goes to `metal_vram_parts` but not into `b[]` (generate.cpp:13535-13537 vs :13538-13542), so a
   window-weights-only change would not retrigger a reprint. Invisible off Metal (`mw == c.w`); fixing it
   would change Metal reprint behaviour, which is outside "fix only what is wrong".
4. **(Info)** a deferred (RAM-refused) request read emits no `sess_event`/`sess_json` note
   (session_dir.cpp:702-709 calls `say` only) - the folder state event simply re-emits on the next successful
   read. Same observability shape as the pre-change failure paths; no action.
5. **(Info)** a preload memory refusal stops the preload thread for the whole process lifetime
   (session_dir.cpp:557-561 `return`) - consistent with the one-shot preload design
   (session_dir.cpp:535 `if (!pick) return;`) and the plan ("preloading stops"); requests still read
   one-shot per turn. Noted so it reads as intended.
6. **(Info, report nit)** REPORT-p125t-mac-sessions.md:81-82 cites "serve/qfn_engine.py" as parsing the VRAM
   lines; the actual parser is serve/gpu_map.py:25-33 (qfn_engine.py only maps engine ERR text to
   RAM/VRAM exceptions). Harmless, but the citation is wrong.

## Build (fresh, from this tip)

`tools/qfn/build.sh scratch/build-p125t-review` (CUDA sm_75, AVX floor, tests ON,
QFN_GGML_DIR=<repo-parent>/strata-main/third_party/llama.cpp): `QFN BUILD OK e8beebfae`. 226 nvcc warnings -
the warning source set is **identical** to the coder's full build (scratch/build-p125t.build.log) and **not
one** warning source is a file this diff touches (checked per file) - no new warnings. `serve/qfn_engine.py`
extractor re-run: byte-identical.

## Tests (this reviewer's own build, one module at a time)

| module | result |
|---|---|
| session_dir_test | **86 OK** (incl. the 5 new refusal checks) |
| conversation_cache_test | 4810 OK |
| conversation_file_test | 235 OK |
| conversation_snapshot_test | **FAIL 6/6** - "A/B/A byte-exact K/V and indexer state" (pre-merge, finding 1) |
| conversation_validation_test | 972 OK |
| conversation_memory_test | 45 OK |
| park_policy_test | 56 OK |
| turn_save_test | 97 OK |
| session_save_reclaim_test | 119 OK |
| serve.test_server | 183 OK |
| serve.test_priority | 80 OK |
| serve.test_savings | 36 OK |
| tools.test_maya_metal | 26 OK |

(The 80/36 counts confirm the report's against the older review's 75/34.) Nothing else was run on the GPU
beyond the plan's own conversation_snapshot_test; no serve/model GPU runs.

## Verdict

MERGE-WITH-NITS. The shared (a)-(d) changes are correct as claimed, the round's fixes do what the last review
prescribed (livelock gone, pinned by a test that would not return otherwise), CUDA/HIP output paths are
byte-identical, and the generated file is honest. Merge with findings 2 (and optionally 6) as follow-ups;
do not call the branch green while finding 1's card is open.
