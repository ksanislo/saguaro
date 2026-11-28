# REPORT-p124y-heap-crash — the 1.2.4-pre QFN engine heap corruption (mangonel SIGSEGVs, mortar crash loop)

## Verdict

The corruption is a **double `cuMemRelease`/`hipMemRelease` of VMM physical-chunk handles on the
P124e elastic-VRAM grow paths**. Fresh chunks gathered into the grow calls' `got` vector were
handed to a `VmmRange` by a `take()` lambda that left them visible in the vector; the grow calls'
"release what the map did not take" cleanup loops then released handles that the range still
held. The tree's own contract (`qsa_slot_grow` / `qsa_slot_trim` / `VmmRange::release`) is that a
landed chunk's handle reference lives in the range and is given back exactly when the range
reclaims the chunk. A double release frees physical VRAM whose **handle value the driver may
already have recycled for another owner's live allocation** — silent cross-owner memory
corruption that surfaces later, far from the cause: mangonel's `malloc` garbage inside
`session_capture`'s error-string build, and mortar's exit-(-11) crash loop inside
`libamdhip64`/libc. Fixed in `engines/qfn/include/strata/core/vmm.hpp` +
`engines/qfn/src/core/layer.cpp` (see below).

## Step 1 — ASan/UBSan over the named unit tests (CUDA-free, standalone g++ 14.2, `-std=c++20 -fsanitize=address,undefined -fno-sanitize-recover=all`)

All modules compiled from source and run **one at a time**; results identical before and after
the fix (none of these translation units include `vmm.hpp`/`layer.hpp`):

| module | result |
|---|---|
| park_policy_test | 56 checks passed |
| park_release_test | 25 checks passed |
| turn_save_test | 97 checks passed |
| conversation_cache_test | 4810 checks passed |
| conversation_snapshot_test | **exit 77 — skipped, no CUDA device** ("No GPU runs") |
| conversation_file_test | 232 checks passed |
| conversation_copy_streams_test | **exit 77 — skipped, no CUDA device** |
| conversation_memory_test | 45 checks passed |
| session_dir_test | 81 checks passed |
| session_save_reclaim_test | 119 checks passed |
| kv_host_growth_test | 25 checks OK |
| read_backing_test | 82 checks OK |

No ASan or UBSan report anywhere. An additional multi-round ASan stress of the crash-window
machinery itself (park → turn capture → background writer → claim/evict → `parked_block_pool_trim`,
with a concurrent churn thread, `scratch/p124y_park_stress.cpp`, scratch-only) is also clean
under ASan and under TSan. Conclusion: the host-side park/turn-save/pool machinery is not the
corruptor — which is exactly what step 2 then localized to the device-side grow paths.

## Step 2 — review findings (file:line as reviewed)

1. **Turn-save adoption (`park::adopt_turn_capture` / `reuse_turn_state`, generate.cpp) is sound.**
   Every rewrite site of the main session and of each batch slot (`turn_main.reset()` sites, the
   slot restart/`begin` sites, the checkpoint-mount site) clears the adoption hold; the whole-capture
   and running-state reuse gates cannot fire on a stale capture. The ASan stress confirms no
   use-after-free along adopt → writer → drop. (Plan bullet "a host buffer adopted by the park AND
   freed/reused by the turn save writer": not present.)
2. **The parked-block pool is sound.** `ParkedBlockPool::take/give/trim` are mutex-guarded and
   allocator-symmetric (`std::allocator<uint8_t>` n/count on both sides); the writer holds
   `shared_ptr<const SavedConversation>` per job, so a block cannot re-enter the pool while an
   image write still reads it. (Plan bullet "a pooled park buffer returned to the pool while a
   snapshot still points at it": not present.)
3. **The real double-free is in the elastic VRAM grow paths** (`layer.cpp`, P124e round 4/7):
   `qsa_follow_grow` (follow pools + the scratch start step), `qsa_follow_grow_at`
   (block-score scratch), the scratch-start step in the scores pool, and `rope_map`
   (RoPE/YaRN rows). Each: `got` ← `vmm_grow_room(_pooled)`, `map_range(lo, hi, take)`, then
   unconditional `for (h : got) vmm_chunk_free(h);` release loops plus the `follow_map_locked`
   failure loop that unmaps+releases fresh chunks. With a `take()` that does not remove from
   `got`, every chunk that landed in the range is released by the loop **and again** at reclaim
   (`qsa_follow_trim` / `VmmRange::release` `vmm_chunk_free(unmap(i))`), and on the failed-map
   path the cleanup loop and the caller loop free the same landed handles twice. This runs on the
   "conversation grows past its backed window" path — mortar parked 68k-token conversations and
   grew follow pools continuously, and its aborts were `libamdhip64` crashes; P124w had already
   documented the same grow machinery mangionel-faulting through `rope_map`/`qsa_follow_grow`.

## Step 3 — the fix, the regression test, the build

* `vmm.hpp`: new `strata::core::take_grown_chunk(got, i)` — hands out `got[i]` **and** zeroes it,
  so a release loop can never free a chunk the range owns (`vmm_chunk_free(0)` is a no-op).
* `layer.cpp`: the four grow paths' `map_range` take lambdas now call it. Landed chunks' handle
  references live in `h_` and are given back exactly once, by `qsa_follow_trim` /
  `VmmRange::release`, matching `qsa_slot_trim`'s long-standing contract; the budget chunks on
  the rope path (`i < from_budget`) are untouched by the release loops as before.
* **Regression test** `engines/qfn/src/core/vmm_grow_ownership_test.cpp` (registered in
  `engines/qfn/CMakeLists.txt` beside `read_backing_test`): models the grow protocol with
  per-handle release counters and asserts **each handle is released exactly once** on both
  outcomes (grow success; failed map with cleanup), and pins the pre-fix protocol (take without
  the handover) as double-releasing on both — so the file fails before the fix (the helper does
  not exist; and the legacy-model pins document the corruption) and passes after, standalone
  under ASan+UBSan and inside the build tree: `vmm_grow_ownership_test: 11 checks passed`.
* **Build (plan-mandated)**: `QFN_GGML_DIR=/data/llm/strata-main/third_party/llama.cpp tools/qfn/build.sh scratch/build-p124y`
  (sm_75, CUDA 13.1, Release): `QFN BUILD OK 3b94f71dc -> scratch/build-p124y` — `strata`,
  `strata_kq`, `strata_kq_tab` + unit tests, `layer.cpp` rebuilt with the fix, `ninja -n strata`
  clean afterwards. Named modules re-run from the build tree, one at a time:
  `vmm_grow_test 56 OK`, `read_backing_test 82 OK`, `kv_host_growth_test 25 OK`,
  `vmm_grow_ownership_test 11 passed`, plus the ten CUDA-free modules above green from the same
  tree. No GPU was used (both CUDA-only tests exit 77 on this host).

## Commits

* `P124y (1.2.4-pre crash hunt): the elastic VRAM grow paths double-released VMM chunk handles`
  — `vmm.hpp`, `layer.cpp`, `vmm_grow_ownership_test.cpp`, `CMakeLists.txt`.

## Open questions

1. **Mangonel (T4, no elastic KV):** the T4 candidate never ran the follow-grow paths, so this
   fix does not explain its `malloc`-path SIGSEGV at `session_capture:492`. Next bisect knobs on
   a T4 rig: run the candidate with `MALLOC_CHECK_=3` / an ASan engine build, and test
   `SAGUARO_YARN_ELASTIC=0` (the `rope_map`/YaRN grow path — the same fixed take/ownership family —
   does run on mangonel's T4s when YaRN rows grow).
2. **Backport:** the fix applies cleanly to both rc124 candidate lines (`saguaro/rc124-cand`,
   `saguaro/rc124-cand2`); recommend cherry-picking before the 1.2.4 cut.
3. **ROCm behavior:** confirm on the mortar RX 9070 XT whether `hipMemRelease` on an
   already-released handle returns an error or releases a recycled handle's allocation there
   (the crash-loop evidence says the latter); if an error is available, a debug-build assertion
   in `vmm_chunk_free` would catch any future violator of the handover contract at the source.
4. Card-level confirmation: re-run the mortar pressure profile (`tools/card_tests/mortar_pressure.sh`
   family) against the fixed engine.
