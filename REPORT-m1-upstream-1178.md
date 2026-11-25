# M1: upstream Project Maya v1.1.76 - v1.1.78 taken in (2026-11-24)

Branch `saguaro/q1-m1-upstream-1178` (worktree ./, off main b535d1fec). Upstream range v1.1.75 (8621f7a71) -> v1.1.78
(1016876a3) = 69 commits, 77 files. Not pushed, not merged into main.

## How the merge was done (one deviation from the plan's letter, worth a look)

`git merge --no-ff v1.1.78` on its own merges against the real merge base, which is **v1.1.50** - the v1.1.75 intake
brought upstream in as cherry-picked commits, so v1.1.75 is not an ancestor of our tree. The raw merge tried to re-apply
all of v1.1.51..v1.1.78 (460+ conflicts, every renamed line included) - it was aborted.

Instead: one plumbing commit `git merge --no-ff -s ours v1.1.75` (b706c3ea4, **no tree change** - it only records the
v1.1.75 lineage), then the planned `git merge --no-ff v1.1.78` (36356a5d2, "Merge upstream Project Maya v1.1.78
(v1.1.76-v1.1.78)"). With v1.1.75 as the merge base the merge covers exactly the 69 new commits and conflicted on 34
files. Later merges of v1.1.79+ will get v1.1.78 as the base and stay this small; that is the real benefit of the
anchor.

Commits (first-parent): b706c3ea4 anchor -> 36356a5d2 merge (plan step 1) -> b2e85aa52 settings rename (step 2) ->
7a364bad7 build-check fixes (step 3) -> a2c8a8173 merge-fallout fixes found by step 4 (model-API think=false,
IK2_VISION_GIB out, stale test assertion, buildlog import) -> 02c95f94a rename fallout (log prefixes, a comment) ->
7a230a65d INKLING2.md in our engine's words -> this report.

## Conflicts and how each was resolved (34)

| File | Resolution |
| --- | --- |
| .gitignore | both sides: our build-saguaro/vision-glm lines + upstream's build-engine.log / build-vision.log; the nightly files listed as `saguaro-nightly-*` (tool re-branded, below) |
| CHANGELOG.md | ours kept on top; short "Upstream Project Maya v1.1.76 - v1.1.78 taken in" section added (our names); upstream's own v1.1.76-78 entries not copied (our tree squashes upstream history, like v1.1.75's) |
| CMakeLists.txt | e755da30f's fetch fix in our names: our FetchContent is `maya_llamacpp` everywhere (v1.1.75 decision), upstream's `strata_llamacpp` reverted to it in the two auto-merged `_maya_gg` lines |
| VERSION | ours (1.2.0) |
| README.ja / .pt-BR / .zh-CN | deleted-by-us kept (translated READMEs are not ours; 82bb64cad's rows noted) |
| README.md | our speed tables (5-column format) kept; upstream's three new rows (#1177 RTX 5090 GLM, #1096 4x/5x 4090 DeepSeek) added in our format; the Inkling #1116 table is new here - the whole "Inkling 2 Mini (experimental preview) speed benchmarks" block was added in our format (main had no such table) |
| cmake/glm_dsa.d/60-glm-gaps.cmake, src/core/glm_dsa_fast.cu | deleted-by-us kept (no glm_dsa / full GLM-5.3 path; see dropped hunks) |
| docs/INKLING2.md | upstream's new drafter-room section taken (pictures take no room, #1185) in `SAGUARO_IK2_*` names; the offload paragraph then rewritten to OUR engine's reality (dynamic cache B4, no mode-A/tier-half/split-measure machinery we never took); our AMD-offload-off + `SAGUARO_IK2_DRAFT_FORCE=2` sentence kept; our `draft_gib` figures (0.65 GiB + 4 KiB) kept over upstream's |
| docs/SETTINGS.md | our settings table kept; upstream's changed rows ported into it in our names: `SAGUARO_GLM_CPU_PIN` (v1.1.77 shared-pool NUMA pin), `SAGUARO_GLM_CPU_CAL` (per-PC timing, `SAGUARO_GLM_CPU_CAL_REDO`, `saguaro.py --speed-calibration`; the SAME_ANSWERS wording our tree doesn't have left out), `SAGUARO_GLM_DSA_SCORE2` (window's tokens too); the new `MAYA_VRAM_LATE_MB` row merged into our section and renamed; the DS4 shipped-profile paragraph extension applied; upstream's deleted block (RDNA2 GEMMs / indexer / Qwen / Hy4 / Inkling / GLM-5.3 tables) stayed deleted - our tree documents Hy4/Inkling elsewhere and has no glm53/Qwen; the two changed HY4 rows (`HY4_SUM_PER`, `HY4_PREFETCH`) plus new `MAYA_IQ_MV_RW` went into docs/HY4-FLASH.md (where our Hy4 switches live) as `SAGUARO_IQ_MV_RW` |
| include/maya/core/glm_model.hpp | our deletion kept (glm-dsa `state_layout` / `dsa_n_sel_last`; new `fast_ffn_width` serves only the glm_dsa bind we lack) |
| saguaro.py (16) | the picture of the two setups' divergence; each: upstream's change kept where our tree has the feature (build-time texts "a few minutes", vision lean/fallback with our `build-vision-glm.bat`, `--bench-lock-ok` by name, smoke, `--bench --compare` priming + `MB.share_timing`, the #1156 reserve marks ported onto OUR display-reserve rule (`gpu_reserve_mib`/`gpu_reserve_why` kept, AMD marks `mark_reserve`/`mark_kept_reserve` added to GLM/DS4/MM3/Hy4 writers + the start path), `set_vram_reserve` added for our families incl. qfn), ours kept where upstream restructured code we never carried: their `g53_defaults`/`GLM_SETUP_ENV`/`SERVER_KEYS`/`keep_server_settings`, q4x/route first-start helpers, two-layout split measurement, `glm_lane_timing_again` and `build_step`'s `also=` rebuild-all (that test is deleted in our tree) - all dropped with the block that took `maya.py`'s shape |
| serve/frontend.py | new `check_messages` + `_JSON_KINDS` kept (called in both converters); `_clear_thinking` (glm_dsa) and the `glm_dsa` parameters dropped (not ours) |
| serve/server.py (12) | both kept: #1156 `mark_reserve`/`mark_kept_reserve` + `clean_stale_picture_dirs` + early `stop_signals` (#1181) + `_STARTING` + `check_messages` on the three OpenAI/Anthropic/Responses paths + the model-API one; `maya_names.export_env/alias_env/env_in` dropped (our settings have one name - `mark_reserve` compares directly), `attach.expand_request` dropped (no serve/attach.py here), `engine_for("has pictures")` and the old on-demand re-measure in main() dropped for upstream's `vision_plan` (auto-merged) with our `[saguaro]` lines; the DS4_YOCO_CONE fast_prefill env not taken (no such kernel switch here); `vision_plan`'s lend key renamed `SAGUARO_GLM_VISION_LEND_MB`, ik2 keeps no pictures room (upstream #1185) |
| serve/test_model_api.py, serve/test_stop.py | upstream's new tests taken (wrong-shape 400s incl. /api/chat - the check merged into `_mapi_chat`; SIGHUP/SIGINT preexec helpers), program names re-branded |
| src/core/glm_fast_path.cu (6) | kept BOTH: d5f00a0f5's `LanePin`/`lane_pin_cpus` shared-pool pinning and the packed-vs-spread first timing, b6752deed/1a4f9dd48's `cal_pc` + `cal_keep` (25% rule, `SAGUARO_GLM_CPU_CAL_REDO`) with our names; dropped the `glmtier::lane_pools` branch (glm-dsa B8, not ours), the same-answers rounding/`lane_steps` calls and the same-answers drift-drop exemption (our tree kept the plain drift-drop, matching our docs); #1182's shard-map drop (glm_model.cu, auto) applied |
| src/core/glm_fast_state.hpp | upstream's `lane_packed` taken; glm-dsa declarations stay deleted |
| src/core/vram_late.hpp | upstream's `why()` / `asked()` / cap-at-1-GiB lines taken (auto + small conflict), names renamed; our `/data/vram-room.json` etc. kept |
| src/ds4/ds4_iq.cu, src/ds4/ds4_iq.cuh | reverted to OURS, then only bdaeba1c1's hunks applied (warp_sum/lanes_sum through `maya_lane_xor`, lane_xor.hpp include) - upstream's `mv_heads`/`mv_reg` additions in this file sit on their shared hy4/ds4 build; ours keeps them in src/hy4/ds4 (see below) |
| src/hy4/ds4/ds4_iq.cu + .cuh (not conflicted - ported by hand) | the actual Hy4 target: bdaeba1c1 warp sums, b95319d5f `kMvRows` + `mv_heads_rows_kernel` + `mv_rows()` (`SAGUARO_IQ_MV_RW`, default on gfx1100) + `mv_heads_rows_nc` + the `mv_heads_nc` hook + `set_mv_rows`/`mv_rows_on` (tests/hy4_experts_test.cu needs them) |
| src/ds4/ds4_kernels.cu | both: our PLAN-turbo-live `eff_k` kept with upstream's `& ~nf` added (3c74b4b7c, #1114) |
| src/ds4/ds4_model.cu | our all-vendor reserve/`keep_b` kept + the #1156 `late_room` line; upstream's "gone to others" WARNING block stays deleted (we removed its `vis_late` accounting earlier); v1.1.76 `vramlate::why` INFO line taken with our `saguaro ds4:` prefix |
| src/ds4/ds4_prompt_mem.hpp, tests/ds4_prompt_mem_test.cpp | d6bac1e5e's 1.1 ms band (`kRingSlowBandS`) taken - without the `slow_bps` / `MAYA_DS4_RING_FAST_LINK` parameter (never ours): the band applies whenever the rule does; the upstream test case needing that parameter dropped, the rest kept |
| src/ik2/ik2_main.cu (9) | #1136: upstream's `fatal_exit` (auto) now used by our CB/die too and our richer `cu_fail` ends through it; e4864b217 pictures-take-no-room kept (our `SAGUARO_IK2_VISION_GIB` reads removed from the plan AND from the Pascal auto-chunk reserve); our drafting defaults block (Ken's table, `SAGUARO_IK2_*` + AMD-off-off) kept; upstream's offload tier-window room model, lane-parity measure (`g_off_measure`/`g_off_draft` - not in our tree) and the two-run first-start split measurement (`SAGUARO_IK2_DRAFT_PART`, measure.log) dropped with it; d6dff4940's #1107 loop-bank hop applied (auto); our room_ok kept with the pictures clause removed |
| tools/setup_bridge.py | b822a500d's `sink` parameter kept; `output_text` (whose def our tree had dropped) restored - OEM-codepage decode matters now that compilers write to the log |
| tools/test_bench.py | docstring merged (our saguaro-bench names + upstream's new note); body auto (lock/backoff tests), `MAYA_BENCH_LOCK_OK` etc. renamed |
| tools/test_calibrate_draft.py | upstream side dropped: it tests their PhaseEngine/link_calibration/GLM-draft-length suites, which we deleted with those features |
| tools/test_maya_hip.py | our renamed env + upstream's new `SAGUARO_VRAM_RESERVE_SETUP_MB` line (matches our hip_env) |
| tools/test_maya_rebuild_all.py | deleted-by-us kept (#1066 rebuild-all is not in our saguaro.py; compile_engine has no `also=`) |
| tools/vision/CMakeLists.txt | upstream's lean.cmake include + MSVC /utf-8 + `MAYA_VISION_LEAN` taken, our `maya_vision.cpp` name and old_names include kept; tools/vision/strata_vision.cpp's lean check auto-merged into our maya_vision.cpp (rename detection) |

New files taken (clean): docs/RELEASE-CHECKS.md, include/maya/kernels/lane_xor.hpp, serve/test_picture_dirs.py
(temp-dir prefixes re-branded `saguaro-*` incl. the test), serve/test_vision_cache.py, tests/glm_dsa/...
**deleted** (no glm_dsa here), tools/maya_changed.py / maya_golden.py / maya_nightly.py + their tests,
tools/test_setup_buildlog.py (fixed to import `saguaro`), tools/test_setup_vswhere.py, tools/vision/lean.cmake,
tools/vision_cache_e2e.py. tools/maya_changed.py adapted to our tree: saguaro-ik2's real CMake target name,
engines/qfn mapped to Qwen as its own `qfn_engine` build, no Maya-Qwen inside the GLM engine,
`saguaro.py`/`saguaro.sh`/`START-SAGUARO.bat` as the shared files; docs/RELEASE-CHECKS.md notes the full GLM-5.3
release checks are not ours. The nightly tools' files are `saguaro-nightly.*` (matches .gitignore).

## Upstream hunks dropped, with why

- **Full GLM-5.3 / glm_dsa family** (3f6349906 + 695dba4ac #1200 bind/load-check, the new
  `glm_dsa_bind_widths_test`, 695dba4ac's release line): our tree carries GLM-5.3-Flash only (v1.1.75 intake).
- **Maya Qwen / maya_names alias layer** (export_env / alias_env / env_in in server start): excluded by the plan; our
  settings are single-named (rename_env.py).
- **serve/attach.py `expand_request`** call: module not in our tree.
- **GLM same-answers references in new code** (`glmfast::repeatable` / `same_answers` / `lane_steps` calls in
  b6752deed's block): feature skipped at v1.1.75; the timing-per-PC and pinning parts (which don't need it) were
  kept.
- **DS4 `fast_prefill`/`DS4_YOCO_CONE` env** (in the auto-merged child_env): our ds4 build has no such kernel switch
  (we removed the config switch earlier); the deleted block stayed deleted.
- **Inkling first-start split measurement** (e2e006ffa `IK2_DRAFT_PART` runs + measure.log + `SAGUARO_IK2_DRAFT_SPLIT`):
  it measures upstream's offload tier-window layouts; we draft on the offload path through our dynamic cache and
  measure the sources in one start (kept from v1.1.75).  The merge first left the driver-less engine-side helpers
  (`g_part`, `draft_part_*`) behind with `IK2_DRAFT_FILE`/`IK2_DRAFT_PART` unrenamed; a follow-up commit removed
  the block, so the drop is now complete.
- **`MAYA_DS4_RING_FAST_LINK` / slow_bps rule** (d6bac1e5e's second half): our ring rule has no fast-link opt-out;
  only the 1.1 ms band value was taken.
- **README speed-table reformat + #1177/#1096/#1116 rows** (82bb64cad) and translated-README rows: our table kept, the
  three English rows added by hand; #1116's Inkling table taken as a NEW block in our format (main had none).
- **tools/test_maya_rebuild_all.py** change (b822a500d): the test file is deleted in our tree.

## Settings renamed (plan step 2, b2e85aa52)

Upstream's new names -> ours (code + docs + tests; `rename_env.py --check` and `rename_log_prefixes.py --check` pass):
`MAYA_VRAM_LATE_MB` -> `SAGUARO_VRAM_LATE_MB`, `MAYA_VRAM_RESERVE_SETUP_MB` -> `SAGUARO_VRAM_RESERVE_SETUP_MB`,
`MAYA_GLM_CPU_PIN` / `MAYA_GLM_CPU_CAL_REDO` / `MAYA_GLM_CPU_CAL` / `MAYA_GLM_KEEP_MAP` / `MAYA_GLM_POOL_CPUS` /
`MAYA_GLM_CPU_LANE_CHECK` / `MAYA_GLM_LANE_WHERE` / `MAYA_GLM_DSA_SCORE2` (mentions) -> `SAGUARO_GLM_*`,
`MAYA_IQ_MV_RW` -> `SAGUARO_IQ_MV_RW`, `MAYA_DS4_LATE_MB` -> `SAGUARO_DS4_LATE_MB`,
`MAYA_DS4_SLOT_DIR` / `MAYA_DS4_SLOT_KEEP` -> `SAGUARO_DS4_SLOT_*`, `MAYA_BENCH_LOCK_OK` -> `SAGUARO_BENCH_LOCK_OK`,
`MAYA_GPU_LOCKS` / `MAYA_HOT_FLAG` -> `SAGUARO_*` (in the new tests; the names themselves were already ours),
`MAYA_BUILD_OUTPUT` -> `SAGUARO_BUILD_OUTPUT`, `MAYA_TEST_ENCODER_TRACE` -> `SAGUARO_TEST_ENCODER_TRACE`,
`IK2_DEPTH` / `IK2_DRAFT_IDS` / `IK2_DRAFT_TIE` / `IK2_LANE` mentions -> `SAGUARO_IK2_*`. Build options stay upstream's
`MAYA_*` (`MAYA_VISION_FULL` / `MAYA_VISION_TOWERS` / `MAYA_VISION_LEAN` / `MAYA_XOR_BPERMUTE` /
`MAYA_HIPBLASLT_TUNING`...). `--bench`'s lock names and the picture temp dirs are `saguaro-*`; note the stale-dir
cleaner only matches `saguaro-*` prefixes, so pre-update `maya-vision-*` / `maya-ds4-pictures-*` leftovers age out via
the OS tmp policy instead.

Also restored (upstream carries it, our tree had lost it): the `-prerelease` vswhere filter in setup.py's
`find_vcvars` and tools/hip/build_maya_windows.bat (#1128's other half; test_setup_vswhere now passes).

## Build checks (plan step 3)

- `cmake -S . -B build-m1 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
  -DCMAKE_CUDA_ARCHITECTURES=75 -DMAYA_ENABLE_CUDA=ON -DMAYA_NATIVE_EXPERTS=ON` + `nice -n 10 ninja -j 12`: **rc=0**,
  all five engines built (saguaro-glm / -ds4 / -mm3 / -hy4 / -ik2; log ./scratch/build-m1.log). One break fixed
  (7a364bad7): the #1200 load check read `Layer::ff/ffd` + `tp_on()` - fields that came with the GLM tensor-parallel
  work we never took; dropped with a comment.
- `tools/hip_syntax_check.py --rocm /data/llm/toolchains/rocm-7.1-syntax --arch gfx1201
  --ggml build-m1/_deps/maya_llamacpp-src -j 8`: **80 of 80 HIP compiles clean** (log ./scratch/hip-check.log). The
  `--ggml` points at the fetched pinned checkout (the plan's strata-main path was not needed).

## Tests (plan step 4 - one module at a time, `python -m unittest <module>`)

- **tools**: 69 modules, 1069 tests. All pass except exactly the plan's known environment failures -
  `tools.test_maya_models` (3 errors: Exl3Download 'yes' attribute - identical to main),
  `tools.test_refresh_ds4_profile` (ImportError: it imports `refresh_ds4_profile` unqualified, though both
  `tools/refresh_ds4_profile.py` and the test are in the tree - unchanged by the merge, same as main),
  `tools.test_setup_ds4`
  (`test_exl3_on_amd` - same as main) - and `tools.test_metal_serve`, which runs none (as on main).
  `tools.test_iq_pack` needs `SAGUARO_GGUF_PY=build-m1/_deps/maya_llamacpp-src/gguf-py` (this worktree has no
  third_party/llama.cpp; with it set, 9/9 pass).
- **serve**: 46 modules, 999 tests - **all pass**, including the merge's new `test_picture_dirs`, `test_vision_cache`
  and the wrong-shape-400 tests.
- Fixed while getting there: `tools.test_maya_prefill` (assertion stale since 1.2.1-A - main failed it too, verified
  in a throwaway worktree), `tools.test_setup_buildlog` (imported upstream's `maya`), `tools.test_setup_vswhere`
  (the `-prerelease` restore), `serve.test_hy4_frontend` (un-skipped the think=false test - see below),
  `serve.test_server` (IK2_VISION_GIB gone from the engine, per #1185).
- **Real bug the upstream test caught and this merge fixed** (a2c8a8173): our model API's `"think": false` still
  mapped to Anthropic's `thinking` key after /api/chat moved to the OpenAI pipeline - thinking never turned off on
  /api/chat and /api/generate (upstream fixed this as 7645ddd53's twin; our intake had dropped the mapping and the
  test). serve/model_api.py now writes `chat_template_kwargs.enable_thinking`, the re-adopted
  `serve.test_model_api.Api/Pull.test_think_false_turns_thinking_off` and the
  `serve/test_hy4_frontend.py` test (previously skipped for exactly this) pass.

## Upstream changes that move answers or defaults (for the manager's card checks)

1. **DeepSeek turn end (#1206/#1207, de3d62027)**: a turn no longer ends empty after the thinking; a tool call written
   inside the thinking is read out of it; the end-of-thinking token is no one's text. Visible in streaming shape.
2. **DeepSeek router safety (#1114, 3c74b4b7c)**: a routing row with no finite score drops the bad pick (`keep & ~nf`)
   instead of reading outside its tables - affected rows now answer instead of crashing/garbling.
3. **DeepSeek shipped expert profile (#1161, e8c0b8277)**: a pack with no profile starts from `data/expert-profile-ds4.bin`
   (finds it beside the build too); the load says which file it read. Expert priority at first starts changes (upstream:
   +7% served).
4. **DeepSeek prompt ring (#1143, d6bac1e5e)**: the one-a-batch rule has a 1.1 ms band so a link near 1 ms stops
   flipping between starts (our tree: the band replaces the 1 ms line unconditionally; no RING_FAST_LINK switch here).
5. **VRAM reserve (#1026/#1156, 31abc0195/fa7e9078c/adb0097d4)**: the measured extra kept beside a reserve is capped at
   1 GiB, never refuses a model, and names its switch (`SAGUARO_VRAM_LATE_MB=0` off); on AMD a reserve the setup wrote
   by itself now counts as the default (marked `SAGUARO_VRAM_RESERVE_SETUP_MB` in the config env), so nothing extra is
   kept on top of it - a re-run of the setup changes the kept figures (and the start lines). Our Hy4 `--reserve-gb`
   rule kept; the mark applies on AMD only.
6. **GLM CPU lane answers (b6752deed/1a4f9dd48/d5f00a0f5)**: the saved lane timing belongs to the PC, not the build -
   an update keeps it and the answers; `--speed-calibration` / `SAGUARO_GLM_CPU_CAL_REDO=1` re-times and replaces the
   saved value only when >25% faster (=2 takes it). On two-socket Linux with 3+ GPUs the shared CPU lane is pinned
   over the NUMA nodes (packed vs spread timed once, saved with the timing) - fresh installs on such PCs may word
   answers differently from v1.1.75-era ones; updated installs keep theirs.
7. **GLM DFlash forced token (#1137, 9ed0c3d58)**: a token forced while the anchor goes out is row 0's - the first
   answer token after a forced end-of-thinking no longer continues the thinking.
8. **GLM/DS4 kernel swaps, same bits**: window tokens take `SAGUARO_GLM_DSA_SCORE2` (51a70bd29); MLA INT8 latents
   16 a load (0df10dde6); RDNA warp sums in registers, same bits (bdaeba1c1 - now also on our ds4/hy4 IQ paths via
   `maya_lane_xor`).
9. **Hy4 on AMD defaults (cf38358dc/b95319d5f/8e630cb83)**: on gfx1100 the dense weight read-ahead is ON without
   measuring and short attention-head rows are packed 2/4 a warp (`SAGUARO_IQ_MV_RW`, same sums); on any AMD build
   with the CPU lane, a token window's sums run on one thread unless `HY4_SUM_PER` is set. Hy4 has not run on our AMD
   hardware yet - treat as measured-upstream-only.
10. **Inkling pictures vs drafter (#1185 follow-through, e4864b217/2e503daca)**: the encoder takes no room in the
    start plan - the plan is identical with images on or off and the encoder runs on the GPU only while free VRAM
    holds it, else on CPU; `SAGUARO_IK2_VISION_GIB` is gone from engine and server. A 24 GB card at 262,144 keeps its
    drafter with pictures on (the fix's point). Our Ken-defaults table and AMD-offload-off stay as set.
11. **Server behavior**: stop during engine load ends the engine (#1181); picture temp dirs go with the server +
    stale ones cleaned at start (#1180); malformed `messages`/`input` gets a plain 400 on all three APIs + model API
    (77bfd6106); vision embeddings are copied to the request's own file before the next encode can evict them (#915);
    server start no longer waits on host-name lookups (0d4f9dc0b); a note at start when DS4_STREAM_WIN/DS4_LAND are
    set on one AMD GPU (#1120); calibrate_ds4_tp measures on a clean slot (#1203).
12. **Setup / bench**: the picture reader compiles lean (towers + f16-f16 mmq only) with automatic fall-back to the
    full build; each compile step prints one line, full output in build-engine.log / build-vision.log
    (`--full-build-output` or `SAGUARO_BUILD_OUTPUT=full`); build-time texts now say "a few minutes"; `--bench` starts
    the engine with the server's own environment (our `bench_start_command` kept - it already does that, QFN
    included), `--bench-smoke`, new lock backoff/`--bench-lock-ok`, `--bench --compare` primes one start per build and
    no longer sweeps `data/golden`; new tools `maya_changed/maya_golden/maya_nightly` + docs/RELEASE-CHECKS.md
    (adapted to our engine map); `--bench-tokens` / `--bench-text-ref`.
13. **VERSION/CHANGELOG**: ours (1.2.0) kept, as planned.

## Leftovers / open questions

- The anchor merge commit (b706c3ea4, `-s ours`) makes v1.1.75 an ancestor. If main prefers never to see it, the two
  commits can be replaced by a merge with `-X` machinery only at the cost of this exact 34-file conflict dance every
  time; recommend keeping.
- `tools/test_maya_rebuild_all.py` stays deleted: upstream's `--rebuild` compiles every engine (compile_engine's
  `also=`), our saguaro.py never took that. Revisit if we want it back.
- The Inkling first-start plain-vs-drafting layout timing (#1185's other half, e2e006ffa) is not adopted - our
  offload-layout choice is our dynamic cache, measured in one start. If Ken wants upstream's two-run behavior on our
  layouts, it needs the tier-window machinery, not just the measurement.
- docs/INKLING2.md and the defaults table still describe drafting-on-the-offload-path through our dynamic cache; the
  upstream text for mode-A windows / last-positions FP8 backoff was deliberately not adopted.
- The nightly release-check tools assume upstream's model set for their example config (docs/RELEASE-CHECKS.md
  mentions a GLM-5.3-full row - our adapted note says it is Flash-only); the example json may still list models we do
  not ship. No GPU runs of these tools happened here (plan's Don't).
- `serve/web/` and the pages: upstream touched none of them in this range (no page-visible changes to re-brand).
- Pre-existing, unchanged: `tools/test_setup_ds4.test_exl3_on_amd`, `test_maya_models` Exl3Download errors,
  missing `refresh_ds4_profile` module.
- Build dir ./build-m1 (1.2 GB) and logs ./scratch/build-m1.log, ./scratch/hip-check.log,
  ./scratch/test-results.txt, ./scratch/serve-results.txt are untracked scratch; the hip-check used the fetched
  checkout at build-m1/_deps/maya_llamacpp-src for --ggml.
