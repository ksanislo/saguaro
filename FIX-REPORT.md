# Fix report: REVIEW-RESULT.md findings (b535d1fec..a998c7b32 intake)

All nine findings addressed; one commit each. Nothing else in the tree was touched.

## Must fix

1. **tools/vision/CMakeLists.txt parse error** - FIXED (`f9d0f08a6`).
   `endif()target_link_libraries(maya-vision PRIVATE mtmd llama)` separated into two lines and the two joined
   comment lines at :81 split apart, as upstream v1.1.78 has them (our "vision-glm's own" wording, which main
   already had, kept). Verified: `cmake -S tools/vision -B scratch/vision-cfg -DLLAMA_DIR=scratch/vision-stub`
   now reads the whole file - CMake parses every command before executing, and it gets all the way to
   lean.cmake's missing `mtmd` target (this worktree has no third_party/llama.cpp checkout, so the configure
   cannot finish for that unrelated reason); before the fix it stopped with the parse error at the file's own
   read time.

2. **src/ik2/ik2_main.cu half-merged DRAFT_PART block** - FIXED the recommended way, by deleting the dead
   block (`7b7a84783`). Removed: the comment block + `g_part`, `g_part_out`, `part_dup_stdout`,
   `draft_part_file`, `draft_part_done`, `draft_part_read`, `env_set`, `draft_part_run` (old 6885-6991;
   `env_set` had no other user); the parent's `SAGUARO_IK2_DRAFT_PART` block (old 7234-7244); and the
   `if (g_part) draft_part_done(...)` line in the measure path (old 9821 - the one caller the review's line
   list did not count, consistent with the block going, since nothing else can set `g_part` any more).
   The `<process.h>` and `<sys/wait.h>` includes the merge added for this block went with it. No
   `g_part`/`draft_part*`/`IK2_DRAFT_PART`/`IK2_DRAFT_FILE` remains anywhere; `SAGUARO_IK2_DRAFT_FILE` /
   `_MEASURE` (main's own draft-measurement feature, read at 6834/6852) untouched. The engine now matches
   the report's "dropped" claim; the report was amended to say so plainly (finding 9c).
   Verified: `nice ninja -C build-m1 saguaro-ik2` -> clean build and link (the `g_apu` warning is pre-existing
   on main). Chose deletion over renaming because the upstream driver (the `draft_part_run`/`draft_part_read`
   two-run split) was really dropped, and the review recommends matching the docs.

## Nits

3. **`[maya]` log prefixes in serve/server.py** - FIXED (`ef27c2650`). Both spots (the #1206 salvage print at
   5921, `ds4_one_gpu_switch_note`'s text at 8053) now say `[saguaro]`. `grep "[maya]"` over all .py files is
   empty again; `rename_log_prefixes.py --check` rc=0; `serve.test_server` passes (174 tests, no test asserts
   the old text).

4. **CHANGELOG.md advertised switch** - FIXED (`567b49ed2`). The `SAGUARO_DS4_RING_FAST_LINK=0` clause is
   dropped from the v1.1.76-78 DeepSeek entry; the #1143 reference and the band description stay. No other
   file mentions the switch (the three REPORT mentions are accurate - they say it was not taken).

5. **docs/SETUP.md vswhere FAQ** - FIXED (`2d4246a26`). "Maya checks the Installer's ... restart Maya" ->
   "Saguaro ... restart Saguaro". The other `Maya` occurrences in the file are model tier names (Maya-S etc.)
   already on main and were left. `tools.test_setup_vswhere`, `tools.test_changed` and `tools.test_golden`
   (the modules that look at docs) pass.

6. **src/core/glm_fast_path.cu stale program name** - FIXED (`a4a45bd71`): the comment now says
   `saguaro.py --speed-calibration`, as SETTINGS.md and the same block's docs spell it. The two identical
   stale comments in `src/hy4/hy4_main.cpp` (lines 14, 1486) were NOT touched: they are main's own pre-merge
   lines, outside this intake's scope, and the review flagged only the merge's new line.

7. **serve/test_vision_cache.py temp dir prefix** - FIXED (`fe22cd559`):
   `tempfile.TemporaryDirectory(prefix="saguaro-vision-http-")`, matching the rebranded sibling test.
   `serve.test_vision_cache` passes.

8. **dead `kRingSlowS` in src/ds4/ds4_prompt_mem.hpp** - FIXED (`c2bbda8a5`). The constant and its line
   removed; the comment sentence that named it now names `kRingSlowBandS`, which is what the merged rule
   actually uses. No test or source referenced the constant (upstream keeps it only for the `slow_bps`
   off-path we never took). Verified: `build-m1/ds4_prompt_mem_test` -> "ds4 prompt / memory rules: ok", and
   `ninja -C build-m1 saguaro-ds4` rebuilds ds4_model.cu clean.

9. **REPORT-m1-upstream-1178.md inaccuracies** - FIXED in the report (`e9b09d859`):
   (a) both places claiming the Inkling #1116 table "was already ours"/"already present" corrected -
       verified main's README has only the GLM/DeepSeek/MiniMax tables and the merge added the whole
       "Inkling 2 Mini (experimental preview) speed benchmarks" block;
   (b) the `tools.test_refresh_ds4_profile` line no longer says "module not in our tree" - re-ran the module:
       `ModuleNotFoundError: No module named 'refresh_ds4_profile'` from its own unqualified import, both
       `tools/refresh_ds4_profile.py` and the test present, exactly as on main;
   (c) the DRAFT_PART drop bullet now states that the driver-less helpers were left behind by the merge and
       removed in a follow-up commit (true as of finding 2's commit).
   NOT changed, on purpose: `tools/maya_bench.py:628`'s "the build's own stamp" comment - the review itself
   only noted it, and it is byte-identical to upstream's stale comment; diverging from upstream text for a
   word the review did not ask to fix is worse than keeping the faithful merge.

## Final state

- `git status` clean; nine commits on top of a998c7b32, one per finding, only the touched files staged each time.
- `rename_env.py --check` and `rename_log_prefixes.py --check`: rc=0.
- `tools/saguaro/extract_qfn_engine.py` re-run: serve/qfn_engine.py regenerated byte-identical (git stayed
  clean); neither it nor engines/ is in this branch's fix diff.
- Test modules re-run, one at a time (only the plan-named ones the changes can touch): serve.test_server,
  serve.test_vision_cache, tools.test_changed, tools.test_golden, tools.test_setup_vswhere,
  tools.test_refresh_ds4_profile (still the known main-side ImportError, unchanged) - plus the C++
  ds4_prompt_mem_test and the saguaro-ik2 / saguaro-ds4 target builds from build-m1.
