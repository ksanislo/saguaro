# FIX-REPORT: T8b review findings (REVIEW-RESULT.md) - what each one got

Branch `saguaro/q1-t8b-engine-qwen`, on top of the reviewed HEAD `730e785d0`. One commit per finding, in
review order. No engine binary or GPU was run; `engines/` untouched; `serve/qfn_engine.py` not touched
(nothing here changes the QFN engine, so `tools/saguaro/extract_qfn_engine.py` had nothing to extract).
Builds went into the existing `./build-t8b`; logs kept in `./scratch/`.

## 1. HIGH - the step-4 CUDA build fails: `kv_q8_parity` links symbols of the removed `qsa.cu` -> FIXED (`4aefd90ab`)

The review is right and the inventory was wrong: `src/kernels/kv_q8_parity.cpp:66,111` call
`k::kv_append_step`/`k::kv_gather_step`, defined only in the removed `src/kernels/cuda/qsa.cu:586,632`
(kv_q4_parity's `*_q4_step` live in the kept `kv_q4.cu`, which is why only kv_q8 broke). Per the plan's own
step-3 rule a `*_parity` goes when its kernel goes, so the test is dropped like its FP16 siblings: deleted
`src/kernels/kv_q8_parity.cpp` and its four `CMakeLists.txt` lines (the comment line they sat under stayed
attached to the next block). The inventory's claim in `scratch/t8b-inventory.txt` is corrected there.
Nothing else references the name (grep over src/ include/ tests/ bench/ tools/ cmake/ *.py *.sh: no hits -
the test was CMake-only, so no T8a leftover).

Re-ran the full build: `ninja -C build-t8b -j 16` re-ran CMake, compiled `generate.cpp.o` and linked every
executable the stopped build had never reached; further passes exit 0 with zero `FAILED`/`error:` lines and
`ninja -n` lists nothing but the always-re-run `maya_build_id` -> `saguaro-ds4` edges (run-every-build custom
command, pre-existing, header not rewritten when the commit is unchanged). Last lines in
scratch/build-t8b-2.log / build-t8b-3.log, pasted into `REPORT-t8b-engine-qwen.md` (finding 2).
The `-Wunused-function` warning the review quoted (build-t8b.log:373) is gone; the build's remaining three
warnings are the pre-existing `-Wmissing-field-initializers` ones in the untouched GLM path.

## 2. HIGH - `REPORT-t8b-engine-qwen.md` (step 5) does not exist -> FIXED (`6564835e0`)

Written: what was removed (files, line counts), what was kept and why (the grep map the review re-verified,
now backed by the passing build), both build results with the last lines pasted, the T8a leftover names, and
the two "noted, not done" items. It needed `git add -f` because this repo's `.git/info/exclude` carries a
generic `REPORT-*.md` line - the same line the earlier `REPORT-t*.md` commits sit under, so the tracked-file
outcome matches them.

## 3. HIGH - the HIP gfx1201 check never ran: `tools/hip_syntax_check.py` is not on this branch -> FIXED (`09b833e2b`)

The review offers "amend the plan or merge the script"; a deletion that touched CMake source lists and
`tests/hip/*` deserves the compiler pass, so the script is merged verbatim from the lineage commit that
carries it (`a66e2560e`) rather than the plan being bent. `--help` read first, then run for gfx1201 with the
toolchain the plan names (relative `../toolchains/rocm-7.1-syntax`) and `--ggml ../toolchains/llama.cpp-3cf03257`
- the pinned `LLAMA_CPP_COMMIT`, since the script's default `third_party/llama.cpp` is not fetched here.
Result (scratch/hip-syntax-t8b.log): `80 of 80 HIP compiles clean for gfx1201`, exit 0. That the deletions
broke nothing on the HIP side is now compiler-proven, not just grep-proven.

## 4. MEDIUM - KEEP/DROP calls not build-proven -> CLOSED by fix 1

The failing link was the only symbol the review's grep audit missed, and it is now dropped; the full CUDA
build plus the 80/80 HIP pass are the proof the finding asked for. No KEEP was found wrong in the other
direction (nothing missing or unreferenced).

## 5. LOW/nit - dead remnants of the removed driver

- `argmax` -> FIXED (`5e2a18a51`). Its only caller went with the driver, so it is not "another engine uses
  it"; the deletion (8 lines with its blanks) removes the `-Wunused-function` warning. (The `--greedy` help text and a comment
  mention the word, not the symbol.)
- `--control-vector`/`--cvec-*` and `--no-fused-gdn` parsing -> NOT CHANGED, deliberately. The plan's hard
  rule is byte-identical GLM behaviour with deletions only, and the review itself scopes this as a follow-up
  prune; `serve/server.py:1688` still mentions `--control-vector-scaled` and is T8a's file (Python is out of
  T8b's scope), so pruning the parser now would split one contract across two branches. Left to the follow-up.

## 6. LOW/nit - no automated test for the refusal line -> NOTED, NOT DONE

The plan asked for no test, and any test here would have to spawn the engine binary, which T8b is told never
to run. `REPORT-t8b-engine-qwen.md` records the exact reachability (the `--glm-pack` branch at
src/program/generate.cpp:5715-5745 returns first; the refusal is src/program/generate.cpp:5750-5751) so a
later task can test it where running the binary is allowed.

## After the fixes

`git log --oneline e1f43e79a..HEAD` is seven commits once this file is in (two reviewed + five here),
worktree clean, no commit
carries Co-Authored-By / Claude-Session / Assisted-by. Full CUDA build: passes (the whole target graph; the stopped build's 481-step graph minus
kv_q8_parity). HIP gfx1201
syntax check: 80 of 80 clean. The only unverified thing left is the GPU runtime, which this task must not touch.
