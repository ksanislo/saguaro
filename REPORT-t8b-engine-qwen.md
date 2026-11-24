# REPORT-t8b-engine-qwen - T8b: the Maya Qwen path is out of the saguaro-glm engine (step 5)

Written after the T8b review (REVIEW-RESULT.md): the review's FIX-FIRST findings are all addressed in
this report's build evidence, and the commits below are one per plan step, fixes included.

## Commits (plan step -> commit)

| commit | step |
| --- | --- |
| `37de16bb5` | 2 - the qwen4exp driver out of `src/program/generate.cpp`, refusal line in its place |
| `730e785d0` | 3 - the DROP files, kernels, parities, hip tests and their CMake entries |
| `4aefd90ab` | 3 (review fix 1) - `kv_q8_parity` follows the removed `qsa.cu` after all |
| `5e2a18a51` | 3 (review nit 5) - the dead `argmax` helper is gone too |
| `09b833e2b` | 4 (review fix 3) - `tools/hip_syntax_check.py` merged from the amd lineage so the check can run |
| step 1 | inventory in `scratch/t8b-inventory.txt` (gitignored), corrected where the build proved it wrong |

## What was removed (files, lines)

`git diff e1f43e79a..HEAD` = **69 files changed, 129 insertions(+), 26582 deletions(-)**; the 129 are the
5-line refusal + comment in `generate.cpp`, 4 unavoidable CMake list-continuation lines, and the 120-line
merged `tools/hip_syntax_check.py`. Everything else is pure deletion. 67 whole files go:

- **generate.cpp**: -4936 lines - the qwen4exp session/graph machinery, serve loop, verifier, drafter,
  PLE/QSA/KV-streaming driver and its private helpers (refill_blocking, vision_lend_need, model_shards,
  drive pools, GpuStage, mem_mark, fnv1a, checkpoint_*, load_control_vectors, probe_ple/pcie,
  cuda_context_broken, and later the dead `argmax`). What remains where the driver was
  (src/program/generate.cpp:5747-5751): no `--glm-pack` -> one line on stderr,
  `saguaro generate: this model runs on saguaro-qfn`, exit 2.
- **core**: session, mtp, verify, layer, native_head, native_dense (.cpp + .hpp).
- **kernels**: ple, qsa, qsa_select, qsa_{decode,prompt}_attn, gr, gdn, fused_gr, fused_gdn,
  native_{gr_norm,gr_postops,gdn,gdn_preprocess,ple_postops,qsa,qsa_indexer,qsa_score}, ngram
  (.cu + .hpp), the ple/qsa oracle vectors, and the parities that belong to them (cvec, gdn, gr,
  kv_hybrid, kv_stream, ple, qsa, qsa_prompt_attn, qsa_select_bench) - plus kv_q8_parity (review fix 1:
  its `kv_append_step`/`kv_gather_step` calls lived only in the removed `cuda/qsa.cu`, so per the plan's
  step-3 rule "*_parity only DROP when the kernel is DROP" it goes with qsa.cu).
- **ngram/ple_reader** (+ its test), **prefill/prefill.cpp** (+ prefill.hpp), the CMake entries for all of it.
- **tests/hip**: native_qsa_score, ple_iq4, prefill_native_batch.

## What was kept and why

- `expert_source` / `remote_experts` - kept consumers: `include/maya/core/glm_fast_state.hpp`,
  `tests/hip/glm_expert_helper.cpp`, `tests/core/file_expert_source_test.cpp`.
- `strata-plan` - `src/core/device_main.cpp` includes `maya/plan/plan.hpp`.
- `mrope` (`rope.cu`, `native_rope.cu`) - `src/prefill/kernels.cu` still calls it; `router_sigmoid` -
  `src/core/glm_model.cu`. Shared prefill MMQ (`gemm.cu`, `kernels.cu`) stays.
- `qsa.hpp` / `kv_stream.hpp` headers - the kept prefill `kernels.hpp` includes them.
- `mtp_rows.hpp` - `generate.cpp`'s GLM path uses `maya::core::MtpTake` from it (not the removed `mtp.hpp`);
  `plan.hpp`'s `n_gdn_layers()`/`n_qsa_layers()` are model-metadata accessors that stayed.
- `kv_q4_parity` - its kernels live in the kept `src/kernels/cuda/kv_q4.cu`, so unlike kv_q8_parity it
  links against nothing removed.
- `tools/iq_pack.py` and `LICENSE.qwen-community-1.0` - the plan says they stay.

The KEEP/DROP calls are now build-proven: the full CUDA build links every remaining target (below), and
grep finds no `#include` or symbol reference to any removed file (the review's finding 4 re-check agrees;
its one miss, kv_q8_parity, is fixed in `4aefd90ab`).

## Build results (plan step 4)

**CUDA** - same `./build-t8b` directory the plan configured (setup.py compile_engine flags:
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc`, arch 75, `MAYA_ENABLE_CUDA=ON`,
`MAYA_BUILD_TESTS=ON`, Release), `ninja -C build-t8b -j 16`. The pre-review run reached 453/481 and
stopped at the `kv_q8_parity` link; after `4aefd90ab` the re-run finished the whole graph (34 steps:
CMake re-run, `generate.cpp.o` after the argmax fix, and the executables that had never reached the link
step). A later `ninja -n` lists nothing but the three ds4 build-id edges below, i.e. every target is built:

```
[0/1] Re-running CMake...
-- Saguaro: CUDA enabled, arch 75
...
[31/34] Linking CXX executable saguaro-ds4
[32/34] Building CXX object CMakeFiles/maya.dir/src/program/generate.cpp.o
[33/34] Linking CXX executable saguaro-glm
```

Two further full passes exit 0 with zero `FAILED`/`error:` lines (scratch/build-t8b-2.log, build-t8b-3.log);
the only edges ninja still plans are the three `maya_build_id` -> `ds4_main.cpp` -> `saguaro-ds4` edges that
re-run on every invocation by design (cmake/build_id.cmake is a run-every-build custom command; the header
it guards is not rewritten when the commit is unchanged). The `-Wunused-function` warning for `argmax` is
gone; the 3 remaining warnings are the pre-existing `-Wmissing-field-initializers` ones in the untouched
GLM path. No GPU, no engine binary was run.

**HIP gfx1201** - `tools/hip_syntax_check.py` was never on this branch (the review's finding 3: absent from
every history of it; it lives on the amd-ds4mm3 lineage), so plan step 4 could not run. It is now merged
verbatim from `a66e2560e` (`09b833e2b`), `--help` read first, and run for gfx1201 (its default arch) with
the toolchain the plan names (as a relative path from the repo: `../toolchains/rocm-7.1-syntax` = the plan's
`/data/llm/toolchains/rocm-7.1-syntax`) and `--ggml ../toolchains/llama.cpp-3cf03257` - the pinned
`LLAMA_CPP_COMMIT` checkout, since `third_party/llama.cpp` (the script's default) is not fetched on this
machine. Result (scratch/hip-syntax-t8b.log, exit 0):

```
80 of 80 HIP compiles clean for gfx1201
```

## Python / test names that still refer to removed things (for T8a - not edited here)

- GGUF arch name `qwen4exp`: `tools/iq_pack.py`, `tools/maya_quants.py`, `tools/test_iq_pack.py`,
  `tools/test_iq_pack_glm.py`, `tools/mtp_pack.py`, `tools/mtp_from_gguf.py`, `tools/gguf_writer.py`,
  `tools/test_gguf_fix_arch.py`, `saguaro.py`.
- CLI flags whose only consumer was the qwen driver but that the parser still accepts (kept deliberately:
  the plan's hard rule keeps the argument parser and `usage()` byte-identical; flagged for a follow-up
  prune by the review): `--control-vector`/`--cvec-*`, `--no-fused-gdn`; `serve/server.py:1688` and
  `serve/qfn_engine.py:1310` mention `--control-vector-scaled` in comments (`serve/qfn_engine.py` belongs
  to the QFN engine and stays untouched).
- No `.py`/`.sh` under tools/, tests/, serve/ or bench/ references any removed CMake target or hip test
  by name (grep: `qsa_select`, `ple_reader`, `kv_q8_parity`, the removed `*_parity`s,
  `native_qsa_score`, `ple_iq4`, `prefill_native_batch` - zero hits).

## Noted, not done

- No automated test covers the refusal line (the plan asked for none; reading the code: the `--glm-pack`
  branch at src/program/generate.cpp:5715-5745 returns first, so src/program/generate.cpp:5750-5751 is
  reached only without `--glm-pack`).
- `bench/micro/` holds nothing qwen-related on this branch (the inventory's bench lines describe the
  other lineage's tree; `git ls-tree e1f43e79a bench/micro` has no such files), so there was nothing to drop there.
