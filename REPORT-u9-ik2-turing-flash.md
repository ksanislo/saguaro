# U9 report: Inkling on Turing - upstream PR #1233 (flash prompt attention for sm_75, GPU 1 runs the missing experts, peer copies)

Branch `saguaro/q1-u9-ik2-turing-flash`, four commits:

1. `30f1eede0` ik2: upstream PR #1233 on our tree (src/ik2/ik2_kernels.cu, src/ik2/ik2_main.cu, tests/ik2_kernels_test.cu)
2. `8ab92b6bf` ik2: check tool + peer-copy probe (tools/ik2_check.py, tests/ik2_p2p_stream.cu)
3. `7072f1fc2` docs/INKLING2.md: the switches section
4. step 4 needed no source change (the build passed as committed); this report is its commit.

## How the patch went on

`git apply --3way ref/pr1233.patch` (the upstream base blobs are in our object store, so the 3-way was real):
`tools/ik2_check.py` and `tests/ik2_p2p_stream.cu` came in as new files directly ("falling back to direct
application"); the four modified files came in **with conflicts**: ik2_kernels.cu 7, ik2_main.cu 11,
ik2_kernels_test.cu 1, docs/INKLING2.md 1. All were resolved by hand and are listed below.

### The structural difference that drove most resolutions

Our tree runs the fused prompt attention on sm_75 **by default** (commit `089b554d9`: the `IK2_SM75` compile-time
layout - one K/V tile loaded where it is used, `kFaSmem75`, and FP8 tiles made FP16 as they load; main's
`if (cls < 75) g_flash = false;`). Upstream v1.1.78 still had flash at sm_80+ and PR #1233 adds Turing layouts as
opt-in (`TU`/`TV` template parameters, `if (cls < 80) g_flash = cls == 75 && getenv("IK2_FLASH_TURING")...`).
Per the plan the default had to stay exactly ours, so:

- **ik2_kernels.cu** (7 conflicts): our `kAhead = !IK2_SM75` machinery stayed and the PR's parameters folded around
  it: `kAhead = !IK2_SM75 && !TU && !TV`, `kBN/kBld` from the patch, `Vs = TV ? fa_sm : ...`, `Bw` offset
  `TV ? 1 : kAhead ? 4 : 2`, `kTileLd` keyed on `!kAhead`, the in-place load now `if constexpr (!kAhead && !TV)`
  (TV loads K at the loop top with the patch's `fa_load_half`, V between Q K^T and the softmax), the prefetch now
  `else if (kAhead && t + 1 < t1)`. `fa_load_half`, the `kFaBNv/kFaSmemTv/kFaSmemTu` constants, the `float s[]`
  hoist, the V-over-K block and the template parameters merged as upstream wrote them. For the default
  instantiation (TU=TV=false) every expression reduces to HEAD's, and the launch keeps our per-device smem choice
  (`major < 8 ? kFaSmem75 : kFaSmem`, extended with `TV ? kFaSmemTv : TU ? kFaSmemTu : ...`; the `attr`/`smem_of`
  caches are per template instantiation, so the layouts cannot collide). `need_class(75)` stayed (upstream's
  `TU ? 75 : 80` would kill our default on Turing). In `attn_flash_f16` the PR's `arch_class() < 80` branches got a
  gate of their own that upstream did not need: `SAGUARO_IK2_FLASH_TURING=1` must be set to select TV (default) or
  TU; unset, sm_75 keeps our kernel. `SAGUARO_IK2_FLASH_TV=0` picks the 64-key TU layout - which on an sm_75 build
  *is* our default layout (their `kFaSmemTu` equals our `kFaSmem75`); the genuinely new one is TV's 32-key tile.
- **ik2_main.cu** (11 conflicts):
  - `g_flash_tu` and the call-site `!(g_flash_tu && g_kv_fp8)` were **not taken**: our sm_75 FP8 prompt attention
    works (FP8 tiles made FP16 at load) and is the default; upstream's FP8-cuBLAS fallback there would have changed
    the default. Documented in the docs and the commit message.
  - arch block: kept our `cls < 75 g_flash=false`, our `SAGUARO_IK2_GRP`/`g_block`/`g_attn_mma`/`g_draft_force` and
    our Pascal/Volta-aware drafting-off NOTEs; took the PR's `g_pa70` expression (adds sm_75 via
    `SAGUARO_IK2_PA70_TURING`, `g_pa70_min = 16384` on sm_75 - inert unless the switch is set) and PA70_MIN.
  - `Prof`: took the new fields (`attn_swa`, `p0[3]`, `p0s[4]`, `p1w`, `ex[6]`, `ph[2][5]`) and our `start()`;
    our `lap(acc, tag)` gained a default tag `"lap"` because the PR's new points call it without one (our
    SAGUARO_IK2_EVENTS timeline records `"lap"` for them).
  - `run_prefill_moe`: the PR's `p1skip`/`pf1_pick`/`pf1_run` insertions took, with our `grp` grouped path and
    `mix`/`up`/`dn` packed-expert handling kept inside `run_exp` and the PR's `g_pf.ex[*]` laps woven around them;
    `pf1_add` after `exec(-1, ...)` took (guarded by `p1`, which requires `g_off && !g_ep && !g_tp`).
  - the `run_layer_attn`/`forward` lap points took; where ours and theirs both touched (tp's `g_tp_part` early
    return, the sliding/full split of the attn lap) both were kept.
  - drafter: `DraftRemote`, `DevScope`, `drafter_copy_taps/write_context/set_f_rnn/propose` and all call-site
    renames took; the `IK2_DRAFT_GPU` setup block took **minus** the upstream-only `off_dr/off_keep/g_draft_kv8`
    context (v1.1.78 items our v1.1.75 tree does not have) and our own `for (kv8...)` line; the big serve-mode
    prompt loop kept our structure (`big_end`, the beat/cancel lines) with the `drafter_*` renames; one
    `write_context(..., 0, n, p0)` stayed at our `p0` (upstream's line there differs from ours in that argument;
    ours is the default behaviour).
  - the SAGUARO_IK2_PROF per-request phase report took as upstream wrote it.
- **tests/ik2_kernels_test.cu** (1 conflict): upstream's comment/message updates for the sm_75 flash test took;
  our Pascal `tc` gating of `test_attn_decode_mma` stayed (upstream calls it unconditionally).
- **docs/INKLING2.md** (1 conflict): did **not** take the PR's block verbatim; wrote our own short section (step 3).

All new code reads `getenv("SAGUARO_IK2_*")`: our ik2's convention (checked: the whole engine reads
`SAGUARO_IK2_*`; only internal compile-time names like `IK2_SM75` keep the bare prefix). Engine messages from the
patch say `NOTE saguaro ik2:` like our other ones.

## Env names (upstream -> ours)

| PR #1233 | ours | notes |
| --- | --- | --- |
| IK2_FLASH_TURING | SAGUARO_IK2_FLASH_TURING | picks the PR's layouts on sm_75; **our flash stays the default when unset** |
| IK2_FLASH_TV | SAGUARO_IK2_FLASH_TV | =0: the 64-key layout (our default's form); default: the 32-key one |
| IK2_PF_GPU1 | SAGUARO_IK2_PF_GPU1 | two-card split, offload path, not with --ep/--tp |
| IK2_PF_PEER | SAGUARO_IK2_PF_PEER | source-stream copies, checked at load both ways |
| IK2_PF_HOST | SAGUARO_IK2_PF_HOST | force host staging |
| IK2_DRAFT_GPU | SAGUARO_IK2_DRAFT_GPU | drafter on GPU 1 of the split, Q8 head |
| IK2_PA70_TURING | SAGUARO_IK2_PA70_TURING | Volta's prompt kernel on sm_75 past 16384 (SAGUARO_IK2_PA70_MIN) |
| IK2_PROF | SAGUARO_IK2_PROF | name existed here (g_prof); the PR adds the phase report |
| IK2_HOST_STAGE / IK2_PEER / IK2_KV / IK2_EP / IK2_TP / IK2_TWO_GRAPH | SAGUARO_IK2_HOST_STAGE / ... | pre-existing names used by the new code |

## Step 2 adaptations

- `tools/ik2_check.py`: starts `serve/server.py -B --config ... --port ... --host 127.0.0.1` (not `maya.py
  --plain`), default base config `configs/Inkling-2-Mini-UD-Q4_K_XL.json`, all settings/KV under SAGUARO_IK2_*,
  chat uses the config's `model_name`; the /v1/models and `timings` shapes match our server. `list` verified.
- `tests/ik2_p2p_stream.cu`: header comments renamed to `saguaro-ik2` / `SAGUARO_IK2_PF_PEER`; `src/mm3/mm3_link.cuh`
  referenced by the header exists in our tree. Compiles with one nvcc line for sm_75.

## Build / check results (step 4, no GPU used)

- `cmake -S . -B build-u9 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75
  -DMAYA_ENABLE_CUDA=ON`: configured ("Saguaro: saguaro-ik2 for 75").
- `cmake --build build-u9 --target saguaro-ik2 ik2_kernels_test -j 16`: **built both**, plus the `ik2_kernels`
  library (ik2_kernels.cu, ik2_off.cu, ik2_pascal.cu, ik2_main.cu, ik2_draft.cu, the lane/ds4 C++ all compile).
  3 nvcc warnings, all pre-existing on HEAD in code the patch does not touch (`kFaRawQ4`, `Shards::num`,
  `g_apu` unused on a CUDA/sm_75 build). No errors. `ik2_kernels_test` builds without a GPU; it was **not run**
  (it needs one).
- `SAGUARO_HIP_SYNTAX_ROCM=... tools/hip_syntax_check.py --rocm /data/llm/toolchains/rocm-7.1-syntax --arch
  gfx1201 --ggml /data/llm/strata-main/third_party/llama.cpp -j 8`: **80 of 80 HIP compiles clean**, the sweep
  includes src/ik2/*.cu, so the merged kernel and main files stay HIP-syntax clean.
- `/opt/llm/saguaro/.venv/bin/python tools/ik2_check.py list`: runs, prints the six rows with SAGUARO_IK2_* names.
- `/usr/local/cuda-13.1/bin/nvcc -O2 -arch=sm_75 -c tests/ik2_p2p_stream.cu`: compiles clean.

## Can SAGUARO_IK2_PF_GPU1 / PF_PEER work under our --tp 2? (from reading the code)

No - they are inert there by construction, which matches the PR not having tested this mode:

- `pf1_pick` returns false unless `g_off && !g_ep && !g_tp` **and** `g_two.on`; `pf1_run`/`pf1_peer` work entirely
  through the two-split's state (`g_two.own`, `g_two.tab` - GPU 1's expert-block table, `g_two.link.all_peer/say`).
  `g_two.want` is set only for `n_gpu == 2 && g_tpn == 1` with neither `SAGUARO_IK2_EP` nor `SAGUARO_IK2_TP`, so
  under `--tp N` (g_tpn > 1) `g_two.on` is false and PF_GPU1 never runs; under `SAGUARO_IK2_TP=1` the `!g_tp`
  guard turns it off too.
- Deeper than the guard: `--tp 2` splits every expert's FFN width across the cards and **refuses to start when a
  member's expert share does not fit in VRAM** - there is no offload tier on GPU 0 whose misses PF_GPU1 could
  offload to GPU 1, the model this feature assumes. And its token sums are already peer copies between every pair
  of members checked at load by its own link test, so PF_PEER's staging decision has nothing to govern.
- `SAGUARO_IK2_DRAFT_GPU` is likewise split-only (`g_two.want && dg == g_two.dev && !dual()`).
- The features therefore apply to our `--gpus 2` two-card split (the offload path, no --ep/--tp), as upstream's
  configuration had it. On a pair like the PR's (2080 Ti + NVLink), our engine's own load-time link test - like
  `mm3_link.cuh`'s - issues its test copies on the destination's stream, which is the direction their measurement
  found wrong; `SAGUARO_IK2_PF_PEER=1` re-checks with source-stream copies both ways before relying on them,
  exactly as upstream wrote.

## Open questions

- Nothing was measured or answer-checked on hardware here (no GPU use was allowed): the TV/TU layouts' answers vs
  our default, PF_GPU1's same-answers claim and the p2p probe's exit codes all still want a run of
  `tools/ik2_check.py run one one-tuned two two-nvlink --what answers,long` + `compare` on a Turing pair, and
  `tools/ik2/t4_check.sh`'s kernel test (both flash layouts, run twice with `SAGUARO_IK2_FLASH_TV=0`).
- Our T4s are 16 GB: the PR's UD-Q4_K_XL is 17.6 GB, so their single-card rows cannot be reproduced on one T4 as
  is; their numbers were cited as theirs in docs/INKLING2.md.
- Should our sm_75 default eventually *become* the 32-key TV layout (their faster one on their cards)? Kept
  opt-in for now per the plan; a measured answer needs the runs above.
- In a multi-arch fat binary the TV/TU launch sets the smem attribute per instantiation per device (fine), but the
  combination has never executed; only compile-checked here.
- `--tp 2` + PF_GPU1 stays refused by design; if someone wants GPU 1's idle VRAM during tp2 prompt reads, the
  pf1 model would have to be rethought around tp2's half-expert layout (not attempted).
