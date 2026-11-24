# T2 report: QFN setup on NVIDIA + its own fit check (QFN-first task 2)

Branch `saguaro/q1-t2-setup-nvidia`, six commits (one per plan step) on top of `d2bdeabe6`:

| Step | Commit | What changed |
| --- | --- | --- |
| 1 | `593a53b77` | `saguaro.py compile_engine` (the CUDA build): with `target == QFN_TARGET` it now passes `-DSAGUARO_QFN=ON` and `-DSAGUARO_QFN_GGML_DIR=<llama>` exactly as `compile_engine_hip` does; CMakeLists.txt already had the `SAGUARO_QFN AND MAYA_ENABLE_CUDA` branch (ExternalProject `qfn_engine` -> `build/qfn/saguaro-qfn`, target `saguaro-qfn`). `refresh_engine` already maps `qfn_exe()` to that target, so update rebuilds take the same path. No engine sources or `serve/` touched; nothing was built here (the plan forbids engine builds). |
| 2 | `938e3b9e1` | New `tools/qfn_fit.py`, modelled on `tools/ik2_fit.py`: pure functions, no hardware access (the caller passes cards `{"name", "vram_gb", "arch"}`, RAM GiB and the quant) plus a small CLI (`tools/qfn_fit.py --ram 220 16:75 16:75 16:75 16:75`). Model facts it encodes: the quant sizes summed from `tools/qfn/download.json` (UD-Q4_K_XL 103.7 GiB ~ 111 GB, Q8_0 175.3 GiB), 12,672 K/V bytes a context token (`--kv int8`, measured) with only the newest `--kv-resident` cells per QSA layer in VRAM, 48 layers, the manifest's "needs" (a 16 GB card, ~96 GB of RAM). `plan()` answers fits-or-not with a reason, the recommended context (65536 one card, 262144 on a split), the RAM tier (`RAM - 20 GiB`) and the layout: one card, or the layer split - the live 4x Tesla T4 configs' own `layer_split` ("11,24,36" / "10,22,35") on four "16 GB" sm_75 cards, an even split of the 48 elsewhere, with `--trim-stage-weights --prefill-pipe <1536|6144|1>`. |
| 3 | `49e6a9c0c` | `saguaro.py`: `NO_ENGINE["cuda"]["qfn"]` lifted; `import qfn_fit as QF`; `check_pc` gives family `qfn` `QF.MIN_CC` (70, Volta on) and a refusal hint that names both vendors; `choose_gpus` refuses `--gpus` with more than four cards for qfn and asks through the new `choose_qfn_gpus` (one 20 GB+ card alone, else up to four equal ones - the live server's 4x T4 layout is the recommendation - else the largest single card); `set_up_qfn` keeps the AMD one-GPU + APU checks untouched and on NVIDIA lays out `QF.plan(...)` (refusing below its fit) with the layout's context recommendation; `write_config_qfn` adds the layout's args (`--trim-stage-weights --prefill-pipe`) and `layer_split` config key only on a split and merges the T4 tuning env (`SAGUARO_SM75_*`, `SAGUARO_KQL`, `SAGUARO_BF16_TC*`, `SAGUARO_MMQ_DIRECT/KTAIL/SWIGLU_Q8` - `SAGUARO_MMQ_KQ` stays the generic line it already was) **only when every chosen card is compute capability 7.5**. What the AMD path writes is unchanged (single card -> the layout adds nothing; its config test passes untouched). `qfn_about` now tells both vendors. |
| 4 | `366980320` | `family_unavailable`: Windows refuses QFN in one plain line ("its engine, saguaro-qfn, is not built on Windows here yet") for either backend; `check_pc`'s existing qfn gate says it before anything is built or downloaded, and the model question marks the row "not on this PC's GPU yet". |
| 5 | `868d5aad3` | `--check --model qwen3.8-flash-next[-q8_0]` gets its own branch (it used to fall through to GLM's fit lines): it prints what `qfn_fit` would write for these cards, or the reason it does not fit (AMD: still refused above one GPU, as the setup does). The start screen's "Loading" text is split so QFN no longer mentions `SAGUARO_EXPERT_RAM_GB` (which it ignores) and names `--resident-budget-gib` instead; CYBER-FROST's text is unchanged. `REPORT_LINES` learned QFN's engine lines: `PP ... new_read=...` (per-chunk read stats), `RDONE rid=` and `VRAM dev=` (the lines `engines/qfn/serve/server.py` reads); a plain `PP` progress line is deliberately not a report line. |
| 6 | `ec21c82ea` | New `tools/test_qfn_fit.py` (18 tests: quant sizes from the manifest, the KV/RAM-tier maths, one T4, the 4x T4 split for both quants incl. the T4 env gate, a 24 GB card, a 16 GB AMD card, the too-small cases - 8 GB card, 60 GB RAM, mixed-kind cards get no T4 env - the card choice and the CLI). `tools/test_setup_qfn.py` extended: the NVIDIA 4x T4 config (no `-hip` suffix, `gpu [0,1,2,3]`, `layer_split "11,24,36"`, 262144 context, `--resident-budget-gib 200`, `--trim-stage-weights --prefill-pipe 1536`, the SM75/KQL env, `engine_kind == "batch"`), a plain 24 GB card (no split, no T4 env), the Windows refusal, `choose_qfn_gpus` (4x T4 all four, a 24 GB card alone, `--gpus 0,1,2,3,4` refused) and the report lines; the old "NVIDIA: refused" assertion became "offered on cuda too". While writing the tests, `qfn_fit.layer_split` was tightened: the live T4 splits apply only when every card is also sm_75 (this rides in the step-6 commit). |

## Files

- changed: `saguaro.py`, `tools/test_setup_qfn.py`
- new: `tools/qfn_fit.py`, `tools/test_qfn_fit.py`
- untouched on purpose: `setup.py` (the plan's line numbers name saguaro.py's setup code: `compile_engine` is saguaro.py:2130), `engines/qfn/**` (read only), `serve/**`, the web UI, `tools/test_setup_amd.py` / `tools/test_setup_ik2.py`.

## Tests (repo root, one module at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest ...`)

| Module | Before | After |
| --- | --- | --- |
| `tools.test_qfn_fit` | (did not exist) | **Ran 18 - OK** |
| `tools.test_setup_qfn` | Ran 12 - OK | **Ran 17 - OK** |
| `tools.test_setup_amd` | Ran 29 - OK | **Ran 29 - OK** |
| `tools.test_setup_ik2` | Ran 54 - OK | **Ran 54 - OK** |

No engine was built (`--check`/setup GPU probes never ran; the tests mock the cards). One regression was caught and fixed mid-flight: an edit had clipped `choose_ik2_gpus`'s `return opts[pick]["gpus"]`, which `tools.test_setup_ik2.Picker.test_gpus` caught before the step-3 commit.

## Open questions

1. `docs/MODELS.md` still says "on NVIDIA it is set up by hand for now" (and `docs/AMD_SAGUARO.md`'s QFN table row). Doc updates were not in this plan's steps - want a follow-up?
2. The Models page (`serve/model_pull.qfn_fit`) still checks "the first AMD GPU" only, so on an NVIDIA PC the page does not say the model fits. The plan said not to touch the web UI - follow-up?
3. `--pool-workers 19` was left out of the written configs: it is the mangonel box's CPU count, not a card fact (`--pool-workers` defaults to every physical core). `--prefill-pipe`'s live values are that box's measured b/a too - they go only on the four-sm_75-T4-cards layout; an unmeasured split gets `1` (the engine's "even out the chunk counts"). Say the word if the T4 layout should also carry `--pool-workers` or `--prompt-cache 64`.
4. The recommended context is 262144 on a split (the largest `QFN_CONTEXTS` the setup offers; the live configs hand-run 1048576) and 65536 on one card. Confirm those are the numbers you want `--check` to print.
5. The general layer-split rule (48 layers shared evenly, no first-card deficit) only runs on cards the live configs never saw; four equal 16 GB sm_75 cards get the live values verbatim. A measurement on e.g. 2x T4 would let `qfn_fit` pin those too.
6. `QF.MIN_CC = 70`: saguaro-qfn's CMake path builds pre-Turing archs through its experimental SM60 build; the setup treats Volta as the floor, as the Qwen path and hy4 do. Pascal stays unofferable until someone measures it.
