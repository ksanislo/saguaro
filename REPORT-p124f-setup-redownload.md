# P124f report - `--setup` on an existing install asks to download the model again

Saguaro 1.2.4 branch `saguaro/q1-p124f-setup-redownload`, off 88809c4ea (1.2.3).  Line numbers in the
step 1 section are as of 88809c4ea; the step 2 section says what moved.  Nothing in serve/ or engines/
was touched (serve/qfn_engine.py untouched, extract_qfn_engine.py not needed); nothing ran on a GPU.

## Step 1 - how the setup decided whether to download, before the fix

`--setup --config <cfg>` reads that config as `prev` (saguaro.py:9929) and `set_up` (5891) dispatches on
`choose_family` to the family's setup.  The model-folder / download decision per family:

- **GLM** - `choose_model` (2260), step 5 `model_step` (3431).  With no `--gguf-dir` / `--model` /
  `--model-dir` it lists the downloadable quants and marks one "downloaded, in {d}" only when
  `local_shards(m, download_dir(models, q))` all exist (2297); `download_dir` (2106) searches
  `<models>/<quant>`, `<models>/glm-5.3-flash-<quant>` and the models folder itself (older HF names via
  `old_names`).  The default is the previous `installer.quant` (2293).  The `installer` block written at
  4138-4142 records `gguf` / `gguf_dir` (a `--gguf-dir` setup) and `model_dir` (an EXL3 folder) - but
  neither was consulted: a config set up from its own GGUF/EXL3 folder asked to download again.
  `model_step`'s download branch offers only the missing shards (`offer_download` 3340 computes `missing`
  with `file_incomplete`).  This was the bug for GLM.
- **Inkling (ik2)** - `set_up_ik2` (9413).  `ik2_dir` (9057) reused the previous `installer.model_dir`
  when the family is ik2 and the same `ik2_quant`; `ik2_model_step` (9081) checked the published
  manifest file names in that folder and offered only the missing ones.  A GGUF of your own (recorded as
  `installer.ik2_file` by `write_config_ik2`, 9364-9365) was recognized by `ik2_local_gguf` (9070) only
  when `--model-dir` was given again; a setup again without the flag looked for the published file name
  in the recorded folder, did not find it, and offered to download.  This was the bug for Inkling (the
  9070 XT box).
- **DeepSeek (DS4)** - `ds4_dir` (6703) reused the parent of the previous config's `--pack` argument when
  `installer.ds4_quant` matched and that folder held `config.json` (EXL3) or any `*.gguf`;
  `ds4_model_step` (6751) checked the manifest's file names there and `offer_download` took only the
  missing ones.  The recorded `installer.model_dir` (6906-6907) was not a second source for that folder.
- **MiniMax (MM3)** - `mm3_dir` (7233) reused the pack's parent when `installer.mm3_quant` matched and
  `installer.own` was not set and the folder held `config.json` / a `*.gguf`.  A folder of your own
  (recorded `own: true` + `model_dir`, 7508-7510) was used only when `--model-dir` named it again
  (`mm3_own_dir` 7282, `mm3_own_step` 7302 checking it by its own index); a setup again without the flag
  fell to the pinned download and `mm3_model_step` (7319) offered it.  Same bug for a MiniMax folder of
  your own.
- **Hy4-Flash** - `hy4_dir` (8735) reused `installer.model_dir` when `installer.hy4_quant` matched;
  `hy4_model_step` (8756) hard-linked a file another Hy4 folder here holds (`hy4_places` 8745) and offered
  only the missing ones.  Already behaved as the plan wants.
- **QFN** - already the pattern the plan asks for: `set_up_qfn` set `a.qfn_prev_files_dir =
  installer.files_dir` (8424) and `qfn_files_step` (7814-7824) used that folder when it held every pinned
  file (sizes via `file_incomplete`), else offered only the missing ones.

The "offer only what is missing" half already existed everywhere (`offer_download` / `file_incomplete`);
what was missing was "the config names a folder whose files are all present - take it, do not download"
for GLM, Inkling and a MiniMax folder of your own, and DeepSeek read its folder from the args only.

## Step 2 - the fix (saguaro.py only; committed 9ae0c31c8)

The same way for every family: when the config being set up names a model folder (its installer record,
or its args) and the files are all present at the sizes/headers the setup expects, that folder is the
answer - no download, no question; what is missing is still named and offered, file by file, by the
existing `offer_download` machinery.

- **GLM, `choose_model`** (saguaro.py:2281-2292): after the `--gguf-dir` / `--model` options, the
  installer's `gguf` and `gguf_dir` are tried with `local_first` + `incomplete(shard_set(first))`; all
  whole -> `("local", first)`, which `model_step` uses in place (its own arch/whole-shard checks).  The
  installer's `model_dir` (an EXL3 folder given with `--model-dir` then) -> `("exl3", folder)` when its
  `config.json`, `tokenizer.json` and `tokenizer_config.json` are there; `exl3_folder_files` names
  anything else that is out.  A download quant keeps the old flow: the row's "downloaded, in {d}" check,
  and `offer_download` naming only the missing shards.
- **Inkling** - `set_up_ik2` now hands the recorded `installer.ik2_file` to the steps as
  `a.ik2_prev_file` (the QFN pattern, saguaro.py:9449); `ik2_local_gguf` (9096-9108) takes
  `<recorded folder>/<that name>` when the published GGUF and no Strata pack are there - so a setup
  again of a config that ran its own GGUF needs no `--model-dir` named again.  `ik2_dir` already
  returned the recorded `model_dir`; a missing published file is still `offer_download`ed (only it).
- **MiniMax** - `mm3_dir` (7263-7265) returns the recorded own folder (`installer.own` + `model_dir`,
  same `mm3_quant`) without `--model-dir`; `set_up_mm3` (7566-7569) pairs that with `own =
  Path(installer.model_dir)`, so step 5 is `mm3_own_step` (the folder whole by its own index; it says
  what is missing) and the pinned EXL3/GGUF download is not offered for it.  A MiniMax config from
  before the `model_dir` record keeps the old behavior.
- **DeepSeek** - `ds4_dir` (6716-6729): the folder of the setup before is now sought in two places, the
  args' `--pack` parent and the recorded `installer.model_dir`, same quant and the files present
  (`config.json` for EXL3, any `*.gguf` for a GGUF); `ds4_model_step` then says and downloads only what
  is missing.
- **Hy4 and QFN - no change**: both already took the recorded folder when its files were present
  (step 1), so they needed nothing.

## Step 3 - tests (committed b018540eb)

The plan named `tools/test_setup_glm*.py`, which does not exist; GLM's setup tests live in
`tools/test_maya_models.py` / `tools/test_maya_own_quants.py`, and those are where the GLM tests went.

- `tools/test_maya_models.py` - new class `SetupAgainOfAnExistingModel`:
  - `test_the_recorded_gguf_dir_is_taken_and_nothing_is_offered`: a GLM config whose installer records
    `gguf`/`gguf_dir` (its whole 3-shard Maya-M in its own folder) -> `choose_model` returns
    `("local", first)` and `model_step` takes the shards in place; `offer_download` is patched to raise
    - it is never reached.
  - `test_a_missing_shard_is_the_only_one_offered`: the config's Maya-M download in the models folder is
    missing shard 3 -> the setup again keeps Maya-M as the default and the offer says "1 still to
    download", with exactly one curl command, for exactly that shard (offer_download run for real, ask
    answered "n").
- `tools/test_setup_ik2.py` - in `Steps`:
  - `test_a_setup_again_takes_the_recorded_folder_and_its_own_file`: installer records model_dir +
    ik2_file -> `ik2_dir` returns the folder, `ik2_local_gguf` (with `a.ik2_prev_file`) the own GGUF, and
    `ik2_model_step` returns it - `offer_download` patched to raise, never reached.
  - `test_a_setup_again_offers_only_the_file_that_is_missing`: the recorded folder's published
    UD-Q4_K_XL gone -> the offer lists exactly that file.
- `tools/test_maya_own_quants.py`:
  - `GlmExl3Folder.test_a_later_setup_takes_the_recorded_folder_of_your_own`: the recorded EXL3 folder is
    `choose_model`'s answer without `--model-dir`.
  - `MiniMaxExl3Folder.test_a_later_setup_takes_the_recorded_folder_of_your_own` (renamed from
    `test_a_later_setup_does_not_find_the_pinned_files_in_the_folder_of_your_own`, which encoded the
    pre-P124f behavior): with the `model_dir` record the own folder is returned; without it (a config
    from before the record) the pinned default stays; a pinned setup before is as it was.
- `tools/test_setup_ds4.py` - `test_own_folder_and_config` extended: the `--pack` parent and the recorded
  `installer.model_dir` are both taken for the same quant; another quant still gets its own folder.

Test runs (one module at a time, `/opt/llm/saguaro/.venv/bin/python tools/<module>`):

| module | result |
| --- | --- |
| tools/test_maya_models.py | 57 tests OK (55 before, +2 new) |
| tools/test_setup_ik2.py | 56 tests OK (54 before, +2 new) |
| tools/test_maya_own_quants.py | 35 tests OK, 1 skipped (34 before, +1 new, 1 renamed) |
| tools/test_setup_ds4.py | 82 tests OK (new asserts in an existing test) |
| tools/test_setup_mm3.py | 31 tests OK (code touched, tests not changed) |
| tools/test_maya_quants.py | 26 tests OK, 1 skipped (choose_model callers) |
| tools/test_maya_metal.py | 26 tests OK (choose_model callers) |
| tools/test_setup_amd.py | 35 tests OK (choose_model callers) |

## Files changed

- `saguaro.py` - `choose_model`, `ds4_dir`, `mm3_dir`, `set_up_mm3`, `ik2_local_gguf`, `set_up_ik2`.
- `tools/test_maya_models.py`, `tools/test_setup_ik2.py`, `tools/test_maya_own_quants.py`,
  `tools/test_setup_ds4.py` - the tests above.
- `REPORT-p124f-setup-redownload.md` - this file (it is listed in `.git/info/exclude`; committed with
  `git add -f` as the plan asks).
- No CHANGELOG entry: in this repo the CHANGELOG moves at a release commit, not at fixes.

## Open questions

1. **A half-copied folder of your own**: if the recorded `--gguf-dir` folder exists but a shard is
   incomplete, the setup falls through to the download list and (for a listed quant) re-fetches the whole
   quant into the models folder rather than completing that folder in place - the same shape as QFN's
   existing "does not hold all of ...'s files ... or leave it out to download them".  Offering the
   missing shards into the custom folder (works when its files carry the HF names) was left undone;
   the 9070 XT report is about a folder whose files are all present.
2. **Inkling with `--yes`**: `choose_ik2_quant` takes the recommended quant, not the config's, when they
   differ - then the recorded folder (a different file set) is not consulted and the recommended one is
   offered for download.  Pre-existing behavior, untouched.
3. The MM3 test renamed above encoded the old behavior on purpose (its title); the rename keeps a test
   for the old path (a config from before the `model_dir` record) inside the new one.
