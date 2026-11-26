# P124f report - `--setup` on an existing install asks to download the model again

## Step 1 - how the setup decides whether to download today

`--setup --config <cfg>` reads that config as `prev` (saguaro.py:9929) and `set_up` (saguaro.py:5891)
dispatches on `choose_family` to the family's setup.  The model-folder / download decision per family:

- **GLM** - `choose_model` (saguaro.py:2260), step 5 `model_step` (3431).  With no `--gguf-dir` /
  `--model` / `--model-dir` it lists the downloadable quants and marks one "downloaded, in {d}" only
  when `local_shards(m, download_dir(models, q))` all exist (2297); `download_dir` (2106) searches
  `<models>/<quant>`, `<models>/glm-5.3-flash-<quant>` and the models folder itself (older HF names
  via `old_names`).  The default is the previous `installer.quant` (2293).  The `installer` block
  written at 4138-4142 records `gguf` / `gguf_dir` (a `--gguf-dir` setup) and `model_dir` (an EXL3
  folder) - but **neither is ever consulted**: a config set up from its own GGUF/EXL3 folder asks to
  download again.  `model_step`'s download branch then offers only the missing shards
  (`offer_download` 3340 computes `missing` with `file_incomplete`).  **This is the bug for GLM.**
- **Inkling (ik2)** - `set_up_ik2` (9413).  `ik2_dir` (9057) reuses the previous `installer.model_dir`
  when the family is ik2 and the same `ik2_quant`; `ik2_model_step` (9081) then checks the published
  manifest file names in that folder and offers only the missing ones.  A GGUF of your own (recorded
  as `installer.ik2_file` by `write_config_ik2`, 9364-9365) is recognized by `ik2_local_gguf` (9070)
  **only when `--model-dir` is given again**; a setup again without the flag looks for the published
  file name in the recorded folder, does not find it, and offers to download.  **This is the bug for
  Inkling.**
- **DeepSeek (DS4)** - `ds4_dir` (6703) reuses the parent of the previous config's `--pack` argument
  when `installer.ds4_quant` matches the quant and that folder holds `config.json` (EXL3) or any
  `*.gguf` (a GGUF); `ds4_model_step` (6751) checks the manifest's file names there and
  `offer_download` takes only the missing ones.  The folder is taken from the args only - the
  recorded `installer.model_dir` (6906-6907) is not used as a second source.
- **MiniMax (MM3)** - `mm3_dir` (7233) reuses the pack's parent when `installer.mm3_quant` matches and
  `installer.own` is not set and the folder holds `config.json` / a `*.gguf`.  A folder of your own
  (recorded `own: true` + `model_dir`, 7508-7510) is used only when `--model-dir` names it again
  (`mm3_own_dir` 7282, then `mm3_own_step` 7302 checks the folder by its own index); a setup again
  without the flag falls to the pinned download and `mm3_model_step` (7319) offers it.  **Same bug
  for a MiniMax folder of your own.**
- **Hy4-Flash** - `hy4_dir` (8735) reuses `installer.model_dir` when `installer.hy4_quant` matches;
  `hy4_model_step` (8756) first hard-links a file another Hy4 folder here holds (`hy4_places` 8745)
  and offers only the missing ones.  Already behaves as the plan wants.
- **QFN** - already the pattern to copy: `set_up_qfn` sets `a.qfn_prev_files_dir =
  installer.files_dir` (8424) and `qfn_files_step` (7814-7824) uses that folder when it holds every
  pinned file (checked by size with `file_incomplete`), else offers only the missing ones.

Common ground: the "offer only what is missing" half already exists everywhere
(`offer_download` / `file_incomplete`); what is missing is the "the config names a folder whose files
are all present - take it, do not download" half for GLM, Inkling and MM3-own (and DS4's folder is
read from the args only).
