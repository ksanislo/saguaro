# REPORT-p6-wording — P6: no "experimental" / "beta" / "not validated" labels in user-visible text

Rule followed: the whole project is experimental, so a page or a setup line says what was measured (the hardware, the
number) or that it is not measured yet - never a model, engine, quant or platform tagged "experimental" / "beta" /
"preview" / "not validated".

## What changed

Three commits on `saguaro/q1-p6-wording` (base `1862ecf7d`); the plan's "web" area needed no commit - see below.

- Step 1, the sweep (`scratch/raw_hits.txt`, verdicts in `scratch/hits.txt`): 147 raw matches over
  `serve/web/*.js serve/*.py saguaro.py setup.py README.md docs/*.md`. Almost all are keeps: `preview` is the
  dashboard's tool-call preview and its HTML Preview dialog (`chat-view.js`, `md.js`, `models.js`'s `MS.preview`,
  `ui.js`, `app.css`, `components.css`); `beta` is a local variable in `ds4_vision.py` / `mm3_vision.py` and test
  payload text; `?beta=true` in `serve/server.py:7516` and `serve/test_server.py` is what Claude Code sends;
  `experimental_speed_projection` (`serve/server.py`, `serve/qfn_engine.py`, `serve/test_server.py`) is a request
  key; `serve/qfn_engine.py` is generated; `--yarn-beta-fast` / `--yarn-beta-slow` and `SAGUARO_GLM_CONV_UNION_BETA`
  are flags/env names; the GDN's `alpha` / `beta` are math names; `setup.py` has no hits at all.
- `409534407` — step 2, the setup (saguaro.py + the one test that pinned a changed line):
  - module docstring (`--help`'s text): the CUDA / HIP / Metal scopes say what they are, `gfx1030` named as run on a
    Radeon PRO V620 (ROCm 7) and not on another RDNA 2 card yet;
  - `VALIDATED_CC` / `HIP_ARCHS_UNVALIDATED` / `HIP_ARCHS_WAVE64` comments say measured / not measured (the identifiers
    are unchanged);
  - `--gguf-dir` and the `Maya-M-Derisked` "about" line: the tag is gone, the description stays;
  - the HIP PC check: the "no supported AMD GPU" help line and the RDNA 2 warning carry the PRO V620 fact;
    `first_run_backend`'s docstring says "the not-measured-yet ones";
  - the Metal PC check line and `--backend`'s help: no tag;
  - Pascal: `can be used (Pascal: not measured yet)`, the refusal line, and the pre-Turing warning keep their facts
    (portable kernels, F32 prompt attention, L1 global-load caching, "not measured on such a card yet"), and
    `tools/test_setup_qfn.py:1420` follows the new card line;
  - `arch_setting`, the `--gguf-dir` / own-GGUF warnings in `model_step` ("not measured with Saguaro yet - only Maya-S,
    ... has been measured"), Inkling 2 Mini on AMD ("checked on an RX 9070 XT (gfx1201) only, no other AMD card yet"),
    and `--check`'s Mac / HIP closing lines ("to set Saguaro up on AMD");
  - DeepSeek's EXL3-on-AMD warning is gone: it existed only to say "experimental", and
    `tools/test_setup_ds4.py::Quant35::test_exl3_on_amd` has pinned *no* warning there since v1.1.53, when upstream
    `ad43571b1` dropped these marks (`git show HEAD:saguaro.py` + `scratch/check_head.py` show it fired at HEAD, so that
    test was already failing before this branch). The facts stay in docs/EXL3_HIP.md and docs/SETUP.md.
- `dddde0894` — the RDNA 2 lines say the measured thing (run on a Radeon PRO V620 with ROCm 7, not on another card of
  these yet); docs/AMD_SAGUARO.md and README.md already said so, so "not measured on such a card yet" would have been
  wrong.
- `54b1fe434` — step 2, the docs (README.md + 13 docs files): titles and headings lose the tag
  (`AMD_RDNA4.md`, `AMD_SAGUARO.md`, `EXL3-HELPER.md`, `EXL3_HIP.md`, `MI50_VALIDATION.md`, `SETTINGS.md`'s three
  DeepSeek sections, `AMD_SAGUARO.md`'s "Expert helper" - and `SETTINGS.md`'s link to it follows the new anchor), and
  table cells / sentences keep only facts: which card and which number measured, or "not measured yet"
  (Pascal in README, MI300X / MI325X speeds, the one-T4 quant's KL 0.012 against UD-Q4_K_XL, the EXL3 quants on AMD:
  kernels pass `exl3_test` on gfx1201 / gfx1151, a smoke test only, the GGUFs are the tested way; Metal: built and
  unit-tested on GitHub's Apple Silicon runners, the team has no Mac and has not run the real model; Windows: developed
  and measured on Linux, a user runs Maya-L on an RTX 4090). Inkling 2's "the preview" becomes "the setup's path" /
  "In the setup". docs/EXL3_HIP.md no longer promises a warning the setup no longer prints.
  Keeps, with reasons in `scratch/hits.txt`: `docs/EXL3_RDNA4.md:88,101` and `docs/HIP_PEER_PREFILL.md:46,48` tell
  fork / older-stack history (like CHANGELOG), not a tag on a Saguaro model or platform.

## Files

- `saguaro.py` (−4 net: the wording spots listed above; the DS4 EXL3 warning's 4 lines removed)
- `tools/test_setup_qfn.py` (1 line: the Pascal card line the setup prints)
- `README.md`, `docs/AMD_RDNA4.md`, `docs/AMD_SAGUARO.md`, `docs/DGX_SPARK_BUILD.md`, `docs/EXL3-HELPER.md`,
  `docs/EXL3_HIP.md`, `docs/INKLING2.md`, `docs/INKLING2-MINI.md`, `docs/METAL.md`, `docs/MI50_VALIDATION.md`,
  `docs/MODELS.md`, `docs/SETTINGS.md`, `docs/SETUP.md`, `docs/WINDOWS.md` (89 changed lines)
- `scratch/hits.txt`, `scratch/raw_hits.txt`, `scratch/check_head.py`, `scratch/saguaro_head.py` (not committed)
- no `serve/` file changed: the web UI's only matches are its own "preview" feature names, so the plan's web area
  produced no commit. `setup.py` has no hits. `serve/qfn_engine.py` was not touched, so no re-extract was needed.

## Tests run (only the modules the plan names, one at a time, with /opt/llm/saguaro/.venv/bin/python)

- `tools.test_setup_qfn` — after the edits: **89 tests, OK (skipped=1)** (1 skip pre-existing).
- `tools.test_setup_ds4` — **82 tests, OK**; at HEAD this module failed `Quant35::test_exl3_on_amd` (the removed
  warning), so the change fixed it rather than the other way round.
- `tools.test_setup_amd` **34 OK**, `tools.test_setup_mm3` **31 OK**, `tools.test_setup_ik2` **54 OK**,
  `tools.test_setup_hy4` **33 OK** after the last commit; `tools.test_setup_dash` 20 (20 skipped),
  `tools.test_setup_buildlog` 6, `tools.test_setup_download` 2, `tools.test_setup_tui` 23 (12 skipped),
  `tools.test_setup_vswhere` 10 OK before it (saguaro.py's last commit only reworded three RDNA 2 strings).
- `tools.test_docs_new_engines` — **7 tests, OK** (before the doc pass as a baseline, and with the docs final).
- `serve.test_dashboard` — **62 OK**; `serve.test_new_engines_ui` — **35 OK**; `serve.test_model_api` — **28 OK**.
- `python -m py_compile saguaro.py` clean after each edit round.
- Not run (not in the plan's run list, and no string they pin was touched): `tools.test_hip_archs`,
  `tools.test_maya_hip`, `tools.test_maya_models`, `tools.test_maya_pascal`. Checked by hand instead: the three
  `test_maya_models.py` assertions on an "(experimental)" download label are `assertNotIn`, so removing tags can only
  help; `test_maya_pascal.py` and `test_maya_hip.py:129` assert arch numbers, not these words; `tools.test_hip_archs`
  pins CMake's own wording, which is untouched (see below).

## Open questions

- `cmake/hip_backend.cmake` still answers an unlisted arch with "not validated" (that is what
  `tools/test_hip_archs.py:33,36` pins). It is user-visible in a build, but the plan's grep list stops at
  serve/web, serve, saguaro.py, setup.py, README.md and docs/*.md - change it there, and its test, in a later step?
- `tools/ik2_fit.py:338` says "an NVIDIA Tesla P40 / GTX 10 (Pascal, experimental) or newer GPU" - that string is
  printed by the setup, but `tools/*.py` is outside the plan's grep list, so it stands. Same for
  `tools/qfn_fit.py:35`'s comment naming engines/qfn's pre-Turing path.
- DeepSeek's EXL3-on-AMD warning was removed rather than rewritten, because the test pinned "no warning on AMD at all".
  If the setup should still say there that the kernels pass their tests and the whole model has had a smoke test only,
  the line can come back without the tag and `test_exl3_on_amd` would have to be re-taken.
- `docs/MODELS.md`'s AMD-pictures line now adds "no AMD run of one is measured here yet", the fact
  docs/AMD_SAGUARO.md states; saguaro.py:1684's comment ("every model the setup takes on AMD reads pictures there")
  was left as a code comment. Worth a look if that comment should match.
- CHANGELOG.md keeps its old wording (the plan excludes history), so the words still appear there.
- `serve/model_pull.py` and `tools/setup_tui.py` are outside the plan's grep list; grepped as a check they carry no
  such labels (`serve/models.py`'s one match is its command-preview docstring). `tools/ik2_fit.py:338`'s Pascal tag is
  the one user-visible string outside the plan's list that still carries a tag.
