# REPORT-rn1-rename-tool: the upstream rename layer as one tested command (PLAN-rn1-rename-tool.md, docs/UPSTREAM.md section 2)

## What changed

- **tools/saguaro/rename_rules.py (new)** - the rule table in one place, read by all four tools: the settings
  family (rename_env's prove-then-rename names, prefixes, allowlist), the internal family (rename_internal_maya's
  strata->maya table, CODE_WORDS/KEEP_LOWER, its sub-functions and per-file `rename_internal()`), the engine log
  prefixes (rename_log_prefixes' RULES/LINE_RULES), plus the two families the M1 intake had to do by hand:
  NAMES_RULES/LOGTAG_RULES/CMD_RULES (maya.py, ./maya.sh, START-MAYA.bat, maya-nightly*, the saguaro-* temp-dir
  prefixes, `[maya]` -> `[saguaro]`, `maya --speed-calibration` -> `saguaro.py ...`) and BRAND_RULES ("restart
  Maya" / "Maya checks ..." -> Saguaro, .md only). The explicit exception list (EXC_PATHS/EXC_TEXT + the skip
  set) keeps: credits and upstream URLs, LICENSE texts, vendored dirs, history files, engines/qfn's own names
  (its `-DSTRATA_*` options, their docs and tests), the internal CMake targets (maya, maya-ds4, maya-mm3,
  maya-hy4, maya-vision, maya_kernels, maya_mmq, MAYA-BUILD.json, `-DMAYA_*` build options), the maya-*.json
  config file names, `.maya-*.json`, the `maya.` browser keys, the serve `--engine maya` / `"maya"` API fields,
  ds4/hy4/ik2/mm3 internal names, the Maya quant names (Maya-S/M/L/XS/S24) and "Project Maya v1.1.x".
- **tools/saguaro/rename_upstream.py (new)** - the one command: `--check` (exit 1 while a renameable use is
  left), `--diff`, `--paths` (old<TAB>new lines for `git mv`: maya.py, maya.sh, START-MAYA.bat, MAYA_* docs,
  include/strata/), a tree or file list, default the whole worktree. Passes: internal -> settings (rename_env's
  own decision pass over the internal-renamed text, so a NEW MAYA_FOO is caught by pattern) -> names/brand ->
  log prefixes. Idempotent (no replacement is ever a match; tested twice over). Exception hits are REPORTED
  without failing `--check`.
- **tools/saguaro/rename_env.py** - no rule change; the decision pass split into `plan(files)` + `apply(text, oc)`
  (shared with the new tool) and its table constants now come from rename_rules.py. `--check` output is
  byte-identical to before over this tree (verified against the pre-change script from HEAD).
- **tools/saguaro/rename_internal_maya.py / rename_log_prefixes.py** - no rule change; their regexes/word lists
  moved to the shared table and are imported back. Verified equal to the HEAD versions over all 1747 text files
  (internal) and on real + synthetic samples (log prefixes); `--check` outputs identical.
- **tools/test_rename_upstream.py (new)** - the tests (25, below).
- **docs/UPSTREAM.md section 2** - rewritten: the command, what it renames, what stays, and the intake's use
  (throwaway branch: --paths -> write -> --check -> merge the renamed copy -> --check again; a missed name is
  a rule-table bug, fixed there and in the tests, never by hand). The tool skips this file (it spells the old
  names to describe them, like its own tests and tools/saguaro/).
- No product code changed; the tool was never run in write mode over the real tree (only `--check`).

## The M1 hand renames and where they live now

scratch/notes.md lists every rule and every M1 follow-up commit (b2e85aa52, 02c95f94a, ef27c2650, fe22cd559,
a4a45bd71, 2d4246a26). The mechanical ones became rules (settings pattern, names, logtag, cmd, brand); the
non-mechanical ones stay human and are named as such in the notes: INKLING2.md's drafter-room rewrite (7a230a65d),
RELEASE-CHECKS' GLM-row / SETTINGS / CHANGELOG text about parts our tree drops (b2e85aa52, e2cdfb25b, 567b49ed2),
and maya_changed.py's engine map (saguaro-ik2 / qfn_engine keys - data about our build, not a pattern).

## Tests run

- `python -m unittest tools.test_rename_upstream` - **OK, 25 tests**: every rule positive + negative (settings
  by pattern incl. the STRATA_ merge, kept words/build options/allowlist; internal incl. the pack-format keeps;
  log prefixes incl. non-prefixes and the tests'-own-banner scope; every name rule incl. maya_kernels/maya_mmq,
  maya-*.json, `maya.` keys, serve-maya.sh, engine_kind=ds4 staying; brand incl. Project/version/quant keeps);
  idempotence over a 7-file fixture tree and over v1.1.78's text; the M1 cases from `git show v1.1.78:<path>` to
  our committed text (vram_late.hpp and test_vision_cache.py whole-file; the `[saguaro]` server line, the
  `saguaro ds4:` expert-profile line, `SAGUARO_DS4_PROMPT_SKIP`, the `SAGUARO_GLM_DSA_SCORE2` comment - the
  02c95f94a collision case, provable once glm_fast.cu is in the run - `./saguaro.sh`/`saguaro-nightly.json`,
  both SETUP.md brand lines and the `saguaro.py --speed-calibration` comment); exceptions untouched; and
  `rename_upstream.py --check` over the real tree exits 0 (CHECK OK).
- `python -m unittest tools.test_docs_new_engines` - **OK, 7 tests** (docs untouched by the tool's scope).
- The three older scripts: no test modules exist for them (grep); their behavior was verified directly as
  described above (`--check` identical, per-file transforms equal to HEAD's).
- `rename_upstream.py --check` over the tree: exit 0; over v1.1.78's RELEASE-CHECKS/SETUP/server/ds4/glm files:
  renames exactly the M1 lines (asserted in the tests).

## What --check found on this tree (kept, reported - product text the RN1 plan's "don't" protects)

tools/test_maya_hip.py's `./maya.sh` comment; src/hy4/hy4_main.cpp's two `maya --speed-calibration` comments
(a4a45bd71 fixed only glm_fast_path.cu); src/hy4/hy4_model.cu's `maya.py's` comment; saguaro.py's three
"start Maya again" user strings (two tests assert the wording); src/prefill/moe_fused_rdna4.cu's "Maya keeps
..." credit-style comments and tools/hy4/download.json's `maya.py's hy4_need_gib` note (outside the names/brand
scopes: .cu brand is docs-only, .json is data). These are rename-layer leftovers from before M1, one small
product pass away; the tool lists them on every run so they cannot get lost.

## Open questions

1. Should the next product pass rewrite the leftover comments/strings above (and the two asserting tests)?
   The tool would then drop those EXC_TEXT entries.
2. rename_internal_maya.py --check exits 1 on main by design (it targets strata-era branches; the kept hits are
   engines/qfn's options and SETTINGS' STRATA_MMQ_TAB). rename_upstream --check stays green because the
   exception list names them - if those qfn names ever go away, delete the EXC_TEXT entries.
3. rename_env's collision rule (SAGUARO_X exists -> OLD_X mention left) still needs a human for merged upstream
   text (the glm_window case: provable by the tool once the reading file is in the same run; blocked when only
   the mentioning file is passed). A "collision but the suffix is proved in the same run -> rename" mode could
   replace the 02c95f94a-style hand fix; it changes rename_env's documented behavior, so it is not in this pass.
4. tools/maya_*.py file names, `import saguaro as maya` aliases and the serve "maya" API fields stay (decided
   before RN1); the tool matches that line, but a future rebrand of the API fields would need the exceptions
   updated deliberately.

Commits: step 2 (rule table + command + three scripts onto it), step 3 (tests), step 4 (docs/UPSTREAM.md +
this report). notes.md lives in ./scratch/ (git-ignored scratch, as the plan's rules put it).
