# REVIEW-RESULT — RN1: the upstream rename layer as one tested command
Review of `git diff e2cdfb25bea924a87d3b5b740edc90e04f21b23e..HEAD` against PLAN-rn1-rename-tool.md /
REPORT-rn1-rename-tool.md. Read-only review; only this file was written.

## Verdict: MERGE-WITH-NITS

No correctness bug found that blocks the merge. The behaviour claims verified; the nits are exception-bookkeeping
and latent-scope issues.

## How this review verified the claims (all run locally, read-only or on in-memory/scratch data)

- `tools.test_rename_upstream`: 25/25 OK. `tools.test_docs_new_engines`: 7/7 OK (the only test modules the plan
  names; grep confirms no test modules exist for the three older scripts, as the REPORT says).
- Rule table shared: every constant/regex the three old scripts used was compared, pattern-string by
  pattern-string, against the pre-change script from `git show e2cdfb25:...` — all equal; per-file transforms of
  the old internal `rename()` vs `T.rename_internal` compared over all 1751 text files (`git ls-files`) — 0 differ.
- Feature OFF: the diff touches only `tools/saguaro/*`, `tools/test_rename_upstream.py`, `docs/UPSTREAM.md` and
  the REPORT — no product code, no CMake, no serve file. Full `--check` runs of the three old scripts are
  byte-identical old-vs-new on this tree (rename_env rc 1/1, rename_log_prefixes 0/0, rename_internal_maya 1/1;
  the 1s predate RN1: kept qfn `-DSTRATA_*` texts, `docs/SETTINGS.md` STRATA_MMQ_TAB, a docs/triage file — and,
  for rename_env, only the coder's untracked `scratch/up178/` upstream copy; on tracked files the plan wants 0
  renames, so a clean checkout stays CHECK OK).
- Generated file: `serve/qfn_engine.py` is NOT in the diff, and the tool skips it and all of `engines/qfn/`
  (rename_rules.py:235-236), so regeneration via `tools/saguaro/extract_qfn_engine.py` stays the only path.
- Locks/threads/stdout: the new code is single-threaded CLI code — no threading, no locks, no concurrent writers
  anywhere in the diff, so there is nothing to interleave and no nested lock acquisition is possible.
- Paths/secrets: writes only to files already collected relative to the repo root; out-of-tree input exits
  (rename_upstream.py:60-62, :186-188); no absolute paths, no secrets in the diff; the tool was never run in
  write mode in these commits (commits are tools/docs only).
- Commits: step 2 → 742dcbe25 (table + command + three scripts), step 3 → 0f21c2848 (tests), step 4 → 460d2cfd1
  (docs + REPORT); per-commit file lists are exactly the step's files; no Co-Authored-By / Claude-Session /
  Assisted-by lines.

## The brief's three questions

1. **Is the rule table really shared?** Yes for the rules: all four tools import every pattern/word-set from
   `tools/saguaro/rename_rules.py`, and the old scripts were shown to behave identically to their pre-change
   versions. Not shared: the SKIP/exception bookkeeping — `rename_env.py:42-46` and `rename_internal_maya.py:31-32`
   still carry their own copies, which have already diverged from `rename_rules.py:233-241` (see NIT-1).
2. **Would a new MAYA_ setting be caught?** Yes. The pattern is `\b(MAYA|STRATA)_([A-Z0-9_]+)\b`
   (rename_rules.py:30) proved-by-context in `rename_env.plan` (rename_env.py:292-335); a new name needs no list
   entry. Verified directly: the test fixture's `std::getenv("MAYA_SOMETHING_NEW")` (tools/test_rename_upstream.py:46)
   is classified a strong use by `strong_use`. The settings pass runs after the internal pass
   (rename_upstream.py:145-158), so a new upstream `STRATA_FOO` (→ `MAYA_FOO` via `STRATA_X -> MAYA_X`) is caught
   too — `test_strata_prefix_merges_into_the_one_name` and `test_new_setting_caught_by_pattern` cover both.
3. **Are the exceptions justified, nothing over-broad (LICENSE, vendored)?** Largely yes. LICENSE*
   (rename_rules.py:240) and `third_party/`/vendored (rename_rules.py:233) are justified (the plan lists both) and
   harmless in practice — renaming a license text would itself be a bug. `EXC_TEXT` entries
   (rename_rules.py:221-230) are narrow per-file regexes and every hit is re-REPORTED on each run
   (rename_upstream.py:124-132, :217-220), so nothing rots silently. The broadest entries are the whole-directory
   skip of `engines/qfn/` (NIT-3) and the `saguaro.py` `STRATA_[A-Z0-9_]+` wildcard, which today matches exactly
   the three `-DSTRATA_*` arguments for qfn's CMake project (saguaro.py:7679-7694) — justified, and REPORT open
   question 1 schedules their removal.

## Findings by severity

### Minor

1. **The exception/skip lists are not shared, and have already drifted.** The unified tool skips
   `docs/UPSTREAM.md` and `tools/test_rename_upstream.py` (rename_rules.py:235-238), but `rename_internal_maya.py`
   — still on its own `SKIP_PREFIX` (rename_internal_maya.py:31-32) — now lists both RN1's own new files as
   strata residue in its `--check` output, and `rename_env.py`'s own copy (rename_env.py:42-46) leaves the test
   fixture names in its collisions report. Same duplication class the rule table was created to remove
   (the plan's "share the rule table (no duplicated rules)"). Behaviour is unchanged, but the lists will keep
   drifting.
2. **`--paths` applies no skip filter.** rename_upstream.py:180-199 walks `root.rglob("*")` without
   `T.skipped(rel)`, so a vendored or scratch file named `maya.py` / `maya.sh` / `START-MAYA.bat` / `MAYA_*.md`
   would be printed as a `git mv` suggestion (and `NAME_PATH_PREFIXES` at rename_rules.py:209 matches `MAYA_*`
   in any directory, not just docs/). Nothing matches today, so it is latent — but an upstream checkout is exactly
   where such vendored names can appear.
3. **The unified command is narrower than the union of the old scripts for `engines/qfn/`.** The whole directory
   is skipped (rename_rules.py:235-236), so a new `MAYA_FOO` setting in engines/qfn's sources is caught by
   `rename_env.py` alone (which only skips `engines/qfn/bench/results/`, rename_env.py:44) and a new `strata:`
   log prefix by `rename_log_prefixes.py` (whose DEFAULT_PATHS include engines/qfn, rename_log_prefixes.py:36-37)
   — but not by `rename_upstream.py`. This matches the report's stated exception policy, but the docstring's
   "ALL of them" (rename_upstream.py:9) deserves the qualifier, or qfn should get the settings/log passes.

### Nits

4. **The test module keeps `rename_env.py --check` green only by coincidence.** Each fixture's expected
   `SAGUARO_*` string appears in the same file, so every fixture setting collides with itself and is kept
   (collision rule, rename_env.py:320-321; fixtures at tools/test_rename_upstream.py:46-58). Rewording an
   assertion makes `rename_env.py --check` red without touching product code; skipping the module from the
   shared table (NIT-1) would make it inert by design.
5. **`exception_hits` reports line numbers from the post-plan text** (rename_upstream.py:203 passes `texts`,
   consumed at :124-132); when `--check` is red, reported lines can drift from the file on disk. Harmless in the
   green case, cosmetic when red.
6. **Untested options:** `--paths` and `--diff` have no tests (plan step 2 defines both); and
   `test_own_banner_outside_the_engine_sources_stays` (tools/test_rename_upstream.py:114-117) actually exercises
   the `LOG_SRC_PATHS` scope gate, not the prefix rules — the rules alone would rename that line.
7. Masked regions are restored by plain `str.replace` (rename_upstream.py:81-83, 149-150, 109-110): benign here
   (a placeholder can't match any rule and the NUL can't appear in a loaded file, rename_env.py:253-254), but it
   masks every textual copy of an exception string, including substrings no rule would have matched.

## Tests really exercise the change

`M1FromV1178` feeds `git show v1.1.78:<path>` through the full pipeline and asserts equality with the committed
text (whole files for two, the settings/log-prefix/name/brand lines the plan names, including the 02c95f94a
collision case made provable by adding `glm_fast.cu` to the run) — tools/test_rename_upstream.py:221-262.
Idempotence is asserted twice (fixture tree :213-218, upstream text :264-268), exceptions untouched
(:143-154), and the real-tree `--check` runs end-to-end via subprocess (:271-276). Every family has positive and
negative cases (:44-141). This matches the plan's step 3 requirements.

## Notes (not RN1's fault, worth tracking)

- REPORT claims "rename_env's --check output is byte-identical over this tree" — true, and verified, but on this
  working tree it is byte-identical *while red* (rc 1), entirely because of the untracked `scratch/up178/`
  upstream copy; a clean checkout of this commit is CHECK OK. A cleaner scratch layout (or the shared skip set)
  would avoid the noise.
