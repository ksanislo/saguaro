# Staying based on upstream: how Saguaro takes Project Maya and Strata changes

Goal: Saguaro stays upstream Project Maya + our changes on top, with our own names (SAGUARO_*, saguaro-*, Saguaro
branding). Every upstream release is taken in full. Intakes must stay routine: a few hours, not a rewrite.

## 1. Branches
- `upstream/main` + upstream tags: a read-only mirror (git remote `origin` = mw00/project-maya).
- `saguaro/qfn-vram` = our main (GitHub ksanislo/saguaro main). Releases are tags on it (v1.2.0, ...).
- Our work: one topic branch per change, merged into main (fork-branch-layout rule). The integration branch for a
  release (today saguaro/qfn-first) collects them before main.

## 2. The rename layer (the main source of pointless conflicts)
- Upstream code arrives with upstream names; ours are SAGUARO_* settings, saguaro-* programs, Saguaro branding.
- ONE command applies the layer: `python3 tools/saguaro/rename_upstream.py [--diff] [--paths] [paths...]`
  (tools/test_rename_upstream.py tests it; the table is tools/saguaro/rename_rules.py). It renames, in one
  idempotent pass: the internal names (include/strata, `strata::`, strata_* cmake words, STRATA_X options ->
  maya / MAYA_X, upstream's own v1.1.75 naming), the settings (`MAYA_X`/`STRATA_X` -> `SAGUARO_X` by pattern,
  so a new upstream setting is caught without being listed), the engines' log prefixes ("strata part:",
  "maya-ds4:", "ds4:" -> "saguaro ..."), the program, launcher and artifact names in text (maya.py ->
  saguaro.py, ./maya.sh -> ./saguaro.sh, START-MAYA.bat, maya-nightly*, the saguaro-* temp-dir prefixes,
  `[maya]` -> `[saguaro]`) and the brand words in user-visible docs ("restart Maya", "Maya checks ..." ->
  Saguaro). The explicit exceptions stay: credits and upstream URLs, LICENSE texts, vendored dirs, history
  files, engines/qfn's and the CMake targets' own names (maya, maya-ds4, maya-vision, maya_kernels, maya_mmq,
  -DMAYA_* options), the maya-*.json config names, `maya.` browser keys, ds4/hy4 internal names and the
  Maya quant names (Maya-S/M/L/XS/S24). The three earlier scripts (rename_env.py, rename_internal_maya.py,
  rename_log_prefixes.py) keep working; they read the same table.
- An intake uses it on a throwaway branch: check the release tag out -> `rename_upstream.py --paths` for the
  `git mv`s (maya.py, maya.sh, START-MAYA.bat, include/strata/) -> `rename_upstream.py` rewrites the tree ->
  `rename_upstream.py --check` says CHECK OK (it also reports what the exception list keeps, and --diff
  reviews everything before a write). That renamed copy is what gets merged, so the merge shows only REAL
  conflicts (resolved against docs/DIVERGENCE.md). After the merge, `--check` again on the integrated tree:
  a missed name is a tool bug - fixed in the rule table and its tests, never by hand in the tree.
- Optional later: a compatibility map so a user's MAYA_* setting still works (read as SAGUARO_*).

## 3. Divergence register (why we differ)
- docs/DIVERGENCE.md: per engine, each of our modifications - what, why, where (files/functions), its switch, its
  measured gain, and whether upstream could take it. Plus what we DROP from upstream on purpose (the Maya Qwen path:
  its hunks are dropped automatically by path). A conflict is resolved by reading the register, not by guessing.
- Every new topic branch adds its line when merged.

## 4. Keeping our changes easy to carry
- Prefer switch-gated additions in our own files/functions with a small hook in the shared code, over edits inside
  upstream's hot functions. Where we must edit upstream code, keep the edit small and marked (`// saguaro:` comment).
- Changes upstream might want stay as clean topic branches (PR-ready by construction), even though we don't send PRs.

## 5. Cadence
- Each upstream release: intake within a day or two (rename run -> merge -> gates below -> into the next Saguaro release).
- Open upstream PRs: a triage pass about weekly (take anything supportable for ANY hardware; inexact ones opt-in or
  KL-checked). Each pass resumes from the last one recorded here: new PRs above its newest number, plus older open
  PRs updated since its snapshot.
- QFN (engines/qfn, our own port of Strata): a sweep of upstream Strata about every two weeks; `docs/DIVERGENCE.md`
  records the last swept Strata commit. Exact gains taken, inexact ones opt-in.

## 6. Gates for every intake
- Build all engines (CUDA sm_75 + the HIP syntax check for gfx1201 / gfx1100); tools + serve test suites = main's pass
  set + new tests.
- Where upstream says "same tokens": T4 SASS / output identity spot check. Where upstream changes answers: KL vs the
  previous release on our T4s (opt-in if it fails our bar). Q8_0 stays exact.
- Smoke every live profile (mangonel llama-swap, mortar :8080); then it ships in the next Saguaro release.

## 7. Triage log (open upstream PRs)
| Date | Newest PR checked | PR list fetched | PRs | Report |
|---|---|---|---|---|
| 2026-11-24 | #1236 | 2026-11-24 23:38 UTC (16:38 MST) | 143 open | `docs/triage/TRIAGE-upstream-prs-20261124.md` |

Next pass, from the last row: `gh pr list -R mw00/project-maya --state open --search "updated:>2026-11-24T23:38:25Z"`
(new PRs and older ones changed since; a PR merged or closed since arrives with the next release intake).
