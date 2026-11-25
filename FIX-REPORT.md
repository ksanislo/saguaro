# FIX-REPORT: the D3 review's findings, each mapped to what changed

Branch `saguaro/q1-d3-divergence`, review `REVIEW-RESULT.md` (e38566f0f): one high, two mediums, five lows, three
nits. Every finding was re-checked against this tree before editing; all twelve were real and all twelve are fixed,
one text-only commit each (a62039538 .. c0d4667e9, plus the report-count follow-up 3c38b2563). No finding was
refused; nothing in the register was found to be wrongly reported by the review. Docs only, as before - no code, no
generated files (`serve/qfn_engine.py` untouched), no build, no engines, no network.

## The findings

1. **HIGH - the "Dropped from upstream on purpose" section described the pre-removal tree.** Fixed in a62039538.
   Verified first: `git merge-base --is-ancestor` confirms 730e785d0 (T8b engine), 37de16bb5 (the driver out of
   `generate.cpp`) and 5bf926032 (T8a Python/web merge) are all ancestors of HEAD; of the nine `Where` files only
   `src/core/expert_source.cpp` and `src/core/remote_experts.cpp` exist (730e785d0's own message names them, plus
   mrope / router_sigmoid / qsa.hpp / kv_stream.hpp / strata-plan, all now cited); `src/program/generate.cpp` is
   5,752 lines and ends at `:5747-5751` with "this model runs on saguaro-qfn"; `rename_internal_maya.py` indeed
   carries no exclusion for any of these paths (confirmed by grep); the intake rule is the CONTRIBUTING.md:29-33
   line d8e8faf84 added. The rewrite: "kept in our tree / removal not done" becomes the executed removal (T8a + T8b,
   CHANGELOG.md:43-57), the "~4.5K of 10.7K" becomes "of its then-10.7K lines" with today's count, `Where` names what
   remains and says the rest lives only under `engines/qfn/src/`, and the new `Intake` bullet states the rule and the
   one-way config read (`saguaro.py:2024`, `:8951`, `:706` - checked in place). The Why bullet (gains, KL) was
   sampled-correct and stays verbatim. REPORT open question 3 was the same stale premise and now records the
   correction instead (same commit).
2. **MEDIUM - `ref/reports/REPORT-amd-ds4mm3.md` does not exist under that name.** Fixed in 68107aa1a: all six
   cites (`ls ref/reports/` shows the real file `REPORT-amd-ds4-mm3.md`) renamed to the hyphenated form; the branch
   name `saguaro/amd-ds4mm3` is spelled right and was left alone (grep confirms it is the only remaining match).
3. **MEDIUM - GLM 1 stated `SAGUARO_GLM_RESIDENT_TOPK=5` as the default.** Fixed in 3fe5e0968: the engine reads
   `v ? atoi : 0` (glm_fast.cu:6625-6627, re-read), so the entry now says "default 0 = off, 5 in the mortar KL runs
   (= drop 3-of-8, ref/reports/REPORT-mortar-glm.md:20)".
4. **MEDIUM - QFN 1 called `nv_default_env` the umbrella of `SAGUARO_SM75_PERF`.** Fixed in a04a3b149: the entry
   now names `strata::sm75_series()` / `sm75_default()` (engines/qfn/include/strata/core/sm75_series.hpp, read at
   :27) as the umbrella and says plainly that `nv_defaults.cpp:nv_default_env` only gates the separate
   `SAGUARO_SM75_STACK` opt-in table for sm_75 (`:37-40`, both re-read); the sub-switch, gain and default cites
   were right and stand.
5. **LOW - IK2 1's gains cited the wrong source line.** Fixed in e086ab670: the quoted +8.6 / +4.4 / +11.3 % are
   REPORT-ik2-draft-gain.md:22-26 (re-read); the ledger cite was kept but marked, not dropped - PERF-LEDGER.md:877-878
   carries the preliminary +8.3 / +4.3 / +12.6 %, and saying so stops a later reader from "correcting" the entry to
   the preliminary row.
6. **LOW - serve 3 cited `bench/saguaro-one/multichat.py` as if in-tree.** Fixed in 54226d1a9: `bench/` holds hy4,
   micro, results (checked); the entry now says the check was the live box's script, a script that is not in this
   repo, with the same ledger cite.
7. **LOW - the step report's counts were stale.** Fixed in 38ae0fc86, settled in 3c38b2563: `docs/DIVERGENCE.md`
   now reads 708 lines / 65 KB after the fix-ups, 694 / 64 KB at step 5, and the diff-stat line says docs files
   only (register, report, REVIEW-RESULT.md), insertions only - no code touched.
8. **LOW - DS4 16's gain cites each stated neither both ends.** Fixed in e233a199f: the entry now puts 4.4 -> 11.1
   at REPORT-amd-ds4-mm3.md:57-58 and the 13.2 at :19 and :146 (each re-read in the real file).
9. **LOW - Renamed said "entries below".** Fixed in 6118e24f3: the DS4 section is above Renamed; reads "above".
10. **NIT - "Entry format as specified" overstated.** Fixed in c40a09519: the report now claims the specified
    fields (all present, as the review confirmed) and admits many entries run past the plan's 2-4 lines to carry
    them. The entries themselves were left alone - the review called the extra content useful.
11. **NIT - HY4 intro cited live-stack-notes.md:29.** Fixed in ef4b15184: "installed, no profile" is at :30 (the
    IK2 intro's :28 re-read and exact, left as written).
12. **NIT - supervisor_name() cited at :8185.** Fixed in c0d4667e9: the def is `serve/server.py:8183` (grep
    confirmed; the old cite was inside the body).

## Housekeeping in this commit

The step report's commit list gains two lines pointing at the review fix-ups (twelve one-finding commits, the
count follow-up, and this report; the list named only the original six). FIX-REPORT.md is force-added, as every
REPORT-*.md and the earlier FIX-REPORT.md was: the main checkout's `info/exclude` ignores this filename.

## Tests run

None. The plan names no test module and forbids building (unchanged by the review), and every change here is a docs
line. What was verified instead: each finding re-checked in the tree before its edit; no register line exceeds 124
characters after the edits; `git status` clean before and after every commit; each commit staged its one or two
files by name, never `git add -A`; no commit message carries a Co-Authored-By / Claude-Session / Assisted-by line.
