# REVIEW-RESULT: D3 - docs/DIVERGENCE.md (branch saguaro/q1-d3-divergence, base e6d840db2)

FIX-FIRST - the register is accurate in nearly everything sampled, but its "Dropped from upstream on purpose"
section describes a tree that no longer exists, and one cited source does not exist under the name given six
times. Both would misdirect the first intake that consults them. The two fixes are text-only; everything else
is optional.

Scope (brief checklist): the diff `e6d840db2..HEAD` is exactly two new docs files (`docs/DIVERGENCE.md` 697
lines, `REPORT-d3-divergence.md` 99 lines) - zero code moved, so feature-OFF behaviour is byte-identical by
construction, no locks/threads/stdout paths were touched, and `serve/qfn_engine.py` is not in the diff (its
generated content checked in sync with `engines/qfn/serve/server.py`, e.g. `SAGUARO_BATCH_WRAP_SPILL` present in
both). The plan names no test module and forbids building, so no tests were run (none run here either). No
secrets; writes confined to the two files + gitignored `scratch/names.txt`; every commit stages one file by name
and no message carries a Co-Authored-By / Claude-Session / Assisted-by line.

## HIGH

1. `docs/DIVERGENCE.md:631-658` (Dropped section) describes the pre-base tree. The qwen4exp engine paths were
   deleted by 730e785d0 ("T8b: the qwen4exp files, kernels and their tests are out of the tree", 2026-11-24),
   which is an ancestor of the base `e6d840db2` (`git merge-base --is-ancestor` confirmed):
   - "kept in our tree" (:633) and "our copy stays where it was: the v1.1.50 Qwen path" (:639) are false at
     HEAD: of the nine `Where` files (:648-650), all are gone except `expert_source.cpp` and
     `remote_experts.cpp` - `src/core/session.cpp`, `src/core/mtp.cpp`, `src/core/verify.cpp`,
     `src/core/layer.cpp`, `src/core/native_dense.cpp`, `src/core/native_head.cpp` and
     `src/prefill/prefill.cpp` do not exist in this tree (they remain only under `engines/qfn/src/core/`).
   - "~4.5K of its 10.7K lines" (:635): `src/program/generate.cpp` here is 5,752 lines; the figure is the
     pre-removal tree's (ref/reports/REPORT-qwen-engines.md:116).
   - "Nothing is deleted yet; the drop happens at intake (path exclusions in
     `tools/saguaro/rename_internal_maya.py`)" (:651-653): that tool contains no exclusion for any of these
     paths (no qwen/session/prefill references in it).
   - What is actually left is the setup/server side: `saguaro.py:706`, `:2024`, `:8951` still map old "qwen"
     family configs onto the QFN config. The section should record the post-removal state instead of "Removal
     ... is not done" (:654).
   `REPORT-d3-divergence.md:82-85` (open question 3) repeats the same stale premise ("dropped from intakes but
   not from the tree").

## MEDIUM

2. A cited source does not exist under the name used - six times: `ref/reports/REPORT-amd-ds4mm3.md`
   (`docs/DIVERGENCE.md:227`, `:228`, `:318`, `:324`, `:362`, `:596`). The tree file is
   `ref/reports/REPORT-amd-ds4-mm3.md` (extra hyphen). The cited line numbers do land on the intended text in
   the real file (:14, :30, :57-58, :19/:146, :118-133, :196-210), so only the filename needs correcting; the
   branch name `saguaro/amd-ds4mm3` is spelled right and stays.
3. `docs/DIVERGENCE.md:31` (GLM 1): "engine-wide `SAGUARO_GLM_RESIDENT_TOPK=5`" reads as the default; the
   engine default is 0 = off (`src/kernels/cuda/glm_fast.cu:6625-6627`) and 5 is the mortar run value
   equivalent to drop 3-of-8 (ref/reports/REPORT-mortar-glm.md:20). Other entries phrase it correctly
   ("default 2; live 1" - GLM 4); this one should say "default off; 5 in the mortar KL runs".
4. `docs/DIVERGENCE.md:425` (QFN 1): `nv_defaults.cpp:nv_default_env` is called the umbrella of
   `SAGUARO_SM75_PERF`. In this tree `nv_default_env` gates only the `SAGUARO_SM75_STACK` opt-in table for
   sm_75 (`engines/qfn/src/core/nv_defaults.cpp:37-40`); `SAGUARO_SM75_PERF` is read via
   `strata::sm75_default()` in `engines/qfn/include/strata/core/sm75_series.hpp:24-36`. The sub-switch cites
   are right (`kql_bits` iq_kernels.cu:4101, `q8g_on` native_mmvq.cu:2896, `gdn_keyhead_ok`
   prefill/kernels.cu:850) and the gain/default are correct (notes :65/:68).

## LOW

5. `docs/DIVERGENCE.md:381` (IK2 1): the gains "+8.6 % / +4.4 % / +11.3 %" cite ref/PERF-LEDGER.md:877-878,
   which carries the preliminary +8.3 / +4.3 / +12.6 %; the quoted numbers are in the entry's second source,
   ref/reports/REPORT-ik2-draft-gain.md:22-26. Re-drop the ledger cite or keep only the report one.
6. `docs/DIVERGENCE.md:546` (serve 3): "the check is `bench/saguaro-one/multichat.py`" - no such path exists in
   the tree (bench/ holds hy4, micro, results). The ledger (:95) quotes the same live-box script, but the
   register reads as if the resolver could find it in-tree.
7. `REPORT-d3-divergence.md:9` and `:70`: "694 lines" - the file is 697; "git diff --stat e6d840db2..HEAD: one
   file, 694 insertions" - 697 insertions (and two files as of HEAD).
8. `docs/DIVERGENCE.md:227` (DS4 16): "decode 4.4 -> 11.1-13.2 tok/s ... :58, 150" - in the real (misnamed, see
   finding 2) file the 4.4 -> 11.1 sentence is at :57-58 and the 13.2 at :19/:146; neither cited line states
   both ends of the range.
9. `docs/DIVERGENCE.md:691` (Renamed): "so its entries below name the `DS4_*` form" - the DS4 section (:128) is
   above Renamed, not below.

## NIT

10. The plan's "Entry = 2-4 lines" is exceeded by many entries (e.g. GLM 1 at :27-35, DS4 1 at :134-142 run
    9 lines), while `REPORT-d3-divergence.md:11` claims "Entry format as specified". The extra content is
    useful; only the claim is off.
11. `docs/DIVERGENCE.md:329` (HY4 intro): the "installed, no profile" text is at ref/live-stack-notes.md:30,
    not :29 (the IK2 intro's :28 is exact).
12. `docs/DIVERGENCE.md:558` (serve 6): `supervisor_name()` cited at `serve/server.py:8185`; the def is at
    :8183 - the cited line is inside the function body, harmless.

## What was checked and passed

- Plan compliance: the ten headings exactly as named, in order; 89 numbered entries matching the report table
  (16/17/13/6/6/16/8/7); every sampled entry carries what/why, `file:function`, switch + default, the measured
  gain with a source line or "not measured", and exact vs answer-changing; max line length 124.
- Major areas all present: GLM CPU lane (GLM 3/4/11), shadow caches (GLM 2, DS4 10), elastic contexts (GLM 13,
  DS4 3, MM3 3, QFN 5/7), DS4 reader / prefetch / NUMA / elastic TAKE (DS4 2/5/4/3), MM3 PAR_STREAM (MM3 5) and
  the vision tower (DS4 15 - MM3 itself has none), AMD hip_compat (setup 3, reporting the missing wiring and
  CONTRIBUTING.md:42-43 truthfully), QFN T4 switches (QFN 1/4/10/11/14), serve one-server + web (serve 1/2).
- Switch existence: every sampled name exists in-tree and/or in ref/switch-names.txt; the branch-only names are
  absent exactly as flagged (spot-checked `SAGUARO_IK2_DRAFT_Q8` and the `SAGUARO_MM3_POOL_REGISTER`/`_FILE_*`/
  `_RAM_HEADROOM_GB` group in src/mm3; `SAGUARO_SM75_Q8G_ROWS` in-tree with kernels out, as written).
- Numbers: ~25 sampled gains matched their source lines word-for-word across all eight sections
  (REPORT-mortar-glm :20/:27-28/:46/:132-149; PERF-LEDGER :35-45/:79-84/:94-100/:102-108/:143-148/:178-187/
  :205/:208/:228/:249/:253/:396-:418/:552-562/:583-587/:606-609/:722-723/:759-763/:802-805/:849-852/:866-867;
  live-stack-notes :65/:68/:73/:82-83; ROLLBACK :256-259/:267/:317-319; qwen-engines :15-21/:101;
  p40-batch :34/:45-46; upstream-v1.1.75 :150-157; ik2-draft-gain :100-101/:116; ds4-avx1-pr1204 :29-30/:101).
  No invented numbers found; the findings above are wrong-file / wrong-line attributions, not wrong values.
- file:function spot-checks were exact: ds4_model.cu `el_take_env` :10384, `el_take` :10630, `ds4_env`
  :15715-15718, the guard wait :20346 inside `run_dev_chunks` (:20156); ds4_main.cpp :2166 / :2176 / :836 /
  :1534; saguaro.py:6557; ds4_store.hpp:40; ds4_marks.hpp:19; hy4_model.cu :664/:741/:743/:786/:6558/:6596;
  ik2_main.cu :4889/:4982/:5026/:7727/:9182; ik2_draft.cu:1099; serve/server.py :247/:536-549/:2981/:2988/
  :6243/:7147/:7698/:8185/:8193/:8255/:8738; qfn generate.cpp:18776; qfn_fit.py :29/:33/:93-94/:146; ik2_fit.py
  :364/:373; maya_bench.py:34; extract_qfn_engine.py:43; test_dashboard.py:429; hip_syntax_check.py :12/:78
  with test_hip_compat.py and CI carrying no reference to it (as the entry and open question 6 state);
  hip_backend.cmake:96; tests.yml:53; chats.js :61/:88; mm3_cpu.cpp:159; native_expert.cpp:118; glm files for
  entries 2/5/6/9/10/15; mm3_model.cu :642/:681/:1659.
- All 18 cited commit IDs (730e785d0 aside) resolve in this history; docs/UPSTREAM.md §3 (:21) and §5 (:36-37)
  say what the register claims; the missing last-swept Strata commit is honestly left open (report Q1), not
  invented.
- No paths outside the repo except runtime-location strings (/opt/llm/...) that mirror the sources' own wording.
