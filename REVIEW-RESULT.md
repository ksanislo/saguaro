# REVIEW-RESULT: D3 round 2 - docs/DIVERGENCE.md at a0f5ec34c (branch saguaro/q1-d3-divergence, base e6d840db2)

MERGE-WITH-NITS - the twelve findings of the first review (e38566f0f) are all fixed and re-verified, no invented
number was found in ~45 numbers checked this pass, and the diff is docs-only; but the re-read turned up one wrong
quality-gate sentence (finding 1), three cite/behaviour slips (2-5) and five nits. All text-only.

Scope (brief checklist): `git diff e6d840db2..HEAD` adds exactly four docs files (`docs/DIVERGENCE.md` 708 lines,
`REPORT-d3-divergence.md`, `REVIEW-RESULT.md`, `FIX-REPORT.md`), 991 insertions, zero deletions - no code moved, so
feature-OFF behaviour is byte-identical by construction, no lock/thread/stdout path exists in the diff, and
`serve/qfn_engine.py` is not in it; the generated pair is in sync (`BATCH_WRAP_SPILL` 6x in `serve/qfn_engine.py`,
13x in `engines/qfn/serve/server.py`; `tools/saguaro/extract_qfn_engine.py:43` writes it and `serve/test_dashboard.py:429`
reads the generated file). 22 commits, each staging one of the four docs files by name; no message carries a
Co-Authored-By / Claude-Session / Assisted-by line. No secrets in the added files. The plan names no test module and
forbids building, running engines or the network, so nothing was executed (none run here either).

## MEDIUM

1. `docs/DIVERGENCE.md:487-488` (QFN 10): "router alone on TC gives top-1 96.91 %, all three 96.58 %, **below the
   96.24 % bar**" - inverted. The bar (ref/PERF-LEDGER.md:28) is a FLOOR, "top-1 agreement vs live >= 96.24 %", and
   both quoted numbers sit above it; the ledger's verdict at :451-452 is "FAIL quality gate - skip list justified"
   without ever saying below-bar (the operating-point FAIL is the KL cap, :271, and KL 0.0101 > 0.01 at :541-542 -
   the GLM sibling). The entry's conclusion (the skip list stays) matches the ledger, but the stated reason
   contradicts the line it cites and teaches the gate's direction backwards.

## LOW

2. `docs/DIVERGENCE.md:96` (GLM 12): "65K read -5.4 % at split **11,23,34**" - ref/PERF-LEDGER.md:317-318 ran that
   A/B at split 11,23,34; the string 11,23,34 occurs nowhere in ref/. The shipped split 12,23,34 is right.
3. `docs/DIVERGENCE.md:177` (DS4 7): "cards -6.1 % / host -6.7 % energy per read (ref/PERF-LEDGER.md:35, **494**)" -
   that scoring line is at :495; :494 is the Prometheus header of the block. GLM 4 (:54) cites the same line
   correctly as :495.
4. `docs/DIVERGENCE.md:506-508` (QFN 13): "the hipBLASLt table is **+12 % on 8K reads** but makes two loads differ,
   and gfx1151's decode table is -13 % on gfx1201 (ref/reports/REPORT-qfn-amd.md:169)" - :169 carries only the
   nondeterminism and the -13 %; the +12 % on 8K reads is :180 of that report.
5. `docs/DIVERGENCE.md:169-170` (DS4 6): "`DS4_PICK_MAX` (**0 = auto**)" - in `src/ds4/ds4_model.cu:5398-5399` an
   UNSET switch is auto (kPickMax = 2048, :1239); a literal 0 is clamped to 1, i.e. every prompt takes the flow
   path - the opposite of auto. Nothing in the sources says "0 = auto" (ref/PERF-LEDGER.md:528 has "auto below N").
   A resolver trying `DS4_PICK_MAX=0` to get the upstream behaviour would get the maximally-changed one.

## NIT

6. `REPORT-d3-divergence.md:74`: "git diff --stat e6d840db2..HEAD: docs files only (`docs/DIVERGENCE.md`, this
   report, `REVIEW-RESULT.md`)" - at HEAD the diff carries four; `FIX-REPORT.md` (a0f5ec34c) is not listed.
7. `REPORT-d3-divergence.md:9`: "694 lines / 64 KB at step 5" - at the end of step 5 (`20e518c7a`) the register is
   697 lines / 64.2 KiB; 694 was the end of step 4 (`8012848bb`), the name sweep `c6a8cc8e3` added three lines.
   "708 lines / 65 KB" at HEAD is right (66,926 bytes).
8. `docs/DIVERGENCE.md:603-607` (setup 4): the cite `CMakeLists.txt:1787-1808` covers the ISA floor (:1795) and the
   SM60 rule (:1802) but not the ninja inheritance the same sentence names (`CMAKE_MAKE_PROGRAM`, :1816; the option
   is :1776).
9. `docs/DIVERGENCE.md:119-120` (GLM 16): "KL at the cap (0.0101 vs off 0.0094, every on-arm below every off-arm)" -
   the parenthetical's "every on below every off" is the ledger's *same-top* comparison (96.19 vs 96.68,
   ref/PERF-LEDGER.md:541-542); for KL the on-arm is above (0.0101 > 0.0094).
10. `docs/DIVERGENCE.md:112` (GLM 15): "Where: `kda_graph_wanted()` and `src/core/glm_layer_graphs.hpp`" - the
    function that reads `SAGUARO_GLM_KDA_GRAPH` is `src/core/glm_fast_path.cu:7292-7293`; the .hpp is the capture
    machinery but does not define the function (a file:function pairing would name glm_fast_path.cu).

## The first review's twelve findings: all fixed, re-verified this pass

Dropped section now the post-removal state (`src/program/generate.cpp` is 5,752 lines and ends at :5747-5751 with
"this model runs on saguaro-qfn"; of the nine Where files only `src/core/expert_source.cpp` and
`src/core/remote_experts.cpp` exist; 730e785d0 / 37de16bb5 / 5bf926032 / d8e8faf84 are ancestors of HEAD;
CONTRIBUTING.md:29-33 is the intake rule; `saguaro.py:2024/:8951/:706` are the one-way config read). The six cites
now name `ref/reports/REPORT-amd-ds4-mm3.md` and the lines land (:14, :19, :30 - which really does carry the
"exactly those 12 errors plus the `'+f'` constraint" sentence, :57-58, :118-133, :146, :196-210). GLM 1 reads
"default 0 = off, 5 in the mortar KL runs" (code `glm_fast.cu:6624-6627`, report :20). QFN 1 points at
`sm75_series.hpp` and says `nv_default_env` gates only `SAGUARO_SM75_STACK`. IK2 1's +8.6/+4.4/+11.3 % come from
`REPORT-ik2-draft-gain.md:22-26` with the ledger's preliminary row marked. serve 3 says multichat.py is the live
box's, not the tree's (bench/ holds hy4, micro, results). supervisor_name() at :8183, HY4 intro at notes :30,
Renamed says "above". The one thing that did not survive its own fix is the fix's line count (nit 7).

## Sampled and passed (independent of the first review)

- Plan compliance: the ten headings exactly as named and in order; 89 numbered entries, no gaps, 16/17/13/6/6/16/8/7
  matching the report table; every sampled entry carries what/why, where, switch + default, the gain with its source
  line or "not measured", exact vs answer-changing; max line length 124; the Dropped and Renamed sections were
  written first (step 1, 120 lines at d1e8fc5fd).
- Entries whose code cites were re-opened this pass and every file:function lands: GLM 1-7, 9-12, 14-16 (incl.
  `prefill_half` :2108 + `pf_cpu_gate` :2499 default 2, `row_halves_of` :518, `arena_par_threads` :555,
  `slot_fast_on` :9157, `decode_spec` :8349 + `SAGUARO_GLM_SPEC_TAIL_THREAD` :8766, `SAGUARO_GLM_SPLIT` :2013 with
  0 = one part); DS4 1-3, 5-17 (incl. `el_take_env` :10384, `el_take` :10630, `ds4_env` :15715-15718, the guard at
  :20346, `ds4_main.cpp` :836/:1531/:2166/:2176, `ds4_marks.hpp:19`, `ds4_store.hpp:40`, `saguaro.py:6557`, prefetch
  defaults 16/2/6 at :13970-13972, elastic pool 1048576/256/"on unless =0" at :10342-10381, INDEX_TC modes at
  ds4_kernels.cu:4314-4328, vision :276/:454-456, restore default 4 at :11145); MM3 1-6, 8-13 (incl.
  `kv_gather_mode` :642, `read_end` :681, `mm3_main.cpp` :389/:584); HY4 1-6 (all fourteen line cites in hy4_model.cu
  exact, `hy4_model.hpp:61`, `saguaro.py:771-772`); IK2 1-6 (ik2_main.cu :4889/:4982/:5026/:7727/:9182,
  ik2_draft.cu:1099, ik2_pascal.cu :498-534); QFN 1-16 (incl. `SAGUARO_PIPED_MTP_MAX` default 2 at generate.cpp:5760,
  the turbo-skip line :18776, `host_elastic_on` :666, `park_fast` :157, `SAGUARO_PARALLEL_LOAD_REGISTER` :5711,
  `gdn_conv` :2596, `bf16_path` :824, `dense_f32_on` :872, `ExpertPool` + `SAGUARO_NUMA_POOL` :750/:762,
  qfn_fit.py :29/:33/:93-94/:146, hip_backend.cmake:96); serve 1-8 (server.py :247/:536-549/:2981/:2988/:6243/:7147/
  :7698/:8183/:8193/:8255/:8738, loop_guard.py:163, chats.js :61/:88, ds41_encoding.py); setup 1-7 (hip_syntax_check.py
  :12/:78 with zero refs to `SAGUARO_HIP_SYNTAX_ROCM` in this tree's tests or CI as the entry says, 90fcee373's
  check_hip_compat PTX work confirmed in that commit, ik2_fit.py :364/:373, maya_bench.py:34, saguaro.py :771).
- Numbers: ~45 gains re-read against their cited source lines this pass, verbatim throughout - PERF-LEDGER
  :28/:30/:35-45/:45/:67-68/:79-84/:94/:97/:100/:102-108/:109-111/:112-119/:123-128/:143-148/:150-155/:178/:193-194/
  :205/:208/:210/:226/:228/:230/:238/:245/:247/:249/:253/:265/:271-281/:283-287/:317-328/:418-420/:424-425/:451-453/
  :495/:503-505/:540-548/:552-555/:556-562/:574-582/:583-587/:594-596/:600/:606-609/:610-613/:632-635/:647/:654/:662/
  :668-669/:688/:722-723/:756/:759-763/:789-791/:806/:836-838/:849-852/:859-863/:866-867/:883-884/:886-888;
  live-stack-notes :20-25/:26-27/:28/:30/:65/:68/:71/:73/:82-83; ROLLBACK :14-19/:27-28/:45/:72-74/:95/:128-132/
  :256-257/:260/:264/:267/:315-319; REPORT-mortar-glm :20/:27-28/:46/:132-149; REPORT-qwen-engines :15-21/:78-82/:101;
  REPORT-upstream-v1.1.75 :11-21/:104-105/:150-157/:162; REPORT-ik2-draft-gain :22-26/:89-95/:100-101/:116;
  REPORT-ik2-p40-tuning :28-34/:48/:62/:64-68; REPORT-p40-batch :18-30/:34/:45-46; REPORT-qfn-amd :119/:169 (nit 4);
  REPORT-amd-ds4-mm3 as in finding list. The register's own reserve figures (689/-later 687, not :756's test config
  691) match ROLLBACK:264 and live-stack-notes:65. No invented number found.
- Switch existence / absence: every sampled switch was found in code (`DS4_ATTN_V7_T4` :5012, `HY4_TP_*` all read at
  the cited lines, `SAGUARO_SELECT_SM75` :1872, `SAGUARO_BF16_TC(_SKIP)` :826 / prefill.cpp:366, the IK2 Pascal gates,
  the DS4 elastic/prefetch/INDEX families above), and every absent-as-flagged name is absent for real - zero hits in
  src/ engines/ tools/ serve/ include/ for `SAGUARO_MM3_POOL_REGISTER`/`_FILE_EVICT`/`_FILE_READERS`/
  `_RAM_HEADROOM_GB`/`_DIRECT`, `SAGUARO_IK2_DRAFT_Q8`, `DS4_EP`, `DS4_ENGRAM_DECODE_KEEP`, `DS4_MODULE_WARM`,
  `DS4_SETTLE_FEED`, `SAGUARO_Q8_ROWS_SM75`; `09367e7ed` is not an ancestor of HEAD; `SAGUARO_SM75_Q8G_ROWS` is in
  the tree (native_mmvq.cu:2918, default 2) while the pr/qfn-q8rows kernels are not - as the entries write.
  (ref/switch-names.txt is a prefixed-name list: bare `DS4_*`/`HY4_*` forms live only in the tree and the configs,
  which is exactly what the DS4 intro and Renamed bullet say.)
- Commit IDs: ~30 sampled (730e785d0, 37de16bb5, 5bf926032, d8e8faf84, c38b9747, 8d73f40, fbbf82c17, 90fcee373,
  bdbc727ba, c2ca8a1a6, dc6e746a4, f784321fd, d700057c5, e58d740a3, 3d1b67905, a92df34f6, 7f8804760, 1823471fb,
  09367e7ed, 6e747b976, 7e998686d, 53b3fd4fe, eed11a966, adf1aeb, 6230a6d, df41596, 3657d62, 2fd646a, eccfbd4,
  da7e59f, aa3365d) - all resolve; every live-default claim sampled ("live since ...") matches ROLLBACK.md or
  live-stack-notes.md.
- Major areas (the brief's list) are all present: GLM CPU lane (GLM 3/4/11), shadow caches (GLM 2, DS4 10), elastic
  contexts (GLM 13, DS4 3, MM3 3, QFN 5/7), DS4 reader/prefetch/NUMA/elastic TAKE (DS4 2/5/4/3), MM3 PAR_STREAM
  (MM3 5); MM3 has no vision tower of its own - the ledger's only MiniMax picture mention is :128, a saved-slot
  limit - and vision is recorded where it exists (DS4 15), consistent with the first review. AMD hip_compat
  (setup 3, truthful about the un-wired syntax check), QFN T4 switches (QFN 1/4/10/11/14), serve one-server + web
  (serve 1/2). No sampled entry misstates what is shipped or live; findings 1-9 are about gate reasoning and
  cite precision, not about the live state an intake resolver reads off the register.
- Housekeeping: no path outside the repo appears except live-stack location strings (/opt/llm/...) that mirror the
  sources; the setup/tools 7 entry's own "paths out of configs/" claim holds (the cleanup is reported as a done
  item from REPORT-mortar-fresh-install).
