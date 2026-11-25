# REPORT: D3 - docs/DIVERGENCE.md (the divergence register)

Branch `saguaro/q1-d3-divergence`, base `e6d840db2`. Docs only: one new file, no code, no build, no engines, no
network, no scripts (PLAN-d3-divergence.md's "Don't"). Nothing extracted, generated or diffed from switches - the
previous run's mistake is named in the plan and was avoided.

## What changed

`docs/DIVERGENCE.md` (new, 694 lines / 64 KB): the register docs/UPSTREAM.md §3 asks for, with the ten headings the
plan named and 89 numbered entries. Entry format as specified: what, why, where (`file:function`), switch and default,
the measured gain with the source line that states it (or "not measured"), and exact vs answer-changing. Numbers are
copied from the sources, never re-derived.

| section | entries | note |
|---|---:|---|
| GLM | 16 | Turbo, CONV_SHADOW, the AVX1 rows, the CPU gate, row halves, chunk 4096 + ramp off, the even split, prestage, ARENA_PAR, SLOT_FAST, the CPU lane, the spec tail thread, elastic contexts, the tier fix, KDA graphs pinned off, and the four not-taken / parked ones |
| DS4 | 17 | read helper + BORROW, the reader, elastic conversations/TAKE, NUMA staging, prefetch, PICK_MAX, the indexer int8 TC, the AVX1 rows + PR #1204's scale table, v7t4 attention, LEND_SHADOW/PIN_OVERLAP, the read-help guard, restore points/rewind, slot keep, DSpark always loaded, the FP16 vision tower, AMD, and ten measured-and-not-shipped |
| MM3 | 13 | ATTN_TC, KV_GATHER, ELASTIC, READ_HELP + STAGE_OVERLAP, PAR_STREAM, saved conversations, the expert profile, chunk/staging, 262K vs the 1M dead end, Turbo/lane, not-shipped, the AMD branch, the router warning |
| HY4 | 6 | tensor parallel, KV spill, the Engram table, PCIE_SAFE, HOT_SEED, and why it is not on AMD or Pascal |
| IK2 | 6 | the drafting switches and defaults, the Pascal kernels, the `--tp 2` pair, the drafter we dropped, AMD/Volta, measured and not built |
| QFN | 16 | the sm_75 stack, piped MTP + Lamina, the prompt-MMQ stack, SELECT_SM75, the batch engine, park/sessions, elastic YaRN, wrap-spill, tail skip (closed), the BF16-TC skip list, GDN (power cap), the pool split, RDNA4, Pascal, Turbo, and the parked exact near-misses |
| serve + web | 8 | one server with QFN's generated driver inside, the multi-chat page and `/health` capabilities, the `jobs_put` fix, logprobs without a 400, the rebrand and what it kept, the supervisor refusal, the loop guard and fair read, the `gt=` skip and remaining gaps |
| setup/tools | 7 | the rename tools, `extract_qfn_engine.py`, the AMD checks, QFN's in-tree build, `qfn_fit`, the per-engine setup rules, `--bench` and the path cleanup |
| Dropped from upstream on purpose | - | the Maya Qwen path, written in step 1 |
| Renamed | - | the rename layer, written in step 1 |

Commits, one per plan step (docs/DIVERGENCE.md only, `git add` of that file alone each time):

    d1e8fc5fd  step 1: the register's shape + "Dropped from upstream on purpose" + "Renamed"
    2da3029c8  step 2: QFN (16) and GLM (16)
    f9c91f9eb  step 3: DS4 (17) and MM3 (13)
    8012848bb  step 4: HY4 (6), IK2 (6), serve + web (8), setup/tools (7)
    (this report)  step 5

## Files

- Added: `docs/DIVERGENCE.md`, `REPORT-d3-divergence.md` (this file).
- Read only: `PLAN-d3-divergence.md`, `docs/UPSTREAM.md`, `ref/PERF-LEDGER.md`, `ref/ROLLBACK.md`,
  `ref/live-stack-notes.md`, `ref/switch-names.txt`, all 13 files in `ref/reports/`, and the code the entries cite
  (`src/`, `engines/qfn/`, `include/`, `serve/`, `tools/`, `saguaro.py`, `CMakeLists.txt`, `configs/`, `data/`,
  `.github/workflows/`).
- `scratch/` (gitignored) holds one throwaway file, `scratch/names.txt`, the list of switch names swept for the check
  above. Nothing else was written outside the two files in "Files".

## Tests run

None. The plan names no test modules and forbids building, running engines or the network, so nothing was executed
beyond read-only greps. What was verified instead:

- `grep -n` (or the Grep tool) per entry to fill in `file:function`, and the enclosing function checked by reading
  back from the line where the switch is read (several of these files put everything in one `struct Model::Impl`, so a
  name-only grep gives the wrong function - `prefill_half`, `run_dev_chunks`, `spill_layer` and `kv_gather_mode` were
  found this way).
- Every switch named in the register was checked to exist. All 166 distinct `SAGUARO_*` / `MAYA_*` / `STRATA_*` /
  `DS4_*` / `MM3_*` / `IK2_*` / `HY4_*` tokens in the file were swept against `ref/switch-names.txt` and against the
  tree; the ones that do not exist are named in the text as branch or upstream-side names (the `MAYA_GLM_*` trio in
  the Renamed section is upstream's, which we renamed away from). Six groups are flagged in place as *not in this
  tree*: `SAGUARO_MM3_POOL_REGISTER` / `_FILE_EVICT` / `_FILE_READERS` / `_RAM_HEADROOM_GB` and `_DIRECT`
  (saguaro/amd-ds4mm3 - confirmed with `git merge-base --is-ancestor`: 09367e7ed is not an ancestor of HEAD),
  `STRATA_GLM_FP16TC` (pr/glm-fp16tc), `SAGUARO_IK2_DRAFT_Q8` (dropped on the merge for upstream's GGUF drafter), and
  the DS4 experiments `DS4_EP` / `DS4_ENGRAM_DECODE_KEEP` / `DS4_MODULE_WARM` / `DS4_SETTLE_FEED`
  (`SAGUARO_DS4_READ_SHARE`, by contrast, *is* in the tree at `src/ds4/ds4_main.cpp:2166`). QFN's
  `SAGUARO_SM75_Q8G_ROWS` is in the tree while the q8rows kernels are not.
- Three claims were corrected during that sweep rather than left as written in the sources: the MiniMax AMD switches
  are branch work, DeepSeek's `--ssd-cache auto` setup rule is branch work too (the engine flag exists, `saguaro.py`
  does not write it), and `tools/test_hip_compat.py` in this tree does not run `tools/hip_syntax_check.py` although the
  tool's own header says it does.
- `git diff --stat e6d840db2..HEAD`: one file, 694 insertions, no deletions - no code touched.
- Line lengths kept under 125 characters, in line with `docs/UPSTREAM.md` (max 122).

## Open questions

1. **The last swept Strata commit is nowhere in the sources.** docs/UPSTREAM.md §5 says `docs/DIVERGENCE.md` records
   it; the plan did not ask for it and inventing it would be worse. The QFN preamble states the port base that *is*
   documented (the subtree of the live Strata tree 82c389f5e + server 0c800169d, ref/live-stack-notes.md:65) and says
   the sweep record is missing. Ken: which Strata commit was last swept?
2. **"Whether upstream could take it" is in docs/UPSTREAM.md §3 but not in the plan's entry format**, so no entry has
   that field. A few entries imply it (e.g. upstream's drafting gate being unstable on Turing, PR #1204's rows being
   slower than ours). Add a per-entry "upstream: yes / no / already asked" line, or leave §3's wording narrower?
3. **The Maya Qwen path is dropped from intakes but not from the tree.** The register records the current state
   (upstream hunks dropped by path, our copy still v1.1.50's, removal recommended and Ken's call). Should it instead
   read as a decision - i.e. did Ken pick option 1? - and should the Dropped section name the branch that would do the
   deletion (qfn/in-file-nextn d025b3d9f, not merged)?
4. **"Live" defaults are quoted from the notes, not from a file.** The live configs are outside the repo
   (`/opt/llm/saguaro/configs/...`), so every "live since ..." line comes from `ref/live-stack-notes.md` or
   `ref/ROLLBACK.md`. If a config and those notes disagree, the config wins and the register is wrong. Anything to
   check them against inside the tree?
5. **Seven entries say "not measured"** where the sources give no number: GLM elastic contexts as a whole (the ships
   are documented, not one A/B), the DS4 restore points / rewind and slot keep, the HY4 tensor parallel, KV spill and
   Engram table (no Hy4 profile exists), and the MM3 learned expert profile - the ledger explicitly defers its decode
   re-measure ("re-measure decode after a few days", ref/PERF-LEDGER.md:122). Do you want a measurement pass, or is
   "not measured" acceptable in the register?
6. **Two inconsistencies found while checking, reported rather than fixed** (the plan forbids code changes):
   `tools/hip_syntax_check.py:12` says `tools/test_hip_compat.py` runs it when `SAGUARO_HIP_SYNTAX_ROCM` is set, but no
   line of this tree's `tools/test_hip_compat.py` or CI refers to the tool; and `CONTRIBUTING.md:42-43` still says "we
   can't test AMD ourselves" while the AMD checks exist. The register states both as they are, with the wiring
   attributed to the `saguaro/amd-ds4mm3` lineage. Fix the tool's header, or merge the wiring?
