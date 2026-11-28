# FIX-REPORT - P125o review (FIX-FIRST) - the K / V grows on demand, step 2

The review (REVIEW-RESULT.md, verdict FIX-FIRST) over `git diff 37c7523ca..HEAD` (the a2 / a3 step) named one
CRITICAL, one HIGH, one MEDIUM, one LOW, two NITs and one INFO. Every finding but the INFO is fixed here, one
numbered fix per commit; the nits were cheap and are in too. No kernels were touched - each fix is where the
review's own fix direction pointed. No GPU was run and none exists on this box (nor Inkling weights), so the
verification is the plan's host-side tests, `bash -n`, and rebuilds of `saguaro-ik2` for CUDA sm_75 in
scratch/build-ik2-elastic-2. `serve/qfn_engine.py`, `serve/server.py` and `tools/saguaro/extract_qfn_engine.py`
appear in no commit of the range, as the manager re-checks.

## Finding -> fix

1. **CRITICAL - the K / V-write epilog wrapped `% kv_cap` over the committed start step** - **fixed** (e0efdcb7d).
   The three full-visit `prep_qkv_r_*` groups (the prompt path, the token path, the block / graph windows) now pass
   the ring the attention already reads by: `kv_cap = 0` - absolute - for a non-sliding visit when `g_kv_vmm`, via
   `write_kv_cap` / `wcap` locals beside the existing `ring` ones. Sliding layers keep their ring; every plain
   layout still passes `Ly.kv_cap`, where `pos < kv_cap` made the wrap the identity already - the OFF path
   unchanged. Past the start step a row now lands at its own position instead of trampling the conversation's
   early rows while attention read the unwritten absolute rows as zeros, and with fix 2's commits the stretch is
   mapped before the write. The `--tp` and `--parallel` prep sites are untouched on purpose: the elastic gate
   excludes those modes, so there `kv_cap` is never a commit and the wrap stays the identity. The a2 comment at the
   allocation tells the truth now (the reads always passed 0, the epilogs did not), and the report's "the kernels
   needed nothing / no wrap to rebase" claim is superseded by this fix and fix 4's note.
   *The card test's part*: `ask` now prints the reply's first word and the growth read's verdict requires `alpha` -
   tokens alone were no verdict, and this bug emitted tokens fine. Canned-row runs of the verdict python: answered
   `alpha` -> PASS, answered `bravo` -> FAIL naming the write epilog, refusal on a growing start -> FAIL as
   before, refusal on a fallback -> unchanged, `GROW=0` -> unchanged.

2. **HIGH - the `--drafting auto` trial wrote its prompt rows before committing any of them** - **fixed**
   (5170e0505). `trial()` commits the request's whole span (its prompt plus its generation, clamped to `ctx` like
   the run) host-side before the first `forward`, so the prompt pass never runs ahead of a commit - no wrapped
   rows (pre-fix-1) and no unmapped-write fault killing the engine before READY (post-fix-1). VRAM short, the
   measurement stops there with one NOTE naming how many of the requests it ran; the unfinished request adds no
   tokens and no seconds. Both side-effects are in the same commit: an empty `out` from `decode_greedy` no longer
   feeds `made += out.size() - 1` a negative token count (the trial drops the request), and `decode_greedy`
   clamps its commit to `ctx`, so an oversized `SAGUARO_IK2_DRAFT_TOKENS` cannot set `g_kv_chunks` past the
   ceiling's chunk count. The commit stays outside every token loop, as the manager's first check demands.

3. **MEDIUM - `unpark` failures failed every request with a false OOM on all cards** - **fixed** (41072c757).
   `full_rows` and `unpark` now return the kind of result (0 done, -1 the pre-existing failure, 1 the grow failure)
   instead of a bool. Only the 1 - the stretch of the elastic K / V that would not commit - fails the request with
   the out-of-memory ERR + DONE it earned. The old falses (the folder file that would not read, the missing
   snapshot at the join) fall through to the from-scratch re-read the call ignored the result for at
   37c7523ca:10802: cache empty, request goes on, no misleading message - on every card, including ones where
   `g_kv_vmm` is false and VRAM cannot be the cause. The feature-OFF path is byte-identical to P125e again.

4. **LOW - the growable cache depended on freshly mapped pages reading as zero** - **fixed** (593527e06).
   `kv_grow` memsets each freshly committed span (`[old, chunks_for(n) * chunk_b)`, clamped to the ceiling's chunk
   count) on the request stream at the commit - one memset a chunk, host-side, never per token, ordered with the
   request's kernels on `g_s`. `cuMemCreate` / `hipMemCreate` promise no more than `cudaMalloc` does about what a
   new page reads as, and the load's DEPTH memset stops at the start step; now the zeros the DEPTH test and the
   tiles' not-yet-written rows read are zeros the engine wrote. The claim was replaced where it lived
   (ik2_vmm.hpp's header, and the a2 comment already said so via fix 1's edit). The load-time DEPTH memset and the
   initial commit at the load are unchanged.

5. **NIT - the drafter-drop comparisons could change non-elastic behavior under rare env knobs** - **fixed**
   (deadadfc4). All three - `decode_greedy`'s trial hook, the serve request's `spec_rows` factor, the CLI's drop
   and its printed line - are gated on `g_kv_vmm`: a run can only reach past the drafter's commit-sized rows where
   this step's growable cache is on, so with the step OFF the `--drafting auto` measurement and the CLI's drafting
   are exactly P125e's whatever `SAGUARO_IK2_DRAFT_CTX` / `SAGUARO_IK2_DRAFT_TOKENS` hold. (On a VMM-backed card
   the drop still does its job - the answers were always identical, drafting verifies.)

6. **NIT - the `kv cache:` line counts the layout, not the mapped bytes** - **fixed** (3a77926da): a comment at the
   print says exactly that - `Vi.kv_cap` is the commit while the commit calls round up to whole chunks (10000 ->
   12288 rows), exact at the default 32768, the card test's 2 % rule covers the slack. The printed number itself
   is deliberately left: the load-memory comparison the plan asks for ("memory at load stays the same at 32768 and
   262144") is a comparison of the layout.

7. **INFO - the grow-failure `@END error out_of_memory` field order** - **not changed, nothing to change**: the
   review raised no defect and offered no fix; the ordering rests on the sibling `context_length` line and the
   legacy `@END` reader lives outside this repo, so there is nothing here to verify against or act on.

## Tests run (plan-named modules only, one at a time)

- `g++ -std=c++17 -Wall -o scratch/ik2_plan_test_fix tests/ik2_plan_test.cpp && scratch/ik2_plan_test_fix`
  -> `ik2_plan_test: ok` (45 checks; run before the fixes as a baseline and after every change).
- `bash -n tools/card_tests/ik2_kv_start.sh` -> clean; the verdict python extracted and exercised on canned rows in
  both directions (correct word, wrong word, refusals, no growth read) as in finding 1 above.
- `ninja -C scratch/build-ik2-elastic-2 saguaro-ik2` after each engine fix -> clean, only the two pre-existing
  nvcc warnings (`Shards::num`, `g_apu`); the tree was verified current at the review's HEAD before the first
  change. No GPU run anywhere - the card script still needs Ken's cards for the end-to-end growth read, which now
  judges the answer, not the token count.

## Commits

e0efdcb7d fix 1 (CRITICAL, engine + card test), 5170e0505 fix 2 (HIGH), 41072c757 fix 3 (MEDIUM),
593527e06 fix 4 (LOW, ik2_main.cu + ik2_vmm.hpp), deadadfc4 nit 5, 3a77926da nit 6, and this report.

Force-added: FIX-REPORT.md sits in the main checkout's info/exclude, as the reports and the review do.
