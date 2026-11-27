# REVIEW-RESULT - P125e (PLAN-p125e-ik2-elastic-1 / REPORT-p125e-ik2-elastic-1)

MERGE-WITH-NITS

Reviewed `git diff d210adc76e47b9883484a6b20b32372ff98a0ce1..HEAD` (6 commits, 9 files). Re-ran the plan's tests
in this worktree, one at a time: `scratch/build-ik2-elastic/ik2_plan_test` ok (ninja: no work to do - binary
current with HEAD); `tools.test_ik2_kv_start` 11 OK; `tools.test_setup_ik2` 58 OK; `tools.test_docs_new_engines`
7 OK; `serve.test_models` 64 OK; `serve.test_new_engines_ui` 38 OK; `serve.test_model_api` 28 OK;
`bash -n tools/card_tests/ik2_kv_start.sh` clean; no GPU run. The build dir is configured
`CMAKE_CUDA_ARCHITECTURES=75`, `MAYA_ENABLE_CUDA=ON`, and `ninja saguaro-ik2` is up to date with the committed
source; `scratch/build-ik2-elastic.log` shows only the three pre-existing nvcc warnings
(Shards::num, g_apu, kFaRawQ4), so the report's "clean" claim holds at HEAD.

## Findings, by severity

1. MEDIUM - elastic start: three hooks still bound positions by the ceiling `ctx` instead of `g_kv_run`, so they can
   touch global-layer K/V above the committed layout.
   - src/ik2/ik2_main.cu:10139 - the SAGUARO_IK2_DRAFT_IDS filter is `v.size() + 256 < ctx`; with elastic (g_kv_run
     = 32768, ctx = 262144) a line of, say, 40 000 ids passes, and the measurement then writes K/V at positions
     above the rows allocated at src/ik2/ik2_main.cu:8465/:8501 and decodes from `P = ids.size()`.
   - src/ik2/ik2_main.cu:10180 - `decode_greedy(..., ctx, ...)` bounds `P < ctx` and `room = ctx - P`
     (src/ik2/ik2_main.cu:7116/:7119): generation runs to the ceiling, not the commit.
   - src/ik2/ik2_main.cu:11308 - SAGUARO_IK2_BLOCK_WARM's gate is `prompt.size() + kBlockMax <= ctx`.
   What the over-commit writes do: the K/V-write epilog is called with `Ly.kv_cap` for the full layers
   (src/ik2/ik2_main.cu:3404/:3414/:3435; the wrap rule is src/ik2/ik2_kernels.cuh:289/:292), so writes past the
   commit wrap `% kv_cap` and silently overwrite the cache's first rows; the decode attention for full layers is
   called with ring 0 (src/ik2/ik2_main.cu:3410/:3430/:3449) and reads `[0, P)` linearly - past the allocated
   rows (an out-of-bounds read). The drafter's context rows are commit-sized too
   (src/ik2/ik2_main.cu:5462/:9174/:9179).
   Reachability: guarded request paths are safe - the GEN guard uses g_kv_run (src/ik2/ik2_main.cu:10732/:10736)
   and the CLI die uses g_kv_run (src/ik2/ik2_main.cu:11205), and serve's own generated SAGUARO_IK2_DRAFT_IDS file
   is short doc questions (serve/server.py:2906-2935, :2992-2995). But the hooks are operator-settable, and a
   small SAGUARO_IK2_KV_START makes even the auto-measure reachable. Fix: use `g_kv_run` at :10139, :10180 and
   :11308 (the elastic design - "never writes past what it laid out", src/ik2/ik2_plan.hpp:34-35 - should hold
   for these paths too).

2. LOW - the CLI die message changed for every start, not just elastic ones: "prompt + gen past --ctx" became
   "prompt + gen past the context" (src/ik2/ik2_main.cu:11205). With SAGUARO_IK2_KV_START=0 the path is otherwise
   byte-identical, so this is a visible string change with the feature OFF. Harmless; keep it or restore the old
   wording when not elastic.

3. LOW - dead variable in the b1 fill: `const double tw = now();` at src/ik2/ik2_main.cu:9259 is never read (the
   warm loop prints its elapsed time, src/ik2/ik2_main.cu:9330; the preload's "tier:" line at :9276 does not).
   nvcc does not flag it, but it is a copy-paste leftover - print the seconds like warm does, or drop it.

4. LOW - PC-side mirror: tools/ik2_fit.py:154-156 - plan_two() now returns the elastic answer ("resident",
   "kv_committed") for two-card calls, but the engine never takes elastic with two GPUs
   (src/ik2/ik2_plan.hpp:40-41; the gate src/ik2/ik2_main.cu:8048 passes g_two.want). No wrong setup ships:
   layout() falls through to the one-card "--gpus 1" branch in that case (tools/ik2_fit.py:415-434) and there the
   engine does take elastic - but plan_two's own contract now disagrees with the engine for direct callers; drop
   the key in plan_two (or say it in the docstring).

5. INFO - an elastic start still sizes two ctx-based buffers by the ceiling: the attention logits/probabilities
   scratch `W.lg_cap = 8 * c.H * ctx + ...` (src/ik2/ik2_main.cu:8924-8926) and the pinned host staging
   `W.T_host` (:8984). Safe (the scratch is more than the commit needs, never overflowed), but ~200+ MiB more
   VRAM than a true 32K config at a 262144 ceiling; a candidate for step 2 along with the growth machinery.

6. INFO - b1 is on for every plain offload start, not gated by SAGUARO_IK2_KV_START: that is the plan
   ("never 0"), and the report says so (:72-76 of the report) - recorded here so "feature OFF byte-identical" is
   read as "a1 OFF byte-identical, plus b1's full tier and its new `tier:` line" (src/ik2/ik2_main.cu:9334-9335).
   GPU 1 of a two-card split gets no preload (the `d == 0` gate, :9334) - matches report open Q 7; no kill-switch
   yet - open Q 6.

## What was checked and is sound

- Feature OFF (SAGUARO_IK2_KV_START=0 or >= --max-context): `kv_committed` returns the ceiling
  (src/ik2/ik2_plan.hpp:21-23), the `kv_b_start` loop is a byte-for-byte mirror of the `kv_b` loop
  (src/ik2/ik2_main.cu:7878-7888 vs :7893-7904, q4 branches included), the elastic gate cannot fire
  (src/ik2/ik2_plan.hpp:40), and `g_kv_run = ctx` at :7479 makes every ctx -> g_kv_run substitution evaluate
  identically (:8190/:8195, :8465, :9174/:9179, :10228, :10253, :10255, :10321, :10732/:10736, :11205).
  `experts_fit` reproduces the old inline expression exactly (src/ik2/ik2_plan.hpp:27-29). Only finding 2 is an
  OFF-visible change.
- Elastic gate ordering: `g_par` (:7468), `g_two.want` (:7561), `g_ep` (:7722) and `g_kv_streaming` (:8031) are
  all set before the gate at :8048-8050; the block sits inside the one-GPU plan section (:7870). The KV-streaming
  rewrite of `kv_b` (:8010) can only happen when g_kv_streaming is set, which the gate excludes - no interaction.
- b1 fill matches the warm fill's rules: same `EC.get(..., EC.tick, g_cs, nullptr, true, true)` call
  (:9270 vs :9312), the same `++EC.tick` rule (:9275 vs :9317), the same save/restore of the request counters
  (:9260/:9280 vs :9298/:9332); the dynamic device cache copies `EC.loc` after the fill
  (src/ik2/ik2_main.cu:9480-9483, after :9336) and `pin_later` skips the tier-held experts (:9341) - the report's
  claims check out. Round-robin: the inner loop is over the pool's layers (:9267-9272), so each layer gets its
  share; unreadable profile preloads too (:9295), warm picks still win.
- Locks/threads/stdout: all new prints and the preload lambda run single-threaded at the load, before the stdin
  thread starts (:10259+); each new line is one printf + fflush; EC has no concurrent access during the load;
  cudaStreamSynchronize before the "tier:" line. No lock is taken anywhere in the new code.
- Generated file: the diff touches no generated file - serve/qfn_engine.py is not in the diff at all (9 files:
  CMakeLists.txt, ik2_main.cu, ik2_plan.hpp, ik2_plan_test.cpp, ik2_fit.py, test_ik2_kv_start.py,
  test_setup_ik2.py, card script, report); nothing to regenerate.
- Tests exercise the change: ik2_plan_test holds all three helpers to arithmetic (21 expects, incl. the mortar
  figures and every gate); tools/test_ik2_kv_start.py covers the ik2_fit mirror (the 262144 mortar row, tight and
  streaming cards unchanged, 1M-fp8 flip); the two flipped assertions in tools/test_setup_ik2.py are the intended
  recommendation flips, said in the step-4 message. The engine's elastic wiring is not host-testable; the card
  script (grep patterns match the real lines: src/ik2/ik2_main.cu:9570/:9632 vs tools/card_tests/ik2_kv_start.sh:100,
  the NOTE regex :157 matches :8056-8060) is the plan's answer and was left unrun here, as the plan requires.
- Paths/secrets: the card script works only under ROOT-relative paths and its own `scratch/ik2-kv-start-card`
  folder (tools/card_tests/ik2_kv_start.sh:31-40), copies the config with `api_key` dropped (:123), and keeps the
  install's own config/log/sessions untouched; no absolute paths outside the repo and no secrets in any diff.
- Announce-everywhere claim: INFO ctx (:10228), READY (:10253/:10255), STAT kv_cap (:10321) with kv_total summed
  from kv_cap (:10220-10221), the GEN guard (:10732-10736), serve/server.py takes max_context from READY
  (serve/server.py:1515-1517, /api/context at :3792, ik2_log_facts at :1004-1030) - a past-the-commit ask meets
  the clean ContextOverflow refusal (serve/server.py:851), as the report says.

## MANAGER

Plan's "small commits, one per step": 6 commits match the plan's list exactly, each touching only its step's
files (helpers+test / engine a1 / engine b1 / PC mirror+tests / card script / report); no Co-Authored-By,
Claude-Session or Assisted-by trailers; the report commit states its force-add (REVIEW-RESULT.md is likewise in
the main checkout's info/exclude). No GPU was run by the coder or by this review.
