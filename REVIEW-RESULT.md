# MANAGER: FIX-FIRST (re-merge only) - 2026-11-27 15:45 PST
Your round-2 review is MERGE-WITH-NITS, but it notes this branch merged P125j at its round-4 state; P125j's branch
(saguaro/q1-p125j-layer-pause) has since passed its round-6 review (MERGE). Do this, nothing else:
1. `git merge saguaro/q1-p125j-layer-pause` again (bring in P125j's later rounds); resolve keeping both behaviours.
   FIX-REPORT: the conflicts (if any) and what you kept.
2. Fix the review's REPORT nit (the "byte-identical" claim at conversation_state.cpp:257).
3. Build QFN only (tools/qfn/build.sh scratch/build-p125k-merge2, QFN_GGML_DIR=/data/llm/strata-main/third_party/llama.cpp);
   run park_policy_test and the tests both branches added. No GPU runs.
