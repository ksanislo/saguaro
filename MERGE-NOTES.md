# saguaro/ik2: merge notes (2026-11-22)

Branch `saguaro/ik2` (worktree /data/llm/saguaro-ik2port), merged with `saguaro/qfn-vram` at ea21f5d1 (one conflict,
serve/web/monitor.js, resolved keeping both sides). Since the last merge of this branch (d5d2ed84):

- **Engine:** conversations kept beside the one in the cache (up to 4 in RAM, `SAGUARO_IK2_KEEP` / `_KEEP_GB`), saved to
  and read back from `SAGUARO_IK2_SLOT_DIR` across restarts, rewind to a message's snapshot (c341f273); an expert
  tensor's row check uses its own row length, so 256-value types (IQ3_XXS ...) load on gate_up (702569ea).
- **Quant tools:** tools/ik2/quant_finish.py, bench_model.py (83311970); the one-T4 quant
  /opt/llm/models/Inkling-2-Mini-T4-NVFP4-IQ3_XXS (build-ik2/night/QUANT-NOTES.md) and its config
  configs/Inkling-2-Mini-T4-NVFP4-IQ3_XXS.1card.json, marked experimental.
- **Server:** Monitor facts and the expert-cache hit rate from saguaro-ik2's start / "decode stats" lines, the picture
  encoder on the config's GPU with the most free memory, top_k up to 1000, stdin closed after QUIT (f1b33187);
  monitor.js shows those figures when present, the setup branch's note otherwise.
- **Defaults** (interleaved T4 A/Bs, NEXT-ik2.md): one card `--chunk 2048` + `SAGUARO_IK2_PIN=0`, two cards
  `--chunk 8192`; the repo configs and the setup's layouts (tools/ik2_fit.py) alike; the setup drops a layout's env keys
  (`SAGUARO_IK2_ROOM_MB`, `SAGUARO_IK2_PIN`) when the cards change. Repo configs also set `SAGUARO_IK2_STATS=1` and
  `SAGUARO_IK2_SLOT_DIR=/opt/llm/var/<model>/sessions`.
- **Docs:** INKLING2.md section 10 (served on the T4s, install steps, proposed llama-swap blocks), INKLING2-MINI.md,
  MODELS.md, README, CHANGELOG with the new speeds and the kept conversations.
- **Tools:** serve_check.py `--model`, and no CLAIM line under IK2_NO_CLAIM.

## Tests

- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_inkling2_frontend serve.test_new_engines_ui
  serve.test_models serve.test_server` (+ tools.test_setup_ik2, tools.test_docs_new_engines, tools.test_setup_hy4,
  tools.test_setup_dash, tools.test_maya_quants): 363 tests OK (21 skipped) at 9764be93.
- On the cards (build-ik2c = 795f6c9f; the later commits touch no engine source): ik2_cpu_test passed; t4_check PASSED
  (kernels, 1 card, 2 cards); conv_check PASSED (set aside / back / edit / saved folder read by a new engine, tokens
  alike); the T4 quant loads and answers (t4_check 1 card); serve_check all PASS on 1 card with UD-Q4_K_XL (overnight)
  and with the T4 quant, pictures included. Results: build-ik2/night/final/.

## Not verified

- Two cards at `--chunk 8192` past a 16K prompt (125K was filled at chunk 2048 only); the picture encoder there
  (serve_check at 131072 / 8192: see NEXT-ik2.md for its result).
- Saved conversations through serve/server.py and llama-swap (only the engine-level conv_check); the setup writes no
  SLOT_DIR or STATS yet.
- The token path's KL against the cache path (1.86e-3 mean) is not upstream's 5.7e-5 figure (theirs is tensor-core vs
  scalar experts within one path); the like-for-like run on T4 is queued (build-ik2/night/final/kl-forms.out).
- The T4 quant: no task-level quality check, 2 cards not run, PIN / chunk not swept for it.
- One-card PIN=0: the first short prompt of a load reads ~0.6 s slower.
- Nothing is installed under /opt/llm/saguaro and no llama-swap profile is applied.
