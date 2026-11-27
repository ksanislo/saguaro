# REPORT-p125e-ik2-elastic-1 - Inkling 2 Mini: the experts keep VRAM until a conversation needs it, step 1 (Saguaro 1.2.5)

The plan's "smallest useful step" from ref/REPORT-p125b-ik2-vram.md - (b1) preload the offload tier at the load and
(a1) split the --max-context ceiling from the K/V committed at the load - done, and nothing more. No kernels were
touched. No GPU was run: this box has neither a CUDA-usable GPU nor Inkling weights, so every check here is a
compile, a host-side unit test, or a script meant for Ken's cards.

## 1. What changed

### a1 - the elastic start step (the decision, then the layout, then the announced ceiling)

- **src/ik2/ik2_plan.hpp (new)**: the decision as three pure functions, unit-tested without a GPU:
  `kv_committed(ctx, kv_start)` (SAGUARO_IK2_KV_START, default 32768, 0 = the whole ceiling as today),
  `experts_fit(...)` (the engine's `fits` arithmetic, byte for byte as it was), and
  `elastic_start(fits_all, fits_committed, streaming_taken, ctx, kv_start, two, q4, parallel, tp)`.
- **src/ik2/ik2_main.cu (a1)**: the plan now computes the global layers' K / V twice - at the ceiling (`kv_b`,
  as today) and at the committed start step (`kv_b_start`) - and gets both `fits` verdicts from the helpers.
  Where the ceiling's cache would crowd every expert out of VRAM but the committed start step leaves them room
  (the mortar's 262144 row on a 16 GB card, and any resident-or-nothing card), the engine takes the elastic
  start: the plan and the layout use the commit (`Vi.kv_cap = g_kv_run` for the global visits), the freed GiB
  goes to the expert pool, and the drafter's context rows are sized for the commit too. It is not taken where
  KV streaming ran (its pinned host rows already hold the whole ceiling), on q4 rows (their own VRAM/host
  split), with two GPUs, `--tp`, `--parallel`, EP, or where the environment refuses the resident path
  (SAGUARO_IK2_OFF=1, SAGUARO_IK2_DENSE, SAGUARO_IK2_FAST=0). The engine says
  `NOTE context: the global layers' K / V is laid out for the first 32768 positions of the 262144 ...` and
  announces the commit everywhere a conversation's ceiling is said: `INFO ... ctx=`, `READY <commit> stop`,
  the legacy `@READY ctx`, the GEN guard's "the prompt and max_tokens pass the context (...)", the CLI's
  "prompt + gen past the context", and STAT's `kv_cap=`. `SAGUARO_IK2_KV_START=0` (or >= --max-context)
  keeps today's layout byte for byte.
- **tools/ik2_fit.py (a1 on the PC side)**: `KV_START = 32768` and `kv_committed()` mirror the engine;
  `plan()` returns `{"resident": true, ..., "kv_committed": <commit>}` for exactly the cards the engine takes
  the elastic start on (the ceiling does not fit, the commit + every expert + the reserve does, and nothing
  streams), leaves every other answer word for word as it was, and `draft_fits()` sizes the drafter's rows for
  the commit on such a card. `verdict()`'s fast line says "every expert in VRAM; the cache is laid out for 32K
  of the 256K ceiling - chats run to 32K until it can grow"; the plain one is unchanged.

### b1 - the offload tier starts full, never empty

- **src/ik2/ik2_main.cu (b1)**: an offload-path start with no SAGUARO_IK2_WARM profile and no EP used to copy
  nothing into its VRAM tier ("offload: 0 of 3072 experts in VRAM", every cold expert a host fetch until the
  dynamic cache learned). It now preloads the pool to its full `n_active` slots with the resident loop's own
  `EC.get` copy, **round-robin over the layers of each pool** so every layer keeps its share instead of the
  first layers eating it all, and prints `tier: %d of %d experts (%.1f GiB) copied at the load, every layer
  its share (no picks file - SAGUARO_IK2_WARM would price them; the cache relearns from here)`. A readable
  warm profile still wins (its picks price the pools better); a profile that does not read ("warm: the cache
  starts empty") preloads too. The copies stay out of the request miss/bytes counters exactly as the warm
  fill's do, and `++EC.tick` follows the warm loop's rule so the first prefetch ticks cannot evict the tier.
  The dynamic device cache picks the tier up unchanged (its `dcache_loc` is copied from `EC.loc` after the
  fill, and preloaded slots start at freq 1 exactly like warmed ones). On the mortar's 262144 row on a card
  where a1 does not fire (experts + commit still do not fit, e.g. UD-Q4_K_XL on a 16 GB card), this alone
  turns 0 of 3072 into roughly "as many as the ~4.5 GiB pool holds" (~1200) - the plan's "never 0".

## 2. What happens to a conversation that is really long (said exactly, as the plan asked)

Step 1 cannot grow the committed K/V by taking VRAM back from the experts - that is step 2's VMM work (a2/a3/a4/b4
of ref/REPORT-p125b-ik2-vram.md), which needs kernels' help to be safe. So on a start that takes the elastic
step:

- the engine never lays out more global-layer K/V than `min(--max-context, SAGUARO_IK2_KV_START)` positions, and
  it says so by announcing that number as the conversation's ceiling (`READY <commit>`). The server's
  `max_context` follows the engine's READY, so a conversation that grows past the commit meets the API's clean
  `maximum context length is 32768 tokens; prompt ...` refusal (serve/server.py's ContextOverflow, the same
  answer a 32768-ceiling config gives today) - never a crash, never a wrong answer: the engine cannot be asked
  to write a position it did not lay out. A CLI prompt past it dies with "prompt + gen past the context".
- Compared with today's behaviour for a truly long conversation at a 262144 ceiling on a card where a1 fires -
  today that conversation ran on the offload path (whole ceiling's cache in VRAM, experts fetched from RAM at
  ~71 tok/s on the RX 9070 XT); after step 1 it runs at full resident speed to the commit and is refused beyond
  it until step 2 grows the cache into the ceiling. This is the one place step 1 shrinks below today, and it is
  deliberate: the same rule the plan asked for ("size the decision from the K/V a conversation USES at start,
  not from --max-context") cannot both keep the experts and keep an unused ceiling usable without the growth
  machinery. The card test prints exactly what a card answers past the commit.
- Everything else is not below today: where the elastic start is not taken (tight cards, streaming cards - the
  24 GB card at the trained 1M keeps streaming its cache with every expert resident - q4 rows, two GPUs,
  --parallel, --tp, EP, forced offload), the behaviour is today's, byte for byte, plus b1's preloaded tier
  instead of an empty one. And `SAGUARO_IK2_KV_START=<ceiling>` (or `0`) restores the whole ceiling on any card
  - today's layout with the tier now preloaded, i.e. strictly above today for long conversations.
- The Models page says the commit in the fast line, and the setup's layout for a 24 GB one-card box now
  recommends --max-context 262144 (the ceiling costs ~0 until used); conversations there run to 32768 until
  step 2.

## 3. Files

- new: src/ik2/ik2_plan.hpp, tests/ik2_plan_test.cpp, tools/test_ik2_kv_start.py, tools/card_tests/ik2_kv_start.sh
- changed: src/ik2/ik2_main.cu (a1 + b1), CMakeLists.txt (register ik2_plan_test), tools/ik2_fit.py,
  tools/test_setup_ik2.py
- commits, one per step: step 1 helpers + test; step 2 engine a1; step 3 engine b1; step 4 the PC-side mirror +
  tests; step 5 the card script; this report.

## 4. Tests run, one at a time, and their results

- `./scratch/build-ik2-elastic/ik2_plan_test` (also `g++ -std=c++17 -Wall` direct): **ok** - 21 checks: the
  commit arithmetic, the mortar-card fits figures (11.45 + 1.8 + 1.5 + cache in 15.9 GiB free), and every
  elastic_start gate (ceiling-fits, too-small, commit==ceiling, start 0, streaming, two GPUs, q4, parallel, tp).
- `python -m unittest tools.test_ik2_kv_start` - **11 OK**: kv_committed; the 24 GB 262144 row plans resident at
  the commit; a 32768 ceiling never gets the key; the tight 16 GB card keeps its offload tier 2.7; the
  streaming 16 GB card keeps streaming; the trained 1M fp8 on 24 GB plans resident at the commit (NVIDIA and
  AMD); two cards still split at 131072; recommended_context reaches 262144 on an elastic card; the fast/commit
  verdict lines; the offload card's line unchanged.
- `python -m unittest tools.test_setup_ik2` - **58 OK** (two assertions followed the intended flip: a 24 GB
  one-card layout now recommends 262144; plan(24, 1048576, fp8) is resident at the commit - with the
  drafting_off_reason/draft_fits rows kept green through the commit-sized drafter room).
- `python -m unittest tools.test_docs_new_engines` - **7 OK** (docs still say what the layouts do).
- `python -m unittest serve.test_models` - **64 OK**; `serve.test_new_engines_ui` - **38 OK**;
  `serve.test_model_api` - **28 OK** (the consumers of plan()/verdict()).
- Build: `cmake -DCMAKE_CUDA_ARCHITECTURES=75 -DMAYA_ENABLE_CUDA=ON` into **scratch/build-ik2-elastic** (CUDA
  13.1, fresh dir) and `ninja saguaro-ik2 ik2_plan_test`: **clean** (only the two pre-existing nvcc warnings,
  Shards::num and g_apu on CUDA). No GPU run anywhere.
- tools/card_tests/ik2_kv_start.sh: `bash -n` clean; **not executed** - no GPU and no Inkling weights here.
  It is for the RX 9070 XT box: 32768 vs 262144 load lines, the announced ceiling, a 20000-token read each,
  and the answer to a past-the-commit ask, with the P124t 2 % rule as its PASS.

## 5. Open questions

1. **The announced ceiling** (ref/REPORT-p125b-ik2-vram.md open Q 3, sharpened): an elastic start says
   `READY 32768` under a 262144 config, so the dashboard's context shows 32768 and the slider's 262144 reads
   back as 32768 after a restart. Right, or should the engine announce the ceiling and ERR past the commit
   (worse UX, keeps the figure), or should serve show "ceiling 262144 / running to 32768" as one line?
2. **mem_vs_ctx (P124t)**: an elastic card now prints "resident: every expert in VRAM (3072)" at both contexts
   and passes the 2 % rule; a card where a1 cannot fire prints resident-vs-offload-N and will (rightly?) FAIL
   the rule until step 2. Should tools/mem_vs_ctx.py treat a `NOTE context:` line as the explanation of the
   placement change instead?
3. **The drafter's room on the borderline**: the engine sizes the drafter for the commit and measures (it can
   still leave it out); ik2_fit's draft_fits sizes it for the commit too but through plan()'s own sums, which
   are looser than the engine's measured fr0. A T4 box at 262144 would show whether they agree.
4. **1M configs**: the model now plans the trained 1M resident at the commit on a 24 GB card (fp8) where it
   said offload before - the engine's own streaming decision (scratch + sliding + reserve counted against real
   free VRAM) may keep such a card streaming instead. Which is the product answer Ken wants for the trained-1M
   configs: keep the ceiling honest, or cap `recommended_context`'s elastic flip at, say, 4x the commit until
   step 2 lands?
5. **HIP**: elastic is untested on gfx1201 at runtime (it only allocates less and copies more at the load, so
   it should be safe, but the offload-note branch and APU fr0 clamping deserve a run).
6. **b1's load-time cost**: ~7 s per 11.45 GiB of tier copies at the load (the resident line's figures). Worth
   a SAGUARO_IK2_TIER_PRELOAD=0 kill-switch, or is a full tier worth it on every cold start?
7. **EP and two-GPU tier fills**: b1 fills GPU 0's tier round-robin, and two_setup then gives GPU 1 what is
   left - correct, but unmeasured against the warmed split. A shipped profile still prices it better.
8. **`--serve-ik2` legacy line set**: the old `@READY ctx` protocol now also carries the commit; anything
   outside this tree parsing it (tools/ik2/*.sh) should be checked before release.
