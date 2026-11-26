# REPORT - P124o: the setup recommends a Qwen3.8-Flash-Next (QFN) quant on lower-end hardware

## Step 1 - where the setup decides the recommended model and quant (the rule before this change)

The model question and its "recommended for this PC" marks:

- `family_fit` - saguaro.py:8580 - the model the setup recommends for a PC's cards when nothing was set up before:
  - **ik2** (Inkling 2 Mini) where every card is under 24 GB and Inkling's engine exists for the backend: on AMD one
    RDNA 3/4 card (`IK2F.amd_unsupported(cards[:1]) is None and len(cards) == 1`, saguaro.py:8591), on NVIDIA one or
    two cards with `IK2F.usable(cards)` (saguaro.py:8592).
  - **hy4** where every card is under 24 GB, there are >= 4 of them, `HY4F.equal_group` finds four equal ones and
    VRAM + RAM >= 68 GB (saguaro.py:8594-8596).
  - **glm** on the HIP backend when VRAM + RAM hold GLM's smallest quant (saguaro.py:8597-8598).
  - **None** elsewhere - no row gets the note and the model question's default stays GLM (`fit or "glm"`,
    saguaro.py:2223). The QFN row said "new" (saguaro.py:2247); the note "recommended for this PC" existed only on the
    glm / hy4 / ik2 rows (saguaro.py:2227, 2248, 2249).
- The QFN quant menu `choose_qfn_quant` - saguaro.py:7715 - always put "recommended" beside `QFN_DEFAULT_QUANT` =
  "ud-q4_k_xl" (saguaro.py:725, 7726, 7731) whatever the PC had; a quant that cannot run is listed with
  `qfn_quant_fit`'s reason (saguaro.py:7693, pricing `qfn_fit.card_min_gib` / `ram_min_gib`, tools/qfn_fit.py:225, 233)
  and refused if picked.

What each of the plan's machines gets today (checked by running `family_fit` with fake cards, WIN False):

| machine | family_fit today | the model the question defaults to |
|---|---|---|
| (a) 1 x Tesla P40 24 GB + 64 GB | `None` - 24 GB is not "under 24", so no small-card branch | GLM-5.3-Flash (no recommendation note on any row) |
| (b) 2 x Tesla T4 16 GB + 256 GB | `ik2` | Inkling 2 Mini, "recommended for this PC" |
| (c) RX 9070 XT 16 GB + 96 GB (AMD) | `ik2` | Inkling 2 Mini |
| (d) 4 x Tesla T4 16 GB + 256 GB | `hy4` | Hy4-Flash Maya-S, "recommended for this PC" |
| (e) one 12 GB card + 32 GB | `ik2` | Inkling 2 Mini |

(Inkling 2 Mini's about-line even says "The model for one or two 16 GB cards", saguaro.py:8634.)

## Step 2 - what Strata itself recommends, per size, inside engines/qfn (grep "recommend", "IQ3_S", "UD-Q4", "Coder", "RAM")

engines/qfn carries **no per-size quant table**; what the greps find is Strata's fit rule and its shrink advice:

- `engines/qfn/strata_control/gpu_fit.py:1-15` - the rule (owner decision): "1. First try the model on ONE card (the
  largest) plus system RAM, the normal hybrid mode. 2. Only when that does not fit, or would be too tight, offer to
  add the second card as a layer split. 3. ... a recommended one is pre-selected (recommend, never force)". Exactly
  one option is recommended (gpu_fit.py:219-251): the single largest card when the model fits it with RAM, else the
  split; "fits / tight / spills" by the RAM left beyond the experts (FITS_SPARE_RAM_GB, gpu_fit.py:30, 132-137).
- When RAM/VRAM runs out the engine says to take a SMALLER quant, not another model: the OOM line "close other
  programs or use a smaller model (Q2_0 / IQ2_XS)" (engines/qfn/serve/server.py:1504-1509) and the tight-RAM warning
  "run START-HERE --setup and pick a smaller size (Q2_0 / IQ2_XS)" (serve/server.py:8456-8461).
- Measured facts naming the quants per card size: Coder IQ1_M on a V100 32 GB keeps 94.8 % of the experts in the GPU
  cache (src/program/generate.cpp:11500); Coder is the pruned variant, fewer than 512 experts (generate.cpp:4111,
  4128); IQ3_S is the multi-card rigs' size (serve/replicas.py:3 "IQ3_S on two R9700s"; generate.cpp:3286 its prefill
  numbers).
- Ours, to line with it: README.md:79-81 "the setup names what fits your PC ... UD-Q4_K_XL is the default ... the
  smaller GSQ-RCO and Coder quants suit one 16-24 GB card", and the P40 box (24 GB, 32 GB RAM) ran GSQ IQ3_S at
  ~15-20 tok/s (tools/qfn_fit.py:319, README's speed table).

So "starting from what upstream Strata recommends per VRAM/RAM size" reads as: take the largest quant this PC runs,
shrink when its memory does not hold it - the plan's ladder UD-Q4_K_XL > IQ3_S > Coder IQ1_M. Q8_0 and CYBER-FROST's
own quant are accuracy picks, never a fit-based default; they stay out of the ladder.

## Step 3 - the new rule (saguaro.py)

- `QFN_RECOMMEND = (QFN_DEFAULT_QUANT, Q.GSQ_IQ3_S, Q.GSQ_CODER_IQ1_M)` - saguaro.py:728 - the ladder, the largest
  that runs first; Q8_0 and CYBER-FROST's own quant are accuracy picks and stay out of it.
- `qfn_recommended_quant(cards, ram, backend)` - saguaro.py:7719 - the largest ladder quant whose `qfn_quant_fit`
  (the setup's own fit check, experts streamed: refused only under the true minimum) passes, priced on the cards the
  QFN setup would take from these (AMD: the largest RDNA GPU that is not an APU - its setup runs on one GPU; NVIDIA:
  `qfn_fit.best_gpus` over the cards at MIN_CC and newer); None where none of the three runs at all.
- `family_fit` - saguaro.py:8619 - tries **QFN first** where the QFN engine exists for the backend, every card is
  24 GB or less (the line GLM's own Maya-S24 note uses: "cards of 24 GB or less", so the P40 box is now covered),
  NVIDIA one or two cards / AMD its one GPU, and `qfn_recommended_quant` finds a quant that runs. Inkling's branch
  stands and is now reached only where no QFN quant runs - Inkling stays offered, not the default. The four-equal-
  small-cards **Hy4** pick, the AMD **GLM** pick (two AMD cards stay GLM's - both small engines use one GPU) and the
  no-recommendation path (GLM / DS4 / MM3 for the bigger PCs) are untouched.
- Reason lines, kept and updated: the model question's QFN row prints
  "recommended for this PC (UD-Q4_K_XL)" - the quant of it this PC would get (saguaro.py:2253); the quant menu marks
  and defaults to the largest quant that RUNS instead of the hardcoded UD-Q4_K_XL (saguaro.py:7733-7761,
  `rec` over QFN_RECOMMEND); Inkling's about-line now says it is "the model for one or two 16 GB cards where
  Qwen3.8-Flash-Next does not run" (saguaro.py:8683).

| machine | before | now (family_fit / quant) |
|---|---|---|
| (a) 1 x P40 24 GB + 64 GB | None -> GLM default | qfn / ud-q4_k_xl |
| (b) 2 x T4 16 GB + 256 GB | ik2 | qfn / ud-q4_k_xl |
| (c) RX 9070 XT 16 GB + 96 GB | ik2 | qfn / ud-q4_k_xl |
| (d) 4 x T4 16 GB + 256 GB | hy4 | hy4 (unchanged, the measured Maya-S pick) |
| (e) one 12 GB card + 32 GB | ik2 | qfn / ud-q4_k_xl |
| one 12 GB card + 13 GB (no QFN fits) | ik2 | ik2 (all three quants refuse: RAM < 14.05 GB) |
| one 8 GB card + 14.5 GB | ik2 | qfn / gsq-coder-iq1_m (the ladder down to what runs) |

## Step 4 - tests, files, results

- tools/test_setup_qfn.py: new class `TheRecommendation` (10 tests): the five profiles above, the ladder
  (UD-Q4_K_XL > IQ3_S on 15.4 GB of RAM > Coder IQ1_M on 14.5 - qfn_fit's minimums 15.9 / 14.93 / 14.05 GiB), the
  no-QFN machine still getting Inkling, bigger PCs keeping their default, and the quant menu's / model question's
  recommended+default lines. Module docstring updated.
- tools/test_setup_ik2.py: Picker - `test_qfn_wins_on_one_or_two_small_cards_now` (the Inkling-recommended case now
  needs a machine under every QFN minimum: `test_recommended_where_no_qfn_quant_runs`); rows() takes a RAM figure.
- tools/test_setup_amd.py: `test_amd_gets_a_recommendation` (9070 XT -> QFN, note "(UD-Q4_K_XL)"),
  `test_nvidia_recommendations` (2x T4 -> QFN, 4x T4 -> Hy4) and `test_family_fit` updated.
- Files changed: saguaro.py (rule, quant menu, reason lines), tools/test_setup_qfn.py, tools/test_setup_ik2.py,
  tools/test_setup_amd.py, this report. engines/qfn, serve/qfn_engine.py and tools/qfn_fit.py were not touched; no
  serve/ file was touched, so serve/qfn_engine.py needed no extraction.
- Run one at a time with /opt/llm/saguaro/.venv/bin/python (no GPU, no network):
  tools.test_setup_qfn 157 OK (1 skip - the pre-existing ./ref live-config parity guard), tools.test_setup_ik2 57 OK,
  tools.test_setup_amd 35 OK, tools.test_setup_hy4 33 OK.

## Open questions

- The Models page (serve/model_pull.py fit_notes, serve/test_new_engines_ui.py) still marks Inkling 2 Mini
  "recommended" on one or two 16 GB cards and says "Hy4-Flash ... Inkling 2 Mini is the model for this PC"; the plan
  scoped the rule to the setup, and "run only the setup test modules you touched" kept serve/ out. Should the page's
  recommended flag move to the QFN rows too (QFN-first, ladder-sized)?
- 4 x T4 + 256 GB keeps the Hy4 Maya-S pick (the plan named only Inkling as the not-the-default model, and Hy4's
  four-card layout is its own measured one; QFN's own live 4x T4 configs say that machine runs both). Flip
  (d) - and QFN's 3-4-card split territory in general - when Ken says.
- The gate counts a PC "lower-end" at every card 24 GB or less (so 1-2 x 4090/3090 boxes now default to QFN, and a
  P40 box gets its first-ever recommendation); single 48 GB+ cards and 3-card NVIDIA boxes keep the GLM default.
- Docs left alone (out of the plan's scope, tests unaffected): docs/AMD_SAGUARO.md:14 still says the setup
  recommends Inkling "for one 16-23 GB RDNA 3 / 4 card", and docs/INKLING2-MINI.md / README's recommendation lines
  now trail the setup's rule.
