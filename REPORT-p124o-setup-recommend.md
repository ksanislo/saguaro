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
