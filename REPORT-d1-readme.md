# D1 Report: README.md - supported models, hardware, readability (docs only)

## What changed

**README.md restructured** (2 commits; only README.md and this report were touched):

- `7a5bb398d` - the restructure:
  - **Opening**: kept the tagline + HF download badges; the two lead paragraphs now say plainly what Saguaro is
    (runs the six MoE models on your own GPUs; one server/dashboard/API; self-tuning) and that it is an
    independent fork uniting Project Maya's engines and Strata's `saguaro-qfn`, pointing to Credits/NOTICE.md.
  - **Quick start**: `git clone` + `./saguaro.sh` (first run sets up and starts; later runs start; `--setup`
    re-sets-up; `./setup.sh` is the same command; Windows `START-SAGUARO.bat`).
  - **New "Supported models" table**: model | engine | quants (default marked) | size | needs | NVIDIA | AMD |
    pictures | `--model` id, one row per family: `saguaro-glm`, `-ds4`, `-mm3`, `-hy4`, `-ik2`, `-qfn`.
    Qwen3.8-Flash-Next shows UD-Q4_K_XL (default), Q8_0 and CYBER-FROST-3.8 Q5_K_M as a `saguaro-qfn` quant
    with a note that its old Maya-engine ids resolve to it - the removed Maya Qwen path is not presented as
    current anywhere. Notes under the table carry the setup's recommendation rule, the 4x Tesla T4 QFN split +
    `--check`, and the community-quant sentence, all linking to docs/MODELS.md.
  - **"What you need" (= the plan's Hardware section; heading kept because docs/WINDOWS.md links
    `README.md#what-you-need`)**: NVIDIA compute capability as the code states it (validated 7.0+, Pascal
    builds for GLM/Inkling only), 1-16 cards with 1-4 for QFN, AMD RDNA 3 / 4 + Strix/Gorgon Halo with ROCm 7
    and exactly which engines run vs only compile there, Windows and Mac engine reach. The word "experimental"
    no longer appears anywhere in the README; each case is stated as what was measured ("its GPU self-tests
    pass on an RX 9070 XT", "compiles for HIP (gfx1201 checked); has not run on an AMD GPU yet", "run once on
    a PRO V620", Mac "unit-tested on GitHub's Macs, not run with the real model").
  - **New "Updating" section**: dashboard Settings > About and updates / Update button, or `git pull` +
    `./setup.sh`, with the releases link.
  - Kept intact and re-wrapped (all lines of prose <= ~120 cols): Quality, the three speed-benchmark tables
    (byte-identical rows, spliced from the previous revision), Using it, Install, Something went wrong?,
    Support, and the full Credits/licences section (word-for-word; only line re-wrapping at existing spaces).
  - Anchors preserved: `<a name>` tags kept on the model and Using-it blocks; external links into the README
    (`#what-you-need`, `#something-went-wrong`, `#how-fast-is-it`) all still resolve.
- `ecd2b2a5e` - final pass: dropped two figures no repo doc or manifest states (see "Left out" below); split
  one over-long credits line.

Facts working file: `./scratch/readme-facts.txt` (per model + per platform, every figure with its source).
Note: `scratch/` is gitignored in this repo, so the Step-1 deliverable could not be committed (no commit made
for Step 1; no empty commits created).

## Files

- `README.md` - rewritten structure (342 -> 355 lines); credits/benchmarks kept verbatim.
- `REPORT-d1-readme.md` - this report.
- `scratch/readme-facts.txt` - local only (gitignored).

## Tests run (plan: none to run - docs only; these are my own verifications, run one at a time)

1. Link/anchor check over README.md (script in ./scratch): 59 local link targets -> **all files exist and all
   `#anchors` resolve** (GitHub-heading slugs + `<a name>` tags, checked inside docs/SETTINGS.md, SETUP.md,
   MODELS.md, AMD_SAGUARO.md).
2. External-anchor check: `docs/WINDOWS.md`, `docs/DGX_SPARK*.md`, `CONTRIBUTING.md` link to
   `README.md#what-you-need`, `#something-went-wrong`, `#how-fast-is-it` -> **all present** in the new README.
3. `grep experimental README.md` -> **0 matches**; `grep qwen4exp` -> 0; CYBER-FROST appears only as the
   saguaro-qfn quant (table row, note, Credits).
4. Line-length check: **no prose/markup line over 120 chars** except Markdown table rows and the
   `<a name="...">` anchor markup (unavoidable: one logical table row per line, and invisible anchors).
5. Table well-formedness: models table 8 rows x 9 columns, hardware table 9 rows x 2 columns, uniform pipes.

## Numbers in the README and their source

| Figure | Source in repo |
| --- | --- |
| 321 B / ~18 B active / 1M ctx (GLM) | docs/MODELS.md "GLM-5.3-Flash" |
| GLM quants 73.4 / 94.7 / 96.5 / 116 / 116 / 156.3 GB; S24 ~1.5 GB more of a 24 GB card, decode +14%; XS for 64 GB + 24 GB @128K; L 99.2% | docs/MODELS.md GLM table |
| GLM EXL3 125.3 / 165.2 GB | docs/MODELS.md "Community quants of GLM-5.3-Flash"; SETUP.md `--model` |
| 748 B; 277 / 340 GB with Engram tables; ~230 GB pool | docs/MODELS.md DeepSeek table; saguaro.py `DS4_MEMORY_GB = 230`; docs/SETUP.md first-run |
| 428 B / 23 B; 93 / 122 / 197 GB; ~250 GB pool | docs/MODELS.md MiniMax table; saguaro.py `MM3_MEMORY_GB = 250`; docs/SETUP.md |
| 220 B; 44.8-89.5 GB + shared 63.4 GB table; ~32 tok/s 4x T4 (clocks locked) | docs/MODELS.md Hy4 table; tools/hy4/download.json file bytes; docs/SETUP.md Hy4 bullet |
| 26 B, 8 of 128 experts; 18.3 GB with tower; ~40 / 51 / 71 tok/s | docs/MODELS.md Inkling section (measured 2026-11-22); tools/ik2/download.json bytes |
| 125 B, 10 of 512; 111 / 177 / 112.5 GB; 48 layers; 1-4 cards; 16 GB + ~96 GB RAM | docs/MODELS.md QFN table; tools/qfn/download.json (`needs`, sizes); tools/qfn_fit.py (`MAX_GROUP=4`); docs/SETUP.md QFN bullet; saguaro.py `qfn_about`/`qfn_quant_fit` |
| V100 / RTX 20 validated (CC 7.0); Pascal builds for GLM + Inkling only, not validated; CC 70 for DS4/MM3/Hy4/QFN | saguaro.py `MIN_CC=60`, `VALIDATED_CC=70`, `DS4_MIN_CC`, `MM3_MIN_CC`; tools/hy4_fit.py, tools/ik2_fit.py, tools/qfn_fit.py `MIN_CC=70`; Pascal warning text + `ik2_about` |
| up to 16 GPUs | docs/SETUP.md options (`--gpus ... up to 16`) |
| AMD gfx1100 / gfx1201 / gfx1151 + ROCm 7; GLM/QFN/IK2 run (RX 9070 XT), DS4/MM3 compile only, Hy4 not offered; RDNA2 ran on a V620 | docs/AMD_SAGUARO.md "What runs on AMD" table + RDNA 2 section; saguaro.py `NO_ENGINE` |
| Windows: GLM/DS4/MM3 engines, not QFN/Hy4/Inkling | docs/MODELS.md tables ("Linux and Windows"); saguaro.py `family_unavailable` (Win qfn/hy4/ik2); docs/WINDOWS.md |
| Mac Metal = GLM only; GitHub-Macs tested only | saguaro.py `NO_ENGINE["metal"]`; README install note (kept); docs/METAL.md |
| CUDA 12.x / CUDA 13 for RTX 20+ | docs/SETUP.md "If the build fails"; saguaro.py arch check |
| disk ~100/~160/~350/~470/~95/~200 GB rows + 1.1 GB encoder; Qwen 111-177; Hy4 +63.4; Inkling 18.3 | carried disk lines vs docs/MODELS.md sizes + docs/SETUP.md (1.1 GB vision files); Qwen/Hy4/Inkling sizes as above |
| 97.7-99.2% / 83-90% (L: 99.2, 90) / 92.6% / 87.6% | docs/MODELS.md quant tables (unchanged section) |
| 40 / 51 / 71 and 32 tok/s bullets; "20+ on 4x T4"; 149 / 124 / 118 / 81 / 30+ headline | docs/MODELS.md measured rows; the README's own benchmark table rows (22.6, 148.6, 123.7, 118, 81.0, 30+) |
| "up to 8 answered together" incl. QFN | saguaro.py `PARALLEL_FAMILIES` + docs/MODELS.md `--parallel ... up to 8`; docs/SETTINGS.md QFN section (parallel > 8 note) |
| 20-40 minutes first-run build (GLM) | docs/SETUP.md first-run |
| CYBER-FROST = saguaro-qfn quant, old ids resolve | CHANGELOG.md (2026-11-24); docs/SETTINGS.md "CYBER-FROST-3.8-PS (the `cyber-frost` quant)" |

## Left out (fact not confirmable in the repo, or conflicting)

- **"32 GB runs it" (RAM)** - was in the old README; no doc or manifest states it. Dropped; kept only
  "more RAM keeps more experts close and is faster" + the sourced Qwen ~96 GB figure.
- **"Inkling 2 Mini's build ~4 minutes on a Ryzen 7 9800X3D"** - was in the old README; no repo doc states it.
  Dropped (GLM's "20-40 minutes" kept: docs/SETUP.md states it).
- **unsloth's DeepSeek UD-Q8_K_XL size** - docs/SETUP.md says 508 GB, docs/MODELS.md says 467 GB; not put in
  the README (only the pre-existing "~470 GB" disk estimate, which matches MODELS.md's 467 GB + the 1 GB
  tower). The docs' contradiction was not resolved (changing other files is on the Don't list).
- **"Thinking ... High (the default)"** - old README claim; tools/maya_quants.py `MODEL_DEFAULTS` gives each
  model its own default (GLM max, DeepSeek high, MiniMax auto, QFN xhigh). Replaced with "each model starts at
  its card's default".
- **MiniMax default quant** - the setup recommends by PC memory (`choose_mm3_quant`), so its row carries no
  "(default)" mark, unlike GLM/DeepSeek/Hy4/QFN which have one.

## Open questions

- The 508 vs 467 GB conflict for DeepSeek's UD-Q8_K_XL between docs/SETUP.md and docs/MODELS.md should be
  fixed in the docs by whoever owns those pages (out of this task's scope).
- Hy4-Flash pictures are stated as "not measured yet" (docs wording); worth revisiting when a measurement
  lands - and Inkling's "four more kept aside" conversation detail was folded away (it stays in
  docs/MODELS.md).
- The old `<a name="the-models">` / `*-new` / `deepseek-*` anchors are kept though nothing in the repo links
  to them (they may be linked from GitHub issues); safe to delete later if confirmed unused.
