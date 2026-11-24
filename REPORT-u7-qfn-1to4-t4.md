# U7 report - the QFN setup picks the right options for 1, 2, 3 or 4 Tesla T4s, and one card gets pictures

## What changed

Step 1 (`./scratch/u7-cases.txt`, gitignored; the findings are carried in the code comments, the step-2/4 commit
messages and here): what `qfn_fit.plan` / `write_config_qfn` wrote for every (1, 2, 3, 4 T4) x (UD-Q4_K_XL, Q8_0,
CYBER-FROST Q5_K_M) x (64, 128, 256 GB) combination, per quant's file bytes (103.7 / 175.3 / 104.8 GiB from
`tools/qfn/download.json`). Most was already right after U6 (context, K/V window, tier, T4 env by class, NUMA only
on more than one socket, pipe 1 off the measured four-card layout); six things were not - P1..P6 below.

Step 2 (`tools/qfn_fit.py`, `saguaro.py`) - the 4x T4 UD-Q4/Q8_0 output untouched, byte for byte:

- **P2 the computed split followed each card's VRAM** (qfn_fit's new `proportional_split`): hand-pick two unequal
  cards and the smaller one used to get the same layer count as the bigger one, i.e. more weight than it can hold;
  now the bounds fall at 48 x (cumulative VRAM share), clamped to strictly rising with a layer for every card.
  Equal cards - every T4 layout - get exactly the old even "24" / "16,32" / "12,24,36". GPU0's extra load is why
  the *measured* four-card splits are 11,24,36 / 10,22,35 and not even; nothing was invented for the computed ones.
- **P3 the measured `--prefill-pipe` needs its four cards**: `live` matched the split *string*, so a computed split
  that merely hit "11,24,36" on four non-T4 cards would have borrowed mangonel's pipe rate; it now also requires
  four all-sm_75 cards, and `measured` collapses to it.
- **P1 one card -> parallel 1** (write_config_qfn): the old `a.parallel or old["parallel"] or ...` kept a previous
  multi-card config's `"parallel": 2` when the setup shrank to a single card - the one-16 GB-card parallel loss
  the AMD box measured. An explicit `--parallel N` still says, and a several-cards setup again still keeps the old
  value; only the one-card layout drops it.
- **P5 the RAM refusal names the quant**: `unsupported()` takes the quant and, when its experts are plainly over
  the layout's VRAM+RAM together (q8_0 on every refused case: 175 GiB over ~79-124; cyber-frost-q5_k_m on one or
  two cards), appends the two numbers ("its 175 GiB of experts are 81 GiB more than these cards' VRAM and this
  PC's RAM together hold (~94 GiB) ... more RAM or more cards, or the smaller quant"). No new numeric refusal:
  the layouts that were *measured* running with SSD reads - the 96 GB Q8_0 AMD box, about 50 GB of experts on the
  SSD per the manifest - keep fitting.
- **P6** the `tools/qfn_fit.py` CLI now offers every manifest quant (`--quant cyber-frost-q5_k_m` ran at last).

Step 3 (`tools/test_setup_qfn.py`): `Layouts1To4T4` - one case per (1, 2, 3, 4 T4) x (UD-Q4, Q8_0, CYBER-FROST) x
(64, 128, 256 GB) as a fresh single-socket setup writes it: the 64 GB row refused in one plain line (with the
quant's own bytes named exactly where they are over the layout), and every written case checked for its split,
context, 32K resident K/V, RAM tier (only Q8_0 at 128 GB pins one: 108 GiB), answers-at-once (no key on one card,
2 on a split), trim/prefill-pipe (1 on 2-3 cards, 1536/6144 only on the four measured T4s, 1 for never-measured
CYBER-FROST at four), the env by class with no NUMA keys and no `--pool-workers` (a two-socket case adds exactly
those), drafting kept, and four T4s + 256 GB UD-Q4 writing exactly U6's pinned live config (LiveT4Parity's
env/args constants, the machine-specific paths/`--max-context`/two dead keys aside).

Step 4 - pictures on every layout. The elastic encoder starts on VRAM the engine lends it, and there are exactly
**two lend paths**: the give path (`SAGUARO_ELASTIC_SLOTS` + `_GIVE` with batch slots from `--batch N` - i.e.
answers at once 2 or more - and VMM caches; the engine then prints "INFO vision_lend=1",
`engines/qfn/src/program/generate.cpp:13119-13164`, and the server sends VLEND, `serve/server.py:3105-3139`), and
the **VRAM command** on a segmented expert cache - which needs the engine started with `--vram-elastic`, NVIDIA
only (`engines/qfn/src/core/expert_cache.cpp:301` "a segmented expert cache is CUDA-only for now", and the engine
answers `VRAM <mib>` only then, `generate.cpp:12738-12769`; the config key `"vram_elastic": true` is what adds the
flag, `engines/qfn/serve/server.py:3762-3768`, extracted into `serve/qfn_engine.py`). So:

- A config that gets the vision entry **without batch slots** (parallel 1 - the one-card case, or a split the
  person set to `--parallel 1`) now writes `"vram_elastic": true`. That is what a one-card config needs for
  working pictures: the server's `_vision_up` fallback reserves the encoder's `elastic_mib` through the VRAM
  command (`serve/server.py:3161-3170`) before starting the reader.
- A layout where **neither path can be configured** - the single AMD card, which has no `--batch` at parallel 1
  and no `--vram-elastic` - gets **no vision entry and no `--vision`**, said in one plain line, instead of an
  elastic entry whose reader starts into a full card and is SIGKILLed (U6's `./ref/vision-fail.log`).
- **The server refuses instead of crashing**: new `StrataEngine.vision_room()` reads the engine's own two start
  facts (`vision_lend=1`, `vram_elastic=1`, `generate.cpp:12407, 13162`); both `_vision_up`s - the shared
  server's and engines/qfn's own - raise a plain `ValueError` ("this engine lends the picture encoder nothing
  ...") when the encoder would start with neither path, instead of the old "starts it anyway, as a resident one
  would". An engine that *can* lend and merely refused one VLEND/VRAM still starts the encoder as before.
  Changed in `engines/qfn/serve/server.py`, then `tools/saguaro/extract_qfn_engine.py` regenerated
  `serve/qfn_engine.py` (147 definitions; only `vision_room` is new there - `serve/qfn_engine.py` was not
  hand-edited).

CYBER-FROST notes: its experts are a hair over UD-Q4's bytes (104.8 GiB), the tier rule follows them by itself
(at 128 GB the page cache holds them, 124.8 <= 128, while Q8_0's 195.3 do not), it gets the even split and pipe 1
on four cards because nothing was measured there, and it is the quant the plan's Reddit 2x T4 + 256 GB user will
pick: tier none, parallel 2, give-path pictures, no `--prompt-cache`.

## Files

- `tools/qfn_fit.py` - `proportional_split`, VRAM-aware `layer_split`, quant-aware `unsupported`, four-card-and-T4
  `live` gate, `manifest_quants` for the CLI.
- `saguaro.py` - `write_config_qfn`: `par` moved before the config is assembled and no longer inherited onto one
  card; the two-lend-path vision decision (`"vram_elastic": true` / pictures plainly off); docstrings of
  `write_config_qfn` and `qfn_vision_step`.
- `engines/qfn/serve/server.py` - `StrataEngine.vision_room`; `_vision_up` refuses with no lend path (standalone
  server).
- `serve/server.py` - the shared `_vision_up` refuses the same way before `v.restart()`.
- `serve/qfn_engine.py` - regenerated by `tools/saguaro/extract_qfn_engine.py` (the `vision_room` closure).
- `tools/test_setup_qfn.py` - new `Layouts1To4T4` (36 cases + the pinned-config case) and `PicturesOnEveryLayout`;
  `Parallel` covers the one-card shrink; the AMD single-card config tests now expect pictures off;
  `LiveT4Parity.flag_pairs` became reusable.
- `tools/test_qfn_fit.py` - new `UnequalCards` and `RefusalsByName`.
- `./scratch/u7-cases.txt` (untracked, gitignored) - the step-1 analysis.

## Commits

1. step 1: nothing to commit (gitignored scratch only)
2. step 2 `666f83039`
3. step 3 `6e36cf579`
4. step 4 `c38fc4e80`
5. this report

## Tests (run one at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest ...`)

- `tools.test_setup_qfn`: **OK, 52 tests** (43 before + 9 new - `Layouts1To4T4` 4, `PicturesOnEveryLayout` 5;
  2 existing ones retargeted: the AMD single-card vision expectations flipped to pictures-off)
- `tools.test_qfn_fit`: **OK, 27 tests** (22 before + 5 new)
- `tools.test_setup_amd`: **OK, 29 tests** (unchanged)

No GPU was used, no engine started, nothing under /opt/llm touched, no engine C++ changed.

## Open questions

- **AMD single card now ships text-only**: the plan's "neither lend path -> pictures off" line applies to it
  (`--vram-elastic` is CUDA-only), so a fresh QFN `--setup` on the RX 9070 XT box no longer writes the vision
  entry, though step 7 still downloads the tower before step 9 turns it off. If ROCm's expert cache leaves enough
  room in practice, the entry could come back with a measured number instead of the plain line - and step 7 could
  skip the download once the rule is known early.
- **256K context on two cards**: kept (the resident K/V window is 0.39 GiB and `SPLIT_CONTEXT` is what U6 ships
  for any split); if 2x T4 users report pressure, per-size contexts are a one-line change in `plan()`.
- **GPU0's overhead in the computed splits**: only the measured four-card splits account for the embeddings/
  vision/reserve weight GPU0 carries; a 2-card split "24" still gives GPU0 the same layer count as GPU1. A measured
  2x T4 box would let `MEASURED_SPLITS` carry one (the parity mechanism is ready).
- **The give path on AMD** was not exercised: the layout where it could apply (AMD plus answers at once) is not
  what `set_up_qfn` writes (AMD stays one card, parallel 1), so the pictures-off rule never contradicts it, but
  nobody measured VMM-give on ROCm.
- `--parallel 2` remains the fresh split default (the live box runs 8; U6's choice, untouched here).
