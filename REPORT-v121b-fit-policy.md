# REPORT: 1.2.1-B - refuse only what cannot run; warn + confirm + scale down the rest; Pascal for QFN

Branch `saguaro/q1-v121b-fit-policy`, four commits on top of `2d0a71bba`, one per numbered
plan step. No engine C++ changed; nothing in `serve/qfn_engine.py` or `engines/qfn/` was touched
(no change there was needed: the scale-down only writes flags the engine already reads -
`--mtp-max-t 1`, `--kv-resident N`), so `tools/saguaro/extract_qfn_engine.py` never had to run.

## What changed

### Step 1 - `413c6e3b3`: the true minimum replaces MIN_VRAM_GB's hard cut  (`tools/qfn_fit.py`, `saguaro.py`)

* New measured facts (mortar Monitor, one card, UD-Q4, named in the code comments):
  `DENSE_GIB = 3.9` (whole model, scaled per quant by the manifest's file size), `KV_RES_GIB = 2.8`
  (the `--kv-resident 32768` window), `ACTIVATIONS_GIB = 1.4`, `RESERVE_GIB = 700 MiB` (1.2.1-A was
  not merged - its branch `saguaro/q1-v121a-display-reserve` sits beside this one - so the *current*
  rule, the `--vram-reserve-mib 700` the setup writes, is the reserve), `KV_RESIDENT_MIN = 4096`,
  `RAM_BUFFER_GIB = 4.0` (a 32K context's pinned K/V and the read staging) and `RAM_SYSTEM_GIB = 8.0`.
* New functions: `dense_gib`, `kv_resident_gib`, `stage_bounds`, `card_min_gib` (a card's share of
  the dense weights + minimum resident K/V **by the layer split**, plus activations + reserve) and
  `ram_min_gib` (dense + buffers + system).  One card: 6.3 GiB at UD-Q4 / CYBER-FROST, 9.0 at Q8_0;
  RAM: ~16 GB (UD-Q4/CF), ~19 GB (Q8_0).
* `unsupported()` now refuses **only** below those minimums, in one plain line carrying the numbers
  ("...its 8 GB cannot hold the 9.0 GB every card of this layout must: its share of the 6.6 GiB of
  dense weights and of the 0.35 GiB minimum resident K/V (50% of the layers), 1.4 GB of activations
  and 0.7 GB kept free...").  The old `ram < 96 GB` refusal and the quant-shortfall refusal are gone.
* `best_gpus()` lost the 15 GB size filter; `check_pc`'s two QFN guards (before the card listing and
  after `--gpus` naming) check `QF.unsupported` - the minimum - instead of "needs GPUs of 16 GB or
  more"; `qfn_about` / the platform intro re-worded ("smaller ones run it too, slowly").
* `MIN_VRAM_GB = 15` / `MIN_RAM_GB = 88` stay - as the **comfortable**, measured level, not a floor.

### Step 2 - `1551cd080`: between the minimum and the comfortable layout (`tools/qfn_fit.py`, `saguaro.py`)

* `qfn_fit`: `comfortable()` + `slow_why()`; `plan()` gains `"slow"`, `"slow_why"` and
  `"turned_off"`, and a slow layout comes **scaled down by itself**: context `SCALED_CONTEXT = 32768`
  (a context that fits: one card 64K -> 32K, a split 256K -> 32K), `--kv-resident` window 4096
  instead of 32768, drafting off.  `slow_why` names the expected cause with the row's numbers -
  thin RAM tier (< 96 GB), small cards (< 16 GB), and the quant whose experts are over VRAM+RAM:
  "q8_0: its 175 GiB of experts are 81 GiB more than these cards' VRAM and this PC's RAM together
  hold (~94 GiB)" for the plan's own 2x T4 + 64 GB example - warn, not refuse, exactly as the plan
  asked.  `said` and the CLI say "expect slow decode" + the scale-down.
* `set_up_qfn`: after the layout lines - `warn(...)` with the cause ("most experts read from RAM /
  the SSD: expect slow decode"), `ok` with what it turned off, then `ask("This will probably be
  very slow. Are you sure? (y/n)", ..., default n)`; `--yes` skips the question, "n" stops the setup
  in one plain line.  `--check` never gets here: `qfn_check_line` **reports** the same (cause +
  scale-down) without asking anyone.
* `write_config_qfn`: `--kv-resident` comes from the plan (`min(ctx, lay["kv_resident"])`); a slow
  layout writes `--mtp-max-t 1` (drafting off, said plainly, recorded as
  `installer.mtp_calibration {"skipped": "slow-pc", "kept": "none"}` so the drafting question is not
  asked and a start does not measure drafting), unless `--drafting on` says otherwise.
* `qfn_quant_fit` refuses a quant only under its own true minimum (8 GB card + Q8_0 = 9.0 needed:
  refused; same card at UD-Q4: runs, slowly).
* Nothing moves for a comfortable layout: the AMD-measured config, the T4 configs and
  **LiveT4Parity's 4x T4 + 256 GB UD-Q4 byte-for-byte live config are unchanged** (their tests pass
  untouched).

### Step 3 - `c3238b375`: Pascal for QFN (`tools/qfn_fit.py`)

* `MIN_CC = 61` (the plan's value): `check_pc` takes a Tesla P40 for QFN; below `VALIDATED_CC = 70`
  it prints "can be used (Pascal: experimental)" and the standing experimental warning, as the
  Inkling/GLM path has always done for Pascal (the "follow how ik2 handles it" instruction: the
  shared `check_pc` machinery - nothing QFN-specific beyond the constant was needed).
* Toolkit: the existing `nvcc_range` / `pick_nvcc` / `find_nvcc` / `toolkit_hint` already handle it -
  for `archs=[61]` the range is `(12.0, 13.0)`, a side-by-side 12.8 is preferred over 13.2, and with
  **only CUDA 13 installed** `check_pc` says plainly "nvcc 13.2 (...) cannot build for these GPUs;
  they need CUDA 12.0 or newer but older than 13.0 (CUDA 13 dropped Volta and Pascal)" with the
  install hint "Volta (V100) and Pascal need CUDA 12.x ...".  Covered by new tests (no hardware).
* The build needs no change: the top-level `CMakeLists.txt` already routes a saguaro-qfn build whose
  archs are all < 75 through `engines/qfn`'s experimental pre-Turing path
  (`STRATA_EXPERIMENTAL_SM60=ON`, which itself refuses a CUDA 13 toolkit).
* No T4 tuning on Pascal: the U6 `sm75_only` class rule already keeps every `SAGUARO_SM75_*` /
  `SAGUARO_KQL` / `SAGUARO_GDN_KEYHEAD` / `SAGUARO_ARGMAX_ROWS_SM70` key off a P40 layout (tested).

### Step 4 - `ce5e0ad37`: the tests (`tools/test_qfn_fit.py`, `tools/test_setup_qfn.py`)

* `tools/test_qfn_fit.py` (27 -> 34 tests): `TrueMinimum` (the 8 GB card: UD-Q4 runs slow, Q8_0
  refused with its numbers; the min's arithmetic; RAM minimum line), `SlowZone` (thin RAM and small
  cards slow-not-refused, scale-down listed, live split kept at 60 GB), `SlowReasonByName` (the
  Q8_0-over-2x-T4+64 GB example says its numbers as a warn; the measured Q8_0 AMD box is neither
  refused nor called slow), `Facts.test_pascal_is_for_qfn_now`, `PolicyMatrix` - the plan's matrix
  (1-4x T4 at 15.0 and 16.0, V100 16/32, 2080 Ti, 3060, 3090/4090, A4000/A5000 single and paired,
  P40 24, RX 7900 XTX 24 and RX 9070 XT 16, an 8 GB card, a mixed 24+16 pair; RAM 32/64/96/256; all
  three quants) judged R/S/F against the plan - refused only where it cannot run (the 8 GB card is
  the one NVIDIA card Q8_0 cannot run on), warned-slow with the scale-down written where slow,
  unchanged where it fits - and `test_4x_t4_256_gb_ud_q4_stays_the_live_layout`.
* `tools/test_setup_qfn.py` (52 -> 64 tests): the quant question refused only under the minimum;
  `Layouts1To4T4` re-rows its RAMs to 32/64/96/256: the thin rows are now **written, scaled**
  (32K, `--kv-resident 4096`, `--mtp-max-t 1` + `slow-pc` record, tier still per the rule) and the
  96/256 rows write the layout as before; `SetupFlow` gains the slow-flow tests (warn/scale-down
  written, exactly one confirm question, "n" stops, `--yes` skips, drafting not asked, `--drafting
  on` overrides, a measured PC asked nothing); `PascalForQfn` (the P40 usable and laid out slow, no
  T4 env, the 12.x toolkit rule of the range/picker/hint, `check_pc` with only CUDA 13 installed
  refusing plainly, `check_pc` with CUDA 12.8 choosing the P40); `QfnGpus` rewritten around the
  minimum (4 GB cards refused in one line with their split-priced numbers, `--gpus` naming one, 12 GB
  of RAM refused; the 8 GB pair *not* refused); `CheckLine` reports the slow row's cause and
  scale-down (and the Q8_0 8 GB refusal keeps the numbers).
* The plan's "4x T4 + 256 GB UD-Q4 unchanged": `LiveT4Parity` and
  `Layouts1To4T4.test_four_t4s_256_gb_ud_q4_is_the_pinned_live_config` pass untouched.

## Files

* `tools/qfn_fit.py` - minimums, comfortable/slow classification, scaled plan, CLI.
* `saguaro.py` - check_pc / quant-fit / choose / set_up_qfn / write_config_qfn / qfn_check_line /
  about wording for the new policy.
* `tools/test_qfn_fit.py`, `tools/test_setup_qfn.py` - the matrix and the setup-side tests.

## Tests run (one module at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest <module>`)

| module | result |
| --- | --- |
| `tools.test_qfn_fit` | 34 tests - OK |
| `tools.test_setup_qfn` | 64 tests - OK (1 skip: `ref/live-ud-q4.json` not in the checkout) |
| `tools.test_setup_amd` | OK (unchanged) |
| `tools.test_setup_ik2` | OK (unchanged; present, so run per the plan) |

Baseline before the work: `test_qfn_fit` 27 / `test_setup_qfn` 52, both green; nothing else was run.

## Open questions

* **The minimum's constants.** DENSE 3.9 / activations 1.4 / buffers 4 / system 8 come
  from the plan's mortar Monitor measurements; the dense weights scale per quant by the files' total
  size (Q8_0 ~1.7x).  The resident K/V window is no longer a Monitor figure: the review found the
  2.8 GiB@32K line contradicted the module's own measured `KV_BYTES` by 7.24x, and the engine's own
  INFO `kv_bytes_token=` (generate.cpp:12401, 12 QSA layers x 1,056 B at `--kv` int8 = 0.39 GiB at
  32768 cells) settled it - `kv_resident_gib` now *is* `kv_vram_gib`, one constant.  One corner the
  constants create: a T4 that reports 14.x GB (ECC state, driver build) is below the comfortable 15
  and gets the slow treatment while `t4_sized()` still hands it the live `11,24,36` split - the
  plan's own 15.0/16.0 rows never hit this.
* **A comfortable row with experts over VRAM+RAM** (e.g. 24 GB card + 128 GB RAM, Q8_0; or the
  measured AMD Q8_0 box) writes no warning, exactly as today: "fits today stays fits" was kept so
  the measured boxes and LiveT4Parity cannot move.  Whether such rows deserve an informational note
  is a follow-up call.
* **Mixed Pascal + newer cards**: the top-level CMake builds saguaro-qfn for all-old archs
  (pre-Turing path) or for the new ones - a P40 beside a T4 builds sm_75 only (pre-existing
  engine-build behaviour; "change engine C++" was off-limits).  A single P40, the hand-run box,
  builds and lays out fine.  `check_pc` now says this plainly for a chosen mixed list (a warning
  naming the missing kernel image and the `--gpu N` one-class-at-a-time way out) instead of
  building past the pre-Turing card in silence.
* **The Models page** (`serve/model_pull.py`) - re-worded with the review: its "needs" line and the
  two thin-PC verdicts now name the *measured* box ("the layout was measured on ...; smaller cards
  / less RAM run it too, slowly") instead of "96 GB are needed", and the quant menu's note reads
  "cannot run on this PC" where it means exactly that (v1.2.1-B made the refusal the true minimum);
  `docs/AMD_SAGUARO.md` says the same about the AMD row.
* **Slow splits get 32K** context unconditionally (the plan's "a context that fits"); a split's RAM
  cost is cheaper than one card's, so 64K might survive on some of them - left conservative.
* The true minimums have not been verified against real cards (no hardware here): the tests prove
  the policy, not that a 8 GB card really decodes.
