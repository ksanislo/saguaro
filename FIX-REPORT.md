# FIX REPORT: the 1.2.1-B fit-policy review (REVIEW-RESULT.md)

One commit per finding (git log, oldest first: f49cdfd88, d7a7fcd1b, f3b24d18e, 0c20085a2,
2d1e4ce0d, e57681bcc, f1ce146d4, then this report). Tests run again one module at a time with
`/opt/llm/saguaro/.venv/bin/python -m unittest <module>`: `tools.test_qfn_fit` 38 OK,
`tools.test_setup_qfn` 69 OK (the pre-existing worktree-local skip of `ref/live-ud-q4.json`),
`tools.test_setup_amd` 33 OK, `tools.test_setup_ik2` 54 OK. `serve/qfn_engine.py` and
`engines/qfn/` stayed untouched (nothing there changed, so `tools/saguaro/extract_qfn_engine.py`
still has nothing to regenerate); no engine C++ moved.

## Finding 1 (MEDIUM, FIX-FIRST): a comfortable card refused when two identical tiny cards sit beside it - FIXED

`f49cdfd88`. Both causes the review named:

* `best_gpus` (tools/qfn_fit.py): `equal_group` ranks by count, so the tiny pair *was* "the group".
  It now drops an equal group that stands below its own `card_min_gib` **when the largest single
  card clears its own single-card minimum** (`_layout_runs` - the group's minimum is a function of
  the group, so the group is tried, then the single card). A tiny pair with nothing better keeps
  the split, so `test_cards_below_the_minimum_are_refused_in_one_line` still passes with its
  2-card-priced refusal (now "the 4.1 GB", see finding 2), and the comfortable-card case sets up:
  reproduced first - `16:86 1:35 1:35` refused before the fix, picks the single card after.
* `check_pc`'s pre-guard (saguaro.py) now applies the same cc filter `choose_gpus` applies
  (`arch >= min_cc`) before pricing `best_gpus`, so a card the picker can never take (and that
  `--gpus` could never name) cannot refuse the PC; with no usable card at all it prices `found`
  and still refuses in one line.

Tests: `Choice.test_a_comfortable_card_beats_a_group_of_tiny_ones` (both cc shapes plus the
groups that still run: the 8 GB pair, four T4s, the bare 4 GB pair) in `test_qfn_fit`, and
`QfnGpus.test_a_tiny_equal_pair_beside_a_comfortable_card_is_not_a_refusal` driving the real
`check_pc` for the A4000 + 2x GT 710 (cc 3.5) and 2080 Ti + 2x GT 1030 (cc 6.1) PCs - the matrix
gap and the check_pc gap the review listed.

## Finding 2 (MEDIUM, FIX-FIRST): the resident-K/V figure contradicted the measured K/V by 7.24x - FIXED

`d7a7fcd1b`. Settled against the engine's own INFO line as directed: generate.cpp:12401 prints
`kv_bytes_token = qsa_kv_bytes_per_cell(g) x n_qsa_layers(g)` (core/layer.cpp:1217), and this
module already carried that measured reading - `KV_BYTES = 12672`, the 12 QSA layers x 1,056 B at
`--kv int8`, which `kv_vram_gib` (and `serve`'s tests against the engine INFO) pinned at ~0.4 GiB
for 32768 cells. `KV_RES_GIB = 2.8` was the outlier; it is **gone**, and `kv_resident_gib(cells)`
now *is* `kv_vram_gib(cells)` - one constant serves both, so the CLI's `--kv-resident` line and
the per-card minimums read the same bytes. The comment records that the plan's "2.8 GB" mortar
Monitor line counted more than the resident window (the whole-K/V-in-pinned-RAM and other
buffers), which is what the review's first branch was weighing; taking KV_BYTES instead keeps the
8 GB card's row slow-but-fits, as the plan's matrix and the review's "checked and clean" do.
New test `Sizes.test_the_two_kv_readings_are_one_number` ties the two functions for 2048/4096/
32768/262144 cells. The verdicts did not move (the review's own 6.03 vs 6.33 check); the pinned
figures did, in `test_qfn_fit` and `test_setup_qfn`: per card 6.3->6.0 (UD-Q4), 9.0->8.7 (Q8_0),
4.2->4.1 (4 GB pair), 2.3->2.2 (GT 710 beside the 4090), the refusal's "minimum resident K/V"
0.35->0.05 GiB, and the matrix's Q8_0 pair 5.6->5.4 GB a card - the minimum formulas in
`test_the_min_per_card_numbers` / `test_the_refusal_prices_the_failing_card` now read
`F.kv_vram_gib(4096)` instead of the 2.8 fraction.

## Finding 3 (LOW): the RAM minimum ignored the tier floor - FIXED (scaled the tier, did not raise the minimum)

`f3b24d18e`. Chose the review's second option: raising `ram_min_gib` by the 8 GiB floor would
*refuse* 16-23 GB PCs, against "don't refuse anyone who can be supported" and against the plan's
own "RAM minimum = dense + buffers + system". Instead a slow layout's tier is scaled to what the
minimum leaves: `budget = min(resident_budget_gib(...), int(ram_gb - ram_min_gib(quant)))` in
`plan()`, and where it leaves nothing no `--resident-budget-gib` is written at all - the experts
stream through the page cache there, which is what the review said such a 16 GB machine is for.
`--ram 16 8` now prints `resident budget -`, and 20 GB (room 4) pins 4 instead of the floor's 8;
32/64 GB stay 12/44 GiB, so every Layouts matrix row and the comfortable tier rule are untouched. The plan's
`said` line and the setup's `qfn_ram_said` phrase say the no-tier case honestly (the old
"(all of them fit in RAM)" parenthetical would have lied there). Tests:
`SlowZone.test_a_thin_ram_slow_pc_scales_its_pinned_tier` (16 -> none, 24 -> 8, 64 -> 44,
comfortable -> 76) and `SetupFlow.test_a_16_gb_pc_pins_no_tier_and_streams_through_the_cache`
(the written config carries no `--resident-budget-gib` and keeps the scaled 4096 window; its
`--max-context` is the 65536 the flow names itself, so that is not asserted).

## Finding 4 (LOW): check_pc accepted a QFN arch list the build cannot serve - FIXED (warning + report)

`0c20085a2`. Engine C++ was off-limits, so `check_pc` says it plainly for the *chosen* list: when
`min(archs) < 75 <= max(archs)` for qfn (CMakeLists.txt:1787-1808 takes the pre-Turing path only
when *every* arch is < 75), it warns that the build serves the newer cards only and the
pre-Turing card would get a binary with no kernel image for it, and points at `--gpu N` for one
class at a time. A warning, not a refusal: the PC runs fine on either class alone (a P40 alone is
the hand-run box; MIN_CC 61 must not strand it) - "refuse only what cannot run" keeps its word.
`PascalForQfn.test_a_mixed_pre_turing_and_newer_list_is_said_plainly` runs the real `check_pc`
both ways (P40+T4 via `--gpus 0,1`: warns, sets up; P40 alone: no warning); the report's open
question on mixed cards now records what check_pc says.

## Finding 5 (LOW): the scale-down line claimed the flag where there is no drafter - FIXED

`2d1e4ce0d`. The claim only ever needs `--mtp-max-t 1` when a drafter exists, and only
`write_config_qfn` (saguaro.py:7576, whose own `ok` line keeps the flag - there it is true)
knows that; step 2 and `--check` cannot know whether the draft layer will be declined at step 8.
So the *generic* lines dropped the flag: `turned_off`'s item is now `drafting off` (true in every
case - a layout with no drafter drafts off), which `qfn_check_line` and the step-2 line repeat,
while the `--drafting on` gate (the previous round's finding 3 fix) keeps filtering the item.
Also the same-family wording: `slow_why`'s "its 8 GB card holds few of **their** layers' experts"
(one card, plural pronoun) became "few of **the model's** layers' experts". Tests: the pinned
strings in `SlowZone` / `Cli` / `CheckLine` / `test_drafting_on_overrides_the_scale_down` moved
without the flag, and `SetupFlow.test_a_slow_pc_with_no_drafter_claims_no_flag` (slow mem +
`_nomtp`) asserts the scale-down line says no `--mtp-max-t` and the config carries neither
`--mtp` nor `--mtp-max-t`.

## Finding 6 (NIT, informational): stale "needs" wording outside the setup - FIXED

`e57681bcc`. `serve/model_pull.py`'s Models-page "needs" line and its two thin-PC verdicts no
longer call the measured box a requirement: "16 GB or more ... about 96 GB are needed" became
"measured on an RX 9070 XT (gfx1201) with 96 GB of RAM ... smaller cards and less RAM run it
too, slowly (only a PC that cannot hold the model at all is refused)". `docs/AMD_SAGUARO.md`'s
QFN row reads the same way. The quant menu's note "does not fit this PC" (saguaro.py) became
"cannot run on this PC", which is what v1.2.1-B made it mean. `ModelsPage.test_catalog_entries`
moved its pinned substrings ("measured on about 96 GB", "runs it too, slowly"); the report's
open question on the Models page records the change. Not touching `serve/qfn_engine.py` was
respected - `model_pull.py` is setup-facing and not generated.

## Finding 7 (NIT): no CHANGELOG entry - FIXED

`f1ce146d4`. A new "Qwen3.8-Flash-Next: refuse only what cannot run (2026-11-24)" section at the
top of the Saguaro notes (the style of 26b6cce3a / dd8031ca9): the true minimum that replaced the
16 GB / 96 GB refusals, the warned + confirmed + scaled slow layouts, Pascal at MIN_CC 61 with
the CUDA 12.x rule and the mixed-arch warning, and the picker behaviour the user now sees.

## Review's "test gaps worth adding" - all covered

The tiny-pair-beside-a-comfortable-card matrix/picker row (`Choice` + the `check_pc` test,
finding 1), the comfortable card pair-vs-alone case through `check_pc` (`QfnGpus`, finding 1),
and the `kv_resident_gib`-to-`kv_vram_gib` tie (`Sizes`, finding 2).
