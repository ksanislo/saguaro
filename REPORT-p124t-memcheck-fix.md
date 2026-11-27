# P124t REPORT - tools/mem_vs_ctx.py misses pinned memory and behaviour changes (Saguaro 1.2.4)

Both gaps of ref/mortar-run-20261126.txt are fixed in the check; no engine was touched and
`serve/qfn_engine.py` was not needed (nothing in serve/ changed).

## What changed

### Step 1 - the pinned RAM is measured twice and judged on the larger (commit 42fa8f486)

The ref run printed "pinned MiB 0.0" for every engine while the engines' own logs said "68.00 GiB
page-locked" (QFN) and "the latent cache in 2.95 GB of pinned RAM" (GLM): ROCm/CUDA page-locked host
memory lands in no process's `VmLck`/`VmPin`. Now, per context, the check records the pinned RAM two ways:

* `pinned_tree_mib` - `VmLck` + `VmPin` summed over EVERY process of the server's tree (children of
  children - `process_tree` already reached them; it is now also what the docstring says);
* `pinned_host_mib` - the host-wide delta of `/proc/meminfo` **Mlocked + Unevictable + Shmem** between
  before the start and after `/health` ok + settle. The check owns the moment (nothing else loads while
  a context runs), so the rise is the run's; a key that FELL counts 0 (a neighbour giving pages back is
  not this run un-pinning). None when the box has no readable `/proc/meminfo`.

The table shows both columns (`pinned host`, `pinned tree`); the verdict judges the pinned figure on the
LARGER of the two. A box without `/proc/meminfo` loses only the host row and still judges on the tree.

### Step 2 - the engines' expert-placement lines become figures, >2% between contexts FAILS (commit a1ef56bb6)

Equal GPU totals hid Inkling going from "resident: every expert in VRAM (3072 copied)" at 32768 to
"offload: 0 of 3072 experts in VRAM" at 262144 and still PASSing. `parse_placement()` now reads the
start-up prints (patterns noted from the engine sources, grepped per the plan) into four figures per
context:

* **experts in VRAM** (count, matches summed - DS4 prints one line a card): QFN "N in the GPU cache" /
  "every expert is in the GPU cache (N of M)", Inkling "resident: every expert in VRAM (N copied" and
  "offload: N of M experts in VRAM", MM3 "loaded ..., N of M experts in VRAM", Hy4 "N of M routed experts
  in VRAM", DS4 "(N of M in VRAM)";
* **experts in RAM** (the largest GiB/GB stated): QFN "resident RAM mode: X GiB of experts in RAM",
  Inkling "experts in pinned [mmap+register] memory: X GiB", MM3 "X GB pinned pool", DS4 "routed experts
  on the CPU lane (X GB ...)";
* **expert pool (GB)** - GLM states no count: "VRAM before the expert pool: A of T GB in use" leaves
  T - A for the experts, summed over the cards (that pool IS its experts in VRAM);
* **pinned RAM declared (GB)** - the largest "X Gi?B of pinned RAM" the engine prints; this is how GLM's
  "the latent cache in 2.95 GB of pinned RAM" now shows up as a figure next to the host-wide delta.

A figure the log said nothing about is None and compared only where two contexts both state it. The
verdict: memory figures as before (fail at >= max(2%, 256 MiB)), placement figures fail at > 2% (the
plan's "more than 2%"), and the FAIL line names the difference furthest past its own limit (the
Inkling case: "FAIL: experts in VRAM differs by 3072 between the contexts (limit 2%: 61.44)" where the
old rule PASSed on a 310 MiB GPU difference). `main` prints one placement line per context above the
budget lines it already printed.

## Files

* `tools/mem_vs_ctx.py` - meminfo parse/read/delta, `pinned_of` + `RAM_FIGURES`, `parse_placement` +
  the pattern tables, `compare_placement`, merged `verdict`, two-column table, placement printout.
* `tools/test_mem_vs_ctx.py` - fixtures from `ref/mortar-run-20261126.txt` (all four engines' lines)
  and fixed `/proc/meminfo` texts; `MeminfoTests` (GLM's 2.95 GB latent cache shows host-wide where
  VmLck read 0.0, the three keys summed, falls clamped, missing file None), `PlacementTests` (Inkling's
  resident/offload flip must FAIL, GLM shows its pinned latent cache and its moved expert pool, QFN
  counts 1729/68 GiB and the 3.35 GiB declared, MM3/DS4/Hy4 equivalents, the 2% boundary, the missing-
  figure rule, the breach-ranking), and pinned-judged-on-the-larger rows; the old table/verdict tests
  moved onto the new row fields.
* `docs/RELEASE-CHECKS.md` - section 4's description of what the check records and its rule.

## Tests run

Only the module the plan names, run one at a time with `/opt/llm/saguaro/.venv/bin/python`:

* `python -m unittest tools.test_mem_vs_ctx` after step 1: **32 tests, OK**; after step 2 (with the
  placement fixtures): **42 tests, OK**. No GPU, no model: the mock-engine cycle test still does the
  real start/stop of `serve.server --engine mock`.

On the ref run itself: the Inkling config's logged lines now FAIL ("experts in VRAM differs by 3072",
verified end-to-end with a scratch render of the ref log), GLM shows the 2.95 GB latent cache and fails
on its expert pool (8.06 GB at 32768 vs 7.75 GB at 262144, 4%), and QFN reads experts_vram 1729 /
experts_ram 68.00 GiB / declared 3.35 GiB with the host delta replacing the 0.0 pinned row.

## Open questions

* The GLM placement rule makes the 1.2.4 run's GLM configs FAIL (the context's +0.30 GB of state/KV
  comes out of the expert pool, 4% > 2%). Read as intended (context not in use must not cost an expert
  tier) or should the expert-pool figure get a 256 MiB-style floor next to the 2%?
* DS4 prints "(N of M in VRAM)" per CUDA card and the counts are summed; engines that print a placement
  line more than once a run (serve stats) could double-count - the fixtures cover the start-up lines
  only, and no such double print was seen in the ref logs.
* The host-wide pinned row is a box-wide reading: RELEASE-CHECKS.md now says the quiet-machine rule
  covers it too, but a neighbour job that pins between the two reads moves the figure (the old GPU-row
  caveat, now for two rows).
