# FIX-REPORT - REVIEW-RESULT findings, one commit each (round 5: `67e1244e6..ec68da4a8`)

The manager's card test (mortar, RX 9070 XT, 2026-11-26 17:38-17:42, run against `67e1244e6`, `REVIEW-RESULT.md`
committed with this report) is the whole finding this round: the HOST side still passes (1.30 GiB pinned K/V at
262144 and 1048576, expert RAM tier 68.00 GiB at both) but the VRAM side fails - 8.67 GiB in use before the expert
cache at 262144 against 12.08 GiB at 1048576 ("~3.4 GiB still allocated by --max-context ... and no log line names
it"), and the cache fell to 1925 slots at 262144 against round 3's 2256 ("the new VRAM path costs ~1 GiB even at
the default context"). The card's three numbered parts map to the three sections below; the code changes are
`9f61edc6a` (step 1) and `ec68da4a8` (step 2), the new build dir is `scratch/build-p124e-r5` (CUDA sm_75, ggml from
`scratch/ggml-pinned`), and nothing here is GPU-testable on this box - the proof stays `scratch/card-test-elastic-kv.sh`
(updated this round) on mortar.

## Step 1 - the load-time accounting lines, and the one-line engagement verdict (`9f61edc6a`)

* `include/strata/core/devnotes.hpp`, `src/core/devnotes.cpp` (new, linked into `strata_engine`): a `DeviceNote`
  wraps one device allocation and records the card's **free-memory delta** across it, under a name; names
  aggregate (bytes, charge, count).  The delta is the point: an address reservation can ask GiBs and cost 0 while
  its backing costs GiBs, and only the delta tells those apart - which is exactly the HIP question the card
  cannot answer without the lines.  Accounted sites: the solo session's carve, each stage's, each batch slot's
  (`generate.cpp`); the native head and the logits buffer (`generate.cpp`, `native_head.cpp`); the MTP drafter's
  dense weights, its 512 expert blobs, its K/V state (`mtp.cpp`); the DFlash2 pack (`dflash2.cpp`); and the round-4
  pools in `layer.cpp` - the follow pools' **addresses for --max-context** and their **start step (backed)** as
  separate lines, the block-score scratch (addresses + start), the grown RoPE table (addresses + the trained
  positions), and the SAGUARO_YARN_ELASTIC table's addresses.
* The dump runs immediately before the expert cache sizes itself (`generate.cpp`, after
  `mem_mark("the weights, the session and the drafter")`, before the sizing's first `device_free_bytes()`):
  `saguaro generate: VRAM in use before the expert cache: X GiB of Y (Z free): the allocations, named` then one
  `VRAM: <name>: A GiB asked, B GiB in use (N allocations)` line per name at or above the card's 64 MiB.  Sum the
  lines against the header and the unnamed remainder is visible too.
* One line, always printed at the same load (`generate.cpp`), naming the verdict and the reason:
  `VRAM follows use: ENGAGED on CUDA0 (<vmm_reason>, ...)` / `FELL BACK on CUDA0 (<why the give cannot run>):
  backing ... whole for --max-context N` / `OFF on CUDA0 (<condition that stopped the rule>)`.  The reason comes
  from `strata::core::vmm_reason()` (`vmm.cpp`, all three build branches: which driver entry point / device
  attribute / granularity query said what, or `SAGUARO_HIP_VMM=0`), and from `qsa_vram_follow_why()`
  (`layer.cpp`) for the rule's own conditions (`--serve`/`SAGUARO_KV_VRAM_ELASTIC`/streaming/host rule/VMM).
  Round 4's ENGAGED line kept its "holds N MiB at load" number inside the new wording.

## Step 2 - the ~3.4 GiB and the ~1 GiB: found, fixed (`ec68da4a8`)

**Read the code path the accounting names.** The path is round 4's own fallback at `generate.cpp:7113-7146`:
"the context's device state cannot grow (the expert cache gives no room: **the experts are not all in the RAM
arena**): backing it whole for --max-context now".  And the card's rig can *never* have the RAM arena:
`--resident-budget-gib` - the option that gives the card its 68 GiB expert RAM tier, the reason the host side now
passes - **implies `--mmap-experts`** (`generate.cpp:3232`, it has since the option existed), so the expert source
is `FileExpertSource`, `srcp != &arena_src`, and round 4's give decision (`srcp == &arena_src` at :7114 and again
inside `give_on` at :13687) made the follow rule **fall back on MORTAR - the acceptance rig - every single run**.
The fallback backs the context state whole:

* the ~3.4 GiB that scaled with --max-context is exactly that: the backed-whole follow arrays' delta
  262144 -> 1048576 (indexer keys 96 chunks per state x 12 states ~ +1152 MiB per holder, x3 holders, plus the
  scratch and the page tables, less the grown RoPE which backs only the trained positions) - the same ~3.4 GiB
  the card measured, in the same range the round-4 table predicted; the fallback line printed the *reason* but no
  bytes, which is why "no log line names it" - step 1's table and the FELL BACK line name it now.
* the ~1 GiB 262144 lost against round 3 is the same fallback's doing: round 4's state arrays move from one
  carved `cudaMalloc` (round 3) to whole-context VMM backing (chunk-rounded hipMemCreate/hipMemMap ranges) on top
  of the reservations - the accounting table will print the per-rig bytes on the next run; after this fix the
  fallback does not run in that configuration at all, so 262144 pays only the start step and should beat
  round 3's carve (round 4's rule frees ~336 MiB per holder that round 3 carved - pooled keys and scratch -
  ~ +350 expert slots).

**The fix:** the give path never needed the RAM arena.  `give()` (`generate.cpp`) evicts the expert cache's
*coldest* residents - it marks them `kNotResident` in the host and device residency tables before unmapping their
chunks, so subsequent reads of an evicted expert take the ordinary "not in the VRAM cache" path, answered by the
same `ExpertSource` interface (`blob_stable`, `prefetch` are virtual; the mapped file tier answers from its
mapping / resident complement exactly as the arena does, and it is precisely the mode where the file already
answers cache misses).  Both gates now accept **any expert source** (`srcp != nullptr`); the guards that genuinely
stop a give stay (`--adapt-async`, `--prefill-head-alias`, `--peer-device`, `--vram-elastic`, helper caches on
other GPUs, "no residency table", "a cache is not in VMM"), and a rig that still cannot give keeps round 4's
behaviour - back whole, FELL BACK line with the reason, today's numbers, byte for byte.

**If HIP lacks what the elastic VRAM path needs** - the card will now say it itself, and here is the reading:
the fallback above is the mechanism the code proves fires (the arena gate makes it certain), so it is what got
fixed.  The one competing candidate for the ~1 GiB, that HIP charges VRAM for a *virtual-address reservation*
(`hipMemAddressReserve`), cannot be shown without the card - so step 1's table measures it directly: the follow
pools' addresses line prints its bytes asked (262144: ~408 MiB per holder, 1048576: ~1560 MiB + the rope
reservations) beside its free-memory delta.  If that delta is ~0 the addresses are free (NVIDIA behaviour) and
the fix above lands both acceptance clauses; if the delta scales with --max-context, ROCm is committing VRAM for
reservations and the remaining way is the card's other option - plain allocation for the start step with growth
by re-allocation outside the hot path (which forfeits the pointer stability the captured graphs need, so it is
deliberately not implemented before the evidence is in).  The ENGAGED line's `vmm_reason()` string covers the
third possibility (the VMM check itself failing on the card) with the attribute-level reason.

## Step 3 - the acceptance (Ken), and what mortar must now show

With the rule engaged: at both contexts every holder's device context state holds the start step (32768 cells)
plus the trained RoPE rows, so "VRAM in use before the expert cache", the expert-cache slots and the 30K read are
the same at 262144 and 1048576; at 262144 the freed carve means the cache should hold MORE slots than round 3's
2256, not fewer.  Nothing here is GPU-testable on this box (no card): `scratch/card-test-elastic-kv.sh` now greps
the accounting table and the engagement line into its report blocks and gained criterion E: both loads must say
ENGAGED (or the line is the finding), the table's start-step line must be equal at both contexts, the addresses
line's "in use" delta too - and if that delta instead scales, the report's fallback sentence above is the next
step.  Criterion D keeps its meaning, restated: with `SAGUARO_KV_HOST_ELASTIC=0` the engagement line must read
OFF with the reason and the carve lines must show the old whole-context numbers.

## The nits (cheap, taken)

* the give/take lines now say the chunks go "to the slots **or the context state**", and on a follow rig without
  elastic slots the give-on line reads "the context's elastic device state on CUDA%d: ... when a conversation
  grows past the start step" instead of claiming "elastic slots ... when its budget runs short" (the r4 report
  itself flagged this wording); the give-off line adds "(the context's device state cannot grow from the cache's
  room)" when the follow rule was the customer.  (`ec68da4a8`)
* `vmm.hpp`'s header comment said "Not under HIP." while `vmm.cpp` has implemented the HIP branch since upstream -
  it now says what is actually true and points at `vmm_reason()`.  (`9f61edc6a`)
* Not taken: KMEM/HMEM stay `--batch`-budget-gated, and follow-trim still frees chunks to the driver rather than
  back into the cache - both are round 4's known, stated gaps; neither touches this card's numbers.

## Verification

* `cmake -S engines/qfn -B scratch/build-p124e-r5 -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
  -DSTRATA_MMQ_TQ_OVERLAY=OFF -DSTRATA_GGML_DIR=$PWD/scratch/ggml-pinned`; `ninja -C scratch/build-p124e-r5` ->
  440/440, `saguaro-qfn` links.  Warnings in the touched files are the pre-existing ones only (`layer.cpp`'s
  dense-line `-Wmisleading-indentation` set and `n_kv`/`hcd` unused vars, `generate.cpp`'s unused `w`,
  `dflash2.cu`'s old `#550-D/#128-D`); nothing new from `devnotes.cpp`, `vmm.cpp` or the edited lines.
* `ctest --test-dir scratch/build-p124e-r5 -R '^kv_host_growth_test$'` run ALONE (the only module the plan names)
  -> 100% passed; the binary says "kv_host_growth_test: 22 checks OK".  No sizing helper changed this round.
* No GPU here: the accounting deltas, the ENGAGED verdict and the give-on-file-tier behaviour are code-true only
  - mortar's card test is the acceptance.  The OFF path is unchanged byte for byte: every new line is printed,
  no allocation moves when `qsa_vram_follow_wanted()`/`vram_follow_active()` say no, and the fallback branch
  still backs whole exactly as round 4 left it.
* `serve/qfn_engine.py` / `engines/qfn/serve/server.py` are still absent from `git diff --name-only 88809c4e..HEAD`
  (this round touched no engine flag, so no `tools/saguaro/extract_qfn_engine.py` re-run).  `scratch/` stays
  gitignored; the new build dir and its logs are uncommitted.
