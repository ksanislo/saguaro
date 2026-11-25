# REPORT - F1: QFN VRAM give-back in the resident low-RAM layout

Plan: `./PLAN-f1-qfn-lowram-giveback.md`. Commits (this worktree, branch
`saguaro/q1-f1-qfn-lowram-giveback`): `scratch/notes.md` (step 1), generate.cpp (step 2),
file_expert_source_test.cpp (step 3), this report (step 4).

## What changed

**Step 1 - analysis only** (`scratch/notes.md`, committed). What the complement layout assumes about
VRAM slots, mapped per structure:

- `host_res` (generate.cpp 8096) - the only slot-number table: residency index -> cache slot /
  kNotResident, mirrored to `d_res` (and the stages' copies) by `res_put`/`res_upload`. The VRAM
  command already maintained it through shrink/grow; nothing else needed changing in it.
- The adaptive tier's swaps (`adapt` 11564, `a_pick` 11831, `resident_stage_swaps` 453): `in` comes
  from `host_res < 0` and `out` from `host_res >= 0` - never from "it must be in RAM". The exchange
  staging (`stage_exchange`, `commit_flip` -> `detail::exchange_cache_complement` 301) is keyed by
  the RAM copy's `complement_offsets_`, not by slots, and refuses an `in` the copy does not hold or
  an `out` it holds already; those swaps run un-staged (`a_from` reads the file through a bounce).
- `exchange_storage_` (SAGUARO_EXCHANGE_ROTATE): ownership table standing in for the offset moves -
  same RAM-keyed rule.
- `resident_lent_slots()` (`complement_lent_slots_`): consumed at STARTUP only (prompt plan 8504,
  warnings 8954/9519); `relend()` re-lays the loan against `xcache.slots()` after a resize.
- The miss path's residency query (expert_source.cpp 2107/2140/2166, `blob()` 4560): per expert,
  `resident_blob(index)` -> RAM, else the file (`file_reads_` counted). The source stores no "in
  VRAM" fact at all: `complement_offsets_` says `kNoCacheComplement` for every expert the RAM copy
  does not hold - which every cache expert already is. **So an eviction needs nothing "told" to the
  source: the evicted expert's existing table entry already means "read it from the file".**
- Why the refusal existed: `git log -S` shows the string was introduced by the #533 feature commit
  itself, 89bad9a4e (2026-10-03, "Hot VRAM resize, opt-in"), as the scope guard for the untested
  combination ("One NVIDIA GPU only (no layer split, helper caches, --peer-device or the resident
  low-RAM mode)"). `--resident-budget-gib`'s file tier inside the complement layout (1e663cc70,
  three days earlier) had already made "expert in neither tier, read from the file" a shipped,
  steady-state configuration - which is what makes the give-back composable now.

**Step 2** - `engines/qfn/src/program/generate.cpp`, `vram_command` only:

- Removed the `src.complement_ready()` refusal.
- Shrink: land the SAGUARO_SHORT_READS lazy refills (`lazy_land(true, e)`) after `adapt_quiesce()`,
  before the segments go - so a pending lazy refill cannot set `host_res` back to a slot the shrink
  takes away (`lazy_land` otherwise runs at the next window tick with the stale slot).
- Grow: the refill read `srcp->blob_stable(layer, pick)` returns the pack's bytes in the complement
  layout (RAM copy for a lend-region expert held in both tiers; mapping/assembled blob otherwise).
  Where the source has no stable pointer (file tier with the mapping closed - Windows, or Linux with
  SAGUARO_TEST_UNMAP_GGUF) it now falls back to `srcp->copy_blob` into a pageable bounce buffer
  instead of failing the whole command.
- Grow error path hardened: a failed read/fill keeps the entry (and every entry after it) in
  `vram_evicted` for the next grow, and the command now lands the queued copies (`sync_queued`) and
  uploads the residency table (`res_upload`) before returning `ERR` - the old code returned with
  host `host_res` already refilled for the earlier entries while `d_res` still said kNotResident
  (host pool would zero their rows; the device would not compute them).
- Header comment extended with the complement behaviour. No other guard touched (peer, open loan,
  batch slots, and the startup `--vram-elastic` exclusions for layer split / helper caches stand).

`engines/qfn/serve/server.py` needs nothing (it forwards `VRAM` to stdin and surfaces ERR);
`serve/qfn_engine.py` untouched (generated; not regenerated).

**Step 3** - `engines/qfn/tests/core/file_expert_source_test.cpp`: new `test_vram_giveback()`
(pure, registered in `main()` beside `test_complement_plan`). It pins the exact invariant the
engine change relies on, at the level the source knows about: with the complement pinned, an evicted
cache expert resolves (via `cache_complement_blob_or_fallback`) to the file exactly like the
budget's file tier; a lend-region expert keeps its RAM place and those bytes equal the file's; and
`exchange_cache_complement` refuses the evicted expert as `in` / a copy-held expert as `out` (so the
tier's staged path can never corrupt the table through an evicted expert), while a real exchange
after a give-back still moves the place correctly.

## Why it stays bit-identical

The weights never move: an evicted expert's bytes are still the pack's file bytes (the RAM copy was
built from them at startup, and adaptive exchanges only move bytes between places, never rewrite
them). After a shrink the expert is computed on the CPU from `blob()` -> the same file bytes a
budget-tier miss reads; after a grow the slot is refilled from `blob_stable` -> the same file bytes
it held before the shrink. The GPU-vs-CPU expert answers are unchanged by construction (the hit/miss
split is the profile cache's ordinary one), so greedy answers are byte-identical across
`VRAM 2048` / `VRAM`; only WHERE misses read from changes - exactly the behaviour #533 already
promised for the plain layout ("their experts become CPU misses, as any expert outside the cache
is").

## Manual check (run on the field-class box; could not run here - no GPU model setup in this worktree)

On a PC whose RAM cannot hold every expert (field case: Tesla P40 24 GB, 31 GB RAM), with the
resident low-RAM setup:

1. Start the server with `--vram-elastic` + a small `--resident-budget-gib` (the fit policy's
   low-RAM arm), `--serve`, greedy sampling. `INFO ... vram_elastic=1` must be present; the
   complement's startup lines say the RAM copy was pinned.
2. Send a prompt greedy -> answer A0 (also serves as the no-`VRAM` baseline).
3. `VRAM 2048` (or POST `/v1/vram` `{"reserve_mib": 2048}`). Expect the `VRAM reserve_mib=2048
   expert_slots=<n<full> ... expert_cache_mib=...` line, no `ERR`.
4. Same prompt greedy -> A1: must match A0 byte for byte.
5. `VRAM` (bare - back to the start reserve; the grow path refills). Expect expert_slots back to the
   full count.
6. Same prompt greedy -> A2: must match A0.
Worthwhile extras: a prompt read between steps 3 and 4 (moves the lend region down, exercises
`relend` and the streamed lend experts), a couple of turns in between (drives the adaptive tier's
swaps over the shrunk table), and SAGUARO_XCACHE_VERIFY=1 on a debug build (its between-request
check compares every slot's bytes with `blob_stable` - it validates the refilled slots in exactly
the complement states this change creates).

## Tests run and results

- New `test_vram_giveback()` added to the tree's existing complement module
  (`file_expert_source_test.cpp`); `expert_cache_segmented_test.cpp` (the elastic cache's
  shrink/grow machinery, unchanged) remains the mechanism test. **Neither binary could be linked or
  run here**: no engine build directory exists under ./scratch, and the modules link the whole CUDA
  engine (several parts are additionally GPU-gated behind `--rotation-gpu`). The plan allows
  skipping a from-scratch build; it was skipped, and it is also why the manual check above is
  spelled out rather than executed.
- Compile check instead (in place of `cmake --build`, per the plan): `g++ -std=c++20
  -fsyntax-only -DSTRATA_VERSION='"0"'` with the engine's include paths and the CUDA 13.1 headers -
  clean on `engines/qfn/src/program/generate.cpp` (the edited engine file), on
  `engines/qfn/tests/core/file_expert_source_test.cpp` (the edited test), and on
  `engines/qfn/src/core/expert_source.cpp` (unmodified, as a control). `expert_source.cpp` and
  `expert_cache.cpp` additionally compile to objects cleanly; the test's final link pulls in the
  kernels/arena/peer TUs (some CUDA-only), so it was not linked by hand.

## Files

- `engines/qfn/src/program/generate.cpp` - `vram_command` (+ its header comment): refusal removed,
  lazy-refill drain before the shrink, bounce-read fallback + safe error path in the grow refill.
- `engines/qfn/tests/core/file_expert_source_test.cpp` - `test_vram_giveback()`.
- `scratch/notes.md` - step 1 write-down.
- Not touched: `serve/qfn_engine.py` (generated), `engines/qfn/serve/server.py`, kernels, the
  complement source (`expert_source.cpp/.hpp`), `expert_cache.cpp`.

## Risks

- A shrunk cache puts the tail slots' experts on the file tier: on a 31 GB-RAM P40 box those decode
  misses read the SSD (or whatever the page cache kept) - slower decode while the reserve is out, by
  design; the picture reader's ~1.5 GB need is what the reserve buys.
- `file_reads_` is now non-zero in a steady `--resident-cpu-experts` run after a shrink; the header
  comment ("0 in a steady resident mode") still describes the no-VRAM-command case. Stats only.
- `resident_lent_slots()` and the startup lend warnings are not re-planned after a resize: a grown-
  back cache's new lend region can include slots whose experts have no RAM copy, so prompts stream
  those from the file until the cache grows back - the documented #669 "only N of the lent experts
  fit in RAM" shape, not a correctness issue.
- A refill that keeps failing (disk error) is retried by every grow and the command keeps answering
  ERR; the tables stay consistent (that is the hardened path), but nothing auto-gives-up.
- The grow failure path change (sync + upload before ERR) also alters the plain layout's behaviour
  on mid-loop failure - from "half-applied, no upload" to "applied what succeeded, uploaded, ERR".

## Open questions

- Should `/v1/status` / the lend-sizing line re-report RAM coverage of the lend region after a
  resize? That would need `pin_cache_complement` to re-plan (it refuses when already pinned); left
  out of scope by the plan.
- Upstream's `docs/VRAM_ELASTIC.md` (not in this pruned tree) still says the command is "not with
  the resident low-RAM mode" there; the doc should be amended upstream when this lands.
- The manual parity check above needs the real P40-class box (weights + GPU); it is the acceptance
  test for this change.
