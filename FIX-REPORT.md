# FIX-REPORT - manager card round 12b (`8e8cf0982`: the 546K stall is a GPU page fault (dmesg) - VA logging first, then a serialised run)

Straight first: this box has no card that stalls, so what it can ship is the instrument - every address-space
op printing itself - plus the two findings the card's serialised run and the merge diff already proved. Three
commits, in the card's order: A, B0, B1. No serve-side change was needed or made: `serve/qfn_engine.py` is
untouched (rule held all twelve rounds - if serve ever changes, the change goes into
`engines/qfn/serve/server.py` and `python3 tools/saguaro/extract_qfn_engine.py` re-extracts; nothing here
required that).

## A (`1dc74054a`) - the resident-budget margin: the merge regression, sized right

The P124x resident-budget margin counted the K/V the host copies COMMIT (the `--max-context` ceiling), not
what they actually back - the merge (57fac6ad8) widened that accounting and made the resident experts plan
67.9 GiB where the pre-merge engine planned 30.8 at the same 1M context. The margin now counts the committed
copies; `resident_policy_test` 19 checks pass.

## B0 (`1b4119a28`) - the faulting reader is the prompt `rope_kernel<true>` (serialised HIP)

The card's serialised run localises the page fault: with kernels serialised, the faulting reader is the prompt
path's `rope_kernel<true>` - a reader of the RoPE tables over positions the prompt walks, running while some
range it reads is not mapped (or just was unmapped). That is exactly the fact B1's log is built to name.

## B1 (`114621b4f`) - `SAGUARO_DEBUG_VA`: every address-space op says its own name

New pure header `core/va_debug.hpp`: `va_debug_on()` (static bool, read once from the env),
`va_debug_format()` (the line), `va_log(op, dev, addr, size, what)`. Off, every hook is one static-bool test.
On, one stderr line per op - `op dev N [0xbase, +bytes) owner` - at:

- `VmmRange` (both the CUDA and the HIP branch, `vmm.cpp`): `reserve` (the cuMemAddressReserve/
  hipMemAddressReserve), `map` (what landed - the partial commit of a run prints the run that made it),
  `unmap`, `release`; `set_va_tag()` names the owner before the first reserve.
- every owner in `layer.cpp`/`expert_cache.cpp` tags its range: `rope cos/sin`, `yarn cos/sin (YaRN-level
  table)`, `scores scratch`, `slot pool (elastic kv/indexer arrays)`, `kv pools (vram-elastic arrays)`,
  `follow pool (device follow state)`, `expert cache arena (vmm)`.
- every pinned-host registration and deregistration in `layer.cpp`: the whole-run copy (`pin`), the elastic
  pieces (`pin`/`unpin`, the give-back and the no-alias bail included), the prefixes (`pin`/`unpin`, the
  refusal bails too), plus the copy's whole host VA range at `reserve`.
- the slot budget's chunk handouts: `budget+` (cache -> budget), `budget-` (budget -> a grow),
  `budget-free` (budget -> driver) - by handle, since a chunk has no address until a pool's `map` gives it one.

A faulting address from dmesg now falls in exactly one logged range, and the log says who reserved it, who
mapped it, and whether anything unmapped it while a kernel could still read it. `docs/SETTINGS.md` gains the
knob's line; `kv_host_growth_test` gained 3 format/gate checks (25 total).

## Build and tests (run ALONE, one at a time)

Fresh `scratch/build-p124e-r13` (same cache as r12 - CUDA sm_75, AVX floor, tests ON; `conversation_snapshot_test`/
`conversation_transfer_test` excluded, see the heads-up below): full build clean, no warnings in the touched files.
Then, each alone: `kv_host_growth_test` 25 OK, `resident_policy_test` 19 OK, `vmm_grow_test` 56 OK,
`park_policy_test` 56 OK. A wider ctest pass (parity excluded) ends with only `pool_cores_test`,
`pool_tasks_test`, `platform_memory_test` failing - and those three fail identically at pristine HEAD in a
throwaway worktree build: this sandbox's CPU topology and mlock ulimit, not this round's changes (the GPU
benches that flaked under -j8 pass alone).

## Heads-up, outside this card, not touched: `conversation_snapshot_save` does not link

The header declares `conversation_snapshot_save` twice - `const QsaState&` (hpp:113) and `const QsaState*`
(hpp:152) - and only the POINTER form is defined. `conversation_snapshot_test.cpp` and
`conversation_transfer_test.cpp` still call the reference form, so a full clean configure with
`STRATA_BUILD_CONVERSATION_TESTS=ON` fails to link those two test binaries. The merge brought it (both sides
changed the signature's neighbourhood); r12's build never reached those targets, so it went unseen. Left for
the manager's call - it is a one-word test change or the missing overload, but it is not this card's file.

## What this box cannot prove, said straight

That the stall's fault address correlates with a logged range - that needs the card's next run:
`SAGUARO_DEBUG_VA=1` with the serialised prompt. What this box proves is that the instrument sees every door:
every reserve/map/unmap/release and every pin/unpin in the elastic machinery now says its own name, so the
`rope_kernel<true>` fault's address either falls inside a range the log shows as still mapped (then the reader
went past the extent - the r8 table question again) or inside one it shows as unmapped mid-flight (then the
unmapper is named on the line above). One of those two lines will be the next card's first sentence.
