# FIX-REPORT - manager FIX-FIRST card round 8 (`e3ec5d022` rig, crash after the staging line)

The manager's round-8 card (`REVIEW-RESULT.md`, committed with this report): LOAD and SPEED pass again, round 7's
YaRN-rows fix works (the rows back on growth, the cache gives way), and the 546,065-token prompt read now
**SIGSEGVs 3.5 s in** - right after "the prompt path's K/V staging holds 786432 cells (a loan of 2030 slots,
chunk 12544)". The card: the staging still sizes itself from a whole-context assumption, not from the elastic
state; check every index against what is BACKED; the crash is an access past a backed range; size the staging
from the state, grow the backing ahead of the chunk about to be touched, never touch an unbacked page; unit
test + new build dir. Three numbered parts, three commits: `12a165e76` (fix 1), `88fa7dc80` (fix 2),
`7ea94ee0e` (fix 3). New build dir `scratch/build-p124e-r8` (CUDA sm_75, ggml from `scratch/ggml-pinned`, cache
diffed against r7's - same flags), 444/444 targets, `saguaro-qfn` links, no GPU touched. No nit items on this
card (three numbered fixes only).

## Finding 1 - the exact access, and every index the read makes through it (`12a165e76`)

**The gate.** `Prefill::run_impl` has the read's one and only index gate: a segment of a run reaching past the
staging was refused with "the K/V staging holds N cells" (prefill.cpp:3282-3289 before this fix). It compared
`sg.p0 + sg.n` with `m.stage_cells` - the LOAN's cells, 786432 on the card, sized by `stage_cells_for` =
reach + a quarter rounded up to the grid, capped only by `max_cells`. The conversation's BACKED cells - the
bytes that actually exist behind the addresses - entered nowhere. Everything the read's kernels index by cell
hangs off that gate:

| what the read indexes by cell | index line | the array's real backed extent |
|---|---|---|
| the pinned host copy - the staging copy's source (`kv_stage_blocks_from_host`, prefill.cpp:3327 / 3805) | page `= cell/4` of each array | the registered pieces: `qsa_host_backed_cells` (layer.cpp:1161) - the card: **546148 cells = 136537 pages** (log lines 149-167, "the host copy backs 546148") |
| the device follow state - page table + pooled indexer keys (append, reset bound, selection's gather) | `table[cell/4]`, pool rows | `qsa_table_backed_pages` / `qsa_follow_backed_cells` (layer.cpp:1802 / 1813) - grown by the same fit, the same number |
| **the RoPE rows - the read rotates every position it touches** (`rope()` prefill.cpp:3755 -> `rope_tab_cs`) | mrope.hpp:95-100: `c = __ldg(t.cos + (size_t)(p - t.min_pos) * 32 + pair)` | **`rope_tab_cs` answers "in range" for any `p < t.max_pos`, and `rope_table_set` registers `max_pos = kRopeYarnOff + max_cells` - the WHOLE context (layer.cpp:2739-2741; the plain table likewise at 2744). The table is a VMM range with only `RopeGrow::cells` rows mapped (layer.cpp:2069, `rope_map` 2119-2136): a position past `r->cells` passes the table's own bound and steps off the mapped prefix. Card: backed 557056 rows ("covers 557056", log 170/175), extent 1048576.** |

That last row is the crash mechanism the card asked for by name: the RoPE table is *registered* for the whole
context and *backed* for a prefix, and the kernel's range test uses the registration. Any read - the prompt
itself, a checkpoint re-read, a verify window - that rotates a position past `r->cells` (or addresses host
pages past 136537, or follow pages past the mapped prefix) takes a load/store at a VA whose VMM range has no
page there: **a fault, not a refusal** - the driver's guard under the reservation. The loan's grid is the
reach of that index space: 786432 cells = 196608 staging pages, and the backed state ends at 546148 / 557056 -
cells (546148, 786432] have addresses and no memory, and the old gate waved through anything up to 786432.

Stated straight, as the card asked: with the crash log's numbers every logged backed prefix (host 546148,
rope/yarn 557056) still covers the prompt's own reach 546065, so this GPU-less box cannot point at the one
in-range test that was defeated on the 9070 XT (the driver's own reservation granularity under memory
pressure is the standing suspect for the follow-pool or rope range). What the audit CAN prove, and what is now
fixed, is that a single gate compared the read's reach with a LOAN SIZE and nothing else - so every access in
the table above was one headroom step away from the guard, and the fix closes all of them at once:

* `qsa_rope_backed_cells(st, yarn)` (new, layer.cpp:2274) - the rows a table actually has; a whole (ungrown)
  table answers its extent.
* `session_backed_cells(s)` (new, session.cpp:201) - host pieces, follow state AND the rope rows (plain, plus
  YaRN while `rope_level`) in one number, capped by the context; pinned-whole sessions answer `max_cells`.
* the run's segment gate (prefill.cpp) now refuses any segment reaching past
  `min(stage_cells, session_backed_cells(segment's own session))` once `Prefill::set_backed_limit(true)` is on -
  on for the solo request's reads wherever the session's host copies are elastic; the refusal line says
  "the K/V staging and the backed context hold N cells, the chunk reaches M", so a card that still trips on a
  real access names the cell and the arrays instead of faulting.

## Finding 2 - staging sized from the state; the read backs up ahead of its chunk (`88fa7dc80`)

New pure header `core/read_backing.hpp` (the sizing and the order, unit-tested - see finding 3):

* `stage_cells_elastic(reach, backed, full)` - the loan's existing grid rule capped at the BACKED cells plus
  one grid step (131072): the step the read's grow runs ahead of its pieces. Below the grid's base the base no
  longer out-promises the backing either (`backed + step`, never a promised 262144 for a 32768-backed state);
  monotonic in the backing; nothing elastic (`backed < 0`) or all backed - today's sizing, word for word.
  `Prefill::stage_cells_elastic` wraps it (session `max_cells`, `SAGUARO_STAGE_HEADROOM`), and `lend()` /
  `stage_covers()` size and match every loan through it (generate.cpp).
* The solo read's `run()` is a piece loop: each piece ends at `read_piece_end = min(r, session-backed,
  staging)`; when that stops short of the piece, the conversation is backed ahead FIRST - `solo_host_fit_r` +
  `solo_vram_fit`, the very fits the request start uses (give-way and relief included), in `kSoloPinAhead`
  steps, capped by the staging and the request's end - then the piece runs. A read that cannot be backed
  further, or reaches past a loan that cannot grow mid-read, ends with "no VRAM/no RAM ... (out of memory)"
  on the read's existing soft-fail path (the request fails, the engine goes on) - it never issues the access.
* The card's own numbers keep today's fast path: prompt 546065, backed 546148 - backed + step rounds to the
  SAME 786432-cell loan and the read is ONE piece with zero grows (asserted in the test). A card backed for
  100000 lays out 231072 staging cells instead of 786432 - a smaller loan is slots the cache keeps for experts.
* Kept as is, on purpose: `SAGUARO_PIPED_CONCURRENT=1` reads call `sp.run` straight (their pump owns the
  stepping) - they gain the backed gate (refuse, not fault) but not the mid-read grow; pinned-whole rigs take
  none of this (`check_backed` off, `backed = max_cells`, the run lambda is the same single call).

## Finding 3 - the tests, the build (`7ea94ee0e`)

`read_backing_test` (new, 82 checks, registered beside `park_policy_test` / `kv_host_growth_test` /
`vmm_grow_test`): the plain grid unchanged (the card's 546065 read is the 786432 loan of the crash log); the
elastic loan (nothing-backed = one step, 100000-backed = 231072, base does not out-promise, monotonicity and
"never past the reach's size" as property loops); the ORDER: a simulated read on a 131072-backed card asserts
per piece that it never touches a cell past the backed or the staged end, walks to exactly the staging's end
in < 16 ahead-step grows and ends on the ERROR where today's engine faults; and the healthy card's numbers stay
the single run they were.

## Verification

* `scratch/build-p124e-r8`: fresh configure (cache diffed against r7's: same flags), `ninja` 444/444 clean.
* `ctest -R '^read_backing_test$'` run ALONE -> passed (82 checks); `kv_host_growth_test` run ALONE -> passed
  (22); `vmm_grow_test` run ALONE -> passed (44). Nothing else run; no GPU here, no card test run here.
* Card test owed by the rig: the 546,065-token request must answer (needle "7319"); load + speed unchanged.
  With this fix the read can only answer, refuse with the cell named, or fail with a no-VRAM/no-RAM line -
  the staging can no longer promise past the backed end, the run gate checks the backed end per segment, and
  every piece is backed before it runs.
