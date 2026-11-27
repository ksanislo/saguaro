# FIX-REPORT - manager FIX-FIRST card round 9 (`8dee8bf1c` rig, the 546K request still does not answer)

The manager's round-9 card (`REVIEW-RESULT.md`, committed with this report): LOAD and SPEED pass again, the
SIGSEGV the last two rounds chased is NOT in the read (it is at exit - finding 2), and the read ends early two
ways. Finding A run 2 is the loud one: "the context's device state could not be grown ... 70 MiB asked",
"the expert cache was asked to give way of 70 MiB and gave 70 MiB" - while `2396 of 2541 slots` still held
experts (~7 GiB). Finding A run 1 is the quiet one: `done: 0 tokens ... (length, cancel=False)` 3.5 s after
"the prompt path's K/V staging holds 786432 cells", no ERR. Three commits: `82ba099e2` (fix 1 - the relief
frees until the ask fits), `22d3cf546` (fix 2 - the silent ends around the read), `26666fe8d` (fix 3 - the
exit order). New build dir `scratch/build-p124e-r10` (CUDA sm_75, ggml from `scratch/ggml-pinned`, the options
byte-match the card-verification r9 build's), clean link; the three named modules each run ALONE -> pass.

## Finding A run 2 - the relief freed once and gave up; now it frees until the ask fits (`82ba099e2`)

The card is right about the mechanism: the give-way hands the slot/context budget chunks, but a follow grow
maps NEW physical chunks (`hipMemCreate`), so what must reach the DRIVER is physical memory - and
`vmm_grow_room` (vmm_grow.hpp) asked the expert cache exactly ONCE per grow (`st.relief_ask == 0` guard),
retried the next chunk once, and failed the whole grow. The cache goes floor by SLOT (`give()` drops `v.lo`
one slot at a time to `give_floor = 128`, and its log says so: "CUDA0's expert cache gives 48/60/62/72 MiB
... 2524 ... 2396 of 2541 slots hold experts"), so a single ask hands back less than a 35-chunk ask needs -
run 2's eight grow attempts shed ~420 MiB in eighths and refused each time, cache nowhere near its floor.
Now (the card's fix, its wording):

* every refusal asks the relief for the WHOLE shortfall still missing and retries; when the retry still fails
  AND the relief handed back bytes, it is ASKED AGAIN - until the ask fits, the relief answers 0 (at the
  floor: `free_to_driver` gives no cache once `gv[k].lo` is at `give_floor` and the budget spare is gone), or
  `kVmmReliefRounds = 24` rounds of bytes that never become chunks.
* `st.relief_ask` / `st.relief_got` accumulate across the rounds, so the grow's failure line says what was
  asked and what was freed IN TOTAL; every individual shed keeps its own stderr line (give()'s "gives N MiB
  ... N of 2541 slots hold experts") - the card's "log what was freed" is satisfied at both granularities.
* `free_to_driver` device-syncs after handing released chunks to the driver. Run 2 "gave 70 MiB" and failed
  the very next `hipMemCreate` with 0.13 GiB free: physical memory a release lets go of is reclaimed once the
  card's work has drained, and the grow's retry must see it. (Straight: on a GPU-less box this is the
  half of the fix I cannot prove; if the card's next run shows the loop asking 24 rounds and still failing
  with the totals growing, the reclamation is NOT the driver's to reuse - then the card's second alternative,
  remapping the given slots' chunks instead of releasing them, is the one to build.)

`vmm_grow_test` (54 checks): the asked-ONCE expectations become the ask-until-the-floor order; new cases:
a cache that frees one chunk per ask covers a 4-chunk grow in four asks (10G asked / 4G given on the line),
a cache that runs dry mid-grow is asked ONE more time at the floor and refuses without mapping, and relief
bytes a `make()` can never use stop at `kVmmReliefRounds`.

## Finding A run 1 - what ends a read as "length" with 0 tokens, said silently (`22d3cf546`)

Stated straight first: run 1's engine log is not in the card's evidence file (the file holds the core, run
2's engine lines and the server output), and this box has no GPU, so I cannot name THE line that ended run 1.
What I did instead: every way the engine can reach DONE with nothing said, checked against the card's
signature - and every one of them that exists is now closed.

* RULED OUT: the r8 refusals. They all carry "(out of memory)", hit the `vram_short` soft-fail, and end
  `DONE ... vram` - the client raises `EngineOutOfVram` and the server prints "(error, cancel=False)" plus
  "the engine reported an error" (the evidence file's own line 96 shows that pair). Run 1 said "(length,
  cancel=False)" with no such line, so its DONE word was CLEAN: the read "succeeded" and nothing produced.
* CLOSED (1): the read's piece loop ignored `Prefill::stopped_at()`. `stop_at_chunk` (the BYIELD / pending
  VLEND hook) makes `sp.run` return TRUE after stopping at a chunk - and the loop advanced `q = qe` over the
  cells it never read, then reported the piece read. A stopped piece now returns "stopped, not failed" with
  the stop cell ABSOLUTE (the driver's `q0 + stopped_at()` recomputation is only right while ONE `sp.run`
  covers the segment - with pieces it would resume at the wrong cell and re-read).
* CLOSED (2): the PIPED-CONCURRENT reader ran bare `sp.run(ids+q0, b0-q0)` - r8's backed cut, back-ahead fits
  and read clock were non-concurrent-only, so on that path only prefill's refusal gate stood between a long
  read and an unmapped cell. It now runs the SAME piece loop (own error string, `read_stop` out).
* CLOSED (3): the silent zero itself. A request that reaches DONE having produced nothing, not cancelled,
  not yielded, finish still "length", now says so on stderr with the numbers that name the culprit:
  "the request ends with no tokens and no reason: the read reached X of N cells, the decode P, max_new M,
  the staging holds S cells, the conversation is backed to B[, the pipelined decode ran]". Next card run
  reads a reason instead of a silence - a staging cap, a reach check and a backed end are all in the line.
* CLOSED (4): `follow_units_cells` under-reported the pooled rows by `idx_block - 1` cells - it answered
  `(U-2) * idx_block` where the grow's need `cells / idx_block + 2` inverts exactly to `(U-1) * idx_block -
  1` - enough for the piece loop to refuse cells the request-start fit had just backed (a spurious
  "no VRAM is left to back the read's context" on a card that had them).

## Finding B - the exit order: no HIP call after the runtime is gone (`26666fe8d`)

The core says it: `exit(0) -> __run_exit_handlers -> ~vector<unique_ptr<ScoresPool>> -> VmmRange::release()
-> call through a null pointer in libamdhip64`. The static pool vectors (layer.cpp's follow pools, block-score
scratch, slot pools, rope tables) are built at load, so their atexit hooks register BEFORE the HIP runtime's
own teardown and run AFTER it. New pure header `core/vmm_exit.hpp`: the first VMM `api()` use - long after
the runtime initialised - arms an atexit handler (LIFO: it runs FIRST), and from then on `VmmRange::release()`
drops its bookkeeping and `vmm_chunk_free()` returns with NO driver call; the ranges are the kernel's to
reclaim at exit. Both guarded branches (CUDA's `cu*` path takes the same static-destructor hit), and the
fallback no-VMM branch needs nothing. The host pools' destructor makes no HIP call and the budget's chunk
list is plain handles, so the flag covers every static owner the card named. `vmm_grow_test` +2 checks (the
flag is false while the engine runs; arming twice registers once; marked = the runtime is gone), 56 total.
The card's "SIGTERM after growth exits 0" is a GPU-rig test; what this box proves is that the exact function
in the core's frame #3 now returns without touching the runtime once the exit handler has run.

## Tests and build

`scratch/build-p124e-r10` (fresh; `-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75` +
`STRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_GGML_DIR=scratch/ggml-pinned
-DSTRATA_MMQ_TQ_OVERLAY=OFF -DSTRATA_PLE_FIXTURE_DIR=engines/qfn/bench/micro` - the cache's options
byte-match the card-verification r9 build's), full build clean, engine links. Run ALONE, one at a time:
`read_backing_test` 82, `kv_host_growth_test` 22, `vmm_grow_test` 56 -> all pass; `serve/test_vram_refusal.py`
4/4 (the r6/r7 ERR-wording contract still holds - no serve file changed, so no re-extraction needed).
