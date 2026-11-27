# FIX-REPORT - manager card round 10 (`REVIEW-RESULT.md` + MANAGER UPDATE 2: the first cause is disproved - the slot guard + 40K repro FIRST)

What the card asked this round, and what this box did, in its own order. Straight first: the card's decisive
step is the 40K-token repro **on the card** with the guard on ("the manager runs it"); this box has no GPU, so
what follows is the guard itself - shipped, tested, off by default - not a speculative rework of the give-ways.

## Finding A (run 1: `done: 0 tokens (length, cancel=False)`; run 2: `hipMemCreate` refusal with ~7 GiB of experts resident) - already fixed in round 9; cause lead superseded by the card; guard now stands

The r9 fixes for both run-modes are committed (`82ba099e2` relief frees until the ask fits, `22d3cf546` the
silent ends closed - the zero-token clean end now prints the numbers that name the culprit). UPDATE 2
disproved the "give takes a slot inside the active loan" cause (the `room_p` clamp keeps a loan at or above
the setup give line, and a loan layout takes its residents off `host_res`), and named the remaining lead:
**the two give-ways start from the same lend line and can hand the same chunks out twice**. That is now
detectable in one card run - see the guard below. The fix for the path it names deliberately waits for that
run: the two bands share one top (`kvg.top = lend_first = v.top`) by design, so any blind clamp of one at the
other's line would simply starve whichever give-way moves second - the guard's abort line names who crossed,
and the fix is then targeted, which is exactly what the card sequences ("DO THIS FIRST ... Fix that path").

## Finding B (exit: SEGFAULT in `VmmRange::release()` after `exit(0)`) - fixed in round 9, nothing new needed

`26666fe8d` (core/vmm_exit.hpp: atexit handler armed at the first VMM api() use; from then on `release()` and
`vmm_chunk_free()` make no driver call). No serve file changed then or now, so `serve/qfn_engine.py` stays as
extracted - it was not touched and nothing needed re-extraction. Card test (SIGTERM after growth exits 0) is
the manager's; a GPU-less box cannot raise a HIP runtime.

## MANAGER UPDATE 1 (the give range fixed at setup overlaps the per-request loan; unit test: a fake cache with a loan resized after setup, a give must not touch `[p.first, slots)`) - cause disproved by UPDATE 2; its TEST and its CHECK ship

UPDATE 2 showed the give as written neither unmaps nor moves into an active loan (room_p + kNotResident). But
the test the update asked for exists (`slot_give_guard_test`: "THE ACTIVE LOAN, including re-sized AFTER
setup"), and the check it encodes is now LIVE on every handout while the guard is armed - a give, move, unmap,
remap or relayout that reaches `[p.first_now, slots)` prints the loan's owner and aborts, whether or not the
r9/r10 clamps are what keeps them apart today.

## MANAGER UPDATE 2 / DO THIS FIRST - the guard: shipped, kept, off by default (`01c43ade7`, `12390e6d2`, `f87b4f143`)

* `engines/qfn/include/strata/core/slot_give_guard.hpp` (new, pure - unit-testable like `vmm_grow.hpp`):
  `slot_give_guard_line` answers, for one op (path + what + slot range + chunk range), (a) is any of it inside
  an ACTIVE loan `[first_now, slots)` of a part on that cache, and (b) is any slot/chunk of it inside the
  region the OTHER give-way handed out (regions are the give-ways' live `[lo, top)` bands, slots AND chunks);
  a crossing is ONE line naming the path, the slots/chunks and who owns them; `slot_give_guard` prints it to
  stderr and `abort()`s - the core dump's last word. Loan before region (the active read's memory is named
  first). Same-path ops never cross their own region; ranges that touch but do not overlap are clean; an op
  that cannot name its slots is not accused of reaching into a loan (chunk checks still apply - that is the
  free-to-driver case).
* Wired in `generate.cpp` at EVERY handout the card lists, each hook gated on `slot_give_guard_on()` (off: one
  test of a static bool - no driver call, no allocation, no log; the off path is what it was):
  kvg_ensure's per-slot give, expert move-into, and chunk unmap; kvg_trim's take-back `map_range` and its
  spare free-to-driver; gv `give()`'s per-slot give, move-into and chunk unmap (the VRAM relief and
  `free_to_driver` run through it); `take_back()`'s budget remap; `free_to_driver`'s release (it knows chunks,
  not slots - the named range is the top `n` chunks of the card's given band); and `lend()`'s loan relayout
  (its own loan is what is being laid out, so (a) checks the OTHER parts on that cache, (b) both give-ways'
  regions). Live providers: `guard_loans` reads every `PfPart::first_now` at call time (the loan is re-sized
  per request), `guard_regions` reads `kvg.lo/top` and every `gv[k].lo/top`. Arming prints one line.
* `slot_give_guard_test` (CMake, beside the other P124e modules): 23 checks - both card scenarios (the loan
  re-sized after setup; two give-ways counting down from one line, the second's give crossing the first),
  loan-before-region, own-region clean, per-cache separation, chunk-only ops, touching ranges, the printed
  line, and off-is-free.
* `docs/SETTINGS.md`: the `SAGUARO_DEBUG_SLOT_GUARD` line.

## "Build ONLY the QFN engine (tools/qfn/build.sh <new dir>, AVX floor)" - done; tests run one at a time

`QFN_GGML_DIR=scratch/ggml-pinned bash tools/qfn/build.sh scratch/build-p124e-r12` (fresh dir; the script's
own AVX floor + sm_75 options) -> **180/180 clean, `saguaro-qfn` links**; `generate.cpp` compiles with only the
pre-existing `w` warning (same line the r10 log shows). Run ALONE, one at a time: `slot_give_guard_test` 23,
`read_backing_test` 82, `kv_host_growth_test` 22, `vmm_grow_test` 56 -> all pass.

## What this box cannot prove, said straight

The crossing itself is a GPU fact: without a card nothing here can show whether kvg and gv really cross on
mortar's rig, which op crosses first, or that the guard's abort fires - the 40K repro at `--max-context
1048576` with `SAGUARO_DEBUG_SLOT_GUARD=1` is what names the culprit (one line, one core). With the env unset
the engine behaves as before (every hook is the flag test; the pure check is proven by its own module). Once
the card's line names the path, the fix is one targeted change - the guard stays in (off) to keep it honest.
