# P8 report: the four opt-in fixes from review-u9 findings 1, 2, 4, 5 (src/ik2/ik2_main.cu)

All of it sits behind the SAGUARO_IK2_* switches the reviewed code already uses; the default path is untouched
(every touched site is inside `dg > 0` / `g_drem.dev >= 0` / `pf1_pick`'s `SAGUARO_IK2_PF_GPU1` gate).
One commit per numbered item, only `src/ik2/ik2_main.cu` changed: 30 insertions, 6 deletions.

## 1. SAGUARO_IK2_DRAFT_GPU: the Q8 head copy at drafter load (commit 015f18123)

The raw `cudaMemcpyPeer` pair (old :9083-9084) ran before `two_setup` established the link - its link test is
what admits peer copies and its setup what enables peer access - so a pair whose access was not enabled then
died at start, and where it was, the load bypassed the test the peer-copy rules exist to gate. `pf1_peer()`
could not be the gate here: it latches on its first call, and at drafter load `g_two.on` is still false, so
calling it would pin the whole run to host staging. The copy is now gated on `g_two.link.all_peer` (the
engine's link test result, false at this point today, kept for a link that is ever tested before the drafter)
and otherwise bounces through a pinned host staging block (`qb + db`, `cudaHostAllocPortable`: both D2Hs - the
`q8` into `stage`, the `q8d` scales into `stage + qb` - with GPU 0 current through a `DevScope`, both H2Ds on
`dg`, freed right after) - the same way every other hand-off of the
feature goes when the test fails. Not moved after `two_setup`: the drafter load (and `fold_q8`/`set_head_q8`
on `R.q8`/`R.q8d`) must precede the cache sizing for the VRAM reasons spelled out above that block.

## 2. pf1_run's grow path: sync with the right device current (commit 5b7e7e000)

In `pf1_run`, the `DevScope on(g_two.dev)` of the grow block now opens before `if (P.s) CU(cudaStreamSynchronize(P.s))`
instead of after it, so GPU 1's stream is not synced while GPU 0 is current (the runtime rejects foreign-device
stream calls on some versions). Everything the block did with GPU 1 current - the frees, the stream/cublas
creation, the allocations - keeps exactly the device it had; the nested `DevScope g0(0)` sites are unchanged.

## 4. pf1_pick: refuse kMIX experts (commit 92dafc9c2)

`pf1_pick` gained a `const Layer& Ly` parameter (forward declaration, definition, the single call site in
`run_prefill_moe` where `Ly` was already in scope) and returns false for `Ly.ty[0] == strata_fmt::kMIX`, with
a comment: `pf1_run` decodes the fixed three-part layout at `Ly.eb`'s offsets, while a kMIX expert's parts sit
under their own MixHdr - routing one to GPU 1 would decode it at wrong offsets silently. `run_exp`'s refusal of a
kMIX pack across the GPUs (`"a Strata pack across two GPUs: not yet"`) sits in its `g_tp` (tensor-parallel)
branch, which the PF_GPU1 prefill path never reaches (`p1` requires `!g_tp`), so this `return false` is the real
guard for the prefill path, not insurance over that die; the token path's `two_send` has no kMIX check of its
own, out of this plan's scope.

## 5. drafter_propose's host-staging branch: record ev_cp (commit ba4ac2cd6)

The staged branch now records `g_drem.ev_cp` on `g_drem.s` after the drafter's stream has read `host_anchor` /
`anchor` and sets `cp_valid = true`, as its peer twin does - today every call site pairs the propose with a
following `write_context`/`set_f_rnn` that records it; a call site without a pair can no longer reorder two
transfers on `host_anchor`. The staged branch still has no `cp_valid` wait at its top (the peer branch's first
line): the plan asked only for the record, and adding the wait would change sequencing on every existing
staged call, so it is left out - noted below.

## Build check

No `-fsyntax-only` in nvcc; instead the single TU was compiled to objects in `./scratch/` with the CUDA 13.1
nvcc, host flags in the ik2 build's spirit: `-std=c++20 -DNDEBUG -arch=compute_75 -DSAGUARO_IK2_PASCAL=1
-DMAYA_DS4_IQ_ONLY -DDS4_IQ_MIN_FORMS` plus the `saguaro-ik2` include dirs. Baseline (unmodified file) and the
file after each fix: all compile; only the two pre-existing unused-declaration warnings (`Shards::num`,
`g_apu`) in baseline and every later pass. No engine runs, no other file compiled.

## Tests run

    /opt/llm/saguaro/.venv/bin/python -m unittest tools.test_docs_new_engines -v
    Ran 7 tests in 0.015s - OK

(the only module the plan names; a docs test, unaffected by these changes, run as the plan's regression gate)

## Open questions

- Finding 1's gate `g_two.link.all_peer` is false on every supported start today (two_setup runs after the
  drafter), so the load always stages; whether to hoist the head copy into two_setup's own timed copy (one
  peer copy at link speed when the test passes) is a shape question beyond this plan.
- Finding 5: should the staged `drafter_propose` also take the peer branch's leading `if (cp_valid) wait`
  before its D2H into `host_anchor`? With the new record a staged-then-staged propose pair stays correct only
  because the paired `write_context` waits at the next `drafter_copy_taps`; the leading wait would make each
  branch self-contained but serializes g_s behind g_drem.s on every staged call.
- Findings 1 and 2 still want the hardware check the review asks for (a two-card start with
  SAGUARO_IK2_DRAFT_GPU=1 / PF_GPU1 with an outgrowing chunk); no such pair is available here.
