# Review: P8 round 2 - the four opt-in fixes plus the round-1 fixes (git diff 7fcda7f2..HEAD)

**Verdict: MERGE-WITH-NITS** - round 1's High (the staged head load dropping the `q8d` D2H) is fixed and
verified, both round-1 Lows were corrected in comment and report, all four fixes match the plan, and the
default path is provably untouched. What is left: the report's change-count sentence went stale, and the
named test still exercises none of it (the plan's choice, stated plainly).

## Findings, by severity

**High / Medium: none.** The round-1 High was `:9104-9106`'s single D2H; HEAD now holds both halves inside
the `DevScope g0(0)` scope - `stage` gets the `q8` and `stage + qb` gets the `q8d` (`src/ik2/ik2_main.cu:9106-9107`),
mirroring the gated peer pair (`:9099-9100`) and the two H2Ds (`:9109-9110`) - so `fold_q8`/`set_head_q8`
(`:9121-9122`) no longer fold uninitialized pinned scales into the drafter's head on the only branch that
runs today. Commit `d9a079601`, report §1 updated in the same commit.

**Nits:**

1. `REPORT-p8-u9-findings.md:5` - "only `src/ik2/ik2_main.cu` changed: 30 insertions, 6 deletions" was true
   at `42150823d` but is stale at HEAD: the committed totals are 34 insertions / 6 deletions
   (`git diff --numstat 7fcda7f2..HEAD -- src/ik2/ik2_main.cu`), and the sentence's "only the .cu" no longer
   covers `FIX-REPORT.md` either. `FIX-REPORT.md` restates the per-commit numbers but not this one.
2. `tools.test_docs_new_engines` is a docs gate and exercises none of the four touched paths - the report
   says so (`REPORT-p8-u9-findings.md:60-65`). The plan names it and forbids other files, so nothing more
   could be done here; the real verification is the single-TU compile (re-run below, exit 0, only the two
   pre-existing warnings `Shards::num` `:198` / `g_apu` `:1544`) and the two-card start the report lists as
   unavailable (`REPORT-p8-u9-findings.md:76-77`) - that hardware check stays the one real open risk for
   findings 1 and 2.

**Info (documented, accepted, no change requested):**

3. `src/ik2/ik2_main.cu:1673` - the staged `drafter_propose` still has no leading `if (cp_valid) wait`
   before its D2H into `host_anchor`, unlike the peer branch's `:1663`. Correct today by pairing: both call
   sites (spec_step `:5532` -> `:5613-5614`, the draft loop `:11349` -> `:11443-11444`) run a
   `drafter_write_context` between proposes, and its `drafter_copy_taps` leading wait (`:1621`) on `g_s`
   (FIFO) orders the next propose's D2H behind the previous propose's `ev_cp` - the newer snapshot the fix
   records (`:1679-1683`) only tightens that. Left open exactly as `REPORT-p8-u9-findings.md:48-50,72-75`
   says; the plan asked for the record only.
4. `src/ik2/ik2_main.cu:9098` - the head-copy gate is `g_two.link.all_peer`, not `pf1_peer()`. Right choice
   (below), with a theoretical wrinkle for some future ordering where the link is tested before the drafter:
   `SAGUARO_IK2_PF_HOST=1` forces host staging for every hand-off (`pf1_peer`, `:3034`) yet the head copy
   would still take the peer branch (`all_peer` true ignores the env). Unreachable in every start today -
   `all_peer` is false at `:9098` - so nothing to fix now.
5. `src/ik2/ik2_main.cu:9103` - the transient ~1.4 GB pinned `cudaHostAlloc` (for a 250K x 5120 head) that
   round 1 rated acceptable: opt-in, one-shot, `CU()`-dies on failure, freed at `:9111`. `FIX-REPORT.md` §4
   answers NO CHANGE; agree.

## The brief's checks

- **Default path untouched (no new statement executes when the switches are off)?** Yes, site by site:
  `drafter_propose` early-returns to the plain call when `g_drem.dev < 0` (`:1661`), so `:1679-1683` runs
  only with a remote drafter; `pf1_pick`'s kMIX test (`:3086`) sits behind the `!en` return (`:3081`, the
  `SAGUARO_IK2_PF_GPU1` gate) and the added parameter is a reference read executing nothing; `pf1_run` is
  reached only through `p1` (`:2453`), which requires `pf1_pick` true, i.e. the switch and `g_two.on`; the
  head-load block is inside `dg > 0` (`:9070`). No device code is touched (all four sites are host
  functions, no kernel added or altered) - HEAD recompiles to an object with CUDA 13.1 nvcc, the build's
  spirit flags, exit 0, only the two pre-existing warnings.
- **DevScope right?** Yes. `cudaStreamSynchronize(P.s)` (`:3102`) is now inside `DevScope on(g_two.dev)`
  (`:3101`), which restores the previous device at `:3136`; the nested `g0(0)` sites (`:3107`, `:3126`)
  still return to GPU 1, and `DevScope` (`:1598-1611`) is a save/restore device swap - no mutex anywhere,
  so no nested lock acquisition, and nothing in the diff spawns a thread or writes stdout/stderr, so no
  line interleaving. In main: the two D2Hs run with GPU 0 current (`:9105-9108`) for GPU-0 pointers, the
  two H2Ds with `dg` current (`:9082`, `:9109-9110`) for `dg`'s pointers, and the peer branch names both
  devices explicitly (`:9099-9100`) - every copy has its buffer's device current.
- **Peer gate right?** Yes. `all_peer` is default-false (`src/mm3/mm3_link.cuh:59`) and set only by
  `Links::setup`, which is what enables peer access (`:116-119`) and only keeps `all_peer` when every used
  pair tested as a peer copy (`:192-198`); `g_two.link.setup` runs only in `two_setup` (`:3731`, called
  `:9477`), after the drafter load (`:9055`), so reading it at `:9098` is well-defined (a global) and false
  on every supported start - the load always stages, the same host bounce the rest of the feature uses.
  `pf1_peer()` could not be the gate: its first-call latch (`:3031-3035`) would fire with `g_two.on` still
  false (set only at `:3735`) and pin the whole run to host staging, plus emit its NOTE early.
- **kMIX refusal right?** Yes: `:3086` returns false before touching `skip`/`g_two.own`, the single call
  site (`:2453`) passes the `Ly` already in scope, and the reworded comment (`:3082-3085`) now matches the
  code - the die is inside `tp_gu` (`:2785`, lambda `:2739`) under `else if (g_tp)` (`:2735`) while `p1`
  requires `!g_tp`, so the refusal is the prefill path's real guard; `two_send` (`:2987-2989`) remains
  kMIX-unchecked, out of P8's scope and noted in the report.
- **ev_cp record right?** Yes: `:1679-1683` mirrors the peer twin (`:1669-1670`); the event exists on this
  path (`:9087`) and the `cp_valid` contract (`:1592`) holds; the cross-device `g_s` waits on `g_drem.s`'s
  event (`:1621`, `:1663`) follow the existing pattern and cannot cycle - `g_drem.s`'s queue up to the new
  record depends only on an earlier-recorded `ev_src`.
- **Generated file:** the diff is three files (`src/ik2/ik2_main.cu`, `REPORT-p8-u9-findings.md`,
  `FIX-REPORT.md`); `serve/qfn_engine.py`, `engines/qfn/serve/server.py` and
  `tools/saguaro/extract_qfn_engine.py` are untouched, so the generator question does not arise.
- **Commits/hygiene:** one commit per numbered item (`015f18123`, `5b7e7e000`, `92dafc9c2`, `ba4ac2cd6`),
  the report its own (`42150823d`), one commit per round-1 finding (`d9a079601`, `d8701d9a7`, `6e9b06a4d`)
  and the fix report (`9e10f52bc`); no Co-Authored-By / Claude-Session / Assisted-by lines anywhere in the
  range; no secrets and no out-of-repo paths in added lines; working tree clean (compile object under
  `scratch/` only).
- **Runs:** `/opt/llm/saguaro/.venv/bin/python -m unittest tools.test_docs_new_engines -v` - ran 7, OK;
  the single-TU nvcc 13.1 compile of `src/ik2/ik2_main.cu` at HEAD - exit 0, the two pre-existing warnings
  only. The two-card hardware check remains unavailable, as both reports state.
