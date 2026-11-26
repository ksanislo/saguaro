# FIX-REPORT - REVIEW-RESULT findings, one commit each (`397d1f852..4e4d6fcec`)

This is round 3 (`REVIEW-RESULT.md` third round, `git diff 88809c4e..397d1f852`). The manager's card test on
mortar OVERRIDES the MERGE-WITH-NITS verdict: the solo session went elastic but the BATCH SLOTS and their DRAFTERS
still pinned their host K/V for the whole `--max-context` at load (262144: 6.18 GiB "the batch slots and the
drafters", 1048576: ~24.7 - the expert tier at 54.36 GiB instead of 68, and the two load lines unequal). That is
the FIX-FIRST item; the review's four round-3 findings follow it. All five are taken, one commit each; the
manager's card-test section of `REVIEW-RESULT.md` was committed with the fix. Nothing here is GPU-testable on this
box (no card, no resident tier): what follows is code-true; the proof stays `scratch/card-test-elastic-kv.sh` on
mortar plus the named unit test (22/22, run alone, prints "22 checks OK" at HEAD).

## 0. FIX-FIRST (manager) - batch slots and slot drafters pinned their host K/V whole at load -> FIXED `b910fa1ab`

Diagnosis: `qsa_state_init` made a streamed state's host copy elastic only when `slot_G > 0 || g_host_main`
(`layer.cpp:2004`). `g_host_main` (--serve) covered the solo session and the stages only - it is switched off
before the slots are built ("the slots' own rule decides theirs") - and the slots' rule was the VRAM one:
without `SAGUARO_ELASTIC_SLOTS=2` + VMM every slot state fell to the `kv_host_alloc` pinned-whole branch, and a
slot drafter's elastic flag follows its slot's (`mtp.cpp:396`), so the drafters pinned whole too. That is the
card test's 6.18/~24.7 GiB. The host copy was always orthogonal to the VRAM ring, so:

* `layer.cpp/.hpp`: new `g_host_slot` + `qsa_set_host_elastic_slots(on)` added to the 2004 condition. It is set
  around the batch-slot construction and the slot-drafters' (`generate.cpp`, both `--serve && --batch` only - the
  CLI `--batch` path keeps today's pinned-whole copies byte for byte) and cleared again before anything else can
  build a state. A slot drafter needs no new plumbing: its `host_elastic` parameter already reads its slot
  session's primary state, and `owner.host_follow = &d->kv_state()` (already wired at load) makes it grow and
  shrink with the slot through the same `session_host_fit`/`session_slot_fit` calls.
* `layer.cpp`: `qsa_host_backed_cells(st)` (the cells an elastic copy is pinned for, min across arrays; -1: not
  elastic) and `qsa_host_step_cells()` (the step in cells; `INT64_MAX` on Windows) answer "already backed / more
  than the start" without touching anything.
* `session.cpp/.hpp`: `session_host_elastic(s)` and `session_host_backed_cells(s)` (states + `host_follow`);
  `session_slot_fit`'s old `slot_cells < 0` early-return becomes a HOST-ONLY branch: same helpers, same step,
  `trim` gives back - the ring half stays untouched and `slot_cells` keeps counting the ring (so the whole
  `SAGUARO_ELASTIC_SLOTS` VRAM-budget give path, `elastic_idle`, `KMEM/HMEM` and `eidle` stay `eslots_on`-gated
  exactly as before).
* `generate.cpp`: a `ehost_on`/`efit_on` (`eslots_on || ehost_on`) + `ehost_step` pair makes every slot-fit site
  see a slot elastic in EITHER half - `slot_admit_fit` (admit: device sync, fit-with-trim to prompt+max_new,
  idle-slot drops now also drop slots holding host cells past the start step, error line keeps its "no RAM"
  wording via `qsa_host_pin_failed`), `slot_fit` (skips only when NEITHER half trails; a host-only failure returns
  "no memory" straight away - the RAM give-way already ran inside `qsa_host_grow`, and there is no VRAM budget to
  hand over), the decode gate `slot_grow_or_end` (its `backed` precheck now checks host-backed cells too, so no
  window can touch cells the slot has not pinned; both call sites `14499`/`15541`), the queued-read and fast-lane
  read fits, `PFMULTI`, and `ram_reclaim` (a slot past `ehost_step` of host cells is a parkable donor: persist,
  trim, zero - and the park line's GiB delta (read from `qsa_kv_host_bytes`) is real released RAM, not a
  promise).
* Load: each slot's states and, via `host_follow`, its drafter pin ONE start step (`session_host_fit(*u, 1)`, per
  stage under the right `OnDevice`) right after the solo session's own start and before the manager's "every
  session" line, which already counts backed bytes when anything is elastic. The number that line splits out as
  "the batch slots and the drafters" is now `holders x one step` - IDENTICAL at 262144 and 1048576 - and all of
  it happens before `pin_cache_complement` sizes the expert tier from the free RAM (generate.cpp:9039), so
  `--resident-budget-gib 68` is filled to the ceiling at both contexts again.   `slot_mem_note` prints a slot's
  host MiB so the "elastic slot N grows at" lines carry numbers in this config too.
* Kept as it is: the solo path (manager's instruction), the pinned-whole fallback when the elastic pool cannot be
  made, and `SAGUARO_KV_HOST_ELASTIC=0`: with the feature off no state is elastic, every new predicate says so,
  and every widened site behaves exactly as it did - the OFF path stays byte-identical, so the card script's
  criterion D holds unchanged. The card script itself needs no edit: criterion A already compares the two load
  lines, and the fix makes them equal.

## 1. MEDIUM - `relieve_ram` held `complement_mu_` across the grace sleep and the madvise -> FIXED `9d4d04f05`

The r2 comment (and FIX-REPORT 3) said the grace wait and the madvise stay OUTSIDE the lock; the `lock_guard` was
function-scoped and they did not. So one give-way stalled every `commit_exchanges`/`commit_flip`/`install_resident`
- the decode path's exchange commits and prefetch installs - for the grace window (250 ms default) plus the
discard, exactly when RAM is short. The lock scope now closes after the phase-1 loop (the snapshot, the sort and
the offset trims), and the sleep, the `MADV_DONTNEED` loop and the log line run without it; `runs`/`gave` were
already locals. This is safe, not just shorter: a trimmed range's offset reads `kNoComplement` to every writer
forever (`exchange_cache_complement` and every copy check take their destination from a LIVE slot's offset, read
under the lock), so a writer released during the grace window can hand an incoming expert some OTHER slot's range -
bytes are never written into a range phase 1 flipped away, and the round-2 writer-race fix stays intact. Lock order
unchanged (one-way `install_resident -> ForesightSwap::hold -> smu_`). The comment now describes what the code
does instead of what it promised.

## 2. LOW - a refused device alias left the registration standing, poisoning every later grow -> FIXED `3abe15417`

`host_prefix_reregister_locked`: `cudaHostRegister` could succeed while `cudaHostGetDevicePointer` refused or
answered null; both loops then fell to rollbacks that unregister only the arrays whose alias HAD arrived (the
step-down give-back skips `rd[i] == null`), so the refused array's run stayed registered, uncounted in
`g_kv_host_bytes`, and every later `cudaHostRegister` at that address failed with it. Both failure branches now do
what the pieces path has always done at its `d != a` bail - `if (e == cudaSuccess) cudaHostUnregister(at)`.

## 3. LOW - a failed prefix TRIM re-registered the run it had just discarded -> FIXED `dc36d24eb` + `4e4d6fcec`

The trim madvises `[prefix(pages), prefix(old_backed))` per array, then - if the smaller prefix is refused - the
fallback started at `pg = p.backed = old_backed`: re-registering that re-faulted the discarded run's ZERO pages
into RAM (re-pinning the room the park came to give) and stood the copy on a prefix whose tail was now zeros. The
walk-down starts at `pages` on the trim path (the r2 nit 5 warn line becomes honest on purpose, not by accident);
the grow path is unaffected (min/`trimming` keeps the old start there).

## 4. LOW - a failed prefix GROW could truncate or null a live copy -> FIXED `4e4d6fcec`

Took the review's proposal and sharpened it: the grow fallback gets exactly ONE attempt - re-registering the old
prefix (`old_backed`) - and no step-down. A step-down below `old_backed` was worse than the null case the finding
names: it stood the conversation on a device base ending BELOW its live cells (the next read writes past it). One
attempt usually succeeds (the failed target attempt's RAM is back by the time the fallback runs), so the copy
stands exactly where it did before the failed grow and only the grow is reported (`ok = false`, the caller's
relief retry then sees the true `p.backed` instead of the walk-down's 0 and can recover); if even the old,
already-populated run is refused (relief already failed - a driver limit), nothing can stand, the null + the
caller's refusal line are the honest answer, `ram_reclaim` parks what can park, and a later grow re-registers the
still-populated pages whole. Pieces mode's rollback-to-`had` never dropped a failed step below the old prefix;
prefix mode now agrees.

## Not taken from the review

* Nothing declined. The one place the report deviates from the finding's literal wording is 4: the review's
  "one-time ok = false return keeping the last full prefix (retrying the old size before stepping down)" is
  implemented as "retry the old size once and do not step down on a grow at all" - stepping below the standing
  prefix re-breaks exactly what the finding asks to make benign.

## Verification

* `ninja -C scratch/build-p124e` -> clean incremental rebuild through every fix, ends "no work to do" at HEAD
  `4e4d6fcec` (`saguaro-qfn` links; the only warning in touched files is the pre-existing HEAD unused-`w`).
* `ctest --test-dir scratch/build-p124e -R '^kv_host_growth_test$'` -> 100% passed, run ALONE (the only test
  module the plan names; the pure helpers it covers are untouched by these fixes and stay 22 checks). No other
  test module was run; no GPU here.
* `serve/qfn_engine.py` / `engines/qfn/serve/server.py` are still absent from `git diff --name-only
  88809c4e..HEAD` (no flag plumbing added, so no `extract_qfn_engine.py` re-run). `scratch/` stays gitignored.
* For mortar (card script, unchanged): A must now pass - the "every session" line shows the SAME small start for
  solo AND slots+drafters at 262144 and 1048576 (about one 32768-cell step per holder, both contexts), the expert
  tier keeps `--resident-budget-gib 68` filled at load in both, and the "grow with their conversations" clause
  stays; C must show slot copies ("elastic slot N grows at ... host K/V") growing per use and BIDLE/ram_reclaim
  lines giving RAM back; D (`SAGUARO_KV_HOST_ELASTIC=0`) must still print the OLD pinned-whole figures byte for
  byte with no "the host copy backs" lines anywhere.
