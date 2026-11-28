# FIX-REPORT - P125j layer pause, re-review round 2 (findings of REVIEW-RESULT.md at 3208d412c)

Every finding of the review is fixed; one commit per finding, no re-review finding left open.
The previous round's ten findings were re-verified fixed BY the review itself ("Previous round's
findings - fixes re-verified here at HEAD") and are untouched here.

## MAJOR 1 - the live path dropped an armed pause without the parked path's trust clamp - FIXED (ec0f6bd0b)

`engines/qfn/src/program/generate.cpp`, the go_on block (was 20321-20335).

- A dropped pause with `resume > read_pause.pos` now caps the trust exactly the way the parked
  adopt does (20124-20148): `resume = pause_pos`, `live.resize(pause_pos)`,
  `live_imgs = imgs_below(live_imgs, pause_pos)`, checkpoints deeper than `pause_pos` erased, and
  one stderr line naming the pause's first token and the shared depth - so the paused chunk is
  re-read over ALL its layers before anything deeper is attended over or parked. The trigger the
  review verified reachable with no env vars (image follow-up: `go_on` fails on `req_imgs.empty()`
  at `resume == live.size() == P+T`) now reads from P, and the request can no longer end as a
  normal `live_ok` turn with cells no layer >= the pause ever wrote.
- The whole pause decision MOVED above `mtp.kv_restore(resume)` (it sat below it) so the drafter's
  ring is truncated to the CLAMPED depth - at the old place a clamped `resume` would have left the
  ring restored to the never-written depth. Nothing between the old and new spot reads the pause or
  writes `resume` (only the drafter restore, `tr`, `req_times`, `set_prompt_len`), and the
  checkpoint-mount above it keeps using the `resume` it mounts at; the RESUME line, `read_from`,
  `turn_main.begin`, `limit_reuse` and `pp_next_check` all see the clamped figure as intended.
- The mid-chunk prompt-edit trigger stays unreachable (full-prefix `resume`, checkpoint/borrow
  limits) - the clamp is the missing invariant guard, as the review said, not a new behaviour.

## MAJOR 2 - the card script's leg-sum could not pass - FIXED engine-side (bb0c65a76)

`engines/qfn/src/program/generate.cpp`, the pause keep branch.

- Took the review's first option ("bump `pp_reached = lp_pos + lp_tokens` in the pause keep"):
  the keep now sets `pp_reached` to the paused chunk's end, so the DONE `read` figure
  (`clamp(pp_reached - resume, 0, fresh)` at 23083+, and the engine's
  "prompt N tokens = R reused + K read" line the script sums) counts the paused chunk on the leg
  that paused it - which is what REPORT:166 already claims. The script is unchanged and its sums
  now close by construction: cut leg K = P+T - resume_leg, resumed leg K = (where it stopped) -
  (P+T), so the legs' reads sum to the prompt within READ_TOL, and the last leg's
  `LAST_R + LAST_K == LAST_N` stays exact (a completed leg reads `fresh = n - resume`; the paused
  chunk is in its R, not its K). The M > 0 reused check is unaffected (RESUME prints the
  un-clamped `resume` = P+T).
- Only reachable-after-the-keep uses of `pp_reached` are the #1620 keep (an `else` of the pause
  keep, so a bumped leg never reaches it), the P124b landing (`!live_ok`, false once the keep ran)
  and the cancelled `read` figure - which is the point. A leg whose pause keep is REFUSED (MAJOR 3
  with the head-alias flag, or no prompt_cache) keeps the old figure: its chunk was not kept and
  is genuinely re-read, so counting it as unread stays right.
- The grace-default flip's card proof is intact: `bash -n` clean; not run (no card on this box,
  the plan excludes GPU runs).

## MAJOR 3 - `--prefill-head-alias` desynced the pause record from its rows - FIXED (df890dd14)

`engines/qfn/src/program/generate.cpp`, three spots.

- Both `sp.reset()` sites (the next request's top rebuild ~19512 and the after-read head restore
  ~21720) now call `drop_read_pause()` beside the reset: the record goes with the rows it was
  pointing at, like everywhere else does.
- The pause keep additionally refuses when `sp.pause_rows_bytes() == 0` - the direct guard for the
  reset that runs BEFORE the keep branch (21720): with the rows gone the keep records nothing
  (and its "0 MiB" line can no longer print), the leg degrades to the #1620 keep at `pp_reached` -
  a completed chunk end, fully written - and the existing pair-check at 22966 drops record+rows
  together on the mismatch. With the pair together, the next request either goes on with a real
  pair or reads on from a clamped trust (MAJOR 1), and a park either has rows to take or no record
  to park by: the half-written-session-dir write (11089-11108) cannot be reached this way again.
- Flag off (the default, no repo config and no server code passes it): untouched - with the flag
  off nothing resets `sp` between the run and the keep, and the keep's new condition was always
  implicitly true there (the pause armed rows).

## NIT 4 - `paused_chunk_index()` doc said 0, restored pauses report -1 - FIXED (6a2ea08e4)

`engines/qfn/include/strata/prefill/prefill.hpp`: the doc now says -1 for a restore-handed-back
pause (give_pause never sets `SavedPause::chunk`; its default -1 stands - the paused run's
run-relative label cannot survive the park) and notes that the engine's DONE/keep line carries the
label captured in the run. Chose "say -1" over "store 0": a restored pause genuinely has no
run-relative index, and the label-only reader never sees it. Doc-only.

## NIT 5 - `state_bytes` wrapped `per_row` before its guard - FIXED (dc4dbea8a)

`engines/qfn/include/strata/program/layer_pause.hpp`: clamp `row_floats > UINT64_MAX / 4` first
(such a row's own byte count is >= 2^64: saturate at once), so `per_row` can no longer wrap - and
the guard below can no longer divide by a wrapped zero. `engines/qfn/src/program/layer_pause_test.cpp`:
two checks added - a 2^62-float row saturates, the widest countable row (2^62-1 floats) stays
exact. Built and run here: "ok: 43 checks, 0 bad", `ctest -R layer_pause_test` passes.

## NIT 6 - a resumed chunk above max(LB,1) still gathered and landed ~84 MB of PLE rows - FIXED (fc4e5f5d8)

`engines/qfn/src/prefill/prefill.cpp`: `ple_reads = l_from <= max(LB, 1)` gates BOTH the chunk-top
gather and `ple_pending` (hence the post-loop land), matching where `m.ple_emb`'s only readers sit
(the l == max(LB,1) blocks, which `l_from` skips). A resumed chunk past layer 1 now gathers
nothing and lands nothing. Where the chunk before such a resume had already read its rows ahead,
the skip waits the gather out and clears `ple_next` at the chunk's end, so the next chunk's top
sees a broken pipeline and restarts the gather at its own chunk - the cold-start path the first
chunk already uses - instead of landing the stale rows as its own. (`ple_on` means LB <= 1, so
`ple_reads` is `l_from <= 1`; a continuation resumes at its own c0 = 0, where `ple_next` starts
empty - the consume is the guard for any layout that changes that.) `strata_prefill` builds exit 0.

## NIT 7 - SETTINGS.md contradicted itself on the grace vs the pause switch - FIXED (79cebc30e)

`docs/SETTINGS.md`, both the `SAGUARO_PRIORITY_IDLE_S` row and the Request-priority paragraph:
with `SAGUARO_LAYER_PAUSE=0` the read stops between chunks again and the grace is what keeps a
follow-up's wait short - set it back to 5. Docs only; the 0 default itself ships with the pause,
as the plan's step 3 and the manager's gate require.

## What I ran (plan-named modules, one at a time, this tree)

- `cmake --build scratch/build-p125j-cuda75 --target strata` exit 0 (links generate.cpp +
  prefill.cpp as committed) and `--target strata_prefill` exit 0.
- `cmake --build ... --target layer_pause_test` + `ctest -R layer_pause_test`: Passed, binary
  prints "ok: 43 checks, 0 bad".
- `python -m unittest serve.test_priority` (the project venv's interpreter): Ran 68, OK.
- `tools/saguaro/extract_qfn_engine.py`: regenerated `serve/qfn_engine.py` byte-identical
  (`git status` clean after) - no server-side change was needed for any finding.
- `bash -n tools/card_tests/layer_pause.sh`: clean; `py_compile` on serve/server.py,
  serve/qfn_engine.py, engines/qfn/serve/server.py, serve/test_priority.py: OK.
- Not run: `tools/card_tests/layer_pause.sh` itself - no card on this box and the plan excludes
  GPU runs; with MAJOR 2 fixed engine-side its checks are the ones that must hold on the card.

---

# FIX-REPORT - P125k merge round (manager FIX-FIRST of REVIEW-RESULT.md at fff608b9f, 2026-11-27)

The manager's instruction: merge `saguaro/q1-p125j-layer-pause` into this branch (a merge, not a
rebase - both are reviewed), resolving `engines/qfn/src/program/generate.cpp` keeping BOTH
behaviours - P125j's layer-boundary stop + its pause record handling, and this branch's
event-based park/restore waits. That is finding 1 and everything: "Do this, nothing else."

## 1 - the generate.cpp conflict, hunk by hunk - FIXED (merge commit 70f67efde)

Merge base d789228f0; the merged ref tip in this repo is d8e25fd95. One file conflicted, two
hunks; everything else merged clean. Both hunks keep BOTH sides:

**Hunk 1 - `if (borrow) {`** (merge-base 19931; merged generate.cpp:20242):
- this branch (P125k) added the narrowed-wait scope for the prefix restore -
  `ConvStreams borrow_streams; const bool borrow_own = park_own_ok() && add_main_streams(...);
  const ConvSyncScope borrow_scope(borrow_own, &borrow_streams)` - the borrow's restores and
  `device_sync()`s wait only on this conversation's streams plus its copy stream.
- P125j added a comment + `drop_read_pause();` - the borrow rewrite kills the armed layer pause
  (with a skipped park it has nothing to go on; with a real park, park_body already dropped it).
- Kept: the scope lines, then P125j's comment and drop, inside the same block. Nothing about
  either side's behaviour moves: `drop_read_pause` touches only host state (the ReadPause record
  and `Prefill::clear_pause`), and the RAII scope covers the whole restore either way.

**Hunk 2 - `if (incoming) {`** (merge-base 19980; merged generate.cpp:20298):
- this branch (P125k) added the comment + `ConvStreams rest_streams; ... const ConvSyncScope
  rest_scope(rest_own, &rest_streams)` - the incoming restore's pre-copy syncs wait on this
  conversation's queued work only, and its post-copy wait is its own copy stream.
- P125j added a comment + `drop_read_pause();` - the restore rewrites the session with another
  conversation's state, so the outgoing conversation's armed pause must go.
- Kept: the scope, then P125j's comment and drop. P125j's re-arm of an incoming paused image
  (`incoming->paused()` / `sp.give_pause`, 20366-20383) auto-merged as-is *inside* the same
  scope: it is host-side (the saved chunk rows are RAM), so it neither waits on nor feeds any
  scoped sync.

Nearby sites that auto-merged and were re-checked line by line (each side kept intact):
- session-file restore (19402-19427): P125j's `drop_read_pause()` before `blocking("transfer")`,
  P125k's `ctx_streams`/`ctx_scope` around `conversation_snapshot_restore` - both, in order.
- the switch's park (20124): P125k's `park_streams`/`park_scope` wraps the park_current call
  sites; P125j's neighbours (the paused-donor borrow guard at 19942, the slot-source
  `drop_read_pause` at 20177, the park_body pause-rows block at 11090 and 11201-11216: estimate
  += `sp.pause_rows_bytes()`, `sp.take_pause_rows` into the image, never `sdir->write` a paused
  image) ride inside that park with no device sync at all - the rows are host RAM.
- the cut read's checkpoint rewind (23220-23244): P125k's `cut_streams`/`cut_scope` before the
  drain `device_sync()`, P125j's `drop_read_pause()` on the landed keep (23237).
- the cancelled-keep chain (23148-23198) + the drop-after-keep rule (23245-23255): P125j's keep
  branch, `pp_reached` accounting and armed-record rule in full; this branch's RDONE/stage_ms
  edits are other lines; the DONE line carries P125j's `paused_layer=/paused_chunk=/paused_at=`
  suffix (23402) and this branch never touched it.
- serve: serve/server.py took P125j's grace default 5->0 (with `serve/test_priority.py` updated
  to match) and its paused-read STOP wording - the locals it reads (`last0` 5796, `fresh` 5919)
  are in scope at its use (5982); this branch's read-rate/stage_ms parsing is untouched.
  `engines/qfn/serve/server.py` took P125j's DONE key parse (1666) on top of this branch's.
- generated file: `serve/qfn_engine.py` auto-merged, and `tools/saguaro/extract_qfn_engine.py`
  reproduces it byte-identical from the merged `engines/qfn/serve/server.py` (run after the
  merge; the regenerated file is what the commit holds) - never hand-edited.
- `engines/qfn/CMakeLists.txt`: P125j's `layer_pause_test` target landed beside this branch's
  test lines.

Merge-ref note: the only `saguaro/q1-p125j-layer-pause` in this repo is refs/heads at d8e25fd95
(that branch's own worktree shares this repo and sits on the same commit). The round-5/6 P125j
fixes the manager's message cites (the batch-lane pause capture, the grace-default gate) are not
in this repo's refs at merge time. If the RC carries a newer P125j tip, re-merging the updated
ref is a round of its own; this resolution keeps both sides for every hunk the ref actually has.

## 2 - "the review's nits, if any are open" - nothing open, nothing changed

No P125k nit list exists in the tree: the review that returned MERGE-WITH-NITS left no file behind
(the REVIEW-RESULT.md on disk before the merge held only the manager's instruction; what the merge
carries in REVIEW-RESULT.md / FIX-REPORT.md is P125j's own round record, whose nits belong to
P125j's rounds). REPORT-p125k-park-sync.md's "Open questions" (ver/ver_b in the slot shape,
multi-GPU still whole-device by the `park_own_ok` gate, the copy-stream contract `copy_done`
rests on) are open questions the review accepted, not findings - unchanged, restated there.

## 3 - build + tests (manager step 3; no GPU)

- `tools/qfn/build.sh scratch/build-p125k-merge` with
  `QFN_GGML_DIR=$PWD/../strata-main/third_party/llama.cpp` (the manager's checkout, named
  relative to this worktree): **QFN BUILD OK 70f67efde**, ninja 179/179
  (strata + strata_kq + strata_kq_tab, sm_75, CUDA 13.1), a new dir, nothing run on a GPU.
- `park_policy_test` (the plan's module, this branch's 65 checks): standalone
  (`g++ -std=c++20 -I engines/qfn/include`) and as a scratch/build-p125k-merge target -
  **65 checks passed**, run one at a time.
- `layer_pause_test` (the unit test P125j added): standalone and as a scratch/build-p125k-merge
  target - **ok: 43 checks, 0 bad**, run one at a time.
- `py_compile`: serve/server.py, serve/qfn_engine.py, engines/qfn/serve/server.py,
  serve/test_priority.py - OK; `bash -n tools/card_tests/layer_pause.sh` - clean.
- No card test run: no card on this box and the manager's step 3 says "No GPU runs".

---

# FIX-REPORT - P125k manager card round 3 (FIX-FIRST of REVIEW-RESULT.md at 28c2111c3, 2026-11-27 15:45 PST)

The card: re-merge P125j, fix the REPORT's :257 "byte-identical" nit, build QFN only + run the
named tests. No GPU runs. Finding -> what was done:

## 1 - "git merge saguaro/q1-p125j-layer-pause again (bring in P125j's later rounds)" - MERGE RAN, NOTHING NEW EXISTS TO BRING IN (320dc3a5f)

`git merge saguaro/q1-p125j-layer-pause` at HEAD: **"Already up to date"** - no conflict, nothing
to resolve, no merge commit (nothing moved). The verification behind that, each in this repo only:

- refs/heads/saguaro/q1-p125j-layer-pause = d8e25fd95 = its round-4 FIX-FIRST tip, the exact commit
  70f67efde already merged in ("p125j tip already in HEAD" by `merge-base --is-ancestor`).
- That branch's worktree is clean at the same commit - no uncommitted round-5/6 work sitting there.
- `git reflog show saguaro/q1-p125j-layer-pause` ends at d8e25fd95: the branch never had a commit
  after it, so there is no amended-away round-5/6 work in this object store (a `git fsck` sweep for
  dangling commits found no P125j message beyond round 4 either).
- No remote carries the branch: `git ls-remote` on saguaro/origin/fork has no `*p125*` ref, and this
  repo holds no remote-tracking p125j ref.

So the round-6-passed P125j the card cites is not in this repo's refs - exactly what the merge round
predicted above ("If the RC carries a newer P125j tip, re-merging the updated ref is a round of its
own"). Both behaviours the round-1 merge kept (P125k's scoped narrow waits + P125j's drop_read_pause
and pause plumbing, hunk-by-hunk in dc436f2f0 §1) stand unchanged at this HEAD; the manager's card
itself is committed here (320dc3a5f) so the instruction rides the branch.

## 2 - the REPORT's "byte-identical" claim at conversation_state.cpp:257 - FIXED (f3d89c6c7)

REPORT-p125k-park-sync.md, three overstatements rewritten, wording only, no code change:

- §2 gate bullet: gated-out shapes do keep the whole-device sync for the ARMED waits, but not
  byte-identical overall - the restore tail `copy_done` (conversation_state.cpp:257) and the
  copy-stream residency refresh are ungated (the §5 "every caller" contract) and ride those shapes
  too: their restores now end on their own copies instead of the old device sync - they wait less,
  never under-wait, and the device-wide pre-sync ahead of the copies still drains everything.
- §2 first bullet: `persist_slot`'s `light` path is not byte-identical either - the hoisted
  `ConvSyncScope` waits the scoped streams PLUS this thread's private copy stream
  (`park::wait_streams` always includes it), so the FAST_LANE slot save gained a narrow sync of its
  own thread's queued copies; it still never device-syncs.
- §5 multi-GPU/pipelined bullet: same correction - the gate keeps those armed waits whole-device;
  the restore-tail change still rides them.

## 3 - build + tests (manager step 3; no GPU)

- `QFN_GGML_DIR=../strata-main/third_party/llama.cpp tools/qfn/build.sh scratch/build-p125k-merge2`:
  a new dir, sm_75, CUDA 13.1 nvcc, QFN-only (strata + strata_kq + strata_kq_tab). The first ninja
  pass was interrupted by the launcher's foreground timeout at [140/179]; the identical ninja
  command (build.sh's own line, same dir/targets/flags) was resumed detached and finished:
  **QFN BUILD OK f3d89c6c7 -> scratch/build-p125k-merge2**, `saguaro-qfn` linked, `ninja -n` for the
  three targets reports *no work to do* at tip. The build log's warnings are the pre-existing set
  (unused `w`/`f`/`hcd`/`n_kv`, misleading-indentation) - no new class. Logs:
  scratch/build-p125k-merge2.{cmake,build,nohup}.log.
- `park_policy_test` (the plan's module, P125k's 65 checks): as a scratch/build-p125k-merge2 target
  and standalone (`g++ -std=c++20 -I engines/qfn/include`) - **65 checks passed** each, one at a time.
- `layer_pause_test` (the unit test P125j added): same build's target and standalone -
  **ok: 43 checks, 0 bad** each, one at a time.
- `serve/qfn_engine.py` untouched (no server-side change this round, so no extraction was due);
  no GPU was touched.

---

# FIX-REPORT - P125k round 4 (FIX-FIRST of REVIEW-RESULT.md at 021500494, 2026-11-27)

The card: fix every FIX-FIRST finding and the cheap NITs, one commit per finding. The BLOCKER's
"to merge" sentence names the two ways out - land the real P125j fix round, **or** revert the grace
default to 5 and arm the pause only where it is captured. The first way was run twice and came up
empty (320dc3a5f: the p125j ref is its round-4 tip d8e25fd95, nothing newer in any worktree reflog,
any dangling object or any remote); so this round takes the second, on this branch, which is what
the review itself offers ("P125k itself does not depend on P125j and could ship without it" - the
P125j merge stays, its inert/armed-anywhere parts are reverted and narrowed).

## BLOCKER 1 - "P125j ships unfixed, and cannot be re-merged" - FIXED (500517c39, 92a62082c, c8f7c4d60)

The re-merge half stands from the round-3 report (nothing to bring in). The three carried findings,
each against the code as it now stands, and the first thing I checked for each - the review's line
numbers were right where they said `serve/server.py` and right in substance, but the capture site is
not inside `read_windows`: it sits in `read_part`'s `run` lambda (20913-20933, `sp.paused_chunk()` at
20927) and `read_windows`/`read_windows_pl` (20779/20683) are the VERIFY-WINDOW readers, which use
`ver.run` (no Prefill at all) - so they cannot produce "layer paused"; they take their STOP between
windows ("cancelled", 20798; the pipelined loop's own check, 20714) and lose only the piece in
flight, exactly as before P125j. The readers that CAN are the two `sp.run` sites with no capture
after them, which is what the fix is about.

**1a. The grace default, 0 in the config nothing compensates for - FIXED (92a62082c).**
`serve/server.py:420`: `_env_seconds("SAGUARO_PRIORITY_IDLE_S", 0.0)` -> `5.0`, docstring and the
history comment say why (the round-4 gate "0 only with a working pause" cannot hold while the cut
is solo-only: see 1b/1c). `serve/test_priority.py:1605-1612`: `IdleGraceDefault` renamed to
`test_the_default_grace_is_five`, asserts 5.0. `tools/card_tests/layer_pause.sh` is unaffected
(it exports `SAGUARO_PRIORITY_IDLE_S` itself).

**1b. "Under --batch the layer pause never fires; the read runs to its end beside a waiting person"
- STANDS, BY DESIGN, and the two lines stay.** `serve/server.py:5844-5847` gates the read
preemption on `keeps_stopped_read and not batch` (and the `QfnEngine` docstring 9600-9602 states the
reason): a batch slot's cancelled read is DROPPED by the engine (`BRA 0 cancel` / BSTOP hand every
checkpoint back, generate.cpp 17284), so cutting one loses the whole chunk - running on is the
cheaper harm, and it is R122's pre-existing rule, not P125j's. What P125j added on top - a
layer-boundary cut - could not help there anyway until the slot's read reader captures it (1c), so
the batch lane keeps the between-chunk pause and the grace 5 (1a) as its protection. Card follow-up
(the review's own wording): giving the slot reader a capture + the batch lane a cut is P125j's real
fix round, owed by P125j, not this branch.

**1c. The pause armed where nothing captures it - FIXED (500517c39).**
`engines/qfn/src/program/generate.cpp:12838` armed `sp.stop_between_layers` once for the whole serve
loop, live for EVERY `sp.run` reader: the batch slot's read thread (17152, joined at 17271, its err
reported at 17289 -> `BRE <slot> layer paused`, rows dropped) and the piped read's own thread
(21176 -> `e = read_err`, 21270-21272). Prefill's own gate (`lp_on`, prefill.cpp:2978) says only the
solo path can pause; the ARM now says the same: the global arm is gone, and `read_part`'s `run`
lambda arms it around its own `sp.run` behind a scope guard (`PauseArm`, disarms on every way out,
an exception included) and reads `paused_chunk()/paused_layer()/...` right after - the one place the
keep chain (23169+, `lp_pos`) can use the record. So: the solo GEN read (the shape the P125j rounds
3 verified and the card script exercises - a GEN line, `win` false past `req_short_read`) keeps its
layer cut; a slot's read, the piped read and the windows readers take the plain between-chunk /
between-window STOP, byte-identical to pre-P125j. `prefill.hpp:348-351`'s doc for the hook says to arm
only where `paused_chunk()` is read. No data race: every arm/disarm is on the serve-loop thread,
between joined reads (the slot path already swaps `should_stop`/`stop_at_chunk` per admission,
17083-17084).

**1d. SETTINGS.md - FIXED (c8f7c4d60).** The `SAGUARO_PRIORITY_IDLE_S` row and the Request-priority
paragraph now say the default is 5 again, and why the P125j rationale does not hold yet; the
`SAGUARO_LAYER_PAUSE` row names the armed reads and says plainly that under `--batch` the server
never cuts a batch read.

## NIT 1 - the ungated restore tail/refresh ride the gated-out shapes - NOTHING TO FIX; CARD OWES ONE RUN

The review's own words: "Correct per the contract... but the card run should cover a gated-out shape
(layer split or multi-card) once". No code change is claimed by that; P125k's REPORT §5 contract
(waits less, never under-wait; the device-wide pre-sync ahead of the copies still drains everything)
is what §2 now states honestly (f3d89c6c7). Noted for the manager's card script: cover a gated-out
shape once.

## NIT 2 - the FAST_LANE light path gained a sync of this thread's copy stream - NO CHANGE

Documented (REPORT §2, f3d89c6c7): `narrow_sync` waits the scoped streams PLUS the thread's private
copy stream - it waits MORE than the old `LightSync::run`, never less, and it is the thread's own
queued copies. Nothing to fix; a behavior change the review accepts as "documented".

## NIT 3 - out-of-repo absolute paths in committed prose - FIXED (7f5ba4f54)

`REPORT-p125k-park-sync.md:128`: the build command is now `QFN_GGML_DIR=../maya/third_party/llama.cpp`
(the sibling worktree's checkout - `scratch/build-p125k/CMakeCache.txt`'s `STRATA_GGML_DIR` confirms
that exact one, so the claim stays true) with "the box's CUDA 13.1 nvcc on PATH" for the compiler,
instead of `/data/llm/maya/...` and `/usr/local/cuda-13.1/bin/nvcc`. `FIX-REPORT.md:109` (the round-2
P125j line) names the venv interpreter descriptively instead of by path. Wording only.

## NIT 4 - the wiring is proven by build + card, not unit - NO CHANGE (accepted)

The review itself closes it: "The plan explicitly allows this". `wait_streams` carries the pure
decision (65 checks); the override hook and the six arming sites are what the build + card run
proves. Card: the manager's, next.

## What I ran (plan-named module only, one at a time, nothing on a GPU)

- `strata` rebuilt in `scratch/build-p125k-merge2`: 4/4, `saguaro-qfn` linked - `generate.cpp` and
  `prefill.cpp` (the doc-only `prefill.hpp` touch) compile clean; the one warning is the pre-existing
  unused `w` (now :15864). `generate.cpp` belongs only to `strata` (engines/qfn/CMakeLists.txt:803),
  so no other target needed a rebuild.
- `park_policy_test` (the only test module PLAN-p125k names): the build target (ninja no-work - the
  fix touches nothing it reads) and standalone `g++ -std=c++20 -I engines/qfn/include` - **65 checks
  passed** each.
- `tools/saguaro/extract_qfn_engine.py`: `serve/qfn_engine.py` regenerated **byte-identical** (run
  after the serve edits: the reverted default lives in the root `serve/server.py`, NOT in the qfn
  engine source - `engines/qfn/serve/server.py` was not touched, so `qfn_engine.py` did not change).
- `py_compile` on `serve/server.py` + `serve/test_priority.py`: OK.
- NOT run: `serve/test_priority.py` itself and `layer_pause_test` - the card says run ONLY the test
  modules the plan names, and PLAN-p125k names only `park_policy_test`; the two python edits are
  one default and the matching assertion, checked by reading `_env_seconds` (unset -> the literal
  default, serve/server.py:393-406) plus `py_compile`. The P125j rounds' own card must run
  test_priority (68) and layer_pause_test (43).
