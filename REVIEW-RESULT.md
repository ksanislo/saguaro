# REVIEW-RESULT - P125j layer pause (review of d789228f0..worktree, redo #2)

FIX-FIRST

Reviewed range: the committed diff `d789228f..HEAD` contains ONLY the two markdown files
(REPORT-p125j-layer-pause.md, the previous REVIEW-RESULT.md) - steps 2-4 exist solely as the
UNCOMMITTED working tree (5 modified files + untracked engines/qfn/include/strata/program/layer_pause.hpp,
+273/-9). I reviewed base..working-tree including the untracked header. The plan's "commit after each
numbered step" was not done for steps 2, 3 and 4.

## BLOCKER

1. The tree does not compile. `lp_resume` is used at prefill.cpp:3214 and prefill.cpp:3274 (C-4 gather
   skip and gr_broadcast guard) but declared at prefill.cpp:3572 - used before declaration in the same
   chunk-loop scope (loop head prefill.cpp:3157). Reproduced on this box:
   `cmake --build scratch/build-p125j-cuda75 --target strata_prefill` fails with
   "error: 'lp_resume' was not declared in this scope" at both lines - matching the coder's own
   scratch/build-strata.log:1891,1894 (build ends in Error 2). The REPORT's claim "build exit 0,
   no GPU run" (REPORT-p125j-layer-pause.md:145-146) is false; the log in scratch/ says otherwise.
   generate.cpp (target `strata`, CMakeLists.txt:795) is downstream of strata_prefill and was never
   reached, so nothing engine-side has ever been compiled.
   Fix: hoist `bool lp_resume = false;` (and the pause-resume block, or at least the flag) to the top
   of the chunk body, before prefill.cpp:3214 - and re-run the build until it exits 0.

## MAJOR

2. Step 4 deliverables named by the REPORT do not exist.
   - `engines/qfn/src/program/layer_pause_test.cpp` and its `add_test` CMake entry (claimed at
     REPORT:141-144): absent - engines/qfn/src/program/ has no such file; no CMakeLists mentions
     layer_pause. Nothing in the repo unit-tests the pure helpers. (My own scratch harness does:
     scratch/review_lp_check.cpp walks pause_point/resumes/state_bytes incl. last-layer, short-last-
     chunk, one-layer-stage and saturation cases - all pass; the helpers themselves are correct.)
   - `tools/card_tests/layer_pause.sh` (claimed REPORT:147-152, port 8092, the four PASS conditions):
     absent - tools/card_tests/ holds only lost_read.sh and park.sh. So no test exercises the change;
     "reused + read = the prompt, answer matches an uninterrupted read" is unverified.
3. Step 3 is incomplete and the REPORT's claims for it are false:
   - `SAGUARO_PRIORITY_IDLE_S` default is still 5.0 at serve/server.py:416; docs/SETTINGS.md:238 and
     :250 still say 5; serve/test_priority.py is untouched (it pins the value in env, so its default
     assertion was the thing to change). REPORT:133-135 claims 5 -> 0 was done. Leaving 5 is the safe
     direction (MANAGER: grace -> 0 only with a working pause), but with the BLOCKER the feature is
     not working, and the claim misstates the tree.
   - The server pause line still prints only "prompt read paused at X of Y prompt tokens"
     (serve/server.py:5875-5877); serve/server.py is not modified at all, while REPORT:131-132 claims
     it names the layer and chunk. The new DONE keys are parsed nowhere: engines/qfn/serve/server.py
     accepts paused_layer/paused_chunk/paused_at (its:1636-1637) but no consumer reads
     `self.last["paused_*"]` anywhere - dead until the line is written.
   - Generated file out of sync: engines/qfn/serve/server.py changed, but serve/qfn_engine.py was NOT
     regenerated - it still has the old key tuple at serve/qfn_engine.py:1194 while the source has the
     new one at engines/qfn/serve/server.py:1636-1637. REPORT:136-137 claims it was regenerated with
     tools/saguaro/extract_qfn_engine.py. The server runs serve/qfn_engine.py, so the parse change is
     dead there too. Regenerate (never hand-edit) as part of step 3.
4. A parked paused read can corrupt a BORROWER of its prefix. The parked-paused image is excluded
   from the session dir (generate.cpp:11108 `!parked_paused`) and its rows park with it
   (generate.cpp:11086-11106), but nothing excludes it as a prefix-borrow donor: the borrow decision
   (generate.cpp:19704 `conversations.borrows(parked)`) and the donor restore
   (generate.cpp:19985-19999, `conversation_snapshot_restore_prefix(donor->..., parked.tokens, ...)`)
   copy state through `parked.tokens` = the paused chunk's END - for which the donor has only layers
   < pause_layer written - while the pause record and rows stay with the donor and never transfer.
   The borrower then treats [pause_pos, pause_end) as fully read and decodes over half-written cells
   (silently wrong tokens). Exclude `paused()` images from the borrow/parked-candidate selection
   (generate.cpp:19696-19716), or cap a donor's borrow depth to its pause_pos.

## MEDIUM

5. Stale pause across session rewrites. `drop_read_pause` (generate.cpp:10453) is not called where
   the live session is replaced WITHOUT a park: the session-file LOAD path (generate.cpp:19184 sets
   live_ok=false, 19210 assigns `live`), the slot-restore failure `live.clear()` (generate.cpp:19928)
   and slot copy-in (generate.cpp:19923-19962), and the incoming/borrow adopt at generate.cpp:20071
   when the outgoing park was skipped by `park_worth`/`park_skip` (its gate at generate.cpp:19858-19908
   - the comment at 19914-19918 says exactly this about the turn track). Correctness then rests only
   on the consumer-side cross-checks (generate.cpp:20259-20272: length + chunk-ids +
   `sp.paused_chunk()/paused_tokens()` agreement) and the run() clear (prefill.cpp:6184); a stale
   armed pause was not rejected for the reason the invariant intends. Call drop_read_pause() at
   every point that rewrites the session.
6. `resumes()` is called tautologically: prefill.cpp:3574-3575 builds the Point from the CURRENT c0
   (`resumes({c0, pause_.layer, pause_.tokens}, c0, n, C)`), so the documented chunk-grid check
   "the pause sits on this run's chunk grid at c0" (layer_pause.hpp:45-58) can never fail. What
   actually protects resume is `pause_.pos0 == pos0 + c0` (prefill.cpp:3574), the rows-size check
   (prefill.cpp:3576) and the id compare (generate.cpp:20264-20266) - which together do hold (I
   checked the cross-run case: leg 1 saves a run-relative chunk mid-run, the continuation run matches
   at c0=0 via the absolute pos0; scratch/review_pausepoint.cpp). Either pass the saved run-relative
   chunk (store it in SavedPause) or drop the dead parameter pair and fix the doc.

## NIT

7. `take_pause_rows` (prefill.cpp:1957) moves the rows out but leaves pause_.pos0/layer/tokens armed,
   so `paused_chunk()/paused_tokens()` keep reporting a pause with no rows. Only the park path uses
   it (and clears right after, generate.cpp:11103-11104), but reset `pause_` in the getter itself.
8. After a second pause whose keep branch is refused (generate.cpp:22823 conditions fail, e.g.
   prompt_cache=0), `read_pause` keeps the FIRST pause's record while the prompt path holds the
   SECOND pause's rows - a later park (generate.cpp:11093-11106) would attach mismatched
   record+rows. The go_on cross-checks reject it (paused_chunk compare, generate.cpp:20267-20268),
   so nothing unsafe happens; still, reset read_pause whenever the keep branch declines a pause.
   Related: the restore at generate.cpp:20083-20091 assigns read_pause before `give_pause` can
   decline (prefill.cpp:1964), same shape.
9. `lp_chunk = (lp_pos - q) / sp.chunk()` (generate.cpp:20587) divides by m.T, but the run's C can be
   a planned chunk (prefill.cpp:3029 `C = ... planned_C > 0 ? planned_C : m.T`) - the
   `paused_chunk=`/log label is then wrong (cosmetic; the resume itself uses absolute pos0 + C).

## Checked and OK

- Feature OFF (SAGUARO_LAYER_PAUSE=0, or callback unset): lp_on false (prefill.cpp:2973) guards every
  new block; the resume-skip degenerates to the old gather path (`!lp_resume` always true); the
  run()-side clear (prefill.cpp:6184) is a no-op with `pause_.pos0 == -1`; generate-side keep needs
  lp_pos >= 0 which only a pause sets; the DONE line gains `%s` but prints the empty string ->
  byte-identical output; SavedConversation::bytes() adds only `capacity()`s (0 when unused,
  conversation_cache.hpp:201-202) and the disk format is untouched. OFF behaviour unchanged.
- Pause mechanics: the layer-pause check sits at the end of a full layer pass after the U5 heartbeat
  (prefill.cpp:5496-5532), stream-syncs before the DtoH, never after the stage's last layer
  (layer_pause.hpp:36), and returns through the existing early-return unwinds (IssuerJoin/StagerDone
  prefill.cpp:3469-3512, the PLE future's blocking dtor prefill.cpp:3119); `normed` stays false so the
  resumed half recomputes the norms from m.R (prefill.cpp:3583). K/V/GDN/indexer state parks for free
  (session state, live runs through the chunk, `conversation_snapshot_save` covers it). Re-read
  fallback idempotence as argued in the REPORT holds by inspection.
- Locks/threads/interleaving: no new mutexes; the stop flag is the same `std::atomic` read by the
  stdin thread (generate.cpp:12688-12692) as `should_stop`; pause log is one fprintf+fflush on stderr,
  the DONE suffix a single printf - no line interleaving. No nested lock acquire in park/restore.
- No secrets, no paths outside the repo in the change; scratch/ build dir is repo-internal.
- Pure helpers (pause_point/resumes/state_bytes): correct per my scratch/review_lp_check.cpp (all
  pass) - they just need the repo-side test the plan promised.

## What I ran

- `cmake --build scratch/build-p125j-cuda75 --target strata_prefill` - FAILS (BLOCKER 1, reproduced).
- g++ scratch harnesses on layer_pause.hpp (scratch/review_lp_check.cpp, scratch/review_pausepoint.cpp)
  - helpers pass, misuse shown.
- py_compile on engines/qfn/serve/server.py, serve/qfn_engine.py, serve/server.py - syntax OK.
- Plan-named test modules: layer_pause_test and tools/card_tests/layer_pause.sh do not exist; the
  card test needs a GPU run the plan itself excludes here; serve/test_priority.py was NOT run - its
  subject (the grace default) is unchanged in the tree, so it exercises nothing of this diff.

## Manager's four

- Resume at the exact layer: implemented (l_from = saved layer, prefill.cpp:3580,3598) but the tree
  cannot build, so unproven.
- Nothing read twice: design holds (paused chunk counted as reused, layers < L skipped on resume,
  run() clears a pause its run re-read whole, prefill.cpp:6184) - unverified on-card (script missing).
- Parks with the conversation: park/restore paths carry record+rows and refuse disk writes
  (generate.cpp:11086-11108, 20083-20091) - but the borrow donor hole (MAJOR 4) and the stale-pause
  sites (MEDIUM 5) are open.
- Idle grace -> 0 only with this: grace still 5.0 (serve/server.py:416) - correct given the BLOCKER,
  but step 3's promised change (and its doc/test updates) is missing.
