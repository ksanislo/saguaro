# FIX-REPORT: P124a2 reviews (REVIEW-RESULT.md) - finding by finding

REVIEW-RESULT.md holds two reviews: the manager's MANAGER: FIX-FIRST card verdict on 568c15a16, and the
diff review `856af24a..568c15a16` (MERGE-WITH-NITS). Every FIX-FIRST finding is fixed; both review nits
are fixed; M1 and L1 landed with the manager round; M2 is declined with its reason below. One commit per
finding, as the plan and the run rules require.

| finding | where |
| --- | --- |
| MANAGER FIX-FIRST - parks still slow: the RUNNING-STATE copy crawls beside the turn save's writeback (checkpoints 574/1560 ms) | `b3166e0ff` |
| MANAGER FIX-FIRST - the card script deleted keys that merely contain "token" (it killed `tokenizer`) | fixed in `scratch/card-test-park-overlap.sh` - no commit: `scratch/` is gitignored and the plan left the script untracked |
| MANAGER FIX-FIRST - the card script must print its PASS/FAIL line | the script already ends on an `echo PASS: ...`; `fail()` now prints its `FAIL: ...` line on stdout too, so the verdict is in the same stream either way - no commit (scratch, untracked) |
| M1 MEDIUM - the RAM admission subtracts the adopted K/V before the adoption can still refuse | `57ff85c44` |
| M2 MEDIUM - only the pure booleans are executed; the adoption builder is compile-only here | unchanged - declined with the reason below |
| L1 LOW - rewrite sites that skip the park leave the hold alive | `a3d99c3a9` |
| N NIT - REPORT line-number drift (`turn_tokens_before` :19760) and the "floors the median" wording | `62dacfde4` |
| N NIT - the external-buffer guard samples one buffer | `900c6fd17` |

## FIX-FIRST (manager, 2026-11-26, 24K conversations) - the running state gets the same treatment (`b3166e0ff`)

The card showed the K/V half fixed (`copy 1 ms` where adoption runs) but two parks still crawled:
`checkpoints 1560 ms` / `574 ms` - the ~237 MB running state copied off the device right beside the
turn save's writeback, exactly what the park had skipped for the K/V. Three changes, the manager's
preferred shape (reuse the copy the turn save just made):

* the capture hold went **strong** (generate.cpp:10702) - the slow parks were the ones arriving after
  the writer finished, where the weak reference had already died; it is released by the park that
  consumes it (:10905), a continuing request, or a rewrite site, never later;
* `park::reuse_turn_state` (park_policy.hpp:168-171) - the five-term pure decision (a whole capture
  held, no layer split, nothing ran since, same ids/mode, no images) that says the capture's HOST
  running state is byte-identical to what this park would copy; `adopt_turn_capture` is now those five
  plus `no_retained` (park_policy.hpp:179-182), so a park whose K/V cannot be adopted (retained
  prefix, or a part that will not alias) still skips the running-state device copy;
* `conversation_snapshot_save` takes a `live_from` checkpoint (conversation_state.cpp:356-369) and
  beats in that host copy, checkpoint-validated, timed into the split's `checkpoints` like the save's
  - the park hands it in at generate.cpp:11034-11037.

The reuse decision is a pure helper, so per the manager it is unit-tested both ways: park_policy_test
41 -> **48 checks**, run as the plan's named module (`./scratch/build-p124a2/park_policy_test` ->
"park_policy_test: 48 checks passed"). Nothing else was executed (run rules), no GPU.

## MANAGER FIX-FIRST - the card script (scratch/, no commit possible)

* key stripping: the blanket `*key*`/`*token*` top-level strip deleted `tokenizer` (an essential
  config key). The filter now drops only api_key-like names (contains "api"+"key", or "apikey"), at
  top level and inside `env`; the `--api-key`/`--api_key` arg strip and the final `grep -qi "api.key"`
  guard stay. Verified against a synthetic config: `tokenizer`, `tokenizer_path`, `HF_TOKEN` and a
  plain `KEY` survive, `api_key` / `OPENAI_API_KEY` (top level and in `env`) go, and the grep guard
  is clean.
* verdict: `PASS: ...` was already the script's last line; `fail()` printed FAIL only on stderr, so
  the FAIL line is now echoed on stdout as well - exactly one PASS/FAIL line either way.
* header comment now says the LIMIT (not the median) is floored: `max(3 x median, 3 x FLOOR_MS)`.

## M1 MEDIUM - honest admission after a refused adoption (`57ff85c44`)

Exactly the review's first suggested shape: on a `conversation_snapshot_from_capture` refusal
(generate.cpp:11013-11029) the park re-reads `additional` without the `adopt_kv` subtraction
(:11022-11025, undoing the :10963-10965 estimate credit) and re-runs the same admit/evict loop
`admit_room()` **before** the fresh capture allocates the full K/V (:11026), with the same
refuse-with-one-line path if the honest figure does not fit. The post-capture floor check (:11046+)
stays as the after-the-fact spill it always was. The REPORT's open question #1 now names the
re-admission (the review noted it named only the refusal shape).

## M2 MEDIUM - declined: an executed adoption-builder test is outside what this run may execute

The plan names one test module ("Tests where a pure helper is added ... park_policy_test.cpp") and the
run rules say to run only the plan-named modules. A `conversation_snapshot_test`-style byte-equality
case (build a `SessionLogTurn`, adopt, compare images; plus a refusal shape and the alias charge) is
exactly what the review describes - but that module needs the device the plan forbids here and is not
plan-named, so a case added there could be compiled and never run: no executed assurance, only the
compile+link proof the eight conversation/session targets already give (REPORT section 3). The manager
round's "same tests; a unit test for the reuse decision if it is a pure helper" is honored where it
can be: both pure decisions (adopt AND reuse) are unit-tested both ways in the executed module
(48 checks), and the end-to-end effect - the adopted/reused bytes on a live park - is what the card
script measures for the manager. Declining follows the P124a precedent (F3/F4/F6: unexecutable-here,
reason recorded).

## L1 LOW - the park block has the last word on the hold (`a3d99c3a9`)

The review's cheap close: one `turn_main.adopt_clear()` after the switch's park block
(generate.cpp:19857-19862), run whether or not the park did (park_gain, the slot-hold `park_skip`,
`pin_sibling`), before any of the three rewrites that follow it (slot copy-in :19866, borrow restore,
incoming restore). The else-branch's own clear (:19798) stays for the same-session restart. Defense
only, as the review says - no wrong-bytes path was found; the REPORT's clear-site list (lines 95-98
now) lists this site too, closing the overstatement the review noted.

## N NIT - guard the whole image, not one buffer (`900c6fd17`)

The restore-keeps-image guard (generate.cpp:20029-20041) now checks EVERY layer's k/v/k_scale/
v_scale/pooled of the main image and of every stage image for `external()` before keeping the image
for the next park, instead of sampling `kv.front().k`. Behavior only widens the "do not keep" side:
a fully-external (aliased or file-mapped) image failed the old sample anyway, and a partly-external
one - the shape the sample assumed away - is now kept out of `retain()` by construction, not by
assumption. (Keeping it out is already forced downstream - a write visit to an external buffer fails
by design - so no wrong-bytes path existed; this makes the invariant local as the review asked.)

## N NIT - REPORT drift and wording (`62dacfde4`)

Both cited spots fixed: `turn_tokens_before` is at generate.cpp:19790 (the switch's `reset()` at
:19794, the park block at :19801-19851), and the card-test description says the LIMIT is floored
(`max(3 x median, 3 x FLOOR_MS)`), not the median. Since the manager-round commits moved every later
line, the rest of the report's generate.cpp/conversation_state.cpp citations were re-anchored to the
branch tip as well, and the three claims the report wrote before that round are now true of the code:
the hold is strong, fallback parks take the capture's host running state, and a refused adoption
re-admits before allocating. park_policy_test's count in the report: 41 -> 48.

## Verification

* `park_policy_test` (the plan's only named module) run once, standalone on the tip: **48 checks
  passed**. The generate.cpp nit edit and the manager-round edits compile in the plan's build dir
  (`scratch/build-p124a2`, sm_75; `generate.cpp.o` built after the last source edit).
* `git diff 856af24a..HEAD --name-only -- serve/ tools/` is empty: `serve/qfn_engine.py` was never
  touched (nothing to extract), no new stdout field, no new env switch.
* The card script passes `bash -n`; its key filter was unit-checked against a synthetic config
  (tokenizer kept, api_key-like dropped). Running it belongs to the manager (needs the card).
