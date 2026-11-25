# FIX-REPORT — P6 wording: REVIEW-RESULT.md finding -> the fix that followed it, or why it stands

Five fix commits on `saguaro/q1-p6-wording` after the review (base `1862ecf7d`, review head `32707f843`), one per
finding area. Tests re-run one at a time with `/opt/llm/saguaro/.venv/bin/python`, all plan modules at the review's
numbers: `tools.test_setup_qfn` 89 OK (skipped=1), `tools.test_setup_ds4` 82, `tools.test_setup_amd` 34,
`tools.test_setup_ik2` 54, `tools.test_setup_mm3` 31, `tools.test_setup_hy4` 33, `tools.test_docs_new_engines` 7,
`serve.test_dashboard` 62, `serve.test_new_engines_ui` 35, `serve.test_model_api` 28; `py_compile saguaro.py` clean.
`serve/` and `engines/` were not touched, so no `serve/qfn_engine.py` re-extract was owed.

## Finding 1 (High) — Pascal claimed "not measured" while the repo's own README has the measurements

Fixed in `d8ce0b6da`. Every user-visible "not measured" Pascal claim now says the known fact, and nothing was
invented - the numbers are the README speed table's own (2x Tesla P40 28.0 / 16.9 tok/s, #212/#214; 3x Tesla P100
17.9 tok/s, #479/#504), the P40 kernel check is docs/INKLING2-MINI.md's own ("checked on one Tesla P40 24 GB ...
CUDA 12.0"):

- `README.md` (What you need, NVIDIA line): "Saguaro has not been measured on such a card yet" -> "has been measured
  on users' cards: 2x Tesla P40 at 28.0 tok/s (#212/#214) and 3x Tesla P100 at 17.9 tok/s (#479/#504), in the speed
  table above". The contradiction with README.md:141/157 is gone. The review suggested a "not validated on one by us"
  tail (Mac-line framing); left out on purpose - docs/INKLING2-MINI.md records a team-side P40 check, so whether the
  team owns a Pascal card is not a fact this repo states.
- `saguaro.py` `check_pc`'s Pascal warning: "...but Saguaro has not been measured on such a card yet" -> "it has been
  measured on users' Tesla P40 / P100 cards (README speed table: #212/#214, #479/#504) - tell us how it goes on yours
  in an issue" (the invitation to report stays: the user's card / model / layout is the new datum).
- `saguaro.py` module docstring (`--help`): "(6.x, Pascal: not measured yet)" -> "(6.x, Pascal: measured on users'
  P40 / P100)".
- `saguaro.py` `VALIDATED_CC` comment: "Pascal not measured yet (#212)" -> "Pascal measured on users' P40 / P100 only
  (#212/#214, #479/#504, README table)" - the thread that made the comment pointedly cite now reads with it, not
  against it.
- `saguaro.py` refusal line ("none of the GPUs can run this model"): "Pascal ... is not measured yet" -> "measured on
  user cards"; GPU list line: "can be used (Pascal: not measured yet)" -> "can be used (Pascal: measured on user
  cards)"; `tools/test_setup_qfn.py:1420` follows the new string with the same `assertIn` strength (the plan allows
  re-pinning, nothing weaker).
- `saguaro.py` `arch_setting`'s docstring carried the same false claim ("Pascal: not measured yet") although the
  review listed only the lines above; fixed the same way ("measured on users' cards").

## Finding 2 (High) — AMD_SAGUARO.md said MI325X speed was not measured, above its own MI325X numbers

Fixed in `f2fde4c2e`. `docs/AMD_SAGUARO.md:305`: "its speed is not measured here yet" -> "the MI325X has been benched
below, the MI300X itself has not been tried yet" - exactly the facts the section carries (the `--bench` run on one
MI325X: 84.0 tok/s answers, 1307 / 1663 tok/s prompts, v1.1.27, Maya-L, 1M; its own "Not tried yet: the MI300X
itself, more than one GPU"). `scratch/hits.txt`'s verdict line for 305 was updated in the worktree so the sweep record
agrees with the artifact (scratch is not committed, as before).

## Finding 3 (Low) — the report's own line counts did not reproduce

Fixed in `e2e16b206`, measured against the *final* diff (the review's -2 / 110 were true at its own head; findings 1,
2 and 5 then moved them again): `REPORT-p6-wording.md` now says saguaro.py 39 insertions / 39 deletions (0 net) and
README.md + the 13 docs files 115 changed lines - `git diff --numstat 1862ecf7..HEAD` reproduces both. The fix-round
commits, their re-run results, and the passages the fixes superseded (the Pascal "keep their facts" bullet, the
MI300X / MI325X example, the ik2_fit / hip_backend open questions) were brought in line with the fixed text.

## Finding 4 (Low) — labels the rule reaches though the plan's grep list does not

- `tools/ik2_fit.py:338`: fixed in `ee95bd09c` - "(Pascal, experimental)" -> "(Pascal)". It is printed by the setup
  (saguaro.py's `IK2F.unsupported` refusal), the fact it carries (P40 / GTX 10 or newer, compute capability 6.0+)
  stands, and `tools.test_setup_ik2` (which pins the "compute capability 6.0" part, not the tag) stays green at 54.
- `cmake/hip_backend.cmake`'s "not validated on a real card yet": **left**, and this agrees with the review's own
  disposition ("a later step" is "the right handling"). Two reasons: the plan's file list excludes `cmake/`, and the
  only module that verifies the line and its pins, `tools.test_hip_archs`, is outside the plan's run list - a wording
  change there could not be run against its pinned test under this branch's rules, and changing both blind is worse
  than the documented later step. The report's open-questions entry now states that reason.
- `CHANGELOG.md:1357`'s dead anchor: untouched, as agreed - history the plan excludes.

## Finding 5 (Nit) — small drifts

Fixed in `ee88a4ab4`.

- `docs/EXL3-HELPER.md:51`: the doc now carries `scratch/hits.txt`'s own verdict words - "It is a budget that is not
  measured yet - a starting point, not an automatic recommendation..." - so artifact and verdict agree (chose the doc
  side because the committed artifact is the one a reader sees; `scratch/hits.txt` stays uncommitted either way).
- Comment columns: the five re-wrapped continuation comments (HIP_ARCHS_UNVALIDATED / HIP_ARCHS_WAVE64) are back on
  the file's 51-space column; the two one-space drifts the branch added besides them - the `Maya-M-Derisked` "about"
  line (17 -> 18 spaces) and the Pascal card-line branch (39 -> 40) - were undone by the same hand.
- "not on another card of these (yet)": now "on no other RDNA 2 card yet" in the `--help` docstring (whose broken
  "Linux / and Windows" line-break was re-flowed too), the HIP refusal, and the RDNA 2 warning; the
  `HIP_ARCHS_UNVALIDATED` comment carries the same wording. docs/AMD_SAGUARO.md:56's prose ("...and not on another
  RDNA 2 card yet") reads fine as a sentence and was left.

## Not acted on

- The review's "Verified clean" and "Answers to the brief's wording questions" sections record no action needed;
  `serve/` stayed untouched (0 diff lines here too), identifiers/fixtures unchanged, no generated file edited.
