# C18 fix report - answers to ./REVIEW-RESULT.md (FIX-FIRST)

One commit per finding/nit, in review order. The verdict's blocker (finding 1) is fixed, findings 2-4 are
fixed, and both cheap nits are fixed; the `engines/` plan-bar nit is left for Ken, as the review itself
framed it. No test-module result regressed: the six named modules are green, run one at a time, counts
under.

## Finding 1 (HIGH) - the live slider move persisted a context the saved factor cannot reach - FIXED
Commit 36e36e484. `_save_context` (serve/server.py) now writes `--rope-scale` with the same `%g` rule as it
writes `--max-context`, so a 1M move on a 512K config saves `--max-context 1048576 --rope-scale 4` and the
next start's `engine_args` spawn reaches its saved context; the mirror move saves 2 with the 512K instead of
keeping 4. The rule is one function now - `models.qfn_yarn_follow` (serve/models.py) - called from
`_save_context` and from `apply`'s qfn branch (the page path it was inline in), the third of the three
writers next to the driver's copy in engines/qfn/serve/server.py's `set_context` (untouched - the C++ and the
engine-side driver stay as the review found them; no extract needed). Garbage in `--yarn-orig-ctx` is read
through `models._int`: a hand-broken config can no longer make `_save_context` drop the whole save, the pair
check just does not fire.
Caught by: the new `Dashboard.test_a_qfn_reload_saves_the_yarn_factor_with_the_context`
(serve/test_dashboard.py) - the review's 1M POST, but on a config that carries the YaRN pair, asserting both
moves through the saved file. It fails without the fix (the config keeps `--rope-scale 2` under 1M).
Knock-on: REPORT-c18 line 35's "a live change rewrite[s] --rope-scale with the new max" is now true of the
saved config as well as the running command, and docs/SETTINGS.md's closing "a move of the slider rewrites
`--rope-scale` with the new max" is true of the file the next start reads.

## Finding 2 (LOW) - the driver's sync branch had no test that ran it - FIXED
Commit 1349f5118. `QfnContext.test_a_past_256k_line_moves_its_yarn_factor_too` (serve/test_dashboard.py):
the 512K spawn line the review names, moved to 1M, asserted `--rope-scale 4`; plus back down to 512K (2) and
a move to the trained 262144 (factor stays - the at-or-below arm), and never a second `--rope-scale` pair.
Runs the real driver through `_qfn_engine_class`, i.e. serve/qfn_engine.py:1101-1109.

## Finding 3 (LOW) - the two syncs gated differently - FIXED
Commit 17a3de1c7. `qfn_yarn_follow` gates on the config's own `--yarn-orig-ctx` and computes with the same
number - the driver's rule - so there is one gate for all three writers. Identical for every value the setup
writes (262144); for a hand-edited origin the page can no longer write a factor under 1 (the constant's gate
with orig 524288 and a 384K context would have written 393216 / 524288 = 0.75, past the engine's own `>= 1`
check). Caught: a hand-edited case added to `test_qfn_context_keeps_the_yarn_scale_in_step`
(serve/test_models.py) - orig 524288, context 384K: the factor stays 2.

## Finding 4 (LOW) - one-card AMD can now be answered "2" - FIXED, as a warning rather than a skipped question
Commit da5f30784. Not skipped, deliberately: the setup's own vision-off line on this very layout says
"--parallel 2 brings the pictures back" (saguaro.py:8192) - answering 2 is the one AMD card's batch-slot lend
path for the picture encoder, and dropping the question on HIP would take that from the interactive flow
while contradicting the message printed beside it. Instead the answer now hears the measurement: an answer
above 1 on the one-card HIP layout warns that parallel decoding lost speed on the measured RX 9070 XT and
that 1 is the measured choice - and the answer stands (it is the user's lend path, and the default stays 1).
"--parallel N given = not asked" keeps its full meaning: given 2 on the command line is neither asked nor
warned ("--parallel N says"). Caught: `test_above_one_on_the_one_amd_card_hears_what_it_measured` and
`test_parallel_given_on_that_card_is_deliberate_and_untold` (tools/test_setup_qfn.py). If Ken wants the
question skipped outright, that is the two-line version of this same spot.

## Finding 5 - nits
- Units - FIXED (commit f91d613a2). The two GiB figures labelled GB in `qfn_context_cost` (the refuse line
  and the warn line; `mem_gb` returns GiB) now read "GiB of RAM". The K/V cost figures stay decimal GB - the
  plan states them that way ("about 12.7 KB a token", ~27 GB at 1M x 2) and they are honestly labelled; the
  docstring now says which figure is which unit. One test string updated (the RAM-poor-box warning).
  `memlock_tier_warn`'s older GiB-labelled-GB line predates C18 and is outside this diff - left alone.
- Refused `--context` printing "context: N tokens" first - FIXED (commit e42b21fa4). The past-1M/part-step
  refusal moved above the `ok("context: ...")` line (the valid-case print order is unchanged); both refusal
  tests now also assert the refused size never says it was taken.
- The plan bars `engines/` while the diff touched its `serve/server.py` - NOT CHANGED, and not changable
  here: step 3's "applies like any context change" forces the driver's `set_context` to sync the factor, and
  the generated-file rule allows only that one route (edit the engine-side driver, re-extract); the engine's
  C++ is untouched. Ken's call, exactly as the review left it. The findings above add nothing under
  `engines/` - finding 1's shared rule lives in serve/models.py and the driver's copy was already right.

## Test runs (each module alone, /opt/llm/saguaro/.venv/bin/python -m unittest <mod>)
tools.test_setup_qfn 147 OK (1 skip; was 145, +2 for finding 4); tools.test_qfn_fit 47 OK;
tools.test_setup_amd 35 OK; serve.test_models 63 OK (one case extended, count unchanged);
serve.test_dashboard 79 OK (was 77, +1 for finding 1, +1 for finding 2); serve.test_new_engines_ui 35 OK.
No card runs, no network. `serve/qfn_engine.py` re-extracted after the work: byte-identical, tree clean.
