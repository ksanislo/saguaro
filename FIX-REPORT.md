# P124q FIX-REPORT - REVIEW-RESULT.md (of 93919888..23e2e9c3f) -> what each finding got

One commit per finding that changed code or tests, then this report.  `serve/qfn_engine.py` stays untouched and
nothing was re-extracted: the change is in `Service.run.new_parser` / the token loop, never in the generator's
roots (`StrataEngine`, `Vision`, `engine_args`, ... - tools/saguaro/extract_qfn_engine.py:14-16), and the review's
own grep (`token_only_end engines/ serve/qfn_engine.py`) found nothing there.  Tests: only the two modules the
plan names, one at a time, all after the last code edit: `serve.test_loop_guard` **67 OK**,
`serve.test_server` **179 OK**.

## HIGH 1 - step 3 was not done: no test exercised the change - FIXED (367205ad6)

serve/test_loop_guard.py, next to the QFN case, per changed engine kind:

- `test_a_quoted_end_tag_stays_text_for_the_other_token_rule_engines` (:1154) runs the QFN shape for `engine_kind`
  "glm" (serve/chat_template.jinja), "minimax" (tools/mm3/chat_template.jinja, closing on its `</mm:think>`, with
  `MiniTok` at :1056 holding that tag as one token) and "hy4" (serve/fixtures/hy4_flash's template; the
  vocabulary check falls back from the suffixed name to the plain `</think>`, which is what the pack's tokenizer
  names it anyway).  Each asserts: one engine call, finish stop, `think_end_id` set, the quote spelled out of
  ordinary pieces (`parse_special=False`) still in the reasoning, the answer after the real token, and the tool
  call written after it kept.  This is the `template=`-hook route the review named; the QFN case's
  `kind=None` branch (:1151) stays as the contrast that pins the text rule for the token-less engines.
- The uncommitted helper edit (`answer(..., template=...)`, the `Service(eng, tok, template)` branch, now
  :1087) is committed as part of this; its `MINIMAX_NS` import was dropped (nothing used it; `MiniTok` needs the
  `</mm:think>` literal, which is spelled out as the file already spells its other tags).

## HIGH 2 - MiniMax with a plain `</think>` token never closed its thinking - FIXED (e627b25a3)

The review's two choices were "show saguaro-mm3 never samples a plain `</think>`" or "let a minimax parser close
on that id too".  The vocabulary question is unanswerable offline (no model file, no GPU here), so the fix took
the second:

- `Service.__init__` (serve/server.py:3339-3344): for a minimax template that has its `</mm:think>` token, the
  plain `</think>` id too - if that vocabulary holds it as one token and it differs - goes to a new
  `think_end_id_alt`.  No other engine, and no MiniMax vocabulary without the second name, gets a value: `None`.
- `run()` reads it once per request (`alt_end`, serve/server.py:6104 - via `getattr` so the hand-built `Service`
  of serve/test_think_budget.py:63-65, which sets three attributes on `object.__new__`, still works) and both
  token checks use it: `end_think` (serve/server.py:6249) closes the reasoning through the existing
  `parser.close_reasoning()` path (:6393-6394), and the text rule (serve/server.py:6257-6262) swallows the
  token's text so the tag never appears in the answer.
- Deliberately kept on the canonical `</mm:think>` id: the budget's forced close (`_think_end`,
  serve/server.py:6139), the restarts' closes (serve/server.py:6291, :6333) and the loop guard's
  `ToolSafeGuard(self.think_end_id, ...)` (:6164 - its `think_end` is thinking-vs-answer classification plus the
  token it may append; an appended canonical id closes the reasoning either way, and loop_guard.py is out of the
  plan's changed set).
- A quote of the tag spelled out of ordinary pieces is still text, not the alt id - the bug class this branch is
  about stays fixed for MiniMax too.
- Regression test: `test_minimax_closes_on_a_plain_end_tag_token_too` (serve/test_loop_guard.py:1180) - both
  names in the vocabulary, the model samples the plain token: the answer and the call after it come out and no
  `THINK_STOP_NOTE`.  Measured genuine: with `git checkout HEAD~1 -- serve/server.py` it reads
  `FAILED (failures=1)`, with the fix `OK`; the reviewer's own scratch/probe_p124q2.py case (b) now prints
  `text='ANSWER OK...'` for `kind='minimax'`.

## LOW 3 - uncommitted step / unfinished report - FIXED (367205ad6 + the report file)

- `git status` is clean of code now: the partial helper went in with 367205ad6, the fix with e627b25a3.
- REPORT-p124q-think-end.md is finished: the step-2 write-up (the per-engine rule, IK2 and every token-less
  vocabulary keeping the text rule, the MiniMax alt-token line) and the step-3 note (what tests, what was run:
  67 / 179 OK), and it now says outright that step 1's line numbers are the pre-change ones, giving the numbers
  after 23e2e9c3f (6241 / 6249-6254 / 6385 / 10009) and after e627b25a3 (6249 / 6257-6262 / 6393-6394 / 10017).
- Nothing to commit there: PLAN-*.md, REPORT-*.md, REVIEW-*.md and FIX-REPORT.md itself are all
  `.git/info/exclude` entries (.git/info/exclude:117, :28, :41-43).  Only code and tests carry commits; this
  report is added with `git add -f`, as P124l's was (e3f322bfc).

## LOW 4 - the gate reads `engine_kind` while the id comes from the template - NOT CHANGED, the reason

The review's `not getattr(self.template, "inkling", False)` form is not a cheap nit, it moves the OFF paths it
itself verified byte-identical: every Service with a vocabulary holding `</think>` and no `engine_kind` - both
modules' mock services, and the QFN case's `kind=None` branch (serve/test_loop_guard.py:1144-1152), which exists
exactly to pin "the text search, the other engines'" - would flip from the text rule to the token rule.  That
needs those tests rewritten and decides the rule for services by their *tokenizer* alone, which is a semantics
change for a hypothetical the review calls harmless today.  And the file-name worry has no live case:
`family_of`'s default for an unrecognized config is "glm", already in the tuple (tools/maya_quants.py:303); DS4
is covered by `template.deepseek` regardless of its family string; and Inkling 2 keeps the text rule twice over -
"ik2" is outside the tuple AND its vocabulary has no `</think>` special, so `think_end_id is None` guards it.
Both forms decide every production engine the same way today; the tuple form is the one the suite pins.

## LOW 5 - nothing recorded the token id per engine - FIXED (in the report, as far as it is measurable)

REPORT step 2 now carries one line per engine: QFN's 248069 (the comment at serve/server.py:5971), Hy4's id as
asserted in serve/test_hy4_frontend.py:280, MiniMax's two names as the single-token lookups at
serve/server.py:3335-3344, GLM's as the very id the budget's forced close sends (serve/server.py:6139 ->
src/program/generate.cpp:2467-2469, 2507-2508).  A measured *number* per model needs the model files and a GPU,
which this task does not have, so the record is the lookup that finds the id, not a number - the report says so.

## LOW 6 - the spelled-out-tag risk spread to three more engines - NOTED (report)

Stated plainly in REPORT step 2: a model that spells the tag out of ordinary pieces no longer ends its reasoning
at the token rule - for GLM and Hy4 (and MM3) as it has been for QFN since 37cd8d829; what is left is the loop
guard's third loop (docs/SETTINGS.md:472), the budget's forced token (serve/server.py:6139) and the salvage at
serve/server.py:6433-6458.  No code change: accepting it is what step 2 of the plan asked for.

## LOW 7 - test_minimax_frontend / test_hy4_frontend could not cover the change - NOT CHANGED, the reason

Both build their Service without an `engine_kind` (serve/test_hy4_frontend.py:95-98), so the gate keyed on it
never fires there - by design: those modules test the frontend and the template, and the gate's engine-kind key
is the OFF-path question of finding 4.  The coverage the plan asked for went where the plan named it -
serve.test_loop_guard now runs the real gate for glm, minimax and hy4 - and neither frontend module is on the
plan's run list, so neither was run or touched here.

## Runs (only the modules the plan names, one at a time, after the last code edit)

    /opt/llm/saguaro/.venv/bin/python -m unittest serve.test_loop_guard  -> Ran 67 tests, OK
    /opt/llm/saguaro/.venv/bin/python -m unittest serve.test_server      -> Ran 179 tests, OK

Plus, as evidence and not as kept tests: the reviewer's scratch/probe_p124q2.py re-run (case (b) now closes and
answers), and the pre-fix check of the new regression (FAILED with the old server.py).  No Task/Explore/General
agents were used; everything was done in this session.
