# P124g report - the last two Python 3.12 generator wrappers (1.2.4)

## 1. What changed

**`serve/server.py` - `openai_chunks`** (commit 521d68958): the bare

    for kind, x in run if run is not None else svc.run(...):

is now, exactly as `88809c4ea` did for `run_with_mcp` / `anthropic_events` / the QFN engine's `generate`,

    with contextlib.closing(run if run is not None else svc.run(...)) as _inner:
        # (closed on any exit: Python 3.12 ran the inner STOP + drain only when the outer one was freed)
        for kind, x in _inner:

body indented, nothing else touched (the whole loop incl. the `^ kind "ping"` comment moved into the `with`; the
one `finish = ...` line kept its single-line form, just indented). This is the plain chat/completions and the
ollama path, so on Paul's 3.12 box a client gone mid-answer used to leave `Service.run`'s STOP + drain to the
frame GC - the next request could read the old answer's tokens.

**`serve/responses.py` - `response_events`: no change.** The plan's fallback ("close the passed `run` in a
`try/finally` if a `with` reads worse") is already there: the whole `for kind, data in run:` loop sits inside a
`try:` whose `finally: run.close()` (:472-473) closes the passed `run` on every exit, GeneratorExit included -
so a `.close()` on the outer `events` closes the inner `run` before `close()` returns, on any Python. Blame:
8a70c37463 (2026-10-20), well before P124c's audit - the audit's "iterates ... with a bare `for`" entry for
`response_events` is a false positive (it checked for `contextlib.closing`, not for the equivalent `finally`).
Re-wrapping the loop in `contextlib.closing` would only re-indent ~70 lines and move `run.close()` before the
`except` clause's `close("incomplete")` yields - churn, no behavior gain. The new test keeps it true anyway.

**`serve/test_priority.py`** (commit 547e3af57): `ClosingTheOtherWrappersClosesTheirRun` (modelled on
`StopReachesTheEngine`, appended after it): one shared fake run generator whose `finally` records that it ran;
per wrapper, drive the outer a few items, keep the reference, `.close()`, assert the record is there - i.e. the
inner `finally` ran before `close()` returned. `openai_chunks` gets a bare dict `req` and a Mock `svc` with its
`run=` override; `response_events` the minimal `("m", {}, run)`. On 3.13 both pass with or without the fix
(expected - the manager runs them on 3.12).

`ref/` (glm.py, load.py, upstream-llama.cpp) holds nothing for these two wrappers - no `closing` there.
Nothing else in `serve/` iterates an event generator with a bare `for` without closing it: `response_events`'s
loop and `response_schema.ResponseValidation.events`'s buffered loop both end in `finally: run.close()`
(response_schema.py:201-204), and `run_with_mcp` / `anthropic_events` / the QFN engine's `generate` were fixed
by `88809c4ea`. `serve/qfn_engine.py` was not touched, so no re-extraction was needed.

## 2. Files

- serve/server.py            (part 1, the `openai_chunks` wrap)
- serve/test_priority.py     (part 2, the two tests)
- REPORT-p124g-py312-closing.md  (this report)

## 3. Tests run (`/opt/llm/saguaro/.venv/bin/python -m unittest`, this box = Python 3.13.5)

- serve.test_priority: 55 tests, OK - incl. the 2 new ones (run alone: 2 tests, OK)
- serve.test_server:   179 tests, OK
- serve.test_responses: 25 tests, OK (2 skipped: "optional OpenAI SDK is not installed", pre-existing)

## 4. Open questions

1. 3.12 proof is by the manager only: no python3.12 exists on this box (the venv is 3.13.5 despite the
   `py312` worktree name), where all three modules pass either way - exactly as the plan expects. On Paul's
   3.12 box `test_closing_the_openai_answer_closes_its_run_before_close_returns` is expected to fail without
   521d68958 (same shape as `StopReachesTheEngine`), and
   `test_closing_the_response_answer_...` passes without any change here - worth running once to confirm the
   audit's `response_events` entry was the false positive it reads as.
2. The audit's §6 line for `response_events` should be corrected in P124c's report when convenient.
