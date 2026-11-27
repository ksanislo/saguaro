# REPORT - P125h: a task-score runner for QFN quant quality

## What changed

The README's Quality section has no Qwen3.8-Flash-Next numbers, and no full-precision reference (~250 GB) can be
run here, so quality per quant (UD-Q4_K_XL, Q8_0, GSQ-RCO IQ3_S, Coder IQ1_M) is now a task score on a fixed
question set. This is the script only - the manager runs it on cards later.

- **tools/quality_score.py** (new) - `quality_score.py <base-url> [--key-file F] [--set gpqa-diamond|agentic]
  [--set-file F] [--limit N] [--out results.jsonl] [--max-tokens N] [--timeout S]`. One normal chat request per
  question: thinking on (`chat_template_kwargs.enable_thinking`), temperature 0, a `--max-tokens` cap (8192) and a
  per-question `--timeout` (600 s). It extracts the answer the way the set defines it - the multiple-choice letter
  for GPQA-style sets (an answer-word mention, the last one winning; a bracketed letter; a lone letter on its own
  line; else exactly one choice named by its own text), or the tool call for agentic sets (the first call; its name
  and every expected argument must match, 5 and "5" the same value, an argument the model added tolerated; a call
  that names nothing scores wrong) - appends one JSON line per question to `--out` (`id, answer, correct, tokens,
  seconds`, plus `finish` when the cap or a cancel cut the reply and `error` when the request failed), resumes by
  skipping ids already in `--out`, and prints `accuracy: k/n = p %  95 % CI [lo, hi] %` (Wilson score interval).
- **Question sets** - the runner reads them only from a local `--set-file` (JSON array or JSONL: gpqa records
  `{id?, question, choices[4], answer}` where the answer may be the letter or the choice text; agentic records
  `{id, question, tools, expect{name, arguments}}`). NO copyrighted question text ships. `--set agentic` with no
  file runs `BUILTIN_AGENTIC_SET`, five tool-call tasks written in the file (calc / file_stat / get_weather /
  json_get / search_docs), so the tool-call path is testable out of the box.
- **tools/test_quality_score.py** (new) - 14 tests: extraction, scoring, the interval, the loader, resume
  (incl. a dead port writing an error line rather than stopping the run), and two real end-to-end runs against
  `python3 -m serve.server --engine mock` on a free port. No GPU, no network beyond the local port.
- **README.md** - the Qwen3.8-Flash-Next bullet in the Quality section now says how its numbers are produced and
  points at the runner.

## Files

- `tools/quality_score.py` (new)
- `tools/test_quality_score.py` (new)
- `README.md` (Quality section, one sentence)
- `REPORT-p125h-quality-runner.md` (this file)

Commits: P125h step 1 (the runner), step 2 (the sets: `--set-file` + `BUILTIN_AGENTIC_SET`), step 3 (the tests),
then README + this report.

## Tests run and their results

    /opt/llm/saguaro/.venv/bin/python -m unittest tools.test_quality_score -v     # the only module the plan names
    Ran 14 tests in 4.793 s ... OK        (Python 3.13.5)

Run twice; stable. The end-to-end cases start a mock server (`--engine mock`, two `--script`s) on a free port and
kill it afterwards; the gpqa case scores a scripted "answer is C." (1/2 - the second question wants B), and the
agentic case scores the built-in set against a scripted GLM-form tool call: calc-sum correct, file-stat wrong on
the name. `serve/test_tools.py` confirmed first what the mock/server produce for a scripted call
(`finish_reason: tool_calls`, `arguments` as a JSON string) so the extractor matches reality.

## Open questions

- **The GPQA sets themselves.** We ship none. GPQA-Diamond is CC BY 4.0, so the manager can take their own copy;
  the record format and `--set-file` are in the script's docstring. Whether the 198-question set is the one to
  standardize on, and per-quant `--max-tokens` big enough for the thinking the small quants do (a reply cut at the
  cap carries `finish: length` and scores wrong unless the letter already landed) - the manager's call on cards.
- **Thinking and the extraction.** The think text arrives as `reasoning_content` and is never read by the scorer;
  if a quant only "answers" inside its think text and stays silent in the visible reply, it scores as no answer.
  That is deliberate (a user-visible answer is the task), but it is a choice.
- **Multiple-choice letters beyond A-D.** The extractor derives the letter range from the number of choices
  (GPQA has four; the loader supports any count up to H). A set whose choices are not letters-only, or whose
  question text itself contains bracketed letters, could confuse it - the `finish`/error line and the raw `answer`
  field in results.jsonl make any such run auditable by hand.
- **Error lines and resume.** Per the plan, an id already in `--out` is skipped - including a line that logged a
  request error (scored wrong). A manager wanting transient failures retried should delete that line before the
  re-run.
- **No `serve/qfn_engine.py` involvement:** this change lives entirely in `tools/` and README, so
  `tools/saguaro/extract_qfn_engine.py` was not needed.
