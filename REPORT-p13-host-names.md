# P13 report: hardware names instead of host names in public files

## What changed

`git grep -inE 'mortar' -- . ':!CHANGELOG.md'` found 10 mentions in 8 files. Every one now reads as the
hardware - "the 9070 XT box" (the one 16 GB RX 9070 XT), keeping the meaning and the dates; a few lines were
re-wrapped to keep the repo's line lengths. Docstrings/comments only: no code behavior, no renames, nothing else
touched. CHANGELOG.md untouched (and it turns out it holds no `mortar` mention either way).

| File | Where | Before -> after |
| --- | --- | --- |
| docs/INKLING2-MINI.md | measured-speeds paragraph | `` mortar `ab-runs/r7` `` -> "the 9070 XT box's `ab-runs/r7`" |
| serve/test_inkling2_frontend.py | test_a_tiny_max_tokens_still_answers docstring | "(mortar, 2026-11-22)" -> "(the 9070 XT box, 2026-11-22)" |
| serve/test_new_engines_ui.py | test_one_rx_9070_xt docstring | "mortar 2026-11-22" -> "the 9070 XT box 2026-11-22" |
| serve/test_server.py | test_a_gpu_out_of_memory... docstring | "(mortar, 2026-11-22)" -> "(the 9070 XT box, 2026-11-22)" |
| serve/test_server.py | comment in test_the_engines_err_reaches_the_request | "on mortar" -> "on the 9070 XT box" |
| tools/qfn_fit.py | v1.2.1-B minimum comment | "mortar Monitor" -> "the 9070 XT box's Monitor" |
| tools/qfn_fit.py | unsupported() docstring | "mortar runs Q8_0" -> "the 9070 XT box runs Q8_0" |
| tools/test_docs_new_engines.py | test_inkling_on_amd_is_said_alike_everywhere docstring | "(mortar, 2026-11-22)" -> "(the 9070 XT box, 2026-11-22)" |
| tools/test_setup_amd.py | Reserve class docstring | "mortar's RX 9070 XT beside an iGPU" -> "the RX 9070 XT beside an iGPU" |
| tools/test_setup_ik2.py | test_each_quant_its_own_config_and_layout docstring | "(mortar, 2026-11-22: ...)" -> "(the 9070 XT box, 2026-11-22: ...)" |

After the pass `git grep -inE 'mortar' -- . ':!CHANGELOG.md'` returns nothing.

## Files

- docs/INKLING2-MINI.md
- serve/test_inkling2_frontend.py
- serve/test_new_engines_ui.py
- serve/test_server.py
- tools/qfn_fit.py
- tools/test_docs_new_engines.py
- tools/test_setup_amd.py
- tools/test_setup_ik2.py

One commit for the pass (d385db116), one for this report.

## Tests run (only the six the plan names, one at a time, /opt/llm/saguaro/.venv/bin/python)

| Module | Result |
| --- | --- |
| tools.test_docs_new_engines | 7 tests, OK |
| tools.test_qfn_fit | 47 tests, OK |
| tools.test_setup_amd | 35 tests, OK |
| tools.test_setup_ik2 | 54 tests, OK |
| serve.test_inkling2_frontend | 22 tests, OK |
| serve.test_new_engines_ui | 38 tests, OK |

Also `python -m py_compile` on every edited .py file: clean.

## Notes / open questions

- serve/test_server.py also carried 2 mentions and was edited (docstring and comment only), but the plan's
  run list does not name serve.test_server, so it was not run per the "Run ONLY" rule; py_compile covers it
  and the edits are non-code, so the risk is nil.
- tools/qfn_fit.py mentions were comments/docstrings: nothing there is extracted into serve/qfn_engine.py
  (extract_qfn_engine.py does not reference qfn_fit and serve/qfn_engine.py held no mention), so no
  re-extraction was needed.
- "mortar Monitor" (tools/qfn_fit.py:41) reads now as "the 9070 XT box's Monitor" - the Monitor itself (the
  tool behind the quoted "2.8 GB" line) is unnamed either way; if its real name matters, say the word.
- No other host names were searched for; the plan named only `mortar`.
