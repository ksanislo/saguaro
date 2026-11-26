# P124l FIX-REPORT - REVIEW-RESULT.md (of 856af24..ed7746621) -> what each finding got

One commit per MANAGER item (then the report itself).  Only the seven planned files changed across the branch;
`serve/qfn_engine.py` / `engines/qfn/serve/server.py` stay untouched (the fixes are bench-side only, so no
extractor run was due).  Tests: only the named modules, one at a time, all after the last code edit:
`tools.test_bench` **48 OK**, `tools.test_maya_hip` **64 OK**, `serve.test_priority` **56 OK**.

## MANAGER 1 - MEDIUM finding 1: a failed batch slot (BRE) hung the round - FIXED (e055d4ffc)

`tools/maya_bench.py`, `parallel_phase`, slots branch: `BRE <slot> <why>` is now terminal for its slot - the
slot leaves `waiting` and its reason goes into a `failed` map, so the round finishes as soon as every other
slot is DONE and never blocks in `readline` on a BDONE that will not come (the review's exact hang: the
ignore-list swallowed the line and the engine, alive, said nothing more for that slot).

- `parallel_phase` now returns each request's own figure in request order: a float, or the string
  `failed: <why>` for a BRE slot (the BRE line's own words - "no RAM is left" reaches the table).  A failed
  slot's produced tokens are unknown, so it counts 0 toward the round total; the answered slots still do.
- `run_parallel` (the console line) and `parallel_section` (saguaro-bench.txt) print a string entry as it is,
  so the failure is named in the table of every round it touched: e.g.
  `  2 at once: total 21.0 tok/s (20.5, failed: no RAM is left)`.
- `BRE` joined the ignore-list, so a malformed one (no numeric slot) is skipped instead of logged as an
  answer.  `BRP` / `BRA` stay log-and-continue, exactly as the review said they should.
- Test (as the manager asked): a scripted QFN mock engine, `MOCK_QFN_BRE`, answers `BRE 1 no RAM is left`
  for one slot after its admission's control DONE; `test_several_at_once_a_slot_dies_with_bre` runs the whole
  1 / 2 / 4 section through it - the rounds finish (old code hangs the test), every answer is still there in
  sent order, the failed one carries the engine's words in both the 2-at-once and 4-at-once lines and in
  `parallel_section`'s text, and the N = 1 plain-GEN round is untouched.  `parallel_phase`'s docstring now
  states BRE and the total's meaning.

## MANAGER 2 - `tools.test_maya_hip` `test_cli_backend_selects_hip_config_for_bench` FAILED - FIXED (50b3cc92a)

The branch made `main()` state the several-at-once default - `bench(..., parallel=a.bench_parallel == "on")`
- and the test's `assert_called_once_with` still listed the five-argument call.  The fix updates the expected
call to the real one (`..., 2, 10.0, parallel=True)`): still `assert_called_once_with`, every argument pinned,
the test still exactly-one-call and HIP-config-selected - not weakened, and the test's point (the CLI never
benchmarks the CUDA config as HIP) is untouched.  `tools.test_maya_hip`: 64 OK.

## NIT 2 - `--bench-parallel off` not byte-identical (saguaro-bench.txt gained a skip line) - FIXED (6eeda7adc)

- `saguaro.py bench()`: `res["parallel"]` is set only for an on run (`if not smoke and parallel:`); an off run
  carries no key, so `bench_text` prints no section - the file's bytes are then the ones from before the
  feature, like `--compare`'s.  `bench_text`'s gate is `res.get("parallel")`.
- `MB.parallel_section(None)` returns `[]` (renders nothing) instead of the skip line; the reason is now said
  on the run's console instead (`say "  several at once: skipped (--bench-parallel off)"`, non-smoke non-compare
  off runs only - `--compare` says nothing it did not say before).
- `tools/test_bench.py`: `test_the_table_text` expects `[]` for a skipped section; `test_bench_text_holds_the_section`
  now asserts an off render holds no "several at once" line at all and keeps the old wall-time line.

## NIT 3 - `Session.multi` rejected a non-numeric `multi=` - FIXED (6eeda7adc)

`tools/maya_bench.py Session.__init__`: `multi=<n>` with a number parses as the server does (serve/server.py's
own pattern); `multi=` with anything else still says "several at once", so it becomes one place on rid-tagged
lines - the old `any(...)` behavior, plain GEN lines with no `rid=` never go to such an engine.  Checked with
`scratch/multi_check.py` (scratch, untracked): multi=2 -> 2, multi=many -> 1, multi=0 -> 0, no multi= -> 0.

## NIT 4 - `plan_parallel` "openings after every window" was loose wording - FIXED (6eeda7adc)

Docstring only, as the review said: it now says the slices open after every window the standard run *opened*
and that an 8K one runs far past its opening - what keeps "nothing reused" true is plan_prompts' own sense:
no two windows share an opening, so no prefix hit.  No code or plan changed; the exact-fit corner the review
noted needs a doc no `--bench` run has.

## NIT 5 - README lines over 120 columns - FIXED (6eeda7adc)

The method paragraph's branch-touched lines re-wrapped to <= 120 (measured with awk: 103 / 105 / 107 / 101 /
18; the pre-existing 162-column lead line was left as the file writes them).  Every added line across all
three fix commits is <= 120.

## NIT 6 - plan-vs-test reading of "mock server" - no change (recorded, as the review itself concluded)

The scripted line-protocol engines stay the right call (serve.server `--engine mock` is an HTTP server the
line-protocol bench cannot talk to); it was disclosed as REPORT open question 1 and the review accepts it.
The BRE mock added for MANAGER 1 is the same idea, one protocol line deeper.

## Also in the nits commit (kept honest, no behaviour)

`REPORT-p124l-bench-parallel.md`: its lines that the fixes make untrue (the skip-line-in-file claim, the
test counts 47 / 8 -> 48 / 9, `parallel_phase`'s protocol list gains BRE, the not-run bullet now says the
review pass ran `tools.test_maya_hip` and `serve.test_priority`).
