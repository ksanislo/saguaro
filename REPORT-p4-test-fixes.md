# P4 report: the three failing test modules (PLAN-p4-test-fixes.md)

Three commits, tests and test helpers only. No product behaviour changed at all: the plan's getattr default turned out
unnecessary (step 1), engines/ untouched, serve/qfn_engine.py untouched, no network.

## 1. tools.test_maya_models - 'types.SimpleNamespace' object has no attribute 'yes'

The attribute the code reads is `a.yes`: `offer_download` (saguaro.py:3283) ends its download question with
`ask(..., "n", a.yes)`. The failing tests (`Exl3Download.test_the_download_names_the_branch_and_asks` and its two
inherited copies, tools/test_maya_models.py:431) faked args as `SimpleNamespace(download_model=False)` only.

**Decision - fixed the test, not the code.** The plan's condition for a getattr default is "the real argparse always
sets it AND other callers exist that do not". The first half holds (`--yes` is `action="store_true"`, saguaro.py:9217,
so argparse always sets it); the second does not: every product caller of `offer_download` passes the full namespace
from `main()` (all ~20 call sites in saguaro.py), so no caller can be missing `yes`. Other tools tests already carry
`yes` in their fake args (e.g. test_setup_ds4.py:24). Fix: the fake args become
`SimpleNamespace(download_model=False, yes=False)` - `False` because the test asserts `ask.assert_called_once()`:
with `yes=True` setup.py's `ask` returns the default without being asked.

File: tools/test_maya_models.py (one line).

## 2. tools.test_refresh_ds4_profile - ModuleNotFoundError 'refresh_ds4_profile'

The tool was **not renamed and is in our tree**: `git log -S refresh_ds4_profile` shows a single commit, 286adfe47
("tools/refresh_ds4_profile.py - refresh DeepSeek startup expert profiles from recent routing counts (by @LostVector)
(#399)"), which added tools/refresh_ds4_profile.py, tools/test_refresh_ds4_profile.py and docs/DS4_PROFILE_REFRESH.md
together; the tool still sits at tools/refresh_ds4_profile.py with MAX_COUNT/read_counts/refresh as the test imports
them. The stale-test branch of the plan does not apply.

The real break: the test does a bare `from refresh_ds4_profile import ...` with no sys.path setup, which only works
under `unittest discover -s tools` (start dir joins sys.path). Run by its module name (`python -m unittest
tools.test_refresh_ds4_profile`) - how the plan's runner names it - nothing puts tools/ on the path, hence the
ModuleNotFoundError. Fix, matching the sibling convention (test_make_profile.py:15, test_golden.py:13):
`sys.path.insert(0, str(Path(__file__).resolve().parent))` before the import.

File: tools/test_refresh_ds4_profile.py (one line added).

## 3. test_metal_serve - NO TESTS RAN

It lives at tools/test_metal_serve.py (not in serve/). It is a **script, not test cases**: it drives the built
maya-metal binary `--serve` on the synthetic fixture and takes `MAYA_METAL GGUF` as its own argv (nothing in CMake or
CI imports it; the CTest `metal_serve_test` in macos.yml is a different, C++ test). Imported by a test runner it
defined no TestCase, so the module looked empty.

Fix: one TestCase, `MetalServeScript`, class-skipped off macOS (`skipUnless(sys.platform == "darwin", ...)`) whose
single method raises `unittest.SkipTest("a script: python tools/test_metal_serve.py MAYA_METAL GGUF")` - so both the
dotted run and `discover -s tools` report `OK (skipped=1)` instead of NO TESTS RAN, on any host. A first attempt
raised SkipTest at module level; Python 3.13 turns that into a hard error for the dotted form (rc=1), so the class
form is used. Nothing about Metal is enabled; running the script directly is unchanged.

File: tools/test_metal_serve.py (9 lines added).

## Tests run (each module on its own, /opt/llm/saguaro/.venv/bin/python, Python 3.13.5)

- `python -m unittest tools.test_maya_models` - 55 tests, OK (was 3 errors before the fix)
- `python -m unittest tools.test_refresh_ds4_profile` - 5 tests, OK (was an import error)
- `python -m unittest tools.test_metal_serve` - 1 test, OK (skipped=1) (was NO TESTS RAN)
- `python -m unittest serve.test_model_api` - 28 tests, OK (untouched, the plan's regression check)
- also re-ran the two fixed modules' CI forms: `python -m unittest discover -s tools -p "test_refresh_ds4_profile.py"`
  (5 OK) and `-p "test_metal_serve.py"` (1, OK skipped=1); `tools/test_maya_models.py` ran directly (55 OK)

## Commits

- test_maya_models: offer_download reads a.yes ... the test's fake args now carry yes=False (tools/test_maya_models.py)
- test_refresh_ds4_profile: put the tools folder on sys.path ... (tools/test_refresh_ds4_profile.py)
- test_metal_serve: ... add one always-skipped TestCase ... (tools/test_metal_serve.py)

## Open questions

- The plan's harness evidently runs tools modules by dotted name; CI's Linux job uses `discover -s tools`, under which
  test_refresh_ds4_profile imported fine. Both forms pass now, but it is worth knowing which the integration gate
  uses if similar breakage should be caught earlier.
- tools/test_metal_serve.py is now always skipped under any runner, including macOS: its real check runs only when the
  script is invoked directly with the engine and the fixture (as the plan asked, nothing Metal was enabled). If a macOS
  job should ever run it as a test, someone must decide where MAYA_METAL and the fixture GGUF come from.
- tools/test_metal_serve.py is referenced by nothing (no CMake/CI entry); only this plan names it. Left in place.
