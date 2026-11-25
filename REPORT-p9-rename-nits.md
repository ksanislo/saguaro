# REPORT-p9-rename-nits - findings 1, 2, 3 and 5 of the rename tool's review (ref/findings.md), one commit each

Branch `saguaro/q1-p9-rename-nits`, four commits on top of 66f754635 (one per finding), touching only the rename
tools and their tests - no product code, and the tool was never run over the real tree in a writing mode (only
`--check`/`--paths`, which write nothing). Files changed over the whole branch:

    docs/UPSTREAM.md                      |  11 ++-
    tools/saguaro/rename_env.py           |  18 ++---
    tools/saguaro/rename_internal_maya.py |  10 +--
    tools/saguaro/rename_rules.py         |  62 +++++++++++++---
    tools/saguaro/rename_upstream.py      | 112 ++++++++++++++++------------
    tools/test_rename_upstream.py         | 133 +++++++++++++++++++++++++++++++++-

## 1. The exception/skip lists are shared now (73e1b803e)

- `rename_rules.py` keeps the ONE skip list; `skipped()` now also rejects build/vendored dir components at any
  depth (an `engines/qfn/third_party/...` or `build-today/...` path can no longer reach a family). The internal
  family's stop line moves to a named `INTERNAL_SKIP_PATHS = ("engines/qfn/", "tools/qfn/", "serve/qfn_engine.py")`
  plus `internal_skipped()`, and the EXC_TEXT masking moves out of `rename_upstream._mask` into shared
  `mask_exceptions()/unmask_exceptions()`; `exc_path()` names the EXC_PATHS check.
- `rename_env.py`: the five private `SKIP_*` copies (rename_env.py:42-46) are gone; `collect()` uses `T.skipped()`.
  The tool module `tools/test_rename_upstream.py` is skipped, so its fixture settings are no longer collisions of
  themselves in the report (nit 4 becomes inert by design, as the findings note).
- `rename_internal_maya.py`: private `SKIP_PREFIX`/`SKIP_NAME` (rename_internal_maya.py:31-32) gone; `files()` uses
  `T.internal_skipped()`, and the main loop wraps the rename in the shared exception mask, so `--check` no longer
  prints the tool's own test module, the triage record, or the exception-list names as strata residue.
- `rename_upstream.py`: uses the same helpers (behaviour of its own scan unchanged in this step).
- Tests: `SharedSkipLists` (one skip list, internal scope, mask round-trip) and two `RealTree` runs:
  `rename_internal_maya.py --check` (was rc=1 printing 6 files, incl. tools/test_rename_upstream.py and
  docs/triage/TRIAGE-upstream-prs-20261124.md; now silent, rc=0) and `rename_env.py --check` (green before too, but
  its COLLISIONS block listed the fixture names MAYA_BENCH_LOCK_OK / MAYA_SOMETHING_NEW / MAYA_DS4_SLOT_KEEP /
  STRATA_OLD_THING; now only the real STRATA_LOGPOS
  collision in tools/qfn/kl.py remains; "distinct old-style names seen" 289 -> 272).

## 2. `--paths` applies the skip filter (22f702f5e)

- `rename_upstream.py`: the `--paths` block becomes `paths_mode()` + `path_new(rel)`. The walk prunes the same skip
  dirs as `collect()` instead of `root.rglob("*")`, and `path_new()` refuses any `internal_skipped` path, so a
  vendored, generated, record or scratch file named `maya.py` / `maya.sh` / `START-MAYA.bat` / `MAYA_*.md` is never
  printed as a `git mv` suggestion.
- `rename_rules.py`: `NAME_PATH_PREFIXES` gains a scope: the `MAYA_*` doc rule counts only under `docs/`
  (rename_rules.py's old line 209 matched `MAYA_*` in any directory).
- Tests: `Paths.test_path_names_and_skips` (positives at root, tools/ and docs/, negatives in scratch/ref/
  third_party/build dirs, in the qfn tree, outside docs/) and a run of `--paths scratch/p9_paths` over a scratch
  fixture holding `maya.py`, `MAYA_NOT_A_DOC.md` and a `build-today/maya.sh`: prints nothing (the fixture is
  created and removed inside the test). `--paths` over the real tree: 0 suggestions, as before.

## 3. engines/qfn is covered like the union of the old scripts (30bc6f5a8)

- `rename_upstream.py`: `collect()` stops skipping QFN's tree wholesale; the settings pass therefore sweeps
  engines/qfn, tools/qfn and serve/qfn_engine.py (rename_env always did), and the internal, names and brand
  families stop at `internal_skipped()`. The docstring's "ALL of them" gets the qualifier: inside that tree only
  the settings and log-prefix passes run.
- `rename_rules.py`: `LOG_SRC_PATHS = ("src/", "tools/vision/", "engines/qfn/src/", "engines/qfn/tools/vision/")`
  - rename_log_prefixes.py's default paths (its DEFAULT_PATHS always included them), so a new `MAYA_FOO` setting
  in a qfn source and a new `strata part:` engine prefix are caught by the one command.
- What deliberately stays in engines/qfn: the sources' `strata` names / `namespace strata` / `include/strata`, the
  CMake project's own STRATA_ options and -D arguments, and its own program names (no names/brand pass there).
- `docs/UPSTREAM.md` section 2 now lists exactly what the command renames there and what stays, and says the three
  older scripts read the same table, skip list and exception list.
- Tests: `EnginesQfn.test_settings_and_prefixes_fire_in_qfn_sources` (a new `MAYA_QFN_NEW_THING` getenv in
  engines/qfn/src/core/n.cu -> SAGUARO_ with the prefix rule firing in the same file, changed families exactly
  ["logprefix", "settings"]; a `ds4:` prefix in engines/qfn/tools/vision -> "saguaro ds4:") and
  `test_the_qfn_tree_itself_stays` (core.h with `namespace strata`, qfn CMakeLists STRATA_ options, qfn README
  `maya.py`/`restart Maya`, tools/qfn/util.py, serve/qfn_engine.py `[maya]` tag - all byte-identical, changed {}).
- Real tree: `rename_upstream.py --check` now reads 1717 files (was 934) and still CHECK OK; its collisions line
  gained `STRATA_LOGPOS` (tools/qfn/kl.py) - visible through rename_env.py for a while, now in the one command too;
  it is "left, review" by design, and the exception report is unchanged (15 hits).

## 5. `exception_hits` reports the lines of the file as it was (28fbc27e1)

- `rename_upstream.py`: `exception_hits(files)` reads each file's collected text (`files[rel][2]`) instead of the
  post-plan `texts` the caller passed; the kept-hit line numbers can no longer drift from the file on disk when a
  red `--check` changes other parts of those files.
- Test: `SharedSkipLists.test_exception_hits_line_up_with_the_file_on_disk` - the on-disk text reports `:2:`, a
  text that gained a leading line would report `:3:`. Real tree: the 15 reported hits and their line numbers are
  byte-identical to before the change (green case).

## Tests run (only the two named modules, one at a time)

    /opt/llm/saguaro/.venv/bin/python -m unittest tools.test_rename_upstream
        baseline (66f754635): 25 tests OK
        after step 1: 30 tests OK    after step 2: 32 tests OK
        after step 3: 34 tests OK    after step 5: 35 tests OK (85 s, the module re-runs --check over the real tree)
    /opt/llm/saguaro/.venv/bin/python -m unittest tools.test_docs_new_engines
        7 tests OK, run once after every commit (untouched by this branch; it pins the docs the rename layer
        must not rewrite)

Manual gates, read-only only: `rename_upstream.py --check / --paths`, `rename_env.py --check`,
`rename_internal_maya.py --check`, `rename_log_prefixes.py --check` (rc=0, as before).

## Open questions

1. The names/brand families deliberately do NOT sweep engines/qfn / tools/qfn / serve/qfn_engine.py (the old
   scripts had no such families; an upstream intake checkout has no qfn tree at all, and qfn's own texts are
   Strata's). The plan's "together with the others" is read as the union of the settings and prefix passes. If
   qfn's docs should follow the program/launcher renames someday, the `internal_skipped` gate in `apply_names`
   is the single line to revisit.
2. `STRATA_LOGPOS` in tools/qfn/kl.py is now in the one command's collisions line. The findings class it with
   "left, review"; if kl.py's old-name setting should become SAGUARO_LOGPOS that is a product change outside
   this plan's run list.
3. Nits 6 (--diff still has no test; --paths now does) and 7 (placeholder restore by plain str.replace, NIT-7)
   were outside this plan's findings and are untouched; nit 4 became inert by design through finding 1.
4. `rename_env.py --check` prints 272 instead of 289 kept/distinct names now (the skipped test module held 17);
   expected, recorded here in case a later doc cites the old count.
