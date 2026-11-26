# P124n fix report - the review round (REVIEW-RESULT.md -> what changed, finding by finding)

All eight findings are addressed; every FIX-FIRST one is a code fix. One commit per finding (nine plus this
report); `python -m unittest tools.test_mem_vs_ctx` (the only module the plan names, venv python 3.13.5) re-run
after every fix that touched code: 24, 25, 26, 26, 26, 26, 27, 27 tests, all OK. No engine and no `serve/`
file was touched, so the `serve/qfn_engine.py` extract rule was not engaged. The tool was again not run on a
real model (the manager does).

## 1. FIX: a machine without /proc silently PASSed on zeros - fixed

- `tools/mem_vs_ctx.py`: new `has_proc()` (reads `/proc/self/status`); `main` calls it before anything else and
  prints "this check reads the process tree through /proc (Linux): ... every figure would be 0.0 and the
  verdict a PASS on nothing - not run" to stderr and exits 2. Exit 2 now means "nothing could be measured"
  (used port, or no /proc) - the docstring says so.
- `tools/test_mem_vs_ctx.py`: the mock-cycle skip keys on the real thing now - `skipUnless(/proc/self/status
  exists)` instead of `skipIf(os.name == "nt")`, so macOS skips (where the old suite failed at RSS > 0) instead
  of lying. Two tests: `has_proc` true here / false in a bare fake root, and `main` refusing (exit 2) with
  `has_proc` patched off - before any config read or server start.
- `REPORT-p124n-mem-vs-ctx.md`: the "Linux/macOS" claims (intro and the Windows open question) corrected to
  Linux-only with the guard described.

Commit `587615997`.

## 2. FIX: stop_server signalled whoever owned the port - fixed

- `tools/mem_vs_ctx.py`: the tree is now always `process_tree(proc.pid)` (plus `proc.pid` itself); the port's
  owner is the SIGTERM target **only when it is in that tree**. An owner outside it - our server died first
  and an unrelated daemon took the port (the pre-check is TOCTOU) - gets no signal at all; the target falls
  back to our own server while it is alive (the pre-bind case, as before), and the SIGKILL sweep only walks
  our tree. The function returns False there, which finding 4 now surfaces.
- Test: `StopServerTests` - a fake /proc where stranger 500 holds the listen inode while 77 (server) / 88
  (engine) are ours; with `os.kill` patched it asserts 500 is never signalled, 77 got the TERM, 88 the KILL,
  and the return is False.

Commit `d58c659e9`.

## 3. FIX: --ctx configs got a second context flag - fixed

- `tools/mem_vs_ctx.py`: `set_context` mirrors `serve/models.py`'s `ctx_flag` - `--ctx` is rewritten in place
  when the args name it and `--max-context` is absent; `--max-context` still wins when both sit there, and it
  is still what gets appended when neither flag is present. No copy carries two context flags whose winner is
  the engine's mood.
- Test: `--ctx 131072` becomes `--ctx <new>` with no append; both-flags rewrites `--max-context` only.

Commit `2ab17fc04`.

## 4. FIX: a failed stop read as a clean one - fixed

- `tools/mem_vs_ctx.py`: `measure_context` keeps `stop_server`'s bool and raises `RuntimeError` naming the
  stale port when it never fell free. `main` catches it, so the run FAILs with "port N is still held after
  the stop - free it before the next context starts"; "port N free again" now prints only when the port
  really is free, and the next context can no longer die at the bind and get blamed on "the server ended
  before /health was ok". No separate test (the stop fallbacks remain admitted-untested in the report; the
  finding-2 test drives the same `freed=False` path from the other side).

Commit `6a85ae489`.

## 5. FIX: the run repointed the repo's "last good config" - documented (the review's doc option)

- `docs/RELEASE-CHECKS.md` section 4: one sentence - every load records the config that worked last, so after
  a run a later failed start on that machine offers `scratch/mem-vs-ctx/<config>/...`, and starting your own
  config overwrites the note.
- Why not suppressed: the only writer is `serve/server.py:10059` (`write_last_good`), which has no env knob;
  adding one edits `serve/server.py` - outside this task's four files and its untouched-`serve/` property that
  the review itself counted clean. The review offered "one sentence in the docs or a way to suppress it"; the
  sentence is the one that keeps the branch's scope promise.

Commit `9c3f1cad3`.

## 6. Nit: the tcp6 test never read tcp6 - fixed

- `tools/test_mem_vs_ctx.py`: `fake_proc` gained `net_tcp6=` and writes `net/tcp6` under the fake root;
  `test_port_owner_by_inode_tcp_and_tcp6` now puts its v6 listener row (own port, own inode, own `fd` link) in
  `net/tcp6` and asserts `port_owner` finds it - the file the reader actually opens - while the tcp row stays.
  The stray `tmp/tcp6` beside the root is gone.

Commit `2e292c4da`.

## 7. Nit: "the boundary equal fails" was untested - fixed

- `tools/test_mem_vs_ctx.py`: `test_a_difference_exactly_at_the_limit_fails` - 10000 vs 10256 is exactly the
  256 MiB floor (2% is 205.12) and 39690 vs 40500 is exactly the 2% (810 of 40500); both FAIL naming the
  figure, and 10000 vs 10255 (one under) passes. The report's rule wording itself needed no change: it was
  already "less than", as planned.

Commit `4c3477794`.

## 8. Nit: docs - fixed

- `docs/RELEASE-CHECKS.md`: section 4 now says quiet machine (the GPU row is the card's *total* used memory,
  so a neighbour job between the two loads can flip the verdict) like the bench sections do; and "All three
  sit on `./saguaro.sh --bench`" became "The first three" under the now-four-tool table (line re-wrapped).

Commit `27f3266e0`.

## Also (not a finding)

- `REPORT-p124n-mem-vs-ctx.md`: its two "22 tests" lines updated to 27 with a pointer here, so the report
  matches the suite the fixes grew.
