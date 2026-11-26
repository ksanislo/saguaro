# P124n review - mem_vs_ctx (diff ba9854dd..HEAD, 4 new files, 854 insertions, 0 deletions in code)

FIX-FIRST

## Findings by severity

1. **FIX: a machine without /proc silently PASSes on zeros - and the report claims macOS.** Every reader
   swallows the missing /proc into zeros: `tools/mem_vs_ctx.py:70-75` (status -> 0.0), `:90-94`
   (pids -> []), `:138-156` (port owner -> None). With no /proc the tree is just the server and every figure
   reads 0.0; with no nvidia-smi/rocm-smi (an Apple box) the GPU row is `-`; `verdict` then compares zeros
   against zeros and prints PASS (`:382-405`). There is no platform guard anywhere in the tool, and
   `tools/test_mem_vs_ctx.py:250` skips only `nt` - on macOS the suite itself would fail at `:266`
   (RSS > 0), so the platform is not supported, merely un-honest about it - while
   `REPORT-p124n-mem-vs-ctx.md:34` claims "Linux/macOS (/proc)". A release check that PASSes when it measured
   nothing is exactly the failure this task exists to catch (QFN's whole-max pin). Fix: fail loudly when
   `/proc/self/status` cannot be read (before starting anything), and make the report/skip honest.
2. **stop_server signals whoever owns the port, not just our tree.** `tools/mem_vs_ctx.py:317-336`: the
   SIGTERM target is `port_owner(port)` with no check it is `proc.pid` or under `process_tree(proc.pid)`,
   and the SIGKILL loop kills that owner's whole tree. If the run's server died first and some unrelated
   process took the port (the pre-check at `:419` is TOCTOU), the tool TERMs/KILLs someone else's daemon and
   its children. Restrict the target to our own tree (the `pid is None` fallback at `:318-319` already
   covers the pre-bind server).
3. **`--ctx`-style configs get a second context flag, not a rewrite.** `set_context`
   (`tools/mem_vs_ctx.py:221-228`) only knows `--max-context`; `serve/models.py:540-542` (and `:407-409`)
   know that ik2 configs may name their context `--ctx`. For one of those the copy ends with `--ctx <old>`
   and an appended `--max-context <new>` - which value wins is the engine's mood: two runs of the *same*
   context (vacuous PASS) or a confused engine. Docs say the checks always run on setup-written configs
   (docs/RELEASE-CHECKS.md:17-18), so the exposure is small, but the mirror of models.py's `ctx_flag` is one
   line.
4. **A failed stop is reported as a clean one, and the next FAIL is misattributed.**
   `tools/mem_vs_ctx.py:366` ignores `stop_server`'s bool and `:433` prints "port N free again" regardless;
   if the port never fell free, the next context's server dies at the bind and the FAIL names
   "the server ended ... before /health was ok" (`:288`) instead of the stale port.
5. **Every real run repoints the repo's "last good config" at the scratch copy.** The server calls
   `write_last_good` when started with `--config` (`serve/server.py:10058-10059`, body `:8851-8856`, note in
   the repo's state folder `:8833-8836`), so the next failed start on that machine offers
   `scratch/mem-vs-ctx/<config>/<ctx>/config.json` as the config to start again. Harmless for the check,
   surprising for the person; worth one sentence in the docs or a way to suppress it.
6. **Nit - the tcp6 test never reads tcp6.** `tools/test_mem_vs_ctx.py:107` writes `tmp/tcp6`, but
   `listening_inodes` reads `net/tcp` and `net/tcp6` (`tools/mem_vs_ctx.py:121-123`); the fake root file is
   inert, so despite its name the test exercises only the tcp branch (same code, but the name promises more).
7. **Nit - a report claim the tests do not cover.** `REPORT-p124n-mem-vs-ctx.md:44-46` says "the boundary
   equal fails - 'less than'" is tested; no test sits exactly at the limit (`verdict` rule
   `tools/mem_vs_ctx.py:399`; the closest cases are 300-vs-256 and 200-vs-256, `:212-218`). The rule itself
   matches the plan's "less than" as written.
8. **Nit - docs.** Section 4 (docs/RELEASE-CHECKS.md:174-193) never says quiet machine, though the GPU row is
   total used memory on the card - a neighbour between the two loads can flip the verdict
   (`tools/mem_vs_ctx.py:183-202`); every bench section spells quiet out (`:15`, `:133`). And "All three sit
   on ./saguaro.sh --bench" (`:15`) now sits under a four-tool table - "The first three" keeps it true.

## Checked and clean

- **Scope / generated file**: the diff is exactly the 4 promised files; `serve/`, `engines/` and the
  generated `serve/qfn_engine.py` are untouched (manager's "no engine changes" confirmed), so the
  extract-generator rule is not engaged.
- **Feature OFF**: purely additive - nothing imports the new tool; every existing path is byte-identical.
- **Locks / threads / stdout**: the tool is single-threaded with no locks; the server's output goes to
  `run_dir/server.log` (`tools/mem_vs_ctx.py:275-277`), never the tool's stdout, so nothing interleaves.
- **Server-side assumptions verified**: `/health` 200 `ok` vs 503 `loading`/`failed`
  (`serve/server.py:7959-7960`, `:8984-8987`) match `wait_health` (`:280-310`, HTTPError before OSError -
  right order); SIGTERM is the server's graceful stop that ends the engine
  (`serve/server.py:9235-9257`, `:10085-10127`), and the listening socket lives until the process exits, so
  "the port fell free" really means the server is gone before the tree's leftovers are KILLed.
  `set_context` is the byte-shape of the server's own (`serve/server.py:1608-1615`); the copy drops
  `api_key` and `SAGUARO_API_KEY`, pins loopback and idle-unload-off, both session-dir shapes
  (`tools/mem_vs_ctx.py:246-259` vs `serve/server.py:9786-9787`, `:9968`); ports pre-checked, exit codes
  0/1/2, strict-less-than rule of 2% of the bigger figure - all as planned.
- **Tests**: `tools.test_mem_vs_ctx` - the only module the plan names - run on the venv python:
  **22 tests, OK**, including the real mock start/stop cycle; parsing and the verdict run on fixed inputs,
  so the module genuinely exercises the change. Untested (report admits): `wait_health`'s failed-page raise
  and the stop fallbacks.
- **Paths / secrets**: repo-relative `scratch/mem-vs-ctx/` copies only (tests use tempfile); `api_key`
  stripped from every copy; no secret appears in the diff; the worktree is clean after the test run.

Re-running finding 1's one-line guard (abort without /proc) and finding 2's tree check turns this into a
merge; nothing else in the diff blocks.
