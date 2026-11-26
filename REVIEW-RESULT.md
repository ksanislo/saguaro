# P124n review, round 2 - mem_vs_ctx (diff ba9854dd..HEAD: new tools/mem_vs_ctx.py + tools/test_mem_vs_ctx.py,
docs/RELEASE-CHECKS.md row/section 4/per-machine bullet, the two reports and the round-1 review committed)

MERGE-WITH-NITS

The round-1 FIX-FIRST findings are all genuinely fixed and covered; what remains is four low-severity nits,
none of which blocks a merge.

## The eight round-1 findings - re-verified at HEAD

1. **FIX (was 1): no-/proc boxes PASSed on zeros - fixed.** `has_proc()` (`tools/mem_vs_ctx.py:81-88`), `main`
   checks it before reading anything and exits 2 (`:440-443`); the mock test skips on a real `/proc` absence
   (`tools/test_mem_vs_ctx.py:309`) and `main`'s refusal is tested with `has_proc` patched off
   (`tools/test_mem_vs_ctx.py:304-306`, before any config read - the config named does not exist).
2. **FIX (was 2): stop_server killed strangers - fixed.** The port's owner is the TERM target only inside
   `process_tree(proc.pid)` (`tools/mem_vs_ctx.py:335-342`) and the SIGKILL sweep walks only that tree
   (`:351-356`); the stranger test patches `os.kill` on a fake `/proc` and asserts 500 is never signalled
   while 77 gets the TERM and 88 the KILL (`tools/test_mem_vs_ctx.py:276-295`).
3. **FIX (was 3): `--ctx` configs got two context flags - fixed.** `set_context` mirrors
   `serve/models.py:541`'s `ctx_flag` (`tools/mem_vs_ctx.py:234-244`); both shapes tested
   (`tools/test_mem_vs_ctx.py:196-202`). Residual corner in the other direction: nit 2 below.
4. **FIX (was 4): a failed stop printed a free port - fixed.** `measure_context` keeps the bool and raises on
   a stale port (`tools/mem_vs_ctx.py:388-391`), and "port N free again" prints only after a freed stop
   (`:462`). No direct test of the raise: nit 4 below.
5. **FIX (was 5): the run repointed last-good - documented.** One sentence in section 4
   (`docs/RELEASE-CHECKS.md:194-196`); the review offered exactly this alternative, and the claim holds -
   `write_last_good` runs only under `--config` (`serve/server.py:10058-10059`), no env knob exists.
6. **Nit (was 6): the tcp6 test never read tcp6 - fixed.** `fake_proc` writes `net/tcp6` where the reader
   looks and the v6 row with its own inode/fd is asserted found (`tools/test_mem_vs_ctx.py:41-52,101-116`).
7. **Nit (was 7): the equal boundary untested - fixed.** 10000-vs-10256 (exactly the 256 MiB floor) and
   39690-vs-40500 (exactly the 2%) both FAIL, one under passes (`tools/test_mem_vs_ctx.py:235-244`).
8. **Nit (was 8): docs - fixed.** Quiet-machine line in section 4 (`docs/RELEASE-CHECKS.md:193-195`, the GPU
   row being the card's *total* used memory), and "The first three" under the now-four-tool table (`:15`).

## New findings (all nits)

1. **`stop_server`'s `proc=None` default reopens the stranger hole.** `tools/mem_vs_ctx.py:329-335`: when
   `proc` is None the tree restriction (`:339-342`) never runs and the bare `port_owner` of `:335` is TERMed
   whatever it is - the round-1 finding's behaviour survives for any future caller that omits `proc`. Today
   no caller does (`measure_context` `:388` always passes it, as does the test), so latent. Making `proc`
   required (or refusing) closes it.
2. **`set_context` rewrites `--ctx` for every family, not just ik2.** `tools/mem_vs_ctx.py:239` picks `--ctx`
   whenever args name it and `--max-context` is absent; `serve/models.py:541` does that only for
   `fam == "ik2"`. A non-ik2 config in that shape would get only the flag its engine ignores rewritten and
   no `--max-context` added - both runs at the engine's default, a vacuous PASS. No setup-written config is
   like that (ik2 owns `--ctx`; the checks run on setup's configs, `docs/RELEASE-CHECKS.md:15-17`), and a
   QFN engine parses both names (`engines/qfn/src/qwen35/q36_engine.cu:3743`), so the exposure is theoretical.
3. **The scratch copies go to a CWD-relative `./scratch`.** `tools/mem_vs_ctx.py:453` builds
   `Path("scratch")/...` from the invoking shell's directory - run the tool from anywhere but the repo root
   and the copies (config, logs) land outside the repo's `scratch/`, while the server child is started with
   `cwd=ROOT` (`:285-293`). Repo tools anchor it otherwise, e.g. `ROOT / "scratch"`
   (`tools/test_rename_upstream.py:379`); `ROOT / "scratch" / ...` here makes the docstring's promise hold
   from any directory.
4. **The fix-4 raise is untested.** `tools/mem_vs_ctx.py:389-391` (port still held after the stop) has no
   direct test - `FIX-REPORT.md:53` admits it; the stranger test exercises `stop_server`'s False return but
   not `measure_context` turning it into the FAIL. The path is two lines; noted, not blocking.

## Checked and clean

- **Scope / generated file**: the diff touches only the promised files; `serve/`, `engines/` and the generated
  `serve/qfn_engine.py` are untouched, so the extract-generator rule is not engaged and the manager's "no
  engine changes" holds. Commits are one per plan part plus one per round-1 finding.
- **Feature OFF**: purely additive - nothing imports `mem_vs_ctx` but its own test
  (`tools/test_mem_vs_ctx.py:25`); every existing path is byte-identical (docs aside).
- **Locks / threads / stdout**: the tool is single-threaded with no locks at all; the server's output goes to
  `run_dir/server.log` (`tools/mem_vs_ctx.py:291-293`), never the tool's stdout, and the tool's own prints
  flush - nothing interleaves.
- **Manager's line, verified**: the figures are real kernel/GPU readings (`VmLck`/`VmPin`/`VmRSS` from
  `/proc/<pid>/status` `:57-78`; nvidia-smi else rocm-smi over the config's cards `:189-215`); both runs are
  cleaned up (TERM our owner, KILL the tree's leftovers, wait for the port, raise when it stays held
  `:329-362,388-391`); the rule is as written - strict "less than max(2%, 256 MiB)", 2% of the bigger figure,
  FAIL naming the biggest difference (`:417-430` vs PLAN-p124n-mem-vs-ctx.md:12-13).
- **Server assumptions verified against the current source**: `/health` 200 `ok` (`serve/server.py:7959-7960`)
  and 503 loading/failed with the error (`:8984-8987`, `:8825-8829`) match `wait_health`, whose HTTPError
  branch precedes OSError (`tools/mem_vs_ctx.py:315,323`) and which also raises on a server that died before
  binding (`:303-304`); no key reaches the run - config copy drops `api_key` (`:262`), `--api-key ""` beats
  the env default (`serve/server.py:9749,9787`), `SAGUARO_API_KEY` is removed from the child's env
  (`tools/mem_vs_ctx.py:290`), and `/health` answers without one (`serve/server.py:7945,8984`); the engine is
  a Popen child, so the tree covers server and engine (`serve/server.py:1480,2558`) with the engine log at
  the config's `log` (`:9415`); `idle_unload_s` (`:9968`) and loopback `host` (`:9786`) are honoured; both
  session-dir shapes are real (`SAGUARO_DS4_SLOT_DIR` `engines/qfn/serve/server.py:4224`, `--session-dir`
  `serve/server.py:7645`) and unset slot dirs default under the relocated log; SIGTERM is the graceful stop
  that ends the engine (`serve/server.py:9236-9257,10085-10127`).
- **Tests**: `python -m unittest tools.test_mem_vs_ctx` (the only module the plan names, venv python) on this
  box: **27 tests, OK**, including the real mock start/stop cycle (`/health` ok, the port's owner is the
  server, RSS > 0, the port freed) and the stranger test with `os.kill` patched - the module genuinely
  exercises the change.
- **Paths / secrets**: repo-relative only; no secret in the diff (`"sekret"` in a test is a fake); the
  worktree is clean after the run and no `serve.server` process is left.
