# P124n - report: a release check for "memory at load must not depend on --max-context"

The rule (Ken, 2026-11-26): context not in use costs nothing - at load, `--max-context 262144` and `1048576`
(or any two settings) must hold the same memory. QFN broke it (host K/V pinned for the whole max at load) and
it was only seen on a 96 GB box. This task builds the CHECK for every engine; it does not fix engines.

## What changed

### 1. `tools/mem_vs_ctx.py` (new) — the check

`python3 tools/mem_vs_ctx.py <config.json> [--contexts 32768,262144] [--port 8093]`, one run per context:

- **The run**: the server starts from a COPY of the config (`scratch/mem-vs-ctx/<config>/<context>/config.json`)
  with its own port (one per context, all pre-checked free), its own session dir (both shapes: the `*_SLOT_DIR`
  env keys and QFN's `--session-dir` arg), its own engine log, host loopback, never idle-unloading and no
  `api_key` (the copy drops it, the start gets `--api-key ""` and the child's `SAGUARO_API_KEY` is removed),
  and `--max-context` set to that context (rewritten when present, appended when not - same shape as the
  server's own `set_context`). `--engine` choice per plan: `python -m serve.server --config ...`.
- **The wait**: `/health` until it answers 200 `ok`. A failed load (503 `status: failed`) raises at once with
  its error; a server process that ends before binding (bad config, tokenizer missing) also raises at once.
- **The reading** (10 s after ok, the load still settles): `VmLck + VmPin` (pinned/locked RAM) and RSS summed
  over the process tree (the server and its engine child, from `/proc/<pid>/stat` ppids, `/proc/<pid>/status`
  in KiB -> MiB); the GPU memory used on the config's cards by `nvidia-smi`, else `rocm-smi` (whichever exists
  and answers; `-` and left out of the rule when neither can say); and the engine's own budget lines - log
  lines containing "pinned RAM" / "resident" (case-insensitive) or "KV" (as written).
- **The stop**: kill by the port's owner - the LISTEN holder found through `/proc/net/tcp{,6}` by socket
  inode - SIGTERM first (the server ends the engine on its way out), wait for the port to fall free, SIGKILL
  the tree's leftovers. The next run only starts once the port is free.
- **The verdict**: a table of every figure per context plus the budget lines; PASS when every figure differs
  by **less than max(2%, 256 MiB)** between the contexts (2% of the bigger one), FAIL naming the biggest
  difference that broke the limit. Exit 0 PASS, 1 FAIL (a run that never got to `/health` ok too - the line
  names the run's log folder), 2 a port already in use before anything started.

Pure stdlib, Linux (`/proc` — `main` checks `/proc/self/status` before anything starts and exits 2 when it cannot
be read, rather than printing zeros and PASSing on nothing; macOS and Windows have no `/proc`); no imports from
`serve/`, so the tests and the tool share one file.

### 2. `tools/test_mem_vs_ctx.py` (new) — the tests

27 tests, no GPU (the 22 first delivered, plus the five the review round added - see `FIX-REPORT.md`):
the `/proc` parsing on fixed inputs (status KiB->MiB and absent keys; the tree through a
fake `/proc` whose comm carries spaces and parens; the tree's sums; the port's owner found by inode in a fake
`/proc/net/tcp{,6}`, an ESTABLISHED line is not a listener); the GPU reader on canned `nvidia-smi` /
`rocm-smi` output (nvidia first, rocm when nvidia is absent, all cards when the config names none, a card the
reader does not list cannot say, no reader is None); the budget grep; the config copy (no api_key, own session
dir both shapes, own log, `--max-context` set or appended, the caller's dict untouched); the table; the PASS/
FAIL rule on fixed figures (the 256 MiB floor below 2%, 2% above it on a big figure, the boundary equal fails
- "less than" - the biggest difference named, an unreadable GPU left out, one row "nothing to compare"); and
one real cycle against `python -m serve.server --engine mock`: `/health` ok, the port's owner is the server,
the tree reads RSS > 0, the stop frees the port again.

### 3. `docs/RELEASE-CHECKS.md`

The tool joins the table ("does the model hold the same memory at load whatever `--max-context` says"), gets
section 4 - per engine, on the SMALLEST-RAM machine available that still runs the model, never only on a big
one - with the two example command lines and the standing note that **QFN fails it until the elastic K/V work
(P124e) lands** (its FAIL is the known state; record the figures anyway), and one line in "What still needs a
person".

`serve/qfn_engine.py` and `engines/` were not touched - the plan builds the check, it does not fix engines.

## Files

- `tools/mem_vs_ctx.py` (new)
- `tools/test_mem_vs_ctx.py` (new)
- `docs/RELEASE-CHECKS.md` (table row, section 4, one per-machine bullet, intro line)
- `REPORT-p124n-mem-vs-ctx.md` (this; force-added, `REPORT-*.md` is excluded)

## Tests run

- `python -m unittest tools.test_mem_vs_ctx -v` (the only module the plan names; venv python 3.13.5):
  **27 tests, OK** (22 as delivered, 27 with the review fixes; the module was re-run after every fix) - run
  twice, once after the two fixes below and once against the final commits. No GPU,
  no model: the only real process it starts is the mock engine.
- By hand while developing (not tests): the tool's own cycle against a live mock server on a free port
  (`/health` ok in 1.0 s, the port's owner was the server pid, the tree read RSS 87 MiB, the stop freed the
  port and the server exited 0); a config with a missing exe (the server ended at once - `wait_health` names
  it, the port stayed free, no orphan); the table/verdict/grep on the shapes above.
- Two fixes the tests caught: `fake_proc` wrote `net/tcp` beside the fake root instead of under `net/` (the
  port-owner test failed honestly before passing), and `start_server` left the log handle open in the tool
  process (ResourceWarning in the mock test) - now closed after `Popen`, the child keeps its copy.
- The tool was deliberately not run on a real model (the plan: the manager does).

## Commits (one per plan part, plus the report)

1. `tools/mem_vs_ctx.py` — the check
2. `tools/test_mem_vs_ctx.py` — the tests (with the log-handle fix they caught)
3. `docs/RELEASE-CHECKS.md` — the release-check entry
4. this report

## Open questions

- **The budget grep marks** ("pinned RAM", "resident", "KV") are the plan's words. Engines word their lines
  differently (`kv=fp8`, `kv_resident=...`, GiB figures in various shapes); the first card runs will show
  whether the marks need widening (or engine-specific patterns) before the lines are actually useful.
- **RSS is summed over the tree**, so pages shared between server and engine count twice. The two contexts
  share the same shape, so the comparison is fair, but the absolute RSS over-reads physical RAM a little.
- **The rule's edges**: "less than max(2%, 256 MiB)" is strict - a difference exactly at the limit FAILs -
  and 2% is taken of the bigger context's figure. The plan says "differs by less than max(2%, 256 MiB)"; if
  "at most" was meant, one comparison flips.
- **AMD boxes**: the `rocm-smi --showmeminfo vram` parser expects the classic `GPU[N] : VRAM Total Used
  Memory (B):` lines; a ROCm that prints its newer table would read None (GPU left out of the rule) until
  its format is added - unmeasured here, this box has no GPU tools to run.
- **No `/proc`, no check**: macOS and Windows have no `/proc` (the process-tree and port-owner reading), so the
  check is Linux-only - it refuses to start when `/proc/self/status` cannot be read, and the mock-cycle test
  skips there; a release check on those boxes would need psutil or `Get-NetTCPConnection` - out of the plan's
  scope, untested.
- Whether the nightly should ever run this automatically (two loads per model per night) - the plan puts it
  in the per-machine (person) list only.
