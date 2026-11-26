# P124l report - `saguaro.py --bench` measures several answers at once too

`--bench` keeps its one-answer standard exactly as it was and, after the prompt-reading phases, in the same warm
engine session, now runs ref/parallel.py's method: for N in 1, 2, 4 - only N within the config's places
(`parallel` / the engine's batch slots) - N different 2K-token prompts are written to the engine before the first
answer is read, 256 greedy tokens each, and the run prints the total tok/s and each answer's own tok/s, each
answer's own time including its prompt read and its turn waiting.  `--bench-parallel off` skips the section; the
default is on.

## What changed

1. **Read-through (step 1, no code):** `--bench` is `saguaro.py`'s `bench()` (~5643) on top of `tools/maya_bench.py`
   (`MB`): a quiet-machine check and GPU lock, the engine started as the server starts it (`bench_start_command`)
   as a line-protocol subprocess, `MB.run_session` driving one session (warm-up, 3 x 256-token answers, the 2K/8K
   prompts 3 times each, QUIT), and `bench_text` (~5785) writing saguaro-bench.txt (header / GPUs / setup /
   decode / prompt reading / wall time).  The section fits after the prompt phase (engine warm, one session,
   nothing after it that could be perturbed) and prints between "prompt reading:" and "wall time:".  Scratch
   notes: `scratch/step1-notes.md` (untracked).
2. **The section** (`tools/maya_bench.py` + `saguaro.py`):
   - `PARALLEL_NS = (1, 2, 4)`; `MB.gen_line` takes the request's `rid` (default 0, so old calls are unchanged).
   - `MB.Session` now reads the engine's places at its start: `READY ... multi=<n>` (GLM / DeepSeek / MiniMax /
     Inkling line engines with `--parallel`) as `multi` (int now; the old bool uses keep working), and QFN's
     `INFO ... batch_slots=<n>` printed before READY as `batch`.  `Session.close` also closes the engine's pipes.
   - `MB.parallel_places(cfg, reported)` - the engine's own word when it said one, else the config's
     `"parallel"` / `--parallel` / `--batch` / `--slots` arg; `MB.parallel_ns(places)` - which Ns to measure and
     which to skip with a note.
   - `MB.plan_parallel(doc, ctx)` - four different 2K windows starting just after every window the standard run
     read (the standard opens at `997 * k`, k = 0..5, plus the document's end for the warm-up; these open at
     `997 * 6..9`), so nothing here is reused; `{}` (skipped with a note) when the docs or context cannot hold it.
   - `MB.parallel_phase` - writes every request first (all sent together), then reads until every one is DONE,
     timing each answer from its own line to its own DONE.  It speaks both several-conversation protocols: the
     rid one (`GEN ... rid=<id>` in, `T / DONE <rid>` out) and QFN's slots (`BGEN <slot>` in, `BT / BDONE <slot>`
     out, the answer that ends at its admission completed from its control `DONE` when `BADM <slot> 0` says so),
     and the plain `GEN` line for a single request on any engine (that is the N = 1 baseline everywhere).
   - `MB.run_parallel` (the rounds, places = what the engine said or the config asks) inside `MB.run_session`
     (new `par` / `cfg_places` params, results under `run["parallel"]`, a new `several` phase in `phases`, and
     `MB.phase_line` prints a phase only when a run measured it); `MB.parallel_section` renders the table text,
     `bench_text` places it between the prompt readings and the wall time.
   - `bench(..., parallel=True)`: the plan is built from the same encoded docs and skipped for `--bench-smoke`
     and `--compare` (an A/B stays two identical one-answer sessions); `res["parallel"]` is None when skipped, so
     the table says "several at once: skipped (--bench-parallel off)".  CLI: `--bench-parallel on|off` (default
     on), wired in `main`.
   - `serve/qfn_engine.py` was not touched (nothing in `engines/qfn/serve/server.py` changed, so no extractor run).
3. **Tests** (`tools/test_bench.py`, class `BenchParallel`, 8 tests): the N selection from a config and from the
   engine's report, `parallel_ns`, `plan_parallel`'s openings (none of them read by the standard plan, too-small
   docs / context -> `{}`), the section's text and its place in `bench_text` (absent of --compare, the off note
   when skipped, `several` only in the wall line of a run that measured it), and measured rounds against three
   scripted mock engines - rid (`READY multi=2`), QFN (`INFO batch_slots=4`, incl. a slot that ends at its
   admission), and one-at-a-time (plain GENs answered in the order sent).  No GPU, no network.
   `tools/test_maya_hip.py`'s scripted `--bench` engine got the one extra DONE the new N = 1 round reads.
4. **Docs:** README's "Measured on Saguaro" method line and docs/RELEASE-CHECKS.md's standard line each gained a
   sentence on the new section.

The one-answer numbers cannot move: the decode / prompt phases, their prompts, their timing and the result line
are untouched; the new rounds run after them.

## Files

- `saguaro.py` - `bench()` + `bench_text()` + `--bench-parallel`.
- `tools/maya_bench.py` - the several-at-once parts, `Session` (multi / batch_slots, close), `phase_line`.
- `tools/test_bench.py` - `BenchParallel`; module docstring.
- `tools/test_maya_hip.py` - its scripted bench engine answers the extra round.
- `README.md`, `docs/RELEASE-CHECKS.md` - the method sentences.

## Commits (one per step, plus the hip-mock fix)

- `1d4e3f4f5` P124l: --bench measures several answers at once too (the section itself)
- `77d71e6fb` P124l tests: the several-at-once section without a GPU
- `e6a44c3da` P124l docs: the method lines say what --bench now measures
- `23ad77b26` P124l: test_maya_hip's scripted --bench engine answers the several-at-once round too

Step 1 changed no tracked file (its notes are in scratch/, gitignored), so it has no commit of its own.

## Tests run

- `python -m unittest tools.test_bench` (the module the plan names, run several times, last after all edits):
  **47 tests, OK** - 39 pre-existing plus the 8 new `BenchParallel` ones; no ResourceWarnings left.
- AST-compiled `saguaro.py`, `tools/maya_bench.py`, `tools/test_bench.py`, `tools/test_maya_hip.py`; all new diff
  lines <= 120 columns; `saguaro.py --help` shows `--bench-parallel {on,off}` with its help.
- `scratch/par_check.py` (scratch, untracked): the pure functions and both mock protocols plus a full
  `run_session` round - checked by hand before the unittests were written.
- Not run per the run-only-named-modules rule: `tools.test_maya_hip` and the rest of the suites (its scripted
  engine was updated by line-by-line reasoning; it now feeds exactly the 12 lines the session reads).

## Open questions

1. **The test's "mock server":** the plan named `serve.server --engine mock`, but that is an HTTP server and
   `--bench` drives the engine's line protocol directly (a server on the port stops the bench by design).  The
   tests therefore use scripted Python mock engines speaking the same GEN / rid / BGEN lines `--bench` drives -
   the same scripted-engine, no-GPU idea as `--engine mock`.  If an HTTP-shaped gate is wanted, it belongs to
   ref/parallel.py itself, not to the bench client.
2. **One round per N** (ref/parallel.py averages two rounds of its ~400-token chats).  One 256-token round each
   keeps the added time to roughly a decode answer's worth a round; averaging two would double it - say the word
   if the standard should repeat.
3. **--compare skips the section** (keeps an A/B symmetric and exactness-first).  A/B'ing parallel speed may be
   a follow-up of its own.
4. **Where QFN's numbers will read:** the parallel round goes through the engine's own slots (like the server's
   `--batch` path), not through HTTP / llama-swap, and QFN admits prompts serially on its control lines while
   the admitted slots decode beside the read - so the bench figures answer "what the engine does with N at
   once", close to but not identical to ref/parallel.py's served numbers when the README table is regenerated.
5. **Places precedence:** what the engine said at its start (`multi=` / `batch_slots=`) outranks the config (the
   engine may run fewer than asked); the config is the fallback only when the engine says nothing, which no
   current parallel-capable engine does.
