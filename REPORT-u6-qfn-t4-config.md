# U6 report - the QFN setup writes the live 4x Tesla T4 server's tuning, and pictures work on it

## What changed

Step 1 - classification (`./scratch/u6-keys.txt`, gitignored; the content is carried in the code comments,
the step-2 commit message and this report). Every key the live configs have and the fresh setup did not write,
each with the engine line it is read at:

- **GENERIC** (any GPU; NVIDIA and AMD alike): `SAGUARO_PARK_FAST` (conversation_copy.hpp:153-166, on by default),
  `SAGUARO_ELASTIC_SLOTS` / `_GIVE` / `_IDLE` / `_IDLE_S=300` (generate.cpp:202-253; with GIVE the engine says
  `INFO vision_lend=1`, generate.cpp:13121-13164), `SAGUARO_SLOT_GRID=6` (generate.cpp:298-305, slot_grid.hpp),
  `SAGUARO_CACHE_MESSAGE_BOUNDARY` (generate.cpp:16197, 20516), and the args `--conversation-cache-mib auto`,
  `--conversation-cache-slots auto`, `--conversation-cache-min-tokens 1024` (generate.cpp:983-1012).
- **T4-ONLY** (every card sm_75): `SAGUARO_ARGMAX_ROWS_SM70` (verify_kernels.cu:985-997, sm_7x opt-in),
  `SAGUARO_GDN_KEYHEAD=1` (prefill/kernels.cu:849-867, "also takes it on sm_75").
- **MULTI-GPU** (only with a layer split): `STAGE_SERVE`, `PARALLEL_LOAD`, `P2P_HANDOFF`, `SLOT_ADMIT` (+`_CHUNK`
  4096, +`_MULTI`), `READ_YIELD` (+`_CHUNK` 4096), `BATCH_READS` (+`_MAX` 8, `_FAST` 512, `_FAST_SHRINK` 512,
  `_DECODE_WINDOWS` 0, `_DECODE_SHARE` 0.3), `FAST_LANE` (+`_MAX` 96), `PIPED_MTP` (+`_DRAFTS` 2, `_MAX` 4),
  `SLOT_COPY_DIRECT`, `SHORT_READS` - gated as the engine gates them (batch_read.hpp:97 "not ... on a layer
  split"; generate.cpp:5977 `!stages.empty()`; 9148 `multi_gpu`).
- **TOPOLOGY** (computed): `NUMA_PLACE=1` + `NUMA_POOL=0` + `--pool-workers` = physical cores across all sockets
  minus the host's (2 x 10-core E5-2660 v2 -> 19) - only on a multi-socket CPU (numa.cpp:100, pool.cpp:758-765,
  pool.cpp:143/181 "the default workers are the local ones").
- **Per quant** (the two live refs differ; kept per quant like the existing `T4_BF16_TC_SKIP`/`T4_SELECT_SM75`):
  `SAGUARO_BATCH_WRAP_SPILL=1` q8_0 only (read by the server: serve/qfn_engine.py:93),
  `SAGUARO_PARALLEL_LOAD_REGISTER=overlap` ud-q4 only, `--prompt-cache 64` ud-q4 only (NVIDIA, t4-gated).
- **Skipped**: `BATCH_READS_OVERTAKE`, `PARK_STAGING_MIB` - no `getenv` anywhere in engines/qfn or serve/.

Step 2 (`tools/qfn_fit.py`, `saguaro.py`): `qfn_fit` got the four tables (`GENERIC_ENV/_ARGS`, `T4_ONLY_ENV`,
`SPLIT_ENV(_QUANT)`, `TOPOLOGY_ENV`) and `plan(..., cpu)` assembles env/args by class (`layout_env`,
`multi_socket`, `pool_workers`; CLI `--sockets/--cores`). `saguaro.py` got `qfn_cpu_info()` (reads
`/sys/.../topology/physical_package_id` + `core_id`; 0/0 off Linux), passes it to `QF.plan` in `write_config_qfn`,
and appends `lay["args"]` unconditionally (GENERIC args reach one-card configs too). serve/qfn_engine.py was not
touched and `extract_qfn_engine.py` was not run: no engine-server change was needed.

Step 3 (`tools/qfn_fit.py`, `saguaro.py`): `resident_budget_gib(ram_gb, quant)` never exceeds the model's own
expert bytes (its files via `quant_gib`, 103 GiB for UD-Q4), and new `page_cache_tier()` says when the page cache
holds every expert with the 20 GB spare untouched - then `--resident-budget-gib` is not written at all, what the
live 252 GB box does (neither ref config pins a tier). `write_config_qfn` follows `lay["resident_budget_gib"]`;
the setup's layout line says "the OS file cache the rest (all of them fit in RAM)" (`qfn_ram_said`). The 91.7 GB
AMD PC keeps its measured 71 GiB tier.

Step 4 (`saguaro.py`, `qfn_vision_step` docstring): pictures died because the server sends `VLEND` only after the
engine's start said `INFO vision_lend=1` (serve/server.py:3105-3117), which the give path says only with
`ELASTIC_SLOTS` + `ELASTIC_SLOTS_GIVE` on and batch slots present; the fresh env had neither, the `vram_reserve`
fallback needs `--vram-elastic`, and `_vision_up` started the encoder into a full CUDA0 anyway (the SIGKILL in
`./ref/vision-fail.log`). The fix is what step 2 writes (the GENERIC keys, exactly as the live configs run them,
with `parallel: 2` -> `--batch 2` giving the slots); the docstring now records the precondition. No server bug
found, so engines/qfn/serve/server.py stayed as it is.

Step 5 (`tools/test_setup_qfn.py`): new `LiveT4Parity` class - 4x T4 + 2-socket 2x10-core + 252 GB - asserts the
written env equals the live UD-Q4 env (the file is copied into the test, `ref/` being worktree-local, with a
ref-file cross-check that skips when the file is absent) bar the listed intended differences, and the args equal
it bar paths and `--max-context`. The AMD test asserts the config is the measured one plus the GENERIC additions
*only*; the plain-card test asserts no T4/split/NUMA keys and the GENERIC ones. All `write_config_qfn` tests now
patch `qfn_cpu_info` (this dev box has 2 sockets; the written configs must not depend on it).

## Files

- `tools/qfn_fit.py` - the four class tables, `plan(..., cpu)`, `layout_env`, `multi_socket`, `pool_workers`,
  `resident_budget_gib(ram, quant)`, `page_cache_tier`, CLI flags.
- `saguaro.py` - `qfn_cpu_info`, `qfn_ram_said`, `import glob`; `write_config_qfn` (class-driven env/args, the
  conditional `--resident-budget-gib`, docstrings); `qfn_vision_step` docstring (step 4); `set_up_qfn` messages.
- `tools/test_qfn_fit.py` - class expectations on every layout, the tier rule, a new `Topology` class.
- `tools/test_setup_qfn.py` - `cpu()` patch helper, AMD/plain-card assertions, new `LiveT4Parity`.
- `./scratch/u6-keys.txt` (untracked, gitignored) - the step-1 classification with the engine citations.

## Commits

1. step 2 `3917184fa` (step 1 produced only the gitignored scratch file - nothing to commit)
2. step 3 `47e7e0f95`
3. step 4 `0dbef586d`
4. step 5 `ad2507748`
5. this report (commit `--force`-added: `REPORT-*.md` is in this worktree's `.git/info/exclude`, and the plan
   says to commit it)

## Tests (run one at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest ...`)

- `tools.test_setup_qfn`: **OK, 43 tests** (40 before + 3 new parity tests)
- `tools.test_qfn_fit`: **OK, 22 tests** (20 before + 2 new Topology tests)
- `tools.test_setup_amd`: **OK, 29 tests** (unchanged)

No GPU was used, no engine started, nothing under /opt/llm touched, no engine C++ changed.

## Open questions

- `SAGUARO_GDN_KEYHEAD`: the refs' value is `"<removed>"`. The setup writes `1`, the only value documented to
  mean anything on sm_75 (prefill/kernels.cu:849; "same bits, A/B until measured"). If mangonel actually ran it
  at some other value, the parity test's note records the deviation.
- `--vram-reserve-mib`: live-q8 runs 689 (+`--vram-reserve-later-mib 687`) where live-ud-q4 runs 700. Those look
  like that box's measured free-VRAM numbers rather than a quant property, so the setup keeps writing 700 for
  both quants; the plan's per-quant list did not name them. Say the word and they become a per-quant table entry.
- The GENERIC conversation-cache args now land in AMD configs too (`--conversation-cache-mib auto` switches the
  park to the shared pool). The plan's class definition says GENERIC goes to AMD as well; it is a behaviour
  change on the RX 9070 XT box the next `--setup` there will apply.
- `--pool-workers` is only computed for multi-socket CPUs. A one-socket server keeps the engine default (that
  socket's cores minus the host's), which is already the same count; the plan phrased the flag as "across
  sockets", so single-socket machines are left on the default.
- `docs/PERF-LEDGER.md` does not exist in this tree, so the per-key evidence cites the engine source lines
  instead of the ledger.
- The `migrate_cyber_frost` path (old-engine CYBER-FROST configs) still pins its own `--resident-budget-gib` the
  old way - out of U6's scope (it is not the fresh setup); worth a look if the rule should apply there too.
