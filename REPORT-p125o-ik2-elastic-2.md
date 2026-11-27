# REPORT-p125o-ik2-elastic-2 - Inkling 2 Mini: the K / V grows on demand, step 2 (Saguaro 1.2.5)

Step 1's plan (ref/REPORT-p125e-ik2-elastic-1.md) did (a1) + (b1) and left a card that keeps every expert at a
262144 ceiling but runs conversations only to the committed 32768. This step does (a2) + (a3) of
ref/REPORT-p125b-ik2-vram.md section 3 and nothing more: the committed K / V becomes a VMM-backed reservation of
the ceiling with physical chunks behind the committed positions, and a request commits the chunks of the positions
it writes before its first pass. No kernels were touched. No GPU was run: this box has neither a CUDA-usable GPU
nor Inkling weights, so every check here is a compile, a host-side unit test, or a script meant for Ken's cards.

## 1. What changed

### a2 - the VMM-backed K / V

- **src/ik2/ik2_vmm.hpp (new)**: the range as the engine needs it - `open()` reserves the whole ceiling's address
  range once, `commit(bytes)` maps granule-sized pieces of physical VRAM behind the next stretch of it and grants
  access to the span (a failure gives back exactly what that call mapped; the earlier commit stands), `shrink` /
  `close` for the give-back step (a4). CUDA through the runtime's driver entry points (src/mm3/mm3_vmm.hpp's
  proven pattern: `cuMemAddressReserve / cuMemCreate / cuMemMap / cuMemSetAccess`, peers in the access list). AMD
  is first-class: `hipMemAddressReserve / hipMemCreate / hipMemMap / hipMemSetAccess` where this ROCm ships them -
  CMake `check_cxx_source_compiles` against `hip::host` sets `SAGUARO_IK2_HIP_VMM` (a configure-time probe, so an
  old ROCm does not break the build, it takes the fallback); where a piece is missing `open()` says so and the
  engine keeps today's plain allocation with one stderr line at the load.
- **src/ik2/ik2_plan.hpp**: the chunk arithmetic as pure functions - `kKvChunkPos` (4096 positions a chunk, the
  report's figure), `kv_rows_per_chunk` (the chunk's bytes rounded up to the granule - never down, so chunks stay
  whole), `kv_chunk_of`, `kv_chunks_for`. The commit order is the position order, ascending.
- **src/ik2/ik2_main.cu (a2)**: where the elastic start is taken **and** the GPU has the virtual memory API, each
  global visit's K and V is now the ceiling's address range with VRAM behind only the committed
  `SAGUARO_IK2_KV_START` positions. The kernels needed nothing (the touch points of the report, checked one by
  one): a full visit passes `kv_cap = 0` to attention and indexes `[0, pos)` by absolute position - no wrap to
  rebase; `Vi.kv_cap` stays the *physical* commit (the STAT `kv_cap=`, the snapshot logic, the DEPTH memset all
  read it as a1 had it, and the memset only ever touches mapped pages); the streaming / swing exclusion, the q4
  split and the sliding rings keep their own layouts and the elastic gate (no streaming, no q4, no two GPUs,
  no `--tp`, no `--parallel`, no EP) means they never share a range with it; `bind_slot`'s slot slices would stay
  valid anyway (the VA is contiguous and the base never moves). Fresh pages map zeroed, which is exactly the
  DEPTH test's semantics. The split of the numbers: `g_kv_commit` is the layout (Vi.kv_cap, the drafter's room
  and rows, STAT `kv_cap=`) and `g_kv_run` is back to the **announced** ceiling - `READY 262144`, INFO `ctx=`,
  the GEN guard and the CLI die included - so serve's `max_context` follows the ceiling again and step 1's
  past-the-commit `maximum context length` refusal is gone. The fallback where the API is missing is step 1
  byte for byte (plain allocation of the commit, `g_kv_run` at it) plus one stderr line.
- The `kv cache: ... (X GiB VRAM ...)` line reports the committed bytes, so an elastic start prints the same
  figure under a 32768 and a 262144 ceiling - the card test judges that.

### a3 - grow the committed K / V on demand

- **`kv_grow(n, err)` (src/ik2/ik2_main.cu)**: commit the chunks covering positions `[0, n)` across all the
  elastic buffers, in position order. Host-side, one call per stretch - never per token, never in a kernel - and
  a no-op on every other layout. On VRAM shortage what the call mapped goes back, the old commit stands, the
  CUDA graphs are unaffected (the base pointer never moves) and the caller fails just its request - there is no
  demotion yet (b4, later), so the experts keep their VRAM and that is that.
- **The commit points**: a serve request commits `ids + max_new` at its start, before the prompt's first pass -
  VRAM short, it is a `NOTE kv:` line, an `ERR` the server passes on and a `DONE`, with no kernel having run
  (the engine and its state untouched, exit code preserved); a kept conversation coming back (`full_rows`,
  park / unpark, a folder file read after a restart) commits its rows' positions first and the request fails the
  same clean way if they cannot land; the CLI commits its `depth + prompt + gen` before the prompt (a load-time
  failure, so it ends the run with the load's exit code) and `SAGUARO_IK2_BLOCK_WARM` commits its windows before
  recording them; the `--drafting auto` trials (`decode_greedy`, run before READY) commit their positions too.
- **The drafter's rows**: the drafter's context rows are still sized for the committed start step (the drafter's
  own growth is not this step). A serve request, or a CLI run, that reaches past the commit runs **undrafted** -
  same answers, plain tokens (the prompt lookup is host-side and stays on). The `--drafting auto` trial past the
  commit likewise runs with the drafter dropped.
- The P125e review's MEDIUM (the `--drafting auto` measure and `SAGUARO_IK2_BLOCK_WARM` bounding positions by the
  ceiling `ctx`, which was an out-of-bounds past step 1's commit) is resolved by construction: the ceiling is now
  address-valid (reserved) and page-valid (committed by these very hooks) where those loops reach.

## 2. What happens to a conversation that grows past the start step (said exactly)

On a VMM-backed elastic start (`NOTE context: ... growable chunks of 4096 - a conversation commits the chunks it
writes and runs to the ceiling`): `READY 262144`, so a 60K read at the 262144 ceiling is admitted; before its
prompt the engine maps the chunks of the 60K + 16 positions (7 more chunks a buffer past the start step, 16 MiB
each for the FP16 row - 112 MiB a buffer, ~224 MiB across the visit's K and V), attention reads `[0, pos)` over
pages that are all mapped, and the answer comes. Decode keeps
staying inside what the request committed (`max_new` was committed up front - no per-token host work). A second
turn past it commits only the new chunks (commit is monotonic; nothing is given back until a4). VRAM short: the
request fails cleanly with the NOTE and the engine keeps serving - it never dies mid-request here, because the
commit happens before any kernel of the request runs. On a card without the API (old ROCm), the start says so,
announces the commit and behaves exactly as step 1 reported. The 20K-inside-the-commit reads, the resident path,
the preloaded tier and every card where the elastic start is not taken are untouched, byte for byte.

## 3. Files

- new: src/ik2/ik2_vmm.hpp
- changed: src/ik2/ik2_plan.hpp (chunk arithmetic), src/ik2/ik2_main.cu (a2 + a3), CMakeLists.txt (the HIP VMM
  configure probe for saguaro-ik2), tests/ik2_plan_test.cpp (the chunk checks), tools/card_tests/ik2_kv_start.sh
  (the growth read + load-time-memory rule), this report
- commits, one per step: helpers + test; the vmm header + probe; a2 in the engine; a3 hooks (plus the @END field
  order); this report

## 4. Tests run, one at a time, and their results

- `./scratch/build-ik2-elastic-2/ik2_plan_test` (also `g++ -std=c++17 -Wall` direct): **ok** - 45 checks: the 21
  of step 1 (unchanged, still green) plus the chunk arithmetic - 4096 positions a chunk; the FP16 4096 B and FP8
  2048 B rows staying whole chunks at the 2 MiB granule and the 16 MiB / 3 MiB granules making chunks bigger,
  never smaller; chunk-of-position and coverage edges (0, 4095, 4096, the ceiling's last position); and the
  commit order: a load commits chunks 0..7 for the 32768 start step (128 MiB a visit, the same under any
  ceiling), a 20K read inside the commit commits nothing new, and a 60K read commits exactly 8..14, ascending.
- Build: `cmake -DCMAKE_CUDA_ARCHITECTURES=75 -DMAYA_ENABLE_CUDA=ON` into **scratch/build-ik2-elastic-2** (a new
  dir; CUDA 13.1) and `ninja saguaro-ik2 ik2_plan_test`: **clean** (only the pre-existing nvcc warnings -
  Shards::num, g_apu, and kFaRawQ4 in the kernels). The header alone also compiles under `nvcc -c -x cu` and
  `g++ -fsyntax-only`. No GPU run anywhere; the HIP branch is only compile-checked at configure on an AMD box.
- tools/card_tests/ik2_kv_start.sh: `bash -n` clean, and its verdict python run on canned rows both ways (grow
  answered -> PASS, grow refused -> FAIL with the reason); **not executed end to end** - no GPU and no Inkling
  weights here. It now also reads GROW (default 60000) tokens at any ceiling above it and PASSes only when that
  read answers *and* the `kv cache:` GiB figure reads the same at 32768 and 262144 (the 2 % rule), on top of
  step 1's expert-placement rule.

## 5. Open questions

1. **Drafting past the commit**: a request that may reach past 32768 runs undrafted. The drafter's context rows
   are the next VMM candidate (one FP16 buffer indexed by position, 4096 B a token - the same range machinery
   would fit); until then long conversations decode at plain speed. Worth its own step?
2. **The committed step vs the chunk**: a request commits everything `ids + max_new` can reach, so a 512-prompt
   with `max_tokens = 230000` at a 262144 ceiling commits 57 chunks a buffer (~0.9 GiB of the K and the V
   together) before its first token, even if it stops early. b4 (demote the coldest experts for room) will make
   that survivable; should the commit instead be capped per request and redone per window while generation runs?
3. **`tools/ik2_fit.py` wording**: the PC model still says "chats run to 32K until it can grow" on an elastic
   card - now optimistic-stale in the other direction (they run to the ceiling and pay per chunk). The plan said
   nothing beyond a2 / a3, so I left the model (and `test_ik2_kv_start`'s assertion of that line) untouched.
   Suggested follow-up: one wording pass, no arithmetic change.
4. **The HIP path is unrun**: `hipMemAddressReserve` / `hipMemCreate` / `hipMemMap` are compiled where the
   configure probe succeeds, but neither the VMM nor the fallback has been executed on gfx1201 - the RX 9070 XT
   box with `tools/card_tests/ik2_kv_start.sh` (watch the `NOTE context:` wording: "growable chunks" vs "no HIP
   virtual memory here") decides whether AMD takes a2 or the fallback today.
5. **Growth under memory pressure on a real card**: the fail-cleanly path is only code-reviewed here. The soak
   that matters: fill the tier to the brim (b1) and walk a conversation chunk by chunk into the ceiling - every
   commit should either land or fail the request cleanly, never the engine.
6. **The announced ceiling and the slider (step 1's Q 1)**: resolved to "announce the ceiling" by this step; the
   `kv_cap=` in STAT now says the physical commit while `ctx=` says the ceiling - if the dashboard wants both in
   one line, that is a serve change.
7. **a4 is now the missing half**: chunks only ever grow (commit is monotonic, `shrink` unused so far). Sessions
   ending and snapshot rewinds should give chunks back; until then a grown card stays grown (which is also why
   the mem_vs_ctx P124t rule is safe to enable on elastic cards - GPU use only moves when a conversation actually
   grows).
