# P124x - QFN aborts on the RX 9070 XT box under memory pressure (Saguaro 1.2.4) - REPORT

Branch `saguaro/q1-p124x-mortar-memory`, base f4b4dac08 (1.2.3+). No GPU was run here; every figure
below is from `ref/mortar-aborts.log` (the 60 engine lines before each abort) and
`ref/mortar-q4-config.json`, read against the engine's own code (all line numbers at the base commit).

## 0. The setup, as the logs show it

`/opt/llm/saguaro/build-hip/qfn/saguaro-qfn` (ROCm/HIP) on a 96 GB RAM / 16 GB VRAM box, Qwen3.8-Flash-Next
UD-Q4_K_XL: `--expert-cache auto --prefill auto --spec 3 --mtp MTP-Lamina --max-context 262144 --kv int8
--kv-resident 32768 --vram-reserve-mib 700 --resident-budget-gib 68`, config `parallel: 2` (`--batch 2`),
agent traffic with conversations of 120K-198K tokens, turn saves of 1.1-2.9 GB, `--session-dir` in place.

## 1. What grows RAM, and what holds VRAM at ~430 MiB - the figures

### 1.1 The thirteen engine ends in the log window (2026-11-24 08:37 - 11-26 20:34)

| # | time | exit | what ended it | VRAM free at req start | RAM avail | swap | RSS | major faults |
|---|------|------|---------------|------------------------|-----------|------|-----|--------------|
| 1 | 11-24 08:37 | -6 | watchdog: "parking the previous conversation" 61 s | 420 MiB | 2595 MiB | 3737 MiB | 89437 MiB | 117,940 |
| 2 | 11-24 09:21 | -9 | right after `ERR verify batch: timed out at layer 20 ... did not finish within 5 s (#267)` | (5 request(s) queued) | - | - | - | - |
| 3 | 11-24 10:25 | -6 | watchdog: prompt read of 145,954 tokens, 60 s no step | 416 MiB | 436 MiB | 5827 MiB | 89073 MiB | 156,346 |
| 4 | 11-24 11:15 | -6 | watchdog: "decode -1" 70 s | 432 MiB | 1993 MiB | 507 MiB | 87194 MiB | 2,247 |
| 5 | 11-25 01:07 | -6 | watchdog: prompt read waiting for the GPU at layer 1, 60 s | 432 MiB | 1453 MiB | 3146 MiB | 86971 MiB | 48,374 |
| 6 | 11-25 01:14 | -6 | watchdog: "decode -1" 76 s | 528 MiB | 1951 MiB | 502 MiB | 87835 MiB | 2,952 |
| 7 | 11-25 01:22 | -9 | right after `ERR verify: timed out at layer 42 ... did not finish within 5 s (#267)` | - | - | - | - | - |
| 8 | 11-25 01:46 | -6 | watchdog: "parking the previous conversation" 60 s | 428 MiB | 2381 MiB | 392 MiB | 86743 MiB | 5,279 |
| 9 | 11-25 06:28 | -6 | watchdog: "decode -1" 82 s | 428 MiB | 864 MiB | 2273 MiB | 88374 MiB | 4,917 |
| 10 | 11-25 08:43 | -6 | watchdog: "mtp draft: waiting for the GPU" 58 s | 428 MiB | 523 MiB | 5300 MiB | 88340 MiB | 81,021 |
| 11 | 11-25 16:05 | -6 | watchdog: prompt read waiting at layer 40, 60 s | 430 MiB | 221 MiB | 5391 MiB | 88851 MiB | 362,363 |
| 12 | 11-26 00:37 | -6 | watchdog: "parking the previous conversation" 61 s | 432 MiB | 3151 MiB | 2269 MiB | 86098 MiB | 31,821 |
| 13 | 11-26 20:34 | -6 | watchdog: "reading the prompt (verify windows)" 60 s | 428 MiB | 1822 MiB | 2571 MiB | 86280 MiB | 4,869 |

Every one of the eleven watchdog aborts (#29) carries the same stall-report line: **"VRAM free when this
request began: 416-432 MiB - LOW"** - below the engine's own `kVramSafeMib` 512 MiB (generate.cpp:1861) -
and every stall report reads the same RAM picture: `87-89 GiB resident, 0.2-5.9 GiB in swap, 0.2-3.2 GiB
available, thousands-to-hundreds-of-thousands of major page faults`. The two `-9` ends are the kernel OOM
killer's signal arriving right behind a #267 verify timeout. The shape is one and the same event: RAM at
the edge -> swap and major-fault storms -> a CPU-expert or GPU step that never finishes -> the watchdog
aborts (or the OOM killer does) -> the server restarts the engine.

### 1.2 What grows RAM during use (all RAM terms the logs name, on a 96 GB host)

* **The resident tier: 68.00 GiB** - `resident RAM: 68.00 GiB of experts in RAM` in every request
  summary; `--resident-budget-gib 68` is 71% of the host's RAM at the start of use, before the engine
  asks for one other byte.
* **The host K/V pool (pinned K/V + the park pool share one budget)** - `--kv int8` with 262,144-token
  context per batch slot. The park figures name the per-token cost: a 45,845-token snapshot is 892 MiB
  (~20.4 KiB/token), so `park_pool = per_token x max_context x batch` = 262144 x 2 slots is **~10.2 GiB**,
  always charged: the streamed K/V of the live conversations lives in it, and parked conversations share
  exactly that budget (`park_policy.hpp`: "Parked conversations share ONE pool with the live K/V").
* **Parked conversations**: 171K-198K-token snapshots of **2.6-3.3 GiB each** ("skip parking (snapshot
  2618 MiB exceeds available budget)", "need 892 MiB plus 2560 MiB floor").
* **The session writer**: turn saves append 140 MiB-2.85 GB per turn; the whole capture is held in RAM
  while the writer runs, and on the thrashing runs the capture itself blocked the serve thread for up to
  **30.9 s** ("the capture took the serve thread 30881 ms", a 2.85 GB save). P124a2 keeps the capture
  hold strong until the park that reuses it, so a multi-GB capture lives past its write.
* **Staging for prompt reads**: "KV staged this read: 2.35 GiB copied host->VRAM in 12 layer passes".
* **Page cache for the file tier**: the file tier wants tens of GiB of cache - "page cache held 41453.9 MB
  at the read, not 66487.8 MB", "the OS read 157887.6 MB from storage (2967 major faults) for 34928.2 MB
  of expert reads" - and when RAM goes short it is squeezed to nothing ("page cache held 0.0 MB at the
  read, not 0.0 MB" on most abort-adjacent requests), which is what turns the tier into **48k-362k major
  page faults** and swap of everything that can move.

Sum against the log: RSS at the aborts 84.8-89.4 GiB = the 68 GiB tier + ~17-21 GiB of pool + captures +
staging + buffers - i.e. the engine alone legitimately commits ~2x its K/V pool beside the tier, and the
file tier needs another ~10+ GiB of page cache to stop re-reading. On a 96 GB box, 68 GiB of budget
leaves neither.

### 1.3 What holds VRAM at 416-432 MiB free - why the log's own 512 MiB floor is not kept

* `--expert-cache auto` sizes the VRAM tier **once, at start**, from `free - reserve`, `reserve =
  --vram-reserve-mib (700) + prefill (0, borrowed) + the draft head + pipeline windows`
  (generate.cpp:6824-6857). Everything allocated **after** the cache comes out of that reserve: the
  verify-window graphs (the log watches the free figure fall 500 -> 446 MiB as the 4/3/2/1-token windows
  capture), the per-slot MTP heads, ROCm's own slack. What the stall report then reads at request start
  is 416-432 MiB - the 700 MiB reserve spent down, the engine's own 512 MiB paging line crossed.
* **Nothing sizes or runs against that 512 MiB line.** `kVramSafeMib` (generate.cpp:1861-1866) is used in
  exactly two places, both words: the stall report's LOW note (:1874) and the request-start block
  commented "said, never changed" (:18811-18828) - it warns and computes a better `--vram-reserve-mib`
  figure for the user, and changes nothing.
* The one automatic raise of the reserve - `kWddmAutoReserveMib = 2560`, added for exactly this paging
  shape under WDDM (#1376) - is `#if defined(_WIN32) && defined(STRATA_USE_HIP)` (:6813): **Linux/ROCm,
  where the same driver behaviour pages GPU memory, never gets it**. And the #496 `reserve_adapted` path
  only ever *lowers* a reserve, and only when `!o.vram_reserve_given` - the setup wrote 700 as the given
  reserve, so it is honoured "as it is" (:6808-6810).
* The relief machinery that does exist runs only on failure or in elastic-VMM mode: `vram_relief()`
  (layer.cpp:1084) is wired **only when `eslots_on`** (CUDA VMM slot budgets, generate.cpp:18594) and the
  verifier calls it only when `cudaGraphInstantiate` returns `cudaErrorMemoryAllocation`
  (verify.cpp:3849-3856). The ROCm driver here does not fail the allocation - it **pages** (the #2065
  shape the LOW line describes) - so the failure path never triggers and there is no proactive floor.

### 1.4 Why the parks' pressure rules did not save the box

They fired, late and small, and could not reach what was actually held:

* **Admission worked** (raw reading): "skip parking (physical RAM admission; need 892 MiB plus 2560 MiB
  floor, 2826 MiB available, source meminfo)" and "skip parking (snapshot 2618 MiB exceeds available
  budget)". Refusing new parks cannot give back RAM already committed.
* **The release floor is credited away**: since P124a the release decision credits the engine's own
  pinned RAM back (`Memory::own_pinned` = the 68 GiB tier + the K/V pool; park_policy.hpp `pressure_need`),
  so with 78+ GiB credited, `SAGUARO_PARK_FREE_MIB` (16 GiB) can only be breached below ~2.5 GiB raw -
  at which point `severe_bytes` (2560 MiB, the hard floor) is the only rule left firing. Swap at 2-5 GiB
  and 100k+ major faults - proof the host ran out - silence nothing: the credit has no swap term.
* **What was left to release was already nothing**: by the time the tick ran, the parks had been refused
  and dropped, and the releases the log shows are the pool's kept-mapped buffers: "memory pressure (2.9
  GiB free): 416 MiB of pooled park buffers go back to the OS", "(0.7 GiB free): 128 MiB",
  "(0.4 GiB free): 160 MiB" - hundreds of MiB against 2-5 GiB of swap. The holders that remained - the
  tier (never a release target), the live K/V pin, the session writer's multi-GB capture holds - are all
  outside the park policy.
* The PSI rule (`SAGUARO_PARK_PSI`, avg10 > 10, one entry per 10 s) gives back one entry a round and only
  when `press == 0 && pending == 0` (park_policy.hpp `plan()`, stalls branch) - a trickle against a flood.

## 2. Fixes (smallest rules that change the shape above)

(Implemented after this section was first committed; see the commits between this file's first commit and
the branch tip; each rule has a test named in section 3.)

### 2a. The resident budget keeps a host margin (engine)
...

### 2b. The expert cache sizes against the engine's own 512 MiB floor (ROCm auto)
...

### 2c. Swap use voids the parks' own-RAM credit (release before the swap storm decides)
...

### 2d. A #267 verify timeout that drained ends the request, not the engine
...

## 3. Tests

...

## 4. Open questions

...
