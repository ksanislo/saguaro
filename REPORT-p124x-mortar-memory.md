# P124x - QFN aborts on the RX 9070 XT box under memory pressure (Saguaro 1.2.4) - REPORT

Branch `saguaro/q1-p124x-mortar-memory`, base f4b4dac08 (1.2.3+). No GPU was run here; every figure
below is from `ref/mortar-aborts.log` (the 60 engine lines before each abort) and
`ref/mortar-q4-config.json`, read against the engine's own code (sections 0-1: line numbers at the base commit f4b4dac08; sections 2-3:
at the branch tip, the fix commits named there).

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
| 2 | 11-24 09:21 | -9 | the server's fatal-line kill (SIGKILL) 17 ms behind `ERR verify batch: timed out at layer 20 ... did not finish within 5 s (#267)` | (5 request(s) queued) | - | - | - | - |
| 3 | 11-24 10:25 | -6 | watchdog: prompt read of 145,954 tokens, 60 s no step | 416 MiB | 436 MiB | 5827 MiB | 89073 MiB | 156,346 |
| 4 | 11-24 11:15 | -6 | watchdog: "decode -1" 70 s | 432 MiB | 1993 MiB | 507 MiB | 87194 MiB | 2,247 |
| 5 | 11-25 01:07 | -6 | watchdog: prompt read waiting for the GPU at layer 1, 60 s | 432 MiB | 1453 MiB | 3146 MiB | 86971 MiB | 48,374 |
| 6 | 11-25 01:14 | -6 | watchdog: "decode -1" 76 s | 528 MiB | 1951 MiB | 502 MiB | 87835 MiB | 2,952 |
| 7 | 11-25 01:22 | -9 | the server's fatal-line kill (SIGKILL) behind `ERR verify: timed out at layer 42 ... did not finish within 5 s (#267)` | - | - | - | - | - |
| 8 | 11-25 01:46 | -6 | watchdog: "parking the previous conversation" 60 s | 428 MiB | 2381 MiB | 392 MiB | 86743 MiB | 5,279 |
| 9 | 11-25 06:28 | -6 | watchdog: "decode -1" 82 s | 428 MiB | 864 MiB | 2273 MiB | 88374 MiB | 4,917 |
| 10 | 11-25 08:43 | -6 | watchdog: "mtp draft: waiting for the GPU" 58 s | 428 MiB | 523 MiB | 5300 MiB | 88340 MiB | 81,021 |
| 11 | 11-25 16:05 | -6 | watchdog: prompt read waiting at layer 40, 60 s | 430 MiB | 221 MiB | 5391 MiB | 88851 MiB | 362,363 |
| 12 | 11-26 00:37 | -6 | watchdog: "parking the previous conversation" 61 s | 432 MiB | 3151 MiB | 2269 MiB | 86098 MiB | 31,821 |
| 13 | 11-26 20:34 | -6 | watchdog: "reading the prompt (verify windows)" 60 s | 428 MiB | 1822 MiB | 2571 MiB | 86280 MiB | 4,869 |

Ten of the eleven watchdog aborts carry the same stall-report line: **"VRAM free when this request began:
416-432 MiB - LOW"** - below the engine's own `kVramSafeMib` 512 MiB (generate.cpp:1861) - and every stall
report reads the same RAM picture: `87-89 GiB resident, 0.2-5.9 GiB in swap, 0.2-3.2 GiB available,
thousands-to-hundreds-of-thousands of major page faults`. The eleventh (#6, log:361) began at 528 MiB -
above the line, no LOW note - and stalled just the same: the 512 MiB line marks where paging starts to
hurt, not where it is survivable (review N1: the first version of this table said "every one ... 416-432").

The two `-9` ends are **not the kernel OOM killer** (review F2; the first version of this section said it
over). Both follow a line that says the GPU did *not* drain - log:119 (`ERR verify batch: timed out at
layer 20 ... did not finish within 5 s (#267)`) and log:443 (layer 42) - and the engine exits 17 ms and
4.8 s behind the server echoing it: that is the server's own fatal-line handler, which `proc.kill()`s the
engine (SIGKILL -> -9, server.py:1369-1380 at the base), not the kernel deciding anything. So both -9s are
the #267 kill of a window whose spin kernel was still on the card - events this branch's rule (d) leaves
fatal, correctly - and the log contains **no** timed-out-with-drained-note event at all: the eleven
"released the verify window's GPU waits (#267): the GPU finished in 0-9 ms" lines are the watchdog's own
pre-kill release (`release_live_verifiers`), each followed by a -6. Rule (d) is therefore justified by the
drain evidence (waits drain in 0-9 ms whenever they are raised in time) and by the code shape (the base
server kills on ANY timed-out ERR prefix, drained or not), not by a killed-while-drained event this log
window happens not to show. The shape is otherwise one and the same event: RAM at the edge -> swap and
major-fault storms -> a CPU-expert or GPU step that never finishes -> the watchdog aborts or a #267 window
is killed -> the server restarts the engine (a minute or two, five requests waiting).

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

Each rule is its own commit; §2's file:line figures are at the branch tip (the commits below), §1's stay at
the base. The review's own findings (F3/F4/F5, N1-N4) that changed these rules are named where they bit.

### 2a. The resident budget keeps a host margin (engine) - 624f95f58

`resident_policy.hpp` (new, header-only like the other pure rules): the engine reads the host total
(`total_physical_memory()`) and the model's shared K/V pool (`qsa_kv_bytes_per_cell x qsa layers x
--max-context x batch`) before the tier pins, and the largest tier the host can carry is

    cap = total - runtime - margin,   runtime = 2 x pool,   margin = max(8 GiB, total / 8)

How much and why, against §1.2's figures: the pool is 10.2 GiB on the mortar box and the engine's own RAM
above the tier measured **18-21 GiB = about two pool passes** (the pool itself, then a turn-save capture of
up to 2.9 GiB held while the writer runs plus the staging riding beside it - the pool is the one budget the
captures are bounded by, hence 2x); the margin is the eighth - 12 GiB of a 96 GiB host - for the OS and
whatever else runs, and for the page cache the file tier reads through (it asked for 41 GiB when it could
hold it and fell to 48k-362k major faults when squeezed to nothing; 12 GiB is what keeps the re-reads away,
8 GiB is the floor a 32 GiB host can still afford). For the mortar box: cap = 96 - 20.4 - 12 = **63.6 GiB
against the 68 asked** - the asked budget is cut to it, said with all four figures. A budget already under
the cap changes nothing; `SAGUARO_RESIDENT_BUDGET_CAP=0` keeps the asked figure as it is (every rule here
has its knob, review N2); on a host so tight no tier fits (cap 0) the engine keeps the asked budget and
**says so loudly** rather than silently flipping the box out of the resident mode its setup named. Rule (c)
and the admission checks keep guarding a host that is full anyway - this rule is the one that stops the
host getting there through the engine's own budget.

### 2b. The expert cache sizes against the engine's own 512 MiB floor (ROCm auto) - 1c190a03a

`vram_policy.hpp` (new) is now the single source of `kVramSafeMib` (512 MiB Linux, 1024 WDDM - §1.3's
label with nothing sizing against it), and `hip_auto_reserve_mib` gives the auto sizing a floor: a HIP card
of 15 GiB and up (the #1709 threshold) on Linux keeps a reserve of **floor + measured slack = 512 + 512 =
1024 MiB**. Why 1024: the mortar box's given 700 MiB reserve was at 416-432 MiB free by request start -
**~272 MiB spent by the window captures and the MTP heads before anything paged** - and the paging that
followed into a host RAM the tier had already filled is the #2065 shape in §1.1; 512 + a rounded 512 leaves
the card at or above the floor when the windows capture. It is the Linux sibling of #1376's
`kWddmAutoReserveMib`, which was `#if defined(_WIN32)`-only; a CUDA build never raises (the same reserve
measured honest there, #1376's A/B arm), Windows keeps its own path, and an **integrated APU is left
alone** (review N3): its reported total is GTT RAM, not a card, and the UMA block above the reserve block
sizes for that shape already. Unlike #496's lowering, this floor does not step back for
`--vram-reserve-mib 700`-style given figures below it - the given 700 is what ended the engine - and it
applies inside `--expert-cache auto` only: `--expert-cache N` still sizes the cache by hand. The cache
size changes no output bit.

### 2c. Swap use voids the parks' own-RAM credit (release before the storm deepens) - c54ba610d, 3b4cae905

The credit that silenced the 16 GiB floor to the 2.5 GiB hard line (§1.4) rests on a premise - the pinned
tier stays resident. Once the kernel has working pages in swap the premise is false, so
`conversation_swap_used()` (SwapTotal - SwapFree, malformed input unknown) feeds `Memory::swap_used` and
`swap_high()` stops the credit above `SAGUARO_PARK_SWAP_MIB`. The threshold's reason (review F5 asked for
one, and for a sane default - the first cut said 16 MiB): every mortar abort line carried **392 MiB to
5.9 GiB** of swap and the storms ran 2-5 GiB, while a calm box's residual swap - idle daemons paged out
once - sits in the tens of MiB; **256 MiB separates dust from storms**, and 16 MiB was not sane because
swap in use does not drain by itself (it frees when its owners swap pages back in or exit), so a
dust-level bar would latch the credit off permanently on exactly the resident hosts whose raw
MemAvailable sits under `SAGUARO_PARK_FREE_MIB` by design - parks emptied and never returning with no
storm in progress. The bar is deliberately an in-use bar, not a growth bar: it holds while swap holds,
which is the truth "the host moved working pages to disk and they are still there", and a park released
under it costs a re-read, not an abort. How it interacts with rule (a): a budget capped by 2a keeps raw
MemAvailable above the free floor, so the credited and raw readings agree and the swap term is silent;
it fires when the operator overrode the cap, when something outside the engine filled the host, or when
the box was booted into pressure - and PSI (avg10 > 10) stays the earlier, softer warning. `=0` turns it
off like the other rules, and the release/trim notes print the swap figure beside the free one (the INFO
`park_pool` line gains `park_swap_mib=`: the server's parser reads the keys it knows and ignores the rest -
an external one keyed on the exact token count should key on names, §4).

### 2d. A #267 verify timeout that drained says so, and the server stops killing the process for it - 652b1cd63, 26feff6db, 9ff4ff922

`release_gpu_waits(timeout, poison, re_armed)`: the serve watchdog (which ends the process right after)
keeps releasing **and poisoning**; a window that timed out on its own 20 s host wait releases with
`poison=false` and **re-arms** once the stream drains - the failed window leaves what a never-rang window
leaves, and the next `stage_inputs` re-arms the per-window handshake words. A stream that did not drain,
and a split's verifier (a next stage, or an incoming hand-off word - the hand-off rings are a running
count the next window must not start under), stay poisoned. `timed_out_note()` words the three shapes
apart (review F3): `...did not finish within 5 s (#267)` / `...released and the GPU finished (#267)` -
printed only when the verifier really took the arm back - / `...released and the GPU finished, but the
verifier cannot take another window (#267)`.

**What this does NOT yet do (re-review F-A): the re-armed process does not survive to take the next
request.** Every serve path that prints a timed-out ERR returns 1 out of the request loop - the solo
prompt read (generate.cpp:21616-21626), the solo decode window (:22665-22667), the batch step (:18768)
and a batch_fatal prompt read (:21073) - so on a drained self-timeout the engine prints its ERR line and
the process exits seconds later; the server logs "the engine had stopped" and reloads. The one-minute
reload with five requests waiting (§1.1) is therefore NOT avoided yet: making the re-armed shape return
to stdin (fail the request, keep the process) is the engine change that claim waits for, in exactly those
paths, gated on the re-arm. The verifier re-arm itself holds (verify.cpp:554-563) and takes another
window - it just never gets one, since the process is gone.

What the rule does deliver today: the server's `fatal_engine_line()` treats only the re-armed note as
recovered, so the request ends with the engine's own error line instead of the pump's SIGKILL of a
process that was ending itself cleanly, the shapes are told apart truthfully (F3), and the protocol is in
place for the engine half: a recovered line means the verifier can take another window. The not-drained
note, both "an earlier window never finished" lines, and a timed-out line with no note (an older engine)
keep the kill. Per review F4 the coupling is stated where it lives: the note-means-recovered rule holds
because this server and the engine binary ship from one build (`serve/qfn_engine.py` is extracted from
this very file) - pairing an older engine, which printed the same words while poisoned, with this server
buys one extra failed request, so keep the two from one build.

## 3. Tests (the plan's: unit tests like core/park_policy_test.cpp, server tests for the setup change)

Ran one at a time (g++ 14.2, -std=c++20, no GPU; scratch/p124x-build):
* `park_policy_test` - 56 checks: the swap rule joins the release policy - the 256 MiB default, at-the-bar
  is not above, 64 MiB of residual dust keeps the credit and the parks, a 2 GiB storm re-asks the raw
  10 GiB and `plan()` releases on-disk-first, `SAGUARO_PARK_SWAP_MIB=0` is exactly the P124a behaviour.
* `conversation_memory_test` - 45 checks: SwapTotal/SwapFree parsing (malformed unknown, no-swap 0,
  SwapFree-above-SwapTotal never negative, I/O failure unknown) and that the host reading answers on Linux.
* `resident_policy_test`, `vram_policy_test` (new; registered in CMakeLists beside `host_limits_test`,
  header-only, no GPU) - 13 and 12 checks: the mortar cap 63.6 GiB against the asked 68, the cap-0 hosts,
  the "one byte over the tight fit" edge; the 9070 XT 700 -> 1024 floor, CUDA/Windows/integrated-APU/
  small-card/unknown-total arms unbumped, at/above the floor kept, a given 0 floored.
* `serve/test_fatal_recovery.py` - 11 checks: the drain/kill protocol through real pipes - the drained note
  ends the request and the (fake) engine idles for the next GEN (P124x d - the fake speaks the protocol the
  server must honor; the real serve loop still exits the process after the line, 2d/F-A), the not-drained
  and still-closed notes and the older engine's note-less line still kill it, `fatal_engine_line` as a
  table (F3's words included). Run via `scratch/run_p124x_server_tests.py`, which registers the root's
  `serve/turbo_live.py` under the name the QFN serve package imports it by - that import has been
  unresolvable standalone since 936183783, at this branch's base as well (pre-existing, not this branch's;
  noted rather than fixed here).
* Build: **CUDA sm_75** - `cmake -S engines/qfn -B scratch/build-p124x-sm75 -DSTRATA_ENABLE_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=75` (nvcc 13.1, ninja): the `strata` target links whole (saguaro-qfn, 172
  steps, no errors; the pre-existing kernel warnings unchanged) - verify.cpp, generate.cpp and
  conversation_memory.cpp all compiled in it; nothing was run on a GPU. **HIP**: no ROCm toolkit in this
  session, so the wave32/HIP build is not run here - the changed sources are backend-neutral host C++
  (they compile against the same `cuda*` names the hip_compat headers map) and the manager's card run
  (§3.6 of the checks list) is where the HIP binary actually gets exercised.
* Card test for the manager: `tools/card_tests/mortar_pressure.sh` (bash -n clean, not run here - no GPU).
  Three conversations of ~100K/175K/250K tokens answer turns for 30 min (the plan's load), RAM and VRAM
  sampled every 10 s; PASS = no abort, VRAM never under `VRAM_FLOOR_MIB` (512), swap never growing more
  than `SWAP_GROW_MIB` (256) above its baseline; the rules' own start-up lines (budget clamp, reserve
  floor) are echoed into the verdict.

## 4. Open questions

* The swap bar is an **in-use** bar that holds while swap holds (2c). If the card run shows resident hosts
  parking fine with real residual swap and still storming, the alternative is a growth bar (swap above
  what the engine saw at start + bar); the park-side cost differs in one direction only (a plateaued storm
  would re-park), and the raw floor plus PSI still guard it. The bar itself is one env away either way.
* Rule (d)'s recovered note trusts the paired engine (F4): stated in the server comment and here, not
  enforced by a handshake. An explicit version/behaviour line in the engine's READY would close it.
* Rule (d)'s engine half (re-review F-A): the re-armed shape should end the request and return to stdin
  (solo generate.cpp:21616/:22665, batch :18768/:21073, gated on the re-arm) instead of `return 1` -
  until it does, every drained #267 still costs the reload. The batch half is the delicate one (other
  slots are mid-window); this branch deliberately keeps the conservative exit and says so rather than
  shipping an unwound batch state that was never GPU-tested.
* `park_swap_mib=` on the INFO `park_pool` line: this repo's parser reads keys by name and ignores
  unknowns (checked, no consumer), but an external parser keyed on the exact token count of that line
  breaks - the token list of the engine's INFO lines is nowhere documented, and this found none as a place
  to fix.
* `test_fatal_recovery`'s recovered case proves the SOLO protocol end to end; the batch shape
  (`ERR verify batch: ... finished (#267)` landing on a slot) is covered by the classifier table and the
  engine's note words, end to end only by the card script's 30-minute run (a #267 may not even appear
  there - it is not inducible without the stall hook).
* The budget cap uses `--max-context x --batch` for the pool even when `--kv-resident` or a smaller pool
  says less is actually held; over-generous (the cap cuts later than it must), never under. The exact pool
  is known later than the pin - sizing the cap from it means moving the clamp after the pool allocation.
