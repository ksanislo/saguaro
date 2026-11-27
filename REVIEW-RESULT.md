# MANAGER: FIX-FIRST round 8 (card tests, mortar RX 9070 XT, 2026-11-27 07:01-07:06 MST, commit e3ec5d022)
LOAD + SPEED PASS again (262144 vs 1048576: 1.30 / 1.30 GiB host K/V, 68 / 68 GiB experts in RAM, GPU cache 2572 / 2511,
30K read 30.0 / 35.6 s). Round 7's fix works: the YaRN rows now back on growth ("CUDA0's YaRN RoPE table covers 557056
positions now (was 0)"), and two "HIP virtual memory: hipMemCreate failed: out of memory" were answered by the expert
cache giving way (41-45 slots each) as designed.

GROWTH NOW CRASHES (SIGSEGV, engine exit -11) 3.5 s into reading the same 546,065-token prompt. The last engine lines
(scratch/r7-crash-tail.log has 30):
  saguaro serve: SAGUARO_YARN_ELASTIC: prompt of 546065 tokens, reads from 0 at level 0
  saguaro: HIP virtual memory: hipMemCreate failed: out of memory
  saguaro batch: elastic slots: CUDA0's expert cache gives 134 MiB ... 2349 of 2511 slots
  saguaro: CUDA0's YaRN RoPE table covers 557056 positions now (was 0)
  saguaro serve: CUDA0: the prompt path's K/V staging holds 786432 cells (a loan of 2030 slots, chunk 12544)
  [saguaro] done: 0 tokens in 5 s (0.0 tok/s) (length, cancel=False)      <- the server saw EngineDied (-11)
The prompt path's K/V staging is sized to 786,432 cells for a 546K prompt and borrows 2030 of the 2349 slots left: it
still sizes itself from a whole-context assumption, not from the elastic state (what is backed now, what the cache
can lend after the slots and the context state took theirs). The SIGSEGV comes right after it.
Fix, commit each:
1. Find the staging sizing (grep "the prompt path's K/V staging holds") and what the read writes / reads through it;
   check every index against what is actually BACKED under elastic VRAM (device state, rope rows, the staging loan) -
   the crash is an access past a backed range. FIX-REPORT: the exact access, file:line.
2. Size the staging from the elastic state (backed cells, the chunk, what the cache can lend now) and make the read
   grow the backing ahead of the chunk it is about to touch (the give-way path) - never touch an unbacked page.
3. A unit test for the staging size / backing order; build QFN for CUDA sm_75 in a NEW build dir under scratch/.
Card test after: the 546K request must answer (needle "7319" found); load + speed unchanged.
