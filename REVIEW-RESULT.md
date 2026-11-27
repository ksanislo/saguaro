# MANAGER: FIX-FIRST round 6 (card tests, mortar RX 9070 XT, 2026-11-26 20:11-20:33 MST, commit d94e12c4b)
LOAD + SPEED NOW PASS (Ken's acceptance): --max-context 262144 vs 1048576 -> host K/V 1.30 / 1.30 GiB, expert RAM tier
68 / 68 GiB, GPU expert cache 2523 / 2561 slots, 30K read 35.9 / 36.4 s (better than round 3 at 262144). Keep all of it.

GROWTH FAILS. One conversation of 546,145 tokens on the 1048576 config (a plain chat request; the host copy grew
correctly: 44 "the host copy backs N cells now" lines up to "546148 cells now (4.91 GiB pinned for K/V in all, 60 ms)"):
  saguaro: the context's device state could not be mapped on CUDA0: no error
  [saguaro] engine said: ERR no RAM is left to pin this request's context (546145 tokens)
  [saguaro] the engine reported an error: no RAM is left to pin the context of this conversation ...
  done: 0 tokens (error)
So growing the DEVICE (VRAM) state past the start step fails: the mapping fails and the error is reported as "no error"
and then as a RAM shortage (it is not: 13 GB were available; 1.2.3 served this request with everything reserved).
Fix, commit each:
1. The mapping failure path: report the real cause (the driver's error code / which allocation / how much was asked /
   how much VRAM was free / whether the expert cache was asked to give way and what it gave), never "no error", never a
   RAM message for a VRAM failure.
2. Make the growth work: when mapping more device state needs VRAM, the expert cache gives way (coldest first) BEFORE
   the mapping is tried, as much as the growth needs; the request then proceeds. A conversation that truly cannot fit
   (VRAM exhausted after the cache gave all it can) must fail with a clear message - but 546K on a 16 GB card with a
   7.3 GiB expert cache must fit (QFN's VRAM K/V is a 32K-cell window: the rest streams from host).
3. A unit test for the give-way-before-map order (pure helper or a fake allocator), and the error text test.
Card test after: the same 546K request must answer (a needle at ~10K must be found).
Build for CUDA sm_75 in a NEW build dir under scratch/; no GPU runs.
