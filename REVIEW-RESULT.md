# MANAGER: FIX-FIRST round 9 (card tests, mortar RX 9070 XT, 2026-11-27 10:06-10:13 MST, commit 8dee8bf1c)
LOAD + SPEED PASS again (262144 vs 1048576: 1.30 / 1.30 GiB host K/V, 68 / 68 GiB experts in RAM, GPU cache 2541 / 2519,
30K read 30.8 / 36.3 s). The 546,065-token request still does NOT answer. A core dump + gdb (scratch/r8-card-evidence.txt:
backtrace, engine log, server output) shows the "SIGSEGV" the last two rounds chased is NOT in the read:

A. (the real failure) the read ends early, two ways in two runs of the same request:
   run 1: "[saguaro] done: 0 tokens in 5 s (0.0 tok/s) (length, cancel=False)" ~3.5 s after "the prompt path's K/V
          staging holds 786432 cells" - a 546K read cannot finish in 3.5 s; no ERR line. Find what ends a read as
          "length" with 0 tokens (a reach/limit check against max_new or the context, a staging cap) and fix it.
   run 2: a clean refusal: "the context's device state could not be grown on CUDA0: hipMemCreate failed: out of
          memory; 70 MiB asked (35 chunks of 2 MiB), 0.13 of 15.92 GiB free on the card" -> "ERR no VRAM is left for
          this request's context (546145 tokens): the card's expert cache gave all it can". It did NOT: right before,
          "CUDA0's expert cache gives 72 MiB ... 2396 of 2541 slots hold experts" - ~7 GiB of experts still on the
          card. The give-way hands slots to the slot/context budget, but a growth that maps NEW physical chunks
          (hipMemCreate) needs the cache to FREE chunks back to the driver. Fix: when the driver is short, the
          relief must free expert-cache physical memory to the driver (or the grow must remap the given slots'
          chunks) until the ask fits; only refuse when the cache is down to its floor. Log what was freed.
B. (secondary, at exit) after SIGTERM the engine exit(0)s and SEGFAULTs in an atexit destructor:
   #3 strata::core::VmmRange::release() <- #4 ~vector<unique_ptr<ScoresPool>> <- __run_exit_handlers. A global
   VMM pool is released after the HIP runtime is gone (call through a null pointer in libamdhip64). Fix: release
   the global VMM pools (ScoresPool and any other static VmmRange owner) explicitly before exit / HIP teardown, or
   skip release at process exit. The server reports this as EngineDied -11 on every shutdown after growth.
Commit each; unit tests where the code allows (the relief freeing to the driver; the exit order); a NEW build dir.
Card test after: the 546K request must answer (needle "7319" found); load + speed unchanged; a SIGTERM after growth
exits 0 with no SIGSEGV.

## MANAGER UPDATE (one read pass, 2026-11-27 ~13:20 PST) - the likely cause; START HERE, skip the broad survey
The gather's buffers (m.emb, m.tok_dev, m.cs) are sized by the CHUNK, but m.emb and the rest of the prompt
workspace are a bump allocation from the BORROWED region = expert-cache slots LENT to the read (prefill.cpp:343,
:538, carve :1738; "a loan of 2061 slots" in the card log). The elastic-slots give-way picks its range ONCE at setup:
  generate.cpp:13726-13729   v.top = v.cache->slots(); for (p : pf_parts) if (p.first >= 0) v.top = min(v.top, p.first);
                             v.lo = v.top;
but the loan is re-sized PER REQUEST (p.first recomputed: generate.cpp:9883, :13247). Round 8 made the 546K read's
staging large (786432 cells) -> its loan grew to 2061 slots -> p.first moved far BELOW the v.top captured at setup.
The give at 13860-13904 then moves experts into / unmaps chunks of slots [v.lo, v.top) that are INSIDE the active
loan (c0..c1 -> qsa_slot_budget_give, then round 9's free-to-driver). When the read crosses the 32768 start step the
context state grows -> a give -> the read's own workspace is unmapped -> the next iq_embed_rows (the gather) touches
it -> SIGSEGV. Fits: in the gather, the first chunk past 32768, right after a give, only since the loan got big.
(The K/V give-way kvg gets this right: kvg_start(lend_first) at generate.cpp:9958 / :23615.)
FIX: the elastic-slots give must never take a slot inside an ACTIVE loan: clamp each give to the CURRENT loan
boundary (min over parts of p.first_now / p.first at the time of the give, not at setup), and when the loan grows
past v.lo, either refuse the give (so the grow takes driver memory / refuses cleanly) or shrink the loan first.
Verify the same for every other path that unmaps / moves cache slots during a read (vram relief, free_to_driver,
trims). Unit test: a fake cache with a loan resized after setup; a give must not touch [p.first, slots).
Build ONLY the QFN engine (tools/qfn/build.sh <new dir>, AVX floor) - not the whole project.
