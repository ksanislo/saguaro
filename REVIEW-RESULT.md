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

## MANAGER UPDATE 2 (deeper read, ~13:40 PST) - the first update's cause is DISPROVED; use this instead
The loan IS kept above the elastic-slots give line (generate.cpp:16221-16228: room_p = slots - give_top(cache); the
chunk halves until need_p <= room_p), and laying a loan out takes every resident in it off host_res
(:16262-16268, kNotResident), so the gives at 13836-13904 neither unmap nor move into an active loan AS WRITTEN.
Open lead (unconfirmed): TWO give-ways start from the same line at setup - kvg (kvg_start(lend_first) :9958; kvg.lo
counts down from kvg.top) and the elastic slots gv (v.top = min(slots, p.first) :13726; v.lo counts down) - and each
counts down on its own. If both are on in this rig (KV streaming + --batch) they can hand out the SAME chunks twice.
DO THIS FIRST (cheap, decisive): a debug guard SAGUARO_DEBUG_SLOT_GUARD=1 that, on every unmap / remap / move-into /
free-to-driver / budget give of a cache slot or chunk (kvg, gv, vram relief, free_to_driver, trims, the loan's own
relayout), checks it against (a) the active loan [p.first_now, slots) of every part on that cache and (b) every
chunk already handed out by the OTHER give-way, and prints one line naming the path, the slots/chunks and who owns
them (then aborts with that line). Then the 40K-token repro at --max-context 1048576 on the card (the manager runs
it) names the culprit in one run. Fix that path; keep the guard (off by default).
Build ONLY the QFN engine (tools/qfn/build.sh <new dir>, AVX floor) - not the whole project.
