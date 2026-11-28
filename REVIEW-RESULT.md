# MANAGER: FIX-FIRST round 13 (card, mortar, 2026-11-27 21:59-22:16 MST, commit d1c28b199)
PASS so far:
- A (1dc74054a) WORKS: --max-context 262144 vs 1048576 now both 68.00 GiB of experts in RAM, 30K read 35.5 / 38.8 s.
- The 546K read with HIP_ENABLE_DEFERRED_LOADING=0 + SAGUARO_DEBUG_VA=1 (scratch/r13-ram-evidence.txt): NO GPU page
  fault this time (dmesg's last fault is 16:52) and the read ran 12 min to 491,520 of 546,065 tokens (90 %).
FAIL - the read then STALLED for lack of RAM (no fault, no crash until the watchdog):
  22:03:55  "KV streaming: the host copy backs 546148 cells now (8.00 GiB pinned for K/V in all)" - the host K/V grows to
            the whole request at its admission (+8 GiB pinned within 1 s)
  22:03:52  "resident RAM mode: 68.00 GiB of experts in RAM (page-locked)" - and it STAYS 68 GiB: no RAM give-back line
            anywhere in the run
  22:15:09  "memory: 88039 MiB resident, 736 MiB in swap, 2937 MiB RAM available" (the memory guard's floor is
            2560 MiB) - progress stuck at 491,520 from 725 s on, then the watchdog.
The elastic promise has a RAM side: when a conversation's host K/V (and its read's host buffers) grow, the experts
pinned in RAM give the same amount back (unpin + drop the coldest experts' RAM copies, which then come from the file
tier) BEFORE the growth pins, so RAM stays above the guard's floor and nothing swaps. A (the start sizes) is right;
the give-back on growth is missing (or never fires on this rig: resident RAM mode, --resident-budget-gib).
Fix (commit each):
1. FIX-REPORT: the growth path of the host K/V (kv_host_growth / session_host_fit) and of the read's host buffers -
   where it should ask the resident arena for RAM back (file:line), and why it did not here.
2. Make it so: each growth asks the arena for (bytes it will pin + margin) first; the arena releases its coldest
   experts' pinned RAM (they fall back to the file tier, like the GPU cache gives VRAM back); one log line per
   give-back ("resident RAM gives N GiB back for the context: M GiB of experts left"). Refuse cleanly (no-RAM ERR) only
   when the arena is at its floor. Never swap.
3. A unit test of the RAM budget: 96 GB box, 68 GiB arena, a 546K conversation -> the arena shrinks by ~8 GiB + margin.
Card after: load/speed, 40K, 546K with the guard on - the 546K must answer (needle "7319" found), no swap.
