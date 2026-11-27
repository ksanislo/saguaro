# MANAGER: FIX-FIRST round 7 (card tests, mortar RX 9070 XT, 2026-11-26 22:54-22:59 MST, commit e2684fc39)
LOAD + SPEED PASS again: 262144 vs 1048576 -> host K/V 1.30 / 1.30 GiB, expert RAM 68 / 68 GiB, GPU expert cache
2571 / 2509 slots, 30K read 35.0 / 37.0 s. Round 6's fix works: the device state now grows (50 "the host copy backs N
cells now" lines to 546148 cells, 8.00 GiB pinned in all; "elastic slots: CUDA0's expert cache gives 62 MiB ...").

GROWTH STILL FAILS, at the NEXT allocation - the YaRN switch. The same 546,145-token conversation on the 1048576 config:
  [saguaro] engine said: ERR no VRAM for the YaRN RoPE rows
  [saguaro] the engine reported an error: no VRAM for the YaRN RoPE rows
At load the engine said: "VRAM: the YaRN level's RoPE table (addresses for --max-context): 0.75 GiB asked, 0.00 GiB in
use (3 allocations)". When the conversation passes 262144 and switches to YaRN (SAGUARO_YARN_ELASTIC), backing those
rows needs VRAM, and that allocation does NOT go through the expert cache's give-way (round 6 put the give-way before
the device-state map only).
Fix, commit each:
1. Route EVERY growth-time VRAM allocation through the same give-way-before-map path: the YaRN RoPE rows, and any other
   allocation made on growth (list them in FIX-REPORT with file:line - grep the VRAM accounting lines you added in
   round 5 for "addresses for --max-context"; every one of them grows on use and must take room from the expert cache).
2. Size the YaRN RoPE rows by the positions in use (back them in chunks as the conversation grows past 262144), not
   all 0.75 GiB at the switch - or compute the rows for the window in use if that is how the kernel reads them; say which.
3. Unit test: the give-way order for each growth allocation (vmm_grow_test style).
Card test after: the 546K request must answer (needle "7319" at ~10K found), and the load + speed figures above unchanged.
Build for CUDA sm_75 in a NEW build dir under scratch/; no GPU runs.
