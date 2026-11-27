# MANAGER: FIX-FIRST round 5 (card test, mortar RX 9070 XT, 2026-11-26 17:38-17:42 MST, commit 67e1244e6)
Host side still PASSES (1.30 GiB pinned K/V at 262144 and 1048576; expert RAM tier 68 GiB at both). VRAM side improved
but FAILS, and 262144 got WORSE than round 3:
- VRAM IN USE BEFORE THE EXPERT CACHE: 262144 "8.67 GiB of 15.92" (7.25 free) vs 1048576 "12.08 GiB of 15.92" (3.84 free)
  -> ~3.4 GiB is still allocated by --max-context somewhere between model load and the expert-cache sizing, and no
  log line names it. GPU expert cache 1925 vs 1057 slots; 30K read 679 tok/s vs 240 tok/s.
- At 262144 the cache fell from 2256 slots (round 3, ae2dac446) to 1925 and the read from 820 to 679 tok/s: the new
  VRAM path costs ~1 GiB even at the default context.
Do, in order, committing each:
1. A load-time accounting line: every VRAM allocation >= 64 MiB made before the expert cache is sized, its name and
   size (one line each, or one table), and ONE line saying whether the elastic VRAM path is ENGAGED on this device
   (VMM / hipMemAddressReserve available and used) or fell back - with the reason.
2. From that accounting (read the code paths it names), find the ~3.4 GiB that still scales with --max-context and the
   ~1 GiB the new path costs at 262144; fix both. If HIP lacks what the elastic VRAM path needs, say so in FIX-REPORT
   with the evidence and implement the fallback that still satisfies the rule on this device (e.g. size the start step
   only and grow by re-allocation outside the hot path, or let the expert cache give VRAM back as a conversation grows).
3. Acceptance (Ken): at 262144 and 1048576 the same VRAM in use before the expert cache, the same expert-cache slots,
   the same 30K read speed; and 262144 no worse than round 3 (2256 slots, ~820 tok/s on mortar).
Build for CUDA sm_75 in a NEW build dir under scratch/; no GPU runs.
