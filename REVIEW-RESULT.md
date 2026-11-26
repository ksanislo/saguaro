# MANAGER: FIX-FIRST (card test, mortar, 2026-11-26 13:24-13:30 MST, commit ae2dac446)
HOST side now PASSES: "1.30 GiB of pinned RAM for the K/V of every session" at BOTH --max-context 262144 and 1048576
(the solo session 0.46 GiB, the batch slots and the drafters the rest), and the expert RAM tier is 68.00 GiB at both.
VRAM side FAILS: the GPU expert cache holds 2256 experts at 262144 but only 843 at 1048576 ("resident RAM mode: 68.00 GiB
of experts in RAM (page-locked), 2256 / 843 in the GPU cache"), and the prompt path's lendable slots are 1915 vs 618
("933 of 1915" / "618 of 618 lendable slots ... will read their experts from the pack"). Result: the same 29,571-token
read takes 36 s (820 tok/s) at 262144 and 256 s (115 tok/s) at 1048576.
So some DEVICE (VRAM) allocation is still sized by --max-context at load, and the expert cache gets what is left.
Fix (Ken's acceptance: just HAVING 1M set must not hurt; memory follows use):
1. List every VRAM allocation made at load whose size depends on --max-context (grep the KV-streaming / QSA / attention
   / RoPE-YaRN tables / position buffers / prompt-chunk scratch / MTP drafter K/V / index buffers in engines/qfn/src;
   compare the two load logs' VRAM lines if the engine prints them). Write each with file:line and its size at 262144 and
   1048576 into FIX-REPORT.md.
2. Make each one sized by use (a start step, grown with the conversation like the host K/V now is) or by the VRAM window
   (32768 cells), never by --max-context. Nothing may change for a conversation that really grows past 262144.
3. The card test must then show the SAME GPU expert-cache count and the same lendable slots at both contexts.
Build for CUDA sm_75 in a NEW build dir under scratch/; no GPU runs. Commit each part.
