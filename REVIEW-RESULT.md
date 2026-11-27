# MANAGER: FIX-FIRST round 11 (a) - the ROOT CAUSE only; (b) is done by the manager (db3373729)
Round 10's slot guard named the crash on mortar (scratch/r10-guard-evidence.txt):
  saguaro SLOT GUARD: gv CUDA0: moves a hotter expert into a slot: slot 2510 is inside the ACTIVE LOAN of
  'CUDA0 prompt loan' (its buffers are laid out from slot 392 to 2510)
(b) - DONE by the manager in db3373729: both give-ways (kvg ~8888, gv ~13928) take move targets only below the lowest
active loan on that cache. Do NOT redo or revert it.
(a) - YOUR TASK: why was an expert RECORDED in slot 2510 (host_res[i] == 2510) while that slot was inside the active
loan? Laying a loan out is meant to take every resident inside it off host_res (generate.cpp ~16262-16268: the loop
over the participant's layers, host_res[i] >= first -> p.lent, kNotResident). That matters beyond the crash: if
host_res points an expert at a slot holding the read's buffers, decode/read could compute with GARBAGE weights for
that expert (a silent quality bug, no crash). Find every path that writes host_res[i] = <slot> and check each against
an active loan: adaptive swaps (apply_pending / swap landing - note the existing "a loan from slot %d met %lld adaptive
swap(s) still pending ... into the loan's slots" line), prefetch (srcp->prefetch after a give), the refill after a
read (refill_one), a loan grown or moved (first_now lowered) WITHOUT re-running the eviction loop, and the batch /
solo / pipelined loan paths. FIX-REPORT: the exact writer (file:line) and the sequence that produced slot 2510.
Fix it at the source (no expert may become resident in an active loan's slots; a loan that grows clears the newly
lent slots first). Add a SAGUARO_DEBUG_SLOT_GUARD check at that writer too. Unit test where the code allows.
Build ONLY QFN (tools/qfn/build.sh <new dir under scratch/>, QFN_GGML_DIR=/data/llm/strata-main/third_party/llama.cpp). No GPU runs.
