# MANAGER: FIX-FIRST round 12 (card, mortar, 2026-11-27 16:38-16:44 MST, commit f9ccea2eb = dev merged in, 57fac6ad8)
Evidence: scratch/r12-merged-evidence.txt. Step 1 (merge dev) is DONE and verified. Two problems, fix BOTH, commit each:

A. A MERGE REGRESSION - the load is no longer elastic in RAM (the release blocker's own promise):
     --max-context 262144:  resident RAM mode: 67.90 GiB of experts in RAM, 30K read 38.92 s
     --max-context 1048576: resident RAM mode: 30.77 GiB of experts in RAM, 30K read 46.66 s   (-37 GiB, +20 %)
   Before the merge (db3373729) both were 68.00 GiB. A P124x rule from dev (the resident-budget margin / the
   reserve floor / the swap-voids-the-credit release - see engines/qfn and docs P124x) sizes its RAM headroom from
   --max-context (the reserved K/V), not from what is BACKED. Under elastic K/V the host copy starts at one 32768-cell
   step and grows by giving RAM back from the experts when a conversation actually grows. Fix: every P124x margin
   counts the BACKED / committed K/V (plus the one growth step it runs ahead), never the ceiling; growth past it takes
   RAM from the experts through the give-back path. FIX-REPORT: the rule (file:line), before/after numbers at both
   contexts from a unit test of the budget. P124x's mortar protections (no swap, no OOM) must still hold.

B0. NEW (manager, 2026-11-27 ~16:52 MST, serialised HIP: AMD_SERIALIZE_KERNEL=3 + HIP_LAUNCH_BLOCKING=1 +
    AMD_LOG_LEVEL=3, eager loading, the merged build): the LAST kernel launched before the 546K read stopped - its
    launch never returned - is the prompt path's RoPE:
      hipLaunchKernel ( 0x2a17678, {20480,1,1}, {32,1,1}, ... )  ShaderName : void
      strata::prefill::(anonymous namespace)::rope_kernel<true>(float*, long, long, long, long, float, float, float, ...)
    (the kernel before it, rms_rows_kernel, returned). So the reader of the unmapped page is the prompt RoPE reading
    the RoPE / YaRN table rows (mrope.hpp's rope_tab_cs: t.cos + (p - t.min_pos) * 32 + pair). The log says "CUDA0's
    YaRN RoPE table covers 557056 positions now (was 0)" - check that the rows the read's positions need are really
    MAPPED at the moment rope_kernel runs (rope_map's backing vs the positions of THIS chunk; the YaRN table vs the
    plain one; the <true> variant = which table; a grow that ran on another device/context; a later trim
    (qsa_rope_yarn_trim) or a give that unmapped rows still needed; the 2 MiB granule rounding at the table's end).
    Fix the backing so every row a kernel can read is mapped first. Then B1 only if this does not explain it.
B. THE 546K STALL IS A GPU PAGE FAULT (dmesg on mortar, scratch/r12-dmesg.txt), 3 s into the read:
     amdgpu: [gfxhub] page fault ... Process saguaro-qfn pid 2129188 ... in page starting at address 0x00007f9ebba01000
     GCVM_L2_PROTECTION_FAULT_STATUS:0x0080113B  Faulty UTCL2 client ID: TCP (0x8)  MORE_FAULTS 0x1  MAPPING_ERROR 0x1  RW 0x0
   A compute kernel READ an UNMAPPED page (4 KiB into a 2 MiB granule = one VMM chunk not mapped) and the wave retries
   forever -> "host idle", the watchdog. DO THIS FIRST, in this order:
   B1. SAGUARO_DEBUG_VA=1: one stderr line for EVERY VMM reserve / map / unmap / release and every pinned host register /
       unregister (cuMemAddressReserve/hipMemAddressReserve, cuMemMap/hipMemMap, the K/V host pieces, the expert-cache
       chunks, the device follow state, the rope/YaRN rows, the staging loan, the slot budget's chunks): the VA range
       [base, base+size), what it is (owner + index), and the op. Off by default; cheap; commit it first.
   B2. Ask the manager (FIX-REPORT line "MANAGER: RUN B2") to run the 546K card with SAGUARO_DEBUG_VA=1 +
       AMD_SERIALIZE_KERNEL=3 + AMD_LOG_LEVEL=3: the fault address then falls in exactly one logged range, and the last
       kernel launched before the dmesg time is the reader. Then fix the missing grow-before-use (or the unmap-while-used).
   Old notes on the stall (superseded): host idle 78848 ms, the GPU rang 0, watchdog main::lambda #75; eager loading +
   P124x's 1024 MiB ROCm reserve did not change it.
Then steps 2-4 of the previous card still apply if needed (runtime reserve, kernels ready early, the host_res-in-loan
writer). Build ONLY QFN (tools/qfn/build.sh <new dir under scratch/>, QFN_GGML_DIR=/data/llm/strata-main/third_party/llama.cpp). No GPU.
