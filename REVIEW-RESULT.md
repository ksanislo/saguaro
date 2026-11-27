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

B. THE 546K READ STILL STALLS with eager kernel loading (HIP_ENABLE_DEFERRED_LOADING=0) and P124x's 1024 MiB ROCm
   reserve in place: "host idle for 78848 ms", the GPU rang 0, then the watchdog aborted (main::lambda #75). So it is
   not (only) the runtime reserve. Find WHERE it waits: the watchdog's dump names the threads/states - add what you
   need to it (the stream/event being waited on, the last kernel enqueued, the read position) and say in FIX-REPORT
   what the read was waiting for. Candidates: a hipEvent/stream wait on work that was never enqueued (a grow that
   failed quietly and left a chunk unmapped / a stage that never rang), a GPU page fault on an unmapped VMM page
   (HIP can hang instead of fault), the K/V staging copy waiting on the host copy's growth.
   The manager will re-run with AMD_SERIALIZE_KERNEL=3 + AMD_LOG_LEVEL=2 on your build if you ask in FIX-REPORT.
Then steps 2-4 of the previous card still apply if needed (runtime reserve, kernels ready early, the host_res-in-loan
writer). Build ONLY QFN (tools/qfn/build.sh <new dir under scratch/>, QFN_GGML_DIR=/data/llm/strata-main/third_party/llama.cpp). No GPU.
