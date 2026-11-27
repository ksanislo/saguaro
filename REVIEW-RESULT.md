# MANAGER: FIX-FIRST round 12 (card, mortar, 2026-11-27 16:05-16:17 MST, commit db3373729) - FOUR steps, in order
Where it stands: load + speed PASS (30K read 31.0 / 35.6 s at 262144 / 1048576); the 40K repro PASSES (needle found);
the manager's db3373729 (a give's move never lands in an active loan) stopped the guard line ("0 hotter experts moved").
The 546K read still crashed in iq_embed_rows - and serialised HIP (AMD_SERIALIZE_KERNEL=3, HIP_LAUNCH_BLOCKING=1,
AMD_LOG_LEVEL=2) gave the REAL cause, not a memory corruption:
  :1:program.cpp :622 : devProgram.load() failed ... HSA_STATUS_ERROR_OUT_OF_RESOURCES
HIP loads a kernel's code object LAZILY at its first launch; iq_embed_rows first launches 546K into the read, when the
card is full (a 2027-slot loan + the grown context) -> the load fails -> a null kernel -> the host SIGSEGV.
With HIP_ENABLE_DEFERRED_LOADING=0 (eager) that is gone - but (a) eager costs ~0.7 GiB of VRAM (cache 2510 -> 2260
slots) and (b) the read then STALLED: host idle 108 s on a GPU fence (dma_fence_default_wait, "the GPU rang 0"), 5.5 GiB
RAM available, then the watchdog aborted (scratch/r11-eager-evidence.txt). Same class: the HIP runtime itself needs
VRAM (code objects, kernel scratch/private segments allocated on demand) and the elastic give-way leaves it none.
ALSO: this branch is missing ALL of dev (saguaro/qfn-first, 215 commits), incl. P124x's mortar memory rules (the
VRAM reserve floor, give-back before swap). Every elastic card so far ran without them.
Do, commit each:
1. DONE (you merged it: 57fac6ad8; the manager verified: QFN builds, the serve suites and the 18 standalone QFN unit tests pass) - MERGE saguaro/qfn-first into this branch (a merge; resolve generate.cpp & co keeping both sides; P124x's reserve
   rules must apply to every elastic growth path). FIX-REPORT: each conflict and the resolution. Run the QFN unit tests
   both sides added (park_policy_test, kv_host_growth_test, vmm_grow_test, read_backing_test, the P124x tests).
2. A RUNTIME RESERVE the elastic growth never takes (HIP and CUDA): enough for the runtime's own allocations (code
   objects if lazy, kernel scratch); size it from what the card test shows, default a few hundred MiB, env override;
   one stderr line at load naming it. Growth that would cut into it refuses cleanly (no-VRAM ERR), never hangs.
3. Kernel loading: either load all kernels eagerly at engine start on HIP (set HIP_ENABLE_DEFERRED_LOADING=0 before
   the runtime starts - and say what it costs in the load line), or keep lazy loading but warm every kernel the read /
   decode use at load (one tiny launch each) - pick the cheaper in VRAM, say which and why in FIX-REPORT.
4. (a) from the previous card, still open: why an expert was RECORDED in a slot inside an active loan (host_res ==
   2510 inside 392..2510) - the writer, file:line; fix at the source. db3373729 stays.
Build ONLY QFN (tools/qfn/build.sh <new dir under scratch/>, QFN_GGML_DIR=/data/llm/strata-main/third_party/llama.cpp). No GPU.
Card after (manager): load/speed, 40K, 546K (needle found), guard on, no OUT_OF_RESOURCES / stall.

MANAGER NOTE: step 1 is done and verified - start at step 2. A mortar run of the merged build with eager loading is running (manager) to see whether P124x's rules alone stop the stall; its result goes in this file when it ends.
