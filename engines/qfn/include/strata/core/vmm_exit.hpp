#pragma once
// P124e (round 9, manager card): the PROCESS is going away - what a global VMM pool's release must call then.
// The engine's static pool vectors (the follow pools, the block-score scratch, the rope tables, the slot pools -
// layer.cpp) are built at load, so their atexit hooks register BEFORE the HIP runtime registers its own
// teardown and run AFTER it: their ~VmmRange called into a torn-down libamdhip64 through a null function
// pointer - the SIGSEGV after exit(0) the server logged as "EngineDied -11" on every shutdown after growth.
// So the first VMM api() use (long after the runtime initialised) arms this handler, and atexit's LIFO runs
// it BEFORE the runtime's own teardown: from then on a destructor holding a VmmRange lets its range go
// UNreleased - at exit the addresses and the physical chunks are the kernel's to reclaim anyway.  Card test:
// a SIGTERM after growth exits 0, no SIGSEGV.  Pure (no CUDA/HIP include) so the grow's test can read it.
#include <atomic>
#include <cstdlib>

namespace strata::core {

inline std::atomic<bool>& vmm_exit_flag() {
    static std::atomic<bool> f{false};
    return f;
}

/// What the process's atexit list calls: from here on the device runtime may already be gone.
inline void vmm_mark_exiting() { vmm_exit_flag().store(true, std::memory_order_relaxed); }

/// True from that handler on: release() / vmm_chunk_free() must NOT call the runtime again.
inline bool vmm_exiting() { return vmm_exit_flag().load(std::memory_order_relaxed); }

/// Registers the handler once, at the first VMM api() use - after the runtime's init, so the LIFO puts this
/// ahead of the teardown the runtime itself registered.
inline void vmm_arm_exit_watch() {
    static std::atomic<bool> armed{false};
    if (!armed.exchange(true)) std::atexit(&vmm_mark_exiting);
}

}  // namespace strata::core
