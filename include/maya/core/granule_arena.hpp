// include/maya/core/granule_arena.hpp - a reserved address range whose granules are pinned only where they are written.
//
// The GLM engine's KV-streaming latent cache (src/core/glm_model.cu's kv_stream_make) was ONE cudaHostAlloc sized by
// --max-context: 2.95 GB of pinned RAM at 262144, 11.8 GB at 1048576, allocated at load whether or not a conversation
// ever writes a position of it (item M of the memory report, ref/REPORT-p125c-glm-mem.md).  This is the replacement the
// report proposes: the whole range is reserved as ADDRESS space (mmap / VirtualAlloc MEM_RESERVE - no bytes of RAM),
// and a granule (64-256 MB) is registered with the driver - cudaHostRegister, hipHostRegister through the HIP shim -
// the first time anything names its bytes.  The addresses never move, so a pointer into the arena taken once (the
// layers' host-copy rows) stays valid across pinning, giving back, and re-pinning; only what a conversation has
// written costs pinned RAM, and giving a conversation back unregisters its granules and hands the pages to the kernel
// (MADV_DONTNEED / MEM_DECOMMIT), ready to be pinned again for the next one.
//
// The registration itself is the caller's (a `Pinner`): the bookkeeping here is host code a CPU test can drive with a
// fake, and the engine passes cudaHostRegister/hipHostRegister.  Pinning is not free - cudaHostRegister of a granule
// costs milliseconds - so it belongs on the granule, not the page or the token.
#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace maya::core {

namespace granule_detail {
constexpr uint64_t kPageSize = 4096;   // the pinning and the decommit both work in these
}  // namespace granule_detail

// What a granule needs from the driver: pin these bytes so a device may read and write them (the pages are already
// resident and zeroed when `pin` is called), and give them back.  The engine implements this over
// cudaHostRegister/cudaHostUnregister (hipHostRegister/hipHostUnregister in HIP builds); a test over a recording fake.
struct Pinner {
    virtual ~Pinner() = default;
    virtual bool pin(void* at, uint64_t bytes) = 0;
    virtual void unpin(void* at, uint64_t bytes) = 0;
};

// An address range of `total` bytes, divided into granules of `granule` bytes (the last one may be shorter).
// Nothing exists until ensure() names a byte range; release() gives all of it back and re-ensure() takes it again.
// Thread-safe: two paths of one engine pin for their positions while another gives an idle conversation's back.
class GranuleArena {
public:
    GranuleArena() = default;
    GranuleArena(const GranuleArena&) = delete;
    GranuleArena& operator=(const GranuleArena&) = delete;
    ~GranuleArena() { close(); }

    // Reserve `total` bytes of address space with granules of `granule` (rounded up to whole 4 KB pages, the least
    // being one page).  No RAM is committed and nothing is pinned: only the mapping's address range is taken.
    bool open(uint64_t total, uint64_t granule, std::string& err) {
        if (total == 0 || base_ != nullptr) {
            err = "granule arena: nothing to reserve" + std::string(base_ != nullptr ? " (already open)" : "");
            return false;
        }
        granule = (granule == 0 ? granule_detail::kPageSize : granule);
        granule = (granule + granule_detail::kPageSize - 1) / granule_detail::kPageSize * granule_detail::kPageSize;
        void* b = reserve_(total);
        if (b == nullptr) {
            err = "granule arena: cannot reserve " + std::to_string(total >> 20) + " MB of address space";
            return false;
        }
        base_ = (uint8_t*) b;
        total_ = total;
        granule_ = granule;
        state_.assign((size_t) ((total + granule - 1) / granule), 0);
        return true;
    }

    // Everything ensure() has pinned so far, in bytes (what the log's "pinned now" and the VramReport's kv_ram carry).
    uint64_t pinned_bytes() const {
        std::lock_guard<std::mutex> lk(mu_);
        return pinned_;
    }
    // The four figures below read without the mutex, on purpose: total_, granule_ and base_ are written only by
    // open() (before the arena is shared) and by close() (once nothing uses it any more), so any read of them -
    // from a pinning thread or a releasing one - sees either the open() value or a null/zero closed arena, and
    // both are answers.  (The thread-safety claim above is by construction, not by test: one non-reentrant mutex
    // serializes ensure()/release()/pinned_at(), and the Pinner's callbacks are driver calls that never come
    // back into the arena.)
    uint64_t reserved_bytes() const { return total_; }
    uint64_t granule_bytes() const { return granule_; }
    // Never null once open() succeeded: the addresses never move, pinned or not.
    uint8_t* base() const { return base_; }
    bool empty() const { return base_ == nullptr; }
    // A test's eye: are the bytes at this offset pinned?
    bool pinned_at(uint64_t off) const {
        std::lock_guard<std::mutex> lk(mu_);
        return base_ != nullptr && off < total_ && state_[(size_t) (off / granule_)] != 0;
    }

    // Pin every granule that [off, off+len) touches, so a device may read or write that range.  Already-pinned
    // granules cost a byte of bookkeeping; the rest cost one Pinner::pin each, zero-filled and backed first.  False
    // with err: the range runs past the arena or the driver refused to pin (the WSL cap of about 1 GB pinned in all,
    // a host out of RAM) - what was pinned by this call before the refusal stays pinned, the caller's request ends.
    bool ensure(uint64_t off, uint64_t len, Pinner& p, std::string& err) {
        std::lock_guard<std::mutex> lk(mu_);
        if (base_ == nullptr) { err = "granule arena: not open"; return false; }
        if (len == 0) return true;
        if (off >= total_ || off + len > total_) {
            err = "granule arena: " + std::to_string(len) + " bytes at " + std::to_string(off) +
                  " run past the arena (" + std::to_string(total_) + ")";
            return false;
        }
        const uint64_t g0 = off / granule_, g1 = (off + len - 1) / granule_;
        for (uint64_t i = g0; i <= g1; ++i) {
            if (state_[(size_t) i] != 0) continue;
            const uint64_t at = i * granule_;
            const uint64_t bytes = at + granule_ <= total_ ? granule_ : total_ - at;
            back_(base_ + at, bytes);
            if (!p.pin(base_ + at, bytes)) {
                unback_(base_ + at, bytes);
                err = "granule arena: the driver refused to pin " + std::to_string(bytes >> 20) + " MB at +" +
                      std::to_string(at >> 20) + " MB";
                return false;
            }
            state_[(size_t) i] = 1;
            pinned_ += bytes;
        }
        return true;
    }

    // Give every pinned granule back: unregister, then hand its pages to the kernel.  The addresses stay, so the
    // pointers into the arena stay valid and a later ensure() re-pins (zero-filled) whatever the next writer needs.
    // The caller must have nothing in flight that touches the arena - its own stream/device synchronize first.
    void release(Pinner& p) {
        std::lock_guard<std::mutex> lk(mu_);
        if (base_ == nullptr) return;
        for (size_t i = 0; i < state_.size(); ++i) {
            if (state_[i] == 0) continue;
            const uint64_t at = (uint64_t) i * granule_;
            const uint64_t bytes = at + granule_ <= total_ ? granule_ : total_ - at;
            p.unpin(base_ + at, bytes);
            unback_(base_ + at, bytes);
            state_[i] = 0;
        }
        pinned_ = 0;
    }

    // The whole arena, mapping and all.  This does NOT release first - unregistering is the Pinner's business
    // and close() takes none: a caller that still holds pins must call release(p) before close() (or before this
    // object dies), or those granules stay registered on a range that is about to be unmapped.  The engine's
    // kv_stream_free() does exactly that; a ~GranuleArena over a still-pinned arena leaks the registrations.
    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        if (base_ == nullptr) return;
        unmap_(base_, total_);
        base_ = nullptr;
        total_ = 0;
        pinned_ = 0;
        granule_ = 0;
        state_.clear();
    }

private:
    void* reserve_(uint64_t bytes) {
#if defined(_WIN32)
        return ::VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE, PAGE_READWRITE);
#else
        void* m = ::mmap(nullptr, (size_t) bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                         -1, 0);
        return m == MAP_FAILED ? nullptr : m;
#endif
    }
    void unmap_(void* p, uint64_t bytes) {
#if defined(_WIN32)
        (void) bytes;
        ::VirtualFree(p, 0, MEM_RELEASE);
#else
        ::munmap(p, (size_t) bytes);
#endif
    }
    // Pages the driver is about to pin must exist and be zeroed (a cudaHostAlloc'd arena starts zeroed; a reader of a
    // position this engine has not written must still read what it has not written as zero, as it did before).
    void back_(void* p, uint64_t bytes) {
#if defined(MADV_POPULATE_WRITE)
        if (::madvise(p, (size_t) bytes, MADV_POPULATE_WRITE) == 0) return;   // the kernel zeroes it faster than we can
#endif
#if defined(_WIN32)
        ::VirtualAlloc(p, (SIZE_T) bytes, MEM_COMMIT, PAGE_READWRITE);
#else
        std::memset(p, 0, (size_t) bytes);
#endif
    }
    // ... and back to nothing: the RAM is given up, the address stays reserved.
    void unback_(void* p, uint64_t bytes) {
#if defined(_WIN32)
        ::VirtualFree(p, (SIZE_T) bytes, MEM_DECOMMIT);
#elif defined(MADV_DONTNEED)
        ::madvise(p, (size_t) bytes, MADV_DONTNEED);
#else
        (void) p, (void) bytes;
#endif
    }

    mutable std::mutex mu_;
    uint8_t* base_ = nullptr;
    uint64_t total_ = 0, granule_ = 0, pinned_ = 0;
    std::vector<uint8_t> state_;   // per granule: pinned
};

}  // namespace maya::core
