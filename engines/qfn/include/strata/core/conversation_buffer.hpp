#pragma once

#include "strata/core/progress.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::core {

// The pool of whole 16 MiB snapshot segments (used by ConversationBuffer's allocator below, read-stall 11-19):
// parks and drops no longer hand these blocks back to the allocator every switch - the pages stay mapped and the
// next park's copy is their first writer.  SAGUARO_PARK_POOL_MIB bounds what it holds (default 8192, 0: off).
// The freed blocks are not counted as cache RAM (they are invisible to the cache's own bytes()); the pool's cap is
// the guard, and the RAM admission checks keep running on real availability.
inline size_t parked_block_pool_cap() {
    static const size_t cap = [] {
        const char* e = std::getenv("SAGUARO_PARK_POOL_MIB");
        const size_t mib = e != nullptr ? (size_t) std::atoll(e) : (size_t) 8192;
        return mib << 20;
    }();
    return cap;
}
struct ParkedBlockPool {
    static constexpr size_t block = 16u << 20;
    std::mutex mu;
    std::vector<void*> free;
    size_t held = 0;
};
inline ParkedBlockPool& parked_block_pool() {
    static ParkedBlockPool p;
    return p;
}
inline void* parked_block_pool_take() {
    ParkedBlockPool& P = parked_block_pool();
    std::lock_guard<std::mutex> lk(P.mu);
    if (P.free.empty()) return nullptr;
    void* p = P.free.back();
    P.free.pop_back();
    P.held -= ParkedBlockPool::block;
    return p;
}
inline bool parked_block_pool_give(void* p) {
    ParkedBlockPool& P = parked_block_pool();
    std::lock_guard<std::mutex> lk(P.mu);
    if (P.held + ParkedBlockPool::block > parked_block_pool_cap()) return false;
    P.free.push_back(p);
    P.held += ParkedBlockPool::block;
    return true;
}

// Gives pooled blocks back to the allocator until at most `keep_bytes` are held (memory pressure: the pool's pages
// are RAM nobody counts).  Returns the bytes released.
inline size_t parked_block_pool_trim(size_t keep_bytes) {
    ParkedBlockPool& P = parked_block_pool();
    std::vector<void*> out;
    {
        std::lock_guard<std::mutex> lk(P.mu);
        while (P.held > keep_bytes && !P.free.empty()) {
            out.push_back(P.free.back());
            P.free.pop_back();
            P.held -= ParkedBlockPool::block;
        }
    }
    for (void* p : out) std::allocator<uint8_t>().deallocate(static_cast<uint8_t*>(p), ParkedBlockPool::block);
    return out.size() * ParkedBlockPool::block;
}
inline size_t parked_block_pool_held() {
    ParkedBlockPool& P = parked_block_pool();
    std::lock_guard<std::mutex> lk(P.mu);
    return P.held;
}

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
// A VIEW (session files read through a mapping, low-RAM restores) reads bytes it does not own: it counts no RAM
// (bytes() == 0), a read visit hands out the mapped bytes, a writing visit fails, and resizing copies it first.
// An ALIAS (PLAN-p124a2: a park taking over a turn save's host K/V) is a view whose bytes are RAM another owner
// holds pinned in RAM - not file cache the OS can hand back - so it counts its full length in bytes() while the
// keeping owner exists. Same read-only rules as a view.
class ConversationBuffer {
public:
    static ConversationBuffer view(const uint8_t* data, size_t n, std::shared_ptr<const void> keep) {
        ConversationBuffer b;
        b.ext_ = data;
        b.size_ = n;
        b.keep_ = std::move(keep);
        return b;
    }
    static ConversationBuffer alias(const uint8_t* data, size_t n, std::shared_ptr<const void> keep) {
        ConversationBuffer b;
        b.ext_ = data;
        b.size_ = n;
        b.keep_ = std::move(keep);
        b.charge_ = n;
        return b;
    }
    bool external() const { return ext_ != nullptr; }
    // PLAN-p124a2: the single block of an owned buffer's bytes, for an alias: nullptr when empty, when this is
    // already a view, or when the payload is split across segments (an alias needs one contiguous run).
    const uint8_t* contiguous() const {
        if (ext_ || segments_.size() != 1) return nullptr;
        return segments_.front().data();
    }
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)),
          ext_(std::exchange(other.ext_, nullptr)), keep_(std::move(other.keep_)),
          charge_(std::exchange(other.charge_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        ext_ = std::exchange(other.ext_, nullptr);
        keep_ = std::move(other.keep_);
        charge_ = std::exchange(other.charge_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        if (ext_) return charge_;   // a mapping's pages are the OS file cache's (charge 0); an alias's are held RAM
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        if (ext_) return std::max(n, size_);   // a resize copies the view into owned segments first
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0, size_))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (ext_) own();
        if (n > size_) {
            const size_t held = size_;
            const bool growing = held != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                segments_.back().resize(segments_.back().size() + extend, value);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing, held));
                segment.resize(count, value);
                segments_.push_back(std::move(segment));
                size_ += count;
                progress_beat();   // #2049: zero-filling 16 MiB of fresh pages is slow when the host is paging
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }
    // resize() without the fill: bytes past the old size are left as the allocator gives them (untouched pages, so
    // the caller's copy is their first touch).  For a caller that overwrites every one of them before any read.
    void resize_for_overwrite(size_t n) {
        if (ext_ || n <= size_) { resize(n); return; }
        const size_t held = size_;
        const bool growing = held != 0;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extend = std::min(n - size_, spare);
        const size_t extra = n - size_ - extend;
        segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
        if (extend) {
            segments_.back().resize(segments_.back().size() + extend);
            size_ += extend;
        }
        while (size_ < n) {
            const size_t count = std::min(segment_bytes, n - size_);
            const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
            Segment segment;
            segment.reserve(segment_capacity(count, previous, growing, held));
            segment.resize(count);   // default-init: no write
            segments_.push_back(std::move(segment));
            size_ += count;
        }
    }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        if (ext_ || other.ext_) {   // a view on either side: compare through reads
            std::vector<uint8_t> a(std::min(size_, segment_bytes)), b(a.size());
            for (size_t at = 0; at < size_; at += a.size()) {
                const size_t n = std::min(a.size(), size_ - at);
                if (!read(a.data(), at, n) || !other.read(b.data(), at, n) || std::memcmp(a.data(), b.data(), n))
                    return false;
            }
            return true;
        }
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    // value-initializes only when asked: resize(n, v) fills as before, resize(n) (resize_for_overwrite) leaves the bytes
    template<class T> struct DefaultInit : std::allocator<T> {
        template<class U> struct rebind { using other = DefaultInit<U>; };
        DefaultInit() = default;
        template<class U> DefaultInit(const DefaultInit<U>&) noexcept {}
        template<class U> void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) { ::new ((void*) p) U; }
        template<class U, class... A> void construct(U* p, A&&... a) { ::new ((void*) p) U(std::forward<A>(a)...); }
        // The read-stall work (11-19): a park allocates and a drop frees ~16 MiB segments every conversation switch,
        // and fresh pages cost faults (the single-threaded 3-5 s parks).  Whole 16 MiB blocks round-trip through a
        // small process-wide pool so the pages stay mapped; the cap bounds the RAM it holds (SAGUARO_PARK_POOL_MIB,
        // 0: off).  Only this buffer's exact segment size is pooled; everything else allocates as before.
        T* allocate(size_t n) {
            if (n * sizeof(T) == segment_bytes && parked_block_pool_cap() > 0)
                if (void* p = parked_block_pool_take()) return static_cast<T*>(p);
            return std::allocator<T>::allocate(n);
        }
        void deallocate(T* p, size_t n) {
            if (n * sizeof(T) != segment_bytes || !parked_block_pool_give(p))
                std::allocator<T>::deallocate(p, n);
        }
    };
    using Segment = std::vector<uint8_t, DefaultInit<uint8_t>>;
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    // The reserve is also bounded by an eighth of what the buffer held before
    // the append (at least 64 KiB): a parked conversation carries it in every
    // K/V buffer, and at up to 16 MiB each it came to 0.4-0.5 GB of a ~1 GB
    // conversation (Qwen3.8-Flash-Next), counted against the cache's budget.
    static size_t segment_capacity(size_t count, size_t previous, bool growing, size_t held) {
        if (!growing) return count;
        const size_t geometric = std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2));
        return std::max(count, std::min(geometric, std::max<size_t>(65536, held / 8)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    // a view becomes owned segments (a resize of a view)
    void own() {
        const uint8_t* src = ext_;
        const size_t n = size_;
        ext_ = nullptr;
        size_ = 0;
        segments_.clear();
        resize(n);
        visit(0, n, [&](uint8_t* p, size_t c, size_t at) { std::memcpy(p, src + at, c); return true; });
        keep_.reset();
        charge_ = 0;   // owned bytes from here: bytes() counts the segments themselves
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        if (self.ext_) {
            if constexpr (!std::is_const_v<Self>) {
                return false;   // a view is read-only: writing into it fails (the caller copies or captures fresh)
            } else {
                while (count) {   // in segment-sized steps, as the owned storage hands them out
                    const size_t n = std::min(count, segment_bytes);
                    if (!fn(self.ext_ + offset, n, offset)) return false;
                    offset += n; count -= n;
                }
                return true;
            }
        }
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
    const uint8_t* ext_ = nullptr;      // a view's or alias's bytes (none: owned segments)
    std::shared_ptr<const void> keep_;  // keeps a view's mapping, or an alias's holding owner, alive
    size_t charge_ = 0;                 // an alias counts these bytes as RAM (a file mapping's view: 0)
};

} // namespace strata::core
