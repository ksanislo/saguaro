#include "strata/core/devnotes.hpp"

#include <cuda_runtime.h>

#include <mutex>

namespace strata::core {
namespace {
std::mutex g_notes_mu;
std::vector<DeviceNoteRec> g_notes;

uint64_t card_free() {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    return (uint64_t) f;
}
}  // namespace

DeviceNote::DeviceNote(const char* what_, uint64_t bytes_) : what(what_), bytes(bytes_), free0(card_free()) {}

DeviceNote::~DeviceNote() {
    const uint64_t f1 = card_free();
    const int64_t used = (free0 != 0 && f1 != 0) ? (int64_t) (free0 - f1) : 0;
    std::lock_guard<std::mutex> lk(g_notes_mu);
    for (DeviceNoteRec& r : g_notes)
        if (r.what == what) {
            r.bytes += bytes;
            r.used += used;
            ++r.n;
            return;
        }
    g_notes.push_back({what, bytes, used, 1});
}

std::vector<DeviceNoteRec> device_notes_taken() {
    std::lock_guard<std::mutex> lk(g_notes_mu);
    std::vector<DeviceNoteRec> out;
    out.swap(g_notes);
    return out;
}

}  // namespace strata::core
