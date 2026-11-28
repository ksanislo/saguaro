// metal/src/engine/stubs_saguaro.cpp - the entry points Saguaro's engines/qfn added or re-signed after it was taken from
// Strata 0.1.49, which the Metal engine does not implement.  Like stubs_144.cpp: each either stops the engine with its name
// (strata::metal_engine::missing) or, where the caller has a documented fallback, answers "not available" / does nothing.
//
// None is reached with the Metal defaults - the Metal window (src/engine/window.cpp) runs decode and prompt itself:
//   * the router, QSA indexer and mapped-row copies gained Turbo / M-RoPE arguments; their Metal side was a stub before
//     (stubs.cpp, the old signatures) and stays one;
//   * SAGUARO_YARN_ELASTIC's re-rotation (yarn_step.hpp, rope.hpp's table builder): the Mac config must not set it;
//   * M-RoPE scopes and per-slot rope tables: the window keeps its own rope table (setup_host.cpp's rope_table_set);
//   * SAGUARO_MMQ_SWIGLU_Q8, SAGUARO_BF16_TC's range counters and the Turing Q8_0 report: CUDA / HIP only.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/yarn_step.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

namespace strata::metal_engine {
void missing(const char* name);
}

namespace strata::kernels {
using strata::metal_engine::missing;

// ---- re-signed: Turbo routing, M-RoPE positions (were stubs on Metal before)
void native_router_top10(const float*, int32_t*, float*, void*, const TurboRoute*) { missing("strata::kernels::native_router_top10"); }
void native_router_top10_multi(const float*, int32_t*, float*, int, void*, const TurboRoute*) {
    missing("strata::kernels::native_router_top10_multi");
}
void turbo_drop_apply(int32_t*, float*, int, int, const TurboRoute&, void*) { missing("strata::kernels::turbo_drop_apply"); }
void native_qsa_indexer_append(const float*, const int32_t*, int32_t, const float*, float, const QsaIndexerBuffers&, const QsaShapes&,
                               int64_t, const RopeScaling&, void*, const int32_t*) {
    missing("strata::kernels::native_qsa_indexer_append");
}
void native_qsa_indexer_append_steps(const float*, const int32_t*, int, int, int32_t, const float*, float, const QsaIndexerBuffers&,
                                     const QsaShapes&, int64_t, const RopeScaling&, void*, const int32_t*) {
    missing("strata::kernels::native_qsa_indexer_append_steps");
}
void copy_rows_from_mapped(float*, const float*, int64_t, int64_t, const int32_t*, const int32_t*, void*, const int32_t*, int) {
    missing("strata::kernels::copy_rows_from_mapped");
}
void copy_rows_from_mapped_after(const uint32_t*, uint32_t, float*, const float*, int64_t, int64_t, const int32_t*, const int32_t*,
                                 void*, const int32_t*, int) {
    missing("strata::kernels::copy_rows_from_mapped_after");
}

// ---- rope tables: the Metal window keeps its own
void rope_table_set(const float*, const float*, int, const RopeScaling&, int) {}
void rope_table_force(bool) {}
MropeScope::MropeScope(const int32_t*) {}
MropeScope::~MropeScope() {}
void slot_rope_enable(bool) {}
bool slot_rope_enabled() { return false; }
void slot_rope_set(int, std::shared_ptr<const SlotRope>) {}
std::shared_ptr<const SlotRope> slot_rope_get(int) { return nullptr; }

// ---- SAGUARO_YARN_ELASTIC's re-rotation: CUDA / HIP only
void build_rope_table_range(int, const RopeScaling&, int, int, float*, float*) { missing("strata::kernels::build_rope_table_range"); }
void rope_plain_rows(float*, int64_t, int64_t, float, void*) { missing("strata::kernels::rope_plain_rows"); }
void kv_q8_rerotate(int8_t*, uint16_t*, const int32_t*, int64_t, int64_t, int64_t, int64_t, const float*, const QsaShapes&, void*,
                    const int32_t*, int64_t) {
    missing("strata::kernels::kv_q8_rerotate");
}
void kv_f16_rerotate(uint16_t*, const int32_t*, int64_t, int64_t, int64_t, int64_t, const float*, const QsaShapes&, void*,
                     const int32_t*, int64_t) {
    missing("strata::kernels::kv_f16_rerotate");
}
void idx_pooled_rerotate(float*, int64_t, int64_t, int64_t, int64_t, const float*, int, void*, const int32_t*, int64_t) {
    missing("strata::kernels::idx_pooled_rerotate");
}

// ---- CUDA-only reports
void native_q8_0_turing_report() {}
}  // namespace strata::kernels

namespace strata::prefill {
bool gemm_tc_range_read(float*, unsigned long long*, unsigned long long*, unsigned long long*, bool) { return false; }
namespace mmq {
bool swiglu_q8_requested() { return false; }
bool swiglu_quantize(const float*, void*, int, int64_t, int64_t, bool, const int32_t*, int, const float*, int64_t, void*) { return false; }
}  // namespace mmq
}  // namespace strata::prefill
