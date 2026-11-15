// include/strata/kernels/logprobs.hpp - the GLM engine's token log-probabilities (GEN logprobs=<k>, the server's OpenAI
// logprobs / top_logprobs), in the DeepSeek engine's rows and LP lines (src/ds4/ds4_logprobs.hpp: same semantics -
// softmax at temperature 1 after the request's penalties, before every cut).  Glm5Model::pick_tokens writes each
// pick's rows into the request's sink; the serve loop prints a token's row after its T line.
#pragma once

#include "../../../src/ds4/ds4_logprobs.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace strata::kernels {

using strata::ds4::LogprobRow;
constexpr int kTopLogprobs = strata::ds4::kTopLogprobs;

/// A request's rows: a ring in mapped pinned memory the kernel writes; each row's key is the position of the token it
/// picks (its history's length + the drafts before it), tagged with the sink's use so a late write of an earlier
/// request never passes for this one's.
struct LpSink {
    static constexpr int kRing = 256;
    int k = 0;                       // top alternatives (0 .. kTopLogprobs)
    uint64_t tag = 0;                // (key's high bits: this use of the sink)
    LogprobRow* rows = nullptr;      // [kRing], mapped: the kernel's
    int64_t* keys = nullptr;         // [kRing], mapped: the kernel's copy of each row's key (written with the row)
    int64_t want[kRing];             // host: the key each slot was queued for
    uint64_t seq[kRing];             // host: the order slots were queued in (the newest row of a key wins)
    uint64_t n = 0;
    int next = 0;
    std::mutex mu;
    int64_t key(int64_t pos) const { return (int64_t) (tag << 40) | pos; }
};

/// A sink for a request with logprobs=<k> (from a pool: pinned memory is allocated once), nullptr + err when the
/// pinned memory did not allocate; lp_sink_put gives it back.
LpSink* lp_sink_get(int k, std::string& err);
void lp_sink_put(LpSink* s);
struct LpPut {
    void operator()(LpSink* s) const { lp_sink_put(s); }
};
using LpHold = std::unique_ptr<LpSink, LpPut>;

/// The rows of a pick on `stream` (after its sample_tokens, which leaves `logits` as they were): T rows of n_vocab,
/// hist / h the penalty rows the pick counted (nullptr / 0: none), ids the picked tokens (device), row t keyed
/// pos0 + t.  Host-side bookkeeping under the sink's lock; the rows land once the stream passes the kernel.
void logprob_pick(LpSink* s, const float* logits, int T, int n_vocab, const int* hist, int h, float repeat, float freq,
                  float present, const int* ids, int64_t pos0, void* stream);

/// The LP line (without newline; rid < 0: untagged) for token `tok` at position `pos`, "" when the sink holds no row
/// of that position for that token (a forced token: the thinking budget, the loop guard).  Call after the pick's wait.
std::string logprob_line(LpSink* s, long long rid, int64_t pos, int tok);

}  // namespace strata::kernels
