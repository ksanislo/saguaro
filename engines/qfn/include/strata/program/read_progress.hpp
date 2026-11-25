#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

// U5b (PLAN-u5b-split-progress): ONE progress figure per reading request, shared by every host point of that read.
// On a layer split each card's host points report their own piece count: the first cards are several pieces ahead of
// the last one, so a figure built from the reporting point alone jumps backwards from beat to beat (4 x T4, a 38910
// token read: done printed 14592, 19456, 14592; 34048, 29184).  This collects the latest host point of every card
// and reports the tokens whose reading is complete, counted as tokens x (layers done / all layers): a token the
// first card has passed and the last has not counts in part.  The figure never decreases and ends at `new` when
// the read ends (report_full of the last stage's chunk end).  Thread-safe: the stages' pipeline threads report from
// their own host points.  One card: its layers are all layers and its own chunk starts are the only positions
// known, so the figure is exactly the position its host points report - a single card prints what it prints today.
//
// Card identity is the stage's first layer (its `lb`, unique across a split's stages): the callback of a queued job
// runs on every stage's thread and carries no stage index, so the layer range is what tells the cards apart.
namespace strata::program {

class ReadProgress {
  public:
    /// The read covers prompt tokens [from, upto); every card together spans `total_layers` layers.
    void reset(int64_t from, int64_t upto, int64_t total_layers) {
        std::lock_guard<std::mutex> lk(mu_);
        from_ = from;
        upto_ = upto;
        total_ = total_layers;
        best_ = 0;
        full_ = 0;
        card_.clear();
        // (no end sentinel on purpose: a chunk end no report revealed stays unknown, so a card alone at the head
        //  of the pipeline gets no part-credit for the chunk it is in - exactly the figure it printed before U5b)
        bounds_.clear();
    }

    /// A host point between the layer passes: card `lb` has finished `pass` of its `passes` layers while working
    /// on the chunk that starts at the absolute position `pos0` (every token below it that card has read).
    void report(int64_t lb, int64_t pos0, int64_t pass, int64_t passes) {
        std::lock_guard<std::mutex> lk(mu_);
        if (upto_ <= from_ || total_ <= 0 || passes <= 0) return;
        Card c;
        c.p0 = clamp_(pos0);
        c.pass = std::max<int64_t>(0, std::min(pass, passes));
        c.passes = passes;
        card_[lb] = c;
        mark_bound_(c.p0);   // a position any card reaches is a real chunk boundary (same chunk sequence for all)
        refresh_();
    }

    /// Tokens below `pos` have passed every card (the last stage finished a chunk there): full credit.
    void report_full(int64_t pos) {
        std::lock_guard<std::mutex> lk(mu_);
        if (upto_ <= from_ || total_ <= 0) return;
        full_ = std::max(full_, clamp_(pos));
        mark_bound_(full_);
        refresh_();
    }

    /// The tokens of this read whose reading is complete: never decreasing, within [0, new].
    int64_t done() const {
        std::lock_guard<std::mutex> lk(mu_);
        return best_;
    }

  private:
    struct Card {
        int64_t p0 = 0, pass = 0, passes = 0;   // p0 relative to `from_`
    };

    int64_t clamp_(int64_t pos) const { return std::max<int64_t>(0, std::min(pos - from_, upto_ - from_)); }

    /// Keep `pos` (relative) as a known chunk boundary, ascending and free of duplicates.  A position a card
    /// reaches, or the last stage's chunk end, is on the chunk sequence every card follows, so it stays a true
    /// boundary even after the card that reported it moves on - that is what bounds a lagging card's part-credit
    /// to the chunk it is really in.
    void mark_bound_(int64_t pos) {
        const auto it = std::lower_bound(bounds_.begin(), bounds_.end(), pos);
        if (it != bounds_.end() && *it == pos) return;
        bounds_.insert(it, pos);
    }

    /// The sum over the cards of the layers each gives every token: `passes` for each token below the card's
    /// position (it has read them) and `pass` for each token of the part of its current chunk that runs to the
    /// next boundary another report revealed (no such boundary: the card is alone at the head and counts no part
    /// of its own chunk - what a single card printed before).  Divided over all layers; `full_` floors it.
    void refresh_() {
        double sum = 0.0;
        for (const auto& [lb, c] : card_) {
            const auto it = std::upper_bound(bounds_.begin(), bounds_.end(), c.p0);
            const int64_t b = it == bounds_.end() ? c.p0 : *it;   // its current chunk's end, if some report gave it
            sum += (double) c.p0 * (double) c.passes + (double) (b - c.p0) * (double) c.pass;
        }
        double v = std::max<double>((double) full_, sum / (double) total_);
        const int64_t newn = upto_ - from_;
        int64_t now = (int64_t) (v + 0.5);
        now = std::max<int64_t>(0, std::min(now, newn));
        if (now > best_) best_ = now;
    }

    mutable std::mutex mu_;
    int64_t from_ = 0, upto_ = 0, total_ = 0;   // the read's range and its total layers
    int64_t best_ = 0, full_ = 0;               // relative to from_
    std::map<int64_t, Card> card_;              // card -> its latest host point
    std::vector<int64_t> bounds_;               // chunk boundaries reported so far (relative, ascending)
};

}  // namespace strata::program
