// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef POLICY_QUIET_H_INCLUDED
#define POLICY_QUIET_H_INCLUDED

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <vector>
#include "movegen.h"

namespace Stockfish {

// A policy vote among plausible history candidates, not a second move generator.
// At depth >= 5 MovePicker's -3560 * depth sort covers the entire good-quiet prefix.
struct QuietPolicyCandidates {
    static constexpr int Limit = 8;
    static constexpr int Prefix = 3;
    static constexpr int HistoryWindow = 2048;
    std::array<ExtMove*, Limit> slots{};
    std::array<float, Limit> logits{};
    int count = 0;

    void collect(ExtMove* begin, ExtMove* end, Move ttMove, int goodQuietThreshold,
                 int limit = Limit, int window = HistoryWindow) {
        count = 0;
        for (auto* it = begin; it != end && count < limit; ++it) {
            if (it->value <= goodQuietThreshold) break;
            if (*it == ttMove) continue;
            assert(!count || it->value <= slots[count - 1]->value);
            if (count && slots[0]->value - it->value > window) break;
            slots[count++] = it;
        }
    }

    // At most one swap. A tie leaves Stockfish's history leader in front.
    int promote_pair(float temperature, int bonus) const {
        assert(count == 2);
        const float advantage = (logits[1] - logits[0]) / temperature;
        if (!(bonus * std::min(advantage, 1.0f) > slots[0]->value - slots[1]->value))
            return 0;
        std::swap(*slots[0], *slots[1]);
        return 1;
    }

    // Stable top-three extraction. The rest keeps its original relative order;
    // every original ExtMove (including its raw history value) is retained.
    int promote(float temperature) const {
        if (count < 2) return 0;
        // Most shortlists contain just two quiets in descending history order.
        // Only the leader can receive a penalty when the challenger has the
        // higher logit. Use the same rounded score and stable tie as below.
        if (count == 2) {
            if (!std::isfinite(logits[0]) || !std::isfinite(logits[1]) || logits[1] <= logits[0])
                return 0;
            const float leader = float(slots[0]->value) + 1024.0f
              * std::clamp((logits[0] - logits[1]) / temperature, -2.0f, 0.0f);
            if (!(float(slots[1]->value) > leader)) return 0;
            std::swap(*slots[0], *slots[1]);
            return 1;
        }
        float maximum = logits[0];
        for (int i = 0; i < count; ++i) {
            if (!std::isfinite(logits[i])) return 0;
            maximum = std::max(maximum, logits[i]);
        }
        std::array<float, Limit> scores;
        std::array<ExtMove, Limit> original;
        std::array<int, Limit> order;
        for (int i = 0; i < count; ++i) {
            original[i] = *slots[i];
            order[i] = i;
            scores[i] = float(original[i].value) + 1024.0f
              * std::clamp((logits[i] - maximum) / temperature, -2.0f, 0.0f);
        }
        int promotions = 0;
        for (int i = 0; i < std::min(Prefix, count); ++i) {
            int best = i;
            for (int j = i + 1; j < count; ++j)
                if (scores[order[j]] > scores[order[best]]) best = j;
            if (best != i) {
                const int chosen = order[best];
                std::move_backward(order.begin() + i, order.begin() + best,
                                   order.begin() + best + 1);
                order[i] = chosen;
                ++promotions;
            }
        }
        if (promotions)
            for (int i = 0; i < count; ++i) *slots[i] = original[order[i]];
        return promotions;
    }
};

// Search-local. MovePicker may permute the ExtMove slots after these logits
// were scored, so retain move identities for just the actionable margins.
class QuietPolicyReductions {
    struct Penalty { Move move; i16 units; };
    std::array<Penalty, QuietPolicyCandidates::Limit> penalties;

   public:
    int count = 0;
    QuietPolicyReductions() {}  // Only the first count entries are live.

    void record(const QuietPolicyCandidates& list) {
        count = 0;
        if (list.count < 2) return;
        const float best = *std::max_element(list.logits.begin(), list.logits.begin() + list.count);
        for (int i = 0; i < list.count; ++i) {
            const float gap = best - list.logits[i];  // finite source logits
            if (gap <= 1.0f) continue;
            const int units = 2 * (int(256 * std::min(gap, 3.0f)) - 256);
            if (units) penalties[count++] = {*list.slots[i], i16(units)};
        }
    }

    int adjustment(Move move, bool improving) const {
        for (int i = 0; i < count; ++i)
            if (penalties[i].move == move)
                return penalties[i].units / (1 + int(improving));
        return 0;
    }
};

// Cache raw move logits, never history-blended ordering or node-specific choices.
// A later visit may have different history scores or a different shortlist.
class QuietPolicyCache {
    static constexpr int Size = 256;
    static constexpr int MovesPerEntry = 16;
    struct Entry {
        Key key = 0;
        int count = 0, replace = 0;
        struct Score { Move move; float logit; };
        std::array<Score, MovesPerEntry> scores;
    };
    std::vector<Entry> entries;
    u64 generation = 0;

public:
    void prepare(bool enabled, u64 modelGeneration) {
        if (!enabled) reset(false);
        else if (entries.empty() || generation != modelGeneration) {
            reset(true);
            generation = modelGeneration;
        }
    }
    void reset(bool enabled) {
        if (!enabled) { entries.clear(); return; }
        entries.resize(Size);
        for (auto& entry : entries) entry.count = entry.replace = 0;
    }
    bool contains(Key key) const {
        if (entries.empty()) return false;
        const auto& entry = entries[key & (Size - 1)];
        return entry.count && entry.key == key;
    }
    bool get(Key key, Move move, float& logit) const {
        if (entries.empty()) return false;
        const auto& entry = entries[key & (Size - 1)];
        if (entry.key != key) return false;
        for (int i = 0; i < entry.count; ++i)
            if (entry.scores[i].move == move) {
                logit = entry.scores[i].logit;
                return true;
            }
        return false;
    }
    int get_candidates(Key key, QuietPolicyCandidates& candidates,
                       std::array<bool, QuietPolicyCandidates::Limit>& missing) const {
        std::fill_n(missing.begin(), candidates.count, true);
        if (entries.empty()) return candidates.count;
        const auto& entry = entries[key & (Size - 1)];
        if (entry.key != key) return candidates.count;
        int remaining = candidates.count;
        for (int j = 0; j < entry.count && remaining; ++j)
            for (int i = 0; i < candidates.count; ++i)
                if (missing[i] && entry.scores[j].move == *candidates.slots[i]) {
                    candidates.logits[i] = entry.scores[j].logit;
                    missing[i] = false;
                    --remaining;
                    break;
                }
        return remaining;
    }
    void put(Key key, Move move, float logit) {
        if (entries.empty()) return;
        auto& entry = entries[key & (Size - 1)];
        if (entry.key != key) {
            entry.key = key;
            entry.count = entry.replace = 0;
        }
        for (int i = 0; i < entry.count; ++i)
            if (entry.scores[i].move == move) {
                entry.scores[i].logit = logit;
                return;
            }
        const int slot = entry.count < MovesPerEntry ? entry.count++ : entry.replace;
        entry.scores[slot] = {move, logit};
        entry.replace = (slot + 1) % MovesPerEntry;
    }
};

// Fixed work accounting, not a wall-clock timer in the search hot path.
// Eight output heads plus a context allowance cost one 256-unit credit.
// A wider network or more missing heads consumes more of the same budget.
inline u64 quiet_policy_cost(int width, int missingHeads) {
    return missingHeads ? u64(width) * (8 + missingHeads) : 0;
}
inline u64 quiet_policy_allowance(u64 nodes) {
    return 4096 * (1 + nodes / 256);
}

} // namespace Stockfish
#endif
