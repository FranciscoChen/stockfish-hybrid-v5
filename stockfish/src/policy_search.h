// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef POLICY_SEARCH_H_INCLUDED
#define POLICY_SEARCH_H_INCLUDED

#include <array>
#include "types.h"

namespace Stockfish {

struct PolicySearchMove {
    Move move = Move::none();
    float probability = 0.0f;
};

struct PolicySearchMoves {
    std::array<PolicySearchMove, MAX_MOVES> moves{};
    int count = 0;

    float probability(Move move) const {
        for (int i = 0; i < count; ++i)
            if (moves[i].move == move) return moves[i].probability;
        return 0.0f;
    }
};

struct PolicySearchCacheEntry {
    Key key = 0;
    bool valid = false;
    PolicySearchMoves list;
};

struct PolicySearchCounters {
    // Completed first-quiet searches at expected cut nodes, split by material
    // bucket and whether the challenger displaced the history leader.
    struct CutoffSample { u64 searched = 0, cutoffs = 0, nodes = 0, researches = 0; };
    std::array<std::array<CutoffSample, 2>, 4> cutoffSamples{};
    u64 nodes = 0, rootVisits = 0, inferences = 0, cacheHits = 0, forced = 0;
    u64 lmrProtected = 0;
    u64 marginLmrReduced = 0, marginLmrFailHigh = 0, marginLmrResearched = 0;
    u64 delayedFirstSearched = 0, delayedFirstFailed = 0, delayedShortlists = 0, delayedCalls = 0;
    u64 delayedNewLeaders = 0, delayedLeaderSearched = 0, delayedLeaderCutoffs = 0;
    u64 quietCalls = 0, quietBudgetSkipped = 0, quietHistorySkipped = 0;
    u64 quietCacheHits = 0, quietHeadCacheHits = 0, quietOutputScores = 0;
    u64 quietNodeSkipped = 0, quietReorders = 0, quietPromotions = 0, quietWork = 0;
};

} // namespace Stockfish
#endif
