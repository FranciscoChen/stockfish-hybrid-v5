/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "search.h"
#ifdef __EMSCRIPTEN__
    #include <emscripten.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <list>
#include <limits>
#include <ratio>
#include <string>
#include <utility>

#include "bitboard.h"
#include "evaluate.h"
#include "history.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "position.h"
#include "syzygy/tbprobe.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"
#ifdef SF_SEARCH_SAMPLER
#include "search_sampler.h"
#endif

namespace Stockfish {

inline int lmr_divisor(int depth) {
    int d = std::min(depth, 16);
    return 3000 + 7 * (d - 8) * (d - 8);
}

namespace TB = Tablebases;

void syzygy_extend_pv(const OptionsMap&            options,
                      const Search::LimitsType&    limits,
                      Stockfish::Position&         pos,
                      Stockfish::Search::RootMove& rootMove,
                      Value&                       v,
                      const usize                  multiPV);

using namespace Search;

namespace {

constexpr u64 NODES_LIMIT_OUTPUT = 10'000'000;

constexpr int SEARCHEDLIST_CAPACITY = 32;
using SearchedList                  = ValueList<Move, SEARCHEDLIST_CAPACITY>;

// Reductions lookup table indexed by depth or move number.
const std::array<int, MAX_MOVES> reductions = [] {
    std::array<int, MAX_MOVES> table{};
    for (usize i = 1; i < table.size(); ++i)
        table[i] = int(2872 / 128.0 * std::log(i));
    return table;
}();

// (*Scalers):
// The values with Scaler asterisks have proven non-linear scaling.
// They are optimized to time controls of 180 + 1.8 and longer,
// so changing them or adding conditions that are similar requires
// tests at these types of time controls.

// (*Scaler) All tuned parameters at time controls shorter than
// optimized for require verifications at longer time controls.

int correction_value(const Worker& w, const Position& pos, const Stack* const ss) {
    const Color us     = pos.side_to_move();
    const auto  m      = (ss - 1)->currentMove;
    const auto& shared = w.sharedHistory;
    const int   pcv    = shared.pawn_correction_entry(pos)[us].pawn;
    const int   micv   = shared.minor_piece_correction_entry(pos)[us].minor;
    const int   wnpcv  = shared.nonpawn_correction_entry<WHITE>(pos)[us].nonPawnWhite;
    const int   bnpcv  = shared.nonpawn_correction_entry<BLACK>(pos)[us].nonPawnBlack;
    const int   cntcv =
      m.is_ok()
          ? 7885
              * ((*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
                 + (*(ss - 4)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()])
            + 6307 * (*(ss - 6)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
          : 80695;

    return 13806 * pcv + 9512 * micv + 11615 * (wnpcv + bnpcv) + cntcv;
}

// Add correctionHistory value to raw staticEval and guarantee evaluation
// does not hit the tablebase range.
Value to_corrected_static_eval(const Value v, const int cv) {
    return std::clamp(v + cv / 131072, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}

void update_correction_history(const Position& pos,
                               Stack* const    ss,
                               Search::Worker& workerThread,
                               const int       bonus) {
    const Move  m  = (ss - 1)->currentMove;
    const Color us = pos.side_to_move();

    constexpr int nonPawnWeight = 186;
    auto&         shared        = workerThread.sharedHistory;

    shared.pawn_correction_entry(pos)[us].pawn << bonus;
    shared.minor_piece_correction_entry(pos)[us].minor << bonus * 150 / 128;
    shared.nonpawn_correction_entry<WHITE>(pos)[us].nonPawnWhite << bonus * nonPawnWeight / 128;
    shared.nonpawn_correction_entry<BLACK>(pos)[us].nonPawnBlack << bonus * nonPawnWeight / 128;

    if (m.is_ok())
    {
        const Square to = m.to_sq();
        const Piece  pc = pos.piece_on(to);
        (*(ss - 2)->continuationCorrectionHistory)[pc][to] << bonus * 130 / 128;
        (*(ss - 4)->continuationCorrectionHistory)[pc][to] << bonus * 70 / 128;
        (*(ss - 6)->continuationCorrectionHistory)[pc][to] << bonus * 35 / 128;
    }
}

// Add a small random component to draw evaluations to avoid 3-fold blindness
Value value_draw(usize nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }
Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply, int r50c);
void  update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void  update_quiet_histories(
   const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus);
void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            ttMove,
                      bool            PvNode);

// Detect shuffling moves in order to limit search explosions. This code was
// added in #6447 as non-regression, and so its parameters should not be tuned.
bool is_shuffling(Move move, Stack* const ss, const Position& pos) {
    if (pos.capture_stage(move) || pos.rule50_count() < 10)
        return false;
    if (pos.state()->pliesFromNull < 6 || ss->ply < 20)
        return false;
    return move.from_sq() == (ss - 2)->currentMove.to_sq()
        && (ss - 2)->currentMove.from_sq() == (ss - 4)->currentMove.to_sq();
}

}  // namespace

Search::Worker::Worker(SharedState&                   sharedState,
                       std::unique_ptr<SearchManager> sm,
                       usize                          threadId,
                       usize                          numaThreadId,
                       usize                          numaTotalThreads,
                       NumaReplicatedAccessToken      token) :
    // Unpack the SharedState struct into member variables
    sharedHistory(sharedState.sharedHistories.at(token.get_numa_index())),
    continuationHistory(sharedHistory.continuationHistory()),
    threadIdx(threadId),
    numaThreadIdx(numaThreadId),
    numaTotal(numaTotalThreads),
    numaAccessToken(token),
    manager(std::move(sm)),
    options(sharedState.options),
    threads(sharedState.threads),
    tt(sharedState.tt),
    network(sharedState.network),
    refreshTable(network[token]) {
    clear();
}

void Search::Worker::ensure_network_replicated() {
    // Access once to force lazy initialization, avoiding initialization during search
    (void) (network[numaAccessToken]);
}

void Search::Worker::start_searching() {

    accumulatorStack.reset();

    // Non-main threads go directly to iterative_deepening()
    if (!is_mainthread())
    {
        iterative_deepening();
        return;
    }

    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                            main_manager()->originalTimeAdjust);
    tt.new_search();
    main_manager()->updates.onStart();

    if (rootMoves.empty())
    {
        main_manager()->updates.onUpdateNoMoves(
          {0, {rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW, rootPos}});
        main_manager()->updates.onBestmove(UCIEngine::move(Move::none()), "");
        return;
    }

    // Main thread starts non-main threads, and begins own search
    threads.start_searching();
    bool uciPvSent = iterative_deepening();
#ifdef SF_SEARCH_SAMPLER
    searchSampler.finish();
#endif
    if (policySearchStats)
        sync_cout << "info string PolicySearchStats {\"nodes\":" << policySearchCounters.nodes
                  << ",\"root_visits\":" << policySearchCounters.rootVisits
                  << ",\"inferences\":" << policySearchCounters.inferences
                  << ",\"cache_hits\":" << policySearchCounters.cacheHits
                  << ",\"forced\":" << policySearchCounters.forced
                  << ",\"quiet_calls\":" << policySearchCounters.quietCalls
                  << ",\"quiet_budget_skipped\":" << policySearchCounters.quietBudgetSkipped
                  << ",\"quiet_history_skipped\":" << policySearchCounters.quietHistorySkipped
                  << ",\"quiet_cache_hits\":" << policySearchCounters.quietCacheHits
                  << ",\"quiet_head_cache_hits\":" << policySearchCounters.quietHeadCacheHits
                  << ",\"quiet_output_scores\":" << policySearchCounters.quietOutputScores
                  << ",\"quiet_node_skipped\":" << policySearchCounters.quietNodeSkipped
                  << ",\"quiet_reorders\":" << policySearchCounters.quietReorders
                  << ",\"quiet_promotions\":" << policySearchCounters.quietPromotions
                  << ",\"quiet_work\":" << policySearchCounters.quietWork
                  << ",\"lmr_protected\":" << policySearchCounters.lmrProtected
                  << ",\"margin_lmr_reduced\":" << policySearchCounters.marginLmrReduced
                  << ",\"margin_lmr_fail_high\":" << policySearchCounters.marginLmrFailHigh
                  << ",\"margin_lmr_researched\":" << policySearchCounters.marginLmrResearched
                  << ",\"delayed_first_searched\":" << policySearchCounters.delayedFirstSearched
                  << ",\"delayed_first_failed\":" << policySearchCounters.delayedFirstFailed
                  << ",\"delayed_shortlists\":" << policySearchCounters.delayedShortlists
                  << ",\"delayed_calls\":" << policySearchCounters.delayedCalls
                  << ",\"delayed_new_leaders\":" << policySearchCounters.delayedNewLeaders
                  << ",\"delayed_leader_searched\":" << policySearchCounters.delayedLeaderSearched
                  << ",\"delayed_leader_cutoffs\":" << policySearchCounters.delayedLeaderCutoffs
                  << "}" << sync_endl;
    if (policySearchStats && cutoffPolicyOrdering)
        for (int bucket = 0; bucket < 4; ++bucket)
            for (int swapped = 0; swapped < 2; ++swapped) {
                const auto& s = policySearchCounters.cutoffSamples[bucket][swapped];
                sync_cout << "info string PolicyCutoffStats {\"bucket\":" << bucket
                          << ",\"swapped\":" << swapped << ",\"searched\":" << s.searched
                          << ",\"cutoffs\":" << s.cutoffs << ",\"nodes\":" << s.nodes
                          << ",\"researches\":" << s.researches << "}" << sync_endl;
            }
    if (bool(options["CompactPolicyStats"]))
        sync_cout << "info string Compact internal policy calls (main worker): "
                  << internalPolicyCalls << sync_endl;

    if (policyDiagnostics) {
        // Main-worker counters; use Threads=1 for complete accounting.
        sync_cout << "info string PolicyDiagnostics {\"calls\":" << internalPolicyCalls
                  << ",\"deep\":" << policyDiagnostic.deep
                  << ",\"tt\":" << policyDiagnostic.tt
                  << ",\"excluded\":" << policyDiagnostic.excluded
                  << ",\"nodeType\":" << policyDiagnostic.nodeType
                  << ",\"check\":" << policyDiagnostic.check
                  << ",\"eligible\":" << policyDiagnostic.eligible
                  << ",\"callbacks\":" << policyDiagnostic.callbacks
                  << ",\"budget\":" << policyDiagnostic.budget
                  << ",\"candidates\":" << policyDiagnostic.candidates
                  << ",\"history\":" << policyDiagnostic.history
                  << ",\"agreement\":" << policyDiagnostic.agreement
                  << ",\"confidence\":" << policyDiagnostic.confidence
                  << ",\"promotionHistory\":" << policyDiagnostic.promotionHistory
                  << ",\"promotions\":" << policyDiagnostic.promotions
                  << ",\"searched\":" << policyDiagnostic.searched
                  << ",\"completed\":" << policyDiagnostic.completed
                  << ",\"cutoffs\":" << policyDiagnostic.cutoffs
                  << ",\"moveNodes\":" << policyDiagnostic.moveNodes
                  << ",\"inferenceNs\":" << policyDiagnostic.inferenceNs
                  << ",\"gateNs\":" << policyDiagnostic.gateNs
                  << ",\"rankSum\":" << policyDiagnostic.rankSum
                  << ",\"rankMax\":" << policyDiagnostic.rankMax << "}" << sync_endl;
        for (const auto& event : policyPromotionEvents)
            sync_cout << "info string PolicyPromotion {\"fen\":\"" << event.fen
                      << "\",\"move\":\"" << event.move << "\",\"rank\":" << event.rank
                      << ",\"nodes\":" << event.nodes << ",\"searched\":" << event.searched
                      << ",\"completed\":" << event.completed << ",\"cutoff\":" << event.cutoff
                      << "}" << sync_endl;
    }

    // When we reach the maximum depth, we can arrive here without a raise of
    // threads.stop. However, if we are pondering or in an infinite search,
    // the UCI protocol states that we shouldn't print the best move before the
    // GUI sends a "stop" or "ponderhit" command. We therefore simply wait here
    // until the GUI sends one of those commands.
    while (!threads.stop && (main_manager()->ponder || limits.infinite))
    {
#ifdef __EMSCRIPTEN__
        emscripten_sleep(10);
#endif
    }

    // Stop the threads if not already stopped (also raise the stop if "ponderhit"
    // just reset threads.ponder).
    threads.stop = true;

    // Wait until all threads have finished
    threads.wait_for_search_finished();

    // When playing in 'nodes as time' mode, subtract the searched nodes from
    // the available ones before exiting.
    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched()
                                              - limits.inc[rootPos.side_to_move()]);

    Worker* bestThread = this;
    Skill   skill =
      Skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);

    if (!limits.depth && !skill.enabled())
        bestThread = threads.get_best_thread()->worker.get();

    main_manager()->bestPreviousScore        = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;

    if (bestThread->rootMoves[0].pv.size() == 1
        && bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos))
        uciPvSent = false;

    // Send PV info if it has changed since last output in iterative_deepening()
    if (!uciPvSent || bestThread != this)
        main_manager()->output_pv(*bestThread, threads, tt, bestThread->rootDepth);

    // In rare cases, output_pv() may change the ponder move through syzygy_extend_pv()
    std::string ponder;
    if (bestThread->rootMoves[0].pv.size() > 1)
        ponder = UCIEngine::move(bestThread->rootMoves[0].pv[1], rootPos.is_chess960());

    auto bestmove = UCIEngine::move(bestThread->rootMoves[0].pv[0], rootPos.is_chess960());
    main_manager()->updates.onBestmove(bestmove, ponder);
}

// Main iterative deepening loop. It calls search() repeatedly with increasing
// depth until the allocated thinking time has been consumed, the user stops
// the search, or the maximum search depth is reached.
bool Search::Worker::iterative_deepening() {

    SearchManager* mainThread = (is_mainthread() ? main_manager() : nullptr);

#ifdef SF_SEARCH_SAMPLER
    if (mainThread)
        searchSampler.start(std::string(options["SearchSampleFile"]),
                            int(options["SearchSampleEvery"]),
                            int(options["SearchSampleLimit"]),
                            int(options["SearchSampleMinDepth"]), rootPos.fen());
#endif
    policySearchEnabled = bool(options["PolicySearch"]);
    policySearchReductions = bool(options["PolicySearchReductions"]);
    policySearchStats = bool(options["PolicySearchStats"]);
    policySearchExploration = bool(options["PolicySearchExploration"]);
    policySearchTemperature = int(options["PolicySearchTemperature"]) / 100.0f;
    policySearchCounters = {};
    policySearchCache.clear();
    rootPolicyEvidence.clear();
    incrementalPolicyEnabled = false;
    economicalMinDepth = int(options["PolicyEconomicalMinDepth"]);
    delayedQuietPolicy = false;
    delayedMinPieces = int(options["PolicyDelayedMinPieces"]);
    cutoffPolicyOrdering = false;
    useRootChallenger = bool(options["PolicyRootChallenger"]);
    verifyPolicyAccumulators = bool(options["PolicySearchVerifyAccumulators"]);
    if (policySearchEnabled) {
        std::string error;
        const std::string policyPath = options["CompactPolicyFile"];
        incrementalPolicyEnabled = IncrementalPolicy::is_format(policyPath);
        if (!(incrementalPolicyEnabled ? incrementalPolicy.load(policyPath, error)
              : compactPolicy.load(policyPath, std::string(options["EvalFile"]), error))) {
            sync_cout << "info string ERROR: PolicySearch requires a valid policy: " << error << sync_endl;
            // Fatal worker-thread errors cannot run global thread destructors.
            // sync_endl above has already flushed the error to the UCI client.
            std::_Exit(EXIT_FAILURE);
        }
        incrementalPolicy.reset();
        cutoffPolicyOrdering = incrementalPolicyEnabled && bool(options["PolicySearchCutoffOrdering"]);
        delayedQuietPolicy = incrementalPolicyEnabled && !cutoffPolicyOrdering && economicalMinDepth == 8
                          && bool(options["PolicyDelayedQuiets"]);
        if (cutoffPolicyOrdering) {
            for (int i = 0; i < 4; ++i)
                cutoffBonus[i] = int(options["PolicyCutoffBonus" + std::to_string(i)]);
            for (auto& rm : rootMoves) rm.policyLogitValid = false;
        }
        // This cache now serves the root only. One entry covers every MultiPV
        // lookup; reserving a full transposition cache just wastes worker memory.
        policySearchCache.resize(!cutoffPolicyOrdering && int(options["PolicySearchCacheEntries"]) > 0 ? 1 : 0);
        if (mainThread)
            sync_cout << "info string PolicySearch hybrid active; quiet margin LMR="
                      << (incrementalPolicyEnabled && policySearchReductions) << sync_endl;
    }
    quietPolicyCache.prepare(incrementalPolicyEnabled && int(options["PolicySearchCacheEntries"]) > 0,
                             incrementalPolicy.generation());
    policyDiagnostics = bool(options["CompactPolicyDiagnostics"]);
    policyDiagnostic = {};
    policyPromotionEvents.clear();
    internalPolicyCalls = 0;
    internalPolicyEnabled = false;
    internalPolicyMinDepth = int(options["CompactPolicyMinDepth"]);
    internalPolicyBonus = int(options["CompactPolicyOrderingBonus"]);
    if (!policySearchEnabled && bool(options["CompactPolicy"]) && bool(options["CompactPolicyInternal"])
        && !bool(options["LeelaPolicy"]) && !rootPos.is_chess960() && internalPolicyBonus > 0) {
        std::string error;
        internalPolicyEnabled = compactPolicy.load(std::string(options["CompactPolicyFile"]),
                                                    std::string(options["EvalFile"]), error);
        if (!internalPolicyEnabled && mainThread && !error.empty())
            sync_cout << "info string Internal policy unavailable: " << error << sync_endl;
    }
    rootPolicy.clear();
    if (!policySearchEnabled && mainThread && bool(options["LeelaPolicy"]) && rootPos.is_chess960())
        sync_cout << "info string Leela policy disabled: Chess960 is not supported yet" << sync_endl;
    if (!policySearchEnabled && mainThread && bool(options["LeelaPolicy"]) && !rootPos.is_chess960()
        && rootMoves.size() > 1) {
        std::vector<std::string> moves;
        for (const auto& rm : rootMoves) moves.push_back(UCIEngine::move(rm.pv[0], true));
        std::vector<float> probabilities;
        std::string error;
        if (leelaPolicy.evaluate(std::string(options["LeelaPolicyLibrary"]),
                std::string(options["LeelaWeightsFile"]), std::string(options["LeelaBackend"]),
                limits.policyStartFen, limits.policyHistory, moves, probabilities, error)) {
            for (usize i = 0; i < rootMoves.size(); ++i)
                rootPolicy.emplace_back(rootMoves[i].pv[0], probabilities[i]);
            sync_cout << "info string Leela root policy loaded for " << rootPolicy.size()
                      << " moves" << sync_endl;
        } else
            sync_cout << "info string Leela policy disabled for this search: " << error << sync_endl;
    }

    // Explicit teacher collection is separate from deployment of the small student.
    const std::string trainingFile = options["PolicyTrainingFile"];
    if (mainThread && !trainingFile.empty() && !bool(options["LeelaPolicy"]))
        for (const auto& rm : rootMoves) rootPolicy.emplace_back(rm.pv[0], 1.0f / rootMoves.size());
    if (mainThread && !trainingFile.empty() && !rootPolicy.empty()) {
        alignas(Eval::NNUE::CacheLineSize) u8 features[Eval::NNUE::FeatureTransformer::BufferSize];
        network[numaAccessToken].policy_features(rootPos, accumulatorStack, refreshTable, features);
        std::ofstream out(trainingFile, std::ios::app);
        out << "{\"features\":[";
        for (int i = 0; i < 1024; ++i) out << (i ? "," : "") << int(features[i]);
        out << "],\"moves\":[";
        for (usize i = 0; i < rootPolicy.size(); ++i) {
            auto f = CompactPolicy::factors(rootPolicy[i].first, rootPos.side_to_move());
            out << (i ? "," : "") << '[' << f[0] << ',' << f[1] << ',' << f[2] << ']';
        }
        out << "],\"targets\":[";
        for (usize i = 0; i < rootPolicy.size(); ++i)
            out << (i ? "," : "") << rootPolicy[i].second;
        out << "]}\n";
        if (!out) sync_cout << "info string Policy training output failed" << sync_endl;
    }
    if (!bool(options["LeelaPolicy"])) rootPolicy.clear();
    if (!policySearchEnabled && mainThread && bool(options["CompactPolicy"]) && !bool(options["LeelaPolicy"])
        && !rootPos.is_chess960() && rootMoves.size() > 1) {
        std::string error;
        if (compactPolicy.load(std::string(options["CompactPolicyFile"]),
                               std::string(options["EvalFile"]), error)) {
            alignas(Eval::NNUE::CacheLineSize) u8 features[Eval::NNUE::FeatureTransformer::BufferSize];
            network[numaAccessToken].policy_features(rootPos, accumulatorStack, refreshTable, features);
            const auto context = compactPolicy.context(features, file_of(rootPos.square<KING>(rootPos.side_to_move())) >= FILE_E, file_of(rootPos.square<KING>(~rootPos.side_to_move())) >= FILE_E);
            for (const auto& rm : rootMoves)
                rootPolicy.emplace_back(rm.pv[0], compactPolicy.score(context, rm.pv[0], rootPos.side_to_move()));
            sync_cout << "info string Compact root policy loaded" << sync_endl;
        } else if (!error.empty())
            sync_cout << "info string " << error << "; using ordinary ordering" << sync_endl;
    }

    PVMoves pv;

    RootPVMoves lastBestMovePV;
    Depth       lastBestMoveDepth = 0;
    Value       lastBestMoveScore = -VALUE_INFINITE;

    Value  alpha, beta;
    Value  bestValue     = -VALUE_INFINITE;
    Color  us            = rootPos.side_to_move();
    double timeReduction = 1, totBestMoveChanges = 0;
    int    delta, iterIdx                        = 0;

    // Allocate stack with extra size to allow access from (ss - 7) to (ss + 2):
    // (ss - 7) is needed for update_continuation_histories(ss - 1) which accesses (ss - 6),
    // (ss + 2) is needed for initialization of cutOffCnt.
    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;

    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory =
          &continuationHistory[0][0][NO_PIECE][0];  // Use as a sentinel
        (ss - i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval                    = VALUE_NONE;
    }

    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    ss->pv = &pv;

    if (mainThread)
    {
        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }

    usize multiPV = usize(options["MultiPV"]);
    Skill skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);

    // When playing with strength handicap enable MultiPV search that we will
    // use behind-the-scenes to retrieve a set of possible moves.
    if (skill.enabled())
        multiPV = std::max(multiPV, usize(4));

    multiPV = std::min(multiPV, rootMoves.size());

    int  searchAgainCounter = 0;
    int  failHighRecovery   = 0;
    bool uciPvSent          = false;

    lowPlyHistory.fill(102);

    for (Color c : {WHITE, BLACK})
        for (int i = 0; i < UINT_16_HISTORY_SIZE; i++)
            mainHistory[c][i] = mainHistory[c][i] * 729 / 1024;

    // Iterative deepening loop until requested to stop or the target depth is reached
    while (rootDepth + 1 < MAX_PLY && !threads.stop
           && !(limits.depth && mainThread && rootDepth >= limits.depth))
    {
        rootDepth++;

        // Do not tune these values. They are not intended for playing strength.
        int seekMateThreshold = 750 + 220000 / (rootDepth * rootDepth);

        // Signal the start of a new iteration
        if (mainThread)
            uciPvSent = false;

        // Save the last iteration's scores before the first PV line is searched and
        // all the move scores except the (new) PV are set to -VALUE_INFINITE.
        for (usize i = 0; i < rootMoves.size(); ++i)
        {
            rootMoves[i].previousScore      = rootMoves[i].score;
            rootMoves[i].previousPV         = rootMoves[i].pv;
            rootMoves[i].previousScoreExact = i < multiPV;
        }

        usize pvFirst = pvLast = 0;

        if (!threads.increaseDepth)
            searchAgainCounter++;

        // MultiPV loop: we perform a full root search for each PV line
        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx)
        {
            if (pvIdx == pvLast)
            {
                pvFirst = pvLast;
                for (pvLast++; pvLast < rootMoves.size(); pvLast++)
                    if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
                        break;
            }

            lastIterationIdxPV = rootMoves[pvIdx].previousPV;

            // Reset UCI info selDepth for each depth and each PV line
            selDepth = 0;

            // Reset aspiration window starting size
            delta     = 5 + threadIdx % 8 + std::abs(rootMoves[pvIdx].meanSquaredScore) / 10193;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha     = std::max(avg - delta, -VALUE_INFINITE);
            beta      = std::min(avg + delta, VALUE_INFINITE);

            // For multiPV we use previousScore because score is often -VALUE_INFINITE
            Value seekMateScore = pvIdx ? rootMoves[pvIdx].previousScore : rootMoves[pvIdx].score;
            seekMate =
              seekMateScore != -VALUE_INFINITE && std::abs(seekMateScore) >= seekMateThreshold;

            // Adjust optimism based on root move's averageScore
            optimism[us]  = 456 * avg / (std::abs(avg) + 85);
            optimism[~us] = -optimism[us];

            // Start with a small aspiration window and, in the case of a fail
            // high/low, enlarge the window progressively.
            int failedHighCnt = 0;
            if (!pvIdx)
                failHighRecovery = std::max(0, failHighRecovery - 2);
            while (true)
            {
                // Adjust the effective depth searched, but ensure at least one
                // effective increment for every four searchAgain steps (see issue #2717).
                Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt - failHighRecovery
                                                    - 3 * (searchAgainCounter + 1) / 4);
                bestValue           = search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);

                // Bring the best move to the front. It is critical that sorting
                // is done with a stable algorithm because all the values but the
                // first and eventually the new best one is set to -VALUE_INFINITE
                // and we want to keep the same order for all the moves except the
                // new PV that goes to the front. Note that in the case of MultiPV
                // search the already searched PV lines are preserved.
                std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);

                // If search has been stopped, we break immediately. Sorting is
                // safe because RootMoves is still valid, although it refers to
                // the previous iteration.
                if (threads.stop)
                    break;

                // When failing high/low give some update before a re-search. To avoid
                // excessive output that could hang GUIs like Fritz 19, only start
                // at nodes > 10M (rather than depth N, which can be reached quickly).
                if (mainThread && multiPV == 1 && (bestValue <= alpha || bestValue >= beta)
                    && nodes > NODES_LIMIT_OUTPUT)
                    main_manager()->output_pv(*this, threads, tt, rootDepth);

                // In case of failing low/high increase aspiration window and re-search,
                // otherwise exit the loop.
                if (bestValue <= alpha)
                {
                    beta  = alpha;
                    alpha = std::max(bestValue - delta, -VALUE_INFINITE);

                    failedHighCnt = 0;
                    if (mainThread)
                        mainThread->stopOnPonderhit = false;
                }
                else if (bestValue >= beta)
                {
                    alpha = std::max(beta - delta, alpha);
                    beta  = std::min(bestValue + delta, VALUE_INFINITE);
                    ++failedHighCnt;
                }
                else
                    break;

                delta += 47 * delta / 128;

                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }

            // Gradually increase depth after reduced depth search
            if (failedHighCnt > 0 && !pvIdx)
                failHighRecovery = (failedHighCnt + 1) / 2 + 2;

            if (threads.stop && pvIdx)
            {
                // In multiPV analysis we do not let aborted searches spoil
                // mated-in/TB loss scores from a completed search in an earlier
                // PV line. Hence we guard against an aborted pvIdx line overtaking
                // pvIdx - 1 when pvIdx - 1 is a proven loss. Moreover, we do not
                // trust an exact loss score from an aborted search.
                if ((is_loss(rootMoves[pvIdx - 1].score) && rootMoves[pvIdx] < rootMoves[pvIdx - 1])
                    || rootMoves[pvIdx].is_exact_loss())
                {
                    // If previousScore is exact and worse than pvIdx - 1, we
                    // can safely use it. If it is equal, we make sure it cannot
                    // overtake pvIdx - 1.
                    if (rootMoves[pvIdx].previousScore != -VALUE_INFINITE
                        && rootMoves[pvIdx].previousScoreExact
                        && rootMoves[pvIdx].previousScore <= rootMoves[pvIdx - 1].score)
                    {
                        rootMoves[pvIdx].score = rootMoves[pvIdx].uciScore =
                          rootMoves[pvIdx].previousScore;
                        rootMoves[pvIdx].previousScore = -VALUE_INFINITE;
                        rootMoves[pvIdx].pv            = rootMoves[pvIdx].previousPV;
                        rootMoves[pvIdx].unset_inexact();
                    }

                    // Otherwise, if we can, we cap the score to the best possible, and mark
                    // the score as inexact (also a valid excuse for the incomplete PV).
                    else
                    {
                        if (is_loss(rootMoves[pvIdx - 1].score))
                        {
                            rootMoves[pvIdx].score = rootMoves[pvIdx].uciScore =
                              rootMoves[pvIdx - 1].score;
                            rootMoves[pvIdx].previousScore = -VALUE_INFINITE;
                            rootMoves[pvIdx].pv.resize(1);
                            rootMoves[pvIdx].inexactUpper = true;
                        }
                        else
                            rootMoves[pvIdx].inexactUpper = false;

                        rootMoves[pvIdx].inexactLower = !rootMoves[pvIdx].inexactUpper;
                    }
                }

                // Finally, we mark all loss scores from partially searched moves as inexact.
                for (usize i = pvIdx + 1; i < multiPV; ++i)
                    if (rootMoves[i].is_exact_loss())
                        rootMoves[i].inexactLower = true;
            }

            // Sort the PV lines searched so far and update the GUI
            std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1);

            if (mainThread && !threads.stop && (pvIdx + 1 == multiPV || nodes > NODES_LIMIT_OUTPUT))
            {
                main_manager()->output_pv(*this, threads, tt, rootDepth);
                uciPvSent = (pvIdx + 1 == multiPV);
            }

            if (threads.stop)
                break;
        }

        const bool forgottenMate = lastBestMoveScore != -VALUE_INFINITE
                                && is_mate_or_mated(lastBestMoveScore)
                                && (std::abs(rootMoves[0].score) < std::abs(lastBestMoveScore)
                                    || rootMoves[0].is_inexact());

        if (!threads.stop)
        {
            if (lastBestMovePV.empty() || lastBestMovePV[0] != rootMoves[0].pv[0])
                lastBestMoveDepth = rootDepth;

            // Do not replace (shorter) mate scores from a previous iteration
            if (!forgottenMate)
            {
                lastBestMovePV    = rootMoves[0].pv;
                lastBestMoveScore = rootMoves[0].score;
            }
        }

        const bool abortedLossSearch = threads.stop && !pvIdx && rootMoves[0].is_exact_loss();

        // An exact mated-in/TB-loss score from an aborted search cannot be
        // trusted: the loss could be delayed or refuted upon exploring the
        // remaining root-moves. Thus here we roll back to the score from the
        // previous iteration. We do the same if a search has failed to recover
        // a mate score that was found in a previous iteration.
        if (abortedLossSearch || (rootMoves[0].score != -VALUE_INFINITE && forgottenMate))
        {
            // Bring the last best move to the front for best thread selection
            if (!lastBestMovePV.empty())
            {
                Utility::move_to_front(rootMoves, [&lastPV = std::as_const(lastBestMovePV)](
                                                    const auto& rm) { return rm == lastPV[0]; });
                rootMoves[0].score = rootMoves[0].uciScore = lastBestMoveScore;
                rootMoves[0].pv                            = lastBestMovePV;
                rootMoves[0].unset_inexact();

                if (mainThread)
                    uciPvSent = false;
            }
            // For an aborted d1 search we label the loss score as inexact
            else if (abortedLossSearch)
                rootMoves[0].inexactLower = true;
        }

        // Have we found a "mate in x" after a completed iteration?
        if (limits.mate && !threads.stop && is_mate_or_mated(rootMoves[0].score)
            && VALUE_MATE - std::abs(rootMoves[0].score) <= 2 * limits.mate)
            threads.stop = true;

        if (!mainThread)
            continue;

        // If the skill level is enabled and time is up, pick a sub-optimal best move
        if (skill.enabled() && skill.time_to_pick(rootDepth))
            skill.pick_best(rootMoves, multiPV);

        // Do we have time for the next iteration? Can we stop searching now?
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit)
        {
            u64 nodesEffort = rootMoves[0].effort * 100000 / std::max(u64(1), u64(nodes));

            double fallingEval = (11.48 + 2.30 * (mainThread->bestPreviousAverageScore - bestValue)
                                  + 1.1 * (mainThread->iterValue[iterIdx] - bestValue))
                               / 100.0;
            fallingEval = std::clamp(fallingEval, 0.576, 1.728);

            // If the bestMove is stable over several iterations, reduce time accordingly
            timeReduction = std::clamp(
              interpolate(double(rootDepth - lastBestMoveDepth), 4.96, 18.79, 0.639, 1.712), 0.629,
              1.544);

            double reduction =
              (1.468 + mainThread->previousTimeReduction) / (2.284 * timeReduction);

            // Age out and update PV variability metric
            totBestMoveChanges /= 2;
            for (auto&& th : threads)
            {
                totBestMoveChanges += th->worker->bestMoveChanges;
                th->worker->bestMoveChanges = 0;
            }

            double bestMoveInstability = 1.077 + 2.229 * totBestMoveChanges / threads.size();

            double highBestMoveEffort = std::clamp(
              interpolate(i64(nodesEffort), i64(75800), i64(104510), 0.969, 0.714), 0.693, 0.838);

            double totalTime = mainThread->tm.optimum() * fallingEval * reduction
                             * bestMoveInstability * highBestMoveEffort;

            if (rootMoves.size() == 1)
                // Cap used time to 0.5s for a better viewer experience
                totalTime = std::min(500.0 * std::max(limits.npmsec, TimePoint(1)), totalTime);

            auto elapsedTime = elapsed();

            // Stop the search if we have exceeded totalTime or maximum time,
            // or if we know that there are no better moves in the analysed line(s).
            if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum()))
                || rootMoves[multiPV - 1].score >= mate_in(3) || rootMoves[0].score == mated_in(2))
            {
                // If we are allowed to ponder do not stop the search now but
                // keep pondering until the GUI sends "ponderhit" or "stop".
                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
            }
            else
                threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.50;
        }

        mainThread->iterValue[iterIdx] = bestValue;
        iterIdx                        = (iterIdx + 1) & 3;
    }

    if (!mainThread)
        return false;

    mainThread->previousTimeReduction = timeReduction;

    // If the skill level is enabled, swap the best PV line with the sub-optimal one
    if (skill.enabled())
        std::swap(rootMoves[0],
                  *std::find(rootMoves.begin(), rootMoves.end(),
                             skill.best ? skill.best : skill.pick_best(rootMoves, multiPV)));

    return uciPvSent;
}


void Search::Worker::do_move(Position& pos, const Move move, StateInfo& st, Stack* const ss) {
    do_move(pos, move, st, pos.gives_check(move), pos.capture_stage(move), ss);
}

template<bool TrackPolicy>
void Search::Worker::do_move(Position&    pos,
                             const Move   move,
                             StateInfo&   st,
                             const bool   givesCheck,
                             const bool   capture,
                             Stack* const ss) {
    // prefetch_key() does not model castling, en passant or promotion exactly.
    // The correction-history prefetches also approximate castling and promotion.
    // For these rare moves the prefetches land on unused lines.
    const Piece pc = pos.moved_piece(move);
    prefetch(tt.get_cluster(pos.prefetch_key(move, pc)));

    if (ss != nullptr)
    {
        const Square to = move.to_sq();

        prefetch(&(*(ss - 1)->continuationCorrectionHistory)[pc][to]);
        prefetch(&(*(ss - 3)->continuationCorrectionHistory)[pc][to]);
        prefetch(&(*(ss - 5)->continuationCorrectionHistory)[pc][to]);
    }

    ++nodes;

    Dirties& dirties = accumulatorStack.push();
    pos.do_move(move, st, givesCheck, dirties, &sharedHistory, pc);
    if constexpr (TrackPolicy)
        if (incrementalPolicyEnabled) incrementalPolicy.push();

    if (ss != nullptr)
    {
        auto& dirtyPiece = dirties.dirtyPiece;
        ss->currentMove  = move;
        ss->continuationHistory =
          &continuationHistory[ss->inCheck][capture][dirtyPiece.pc][move.to_sq()];
        ss->continuationCorrectionHistory =
          &continuationCorrectionHistory[dirtyPiece.pc][move.to_sq()];
    }
}

void Search::Worker::do_null_move(Position& pos, StateInfo& st, Stack* const ss) {
    pos.do_null_move(st);
    if (incrementalPolicyEnabled) incrementalPolicy.push();
    ss->currentMove                   = Move::null();
    ss->continuationHistory           = &continuationHistory[0][0][NO_PIECE][0];
    ss->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
}

template<bool TrackPolicy>
void Search::Worker::undo_move(Position& pos, const Move move) {
    pos.undo_move(move);
    accumulatorStack.pop();
    if constexpr (TrackPolicy)
        if (incrementalPolicyEnabled) incrementalPolicy.pop();
}

void Search::Worker::undo_null_move(Position& pos) {
    pos.undo_null_move();
    if (incrementalPolicyEnabled) incrementalPolicy.pop();
}


// Reset histories, usually before a new game
void Search::Worker::clear() {
    quietPolicyCache.reset(false);
    mainHistory.fill(-5);
    captureHistory.fill(-742);

    // Each thread clears its part of the dynamically-sized shared histories.
    // The constant-size continuation history is initialized by thread 0 of each NUMA node.
    sharedHistory.correctionHistory.clear_range(-5, numaThreadIdx, numaTotal);
    sharedHistory.pawnHistory.clear_range(-1338, numaThreadIdx, numaTotal);

    if (numaThreadIdx == 0)
        for (bool inCheck : {false, true})
            for (StatsType c : {NoCaptures, Captures})
                for (auto& to : continuationHistory[inCheck][c])
                    for (auto& h : to)
                        h.fill(-586);

    ttMoveHistory = 0;

    for (auto& to : continuationCorrectionHistory)
        for (auto& h : to)
            h.fill(5);

    refreshTable.clear(network[numaAccessToken]);
}


// Main search function for both PV and non-PV nodes
// At depth one only the challengers in this TB rank and MultiPV range can
// change order. Carry each raw score with its RootMove across MultiPV sorts.
void Search::Worker::cutoff_root_order(std::vector<Move>& order) {
    order.reserve(pvLast - pvIdx);
    for (usize i = pvIdx; i < pvLast; ++i) order.push_back(rootMoves[i].pv[0]);
    if (order.size() <= 2) return;

    std::array<Move, MAX_MOVES> missing;
    std::array<float, MAX_MOVES> logits;
    std::array<RootMove*, MAX_MOVES> challengers;
    int count = 0;
    for (usize i = 1; i < order.size(); ++i) {
        auto& rm = rootMoves[pvIdx + i];
        challengers[i - 1] = &rm;
        if (!rm.policyLogitValid) missing[count++] = order[i];
    }
    if (count) {
        const auto hidden = incrementalPolicy.context(rootPos, verifyPolicyAccumulators);
        incrementalPolicy.scores(hidden, missing.data(), logits.data(), count, rootPos);
        int j = 0;
        for (usize i = 0; i + 1 < order.size(); ++i) {
            auto& rm = *challengers[i];
            if (rm.policyLogitValid) continue;
            rm.policyLogit = logits[j++];
            if (!std::isfinite(rm.policyLogit)
                || (verifyPolicyAccumulators && rm.policyLogit != incrementalPolicy.score(hidden, rm.pv[0], rootPos)))
                std::abort();
            rm.policyLogitValid = true;
        }
        ++policySearchCounters.inferences;
    }
    std::stable_sort(challengers.begin(), challengers.begin() + order.size() - 1,
                     [](const RootMove* a, const RootMove* b) { return a->policyLogit > b->policyLogit; });
    for (usize i = 1; i < order.size(); ++i) order[i] = challengers[i - 1]->pv[0];
    ++policySearchCounters.nodes;
}

// MovePicker has already tried the incumbent and good captures. Its quiet
// prefix is history-sorted at depth >= 4. Use the cached root priors only
// where recent searches and history both leave room for another challenger.
void Search::Worker::promote_root_challenger(ExtMove* begin, ExtMove* end) {
    const auto eligible = rootMoves.begin() + pvIdx + 1;
    const auto last = rootMoves.begin() + pvLast;
    const int bonus = cutoffBonus[(popcount(rootPos.pieces()) - 1) / 8];
    ExtMove* leader = nullptr;
    ExtMove* best = nullptr;
    float leaderLogit = 0, bestGain = 0;
    Value leaderEval = VALUE_NONE;
    for (auto* it = begin; it != end && it->value > MovePicker::GoodQuietThreshold; ++it) {
        if (leader && leader->value - it->value >= bonus) break;
        const auto rm = std::find(eligible, last, Move(*it));
        if (rm == last) continue;
        if (!leader) {
            if (!rm->policyLogitValid || rm->averageScore == -VALUE_INFINITE) return;
            leader = it;
            leaderLogit = rm->policyLogit;
            leaderEval = rm->averageScore;
            continue;
        }
        if (!rm->policyLogitValid || rm->averageScore < leaderEval - PawnValue / 4) continue;
        const float advantage = (rm->policyLogit - leaderLogit) / policySearchTemperature;
        const float gain = bonus * std::min(advantage, 1.0f) - (leader->value - it->value);
        if (gain > bestGain) {
            best = it;
            bestGain = gain;
        }
    }
    if (best) {
        std::rotate(leader, best, best + 1);
        ++policySearchCounters.rootVisits;
    }
}

void Search::Worker::policy_search_moves(const Position& pos, PolicySearchMoves& result) {
    PolicySearchCacheEntry* entry = policySearchCache.empty() ? nullptr
      : &policySearchCache[pos.key() % policySearchCache.size()];
    if (entry && entry->valid && entry->key == pos.key()) {
        result = entry->list;
        ++policySearchCounters.cacheHits;
        return;
    }
    result.count = 0;
    for (Move move : MoveList<ALL>(pos))
        result.moves[result.count++].move = move;
    if (result.count <= 1) {
        if (result.count) result.moves[0].probability = 1.0f;
        ++policySearchCounters.forced;
    } else {
        ++policySearchCounters.inferences;
        float maximum = -std::numeric_limits<float>::infinity();
        auto scoreMoves = [&](auto&& scorer) {
            for (int i = 0; i < result.count; ++i) {
                const float score = scorer(result.moves[i].move);
                if (!std::isfinite(score)) {
                    sync_cout << "info string ERROR: Non-finite policy score" << sync_endl;
                    // Avoid global thread destructors from a worker-thread error.
                    std::_Exit(EXIT_FAILURE);
                }
                result.moves[i].probability = score;
                maximum = std::max(maximum, score);
            }
        };
        if (incrementalPolicyEnabled) {
            const auto hidden = incrementalPolicy.context(pos, verifyPolicyAccumulators);
            std::array<Move, MAX_MOVES> moves;
            std::array<float, MAX_MOVES> logits;
            for (int i = 0; i < result.count; ++i) moves[i] = result.moves[i].move;
            incrementalPolicy.scores(hidden, moves.data(), logits.data(), result.count, pos);
            int index = 0;
            scoreMoves([&](Move move) {
                const float value = logits[index++];
                if (verifyPolicyAccumulators && value != incrementalPolicy.score(hidden, move, pos))
                    std::abort();
                return value;
            });
        } else {
            alignas(Eval::NNUE::CacheLineSize) u8 features[Eval::NNUE::FeatureTransformer::BufferSize];
            network[numaAccessToken].policy_features(pos, accumulatorStack, refreshTable, features);
            const auto hidden = compactPolicy.context(features,
              file_of(pos.square<KING>(pos.side_to_move())) >= FILE_E,
              file_of(pos.square<KING>(~pos.side_to_move())) >= FILE_E);
            scoreMoves([&](Move move) { return compactPolicy.score(hidden, move, pos.side_to_move()); });
        }
        double sum = 0.0;
        for (int i = 0; i < result.count; ++i) {
            // A numerical floor prevents zero-probability moves, not an
            // assertion that the network's confidence is calibrated.
            result.moves[i].probability = std::max(1e-8f, std::exp((result.moves[i].probability - maximum) / policySearchTemperature));
            sum += result.moves[i].probability;
        }
        for (int i = 0; i < result.count; ++i)
            result.moves[i].probability /= float(sum);
        // The root asks for probabilities by move and orders its eligible
        // challengers itself. Sorting here and calculating unused entropy are
        // redundant. Keep normalization/flooring to preserve all ordering ties.
    }
    if (entry) {
        entry->key = pos.key(); entry->valid = true; entry->list = result;
    }
}


template<NodeType nodeType>
Value Search::Worker::search(
  Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, const bool cutNode) {

    constexpr bool PvNode   = nodeType != NonPV;
    constexpr bool rootNode = nodeType == Root;
    const bool     allNode  = !(PvNode || cutNode);

    // Dive into quiescence search when the depth reaches zero
    if (depth <= 0)
        return qsearch<PvNode ? PV : NonPV>(pos, ss, alpha, beta);

    // Limit the depth if extensions made it too large
    depth = std::min(depth, MAX_PLY - 1);

    // Check if we have an upcoming move that draws by repetition
    if (!rootNode && alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    PVMoves   pv;
    StateInfo st;

    Key   posKey;
    Move  move, excludedMove, bestMove;
    Depth extension, newDepth;
    Value bestValue, value, eval, maxValue, probCutBeta;
    bool  givesCheck, improving, priorCapture, opponentWorsening;
    bool  capture, ttCapture;
    int   priorReduction;
    Piece movedPiece;

    SearchedList capturesSearched;
    SearchedList quietsSearched;

    // Step 1. Initialize node
    ss->inCheck   = pos.checkers();
    priorCapture  = pos.captured_piece();
    Color us      = pos.side_to_move();
    ss->moveCount = 0;
    bestValue     = -VALUE_INFINITE;
    maxValue      = VALUE_INFINITE;

    ss->followPV = rootNode
                || ((ss - 1)->followPV
                    && (static_cast<usize>(ss->ply - 1) < lastIterationIdxPV.size()
                        && (ss - 1)->currentMove == lastIterationIdxPV[ss->ply - 1]));

    // Check for the available remaining time
    if (is_mainthread())
        main_manager()->check_time(*this);

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    if (!rootNode)
    {
        // Step 2. Check for aborted search or immediate draw
        if (threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : value_draw(nodes);

        // Step 3. Mate distance pruning. Even if we mate at the next move our score
        // would be at best mate_in(ss->ply + 1), but if alpha is already bigger because
        // a shorter mate was found upward in the tree then there is no need to search
        // because we will never beat the current alpha. Equal and opposite logic applies
        // when being mated. In either case, return a fail-high score.
        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    Square prevSq  = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    bestMove       = Move::none();
    priorReduction = (ss - 1)->reduction;
    (ss - 1)->reduction        = 0;
    ss->statScore              = 0;
    (ss + 2)->cutoffCnt        = 0;
    (ss + 1)->priorNMPFailHigh = 0;

    const auto correctionValue = correction_value(*this, pos, ss);

    // Step 4. Transposition table lookup
    excludedMove                   = ss->excludedMove;
    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);

    ss->ttHit    = ttHit;
    ttData.move  = rootNode ? rootMoves[pvIdx].pv[0] : ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule50_count()) : VALUE_NONE;
    ss->ttPv     = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);
    ttCapture    = ttData.move && pos.capture_stage(ttData.move);

    // Step 5. Static evaluation of the position
    Value unadjustedStaticEval = VALUE_NONE;

    // Skip early pruning when in check
    if (ss->inCheck)
        ss->staticEval = eval = (ss - 2)->staticEval;
    else if (excludedMove)
        unadjustedStaticEval = eval = ss->staticEval;
    else if (ss->ttHit)
    {
        // Never assume anything about values stored in TT
        unadjustedStaticEval = ttData.eval;
        if (!is_valid(unadjustedStaticEval))
            unadjustedStaticEval = evaluate(pos);

        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // ttValue can be used as a better position evaluation
        if (is_valid(ttData.value)
            && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
    }
    else
    {
        unadjustedStaticEval = evaluate(pos);
        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);

        // Static evaluation is saved as it was before adjustment by correction history
        ttWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED, Move::none(),
                       unadjustedStaticEval, tt.generation());
    }

    // Set up the improving flag, which is true if current static evaluation is
    // bigger than the previous static evaluation at our turn (if we were in
    // check at our previous move we go back until we weren't in check) and is
    // false otherwise. The improving flag is used in various pruning heuristics.
    // Similarly, opponentWorsening is true if our static evaluation is better
    // for us than at the last ply.
    improving         = ss->staticEval > (ss - 2)->staticEval;
    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

    // Hindsight adjustment of reductions based on static evaluation difference
    if (priorReduction >= 3 && !opponentWorsening)
        depth++;
    if (priorReduction >= 2 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > 166)
        depth--;

    // Step 6. At non-PV nodes we check for an early TT cutoff. Note that we
    //         always check the validity of the TT value because of access races.
    if (!PvNode && !excludedMove && is_valid(ttData.value)
        && ttData.depth > depth - (ttData.value <= beta))
    {
        // Case A: TT entry can produce a cutoff
        if ((ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
            && (cutNode == (ttData.value >= beta) || depth > 4))
        {
            // If the ttMove is quiet, update move sorting heuristics on TT hit
            if (ttData.move && ttData.value >= beta)
            {
                // Bonus for a quiet ttMove that fails high
                if (!ttCapture)
                    update_quiet_histories(pos, ss, *this, ttData.move, 131 * depth);

                // Extra penalty for early quiet moves of the previous ply
                if (prevSq != SQ_NONE && (ss - 1)->moveCount < 5 && !priorCapture)
                    update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2210);
            }

            // Partial workaround for the graph history interaction problem.
            // For high rule50 counts don't produce transposition table cutoffs.
            if (pos.rule50_count() < 96)
            {
                if (depth >= 7 && ttData.move && pos.legal(ttData.move)
                    && !is_decisive(ttData.value))
                {
                    pos.do_move(ttData.move, st);
                    Key nextPosKey                             = pos.key();
                    auto [ttHitNext, ttDataNext, ttWriterNext] = tt.probe(nextPosKey);
                    pos.undo_move(ttData.move);

                    // Check that the ttValue after the tt move would also trigger a cutoff
                    if (!is_valid(ttDataNext.value))
                        return ttData.value;

                    if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
                        return ttData.value;
                }
                else
                    return ttData.value;
            }
        }
        // Case B: No cutoff, but depth was sufficient, so the bound points the wrong way.
        else if (depth > 5)
        {
            assert(!(ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)));

            // If such a mismatch is the only reason cutoff failed, the TT entry is now useless
            // (*Scaler) This is very close to secondary aging, a known antiscaler. The depth
            // and window conditions seem to be key to decent scaling.
            ttWriter.penalize(1);
        }
    }

    // Step 7. Tablebases probe
    if (!rootNode && !excludedMove && tbConfig.cardinality)
    {
        int piecesCount = pos.count<ALL_PIECES>();

        if (piecesCount <= tbConfig.cardinality
            && (piecesCount < tbConfig.cardinality || depth >= tbConfig.probeDepth)
            && pos.rule50_count() == 0 && !pos.can_castle(ANY_CASTLING))
        {
            TB::ProbeState err;
            TB::WDLScore   wdl = TB::probe_wdl(pos, &err);

            // Force check of time on the next occasion
            if (is_mainthread())
                main_manager()->callsCnt = 0;

            if (err != TB::ProbeState::FAIL)
            {
                ++tbHits;

                int drawScore = tbConfig.useRule50 ? 1 : 0;

                Value tbValue = VALUE_TB - ss->ply;

                // Use the range VALUE_TB to VALUE_TB_WIN_IN_MAX_PLY to score
                value = wdl < -drawScore ? -tbValue
                      : wdl > drawScore  ? tbValue
                                         : VALUE_DRAW + 2 * wdl * drawScore;

                Bound b = wdl < -drawScore ? BOUND_UPPER
                        : wdl > drawScore  ? BOUND_LOWER
                                           : BOUND_EXACT;

                if (b == BOUND_EXACT || (b == BOUND_LOWER ? value >= beta : value <= alpha))
                {
                    ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, b,
                                   std::min(MAX_PLY - 1, depth + 6), Move::none(), VALUE_NONE,
                                   tt.generation());

                    return value;
                }

                if (PvNode)
                {
                    if (b == BOUND_LOWER)
                        bestValue = value, alpha = std::max(alpha, bestValue);
                    else
                        maxValue = value;
                }
            }
        }
    }

    if (ss->inCheck)
        goto moves_loop;

    // Use static evaluation difference to improve quiet move ordering
    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture)
    {
        int evalDiff = std::clamp(-int((ss - 1)->staticEval + ss->staticEval), -189, 194) + 60;
        mainHistory[~us][((ss - 1)->currentMove).raw()] << evalDiff * 11;
        if (!ttHit && type_of(pos.piece_on(prevSq)) != PAWN
            && ((ss - 1)->currentMove).type_of() != PROMOTION)
            sharedHistory.pawn_entry(pos)[pos.piece_on(prevSq)][prevSq] << evalDiff * 13;
    }


    // Step 8. Razoring
    // If eval is really low, skip search entirely and return the qsearch value
    if (allNode && eval < alpha - 342 * depth && !seekMate)
        return qsearch<NonPV>(pos, ss, alpha, beta);

    // Step 9. Futility pruning: child node
    // The depth condition is important for mate finding. It should NOT be tuned.
    if (!ss->ttPv && depth < (seekMate ? 6 : 19) && eval >= beta && (!ttData.move || ttCapture)
        && !is_loss(beta) && !is_win(eval))
    {
        Value futilityMult = std::min(45 + depth * 4, 85);
        futilityMult -= 20 * !ss->ttHit;

        Value futilityMargin = futilityMult * depth
                             - (2789 * improving + 335 * opponentWorsening) * futilityMult / 1024
                             + std::abs(correctionValue) / 198435;

        if (eval - futilityMargin >= beta)
            return (661 * beta + 363 * eval) / 1024;
    }

    // Step 10. Null move search with verification search
    if (cutNode
        && ss->staticEval + 50 * ss->priorNMPFailHigh >= beta - 13 * depth - 47 * improving + 365
        && !excludedMove && pos.non_pawn_material(us) && ss->ply >= nmpMinPly && beta >= -2000)
    {
        assert((ss - 1)->currentMove != Move::null());

        // Null move dynamic reduction based on depth
        Depth R = 7 + depth / 3 + std::max((ss->staticEval - beta) / 256, 0);
        R -= seekMate * R / 2;
        do_null_move(pos, st, ss);

        Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);

        undo_null_move(pos);

        // Do not return unproven mate or TB scores
        if (nullValue >= beta && !is_win(nullValue))
        {
            if (nmpMinPly || depth < 16)
            {
                ++ss->priorNMPFailHigh;
                return nullValue;
            }

            // Recursive verification is not allowed
            assert(!nmpMinPly);

            // Do verification search at high depths, with null move pruning
            // disabled until ply exceeds nmpMinPly.
            nmpMinPly = ss->ply + 3 * (depth - R) / 4;

            Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);

            nmpMinPly = 0;

            if (v >= beta)
            {
                ++ss->priorNMPFailHigh;
                return nullValue;
            }
        }
    }

    improving |= ss->staticEval >= beta;

    // Step 11. Internal iterative reductions
    // At sufficient depth, reduce depth for PV/Cut nodes without a TTMove.
    // (*Scaler) Making IIR more aggressive scales poorly.
    if (!ss->followPV && !allNode && depth >= 6 && !ttData.move)
        depth--;

    // Step 12. ProbCut
    // If we have a good enough capture (or queen promotion) and a reduced search
    // returns a value much above beta, we can (almost) safely prune the previous move.
    probCutBeta = beta + 241 - 64 * improving;
    if (depth >= 3 && !is_decisive(beta) && !(is_valid(ttData.value) && ttData.value < probCutBeta))
    {
        assert(probCutBeta < VALUE_INFINITE && probCutBeta > beta);

        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory);
        Depth      probCutDepth = depth - (improving ? 5 : 3);

        while ((move = mp.next_move()) != Move::none())
        {
            assert(move.is_ok());

            if (move == excludedMove)
                continue;

            capture = pos.capture_stage(move);
            assert(capture);

            do_move(pos, move, st, pos.gives_check(move), capture, ss);
            // Perform a preliminary qsearch to verify that the move holds
            value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

            // If the qsearch held, perform the regular search
            if (value >= probCutBeta && probCutDepth > 0)
                value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1, probCutDepth,
                                       !cutNode);

            undo_move(pos, move);

            if (value >= probCutBeta)
            {
                // Save ProbCut data into transposition table
                ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                               probCutDepth + 1, move, unadjustedStaticEval, tt.generation());

                if (!is_decisive(value))
                    return value - (probCutBeta - beta);
            }
        }
    }

moves_loop:  // When in check, search starts here

    // Step 13. A small ProbCut idea
    probCutBeta = beta + 428;
    if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4 && ttData.value >= probCutBeta
        && !is_decisive(beta) && is_valid(ttData.value) && !is_decisive(ttData.value) && !seekMate)
        return probCutBeta;

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory, (ss - 3)->continuationHistory,
      (ss - 4)->continuationHistory, (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};


    MovePicker mp(pos, ttData.move, depth, &mainHistory, &lowPlyHistory, &captureHistory, contHist,
                  &sharedHistory, ss->ply);

#ifdef SF_SEARCH_SAMPLER
    SearchSample sample;
    const bool sampleEligible = is_mainthread() && !excludedMove && !pos.is_chess960()
                             && searchSampler.enabled(depth);
#endif
    // Consult only when search has no hash move to lead with. PV nodes and
    // expected cut nodes can repay ordering work; excluded-move searches must
    // retain their normal ordering. The callback adds a history ambiguity gate.
    const bool consultPolicy = !rootNode && internalPolicyEnabled
                            && depth >= internalPolicyMinDepth && !ttData.move
                            && !excludedMove && (PvNode || cutNode);
    if (policyDiagnostics && internalPolicyEnabled && !rootNode && depth >= internalPolicyMinDepth) {
        // Sequential rejection reasons among nodes reaching the move-picker.
        ++policyDiagnostic.deep;
        if (ttData.move) ++policyDiagnostic.tt;
        else if (excludedMove) ++policyDiagnostic.excluded;
        else if (!(PvNode || cutNode)) ++policyDiagnostic.nodeType;
        else if (pos.checkers()) ++policyDiagnostic.check;
        else ++policyDiagnostic.eligible;
    }
    struct QuietPolicyContext {
        Worker* worker;
        const Position* position;
        bool policyEnabled;
        Move transpositionMove;
#ifdef SF_SEARCH_SAMPLER
        SearchSample* sample;
        bool eligible;
        int depth, ply, alpha, beta;
        Move ttMove;
#endif
        Move promoted = Move::none();
        int eventIndex = -1;
    } quietPolicyContext{this, &pos, consultPolicy, ttData.move
#ifdef SF_SEARCH_SAMPLER
        , &sample, sampleEligible, depth, ss->ply, alpha, beta, ttData.move
#endif
    };
    if constexpr (!rootNode) {
        if (!pos.checkers() && (consultPolicy
#ifdef SF_SEARCH_SAMPLER
            || sampleEligible
#endif
            ))
            mp.set_quiet_policy(&quietPolicyContext, [](void* opaque, ExtMove* begin, ExtMove* end) {
                auto& ctx = *static_cast<QuietPolicyContext*>(opaque);
                auto& worker = *ctx.worker;
                const Position& position = *ctx.position;
#ifdef SF_SEARCH_SAMPLER
                if (ctx.eligible)
                    searchSampler.capture(*ctx.sample, position, begin, end, ctx.ttMove,
                                          ctx.depth, ctx.ply, ctx.alpha, ctx.beta, worker.nodes.load(std::memory_order_relaxed));
#endif
                if (!ctx.policyEnabled) return;
                using Clock = std::chrono::steady_clock;
                const bool diagnostic = worker.policyDiagnostics;
                struct GateTimer {
                    u64* target;
                    Clock::time_point start;
                    ~GateTimer() {
                        if (target) *target += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
                    }
                } timer{diagnostic ? &worker.policyDiagnostic.gateNs : nullptr,
                        diagnostic ? Clock::now() : Clock::time_point{}};
                if (diagnostic) ++worker.policyDiagnostic.callbacks;
                // Bound projection work to at most one call per 4096 visited
                // nodes (plus one initial call), per worker and root search.
                if (worker.internalPolicyCalls > worker.nodes.load(std::memory_order_relaxed) / 4096) {
                    if (diagnostic) ++worker.policyDiagnostic.budget;
                    return;
                }
                // Only arbitrate between the first two legal quiet moves in
                // the good-quiet stage. Its prefix is already history-sorted.
                // Never pull a deferred move across the stage boundary.
                ExtMove* historyFirst = nullptr;
                ExtMove* historySecond = nullptr;
                int legalCount = 0;
                for (auto* it = begin; it != end; ++it) {
                    assert(position.legal(*it)); // QUIETS already emits legal moves.
                    ++legalCount;
                    if (it->value > MovePicker::GoodQuietThreshold) {
                        if (!historyFirst) historyFirst = it;
                        else if (!historySecond) historySecond = it;
                    }
                    if (legalCount >= 4 && historySecond) break;
                }
                if (legalCount < 4 || !historySecond) {
                    if (diagnostic) ++worker.policyDiagnostic.candidates;
                    return;
                }
                if (historyFirst->value - historySecond->value > worker.internalPolicyBonus) {
                    if (diagnostic) ++worker.policyDiagnostic.history;
                    return;
                }
                const auto inferenceStart = diagnostic ? Clock::now() : Clock::time_point{};
                alignas(Eval::NNUE::CacheLineSize) u8 features[Eval::NNUE::FeatureTransformer::BufferSize];
                worker.network[worker.numaAccessToken].policy_features(
                  position, worker.accumulatorStack, worker.refreshTable, features);
                const auto hidden = worker.compactPolicy.context(features, file_of(position.square<KING>(position.side_to_move())) >= FILE_E, file_of(position.square<KING>(~position.side_to_move())) >= FILE_E);
                ++worker.internalPolicyCalls;
                // Local ranking needs only two logits, not a calibrated
                // probability over every legal move. Keep history order on ties.
                const float firstScore = worker.compactPolicy.score(hidden, *historyFirst, position.side_to_move());
                const float secondScore = worker.compactPolicy.score(hidden, *historySecond, position.side_to_move());
                const bool finite = std::isfinite(firstScore) && std::isfinite(secondScore);
                const bool promote = finite && secondScore > firstScore;
                if (diagnostic) {
                    worker.policyDiagnostic.inferenceNs += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - inferenceStart).count();
                    if (!finite) ++worker.policyDiagnostic.confidence;
                    else if (!promote) ++worker.policyDiagnostic.agreement;
                }
                if (promote) {
                    ExtMove* best = historySecond;
                    const ExtMove promotedMove = *best;
                    if (diagnostic) {
                        ctx.promoted = promotedMove;
                        const int rank = int(best - begin) + 1;
                        ++worker.policyDiagnostic.promotions;
                        worker.policyDiagnostic.rankSum += rank;
                        worker.policyDiagnostic.rankMax = std::max(worker.policyDiagnostic.rankMax, u64(rank));
                        if (worker.policyPromotionEvents.size() < 256) {
                            ctx.eventIndex = int(worker.policyPromotionEvents.size());
                            worker.policyPromotionEvents.push_back({position.fen(), UCIEngine::move(promotedMove, position.is_chess960()), rank});
                        }
                    }
                    // Exchange only these two candidates, preserving their
                    // history values and the order of every other move.
                    std::swap(*historyFirst, *historySecond);
                }
            });
    }

    struct HybridPolicyContext {
        Worker* worker;
        const Position* position;
        Move ttMove;
        bool allowInference;
        bool useMarginLMR;
        QuietPolicyReductions reductions;
        int delayedDepth;
        bool retry = false;
        Move firstQuiet = Move::none(), promotedAlternative = Move::none();
        Move observed = Move::none();
        PolicySearchCounters::CutoffSample* sample = nullptr;
    } hybridPolicy{this, &pos, ttData.move, cutoffPolicyOrdering ? cutNode
                                : (depth >= economicalMinDepth && (cutNode || PvNode)),
                       policySearchReductions && depth >= 6 && cutNode && !PvNode && !ss->ttPv && !seekMate
                         && !is_decisive(alpha) && !is_decisive(beta), {},
                       delayedQuietPolicy && !PvNode && cutNode && depth >= 5 && depth < 8
                         && (delayedMinPieces == 1 || popcount(pos.pieces()) >= delayedMinPieces)
                         ? depth : 0};
    // Amortize policy over a substantial remaining subtree, at any root ply.
    // Quiescence, evasions and singular verification retain their normal order.
    if (policySearchEnabled && !rootNode && !excludedMove && !ss->inCheck
        && (depth >= 6 || hybridPolicy.delayedDepth
            || (depth == 5 && incrementalPolicyEnabled && !cutoffPolicyOrdering && economicalMinDepth == 5)))
        mp.set_quiet_policy(&hybridPolicy, [](void* opaque, ExtMove* begin, ExtMove* end) {
            auto& ctx = *static_cast<HybridPolicyContext*>(opaque);
            auto& worker = *ctx.worker;
            auto& counters = worker.policySearchCounters;
            const Position& position = *ctx.position;
            if (worker.incrementalPolicyEnabled) {
                if (ctx.delayedDepth && !ctx.retry) {
                    // This prefix is history-sorted. TT has already been tried.
                    const auto* first = begin + (*begin == ctx.ttMove);
                    if (first < end && first->value > MovePicker::GoodQuietThreshold)
                        ctx.firstQuiet = *first;
                    // Pure depth 8 has pre-search cache reuse only from depth 6.
                    if (ctx.delayedDepth == 5) return;
                }
                const Key key = position.key();
                const bool cachedPosition = worker.quietPolicyCache.contains(key);
                // A cold position can be rejected before even collecting heads.
                // Cached positions still get a chance when the work budget is spent.
                if (!cachedPosition && !ctx.allowInference) {
                    ++counters.quietNodeSkipped;
                    return;
                }
                const u64 allowance = quiet_policy_allowance(worker.nodes.load(std::memory_order_relaxed));
                if (!cachedPosition && counters.quietWork
                    + quiet_policy_cost(worker.incrementalPolicy.width(), 2) > allowance) {
                    ++counters.quietBudgetSkipped;
                    return;
                }
                QuietPolicyCandidates candidates;
                const int bucket = worker.cutoffPolicyOrdering ? (popcount(position.pieces()) - 1) / 8 : 0;
                const int bonus = worker.cutoffBonus[bucket];
                if (worker.cutoffPolicyOrdering)
                    candidates.collect(begin, end, ctx.ttMove, MovePicker::GoodQuietThreshold,
                                       2, bonus - 1);
                else candidates.collect(begin, end, ctx.ttMove, MovePicker::GoodQuietThreshold);
                if (candidates.count < 2) {
                    ++counters.quietHistorySkipped;
                    return;
                }
                if (ctx.retry && worker.policySearchStats) ++counters.delayedShortlists;
                std::array<bool, QuietPolicyCandidates::Limit> missing{};
                const int missingCount = worker.quietPolicyCache.get_candidates(
                  key, candidates, missing);
                // Expected all-nodes normally need every move searched. Reuse
                // available logits there, but spend fresh inference on PV/cut nodes.
                if (missingCount && !ctx.allowInference) {
                    ++counters.quietNodeSkipped;
                    return;
                }
                const u64 cost = quiet_policy_cost(worker.incrementalPolicy.width(), missingCount);
                if (cost && counters.quietWork + cost > allowance) {
                    ++counters.quietBudgetSkipped;
                    return;
                }
                if (missingCount || worker.verifyPolicyAccumulators) {
                    const auto hidden = worker.incrementalPolicy.context(
                      position, worker.verifyPolicyAccumulators);
                    if (missingCount) {
                        counters.quietWork += cost;
                        ++counters.quietCalls;
                        if (ctx.retry && worker.policySearchStats) ++counters.delayedCalls;
                        ++counters.inferences;
                    }
                    std::array<Move, QuietPolicyCandidates::Limit> moves;
                    std::array<float, QuietPolicyCandidates::Limit> logits;
                    int count = 0;
                    for (int i = 0; i < candidates.count; ++i)
                        if (missing[i]) moves[count++] = *candidates.slots[i];
                    worker.incrementalPolicy.scores(hidden, moves.data(), logits.data(), count, position);
                    count = 0;
                    for (int i = 0; i < candidates.count; ++i) {
                        if (missing[i]) {
                            candidates.logits[i] = logits[count++];
                            ++counters.quietOutputScores;
                        }
                        if (worker.verifyPolicyAccumulators
                            && candidates.logits[i] != worker.incrementalPolicy.score(
                              hidden, *candidates.slots[i], position))
                            std::abort();
                        if (!std::isfinite(candidates.logits[i])) return;
                    }
                    // Delayed scores are for this ordering decision only. Keep
                    // them out of the shared ordering/LMR cache on later visits.
                    if (!ctx.retry)
                        for (int i = 0; i < candidates.count; ++i)
                            if (missing[i]) worker.quietPolicyCache.put(
                              key, *candidates.slots[i], candidates.logits[i]);
                }
                counters.quietHeadCacheHits += candidates.count - missingCount;
                counters.quietCacheHits += missingCount == 0;
                ++counters.nodes;
                ctx.firstQuiet = Move::none(); // A complete initial cache hit needs no retry.
                if (ctx.useMarginLMR) ctx.reductions.record(candidates);
                const Move historyLeader = *candidates.slots[0];
                const int promoted = worker.cutoffPolicyOrdering
                  ? candidates.promote_pair(worker.policySearchTemperature, bonus)
                  : candidates.promote(worker.policySearchTemperature);
                if (ctx.retry && worker.policySearchStats && historyLeader != *candidates.slots[0]) {
                    ctx.promotedAlternative = *candidates.slots[0];
                    ++counters.delayedNewLeaders;
                }
                if (worker.cutoffPolicyOrdering && worker.policySearchStats && ctx.allowInference) {
                    ctx.sample = &counters.cutoffSamples[bucket][promoted];
                    ctx.observed = *candidates.slots[0];
                }
                counters.quietReorders += promoted != 0;
                counters.quietPromotions += promoted;
                return;
            }
            // Legacy compact-network path retains its previous full-quiet order.
            // At most one quiet inference per 256 visited nodes, plus one
            // initial call. Root inference has its own existing path.
            if (counters.quietCalls > worker.nodes.load(std::memory_order_relaxed) / 256) {
                ++counters.quietBudgetSkipped;
                return;
            }
            // Work only with supplied, legal good quiets. TT was searched already;
            // deferred quiets keep their order and stage membership unchanged.
            struct RankedQuiet { ExtMove move; float score; };
            std::array<RankedQuiet, MAX_MOVES> ranked;
            std::array<ExtMove*, MAX_MOVES> slots;
            int count = 0;
            int first = std::numeric_limits<int>::min();
            int second = first;
            for (auto* it = begin; it != end; ++it) {
                if (it->value <= MovePicker::GoodQuietThreshold || *it == ctx.ttMove) continue;
                assert(position.legal(*it)); // Shared legal generator already checked this.
                slots[count] = it;
                ranked[count++] = {*it, 0.0f};
                if (it->value > first) { second = first; first = it->value; }
                else second = std::max(second, it->value);
            }
            if (count < 4) return;
            // The network can overcome at most 2048 history points. A larger
            // lead cannot change the first move, so avoid the inference entirely.
            if (first - second > 2048) {
                ++counters.quietHistorySkipped;
                return;
            }
            alignas(Eval::NNUE::CacheLineSize) u8 features[Eval::NNUE::FeatureTransformer::BufferSize];
            worker.network[worker.numaAccessToken].policy_features(
              position, worker.accumulatorStack, worker.refreshTable, features);
            const auto hidden = worker.compactPolicy.context(features,
              file_of(position.square<KING>(position.side_to_move())) >= FILE_E,
              file_of(position.square<KING>(~position.side_to_move())) >= FILE_E);
            ++counters.quietCalls;
            ++counters.inferences;
            ++counters.nodes;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < count; ++i) {
                ranked[i].score = worker.compactPolicy.score(
                  hidden, ranked[i].move, position.side_to_move());
                // Preserve the complete original ordering on invalid output.
                if (!std::isfinite(ranked[i].score)) return;
                maximum = std::max(maximum, ranked[i].score);
            }
            // Logit differences are shift invariant. Equal logits leave history
            // untouched; small differences have small effects without claiming
            // calibrated confidence. No softmax, entropy or all-move generation.
            for (int i = 0; i < count; ++i)
                ranked[i].score = float(ranked[i].move.value) + 1024.0f
                  * std::clamp((ranked[i].score - maximum) / worker.policySearchTemperature,
                               -2.0f, 0.0f);
            // Small candidate lists: stable in-place insertion avoids allocation.
            for (int i = 1; i < count; ++i) {
                const auto candidate = ranked[i];
                int j = i;
                while (j && ranked[j - 1].score < candidate.score) {
                    ranked[j] = ranked[j - 1];
                    --j;
                }
                ranked[j] = candidate;
            }
            // Retain raw history values for Stockfish's pruning/reduction logic.
            for (int i = 0; i < count; ++i) *slots[i] = ranked[i].move;
        }, incrementalPolicyEnabled ? 2 : 4);

    if constexpr (rootNode)
        if (cutoffPolicyOrdering && useRootChallenger && depth >= 4 && !ss->inCheck && pvLast - pvIdx >= 3)
            mp.set_quiet_policy(this, [](void* worker, ExtMove* begin, ExtMove* end) {
                static_cast<Worker*>(worker)->promote_root_challenger(begin, end);
            }, 2);

    std::vector<Move> policyOrder;
    usize policyNext = 0;
    if constexpr (rootNode) {
        if (!rootPolicy.empty() && rootDepth == 1) {
            for (usize i = pvIdx; i < pvLast; ++i) policyOrder.push_back(rootMoves[i].pv[0]);
            auto probability = [&](Move candidate) {
                for (const auto& entry : rootPolicy)
                    if (entry.first == candidate) return entry.second;
                return 0.0f;
            };
            // Preserve the incumbent; policy ranks only the eligible challengers.
            std::stable_sort(policyOrder.begin() + (policyOrder.empty() ? 0 : 1), policyOrder.end(),
                [&](Move a, Move b) { return probability(a) > probability(b); });
        }
    }
    if constexpr (rootNode) {
        if (cutoffPolicyOrdering && rootDepth == 1)
            cutoff_root_order(policyOrder);
        else if (policySearchEnabled && rootDepth == 1) {
            PolicySearchMoves priors;
            policy_search_moves(pos, priors);
            ++policySearchCounters.nodes;
            policyOrder.clear();
            for (usize i = pvIdx; i < pvLast; ++i) policyOrder.push_back(rootMoves[i].pv[0]);
            auto priority = [&](Move candidate) { return priors.probability(candidate); };
            // Retain Stockfish's incumbent and MultiPV/TB eligibility range.
            if (policyOrder.size() > 1)
                std::stable_sort(policyOrder.begin() + 1, policyOrder.end(),
                                 [&](Move a, Move b) { return priority(a) > priority(b); });
        }
    }
    auto nextMove = [&]() {
        if constexpr (rootNode)
            if (!policyOrder.empty())
                return policyNext < policyOrder.size() ? policyOrder[policyNext++] : Move::none();
        return mp.next_move();
    };

    value = bestValue;

    int moveCount = 0;

    // Step 14. Loop through all legal moves until no moves remain
    // or a beta cutoff occurs.
    while ((move = nextMove()) != Move::none())
    {
        assert(move.is_ok());

        if (move == excludedMove)
            continue;

        // At root obey the "searchmoves" option and skip moves not listed in Root
        // Move List. In MultiPV mode we also skip PV moves that have been already
        // searched and those of lower "TB rank" if we are in a TB root position.
        if (rootNode && !std::count(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast, move))
            continue;

        ss->moveCount = ++moveCount;

        if (rootNode && is_mainthread() && nodes > NODES_LIMIT_OUTPUT)
            main_manager()->updates.onIter(
              {depth, UCIEngine::move(move, pos.is_chess960()), moveCount + pvIdx});
        if (PvNode)
            (ss + 1)->pv = nullptr;

        extension  = 0;
        capture    = pos.capture_stage(move);
        movedPiece = pos.moved_piece(move);
        givesCheck = pos.gives_check(move);

        // Calculate new depth for this move
        newDepth = depth - 1;

        int r = reduction(improving, depth, moveCount);

        // Decrease reduction for PvNodes
        if (PvNode)
            r -= 512;

        // Increase reduction for ttPv nodes
        // (*Scaler) Larger values scale well.
        if (ss->ttPv)
            r += 929;

        // Step 15. Pruning at shallow depths.
        // Depth conditions are important for mate finding.
        if (!rootNode && pos.non_pawn_material(us) && !is_loss(bestValue))
        {
            // Skip quiet moves if movecount exceeds our threshold
            if (moveCount >= (3 + depth * depth) / (2 - improving))
                mp.skip_quiet_moves();

            // Reduced depth of the next LMR search
            int lmrDepth = newDepth - r / 1024;

            if (capture || givesCheck)
            {
                Piece capturedPiece = pos.piece_on(move.to_sq());
                int   captHist = captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];

                // Futility pruning for captures
                if (!givesCheck && lmrDepth < 8)
                {
                    Value futilityValue = ss->staticEval + 234 + 247 * lmrDepth
                                        + PieceValue[capturedPiece] + 134 * captHist / 1024;

                    if (futilityValue <= alpha)
                        continue;
                }

                // SEE based pruning for captures and checks.
                // Avoid pruning sacrifices of our last piece for stalemate.
                int margin = 177 * depth + captHist * 34 / 1024;
                if ((alpha >= VALUE_DRAW || pos.non_pawn_material(us) != PieceValue[movedPiece])
                    && !pos.see_ge(move, -margin))
                    continue;
            }
            else if (!ss->followPV || !PvNode)
            {
                int history = (*contHist[0])[movedPiece][move.to_sq()]
                            + (*contHist[1])[movedPiece][move.to_sq()]
                            + sharedHistory.pawn_entry(pos)[movedPiece][move.to_sq()];

                // Continuation history based pruning
                if (history < -4136 * depth)
                    continue;

                history += 69 * mainHistory[us][move.raw()] / 32;

                // (*Scaler): Generally, lower divisors scale well
                lmrDepth += history / lmr_divisor(depth);

                Value futilityValue =
                  ss->staticEval + 119 * lmrDepth + 90 * (ss->staticEval > alpha) + 164;

                // Futility pruning: parent node
                // (*Scaler): Generally, more frequent futility pruning scales well
                if (!ss->inCheck && lmrDepth < 12 && futilityValue <= alpha)
                {
                    if (bestValue <= futilityValue && !is_decisive(bestValue)
                        && !is_win(futilityValue))
                        bestValue = futilityValue;
                    continue;
                }

                lmrDepth = std::max(lmrDepth, 0);

                // Prune moves with negative SEE
                if (!pos.see_ge(move, -23 * lmrDepth * lmrDepth))
                    continue;
            }
        }

        // Step 16. Singular Extensions
        //
        // We check for "only moves": if one move fails high on (alpha, beta) but all
        // others fail low on (alpha-s, beta-s), then that move is singular. Singular
        // moves are extended to better estimate the result of the (putatively)
        // forced line. If it's non-singular, we may do some pruning or reduction.
        //
        // Recursive excluded search is avoided. The `excludedMove` mechanism
        // was historically a hack and remains a bit fragile.
        //
        // (*Scaler) Generally, higher singularBeta (i.e closer to ttValue)
        // and lower extension margins scale well.
        if (!rootNode && move == ttData.move && !excludedMove && depth >= 6 + ss->ttPv
            && is_valid(ttData.value) && !is_decisive(ttData.value) && (ttData.bound & BOUND_LOWER)
            && ttData.depth >= depth - 3 && !is_shuffling(move, ss, pos) && !seekMate)
        {
            Value singularBeta  = ttData.value - (59 + 66 * (ss->ttPv && !PvNode)) * depth / 63;
            Depth singularDepth = newDepth / 2;

            ss->excludedMove = move;
            value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
            ss->excludedMove = Move::none();

            if (value < singularBeta)
            {
                int corrValAdj   = std::abs(correctionValue) / 198368;
                int doubleMargin = -2 + 204 * PvNode - 152 * !ttCapture - corrValAdj
                                 - 1175 * ttMoveHistory / 114178 - (ss->ply > rootDepth) * 38;
                int tripleMargin = 70 + 279 * PvNode - 188 * !ttCapture + 81 * ss->ttPv - corrValAdj
                                 - (ss->ply > rootDepth) * 43;

                extension =
                  1 + (value < singularBeta - doubleMargin) + (value < singularBeta - tripleMargin);

                depth++;
            }

            // Multi-cut pruning
            // Our ttMove is assumed to fail high based on the bound of the TT entry,
            // and if after excluding the ttMove with a reduced search we fail high
            // over the original beta, we assume this expected cut-node is not
            // singular (multiple moves fail high), and we can prune the whole
            // subtree by returning a softbound.
            else if (value >= beta && !is_decisive(value))
            {
                ttMoveHistory << -421 - 110 * depth;

                if (!ss->inCheck && value > ss->staticEval)
                {
                    const int bonus =
                      std::clamp(int(value - ss->staticEval) * 664 / 1024,
                                 -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
                    update_correction_history(pos, ss, *this, bonus);
                }

                return value;
            }

            // Negative extensions
            // If other moves failed high over (ttValue - margin) without the
            // ttMove on a reduced search, but we cannot do multi-cut because
            // (ttValue - margin) is lower than the original beta, we do not know
            // if the ttMove is singular or can do a multi-cut, so we reduce the
            // ttMove in favor of other moves based on some conditions:

            // If the ttMove is assumed to fail high over current beta or
            // if we are on a cutNode
            else if (ttData.value >= beta || cutNode)
                extension = -3;
        }

        const bool diagnosticPromotion = policyDiagnostics && move == quietPolicyContext.promoted;
        const bool observeCutoff = hybridPolicy.observed == move;
        bool researched = false;
        u64 nodeCount = (rootNode || diagnosticPromotion || observeCutoff) ? u64(nodes) : 0;
        if (diagnosticPromotion) {
            ++policyDiagnostic.searched;
            if (quietPolicyContext.eventIndex >= 0)
                policyPromotionEvents[quietPolicyContext.eventIndex].searched = true;
        }

        // Step 17. Make the move
        do_move(pos, move, st, givesCheck, capture, ss);

        // Add extension to new depth
        newDepth += extension;

        // Step 18. Compute and apply late moves reductions/extensions (LMR)

        // Decrease reduction for PvNodes (*Scaler)
        if (ss->ttPv)
            r -= 3023 + PvNode * 1004 + (ttData.value > alpha) * 885
               + (ttData.depth >= depth) * (816 + cutNode * 940);

        // Base reduction offset to compensate for other tweaks
        r += 697;

        r -= moveCount * 65;
        r -= std::abs(correctionValue) / 26310;

        // Increase reduction for cut nodes
        if (cutNode)
            r += 4026 + 933 * !ttData.move;

        // Increase reduction if ttMove is a capture
        if (ttCapture)
            r += 1079;

        // Increase reduction if next ply has a lot of fail high
        if ((ss + 1)->cutoffCnt > 1)
            r += 264 + 1095 * ((ss + 1)->cutoffCnt > 2) + 1138 * allNode;

        // For first picked move (ttMove) reduce reduction
        else if (move == ttData.move)
            r -= 2179;

        if (capture)
            ss->statScore = 873 * int(PieceValue[pos.captured_piece()]) / 128
                          + captureHistory[movedPiece][move.to_sq()][type_of(pos.captured_piece())];
        else
            ss->statScore =
              (2252 * mainHistory[us][move.raw()] + 1126 * (*contHist[0])[movedPiece][move.to_sq()]
               + 1093 * (*contHist[1])[movedPiece][move.to_sq()])
              / 1024;

        // Decrease/increase reduction for moves with a good/bad history
        r -= ss->statScore * 439 / 4096;

        if (!capture && !is_decisive(alpha))
            r += 3 * std::clamp(alpha - eval, -64, 96);

        // Scale up reductions for expected ALL nodes
        if (allNode)
            r += r * 276 / (256 * depth + 268);

        // When seeking a mate, reduce non-checking moves more and checking moves less
        if (seekMate && r > 0)
            r += (givesCheck ? -r : r) / 4;

        // Apply the computed LMR
        if (depth >= 2 && moveCount > 1)
        {
            // In general we want to cap the LMR depth search at newDepth, but when
            // reduction is negative, we allow this move a limited search extension
            // beyond the first move depth. To avoid search explosion, extensions
            // are not allowed deep, relative to rootDepth, in the search tree.
            Depth d =
              std::max(1, newDepth + std::min(-r / 1024, ss->ply < 2 * rootDepth ? 2 : 0)) + PvNode;

            // At non-PV cut nodes r is already positive here if a normal
            // reduction applies. A margin can cross at most one ply boundary.
            // Keep at least two plies in the probe. Earlier pruning never sees it.
            bool marginReduced = false;
            if (hybridPolicy.reductions.count && d > 2 && r >= 1024 && !capture && !givesCheck
                && move.type_of() == NORMAL && ss->statScore < 2048)
            {
                const int extra = hybridPolicy.reductions.adjustment(move, improving);
                marginReduced = (r % 1024) + extra >= 1024;
                d -= int(marginReduced);
                if (marginReduced && policySearchStats) ++policySearchCounters.marginLmrReduced;
            }
            ss->reduction = newDepth - d;
            value         = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);
            ss->reduction = 0;

            // Do a full-depth search when reduced LMR search fails high
            // (*Scaler) Shallower searches here don't scale well
            if (value > alpha)
            {
                if (marginReduced && policySearchStats) ++policySearchCounters.marginLmrFailHigh;
                // Adjust full-depth search based on LMR results - if the result was
                // good enough search deeper, if it was bad enough search shallower.
                const bool doDeeperSearch    = d < newDepth && value > bestValue + 53;
                const bool doShallowerSearch = value < bestValue + 8;

                newDepth += doDeeperSearch - doShallowerSearch;

                if (newDepth > d) {
                    if (marginReduced && policySearchStats) ++policySearchCounters.marginLmrResearched;
                    researched = true;
                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
                }

                // Post LMR continuation history updates
                update_continuation_histories(ss, movedPiece, move.to_sq(), 1334);
            }
        }

        // Step 19. Full-depth search when LMR is skipped
        else if (!PvNode || moveCount > 1)
        {
            // Increase reduction if ttMove is not present
            if (!ttData.move)
                r += 1127;

            // If expected reduction is high, we reduce search depth here
            value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha,
                                   newDepth - (r > 5234) - (r > 5487 && newDepth > 2), !cutNode);
        }

        // Step 20. For PV nodes only, do a full PV search on the first move
        // or after a fail high, otherwise let the parent node fail low with
        // value <= alpha and try another move.
        if (PvNode && (moveCount == 1 || value > alpha))
        {
            (ss + 1)->pv = &pv;
            (ss + 1)->pv->clear();

            // Extend move from transposition table if we are about to dive
            // into qsearch. Decisive score handling improves mate finding
            // and retrograde analysis.
            if (move == ttData.move
                && ((is_valid(ttData.value) && is_decisive(ttData.value) && ttData.depth > 0)
                    || ttData.depth > 1))
                newDepth = std::max(newDepth, 1);

            value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
        }

        // Step 21. Undo move
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        if (diagnosticPromotion) {
            const u64 spent = u64(nodes) - nodeCount;
            const bool completed = !threads.stop.load(std::memory_order_relaxed);
            policyDiagnostic.moveNodes += spent;
            policyDiagnostic.completed += completed;
            policyDiagnostic.cutoffs += completed && value >= beta;
            if (quietPolicyContext.eventIndex >= 0) {
                auto& event = policyPromotionEvents[quietPolicyContext.eventIndex];
                event.nodes = spent;
                event.completed = completed;
                event.cutoff = completed && value >= beta;
            }
        }

        // Step 22. Check for a new best move
        // If a stop occurred, the value of the search cannot be trusted,
        // and we return immediately without updating the best move,
        // principal variation or transposition table.
        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (move == hybridPolicy.promotedAlternative) {
            ++policySearchCounters.delayedLeaderSearched;
            policySearchCounters.delayedLeaderCutoffs += value >= beta;
        }
        if (move == hybridPolicy.firstQuiet) {
            hybridPolicy.firstQuiet = Move::none();
            if (policySearchStats) {
                ++policySearchCounters.delayedFirstSearched;
                policySearchCounters.delayedFirstFailed += value < beta;
            }
            if (value < beta) {
                hybridPolicy.retry = true;
                hybridPolicy.allowInference = true;
                hybridPolicy.useMarginLMR = false;
                mp.apply_remaining_quiet_policy();
            }
        }
        if (observeCutoff) {
            auto& sample = *hybridPolicy.sample;
            ++sample.searched;
            sample.cutoffs += value >= beta;
            sample.nodes += u64(nodes) - nodeCount;
            sample.researches += researched;
        }

        if (rootNode)
        {
            RootMove& rm = *std::find(rootMoves.begin(), rootMoves.end(), move);

            rm.effort += nodes - nodeCount;

            u64 N      = nodes - nodeCount;
            u64 E_prev = std::max(u64(1), rm.effort - N);

            // Exponential moving average parameters for root move
            constexpr u64 Scale          = 32;
            constexpr u64 ChiNumerator   = 3;
            constexpr u64 ChiDenominator = 2;   // Chi = 3/2 = 1.5
            constexpr u64 MinWeight      = 12;  // 37.5% minimum weight
            constexpr u64 MaxWeight      = 24;  // 75% maximum weight

            u64 w     = std::clamp((Scale * N * ChiDenominator)
                                     / (N * ChiDenominator + ChiNumerator * E_prev),
                                   MinWeight, MaxWeight);
            u64 w_mss = std::min(w, u64(16));
            i64 v2    = i64(value) * std::abs(value);

            if (rm.averageScore == -VALUE_INFINITE)
                rm.averageScore = value;
            else
                rm.averageScore = Value((value * w + rm.averageScore * (Scale - w)) / Scale);

            if (rm.meanSquaredScore == -VALUE_INFINITE * VALUE_INFINITE)
                rm.meanSquaredScore = value * std::abs(value);
            else
                rm.meanSquaredScore =
                  Value((v2 * w_mss + int64_t(rm.meanSquaredScore) * (Scale - w_mss)) / Scale);

            // PV move or new best move?
            if (moveCount == 1 || value > alpha)
            {
                rm.score = rm.uciScore = value;
                rm.selDepth            = selDepth;
                rm.unset_inexact();

                if (value >= beta)
                {
                    rm.inexactLower = true;
                    rm.uciScore     = beta;
                }
                else if (value <= alpha)
                {
                    rm.inexactUpper = true;
                    rm.uciScore     = alpha;
                }

                rm.pv.resize(1);

                assert((ss + 1)->pv);

                for (Move pvMove : *(ss + 1)->pv)
                    rm.pv.push_back(pvMove);

                // We record how often the best move has been changed in each iteration.
                // This information is used for time management. In MultiPV mode,
                // we must take care to only do this for the first PV line.
                if (moveCount > 1 && !pvIdx)
                    ++bestMoveChanges;
            }
            else
                // All other moves but the PV are set to the lowest value: this
                // is not a problem when sorting because the sort is stable and the
                // move position in the list is preserved -- just the PV is pushed up.
                rm.score = -VALUE_INFINITE;
        }

        // If we have an alternative move equal in value to the current bestmove,
        // we sometimes promote it to bestmove by pretending it just exceeds
        // alpha (but not beta).
        int inc = (value == bestValue && ss->ply + 2 >= rootDepth && (int(nodes) & 14) == 0
                   && !is_win(std::abs(value) + 1));

        if (value + inc > bestValue)
        {
            bestValue = value;

            if (value + inc > alpha)
            {
                bestMove = move;

                // Update PV even in fail-high case
                if (PvNode && !rootNode)
                    ss->pv->update(move, (ss + 1)->pv);

                if (value >= beta)
                {
                    // (*Scaler) Infrequent and small updates scale well
                    ss->cutoffCnt += (extension < 2) || PvNode;
                    assert(value >= beta);  // Fail high
                    break;
                }

                // Reduce other moves if we have found at least one score improvement
                if (depth > 3 && depth < 12 && !is_decisive(value))
                    depth -= 3;

                assert(depth > 0);
                alpha = value;  // Update alpha! Always alpha < beta
            }
        }

        // If the move is worse than some previously searched move,
        // remember it, to update its stats later.
        if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY)
        {
            if (capture)
                capturesSearched.push_back(move);
            else
                quietsSearched.push_back(move);
        }
    }

    // Step 23. Check for mate and stalemate, otherwise update bestmove/countermove stats

    assert(moveCount || !ss->inCheck || excludedMove || !MoveList<ALL>(pos).size());

    // Adjust best value for fail high cases
    if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
        bestValue = (bestValue * depth + beta) / (depth + 1);

    // All legal moves have been searched: if there are no legal moves, it
    // must be a mate or a stalemate (just a fail low score if we are in a
    // singular extension search).
    if (!moveCount)
        bestValue = excludedMove ? alpha : ss->inCheck ? mated_in(ss->ply) : VALUE_DRAW;

    // If there is a move that produces search value greater than alpha,
    // we update the stats of searched moves.
    else if (bestMove)
    {
        update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched, depth,
                         ttData.move, PvNode);
        if (!PvNode)
            ttMoveHistory << (bestMove == ttData.move ? 918 : -747);
    }

    // Bonus for prior quiet countermove that caused the fail low
    else if (!priorCapture && prevSq != SQ_NONE)
    {
        int bonusScale = -241;
        bonusScale -= (ss - 1)->statScore / 98;
        bonusScale += std::min(59 * depth, 420);
        bonusScale += 186 * ((ss - 1)->moveCount > 9);
        bonusScale += 142 * (!ss->inCheck && bestValue <= ss->staticEval - 106);
        bonusScale += 159 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 68);

        bonusScale = std::max(bonusScale, 0);

        // scaledBonus ranges from 0 to roughly 2.3M, overflows happen for
        // multipliers larger than 900
        const int scaledBonus = std::min(150 * depth - 85, 1337) * bonusScale;

        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                      scaledBonus * 263 / 16384);

        mainHistory[~us][((ss - 1)->currentMove).raw()] << scaledBonus * 215 / 32768;

        if (type_of(pos.piece_on(prevSq)) != PAWN && ((ss - 1)->currentMove).type_of() != PROMOTION)
            sharedHistory.pawn_entry(pos)[pos.piece_on(prevSq)][prevSq] << scaledBonus * 324 / 8192;
    }

    // Bonus for prior capture countermove that caused the fail low
    else if (prevSq != SQ_NONE)
    {
        assert(priorCapture);

        Piece capturedPiece = pos.captured_piece();
        assert(capturedPiece != NO_PIECE);
        captureHistory[pos.piece_on(prevSq)][prevSq][type_of(capturedPiece)] << 892;
    }

    if (PvNode)
        bestValue = std::min(bestValue, maxValue);

    // If no good move is found and the previous position was ttPv, then the previous
    // opponent move is probably good and the new position is added to the search tree.
    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

    // Step 24. Write gathered information in transposition table. Note that the
    // static evaluation is saved as it was before correction history.
    if (!excludedMove && !(rootNode && pvIdx))
        ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                       bestValue >= beta    ? BOUND_LOWER
                       : PvNode && bestMove ? BOUND_EXACT
                                            : BOUND_UPPER,
                       moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6), bestMove,
                       unadjustedStaticEval, tt.generation());

    // Adjust correction history if the best move is not a capture and
    // the error direction matches whether we are above/below bounds.
    if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
        && (bestValue > ss->staticEval) == bool(bestMove))
    {
        auto bonus =
          std::clamp(int(bestValue - ss->staticEval) * depth * (bestMove ? 12 : 18) / 128,
                     -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
        update_correction_history(pos, ss, *this, 1061 * bonus / 1024);
    }

#ifdef SF_SEARCH_SAMPLER
    if (sample.active && !threads.stop.load(std::memory_order_relaxed))
        searchSampler.write(sample, bestValue, bestMove, PvNode, nodes.load(std::memory_order_relaxed));
#endif
    // The search is now complete
    assert(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);
    return bestValue;
}


// Quiescence search function, which is called by the main search function with
// depth zero, or recursively with further decreasing depth. With depth <= 0, we
// "should" be using static eval only, but tactical moves may confuse the static eval.
// To fight this horizon effect, we implement this qsearch of tactical moves.
// See https://www.chessprogramming.org/Horizon_Effect
// and https://www.chessprogramming.org/Quiescence_Search
template<NodeType nodeType>
Value Search::Worker::qsearch(Position& pos, Stack* ss, Value alpha, Value beta) {

    static_assert(nodeType != Root);
    constexpr bool PvNode = nodeType == PV;

    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));

    // Check if we have an upcoming move that draws by repetition
    if (alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }

    PVMoves   pv;
    StateInfo st;

    Key   posKey;
    Move  move, bestMove;
    Value bestValue, value, futilityBase;
    bool  pvHit, givesCheck, capture;
    int   moveCount;

    // Step 1. Initialize node
    if (PvNode)
    {
        (ss + 1)->pv = &pv;
        ss->pv->clear();
    }

    bestMove    = Move::none();
    ss->inCheck = pos.checkers();
    moveCount   = 0;

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    // Step 2. Check for an immediate draw or maximum ply reached
    if (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
        return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : VALUE_DRAW;

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Step 3. Transposition table lookup
    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);

    ss->ttHit    = ttHit;
    ttData.move  = ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule50_count()) : VALUE_NONE;
    pvHit        = ttHit && ttData.is_pv;

    // At non-PV nodes we check for an early TT cutoff
    if (!PvNode && ttData.depth >= DEPTH_QS && is_valid(ttData.value)
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return ttData.value;

    // Step 4. Static evaluation of the position
    Value unadjustedStaticEval = VALUE_NONE;
    if (ss->inCheck)
        bestValue = futilityBase = -VALUE_INFINITE;
    else
    {
        const auto correctionValue = correction_value(*this, pos, ss);

        if (ss->ttHit)
        {
            // Never assume anything about values stored in TT
            unadjustedStaticEval = ttData.eval;

            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);

            ss->staticEval = bestValue =
              to_corrected_static_eval(unadjustedStaticEval, correctionValue);

            // ttValue can be used as a better position evaluation
            if (is_valid(ttData.value) && !is_decisive(ttData.value)
                && (ttData.bound & (ttData.value > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttData.value;
        }
        else
        {
            unadjustedStaticEval = evaluate(pos);
            ss->staticEval       = bestValue =
              to_corrected_static_eval(unadjustedStaticEval, correctionValue);
        }

        // Stand pat. Return immediately if static value is at least beta
        if (bestValue >= beta)
        {
            assert(!is_decisive(bestValue));
            bestValue = (441 * bestValue + 583 * beta) / 1024;

            if (!ss->ttHit)
                ttWriter.write(posKey, VALUE_NONE, false, BOUND_LOWER, DEPTH_UNSEARCHED,
                               Move::none(), unadjustedStaticEval, tt.generation());
            return bestValue;
        }

        if (bestValue > alpha)
            alpha = bestValue;

        futilityBase = ss->staticEval + 306;
    }

    const PieceToHistory* contHist[] = {(ss - 1)->continuationHistory};

    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;

    // Initialize a MovePicker object for the current position, and prepare
    // to search the moves. We presently use two stages of move generator in
    // quiescence search: captures, or evasions only when in check.
    MovePicker mp(pos, ttData.move, DEPTH_QS, &mainHistory, &lowPlyHistory, &captureHistory,
                  contHist, &sharedHistory, ss->ply);

    // Step 5. Loop through all legal moves until no moves remain
    // or a beta cutoff occurs.
    while ((move = mp.next_move()) != Move::none())
    {
        assert(move.is_ok());

        givesCheck = pos.gives_check(move);
        capture    = pos.capture_stage(move);

        moveCount++;

        // Step 6. Pruning
        if (!is_loss(bestValue))
        {
            // Futility pruning and moveCount pruning
            if (!givesCheck && move.to_sq() != prevSq && !is_loss(futilityBase)
                && move.type_of() != PROMOTION)
            {
                if (moveCount > 2)
                    continue;

                Value futilityValue = futilityBase + PieceValue[pos.piece_on(move.to_sq())];

                // If static eval + value of piece we are going to capture is
                // much lower than alpha, we can prune this move.
                if (futilityValue <= alpha)
                {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }

                // If static exchange evaluation is low enough, we can prune
                if (!pos.see_ge(move, alpha - futilityBase))
                {
                    bestValue = std::max(bestValue, std::min(alpha, futilityBase));
                    continue;
                }
            }

            // Skip non-captures
            if (!capture)
                continue;

            // Do not search moves with bad enough SEE values
            if (!pos.see_ge(move, -74))
                continue;
        }

        // Step 7. Make and search the move
        // Descendants stay in qsearch, where policy is never consulted.
        do_move<false>(pos, move, st, givesCheck, capture, ss);

        value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha);
        undo_move<false>(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Step 8. Check for a new best move
        if (value > bestValue)
        {
            bestValue = value;

            if (value > alpha)
            {
                bestMove = move;

                // Update pv even in fail-high case
                if (PvNode)
                    ss->pv->update(move, (ss + 1)->pv);

                if (value < beta)  // Update alpha here!
                    alpha = value;
                else
                    break;  // Fail high
            }
        }
    }

    // Step 9. Check for mate and stalemate
    // All legal moves have been searched. A special case: if we are
    // in check and no legal moves were found, it is checkmate.
    if (!moveCount)
    {
        if (ss->inCheck)  // Checkmate!
        {
            assert(!MoveList<ALL>(pos).size());
            return mated_in(ss->ply);  // Plies to mate from the root
        }

        // Only check for stalemate under specific conditions
        Color us = pos.side_to_move();
        if (!(pawn_single_push_bb(us, pos.pieces(us, PAWN)) & ~pos.pieces())
            && !pos.non_pawn_material(us) && type_of(pos.captured_piece()) >= KNIGHT
            && !MoveList<ALL>(pos).size())
            bestValue = VALUE_DRAW;
    }

    if (!is_decisive(bestValue) && bestValue > beta)
        bestValue = (462 * bestValue + 562 * beta) / 1024;

    // Step 10. Save gathered info in transposition table. The static evaluation
    // is saved as it was before adjustment by correction history.
    ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
                   bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
                   unadjustedStaticEval, tt.generation());

    // The search is now complete
    assert(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);
    return bestValue;
}

int Search::Worker::reduction(bool i, Depth d, int mn) const {
    int reductionScale = reductions[d] * reductions[mn];
    return reductionScale + !i * reductionScale * 197 / 512 + 948;
}

// elapsed() returns the time elapsed since the search started. If the
// 'nodestime' option is enabled, it will return the count of nodes searched
// instead. This function is called to check whether the search should be
// stopped based on predefined thresholds like time limits or nodes searched.
TimePoint Search::Worker::elapsed() const {
    return main_manager()->tm.elapsed([this]() { return threads.nodes_searched(); });
}


// Evaluate the current position of the game tree, from the point of view of
// the side to move.
Value Search::Worker::evaluate(const Position& pos) {
    return Eval::evaluate(network[numaAccessToken], pos, accumulatorStack, refreshTable,
                          optimism[pos.side_to_move()]);
}

namespace {

// Adjusts a mate or TB score from "plies to mate from the root" to
// "plies to mate from the current position". Standard scores are unchanged.
// The function is called before storing a value in the transposition table.
Value value_to_tt(Value v, int ply) { return is_win(v) ? v + ply : is_loss(v) ? v - ply : v; }


// Inverse of value_to_tt(): it adjusts a mate or TB score from the transposition
// table (which refers to the plies to mate/be mated from current position) to
// "plies to mate/be mated (TB win/loss) from the root". However, to avoid
// potentially false mate or TB scores related to the 50 moves rule and the
// graph history interaction, we return the highest non-TB score instead.
Value value_from_tt(Value v, int ply, int r50c) {

    if (!is_valid(v))
        return VALUE_NONE;

    // Handle TB win or better
    if (is_win(v))
    {
        // Downgrade a potentially false mate score
        if (is_mate(v) && VALUE_MATE - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        // Downgrade a potentially false TB score
        if (VALUE_TB - v > 100 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        return v - ply;
    }

    // Handle TB loss or worse
    if (is_loss(v))
    {
        // Downgrade a potentially false mate score
        if (is_mated(v) && VALUE_MATE + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        // Downgrade a potentially false TB score
        if (VALUE_TB + v > 100 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        return v + ply;
    }

    return v;
}


// Updates stats at the end of search() when a bestMove is found
void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            ttMove,
                      bool            PvNode) {

    CapturePieceToHistory& captureHistory = workerThread.captureHistory;
    Piece                  movedPiece     = pos.moved_piece(bestMove);
    PieceType              capturedPiece;

    int bonus =
      std::min(133 * depth - 81, 1487) + 364 * (bestMove == ttMove) + (ss - 1)->statScore / 28;
    int malus = std::min(968 * depth - 235, 2244);

    if (!PvNode)
        // Important: don't remove the cast to a 64-bit number else the multiplication
        // can overflow on 32-bit platforms which would change the bench signature
        bonus += int(bonus * u64(quietsSearched.size() + capturesSearched.size()) / 256);

    if (!pos.capture_stage(bestMove))
    {
        update_quiet_histories(pos, ss, workerThread, bestMove, bonus * 899 / 1024);

        // Decrease stats for all non-best quiet moves
        int actualMalus = malus * 1159 / 1024;
        for (Move move : quietsSearched)
        {
            actualMalus = actualMalus * 921 / 1024;
            update_quiet_histories(pos, ss, workerThread, move, -actualMalus);
        }
    }
    else
    {
        // Increase stats for the best move in case it was a capture move
        capturedPiece = type_of(pos.piece_on(bestMove.to_sq()));
        captureHistory[movedPiece][bestMove.to_sq()][capturedPiece] << bonus * 1427 / 1024;
    }

    // Extra penalty for a quiet early move that was not a TT move in
    // previous ply when it gets refuted.
    if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -malus * 713 / 1024);

    // Decrease stats for all non-best capture moves
    for (Move move : capturesSearched)
    {
        movedPiece    = pos.moved_piece(move);
        capturedPiece = type_of(pos.piece_on(move.to_sq()));
        captureHistory[movedPiece][move.to_sq()][capturedPiece] << -malus * 1489 / 1024;
    }
}


// Updates the continuation histories for the move pairs formed by
// the current move and the moves played in previous plies.
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus) {
    static constexpr std::array<ConthistBonus, 5> conthist_bonuses = {
      {{1, 520}, {2, 390}, {3, 145}, {4, 251}, {6, 209}}};

    // Multipliers for positive history consistency
    constexpr int CMHCMultipliers[] = {94, 103, 110, 106, 119, 126};
    int           positiveCount     = 0;

    for (const auto [i, weight] : conthist_bonuses)
    {
        // Only update the first 2 continuation histories if we are in check
        if (ss->inCheck && i > 2)
            break;

        if (((ss - i)->currentMove).is_ok())
        {
            auto& historyEntry = (*(ss - i)->continuationHistory)[pc][to];
            if (historyEntry > 0)
                positiveCount++;

            int multiplier = CMHCMultipliers[positiveCount];
            historyEntry << bonus * weight * multiplier / 65536 + 73 * (i < 2);
        }
    }
}

// Updates move sorting heuristics

void update_quiet_histories(
  const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus) {

    Color us = pos.side_to_move();
    workerThread.mainHistory[us][move.raw()] << bonus;  // Untuned to prevent duplicate effort

    if (ss->ply < LOW_PLY_HISTORY_SIZE)
        workerThread.lowPlyHistory[ss->ply][move.raw()] << bonus * 712 / 1024;

    update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(), bonus * 750 / 1024);

    workerThread.sharedHistory.pawn_entry(pos)[pos.moved_piece(move)][move.to_sq()]
      << bonus * (bonus > -4 ? 1104 : 459) / 1024;
}
}

// When playing with strength handicap, choose the best move among a set of
// RootMoves using a statistical rule dependent on 'level'.
// Idea by Heinz van Saanen.
Move Skill::pick_best(const RootMoves& rootMoves, usize multiPV) {
    static PRNG rng(now());  // PRNG sequence should be non-deterministic

    // With tablebases at the root, rootMoves are ordered by tbRank rather
    // than by score, so compute the score range explicitly to keep 'delta'
    // non-negative.
    Value topScore = rootMoves[0].score;
    Value minScore = rootMoves[0].score;
    for (usize i = 1; i < multiPV; ++i)
    {
        topScore = std::max(topScore, rootMoves[i].score);
        minScore = std::min(minScore, rootMoves[i].score);
    }
    int    delta    = std::min(topScore - minScore, int(PawnValue));
    int    maxScore = -VALUE_INFINITE;
    double weakness = 120 - 2 * level;

    // Choose best move. For each move score we add two terms dependent on
    // weakness. One is deterministic and bigger for weaker levels, and one
    // is random. Then we choose the move with the resulting highest score.
    for (usize i = 0; i < multiPV; ++i)
    {
        // This is our magic formula
        int push = int(weakness * int(topScore - rootMoves[i].score)
                       + delta * (rng.rand<unsigned>() % int(weakness)))
                 / 128;

        if (rootMoves[i].score + push >= maxScore)
        {
            maxScore = rootMoves[i].score + push;
            best     = rootMoves[i].pv[0];
        }
    }

    return best;
}

// Function to detect when we are out of available time and stop the search,
// and to print debug info.
void SearchManager::check_time(Search::Worker& worker) {

    if (--callsCnt > 0)
        return;

    // When using nodes, ensure checking rate is not lower than 0.1% of nodes
    callsCnt = worker.limits.nodes ? std::min(512, int(worker.limits.nodes / 1024)) : 512;

#ifdef __EMSCRIPTEN__
    // Let the worker receive stop without unwinding or resetting the search.
    static TimePoint lastYield = now();
    if (now() - lastYield >= 16)
    {
        emscripten_sleep(0);
        lastYield = now();
    }
#endif
    static TimePoint lastInfoTime = now();

    TimePoint elapsed = tm.elapsed([&worker]() { return worker.threads.nodes_searched(); });
    TimePoint tick    = worker.limits.startTime + elapsed;

    if (tick - lastInfoTime >= 1000)
    {
        lastInfoTime = tick;
        dbg_print();
    }

    // We should not stop pondering until told so by the GUI
    if (ponder)
        return;

    if ((worker.limits.use_time_management() && (elapsed > tm.maximum() || stopOnPonderhit))
        || (worker.limits.movetime && elapsed >= worker.limits.movetime)
        || (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes))
        worker.threads.stop = true;
}

// Used to correct and extend PVs for moves that have a TB (but not a mate) score.
// Keeps the search based PV for as long as it is verified to maintain the game
// outcome, truncates afterwards. Finally, extends to mate the PV, providing a
// possible continuation (but not a proven mating line).
void syzygy_extend_pv(const OptionsMap&         options,
                      const Search::LimitsType& limits,
                      Position&                 pos,
                      RootMove&                 rootMove,
                      Value&                    v,
                      const usize               multiPV) {

    auto t_start      = std::chrono::steady_clock::now();
    int  moveOverhead = int(options["Move Overhead"]);
    bool rule50       = bool(options["Syzygy50MoveRule"]);

    // Do not use more than moveOverhead / 2 ms, if time management is active.
    // Under 'nodestime' the pos.do_move() calls come for free.
    auto time_abort = [&t_start, &moveOverhead, &limits, &multiPV]() -> bool {
        auto t_end = std::chrono::steady_clock::now();
        return !limits.npmsec && limits.use_time_management()
            && 2 * multiPV * std::chrono::duration<double, std::milli>(t_end - t_start).count()
                 >= moveOverhead;
    };

    if (time_abort())
        return;

    std::list<StateInfo> sts;

    // Step 0, do the rootMove, no correction allowed, as needed for MultiPV in TB
    auto& stRoot = sts.emplace_back();
    pos.do_move(rootMove.pv[0], stRoot);
    int ply = 1;

    // Step 1, walk the PV to the last position in TB with correct decisive score
    while (usize(ply) < rootMove.pv.size())
    {
        Move& pvMove = rootMove.pv[ply];

        RootMoves legalMoves;
        for (const auto& m : MoveList<ALL>(pos))
            legalMoves.emplace_back(m);

        TB::Config config = TB::rank_root_moves(options, pos, legalMoves, false, time_abort);
        RootMove&  rm     = *std::find(legalMoves.begin(), legalMoves.end(), pvMove);

        if (legalMoves[0].tbRank != rm.tbRank)
            break;

        ply++;

        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);

        // Do not allow for repetitions or drawing moves along the PV in TB regime
        if (config.rootInTB && ((rule50 && pos.is_draw(ply)) || pos.is_repetition(ply)))
        {
            pos.undo_move(pvMove);
            ply--;
            break;
        }

        // Full PV shown will thus be validated and end in TB. If we cannot
        // validate the full PV in time, we do not show it.
        if (config.rootInTB && time_abort())
            break;
    }

    // Resize the PV to the correct part
    rootMove.pv.resize(ply);

    // Step 2, now extend the PV to mate, as if the user explored syzygy-tables.info
    // using top ranked moves (minimal DTZ), which gives optimal mates only for simple
    // endgames e.g. KRvK.
    while (!(rule50 && pos.is_draw(0)))
    {
        if (time_abort())
            break;

        RootMoves legalMoves;
        for (const auto& m : MoveList<ALL>(pos))
        {
            auto&     rm = legalMoves.emplace_back(m);
            StateInfo tmpSI;
            pos.do_move(m, tmpSI);
            // Give a score of each move to break DTZ ties restricting opponent
            // mobility, but not giving the opponent a capture.
            for (const auto& mOpp : MoveList<ALL>(pos))
                rm.tbRank -= pos.capture(mOpp) ? 100 : 1;
            pos.undo_move(m);
        }

        // Mate found
        if (legalMoves.empty())
            break;

        // Sort moves according to their above assigned rank.
        // This will break ties for moves with equal DTZ in rank_root_moves.
        std::stable_sort(
          legalMoves.begin(), legalMoves.end(),
          [](const Search::RootMove& a, const Search::RootMove& b) { return a.tbRank > b.tbRank; });

        // The winning side tries to minimize DTZ, the losing side maximizes it
        TB::Config config = TB::rank_root_moves(options, pos, legalMoves, true, time_abort);

        // If DTZ is not available we might not find a mate, so we bail out
        if (!config.rootInTB || config.cardinality > 0)
            break;

        ply++;

        Move& pvMove = legalMoves[0].pv[0];
        rootMove.pv.push_back(pvMove);
        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);
    }

    // Finding a draw in this function is an exceptional case, that cannot happen when rule50 is false or
    // during engine game play, since we have a winning score, and play correctly
    // with TB support. However, it can be that a position is draw due to the 50 move
    // rule if it has been reached on the board with a non-optimal 50 move counter
    // (e.g. 8/8/6k1/3B4/3K4/4N3/8/8 w - - 54 106 ) which TB with dtz counter rounding
    // cannot always correctly rank. See also
    // https://github.com/official-stockfish/Stockfish/issues/5175#issuecomment-2058893495
    // We adjust the score to match the found PV. Note that a TB loss score can be
    // displayed if the engine did not find a drawing move yet, but eventually search
    // will figure it out (e.g. 1kq5/q2r4/5K2/8/8/8/8/7Q w - - 96 1 )
    if (pos.is_draw(0))
        v = VALUE_DRAW;

    // Undo the PV moves
    for (usize i = rootMove.pv.size(); i > 0; --i)
        pos.undo_move(rootMove.pv[i - 1]);

    // Inform if we couldn't get a full extension in time
    if (time_abort())
        sync_cout
          << "info string Syzygy based PV extension requires more time, increase Move Overhead as needed."
          << sync_endl;
}

void SearchManager::output_pv(Search::Worker&           worker,
                              const ThreadPool&         threads,
                              const TranspositionTable& tt,
                              Depth                     depth) {

    const auto nodes     = threads.nodes_searched();
    auto&      rootMoves = worker.rootMoves;
    auto&      pos       = worker.rootPos;
    usize      multiPV   = std::min(usize(worker.options["MultiPV"]), rootMoves.size());
    u64        tbHits    = threads.tb_hits() + (worker.tbConfig.rootInTB ? rootMoves.size() : 0);

    for (usize i = 0; i < multiPV; ++i)
    {
        bool usePreviousScore = rootMoves[i].score == -VALUE_INFINITE;

        if (depth == 1 && usePreviousScore && i > 0)
            continue;

        Depth d = usePreviousScore ? std::max(1, depth - 1) : depth;
        Value v = usePreviousScore ? rootMoves[i].previousScore : rootMoves[i].uciScore;

        if (v == -VALUE_INFINITE)
            v = VALUE_ZERO;

        bool isTBScore = worker.tbConfig.rootInTB && !is_mate_or_mated(v);
        v              = isTBScore ? rootMoves[i].tbScore : v;

        // Potentially correct and extend the PV, and in exceptional cases v.
        // Previous PVs have already been extended. Inexact flags indicate an unreliable PV.
        if (is_decisive(v) && !is_mate_or_mated(v) && !usePreviousScore
            && (!rootMoves[i].is_inexact() || isTBScore))
            syzygy_extend_pv(worker.options, worker.limits, pos, rootMoves[i], v, multiPV);

        std::string pv;
        for (Move m : usePreviousScore ? rootMoves[i].previousPV : rootMoves[i].pv)
            pv += UCIEngine::move(m, pos.is_chess960()) + " ";

        // Remove last whitespace
        if (!pv.empty())
            pv.pop_back();

        auto wdl = worker.options["UCI_ShowWDL"] ? UCIEngine::wdl(v, pos) : "";

        // Scores cannot be both exact and inexact
        assert(!(rootMoves[i].inexactLower && rootMoves[i].inexactUpper));
        auto bound = rootMoves[i].inexactLower ? "lowerbound"
                   : rootMoves[i].inexactUpper ? "upperbound"
                                               : "";

        InfoFull info;

        info.depth    = d;
        info.selDepth = rootMoves[i].selDepth;
        info.multiPV  = i + 1;
        info.score    = {v, pos};
        info.wdl      = wdl;

        // TB and previous scores are exact, even though their flags may say otherwise
        if (!(isTBScore || usePreviousScore))
            info.bound = bound;

        TimePoint time = std::max(TimePoint(1), tm.elapsed_time());
        info.timeMs    = time;
        info.nodes     = nodes;
        info.nps       = nodes * 1000 / time;
        info.tbHits    = tbHits;
        info.pv        = pv;
        info.hashfull  = tt.hashfull();

        updates.onUpdateFull(info);
    }
}

// Called in case we have no ponder move before exiting the search,
// for instance, in case we stop the search during a fail high at root.
// We try hard to have a ponder move to return to the GUI, otherwise
// in case of 'ponder on' we have nothing to think about.
bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt, Position& pos) {

    assert(pv.size() == 1 && pv[0] != Move::none());

    StateInfo st;
    pos.do_move(pv[0], st);

    if (!pos.is_draw(1))
    {
        auto [ttHit, ttData, ttWriter] = tt.probe(pos.key());
        if (ttHit && ttData.move && pos.legal(ttData.move))
            pv.push_back(ttData.move);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}


}  // namespace Stockfish
