// Internal-node data collection, compiled only into the dedicated sampler.
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef SEARCH_SAMPLER_H_INCLUDED
#define SEARCH_SAMPLER_H_INCLUDED
#include <fstream>
#include <unordered_set>

namespace Stockfish {
struct SearchSample {
    bool active = false;
    std::string fen, quiets;
    int depth = 0, ply = 0, alpha = 0, beta = 0;
    u64 nodes = 0;
};

class SearchSampler {
    std::ofstream out;
    std::unordered_set<std::string> seen;
    std::string root;
    int every = 100, limit = 1000, minDepth = 6, eligible = 0, selected = 0, written = 0;
public:
    void start(const std::string& path, int interval, int cap, int depth, const std::string& fen) {
        out.close();
        out.clear();
        seen.clear();
        eligible = selected = written = 0;
        every = interval; limit = cap; minDepth = depth; root = fen;
        if (!path.empty()) {
            out.open(path, std::ios::app);
            if (!out)
                sync_cout << "info string Search sampler: cannot open output file" << sync_endl;
        }
    }
    bool enabled(int depth) const { return out.is_open() && out.good() && depth >= minDepth && selected < limit; }
    void capture(SearchSample& s, const Position& pos, ExtMove* begin, ExtMove* end,
                 Move ttMove, int depth, int ply, int alpha, int beta, u64 nodes) {
        if (!enabled(depth) || ++eligible % every) return;
        auto fen = pos.fen();
        if (!seen.insert(fen).second) return;
        std::string quiets;
        for (auto it = begin; it != end; ++it) {
            Move m = *it;
            if (m == ttMove || !pos.legal(m)) continue;
            if (!quiets.empty()) quiets += ',';
            quiets += '"' + UCIEngine::move(m, false) + '"';
        }
        if (quiets.empty()) return;
        s = {true, std::move(fen), std::move(quiets), depth, ply, alpha, beta, nodes};
        ++selected;
    }
    void write(const SearchSample& s, Value score, Move best, bool pv, u64 nodes) {
        // FEN and standard UCI moves contain no JSON quoting characters.
        const char* bound = score >= s.beta ? "lower" : pv && best ? "exact" : "upper";
        out << "{\"schema\":1,\"source\":\"internal_quiet\",\"root_fen\":\"" << root
            << "\",\"fen\":\"" << s.fen << "\",\"depth\":" << s.depth
            << ",\"ply\":" << s.ply << ",\"alpha\":" << s.alpha << ",\"beta\":" << s.beta
            << ",\"quiet_moves\":[" << s.quiets << "],\"score_raw\":" << score
            << ",\"bound\":\"" << bound << "\",\"cutoff\":" << (score >= s.beta ? "true" : "false")
            << ",\"bestmove\":";
        if (best) out << '"' << UCIEngine::move(best, false) << '"'; else out << "null";
        out << ",\"nodes_after_quiets\":" << nodes - s.nodes << "}\n";
        if (!out) {
            sync_cout << "info string Search sampler: output write failed" << sync_endl;
            out.close();
        } else ++written;
    }
    void finish() {
        if (out.is_open()) {
            out.flush();
            sync_cout << "info string Search sampler: " << written << " completed samples"
                      << (out.good() ? "" : " (flush failed)") << sync_endl;
            out.close();
        }
    }
};
// Only the main search worker uses this instance. Helpers never access its output.
static thread_local SearchSampler searchSampler;
}
#endif
