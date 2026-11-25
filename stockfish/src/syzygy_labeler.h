// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef SYZYGY_LABELER_H
#define SYZYGY_LABELER_H
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>
#include <cstdlib>
#include "movegen.h"
#include "position.h"
#include "syzygy/tbprobe.h"
#include "uci.h"
namespace Stockfish {
// Persistent FEN-in/JSON-out labelling mode. No Engine, workers, NNUE or search.
inline int syzygy_labeler(const std::string& path) {
    Tablebases::init(path);
    std::cout << "ready" << std::endl;
    std::string fen;
    while (std::getline(std::cin, fen)) {
        const bool wdlOnly = fen.rfind("wdl ", 0) == 0;
        if (wdlOnly) fen.erase(0, 4);
        Position pos; StateInfo root;
        if (pos.set(fen, false, &root) || pos.count<ALL_PIECES>() > (wdlOnly ? 7 : 5)
            || pos.can_castle(ANY_CASTLING)) {
            std::cout << "{\"error\":\"unsupported_position\"}" << std::endl;
            continue;
        }
        MoveList<ALL> legal(pos);
        if (legal.size() < 2) {
            std::cout << "{\"moves\":[]}" << std::endl;
            continue;
        }
        struct Choice { Move move; int value; int distance; };
        std::vector<Choice> choices;
        bool failed = false;
        int best = -3;
        for (Move move : legal) {
            StateInfo st; pos.do_move(move, st);
            int value = 0, distance = 0;
            if (MoveList<ALL>(pos).size() == 0) value = pos.checkers() ? 2 : 0;
            else {
                Tablebases::ProbeState status;
                value = -int(Tablebases::probe_wdl(pos, &status));
                if (status == Tablebases::FAIL) failed = true;
                // Draws and cursed/blessed outcomes need no DTZ lookup.
                if (wdlOnly) {
                    if (std::abs(value) == 1) value = 0;
                } else if (!failed && std::abs(value) == 2) {
                    distance = std::abs(Tablebases::probe_dtz(pos, &status));
                    if (status == Tablebases::FAIL) failed = true;
                    if (distance + pos.rule50_count() > 100) value = 0;
                } else value = 0;
            }
            pos.undo_move(move);
            if (failed) break;
            choices.push_back({move, value, distance});best = std::max(best, value);
        }
        if (failed) {
            std::cout << "{\"error\":\"missing_table\"}" << std::endl;
            continue;
        }
        int distance = std::numeric_limits<int>::max();
        if (best > 0) for (auto c : choices) if (c.value == best) distance = std::min(distance, c.distance);
        std::cout << "{\"moves\":[";
        bool first = true;
        for (auto c : choices) if (c.value == best && (best <= 0 || c.distance == distance)) {
            if (!first) std::cout << ',';
            first = false;
            std::cout << '"' << UCIEngine::move(c.move, false) << '"';
        }
        std::cout << "],\"wdl\":" << best << "}" << std::endl;
    }
    return 0;
}
}
#endif
