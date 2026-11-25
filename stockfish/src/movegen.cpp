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

#include "movegen.h"

#include <initializer_list>

#include "attacks.h"
#include "bitboard.h"
#include "movegen_callbacks.h"
#include "position.h"

namespace Stockfish {

namespace {

template<Color Us>
int count_pawn_moves(const Position& pos, Bitboard target, bool inCheck) {

    constexpr Color     Them     = ~Us;
    constexpr Bitboard  TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up       = pawn_push(Us);
    constexpr Direction UpRight  = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft   = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~pos.pieces();
    const Bitboard enemies      = inCheck ? pos.checkers() : pos.pieces(Them);
    const Bitboard pinned       = pos.blockers_for_king(Us);
    const Square   ksq          = pos.square<KING>(Us);

    const Bitboard pawnsOn7       = pos.pieces(Us, PAWN) & TRank7BB;
    const Bitboard pawnsNotOn7    = pos.pieces(Us, PAWN) & ~TRank7BB;
    const Bitboard pushable       = ~pinned | file_bb(ksq);
    const Bitboard canCaptureLeft = ~pinned | Attacks::antidiag_bb(ksq);
    const Bitboard canCaptureRight = ~pinned | Attacks::diag_bb(ksq);

    Bitboard singlePushes = shift(pawnsNotOn7 & pushable, Up) & emptySquares;
    Bitboard doublePushes = shift(singlePushes & TRank3BB, Up) & emptySquares;

    if (inCheck)
    {
        singlePushes &= target;
        doublePushes &= target;
    }

    int count = popcount(singlePushes) + popcount(doublePushes);

    Bitboard captureRight = shift(pawnsNotOn7 & canCaptureRight, UpRight) & enemies;
    Bitboard captureLeft  = shift(pawnsNotOn7 & canCaptureLeft, UpLeft) & enemies;
    count += popcount(captureRight) + popcount(captureLeft);

    Bitboard promoteRight = shift(pawnsOn7 & canCaptureRight, UpRight) & enemies;
    Bitboard promoteLeft  = shift(pawnsOn7 & canCaptureLeft, UpLeft) & enemies;
    Bitboard promotePush  = shift(pawnsOn7 & pushable, Up) & emptySquares;

    if (inCheck)
        promotePush &= target;

    count += 4 * (popcount(promoteRight) + popcount(promoteLeft) + popcount(promotePush));

    const Square epSq = pos.ep_square();
    if (epSq != SQ_NONE && !(inCheck && (target & (epSq + Up))))
    {
        Bitboard epPawns = pawnsNotOn7
                         & ((canCaptureRight & shift(square_bb(epSq), -UpRight))
                            | (canCaptureLeft & shift(square_bb(epSq), -UpLeft)));
        count += popcount(epPawns);
    }

    return count;
}

int count_legal_moves_impl(const Position& pos) {

    const Color   us       = pos.side_to_move();
    const Square  ksq      = pos.square<KING>(us);
    const Bitboard checkers = pos.checkers();
    const bool    inCheck  = bool(checkers);

    int count = 0;

    // In double check only king moves are legal.
    if (!inCheck || !more_than_one(checkers))
    {
        const Bitboard target = inCheck ? Attacks::between_bb(ksq, lsb(checkers)) : ~pos.pieces(us);

        count += us == WHITE ? count_pawn_moves<WHITE>(pos, target, inCheck)
                             : count_pawn_moves<BLACK>(pos, target, inCheck);

        for (PieceType pt : {KNIGHT, BISHOP, ROOK, QUEEN})
        {
            Bitboard pieces = pos.pieces(us, pt);
            const Bitboard pinned = pos.blockers_for_king(us);

            Bitboard unpinned = pieces & ~pinned;
            while (unpinned)
            {
                const Square from = pop_lsb(unpinned);
                count += popcount(Attacks::attacks_bb(pt, from, pos.pieces()) & target);
            }

            if (pt != KNIGHT)
            {
                Bitboard pinnedPieces = pieces & pinned;
                while (pinnedPieces)
                {
                    const Square from = pop_lsb(pinnedPieces);
                    Bitboard moves = Attacks::attacks_bb(pt, from, pos.pieces()) & target;
                    moves &= Attacks::line_bb(ksq, from);
                    count += popcount(moves);
                }
            }
        }
    }

    Bitboard kingTargets = Attacks::attacks_bb(KING, ksq) & ~pos.pieces(us)
                         & ~pos.threats_by(ALL_PIECES);
    count += popcount(kingTargets);

    if (!inCheck && pos.can_castle(us & ANY_CASTLING))
        for (CastlingRights cr : {us & KING_SIDE, us & QUEEN_SIDE})
        {
            const Square rookSquare = pos.castling_rook_square(cr);
            const Square to = relative_square(us, rookSquare > ksq ? SQ_G1 : SQ_C1);

            if (!pos.castling_impeded(cr) && pos.can_castle(cr)
                && !(Attacks::between_bb(ksq, to) & pos.threats_by(ALL_PIECES))
                && !(pos.blockers_for_king(us) & rookSquare))
                ++count;
        }

    return count;
}

}  // namespace


// <CAPTURES>     Generates all legal captures plus queen promotions
// <QUIETS>       Generates all legal non-captures and underpromotions
// <EVASIONS>     Generates all legal check evasions
//
// Returns a pointer to the end of the move list.
template<GenType Type>
Move* generate(const Position& pos, Move* moveList) {

    static_assert(Type != ALL, "Unsupported type in generate()");

    MovegenDetail::MoveListWriter emit{moveList};
    MovegenDetail::generate_to<Type>(pos, emit);
    return emit.current;
}

// Explicit template instantiations
template Move* generate<CAPTURES>(const Position&, Move*);
template Move* generate<QUIETS>(const Position&, Move*);
template Move* generate<EVASIONS>(const Position&, Move*);

int count_legal_moves(const Position& pos) {
    return count_legal_moves_impl(pos);
}

// generate<ALL> generates all the legal moves in the given position

template<>
Move* generate<ALL>(const Position& pos, Move* moveList) {
    MovegenDetail::MoveListWriter emit{moveList};
    MovegenDetail::generate_to<ALL>(pos, emit);
    return emit.current;
}

}  // namespace Stockfish
