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

#ifndef MOVEGEN_CALLBACKS_H_INCLUDED
#define MOVEGEN_CALLBACKS_H_INCLUDED

#include <cassert>

#include "movegen.h"
#include "position.h"

#if defined(USE_AVX512ICL)
    #include <immintrin.h>
#endif

namespace Stockfish::MovegenDetail {

#if defined(USE_AVX512ICL)

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    assert(popcount(to_bb) <= 8);  // <= 8 pawns per side

    const __m128i toSquares =
      _mm_cvtepi8_epi16(_mm512_castsi512_si128(_mm512_maskz_compress_epi8(to_bb, AllSquares)));
    const __m128i fromSquares = _mm_subs_epi16(toSquares, _mm_set1_epi16(offset));
    const __m128i moves       = _mm_or_si128(_mm_slli_epi16(fromSquares, Move::FromSqShift),
                                             _mm_slli_epi16(toSquares, Move::ToSqShift));

    _mm_storeu_si128(reinterpret_cast<__m128i*>(moveList), moves);
    return moveList + popcount(to_bb);
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    assert(popcount(to_bb) <= 32);  // Q can attack up to 27 squares

    const __m512i fromVec = _mm512_set1_epi16(Move(from, SQUARE_ZERO).raw());
    const __m512i toSquares =
      _mm512_cvtepi8_epi16(_mm512_castsi512_si256(_mm512_maskz_compress_epi8(to_bb, AllSquares)));
    const __m512i moves = _mm512_or_si512(fromVec, _mm512_slli_epi16(toSquares, Move::ToSqShift));

    _mm512_storeu_si512(moveList, moves);
    return moveList + popcount(to_bb);
}

#else

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    while (to_bb)
    {
        Square to   = pop_lsb(to_bb);
        *moveList++ = Move(to - offset, to);
    }
    return moveList;
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    while (to_bb)
        *moveList++ = Move(from, pop_lsb(to_bb));
    return moveList;
}

#endif

// Preserve packed stores for consumers that need a sortable move list.
struct MoveListWriter {
    Move* current;

    void operator()(Move move, [[maybe_unused]] Piece pc) { *current++ = move; }

    template<Direction offset>
    void pawns(Bitboard destinations, [[maybe_unused]] Piece pc) {
        current = splat_pawn_moves<offset>(current, destinations);
    }

    void moves(Square from, Bitboard destinations, [[maybe_unused]] Piece pc) {
        current = splat_moves(current, from, destinations);
    }
};

// Invoke the consumer directly without creating an intermediate move list.
template<typename Callback>
struct CallbackWriter {
    Callback& callback;

    void operator()(Move move, [[maybe_unused]] Piece pc) { callback(move); }

    template<Direction offset>
    void pawns(Bitboard destinations, [[maybe_unused]] Piece pc) {
        while (destinations)
        {
            const Square to = pop_lsb(destinations);
            callback(Move(to - offset, to));
        }
    }

    void moves(Square from, Bitboard destinations, [[maybe_unused]] Piece pc) {
        while (destinations)
            callback(Move(from, pop_lsb(destinations)));
    }
};


// Keep destinations from one source together for consumers that can reuse
// source-dependent work. Pawn batches can contain different source squares.
template<typename SingleCallback, typename BatchCallback>
struct BatchCallbackWriter {
    SingleCallback& single;
    BatchCallback&  batch;

    void operator()(Move move, Piece pc) { single(move, pc); }

    template<Direction offset>
    void pawns(Bitboard destinations, Piece pc) {
        while (destinations)
        {
            const Square to = pop_lsb(destinations);
            single(Move(to - offset, to), pc);
        }
    }

    void moves(Square from, Bitboard destinations, Piece pc) {
        if (destinations)
            batch(from, destinations, pc, type_of(pc));
    }
};


template<Color Us, GenType Type, Direction D, bool Enemy, typename Emitter>
void make_promotions([[maybe_unused]] Emitter& emit, [[maybe_unused]] Square to) {

    constexpr bool          all  = Type == EVASIONS || Type == ALL;
    [[maybe_unused]] Square from = to - D;

    if constexpr (Type == CAPTURES || all)
        emit(Move::make<PROMOTION>(from, to, QUEEN), make_piece(Us, PAWN));

    if constexpr ((Type == CAPTURES && Enemy) || (Type == QUIETS && !Enemy) || all)
    {
        emit(Move::make<PROMOTION>(from, to, ROOK), make_piece(Us, PAWN));
        emit(Move::make<PROMOTION>(from, to, BISHOP), make_piece(Us, PAWN));
        emit(Move::make<PROMOTION>(from, to, KNIGHT), make_piece(Us, PAWN));
    }
}


// Snapshot reused by all generators in one call. Callbacks must restore Position.
struct GenerationContext {
    Bitboard occupied, pinned, checkers, enemies, own, threats;
    Square king;
};


template<Color Us, GenType Type, bool HasEP, typename Emitter>
void generate_pawn_moves(const Position& pos, Emitter& emit, const GenerationContext& ctx, Bitboard target) {

    static_assert(Type != ALL, "Unsupported type in generate_pawn_moves()");

    constexpr Bitboard  TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up       = pawn_push(Us);
    constexpr Direction UpRight  = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft   = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~ctx.occupied;
    const Bitboard enemies      = Type == EVASIONS ? ctx.checkers : ctx.enemies;
    const Bitboard pinned       = ctx.pinned;
    const Square   ksq          = ctx.king;

    const Bitboard pawnsOn7        = pos.pieces(Us, PAWN) & TRank7BB;
    const Bitboard pawnsNotOn7     = pos.pieces(Us, PAWN) & ~TRank7BB;
    const Bitboard pushable        = ~pinned | file_bb(ksq);
    const Bitboard canCaptureLeft  = ~pinned | Attacks::antidiag_bb(ksq);
    const Bitboard canCaptureRight = ~pinned | Attacks::diag_bb(ksq);

    // Single and double pawn pushes, no promotions
    if constexpr (Type != CAPTURES)
    {
        Bitboard b1 = shift(pawnsNotOn7 & pushable, Up) & emptySquares;
        Bitboard b2 = shift(b1 & TRank3BB, Up) & emptySquares;

        if constexpr (Type == EVASIONS)  // Consider only blocking squares
        {
            b1 &= target;
            b2 &= target;
        }

        emit.template pawns<Up>(b1, make_piece(Us, PAWN));
        emit.template pawns<Up + Up>(b2, make_piece(Us, PAWN));
    }

    // Promotions and underpromotions
    if (pawnsOn7)
    {
        Bitboard b1 = shift(pawnsOn7 & canCaptureRight, UpRight) & enemies;
        Bitboard b2 = shift(pawnsOn7 & canCaptureLeft, UpLeft) & enemies;
        Bitboard b3 = shift(pawnsOn7 & pushable, Up) & emptySquares;

        if constexpr (Type == EVASIONS)
            b3 &= target;

        while (b1)
            make_promotions<Us, Type, UpRight, true>(emit, pop_lsb(b1));

        while (b2)
            make_promotions<Us, Type, UpLeft, true>(emit, pop_lsb(b2));

        while (b3)
            make_promotions<Us, Type, Up, false>(emit, pop_lsb(b3));
    }

    // Standard and en passant captures
    if constexpr (Type == CAPTURES || Type == EVASIONS)
    {
        Bitboard b1 = shift(pawnsNotOn7 & canCaptureRight, UpRight) & enemies;
        Bitboard b2 = shift(pawnsNotOn7 & canCaptureLeft, UpLeft) & enemies;

        emit.template pawns<UpRight>(b1, make_piece(Us, PAWN));
        emit.template pawns<UpLeft>(b2, make_piece(Us, PAWN));

        if constexpr (HasEP)
        {
            const Square epSq = pos.ep_square();
            assert(epSq != SQ_NONE);

            // An en passant capture cannot resolve a discovered check.
            if (!(Type == EVASIONS && (target & (epSq + Up))))
            {
                assert(rank_of(epSq) == relative_rank(Us, RANK_6));

                b1 = pawnsNotOn7
                   & ((canCaptureRight & shift(square_bb(epSq), -UpRight))
                      | (canCaptureLeft & shift(square_bb(epSq), -UpLeft)));
                assert(b1);

                while (b1)
                    emit(Move::make<EN_PASSANT>(pop_lsb(b1), epSq), make_piece(Us, PAWN));
            }
        }
    }
}


template<Color Us, PieceType Pt, typename Emitter>
void generate_moves(const Position& pos, Emitter& emit, const GenerationContext& ctx, Bitboard target) {

    static_assert(Pt != KING && Pt != PAWN, "Unsupported piece type in generate_moves()");

    const Square   ksq    = ctx.king;
    const Bitboard pinned = ctx.pinned;

    Bitboard bb = pos.pieces(Us, Pt) & ~pinned;
    while (bb)
    {
        Square   from = pop_lsb(bb);
        Bitboard b    = Attacks::attacks_bb(Pt, from, ctx.occupied) & target;

        emit.moves(from, b, make_piece(Us, Pt));
    }

    if constexpr (Pt != KNIGHT)  // pinned knights cannot move
    {
        bb = pos.pieces(Us, Pt) & pinned;

        // A pinned bishop or rook can only move if its movement follows the pin line.
        if constexpr (Pt == BISHOP || Pt == ROOK)
            bb &= Attacks::attacks_bb(Pt, ksq);

        if constexpr (Pt == QUEEN)
        {
            const Bitboard diagonal = Attacks::attacks_bb(BISHOP, ksq);

            while (bb)
            {
                const Square from = pop_lsb(bb);
                const PieceType slider = diagonal & from ? BISHOP : ROOK;
                const Bitboard moves = Attacks::attacks_bb(slider, from, ctx.occupied) & target
                                     & Attacks::line_bb(ksq, from);

                emit.moves(from, moves, make_piece(Us, Pt));
            }
        }
        else
        {
            while (bb)
            {
                Square   from = pop_lsb(bb);
                Bitboard b    = Attacks::attacks_bb(Pt, from, ctx.occupied) & target;
                b &= Attacks::line_bb(ksq, from);

                emit.moves(from, b, make_piece(Us, Pt));
            }
        }
    }
}


template<Color Us, CastlingRights Side, typename Emitter>
void generate_castling(const Position& pos, Emitter& emit, const GenerationContext& ctx) {

    static_assert(Side == KING_SIDE || Side == QUEEN_SIDE, "Unsupported castling side");
    constexpr CastlingRights cr = Us & Side;
    assert(pos.can_castle(cr));

    const Square ksq        = ctx.king;
    const Square rookSquare = pos.castling_rook_square(cr);
    const Square to         = relative_square(Us, rookSquare > ksq ? SQ_G1 : SQ_C1);

    if (!pos.castling_impeded(cr)
        && !(Attacks::between_bb(ksq, to) & ctx.threats)
        && !(ctx.pinned & rookSquare))
        emit(Move::make<CASTLING>(ksq, rookSquare), make_piece(Us, KING));
}


template<Color Us, GenType Type, bool HasEP, bool CastleKingSide = false,
         bool CastleQueenSide = false, typename Emitter>
void generate_all(const Position& pos, Emitter& emit) {

    static_assert(Type != ALL, "Unsupported type in generate_all()");
    static_assert(!HasEP || Type != QUIETS, "Quiet generation does not use en passant");
    static_assert(!(CastleKingSide || CastleQueenSide) || Type == QUIETS,
                  "Only quiet generation uses castling rights");

    const GenerationContext ctx{pos.pieces(), pos.blockers_for_king(Us), pos.checkers(),
                                pos.pieces(~Us), pos.pieces(Us), pos.threats_by(ALL_PIECES),
                                pos.square<KING>(Us)};
    const Square ksq = ctx.king;
    Bitboard     target;

    // Skip generating non-king moves when in double check
    if (Type != EVASIONS || !more_than_one(ctx.checkers))
    {
        target = Type == EVASIONS ? Attacks::between_bb(ksq, lsb(ctx.checkers))
               : Type == CAPTURES ? ctx.enemies
                                  : ~ctx.occupied;  // QUIETS

        generate_pawn_moves<Us, Type, HasEP>(pos, emit, ctx, target);
        generate_moves<Us, KNIGHT>(pos, emit, ctx, target);
        generate_moves<Us, BISHOP>(pos, emit, ctx, target);
        generate_moves<Us, ROOK>(pos, emit, ctx, target);
        generate_moves<Us, QUEEN>(pos, emit, ctx, target);
    }

    if constexpr (Type == EVASIONS)
        target = ~ctx.own;

    target &= ~ctx.threats;

    Bitboard b = Attacks::attacks_bb(KING, ksq) & target;
    emit.moves(ksq, b, make_piece(Us, KING));

    if constexpr (CastleKingSide)
        generate_castling<Us, KING_SIDE>(pos, emit, ctx);

    if constexpr (CastleQueenSide)
        generate_castling<Us, QUEEN_SIDE>(pos, emit, ctx);
}


// Dispatch only on flags relevant to this generation type to limit code growth.
template<Color Us, GenType Type, typename Emitter>
void generate_for_side(const Position& pos, Emitter& emit) {

    if constexpr (Type == QUIETS)
    {
        const bool kingSide  = pos.can_castle(Us & KING_SIDE);
        const bool queenSide = pos.can_castle(Us & QUEEN_SIDE);

        if (kingSide)
            return queenSide ? generate_all<Us, Type, false, true, true>(pos, emit)
                             : generate_all<Us, Type, false, true, false>(pos, emit);

        return queenSide ? generate_all<Us, Type, false, false, true>(pos, emit)
                         : generate_all<Us, Type, false, false, false>(pos, emit);
    }
    else
        return pos.ep_square() != SQ_NONE ? generate_all<Us, Type, true>(pos, emit)
                                         : generate_all<Us, Type, false>(pos, emit);
}


template<GenType Type, typename Emitter>
void generate_to(const Position& pos, Emitter& emit) {

    if constexpr (Type == ALL)
    {
        if (pos.checkers())
            generate_to<EVASIONS>(pos, emit);
        else
        {
            generate_to<CAPTURES>(pos, emit);
            generate_to<QUIETS>(pos, emit);
        }
    }
    else
    {
        assert((Type == EVASIONS) == bool(pos.checkers()));

        if (pos.side_to_move() == WHITE)
            generate_for_side<WHITE, Type>(pos, emit);
        else
            generate_for_side<BLACK, Type>(pos, emit);
    }
}

}  // namespace Stockfish::MovegenDetail

namespace Stockfish {

// Calls callback(Move) in generation order. The callback may make and undo moves,
// but must restore the position before returning. Any return value is ignored.
template<GenType Type = ALL, typename Callback>
void for_each_move(const Position& pos, Callback&& callback) {
    MovegenDetail::CallbackWriter<Callback> emit{callback};
    MovegenDetail::generate_to<Type>(pos, emit);
}

}  // namespace Stockfish

#endif  // #ifndef MOVEGEN_CALLBACKS_H_INCLUDED
