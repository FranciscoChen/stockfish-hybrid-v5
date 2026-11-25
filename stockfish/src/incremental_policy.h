// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef INCREMENTAL_POLICY_H
#define INCREMENTAL_POLICY_H
#include <array>
#include <vector>
#include <fstream>
#include <streambuf>
#include <algorithm>
#include <string>
#include <cstdlib>
#include "position.h"
#include "movegen.h"
#include "compact_policy.h"
#include "policy_dot.h"
namespace Stockfish {
#ifdef CHESSIL_EMBED_POLICY
class EmbeddedPolicyBuffer : public std::streambuf {
public:
    static constexpr auto name = "<embedded>";
    void open();
};
#endif

// V4: 16 king buckets x 12 relative piece types x 64 squares, plus
// four castling rights and eight en-passant files. Shared two-perspective
// accumulator; width denotes the concatenated side-to-move/opponent output.
class IncrementalPolicy {
public:
    static constexpr int Features = 12300;
    struct Context { std::array<i16, 1024> hidden{}; int stack = 0; };
    struct Frame {
        std::array<Piece, 64> board{};
        std::array<int, 2> flip{}, bucket{};
        std::array<std::array<bool, 12>, 2> auxiliary{};
        std::array<std::array<i32, 512>, 2> sums{};
    };
private:
    int width_ = 0, half = 0, ply = 0, stacks = 1;
    std::vector<i16> weights, bias, output;
    std::array<bool, 4> ordinaryFolded{};
    // Heap storage is allocated only when a policy is loaded.
    std::vector<Frame> frames;
    std::array<int, MAX_PLY + 1> parentFrame;
    int lastFrame = -1;
    bool loaded = false;
    u64 generation_ = 0;
    std::string configuration;
    static int index(int c, Piece p, int s, const Frame& frame) {
        const int relative = (int(color_of(p)) != c) * 6 + int(type_of(p)) - 1;
        return frame.bucket[c] * 768 + relative * 64 + (s ^ frame.flip[c]);
    }
    static void describe(const Position& pos, Frame& f) {
        f.board = pos.piece_array();
        f.auxiliary = {};
        for (int c = 0; c < 2; ++c) {
            int king = int(pos.square<KING>(Color(c))) ^ (c == BLACK ? 56 : 0);
            const int mirror = (king % 8 >= 4) ? 7 : 0;
            f.flip[c] = (c == BLACK ? 56 : 0) ^ mirror;
            king ^= mirror;
            f.bucket[c] = (king / 8) * 2 + (king % 8) / 2;
            for (int opponent = 0; opponent < 2; ++opponent)
                for (int wing = 0; wing < 2; ++wing) {
                    const int actualWing = wing ^ bool(mirror);
                    auto right = CastlingRights(1 << ((c ^ opponent) * 2 + actualWing));
                    f.auxiliary[c][opponent * 2 + wing] = pos.can_castle(right);
                }
            if (pos.ep_square() != SQ_NONE)
                f.auxiliary[c][4 + ((int(pos.ep_square()) ^ f.flip[c]) % 8)] = true;
        }
    }
    template<int Sign>
    void add(Frame& f, int c, int feature) const {
        const i16* row = weights.data() + feature * half;
        int j = 0;
#if defined(USE_AVX2)
        for (; j + 8 <= half; j += 8) {
            auto sum = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&f.sums[c][j]));
            const auto delta = _mm256_cvtepi16_epi32(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j)));
            if constexpr (Sign > 0) sum = _mm256_add_epi32(sum, delta);
            else sum = _mm256_sub_epi32(sum, delta);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&f.sums[c][j]), sum);
        }
#endif
        for (; j < half; ++j) f.sums[c][j] += Sign * int(row[j]);
    }
    void refresh(Frame& f, int c) const {
        for (int j = 0; j < half; ++j) f.sums[c][j] = bias[j];
        for (int s = 0; s < 64; ++s)
            if (f.board[s] != NO_PIECE) add<1>(f, c, index(c, f.board[s], s, f));
        for (int a = 0; a < 12; ++a) if (f.auxiliary[c][a]) add<1>(f, c, 12288 + a);
    }
    // Reuse the board comparison across perspectives and copy only active units.
    void update(Frame& f, const Frame& parent, int c,
                const int* changed, int changedCount, int pieceCount) const {
        int edits = 0, freshRows = pieceCount;
        for (int i = 0; i < changedCount; ++i) {
            edits += parent.board[changed[i]] != NO_PIECE;
            edits += f.board[changed[i]] != NO_PIECE;
        }
        for (int a = 0; a < 12; ++a) {
            freshRows += f.auxiliary[c][a];
            edits += f.auxiliary[c][a] != parent.auxiliary[c][a];
        }
        if (edits > freshRows) {
            refresh(f, c);
            return;
        }
        std::copy_n(parent.sums[c].begin(), half, f.sums[c].begin());
        for (int i = 0; i < changedCount; ++i) {
            const int s = changed[i];
            if (parent.board[s] != NO_PIECE) add<-1>(f, c, index(c, parent.board[s], s, parent));
            if (f.board[s] != NO_PIECE) add<1>(f, c, index(c, f.board[s], s, f));
        }
        for (int a = 0; a < 12; ++a)
            if (f.auxiliary[c][a] != parent.auxiliary[c][a]) {
                if (f.auxiliary[c][a]) add<1>(f, c, 12288 + a);
                else add<-1>(f, c, 12288 + a);
            }
    }
    Context activate(const Frame& f, Color side, int pieceCount) const {
        Context ctx;
        if (stacks == 4) ctx.stack = std::clamp((pieceCount - 1) / 8, 0, 3);
        for (int p = 0; p < 2; ++p) for (int j = 0; j < half; ++j)
            ctx.hidden[p * half + j] = i16(std::clamp(f.sums[int(side) ^ p][j], 0, 4096));
        return ctx;
    }
public:
    static bool is_format(const std::string& path) {
#ifdef CHESSIL_EMBED_POLICY
        if (path == EmbeddedPolicyBuffer::name) return true;
#endif
        std::ifstream in(path, std::ios::binary); char magic[8]{}; in.read(magic, 8);
        return std::string(magic, 8) == std::string("SFPOLI4\0", 8)
            || std::string(magic, 8) == std::string("SFPOLI5\0", 8);
    }
    bool load(const std::string& path, std::string& error) {
        if (loaded && configuration == path) return true;
        loaded = false;
#ifdef CHESSIL_EMBED_POLICY
        if (path == EmbeddedPolicyBuffer::name) {
            EmbeddedPolicyBuffer buffer;
            buffer.open();
            std::istream in(&buffer);
            return load_stream(in, path, error);
        }
#endif
        std::ifstream in(path, std::ios::binary);
        return load_stream(in, path, error);
    }
private:
    bool load_stream(std::istream& in, const std::string& path, std::string& error) {
        char magic[8]{};
        in.read(magic, 8);
        auto number = [&](int bytes) { u32 n = 0; for (int i = 0; i < bytes; ++i) {
            char b = 0; in.get(b); n |= u32(static_cast<unsigned char>(b)) << (8 * i);
        } return n; };
        const auto w = number(4), features = number(4);
        const bool v5 = std::string(magic, 8) == std::string("SFPOLI5\0", 8);
        if (!in || (!v5 && std::string(magic, 8) != std::string("SFPOLI4\0", 8))
            || w < 2 || w > 1024 || w % 2 || features != Features) {
            error = "Invalid incremental policy header"; return false;
        }
        width_ = int(w); half = width_ / 2; stacks = v5 ? 4 : 1;
        weights.resize(Features * half); bias.resize(half); output.resize(4101 * width_ * stacks);
        const u16 endian = 1;
        for (auto* v : {&weights, &bias, &output}) {
            auto bytes = reinterpret_cast<char*>(v->data());
            in.read(bytes, std::streamsize(v->size() * sizeof(i16)));
            if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
                for (usize i = 0; i < v->size(); ++i) std::swap(bytes[2 * i], bytes[2 * i + 1]);
        }
        if (!in || in.peek() != std::char_traits<char>::eof()) {
            error = "Invalid incremental policy payload"; return false;
        }
        for (int s = 0; s < stacks; ++s)
            ordinaryFolded[s] = fold_policy_ordinary_category(output.data() + s * 4101 * width_, width_);
        frames.assign(MAX_PLY + 1, Frame{});
        loaded = true; configuration = path; ++generation_; reset(); return true;
    }
public:
    bool ready() const { return loaded; }
    int width() const { return width_; }
    u64 generation() const { return generation_; }
    void reset() { ply = 0; lastFrame = -1; }
    void push() { ++ply; }
    void pop() {
        if (lastFrame == ply) lastFrame = parentFrame[ply];
        --ply;
    }
    Frame fresh(const Position& pos) const {
        Frame f; describe(pos, f); for (int c = 0; c < 2; ++c) refresh(f, c); return f;
    }
    // No projection is paid at make/unmake or at nodes that do not request policy.
    // A requested frame updates from the nearest evaluated ancestor. Only rows
    // for changed squares/state are added/subtracted; king bucket/mirror changes
    // refresh the affected perspective. Pop reuses the parent's saved sums.
    Context context(const Position& pos, bool verify = false) {
        const int pieceCount = popcount(pos.pieces());
        if (ply >= int(frames.size())) return activate(fresh(pos), pos.side_to_move(), pieceCount);
        auto& f = frames[ply];
        if (lastFrame == ply) {
            if (verify && f.sums != fresh(pos).sums) std::abort();
            return activate(f, pos.side_to_move(), pieceCount);
        }
        describe(pos, f);
        const int ancestor = lastFrame;
        std::array<int, 64> changed;
        int changedCount = 0;
        if (ancestor >= 0)
            for (int s = 0; s < 64; ++s)
                if (f.board[s] != frames[ancestor].board[s]) changed[changedCount++] = s;
        for (int c = 0; c < 2; ++c) {
            if (ancestor < 0 || frames[ancestor].bucket[c] != f.bucket[c]
                || frames[ancestor].flip[c] != f.flip[c]) refresh(f, c);
            else update(f, frames[ancestor], c, changed.data(), changedCount, pieceCount);
        }
        if (verify && f.sums != fresh(pos).sums) std::abort();
        parentFrame[ply] = ancestor;
        lastFrame = ply;
        return activate(f, pos.side_to_move(), pieceCount);
    }
    // Debug command: exercise real move/undo/null transitions, including an
    // unevaluated intermediate frame and king refreshes, against full sums.
    void verify_tree(Position& pos, int depth = 2) {
        context(pos, true);
        if (!depth) return;
        int count = 0;
        for (Move move : MoveList<ALL>(pos)) {
            StateInfo st;
            pos.do_move(move, st); push();
            verify_tree(pos, depth - 1);
            pos.undo_move(move); pop(); context(pos, true);
            if (depth < 2 && ++count >= 3) break;
        }
        if (!pos.checkers()) {
            StateInfo st; pos.do_null_move(st); push();
            context(pos, true); pos.undo_null_move(); pop(); context(pos, true);
        }
    }
    // Reuse promotion-category factors within a shortlist. Ordinary quiets all
    // share category zero. Add exact integers before the one float conversion,
    // preserving score() bit-for-bit, including promotions and castling mapping.
    void scores(const Context& ctx, const Move* moves, float* logits, int count,
                const Position& pos) const {
        if (count == 1) { logits[0] = score(ctx, moves[0], pos); return; }
        std::array<i64, 5> category{};
        std::array<bool, 5> ready{};
        ready[0] = ordinaryFolded[ctx.stack];
        const Color side = pos.side_to_move();
        const int mirror = file_of(pos.square<KING>(side)) >= FILE_E ? 7 : 0;
        for (int i = 0; i < count; ++i) {
            const auto f = CompactPolicy::factors(moves[i], side);
            const int promo = f[2] - 128;
            if (!ready[promo]) {
                category[promo] = policy_factor_dot(ctx.hidden.data(),
                  output.data() + (ctx.stack * 4101 + 4096 + promo) * width_, width_);
                ready[promo] = true;
            }
            const int from = f[0] ^ mirror, to = (f[1] - 64) ^ mirror;
            const i64 sum = category[promo] + policy_factor_dot(ctx.hidden.data(),
              output.data() + (ctx.stack * 4101 + from * 64 + to) * width_, width_);
            logits[i] = float(sum) / (4096.0f * 4096.0f);
        }
    }
    float score(const Context& ctx, Move move, const Position& pos) const {
        auto f = CompactPolicy::factors(move, pos.side_to_move());
        const int mirror = file_of(pos.square<KING>(pos.side_to_move())) >= FILE_E ? 7 : 0;
        const int from = f[0] ^ mirror, to = (f[1] - 64) ^ mirror;
        const i16* row = output.data() + (ctx.stack * 4101 + from * 64 + to) * width_;
        const int category = f[2] - 128;
        const i64 sum = ordinaryFolded[ctx.stack] && category == 0
          ? policy_factor_dot(ctx.hidden.data(), row, width_)
          : policy_output_dot(ctx.hidden.data(), row,
              output.data() + (ctx.stack * 4101 + 4096 + category) * width_, width_);
        // Exact integer dot product, rounded once to float. Network-file scale
        // and the mathematical logit are unchanged; no additional quantization.
        return float(sum) / (4096.0f * 4096.0f);
    }
};
}
#endif
