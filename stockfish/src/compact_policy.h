// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef COMPACT_POLICY_H
#define COMPACT_POLICY_H
#include <algorithm>
#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <string>
#include "types.h"
#if defined(USE_AVX2)
#include <immintrin.h>
#endif
namespace Stockfish {
// Quantized student: 1024 shared NNUE features -> variable-width ReLU units -> move factors.
// Scores add origin, destination, and promotion embeddings, oriented to side to move.
class CompactPolicy {
    std::vector<i16> projection;
    std::vector<i16> bias;
    std::vector<i16> embedding;
    int width = 32;
    bool joint = false;
    int inputs = 1024;
    std::vector<int> activeUnits;
    static i64 dot(const i16* weights, const u8* features) {
#if defined(USE_AVX2)
        // Each int32 lane sums 128 int16*u8 products, safely below INT32_MAX.
        __m256i accum = _mm256_setzero_si256();
        for (int i=0;i<1024;i+=16) {
            const auto x = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(features+i)));
            const auto w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights+i));
            accum = _mm256_add_epi32(accum, _mm256_madd_epi16(w,x));
        }
        alignas(32) i32 lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),accum);
        i64 total=0;
        for (auto lane:lanes) total+=lane;
        return total;
#else
        i64 total=0;
        for (int i=0;i<1024;++i) total+=i64(weights[i])*features[i];
        return total;
#endif
    }
    void prepare() {
        activeUnits.clear();
        for (int j=0;j<width;++j) {
            for (int f=0;f<(joint ? 4101 : 133);++f)
                if (embedding[f*width+j]) { activeUnits.push_back(j); break; }
        }
    }
    std::string configuration;
    bool ready = false;
 public:
    static std::array<int, 3> factors(Move m, Color side) {
        int flip = side == BLACK ? 56 : 0;
        return {int(m.from_sq()) ^ flip, 64 + (int(m.to_sq()) ^ flip),
                128 + (m.type_of() == PROMOTION ? int(m.promotion_type()) - 1 : 0)};
    }
    bool load(const std::string& path, const std::string& eval, std::string& error) {
        if (configuration == path + "\n" + eval) return ready;
        configuration = path + "\n" + eval;
        ready = false;
        std::ifstream in(path, std::ios::binary);
        char signature[8]{};
        in.read(signature, 8);
        joint = std::string(signature, 8) == std::string("SFPOLI3\0", 8);
        inputs = joint ? 1026 : 1024;
        const bool binary = joint || std::string(signature, 8) == std::string("SFPOLI2\0", 8);
        auto number = [&](int bytes) -> u32 {
            u32 value = 0;
            for (int i = 0; i < bytes; ++i) {
                char c = 0; in.get(c); value |= u32(static_cast<unsigned char>(c)) << (i * 8);
            }
            return value;
        };
        if (binary) {
            const u32 w = number(4), length = number(4);
            if (!in || w == 0 || w > 1024 || length > 4096) {
                error = "Invalid compact policy header"; return false;
            }
            std::string name(length, '\0'); in.read(name.data(), length);
            if (!in || name != eval) { error = "Compact policy EvalFile mismatch"; return false; }
            width = int(w);
            projection.resize(width * inputs); bias.resize(width); embedding.resize((joint ? 4101 : 133) * width);
            for (auto* values : {&projection, &bias, &embedding})
                for (auto& v : *values) { const u32 n = number(2); v = i16(n < 32768 ? int(n) : int(n) - 65536); }
            if (!in || in.peek() != std::char_traits<char>::eof()) {
                error = "Invalid compact policy payload"; return false;
            }
            prepare(); ready = true; return true;
        }
        in.clear(); in.seekg(0);
        width = 32; projection.resize(width * inputs); bias.resize(width); embedding.resize((joint ? 4101 : 133) * width);
        std::string magic, expected;
        if (!(in >> magic >> expected) || magic != "SFPOLICY1" || expected != eval) {
            error = "Missing/incompatible compact policy (must match EvalFile)"; return false;
        }
        auto read = [&](auto& values) {
            for (auto& v : values) {
                int n;
                if (!(in >> n) || n < -32768 || n > 32767) return false;
                v = i16(n);
            }
            return true;
        };
        if (!read(projection) || !read(bias) || !read(embedding)) {
            error = "Invalid compact policy weights"; return false;
        }
        std::string extra;
        if (in >> extra) { error = "Unexpected compact policy data"; return false; }
        prepare(); ready = true; return true;
    }
    struct Context {
        std::array<float, 1024> hidden{};
        mutable std::array<float, 133> squares{};
        mutable std::array<bool, 133> computed{};
    };
    Context context(const u8* features, bool ownMirror, bool opponentMirror) const {
        Context h;
        for (int j : activeUnits) {
            i64 sum = i64(bias[j]) * 127;
            sum += dot(projection.data()+j*inputs, features);
            if (joint) {
                sum += i64(projection[j*inputs+1024]) * (ownMirror ? 127 : 0);
                sum += i64(projection[j*inputs+1025]) * (opponentMirror ? 127 : 0);
            }
            h.hidden[j] = std::max(0.0f, float(sum) / (4096.0f * 127.0f));
        }
        return h;
    }
    float score(const Context& h, Move m, Color side) const {
        float value = 0;
        if (joint) {
            auto f = factors(m, side);
            const int moveIndex = f[0]*64 + f[1]-64;
            const int promotionIndex = 4096 + f[2]-128;
            for (int j : activeUnits)
                value += h.hidden[j] * (int(embedding[moveIndex*width+j]) + int(embedding[promotionIndex*width+j]));
            return value / 4096.0f;
        }
        for (int factor : factors(m, side)) {
            if (!h.computed[factor]) {
                float sum = 0;
                for (int j : activeUnits)
                    sum += h.hidden[j] * embedding[factor * width + j];
                h.squares[factor] = sum / 4096.0f;
                h.computed[factor] = true;
            }
            value += h.squares[factor];
        }
        return value;
    }
};
}
#endif
