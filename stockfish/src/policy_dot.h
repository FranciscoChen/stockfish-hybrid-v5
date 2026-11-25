// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef POLICY_DOT_H_INCLUDED
#define POLICY_DOT_H_INCLUDED
#include <limits>
#include "types.h"
#if defined(USE_AVX2)
#include <immintrin.h>
#endif

namespace Stockfish {

// Rewrite a stack once at load. The ordinary category becomes zero and
// promotions retain just their difference from it. Preflight the entire
// stack so every accepted rewrite is exact and fallback is atomic.
inline bool fold_policy_ordinary_category(i16* rows, int width) {
    constexpr int moves = 4096, categories = 5;
    i16* const ordinary = rows + moves * width;
    auto fits = [](int n) {
        return n >= std::numeric_limits<i16>::min() && n <= std::numeric_limits<i16>::max();
    };
    for (int i = 0; i < moves; ++i)
        for (int j = 0; j < width; ++j)
            if (!fits(rows[i * width + j] + ordinary[j])) return false;
    for (int p = 1; p < categories; ++p)
        for (int j = 0; j < width; ++j)
            if (!fits(ordinary[p * width + j] - ordinary[j])) return false;

    for (int i = 0; i < moves; ++i)
        for (int j = 0; j < width; ++j) rows[i * width + j] += ordinary[j];
    for (int p = 1; p < categories; ++p)
        for (int j = 0; j < width; ++j) ordinary[p * width + j] -= ordinary[j];
    for (int j = 0; j < width; ++j) ordinary[j] = 0;
    return true;
}

// V4 activations are exactly integers in [0, 4096]. Both output rows are int16.
// Four products fit int32 (absolute maximum 536870912); widen before summing
// across blocks, since a full 1024-unit dot product need not fit int32.
inline i64 policy_output_dot(const i16* hidden, const i16* move, const i16* promotion, int width) {
    i64 result = 0;
    int j = 0;
#if defined(USE_AVX2)
    auto low = _mm256_setzero_si256(), high = _mm256_setzero_si256();
    for (; j + 16 <= width; j += 16) {
        const auto h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hidden + j));
        const auto m = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(move + j));
        const auto p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(promotion + j));
        const auto pairs = _mm256_add_epi32(_mm256_madd_epi16(h, m), _mm256_madd_epi16(h, p));
        low = _mm256_add_epi64(low, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(pairs)));
        high = _mm256_add_epi64(high, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(pairs, 1)));
    }
    alignas(32) i64 lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), _mm256_add_epi64(low, high));
    for (i64 lane : lanes) result += lane;
#endif
    for (; j < width; ++j) result += i64(hidden[j]) * (int(move[j]) + int(promotion[j]));
    return result;
}

// Single output factor. Two products fit int32; widen each block before
// accumulating so extreme weights remain exact at every supported width.
inline i64 policy_factor_dot(const i16* hidden, const i16* weights, int width) {
    i64 result = 0;
    int j = 0;
#if defined(USE_AVX2)
    auto low = _mm256_setzero_si256(), high = _mm256_setzero_si256();
    for (; j + 64 <= width; j += 64) {
        auto pairs = _mm256_setzero_si256();
        for (int block = 0; block < 64; block += 16) {
            const auto h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hidden + j + block));
            const auto w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + j + block));
            pairs = _mm256_add_epi32(pairs, _mm256_madd_epi16(h, w));
        }
        // Eight products per lane: magnitude <= 1,073,741,824, safely int32.
        low = _mm256_add_epi64(low, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(pairs)));
        high = _mm256_add_epi64(high, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(pairs, 1)));
    }
    alignas(32) i64 lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), _mm256_add_epi64(low, high));
    for (i64 lane : lanes) result += lane;
#endif
    for (; j < width; ++j) result += i64(hidden[j]) * weights[j];
    return result;
}

} // namespace Stockfish
#endif
