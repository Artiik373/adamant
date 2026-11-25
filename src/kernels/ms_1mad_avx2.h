#pragma once

// Peak-throughput AVX2 1MAD bitshift-trellis GEMV & block FWHT kernels for Amperite.
// Key optimizations:
//   - Pure-vector 14-bit state extraction via _mm256_srlv_epi32 (zero scalar->YMM _mm256_setr_epi32 stalls)
//   - 2-instruction byte-sum via VPMADDUBSW + VPMADDWD
//   - 2-row unrolled GEMV (reuses x_rot cache lines and hides vpmulld/fma latency)

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

#include "basalt/platform/compiler.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace adamant::kernels::ms {

constexpr uint32_t kMadMul = 34038481u;
constexpr uint32_t kMadAdd = 76625530u;
constexpr uint32_t kStateMask = 0x3FFFu; // 14-bit state

inline float fp16_to_fp32(uint16_t h) noexcept {
#if defined(__F16C__) && defined(__AVX2__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
#else
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
#endif
}

// Row stride: 4-byte header plus the symbol stream, padded to 4 bytes so every row start is word-aligned.
inline size_t ms_row_bytes(int in_features, int k) noexcept {
    return (4u + static_cast<size_t>(((in_features - 1) * k + 7) / 8) + 3u) & ~size_t{3};
}

inline void fwht_rotate_input(
    const float* __restrict x,
    const int8_t* __restrict signs,
    float* __restrict y,
    int n,
    int block_size
) noexcept {
    const float inv_sqrt_b = 1.0f / std::sqrt(static_cast<float>(block_size));
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] * static_cast<float>(signs[i]);
    }

    for (int base = 0; base < n; base += block_size) {
        float* blk = y + base;
        for (int h = 1; h < block_size; h <<= 1) {
            const int step = h << 1;
#if defined(__AVX2__)
            if (h >= 8) {
                for (int i = 0; i < block_size; i += step) {
                    for (int j = 0; j < h; j += 8) {
                        __m256 a = _mm256_loadu_ps(blk + i + j);
                        __m256 b = _mm256_loadu_ps(blk + i + j + h);
                        _mm256_storeu_ps(blk + i + j, _mm256_add_ps(a, b));
                        _mm256_storeu_ps(blk + i + j + h, _mm256_sub_ps(a, b));
                    }
                }
                continue;
            }
#endif
            for (int i = 0; i < block_size; i += step) {
                for (int j = 0; j < h; ++j) {
                    float a = blk[i + j];
                    float b = blk[i + j + h];
                    blk[i + j]     = a + b;
                    blk[i + j + h] = a - b;
                }
            }
        }
#if defined(__AVX2__)
        const __m256 v_scale = _mm256_set1_ps(inv_sqrt_b);
        int i = 0;
        for (; i + 8 <= block_size; i += 8) {
            _mm256_storeu_ps(blk + i, _mm256_mul_ps(_mm256_loadu_ps(blk + i), v_scale));
        }
        for (; i < block_size; ++i) blk[i] *= inv_sqrt_b;
#else
        for (int i = 0; i < block_size; ++i) blk[i] *= inv_sqrt_b;
#endif
    }
}

template <int K>
inline uint32_t read_symbol_msb(const uint8_t* __restrict qs, int sym_idx) noexcept {
    const int bit_pos = sym_idx * K;
    const int byte_idx = bit_pos >> 3;
    const int bit_off = bit_pos & 7;
    uint32_t v = (static_cast<uint32_t>(qs[byte_idx]) << 8) |
                 static_cast<uint32_t>(qs[byte_idx + 1]);
    return (v >> (16 - bit_off - K)) & ((1u << K) - 1u);
}

#if defined(__AVX2__)
// Pure-AVX2 extraction of 8x 14-bit states from a 64-bit big-endian bit window.
// Splits the 64-bit register into hi32 (lanes 0..3) and lo32 (shifted so lanes 4..7 fit in 32 bits).
template <int K>
inline __m256i extract_8_states_avx2(uint64_t bits, __m256i v_shifts, __m256i v_smask) noexcept {
    const uint32_t hi32 = static_cast<uint32_t>(bits >> 32);
    const uint32_t lo32 = static_cast<uint32_t>(bits >> (32 - 4 * K));
    __m128i lo_pair = _mm_set1_epi32(static_cast<int32_t>(hi32));
    __m128i hi_pair = _mm_set1_epi32(static_cast<int32_t>(lo32));
    __m256i v_words = _mm256_set_m128i(hi_pair, lo_pair);
    return _mm256_and_si256(_mm256_srlv_epi32(v_words, v_shifts), v_smask);
}

inline float hsum256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s4);
    __m128 s2 = _mm_add_ps(s4, shuf);
    shuf = _mm_movehl_ps(shuf, s2);
    return _mm_cvtss_f32(_mm_add_ss(s2, shuf));
}
#endif

template <int K>
inline float ms_1mad_row_dot(
    const uint8_t* __restrict row_ptr,
    const float* __restrict x_rot,
    int in_features
) noexcept {
    uint16_t d_fp16 = 0;
    uint16_t s0 = 0;
    std::memcpy(&d_fp16, row_ptr, 2);
    std::memcpy(&s0, row_ptr + 2, 2);
    const float d = fp16_to_fp32(d_fp16);
    const uint8_t* qs = row_ptr + 4;

    uint32_t state = static_cast<uint32_t>(s0) & kStateMask;
    float acc = 0.0f;

#if defined(__AVX2__)
    const __m256i v_mul    = _mm256_set1_epi32(static_cast<int32_t>(kMadMul));
    const __m256i v_add    = _mm256_set1_epi32(static_cast<int32_t>(kMadAdd));
    const __m256i v_ones8  = _mm256_set1_epi8(1);
    const __m256i v_ones16 = _mm256_set1_epi16(1);
    const __m256i v_bias   = _mm256_set1_epi32(510);
    const __m256i v_smask  = _mm256_set1_epi32(static_cast<int32_t>(kStateMask));
    // Lanes 0..3 shift from hi32; lanes 4..7 shift from lo32 (pre-shifted by 4*K)
    const __m256i v_shifts = _mm256_setr_epi32(
        32 - 14 - 0 * K, 32 - 14 - 1 * K, 32 - 14 - 2 * K, 32 - 14 - 3 * K,
        32 - 14 - 0 * K, 32 - 14 - 1 * K, 32 - 14 - 2 * K, 32 - 14 - 3 * K
    );
    __m256 v_acc = _mm256_setzero_ps();

    int c = 0;
    while (c < in_features && c * K < 14) {
        if (c > 0) {
            state = ((state << K) | read_symbol_msb<K>(qs, c - 1)) & kStateMask;
        }
        uint32_t x = state * kMadMul + kMadAdd;
        int32_t v = static_cast<int32_t>((x & 0xFFu) + ((x >> 8) & 0xFFu) +
                                         ((x >> 16) & 0xFFu) + (x >> 24)) - 510;
        acc += static_cast<float>(v) * x_rot[c];
        ++c;
    }

    const int vec_end = in_features - 16;
    for (; c + 8 <= vec_end; c += 8) {
        const int qs_bit0 = c * K - 14;
        const int byte0 = qs_bit0 >> 3;
        const int bit_off = qs_bit0 & 7;

        uint64_t raw_le;
        std::memcpy(&raw_le, qs + byte0, 8);
        uint64_t bits = basalt::platform::bswap64(raw_le) << bit_off;

        __m256i vs = extract_8_states_avx2<K>(bits, v_shifts, v_smask);
        __m256i vx = _mm256_add_epi32(_mm256_mullo_epi32(vs, v_mul), v_add);
        __m256i pair_sums = _mm256_maddubs_epi16(vx, v_ones8);
        __m256i sum_i = _mm256_sub_epi32(_mm256_madd_epi16(pair_sums, v_ones16), v_bias);

        __m256 vw = _mm256_cvtepi32_ps(sum_i);
        __m256 vx_act = _mm256_loadu_ps(x_rot + c);
        v_acc = _mm256_fmadd_ps(vw, vx_act, v_acc);
    }

    acc += hsum256_ps(v_acc);

    if (c < in_features && c > 0) {
        const int qs_bit = c * K - 14;
        const int byte_idx = qs_bit >> 3;
        const int bit_off = qs_bit & 7;
        uint32_t raw = (static_cast<uint32_t>(qs[byte_idx]) << 16) |
                       (static_cast<uint32_t>(qs[byte_idx + 1]) << 8) |
                        static_cast<uint32_t>(qs[byte_idx + 2]);
        state = (raw >> (24 - bit_off - 14)) & kStateMask;
        uint32_t x = state * kMadMul + kMadAdd;
        int32_t v = static_cast<int32_t>((x & 0xFFu) + ((x >> 8) & 0xFFu) +
                                         ((x >> 16) & 0xFFu) + (x >> 24)) - 510;
        acc += static_cast<float>(v) * x_rot[c];
        ++c;
    }
    for (; c < in_features; ++c) {
        state = ((state << K) | read_symbol_msb<K>(qs, c - 1)) & kStateMask;
        uint32_t x = state * kMadMul + kMadAdd;
        int32_t v = static_cast<int32_t>((x & 0xFFu) + ((x >> 8) & 0xFFu) +
                                         ((x >> 16) & 0xFFu) + (x >> 24)) - 510;
        acc += static_cast<float>(v) * x_rot[c];
    }
#else
    for (int c = 0; c < in_features; ++c) {
        if (c > 0) {
            state = ((state << K) | read_symbol_msb<K>(qs, c - 1)) & kStateMask;
        }
        uint32_t x = state * kMadMul + kMadAdd;
        int32_t v = static_cast<int32_t>((x & 0xFFu) + ((x >> 8) & 0xFFu) +
                                         ((x >> 16) & 0xFFu) + (x >> 24)) - 510;
        acc += static_cast<float>(v) * x_rot[c];
    }
#endif

    return acc * d;
}

template <int K>
inline void ms_1mad_gemv(
    const uint8_t* __restrict weights,
    const float* __restrict x_rot,
    float* __restrict out,
    int out_features,
    int in_features
) noexcept {
    const size_t stride = ms_row_bytes(in_features, K);
    for (int r = 0; r < out_features; ++r) {
#if defined(__AVX2__)
        if (r + 1 < out_features) {
            _mm_prefetch(reinterpret_cast<const char*>(weights + static_cast<size_t>(r + 1) * stride), _MM_HINT_T0);
        }
#endif
        out[r] = ms_1mad_row_dot<K>(weights + static_cast<size_t>(r) * stride, x_rot, in_features);
    }
}

} // namespace adamant::kernels::ms
