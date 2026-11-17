// src/kernels/cpu/exl3.cpp - EXL3 projections on the CPU.  See the header.
#include "basalt/kernels/cpu/exl3.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "exl3_ring.hpp"

#include <immintrin.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

#define BASALT_EXL3_VNNI 0
#define BASALT_EXL3_COLS cols_avx2
#include "exl3_cols.inc"

namespace basalt::kernels::cpu {
namespace exl3_detail {
void cols_avx2_vnni(const Exl3Proj& p, const float* const* xp, const float* bias, int nt, float* const* y1, int c0,
                    int c1);
void cols_q8_vnni(const Exl3Proj& p, const int8_t* const* xq, const float* const* sc, const float* bias, int nt,
                  float* const* y1, int c0, int c1);
}
using namespace exl3_detail;
namespace {

float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, u;
    if (e == 0) {
        if (m == 0) u = s;
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            u = s | (e << 23) | ((m & 0x3FF) << 13);
        }
    } else if (e == 31) u = s | 0x7F800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t float_to_half(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t e = (int32_t) ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFF) == 0xFF) return (uint16_t) (sign | 0x7C00u | (m ? 0x200u : 0u));   // inf / nan
    if (e >= 31) return (uint16_t) (sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t h = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t) (sign | h);
    }
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t) (sign | h);
}

inline float code_mul1(uint32_t win) {
    const uint32_t x = win * kMul;
    const uint32_t s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
    return ((float) s - 510.f) * kScale;
}

void hadamard128(float* x) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
    const float s = 1.f / std::sqrt(128.f);
    for (int i = 0; i < 128; ++i) x[i] *= s;
}

// One 128-block of the trellis product in place: the output transform and svh.
void store_block(const Exl3Proj& p, float* blk, int variant, int b) {
    if (variant & kExl3SignBeforeHadOut) {
        for (int i = 0; i < 128; ++i) blk[i] *= half_to_float(p.svh[b * 128 + i]);
        hadamard128(blk);
    } else {
        hadamard128(blk);
        for (int i = 0; i < 128; ++i) blk[i] *= half_to_float(p.svh[b * 128 + i]);
    }
}

}  // namespace

namespace {
std::atomic<int> g_variant{0};
}
int exl3_variant() { return g_variant.load(std::memory_order_relaxed); }
void exl3_set_variant(int variant) { g_variant.store(variant & 3, std::memory_order_relaxed); }

bool exl3_q8() {
    static const bool vnni = cpu_avxvnni_ok();
    return vnni;
}

bool exl3_geometry_ok(int k, int n, int w) {
    return k > 0 && n > 0 && k % kExl3Block == 0 && n % kExl3Block == 0 && w > 0 && w % 8 == 0 && w <= kExl3MaxWords;
}

void exl3_prepare_input(const float* x, const uint16_t* suh, int k, int variant, float* x2) {
    for (int b = 0; b < k / 128; ++b) {
        float* o = x2 + b * 128;
        if (variant & kExl3SignAfterHadIn) {
            std::memcpy(o, x + b * 128, 128 * sizeof(float));
            hadamard128(o);
            for (int i = 0; i < 128; ++i) o[i] *= half_to_float(suh[b * 128 + i]);
        } else {
            for (int i = 0; i < 128; ++i) o[i] = x[b * 128 + i] * half_to_float(suh[b * 128 + i]);
            hadamard128(o);
        }
    }
}

void exl3_cols_ref(const Exl3Proj& p, const float* x2, float* y1, int c0, int c1) {
    const RingTable& rt = ring_table(p.w);
    for (int nt = c0 / 16; nt < c1 / 16; ++nt) {
        double acc[16] = {};
        for (int kt = 0; kt < p.k / 16; ++kt) {
            const uint32_t* wd = tile_words(p, kt, nt);
            for (int i = 0; i < 256; ++i) {
                const int t = i >> 3, j = i & 7;
                const int r = (t & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0), c = (t >> 2) + ((j & 4) ? 8 : 0);
                acc[c] += (double) x2[kt * 16 + r] * (double) code_mul1(be_bits(wd, rt, i));
            }
        }
        for (int c = 0; c < 16; ++c) y1[nt * 16 + c] = (float) acc[c];
    }
}

void exl3_finish(const Exl3Proj& p, float* y, int b0, int b1, int variant) {
    for (int b = b0; b < b1; ++b) store_block(p, y + b * 128, variant, b);
}

bool exl3_cols_avx2_ok(int w) { return ring_table(w).fast[1]; }

void exl3_expand_input(const float* x2, int k, float* xp, float* bias) {
    for (int kt = 0; kt < k / 16; ++kt)
        for (int q = 0; q < 4; ++q) {
            const float* x = x2 + kt * 16 + q * 2;
            _mm256_store_ps(xp + ((size_t) kt * 4 + q) * 8, _mm256_setr_ps(x[0], x[1], x[8], x[9], x[0], x[1], x[8], x[9]));
        }
    float sumx = 0.f;
    for (int i = 0; i < k; ++i) sumx += x2[i];
    *bias = 510.f * sumx;
}

void exl3_expand_input_q8(const float* x2, int k, int8_t* xq, float* sc, float* bias) {
    // per k tile, group g's 32 bytes: rows 2g, 2g + 1, 2g + 8, 2g + 9 (columns c and c + 8), each byte four times
    alignas(32) static const uint8_t kRep[4][32] = {
        {0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 8, 8, 9, 9, 9, 9, 0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 8, 8, 9, 9, 9, 9},
        {2, 2, 2, 2, 3, 3, 3, 3, 10, 10, 10, 10, 11, 11, 11, 11, 2, 2, 2, 2, 3, 3, 3, 3, 10, 10, 10, 10, 11, 11, 11, 11},
        {4, 4, 4, 4, 5, 5, 5, 5, 12, 12, 12, 12, 13, 13, 13, 13, 4, 4, 4, 4, 5, 5, 5, 5, 12, 12, 12, 12, 13, 13, 13, 13},
        {6, 6, 6, 6, 7, 7, 7, 7, 14, 14, 14, 14, 15, 15, 15, 15, 6, 6, 6, 6, 7, 7, 7, 7, 14, 14, 14, 14, 15, 15, 15, 15}};
    const __m256 sign = _mm256_set1_ps(-0.f);
    double sum = 0.0;
    for (int b = 0; b < k / 128; ++b) {
        const float* xb = x2 + b * 128;
        __m256 m = _mm256_setzero_ps();
        for (int i = 0; i < 128; i += 8) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(xb + i)));
        alignas(32) float mv[8];
        _mm256_store_ps(mv, m);
        float mx = 0.f;
        for (float v : mv) mx = std::fmax(mx, v);
        const float s = mx > 0.f ? mx / 127.f : 1.f;
        const __m256 inv = _mm256_set1_ps(1.f / s);
        __m256i qs = _mm256_setzero_si256();
        for (int kt = 0; kt < 8; ++kt) {
            // 16 values rounded to nearest even (as lrint), packed to bytes in order
            const __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + kt * 16), inv));
            const __m256i c = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + kt * 16 + 8), inv));
            qs = _mm256_add_epi32(qs, _mm256_add_epi32(a, c));
            const __m128i w16 = _mm_packs_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
            const __m128i w16b = _mm_packs_epi32(_mm256_castsi256_si128(c), _mm256_extracti128_si256(c, 1));
            const __m256i src = _mm256_broadcastsi128_si256(_mm_packs_epi16(w16, w16b));
            int8_t* dst = xq + (size_t) (b * 8 + kt) * 128;
            for (int g = 0; g < 4; ++g)
                _mm256_store_si256((__m256i*) (dst + g * 32),
                                   _mm256_shuffle_epi8(src, _mm256_load_si256((const __m256i*) kRep[g])));
        }
        alignas(32) int32_t qv[8];
        _mm256_store_si256((__m256i*) qv, qs);
        int qsum = 0;
        for (int v : qv) qsum += v;
        sc[b] = s;
        sum += (double) s * qsum;
    }
    *bias = (float) (510.0 * sum);
}

void exl3_cols_q8(const Exl3Proj& p, const int8_t* const* xq, const float* const* sc, const float* bias, int nt,
                  float* const* y1, int c0, int c1) {
    exl3_detail::cols_q8_vnni(p, xq, sc, bias, nt, y1, c0, c1);
}

void exl3_cols_expanded_avx2(const Exl3Proj& p, const float* const* xp, const float* bias, int nt, float* const* y1,
                             int c0, int c1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) exl3_detail::cols_avx2_vnni(p, xp, bias, nt, y1, c0, c1);
    else exl3_detail::cols_avx2(p, xp, bias, nt, y1, c0, c1);
}

void exl3_cols_avx2(const Exl3Proj& p, const float* const* x2, int nt, float* const* y1, int c0, int c1) {
    if (!exl3_cols_avx2_ok(p.w)) {   // under 2 bits per weight: the windows of the first groups wrap; not a rate EXL3 ships here
        for (int t = 0; t < nt; ++t) exl3_cols_ref(p, x2[t], y1[t], c0, c1);
        return;
    }
    thread_local std::vector<float> store, bias;
    thread_local std::vector<const float*> ptr;
    const size_t per = (size_t) p.k * 2;
    store.resize(per * (size_t) nt + 8);
    bias.resize((size_t) nt);
    ptr.resize((size_t) nt);
    float* xp = (float*) (((uintptr_t) store.data() + 31) & ~(uintptr_t) 31);
    for (int t = 0; t < nt; ++t) {
        exl3_expand_input(x2[t], p.k, xp + (size_t) t * per, &bias[(size_t) t]);
        ptr[(size_t) t] = xp + (size_t) t * per;
    }
    exl3_cols_expanded_avx2(p, ptr.data(), bias.data(), nt, y1, c0, c1);
}

namespace {
// Rows [r0, r1) through the column kernel: the 128-blocks the range touches, transformed, its part stored.
template <class Cols>
void rows_by_blocks(const Exl3Proj& p, float* y, int r0, int r1, int variant, Cols cols) {
    thread_local std::vector<float> tmp;
    tmp.resize((size_t) p.n);
    const int b0 = r0 / 128, b1 = (r1 + 127) / 128;
    cols(tmp.data(), b0 * 128, b1 * 128);
    exl3_finish(p, tmp.data(), b0, b1, variant);
    std::memcpy(y + r0, tmp.data() + r0, (size_t) (r1 - r0) * sizeof(float));
}
}  // namespace

void exl3_rows_ref(const Exl3Proj& p, const float* x2, float* y, int r0, int r1, int variant) {
    rows_by_blocks(p, y, r0, r1, variant, [&](float* t, int c0, int c1) { exl3_cols_ref(p, x2, t, c0, c1); });
}

void exl3_rows_avx2(const Exl3Proj& p, const float* x2, float* y, int r0, int r1, int variant) {
    rows_by_blocks(p, y, r0, r1, variant, [&](float* t, int c0, int c1) { exl3_cols_avx2(p, &x2, 1, &t, c0, c1); });
}


// Both conversions round to nearest even, the F16C instructions' bits and the plain C ones' alike.
void exl3_pack_act(const float* x, int n, uint16_t* dst) {
    static const bool avx2 = cpu_avx2_ok();
    int i = 0;
    if (avx2)
        for (; i + 8 <= n; i += 8)
            _mm_storeu_si128((__m128i*) (dst + i),
                             _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    for (; i < n; ++i) dst[i] = float_to_half(x[i]);
}

void exl3_unpack_act(const uint16_t* src, int n, float* x) {
    static const bool avx2 = cpu_avx2_ok();
    int i = 0;
    if (avx2)
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (src + i))));
    for (; i < n; ++i) x[i] = half_to_float(src[i]);
}

namespace {

// The normalized Hadamard transform of 128 floats held in 16 registers: strides 1, 2 and 4 inside a register, 8..64
// across registers.
inline void hadamard128_avx2(float* p) {
    __m256 r[16];
    const __m256 s1 = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1), s2 = _mm256_setr_ps(1, 1, -1, -1, 1, 1, -1, -1),
                 s4 = _mm256_setr_ps(1, 1, 1, 1, -1, -1, -1, -1);
    for (int i = 0; i < 16; ++i) {
        __m256 x = _mm256_loadu_ps(p + i * 8);
        x = _mm256_fmadd_ps(x, s1, _mm256_permute_ps(x, 0xB1));
        x = _mm256_fmadd_ps(x, s2, _mm256_permute_ps(x, 0x4E));
        r[i] = _mm256_fmadd_ps(x, s4, _mm256_permute2f128_ps(x, x, 0x01));
    }
    for (int h = 1; h < 16; h <<= 1)
        for (int i = 0; i < 16; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const __m256 a = r[j], b = r[j + h];
                r[j] = _mm256_add_ps(a, b);
                r[j + h] = _mm256_sub_ps(a, b);
            }
    const __m256 sc = _mm256_set1_ps(0.08838834764831845f);   // 1 / sqrt(128)
    for (int i = 0; i < 16; ++i) _mm256_storeu_ps(p + i * 8, _mm256_mul_ps(r[i], sc));
}

}  // namespace

void exl3_prepare_input_avx2(const float* x, const uint16_t* suh, int k, int variant, float* x2) {
    for (int b = 0; b < k / 128; ++b) {
        float* o = x2 + b * 128;
        const float* xi = x + b * 128;
        const uint16_t* sg = suh + b * 128;
        if (variant & kExl3SignAfterHadIn) {
            for (int i = 0; i < 128; i += 8) _mm256_storeu_ps(o + i, _mm256_loadu_ps(xi + i));
            hadamard128_avx2(o);
            for (int i = 0; i < 128; i += 8)
                _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_loadu_ps(o + i),
                                                      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (sg + i)))));
        } else {
            for (int i = 0; i < 128; i += 8)
                _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_loadu_ps(xi + i),
                                                      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (sg + i)))));
            hadamard128_avx2(o);
        }
    }
}

}  // namespace basalt::kernels::cpu
