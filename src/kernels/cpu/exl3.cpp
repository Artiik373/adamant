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
void cols_avx2_vnni(const Exl3Proj& p, const float* const* x2, int nt, float* const* y1, int c0, int c1);
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

void exl3_cols_avx2(const Exl3Proj& p, const float* const* x2, int nt, float* const* y1, int c0, int c1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) exl3_detail::cols_avx2_vnni(p, x2, nt, y1, c0, c1);
    else exl3_detail::cols_avx2(p, x2, nt, y1, c0, c1);
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


void exl3_pack_act(const float* x, int n, uint16_t* dst) {
    for (int i = 0; i < n; ++i) dst[i] = float_to_half(x[i]);
}

void exl3_unpack_act(const uint16_t* src, int n, float* x) {
    for (int i = 0; i < n; ++i) x[i] = half_to_float(src[i]);
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
