// include/basalt/kernels/cpu/exl3.hpp - EXL3 (QTIP trellis) projections on the CPU.
//
// A projection W[k][n] is stored as 16x16 tiles of trellis bits, a per-input vector `suh` and a per-output vector
// `svh` (fp16).  y = x . W is evaluated as
//     x1 = H(x * suh)   ->   y1 = x1 . W_inner   ->   y = H(y1) * svh
// with H the normalized Hadamard transform on blocks of 128 and W_inner decoded tile by tile from the trellis.
// `variant` moves the sign vectors relative to the transforms (kExl3SignAfterHadIn, kExl3SignBeforeHadOut).
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

/// Expert type ids of EXL3 layers: kExl3TypeBase + words-per-tile / 8 (a tile of 16 bits per weight x bits).
inline constexpr int kExl3TypeBase = 96;
inline constexpr int kExl3MaxWords = 128;                 // 8 bits per weight
inline constexpr int kExl3Block = 128;                    // the Hadamard block
inline bool exl3_type(int t) { return t > kExl3TypeBase && t <= kExl3TypeBase + kExl3MaxWords / 8; }
inline int exl3_words(int t) { return (t - kExl3TypeBase) * 8; }

inline constexpr int kExl3SignAfterHadIn = 1;             // input: H(x) * suh instead of H(x * suh)
inline constexpr int kExl3SignBeforeHadOut = 2;           // output: H(y * svh) instead of H(y) * svh

/// The sign placement every EXL3 projection of the process uses (0 unless set).
int exl3_variant();
void exl3_set_variant(int variant);
/// Whether the CPU tier takes int8 activations (exl3_expand_input_q8 / exl3_cols_q8): when the CPU has AVX-VNNI.
bool exl3_q8();

struct Exl3Proj {
    const uint8_t* trellis = nullptr;   ///< int16 [k/16][n/16][w], little endian
    const uint16_t* suh = nullptr;      ///< fp16 [k]
    const uint16_t* svh = nullptr;      ///< fp16 [n]
    int k = 0, n = 0, w = 0;
};

/// Bytes of one projection record in a blob: trellis, then suh, then svh.
inline size_t exl3_proj_bytes(int k, int n, int w) {
    return (size_t) (k / 16) * (size_t) (n / 16) * (size_t) w * 2 + (size_t) (k + n) * 2;
}
/// The record's pointers (`p` points at its first byte).
inline Exl3Proj exl3_proj_at(const uint8_t* p, int k, int n, int w) {
    Exl3Proj r;
    r.trellis = p;
    r.suh = (const uint16_t*) (p + (size_t) (k / 16) * (size_t) (n / 16) * (size_t) w * 2);
    r.svh = r.suh + k;
    r.k = k; r.n = n; r.w = w;
    return r;
}

/// Whether a geometry is supported: k and n multiples of 128, w a multiple of 8 up to kExl3MaxWords.
bool exl3_geometry_ok(int k, int n, int w);

/// x2 = the input of the trellis product: x (k floats) with suh and the block Hadamard applied.
void exl3_prepare_input(const float* x, const uint16_t* suh, int k, int variant, float* x2);

/// The same on AVX2 + F16C (call only behind cpu_avx2_ok()).
void exl3_prepare_input_avx2(const float* x, const uint16_t* suh, int k, int variant, float* x2);
/// Activations travel between the pool's quantize and row steps as fp16.
void exl3_pack_act(const float* x, int n, uint16_t* dst);
void exl3_unpack_act(const uint16_t* src, int n, float* x);

/// y1[c] for output columns c in [c0, c1) (multiples of 16) from the prepared input, before the output transform
/// (plain C).  Columns are independent: any split gives the same values.
void exl3_cols_ref(const Exl3Proj& p, const float* x2, float* y1, int c0, int c1);
/// The same on AVX2 for `nt` prepared inputs, the trellis decoded once for all of them (call only behind
/// cpu_avx2_ok()).  A token's values do not depend on the others.
void exl3_cols_avx2(const Exl3Proj& p, const float* const* x2, int nt, float* const* y1, int c0, int c1);
/// The column kernel's own form of a prepared input, for callers that run many column ranges on one input:
/// exl3_expand_input fills xp (2 k floats, 32-byte aligned) and the bias; exl3_cols_expanded_avx2 is exl3_cols_avx2 on
/// them, bit for bit.  Only for a rate exl3_cols_avx2_ok accepts (2 bits per weight and more).
bool exl3_cols_avx2_ok(int w);
void exl3_expand_input(const float* x2, int k, float* xp, float* bias);
void exl3_cols_expanded_avx2(const Exl3Proj& p, const float* const* xp, const float* bias, int nt, float* const* y1,
                             int c0, int c1);
/// The int8 form of a prepared input: one scale a 128-block (sc, k / 128 floats), the values rounded to int8 and laid
/// out four times a lane (xq, 8 k bytes, 32-byte aligned), and the bias from the rounded values.  exl3_cols_q8 is the
/// column kernel on them (call only when exl3_q8()).
void exl3_expand_input_q8(const float* x2, int k, int8_t* xq, float* sc, float* bias);
void exl3_cols_q8(const Exl3Proj& p, const int8_t* const* xq, const float* const* sc, const float* bias, int nt,
                  float* const* y1, int c0, int c1);
/// The output transform and svh over 128-blocks [b0, b1) of y1, in place.
void exl3_finish(const Exl3Proj& p, float* y, int b0, int b1, int variant);

/// y[r] for r in [r0, r1) from the prepared input: the columns of the 128-blocks the range touches, then exl3_finish.
void exl3_rows_ref(const Exl3Proj& p, const float* x2, float* y, int r0, int r1, int variant);
/// The same on AVX2 (call only behind cpu_avx2_ok()).
void exl3_rows_avx2(const Exl3Proj& p, const float* x2, float* y, int r0, int r1, int variant);

}  // namespace basalt::kernels::cpu
