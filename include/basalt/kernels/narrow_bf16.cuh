// include/basalt/kernels/narrow_bf16.cuh - FP8 E4M3 and FP4 E2M1 codes as bf16x2 for a kernel's mma operands.  Device
// code only (no kernel): include from a .cu.
//
// Every E4M3 and E2M1 value is exact in bf16.  `cvt.rn.bf16x2.e4m3x2` and `cvt.rn.bf16x2.e2m1x2` (one instruction
// each) need PTX ISA 9.2, CUDA 13.2: the ptxas of CUDA 13.0 and 13.1 rejects them.  There the codes go to f16x2
// (`cvt.rn.f16x2.*`, also exact) and each half widens through f32 to bf16, which gives the same bits, NaNs included.
// tests/kernels/narrow_bf16_parity.cu compares the two ways on every input. Below compute capability 10.0 (no narrow
// cvt) the codes are decoded in integer arithmetic and mul.bf16x2 runs as fma.bf16x2 with a -0 addend: the same bits.
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define BASALT_CVT_BF16X2_NARROW 1
#else
#define BASALT_CVT_BF16X2_NARROW 0
#endif

namespace basalt::kernels {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#define BASALT_NARROW_SOFT 1
#else
#define BASALT_NARROW_SOFT 0
#endif

/// a * b per half, rounded once (exact for the products below)
__device__ __forceinline__ unsigned bf16x2_mul_rn(unsigned a, unsigned b) {
    unsigned d;
#if BASALT_NARROW_SOFT
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
#else
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
#endif
    return d;
}

/// one E4M3 byte as bf16 bits (exact; NaN as the canonical bf16 NaN)
__device__ __forceinline__ unsigned e4m3_bf16_soft(unsigned b) {
    const unsigned s = (b & 0x80u) << 8, e = (b >> 3) & 15u, m = b & 7u;
    if (e == 15u && m == 7u) return 0x7FFFu;
    if (e) return s | ((e + 120u) << 7) | (m << 4);
    return s | (__float_as_uint((float) m * 0x1p-9f) >> 16);
}

/// two E4M3 bytes (the low 16 bits of x, the low byte .x) as fp32, exact. Below compute capability 8.9 (no E4M3 cvt):
/// the code's bits as an f16 of 2^-8 times its value, widened and times 256, both exact (the NaN code 0x7F/0xFF reads
/// as 480: weights and activations hold no NaN codes)
__device__ __forceinline__ float2 e4m3x2_float2(unsigned x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    const unsigned t = __byte_perm(x, 0u, 0x4140u);   // bytes [b0, 0, b1, 0]
    const unsigned h = ((t & 0x007F007Fu) << 7) | ((t & 0x00800080u) << 8);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
    return make_float2(f.x * 256.0f, f.y * 256.0f);
#else
    return __half22float2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (x & 0xFFFFu), __NV_E4M3));
#endif
}

/// one E2M1 nibble as bf16 bits (exact): magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6
__device__ __forceinline__ unsigned e2m1_bf16_soft(unsigned n) {
    const unsigned m = n & 7u;
    const unsigned mag = m < 2u ? m * (126u << 7) : ((126u + (m >> 1)) << 7) | ((m & 1u) << 6);
    return ((n & 8u) << 12) | mag;
}

/// f16x2 as bf16x2, for values bf16 holds exactly (each half through f32)
__device__ __forceinline__ unsigned f16x2_bf16x2(unsigned f) {
    unsigned h;
    asm("{\n"
        ".reg .f16 l, u;\n"
        ".reg .f32 a, b;\n"
        "mov.b32 {l, u}, %1;\n"
        "cvt.f32.f16 a, l;\n"
        "cvt.f32.f16 b, u;\n"
        "cvt.rn.bf16x2.f32 %0, b, a;\n"
        "}\n"
        : "=r"(h)
        : "r"(f));
    return h;
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 through f16x2
__device__ __forceinline__ unsigned e4m3x2_bf16x2_via_f16(unsigned x) {
    unsigned f;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f) : "h"(v));
    return f16x2_bf16x2(f);
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 through f16x2
__device__ __forceinline__ void e2m1x8_bf16x2_via_f16(unsigned w, unsigned (&h)[4]) {
    unsigned f[4];
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3])
        : "r"(w));
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = f16x2_bf16x2(f[i]);
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 (exact)
__device__ __forceinline__ unsigned e4m3x2_bf16x2(unsigned x) {
#if BASALT_NARROW_SOFT
    return e4m3_bf16_soft(x & 0xFFu) | (e4m3_bf16_soft((x >> 8) & 0xFFu) << 16);
#elif BASALT_CVT_BF16X2_NARROW
    unsigned h;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
    return h;
#else
    return e4m3x2_bf16x2_via_f16(x);
#endif
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 times the bf16x2 scale sc (exact:
/// a code times a UE4M3 scale has at most five significant bits)
__device__ __forceinline__ void e2m1x8_bf16x2_scaled(unsigned w, unsigned sc, unsigned (&h)[4]) {
#if BASALT_NARROW_SOFT
#pragma unroll
    for (int i = 0; i < 4; ++i)
        h[i] = bf16x2_mul_rn(e2m1_bf16_soft((w >> (8 * i)) & 15u) | (e2m1_bf16_soft((w >> (8 * i + 4)) & 15u) << 16), sc);
#elif BASALT_CVT_BF16X2_NARROW
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        ".reg .b32 t0, t1, t2, t3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 t0, b0;\n"
        "cvt.rn.bf16x2.e2m1x2 t1, b1;\n"
        "cvt.rn.bf16x2.e2m1x2 t2, b2;\n"
        "cvt.rn.bf16x2.e2m1x2 t3, b3;\n"
        "mul.rn.bf16x2 %0, t0, %5;\n"
        "mul.rn.bf16x2 %1, t1, %5;\n"
        "mul.rn.bf16x2 %2, t2, %5;\n"
        "mul.rn.bf16x2 %3, t3, %5;\n"
        "}\n"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w), "r"(sc));
#else
    unsigned t[4];
    e2m1x8_bf16x2_via_f16(w, t);
#pragma unroll
    for (int i = 0; i < 4; ++i) asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(h[i]) : "r"(t[i]), "r"(sc));
#endif
}

}  // namespace basalt::kernels
