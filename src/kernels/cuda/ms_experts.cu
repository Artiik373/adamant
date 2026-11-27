// src/kernels/cuda/ms_experts.cu - MS (1MAD trellis) routed experts on the GPU: native_expert_grouped's MS path.
// A blob is three MS payloads [gate | up | down] (kernels/cpu/native_expert.hpp, ms_k).  Per layer:
//   1. each entry's input rotated once by the gate's and the up's signs, H(signs * x), into scratch
//   2. gate and up rows, a warp per 4 rows, each weight row read and decoded once per <= 4 entries of its group
//   3. h = SwiGLU, rotated by the down's signs in the same pass
//   4. down rows as 2.
#include "basalt/kernels/ms_kernels.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/log.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "../ms_1mad_cuda.cuh"

#include <cstdlib>
#include <type_traits>

namespace basalt::kernels {
namespace {

namespace mc = adamant::kernels::ms_cuda;

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;

// The gate payload's bytes (gate and up are the same size): the up payload's offset, half the down's.
template <int K>
__device__ __forceinline__ size_t gu_bytes() { return (size_t) 640 * mc::ms_row_bytes(2560, K) + 2560; }

__device__ __forceinline__ size_t gu_bytes_k(int k) { return (size_t) 640 * mc::ms_row_bytes(2560, k) + 2560; }

// A mixed layer's blob: projection proj's K and payload.
__device__ __forceinline__ const uint8_t* mixed_proj(unsigned long long ptr, int proj, int& k) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(ptr);
    const int kg = __ldg(b);
    const int ku = __ldg(b + 1) ? (int) __ldg(b + 1) : kg;
    const int kd = __ldg(b + 2) ? (int) __ldg(b + 2) : kg;
    k = proj == 0 ? kg : proj == 1 ? ku : kd;
    const size_t off = proj == 0 ? 0 : proj == 1 ? gu_bytes_k(kg) : gu_bytes_k(kg) + gu_bytes_k(ku);
    return b + basalt::kernels::cpu::kMsMixHead + off;
}

// The signs of projection proj (n_in inputs, n_out rows) of a group's blob; K = 0: a mixed layer's, from its head.
template <int K>
__device__ __forceinline__ const int8_t* proj_signs(unsigned long long ptr, int proj, int n_in, int n_out) {
    if constexpr (K > 0) {
        return reinterpret_cast<const int8_t*>(reinterpret_cast<const uint8_t*>(ptr) + (size_t) proj * gu_bytes<K>() +
                                               (size_t) n_out * mc::ms_row_bytes(n_in, K));
    } else {
        int k;
        const uint8_t* p = mixed_proj(ptr, proj, k);
        return reinterpret_cast<const int8_t*>(p + (size_t) n_out * mc::ms_row_bytes(n_in, k));
    }
}

// xr[e][z] = H(signs_z * x[ent_tok[e]]) for z = gate, up (blockIdx.z): a warp per Hadamard block (blockIdx.x),
// two warps per block for multi-entry groups, a group per blockIdx.y.
template <int K>
__global__ void __launch_bounds__(64) ms_rotate_in_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok, const mc::MsQ81* __restrict__ x,
    float* __restrict__ xr, int n_ff) {
    constexpr int N = 2560, HB = 512;
    const int ng = *n_groups;
    const int z = blockIdx.z;
    const int first = blockIdx.x * HB;
    const int w = threadIdx.x >> 5;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const int8_t* sg = proj_signs<K>(grp_ptr[g], z, N, n_ff);
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0 + w; e < e1; e += 2) {
            const mc::MsQ81* xq = x + (size_t) ent_tok[e] * (N / 32);
            mc::ms_rotate_block_warp(xr + ((size_t) e * 2 + z) * N, [&](int i) {
                const mc::MsQ81& b = xq[i >> 5];
                return __half2float(b.d) * (float) b.qs[i & 31] * (float) __ldg(sg + i);
            }, first, HB);
        }
    }
}

// h[e] = H(signs_down * SwiGLU(gate[e], up[e])) (gate capped at `limit`, up clamped to +-limit, when limit > 0)
template <int K>
__global__ void __launch_bounds__(64) ms_swiglu_rot_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ h, float limit) {
    constexpr int N = 640, N_OUT = 2560, HB = 128;
    const int ng = *n_groups;
    const int first = blockIdx.x * HB;
    const int w = threadIdx.x >> 5;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const int8_t* sg = proj_signs<K>(grp_ptr[g], 2, N, N_OUT);
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0 + w; e < e1; e += 2) {
            const float* ge = gate + (size_t) e * N;
            const float* ue = up + (size_t) e * N;
            mc::ms_rotate_block_warp(h + (size_t) e * N, [&](int i) {
                float a = ge[i], b = ue[i];
                if (limit > 0.0f) {
                    a = fminf(a, limit);
                    b = fmaxf(fminf(b, limit), -limit);
                }
                return a / (1.0f + __expf(-a)) * b * (float) __ldg(sg + i);
            }, first, HB);
        }
    }
}

// NE (1 or 2) entries of 4 rows: keeps register count within __launch_bounds__(128, 6) while halving 2+ entry loads.
template <int K, int N_IN, int NE, bool IS_GU>
__device__ __forceinline__ void proj_4row_ne(const uint8_t* __restrict__ p, int eb,
                                             const int32_t* __restrict__ ent_dst, const float* __restrict__ xr,
                                             float* __restrict__ out, int r, int n_out, int lane, float rs, int z) {
    constexpr int STEPS = N_IN / 128;
    const int stride_words = (int) (mc::ms_row_bytes(N_IN, K) >> 2);
    const int lane_bit = 4 * lane * K + 18, w_lane = lane_bit >> 5, b_lane = lane_bit & 31;
    const uint32_t* g32_0 = reinterpret_cast<const uint32_t*>(p) + (size_t) r * stride_words;
    const uint32_t* g32_1 = g32_0 + stride_words;
    const uint32_t* g32_2 = g32_1 + stride_words;
    const uint32_t* g32_3 = g32_2 + stride_words;
    const uint32_t hdr0 = __ldg(g32_0), hdr1 = __ldg(g32_1), hdr2 = __ldg(g32_2), hdr3 = __ldg(g32_3);
    const float s0 = __half2float(__ushort_as_half((unsigned short) (hdr0 & 0xFFFFu))) * rs;
    const float s1 = __half2float(__ushort_as_half((unsigned short) (hdr1 & 0xFFFFu))) * rs;
    const float s2 = __half2float(__ushort_as_half((unsigned short) (hdr2 & 0xFFFFu))) * rs;
    const float s3 = __half2float(__ushort_as_half((unsigned short) (hdr3 & 0xFFFFu))) * rs;

    const float4* xv4_0 = reinterpret_cast<const float4*>(
        IS_GU ? xr + ((size_t) eb * 2 + z) * N_IN : xr + (size_t) eb * N_IN);
    const float4* xv4_1 = NE == 2 ? reinterpret_cast<const float4*>(
        IS_GU ? xr + ((size_t) (eb + 1) * 2 + z) * N_IN : xr + (size_t) (eb + 1) * N_IN) : nullptr;
    float acc0_0 = 0.0f, acc1_0 = 0.0f, acc2_0 = 0.0f, acc3_0 = 0.0f;
    float acc0_1 = 0.0f, acc1_1 = 0.0f, acc2_1 = 0.0f, acc3_1 = 0.0f;

    constexpr int CHUNK = (STEPS % 4 == 0) ? 4 : STEPS;
    for (int st0 = 0; st0 < STEPS; st0 += CHUNK) {
        uint32_t wa0[CHUNK], wb0[CHUNK], wa1[CHUNK], wb1[CHUNK];
        uint32_t wa2[CHUNK], wb2[CHUNK], wa3[CHUNK], wb3[CHUNK];
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            const int wi = w_lane + 4 * K * (st0 + i);
            wa0[i] = __ldg(g32_0 + wi); wb0[i] = wi + 1 < stride_words ? __ldg(g32_0 + wi + 1) : 0u;
            wa1[i] = __ldg(g32_1 + wi); wb1[i] = wi + 1 < stride_words ? __ldg(g32_1 + wi + 1) : 0u;
            wa2[i] = __ldg(g32_2 + wi); wb2[i] = wi + 1 < stride_words ? __ldg(g32_2 + wi + 1) : 0u;
            wa3[i] = __ldg(g32_3 + wi); wb3[i] = wi + 1 < stride_words ? __ldg(g32_3 + wi + 1) : 0u;
        }
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            const int st = st0 + i;
            const int wi = w_lane + 4 * K * st;
            const uint32_t a0 = wi == 0 ? (hdr0 >> 16) & mc::kStateMask : __byte_perm(wa0[i], 0, 0x0123);
            const uint32_t v0 = __funnelshift_l(__byte_perm(wb0[i], 0, 0x0123), a0, b_lane);
            const uint32_t a1 = wi == 0 ? (hdr1 >> 16) & mc::kStateMask : __byte_perm(wa1[i], 0, 0x0123);
            const uint32_t v1 = __funnelshift_l(__byte_perm(wb1[i], 0, 0x0123), a1, b_lane);
            const uint32_t a2 = wi == 0 ? (hdr2 >> 16) & mc::kStateMask : __byte_perm(wa2[i], 0, 0x0123);
            const uint32_t v2 = __funnelshift_l(__byte_perm(wb2[i], 0, 0x0123), a2, b_lane);
            const uint32_t a3 = wi == 0 ? (hdr3 >> 16) & mc::kStateMask : __byte_perm(wa3[i], 0, 0x0123);
            const uint32_t v3 = __funnelshift_l(__byte_perm(wb3[i], 0, 0x0123), a3, b_lane);
            const int col = lane + 32 * st;
            const float4 xv0 = __ldg(xv4_0 + col);
            if constexpr (NE == 1) {
                acc0_0 = fmaf(mc::decode_1mad_dp4a((v0 >> 18) & mc::kStateMask), xv0.x,
                         fmaf(mc::decode_1mad_dp4a((v0 >> (18 - K)) & mc::kStateMask), xv0.y,
                         fmaf(mc::decode_1mad_dp4a((v0 >> (18 - 2 * K)) & mc::kStateMask), xv0.z,
                         fmaf(mc::decode_1mad_dp4a((v0 >> (18 - 3 * K)) & mc::kStateMask), xv0.w, acc0_0))));
                acc1_0 = fmaf(mc::decode_1mad_dp4a((v1 >> 18) & mc::kStateMask), xv0.x,
                         fmaf(mc::decode_1mad_dp4a((v1 >> (18 - K)) & mc::kStateMask), xv0.y,
                         fmaf(mc::decode_1mad_dp4a((v1 >> (18 - 2 * K)) & mc::kStateMask), xv0.z,
                         fmaf(mc::decode_1mad_dp4a((v1 >> (18 - 3 * K)) & mc::kStateMask), xv0.w, acc1_0))));
                acc2_0 = fmaf(mc::decode_1mad_dp4a((v2 >> 18) & mc::kStateMask), xv0.x,
                         fmaf(mc::decode_1mad_dp4a((v2 >> (18 - K)) & mc::kStateMask), xv0.y,
                         fmaf(mc::decode_1mad_dp4a((v2 >> (18 - 2 * K)) & mc::kStateMask), xv0.z,
                         fmaf(mc::decode_1mad_dp4a((v2 >> (18 - 3 * K)) & mc::kStateMask), xv0.w, acc2_0))));
                acc3_0 = fmaf(mc::decode_1mad_dp4a((v3 >> 18) & mc::kStateMask), xv0.x,
                         fmaf(mc::decode_1mad_dp4a((v3 >> (18 - K)) & mc::kStateMask), xv0.y,
                         fmaf(mc::decode_1mad_dp4a((v3 >> (18 - 2 * K)) & mc::kStateMask), xv0.z,
                         fmaf(mc::decode_1mad_dp4a((v3 >> (18 - 3 * K)) & mc::kStateMask), xv0.w, acc3_0))));
            } else {
                const float4 xv1 = __ldg(xv4_1 + col);
                const float w00 = mc::decode_1mad_dp4a((v0 >> 18) & mc::kStateMask);
                const float w01 = mc::decode_1mad_dp4a((v0 >> (18 - K)) & mc::kStateMask);
                const float w02 = mc::decode_1mad_dp4a((v0 >> (18 - 2 * K)) & mc::kStateMask);
                const float w03 = mc::decode_1mad_dp4a((v0 >> (18 - 3 * K)) & mc::kStateMask);
                acc0_0 = fmaf(w00, xv0.x, fmaf(w01, xv0.y, fmaf(w02, xv0.z, fmaf(w03, xv0.w, acc0_0))));
                acc0_1 = fmaf(w00, xv1.x, fmaf(w01, xv1.y, fmaf(w02, xv1.z, fmaf(w03, xv1.w, acc0_1))));

                const float w10 = mc::decode_1mad_dp4a((v1 >> 18) & mc::kStateMask);
                const float w11 = mc::decode_1mad_dp4a((v1 >> (18 - K)) & mc::kStateMask);
                const float w12 = mc::decode_1mad_dp4a((v1 >> (18 - 2 * K)) & mc::kStateMask);
                const float w13 = mc::decode_1mad_dp4a((v1 >> (18 - 3 * K)) & mc::kStateMask);
                acc1_0 = fmaf(w10, xv0.x, fmaf(w11, xv0.y, fmaf(w12, xv0.z, fmaf(w13, xv0.w, acc1_0))));
                acc1_1 = fmaf(w10, xv1.x, fmaf(w11, xv1.y, fmaf(w12, xv1.z, fmaf(w13, xv1.w, acc1_1))));

                const float w20 = mc::decode_1mad_dp4a((v2 >> 18) & mc::kStateMask);
                const float w21 = mc::decode_1mad_dp4a((v2 >> (18 - K)) & mc::kStateMask);
                const float w22 = mc::decode_1mad_dp4a((v2 >> (18 - 2 * K)) & mc::kStateMask);
                const float w23 = mc::decode_1mad_dp4a((v2 >> (18 - 3 * K)) & mc::kStateMask);
                acc2_0 = fmaf(w20, xv0.x, fmaf(w21, xv0.y, fmaf(w22, xv0.z, fmaf(w23, xv0.w, acc2_0))));
                acc2_1 = fmaf(w20, xv1.x, fmaf(w21, xv1.y, fmaf(w22, xv1.z, fmaf(w23, xv1.w, acc2_1))));

                const float w30 = mc::decode_1mad_dp4a((v3 >> 18) & mc::kStateMask);
                const float w31 = mc::decode_1mad_dp4a((v3 >> (18 - K)) & mc::kStateMask);
                const float w32 = mc::decode_1mad_dp4a((v3 >> (18 - 2 * K)) & mc::kStateMask);
                const float w33 = mc::decode_1mad_dp4a((v3 >> (18 - 3 * K)) & mc::kStateMask);
                acc3_0 = fmaf(w30, xv0.x, fmaf(w31, xv0.y, fmaf(w32, xv0.z, fmaf(w33, xv0.w, acc3_0))));
                acc3_1 = fmaf(w30, xv1.x, fmaf(w31, xv1.y, fmaf(w32, xv1.z, fmaf(w33, xv1.w, acc3_1))));
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc0_0 += __shfl_down_sync(0xFFFFFFFFu, acc0_0, o);
        acc1_0 += __shfl_down_sync(0xFFFFFFFFu, acc1_0, o);
        acc2_0 += __shfl_down_sync(0xFFFFFFFFu, acc2_0, o);
        acc3_0 += __shfl_down_sync(0xFFFFFFFFu, acc3_0, o);
        if constexpr (NE == 2) {
            acc0_1 += __shfl_down_sync(0xFFFFFFFFu, acc0_1, o);
            acc1_1 += __shfl_down_sync(0xFFFFFFFFu, acc1_1, o);
            acc2_1 += __shfl_down_sync(0xFFFFFFFFu, acc2_1, o);
            acc3_1 += __shfl_down_sync(0xFFFFFFFFu, acc3_1, o);
        }
    }
    if (lane == 0) {
        const size_t b0 = (size_t) (IS_GU ? eb : ent_dst[eb]) * n_out + r;
        out[b0] = acc0_0 * s0; out[b0 + 1] = acc1_0 * s1; out[b0 + 2] = acc2_0 * s2; out[b0 + 3] = acc3_0 * s3;
        if constexpr (NE == 2) {
            const size_t b1 = (size_t) (IS_GU ? eb + 1 : ent_dst[eb + 1]) * n_out + r;
            out[b1] = acc0_1 * s0; out[b1 + 1] = acc1_1 * s1; out[b1 + 2] = acc2_1 * s2; out[b1 + 3] = acc3_1 * s3;
        }
    }
}

template <int K, int N_IN, bool IS_GU>
__device__ __forceinline__ void proj_4row_group(const uint8_t* __restrict__ p, int e0, int e1,
                                                const int32_t* __restrict__ ent_dst, const float* __restrict__ xr,
                                                float* __restrict__ out, int r, int n_out, int lane, float rs, int z) {
    int eb = e0;
    if constexpr (IS_GU) {
        for (; eb + 1 < e1; eb += 2) proj_4row_ne<K, N_IN, 2, IS_GU>(p, eb, ent_dst, xr, out, r, n_out, lane, rs, z);
    }
    for (; eb < e1; ++eb) proj_4row_ne<K, N_IN, 1, IS_GU>(p, eb, ent_dst, xr, out, r, n_out, lane, rs, z);
}

template <int K, bool MIX>
__global__ void __launch_bounds__(kThreads, 6) ms_gu_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const float* __restrict__ xr, float* __restrict__ out0,
    float* __restrict__ out1, int n_out) {
    const int ng = *n_groups;
    const int lane = threadIdx.x & 31;
    const int r = 4 * (blockIdx.x * kWarps + (threadIdx.x >> 5));
    if (r >= n_out) return;
    float* __restrict__ out = blockIdx.z ? out1 : out0;
    const float rs = rsqrtf((float) mc::ms_hadamard_block(2560));
    const int proj = blockIdx.z ? 1 : 0;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if constexpr (MIX) {
            int gk;
            const uint8_t* p = mixed_proj(grp_ptr[g], proj, gk);
            switch (gk) {
            case 2: proj_4row_group<2, 2560, true>(p, e0, e1, nullptr, xr, out, r, n_out, lane, rs, blockIdx.z); break;
            case 3: proj_4row_group<3, 2560, true>(p, e0, e1, nullptr, xr, out, r, n_out, lane, rs, blockIdx.z); break;
            case 4: proj_4row_group<4, 2560, true>(p, e0, e1, nullptr, xr, out, r, n_out, lane, rs, blockIdx.z); break;
            case 5: proj_4row_group<5, 2560, true>(p, e0, e1, nullptr, xr, out, r, n_out, lane, rs, blockIdx.z); break;
            default: proj_4row_group<6, 2560, true>(p, e0, e1, nullptr, xr, out, r, n_out, lane, rs, blockIdx.z); break;
            }
        } else {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(grp_ptr[g]) + (size_t) proj * gu_bytes<K>();
            proj_4row_group<K, 2560, true>(p, e0, e1, nullptr, xr, out, r, n_out, lane, rs, blockIdx.z);
        }
    }
}

template <int K, bool MIX>
__global__ void __launch_bounds__(kThreads, 6) ms_down_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst, const float* __restrict__ xr,
    float* __restrict__ out, int n_out) {
    const int ng = *n_groups;
    const int lane = threadIdx.x & 31;
    const int r = 4 * (blockIdx.x * kWarps + (threadIdx.x >> 5));
    if (r >= n_out) return;
    const float rs = rsqrtf((float) mc::ms_hadamard_block(640));
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if constexpr (MIX) {
            int gk;
            const uint8_t* p = mixed_proj(grp_ptr[g], 2, gk);
            switch (gk) {
            case 2: proj_4row_group<2, 640, false>(p, e0, e1, ent_dst, xr, out, r, n_out, lane, rs, 0); break;
            case 3: proj_4row_group<3, 640, false>(p, e0, e1, ent_dst, xr, out, r, n_out, lane, rs, 0); break;
            case 4: proj_4row_group<4, 640, false>(p, e0, e1, ent_dst, xr, out, r, n_out, lane, rs, 0); break;
            case 5: proj_4row_group<5, 640, false>(p, e0, e1, ent_dst, xr, out, r, n_out, lane, rs, 0); break;
            default: proj_4row_group<6, 640, false>(p, e0, e1, ent_dst, xr, out, r, n_out, lane, rs, 0); break;
            }
        } else {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(grp_ptr[g]) + 2 * gu_bytes<K>();
            proj_4row_group<K, 640, false>(p, e0, e1, ent_dst, xr, out, r, n_out, lane, rs, 0);
        }
    }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <int K>   // 0: a mixed layer
void grouped_k(float limit, const unsigned long long* grp_ptr, const int32_t* grp_start,
               const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, unsigned gy,
               const void* x_q8_1, float* gate, float* up, float* h, float* xr, float* out, cudaStream_t s) {
    constexpr int NE = 2560, NF = 640;
    ms_rotate_in_kernel<K><<<dim3(5, gy, 2), 64, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok,
                                                          static_cast<const mc::MsQ81*>(x_q8_1), xr, NF);
    check("ms_expert_grouped/rotate");
    ms_gu_kernel<K, K == 0><<<dim3(NF / (4 * kWarps), gy, 2), kThreads, 0, s>>>(
        grp_ptr, grp_start, n_groups, xr, gate, up, NF);
    check("ms_expert_grouped/gu");
    ms_swiglu_rot_kernel<K><<<dim3(5, gy), 64, 0, s>>>(grp_ptr, grp_start, n_groups, gate, up, h, limit);
    check("ms_expert_grouped/swiglu");
    ms_down_kernel<K, K == 0><<<dim3(NE / (4 * kWarps), gy, 1), kThreads, 0, s>>>(
        grp_ptr, grp_start, n_groups, ent_dst, h, out, NE);
    check("ms_expert_grouped/down");
}

}  // namespace

bool ms_expert_supported(int type, int64_t n_embd, int64_t n_ff) noexcept {
    return basalt::ms_expert_type((uint32_t) type) && n_embd == 2560 && n_ff == 640;   // grouped_k's geometry
}

void ms_expert_grouped(const NativeExpertLayout& L, float limit, const unsigned long long* grp_ptr,
                       const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                       const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,
                       void* scratch, float* out, void* stream, int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (!ms_expert_supported(L.gu_type, L.n_embd, L.n_ff) || L.d_type != L.gu_type || L.up_type != L.gu_type) {
        basalt::log::error("basalt", "ms_expert_grouped: types %d/%d/%d, %lld x %lld", L.gu_type, L.up_type, L.d_type,
                           (long long) L.n_embd, (long long) L.n_ff);
        std::exit(1);
    }
    // native_expert_scratch_bytes: gate, up, h (cap x n_ff floats each), then the rotated inputs (2 x n_embd an entry)
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    float* xr = (float*) ((uint8_t*) scratch + 3 * fa);
    const unsigned gy = (unsigned) ((grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups);
    cudaStream_t s = (cudaStream_t) stream;
    switch (basalt::ms_mix_type((uint32_t) L.gu_type) ? 0 : basalt::ms_bits((uint32_t) L.gu_type)) {
    case 0: grouped_k<0>(limit, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, gy, x_q8_1, gate, up, h, xr, out, s); break;
    case 2: grouped_k<2>(limit, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, gy, x_q8_1, gate, up, h, xr, out, s); break;
    case 3: grouped_k<3>(limit, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, gy, x_q8_1, gate, up, h, xr, out, s); break;
    case 4: grouped_k<4>(limit, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, gy, x_q8_1, gate, up, h, xr, out, s); break;
    case 5: grouped_k<5>(limit, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, gy, x_q8_1, gate, up, h, xr, out, s); break;
    default: grouped_k<6>(limit, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, gy, x_q8_1, gate, up, h, xr, out, s); break;
    }
}

}  // namespace basalt::kernels