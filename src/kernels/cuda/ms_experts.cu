// src/kernels/cuda/ms_experts.cu - MS (1MAD trellis) routed experts on the GPU: native_expert_grouped's MS path.
// A blob is three MS payloads [gate | up | down] (kernels/cpu/native_expert.hpp, ms_k).  Per layer:
//   1. each entry's input rotated once by the gate's and the up's signs, H(signs * x), into scratch
//   2. gate and up rows, a warp a row, each row decoded once for up to kNe entries of its group
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

constexpr int kNe = 4;        // entries a row is decoded for at once
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

template <int K, int N_IN>
__device__ __forceinline__ const int8_t* signs_of(const uint8_t* payload, int n_out) {
    return reinterpret_cast<const int8_t*>(payload + (size_t) n_out * mc::ms_row_bytes(N_IN, K));
}

// The gate payload's bytes (gate and up are the same size): the up payload's offset, half the down's.
template <int K>
__device__ __forceinline__ size_t gu_bytes() { return (size_t) 640 * mc::ms_row_bytes(2560, K) + 2560; }

// f(K, blob) for one group's blob.  K > 0: the layer's K.  K == 0: a mixed layer, the K in the blob's head.
template <int K, class F>
__device__ __forceinline__ void with_k(unsigned long long ptr, F&& f) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(ptr);
    if constexpr (K > 0) {
        f(std::integral_constant<int, K>{}, b);
    } else {
        const uint8_t* p = b + basalt::kernels::cpu::kMsMixHead;
        switch (__ldg(b)) {   // block-uniform
        case 2: f(std::integral_constant<int, 2>{}, p); break;
        case 3: f(std::integral_constant<int, 3>{}, p); break;
        case 4: f(std::integral_constant<int, 4>{}, p); break;
        case 5: f(std::integral_constant<int, 5>{}, p); break;
        default: f(std::integral_constant<int, 6>{}, p); break;
        }
    }
}

// xr[e][z] = H(signs_z * x[ent_tok[e]]) for z = gate, up (blockIdx.z): a block per (entry slot, group, z), a warp per
// Hadamard block.
template <int K>
__global__ void __launch_bounds__(kThreads) ms_rotate_in_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok, const mc::MsQ81* __restrict__ x,
    float* __restrict__ xr, int n_ff) {
    constexpr int N = 2560;
    const int ng = *n_groups;
    const int z = blockIdx.z;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) with_k<K>(grp_ptr[g], [&](auto kc, const uint8_t* b) {
        constexpr int KK = decltype(kc)::value;
        const int8_t* sg = signs_of<KK, N>(b + (z ? gu_bytes<KK>() : 0), n_ff);
        for (int e = grp_start[g] + blockIdx.x; e < grp_start[g + 1]; e += gridDim.x)
            mc::ms_stage_q8_rotated(xr + ((size_t) e * 2 + z) * N, x + (size_t) ent_tok[e] * (N / 32), sg, N,
                                    mc::ms_hadamard_block(N));
    });
}

// h[e] = H(signs_down * SwiGLU(gate[e], up[e])) (gate capped at `limit`, up clamped to +-limit, when limit > 0)
template <int K>
__global__ void __launch_bounds__(kThreads) ms_swiglu_rot_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ h, float limit) {
    constexpr int N = 640, N_OUT = 2560;
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) with_k<K>(grp_ptr[g], [&](auto kc, const uint8_t* b) {
        constexpr int KK = decltype(kc)::value;
        const int8_t* sg = signs_of<KK, N>(b + 2 * gu_bytes<KK>(), N_OUT);
        for (int e = grp_start[g] + blockIdx.x; e < grp_start[g + 1]; e += gridDim.x) {
            const float* ge = gate + (size_t) e * N;
            const float* ue = up + (size_t) e * N;
            mc::ms_stage_rotated(h + (size_t) e * N, [&](int i) {
                float a = ge[i], b = ue[i];
                if (limit > 0.0f) {
                    a = fminf(a, limit);
                    b = fmaxf(fminf(b, limit), -limit);
                }
                return a / (1.0f + expf(-a)) * b * (float) __ldg(sg + i);
            }, N, mc::ms_hadamard_block(N));
        }
    });
}

// One row (its words loaded) against NE entries from c0: the entry count is a constant, so the accumulators and the
// input loads are exactly the entries'.
template <int K, bool GU, int STEPS, int NE>
__device__ __forceinline__ void row_entries(const uint32_t (&wa)[STEPS], const uint32_t (&wb)[STEPS], uint32_t hdr,
                                            float scale, const float* __restrict__ xr,
                                            const int32_t* __restrict__ ent_dst, float* __restrict__ out, int c0,
                                            int r, int n_out, int lane, int w_lane, int b_lane) {
    constexpr int N_IN = 128 * STEPS;
    const float4* xv4[NE];
    #pragma unroll
    for (int j = 0; j < NE; ++j)
        xv4[j] = reinterpret_cast<const float4*>(xr + (GU ? (size_t) (c0 + j) * 2 + blockIdx.z : (size_t) (c0 + j)) * N_IN);
    float acc[NE] = {};
    #pragma unroll
    for (int st = 0; st < STEPS; ++st) {
        const int wi = w_lane + 4 * K * st;
        const uint32_t a = wi == 0 ? (hdr >> 16) & mc::kStateMask : __byte_perm(wa[st], 0, 0x0123);
        const uint32_t v = __funnelshift_l(__byte_perm(wb[st], 0, 0x0123), a, b_lane);
        const float d0 = mc::decode_1mad_dp4a((v >> 18) & mc::kStateMask);
        const float d1 = mc::decode_1mad_dp4a((v >> (18 - K)) & mc::kStateMask);
        const float d2 = mc::decode_1mad_dp4a((v >> (18 - 2 * K)) & mc::kStateMask);
        const float d3 = mc::decode_1mad_dp4a((v >> (18 - 3 * K)) & mc::kStateMask);
        const int c4 = lane + 32 * st;
        #pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float4 xv = __ldg(xv4[j] + c4);
            acc[j] = fmaf(d0, xv.x, fmaf(d1, xv.y, fmaf(d2, xv.z, fmaf(d3, xv.w, acc[j]))));
        }
    }
    #pragma unroll
    for (int j = 0; j < NE; ++j) {
        float sum = acc[j];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xFFFFFFFFu, sum, o);
        if (lane == 0) out[(GU ? (size_t) (c0 + j) : (size_t) ent_dst[c0 + j]) * n_out + r] = sum * scale;
    }
}

// Rows of one payload for every entry of every group, a warp a row.  GU: payload z = blockIdx.z (gate, up) at offset
// off_z, input xr[e][z], output row e of out_z; down: input xr[e], output row ent_dst[e] of out0.  A lane decodes 4
// columns a step from one 32-bit window of the row's big-endian stream; its bit offset is the same every step and its
// words advance by 4 K, so all of a row's loads are issued before the first is used.
template <int KL, bool GU, int STEPS>   // KL: the layer's K, 0 for a mixed layer
__global__ void __launch_bounds__(kThreads, STEPS <= 5 ? 4 : 3) ms_rows_kernel(   // short rows: more warps in flight
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst, const float* __restrict__ xr,
    float* __restrict__ out0, float* __restrict__ out1, int n_out) {
    constexpr int N_IN = 128 * STEPS;
    const int ng = *n_groups;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (r >= n_out) return;
    float* __restrict__ out = blockIdx.z ? out1 : out0;
    const float rs = rsqrtf((float) mc::ms_hadamard_block(N_IN));
    for (int g = blockIdx.y; g < ng; g += gridDim.y) with_k<KL>(grp_ptr[g], [&](auto kc, const uint8_t* b) {
        constexpr int K = decltype(kc)::value;
        constexpr int stride_words = (int) (((4 + ((N_IN - 1) * K + 7) / 8) + 3) & ~3) / 4;
        const int lane_bit = 4 * lane * K + 18, w_lane = lane_bit >> 5, b_lane = lane_bit & 31;
        // gate, up: payload z; down: the third
        const size_t off = GU ? (blockIdx.z ? gu_bytes<K>() : 0) : 2 * gu_bytes<K>();
        const uint32_t* g32 = reinterpret_cast<const uint32_t*>(b + off) + (size_t) r * stride_words;
        uint32_t wa[STEPS], wb[STEPS];
        #pragma unroll
        for (int st = 0; st < STEPS; ++st) {
            const int wi = w_lane + 4 * K * st;
            wa[st] = __ldg(g32 + wi);
            wb[st] = wi + 1 < stride_words ? __ldg(g32 + wi + 1) : 0u;
        }
        const uint32_t hdr = __ldg(g32);
        const float scale = __half2float(__ushort_as_half((unsigned short) (hdr & 0xFFFFu))) * rs;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int c0 = e0; c0 < e1; c0 += kNe) {
            switch (min(kNe, e1 - c0)) {   // warp-uniform
            case 1: row_entries<K, GU, STEPS, 1>(wa, wb, hdr, scale, xr, ent_dst, out, c0, r, n_out, lane, w_lane, b_lane); break;
            case 2: row_entries<K, GU, STEPS, 2>(wa, wb, hdr, scale, xr, ent_dst, out, c0, r, n_out, lane, w_lane, b_lane); break;
            case 3: row_entries<K, GU, STEPS, 3>(wa, wb, hdr, scale, xr, ent_dst, out, c0, r, n_out, lane, w_lane, b_lane); break;
            default: row_entries<K, GU, STEPS, 4>(wa, wb, hdr, scale, xr, ent_dst, out, c0, r, n_out, lane, w_lane, b_lane); break;
            }
        }
    });
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
    // entry slots a group: a decode window has one entry an expert, a verify window a few
    const dim3 grot(2, gy);
    ms_rotate_in_kernel<K><<<dim3(2, gy, 2), 160, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok,
                                               static_cast<const mc::MsQ81*>(x_q8_1), xr, NF);
    check("ms_expert_grouped/rotate");
    ms_rows_kernel<K, true, NE / 128><<<dim3(NF / kWarps, gy, 2), kThreads, 0, s>>>(
        grp_ptr, grp_start, n_groups, nullptr, xr, gate, up, NF);
    check("ms_expert_grouped/gu");
    ms_swiglu_rot_kernel<K><<<grot, 160, 0, s>>>(grp_ptr, grp_start, n_groups, gate, up, h, limit);
    check("ms_expert_grouped/swiglu");
    ms_rows_kernel<K, false, NF / 128><<<dim3(NE / kWarps, gy, 1), kThreads, 0, s>>>(
        grp_ptr, grp_start, n_groups, ent_dst, h, out, out, NE);
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
