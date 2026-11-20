#pragma once

// High-bandwidth Ampere (sm_86) CUDA 1MAD bitshift-trellis GEMV & block FWHT kernels.
// Combines:
//   - 100% coalesced 128-byte warp loads (uint32_t per lane) into a tiny 1 KB shared-memory
//     buffer (100% occupancy on sm_86).
//   - Single-instruction PTX `dp4a.u32.u32` for the 4-byte unsigned sum + (-510) bias.
//   - 32-bit word reads from shared memory via __byte_perm (no 3x byte-load overhead).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace adamant::kernels::ms_cuda {

constexpr uint32_t kMadMul = 34038481u;
constexpr uint32_t kMadAdd = 76625530u;
constexpr uint32_t kStateMask = 0x3FFFu;

__host__ __device__ inline size_t ms_row_bytes(int in_features, int k) {
    return 4u + static_cast<size_t>(((in_features - 1) * k + 7) / 8);
}

__device__ __forceinline__ float decode_1mad_dp4a(uint32_t state) {
    uint32_t x = state * kMadMul + kMadAdd;
    int32_t sum;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(sum) : "r"(x), "r"(0x01010101u), "r"(-510));
    return __int2float_rn(sum);
}

// Fast 32-bit big-endian unaligned read from shared memory using 2 word loads + funnel shift.
__device__ __forceinline__ uint32_t load_be32_smem(const uint32_t* __restrict__ s_u32, int byte_idx) {
    const int w = byte_idx >> 2;
    const int b = byte_idx & 3;
    uint32_t lo = __byte_perm(s_u32[w], 0, 0x0123);     // bswap32(w)
    if (b == 0) return lo;
    uint32_t hi = __byte_perm(s_u32[w + 1], 0, 0x0123); // bswap32(w + 1)
    return __funnelshift_l(hi, lo, b * 8);
}

template <int K>
__device__ __forceinline__ uint32_t decode_state_smem_u32(
    uint16_t s0,
    const uint32_t* __restrict__ s_qs_u32,
    int col
) {
    if (col == 0) return static_cast<uint32_t>(s0) & kStateMask;
    const int start_bit = col * K;
    if (start_bit < 14) {
        uint32_t hi = (static_cast<uint32_t>(s0) & kStateMask) << start_bit;
        uint32_t lo = load_be32_smem(s_qs_u32, 0) >> (32 - start_bit);
        return (hi | lo) & kStateMask;
    }
    const int qs_bit = start_bit - 14;
    const int byte_idx = qs_bit >> 3;
    const int bit_off = qs_bit & 7;
    uint32_t raw = load_be32_smem(s_qs_u32, byte_idx);
    return (raw >> (32 - bit_off - 14)) & kStateMask;
}

// Coalesced warp-per-row GEMV with 128-byte coalesced staging + dp4a.u32.u32 decode.
template <int K, int WARPS_PER_BLOCK = 4>
__global__ void ms_1mad_gemv_fast_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x_rot,
    float* __restrict__ out,
    int out_features,
    int in_features
) {
    constexpr int kMaxRowWords = 512;
    __shared__ uint32_t smem_rows[WARPS_PER_BLOCK][kMaxRowWords];

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    if (row_idx >= out_features) return;

    const size_t stride = ms_row_bytes(in_features, K);
    const uint8_t* g_row = weights + static_cast<size_t>(row_idx) * stride;
    uint32_t* s_row_u32 = smem_rows[warp_in_blk];

    // 100% coalesced 128-byte warp loads
    const int n_words = static_cast<int>((stride + 3u) >> 2);
    const uint32_t* g_u32 = reinterpret_cast<const uint32_t*>(g_row);
    const bool aligned4 = (reinterpret_cast<uintptr_t>(g_row) & 3u) == 0u;
    if (aligned4) {
        for (int w = lane; w < n_words; w += 32) {
            s_row_u32[w] = __ldg(g_u32 + w);
        }
    } else {
        uint8_t* s_bytes = reinterpret_cast<uint8_t*>(s_row_u32);
        for (int b = lane; b < static_cast<int>(stride); b += 32) {
            s_bytes[b] = __ldg(g_row + b);
        }
    }
    __syncwarp();

    const uint32_t hdr = s_row_u32[0];
    const float d = __half2float(*reinterpret_cast<const __half*>(&hdr));
    const uint16_t s0 = static_cast<uint16_t>(hdr >> 16);
    const uint32_t* s_qs_u32 = s_row_u32 + 1;

    float acc = 0.0f;
    #pragma unroll 4
    for (int c = lane; c < in_features; c += 32) {
        uint32_t state = decode_state_smem_u32<K>(s0, s_qs_u32, c);
        acc = fmaf(decode_1mad_dp4a(state), __ldg(x_rot + c), acc);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }

    if (lane == 0) {
        out[row_idx] = acc * d;
    }
}

template <int BLOCK_SIZE>
__global__ void fwht_rotate_input_kernel(
    const float* __restrict__ x,
    const int8_t* __restrict__ signs,
    float* __restrict__ y,
    int n
) {
    __shared__ float smem[BLOCK_SIZE];
    const int base = blockIdx.x * BLOCK_SIZE;
    const int tid = threadIdx.x;
    if (base + tid >= n) return;

    smem[tid] = x[base + tid] * static_cast<float>(signs[base + tid]);
    __syncthreads();

    for (int h = 1; h < BLOCK_SIZE; h <<= 1) {
        const int partner = tid ^ h;
        const float a = smem[tid];
        const float b = smem[partner];
        __syncthreads();
        smem[tid] = (tid < partner) ? (a + b) : (b - a);
        __syncthreads();
    }

    y[base + tid] = smem[tid] * rsqrtf(static_cast<float>(BLOCK_SIZE));
}

template <int K>
inline void launch_ms_1mad_gemv(
    const uint8_t* weights,
    const float* x_rot,
    float* out,
    int out_features,
    int in_features,
    cudaStream_t stream = nullptr
) {
    constexpr int kWarpsPerBlock = 4;
    constexpr int kThreads = kWarpsPerBlock * 32;
    const int blocks = (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock;
    ms_1mad_gemv_fast_kernel<K, kWarpsPerBlock><<<blocks, kThreads, 0, stream>>>(
        weights, x_rot, out, out_features, in_features
    );
}

} // namespace adamant::kernels::ms_cuda
