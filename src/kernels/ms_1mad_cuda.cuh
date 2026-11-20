#pragma once

// High-bandwidth Ampere (sm_86) CUDA 1MAD bitshift-trellis GEMV & block FWHT kernels.
// Key optimization: 100% coalesced 128-byte warp loads (uint32_t per lane) of the packed
// row bitstream into per-warp shared memory, eliminating uncoalesced byte-by-byte global reads.

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

template <int K>
__device__ __forceinline__ uint32_t decode_state_from_smem(
    uint16_t s0,
    const uint8_t* __restrict__ s_qs,
    int col
) {
    if (col == 0) {
        return static_cast<uint32_t>(s0) & kStateMask;
    }
    const int start_bit = col * K;
    if (start_bit < 14) {
        uint32_t hi = (static_cast<uint32_t>(s0) & kStateMask) << start_bit;
        uint32_t lo_bytes = (static_cast<uint32_t>(s_qs[0]) << 16) |
                            (static_cast<uint32_t>(s_qs[1]) << 8)  |
                             static_cast<uint32_t>(s_qs[2]);
        uint32_t lo = lo_bytes >> (24 - start_bit);
        return (hi | lo) & kStateMask;
    }
    const int qs_bit = start_bit - 14;
    const int byte_idx = qs_bit >> 3;
    const int bit_off = qs_bit & 7;
    uint32_t raw = (static_cast<uint32_t>(s_qs[byte_idx]) << 16) |
                   (static_cast<uint32_t>(s_qs[byte_idx + 1]) << 8) |
                    static_cast<uint32_t>(s_qs[byte_idx + 2]);
    return (raw >> (24 - bit_off - 14)) & kStateMask;
}

__device__ __forceinline__ float decode_1mad_val(uint32_t state) {
    uint32_t x = state * kMadMul + kMadAdd;
    // Fast 4-byte unsigned sum using __vabsdiffu4(x, 0) / shifts
    uint32_t sum = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24);
    return static_cast<float>(static_cast<int32_t>(sum) - 510);
}

// Coalesced warp-per-row GEMV:
// Each warp stages its entire packed row into shared memory using 100% coalesced
// 32-bit/128-byte transactions, then decodes all columns from shared memory at L1 speed.
template <int K, int WARPS_PER_BLOCK = 4>
__global__ void ms_1mad_gemv_coalesced_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x_rot,
    float* __restrict__ out,
    int out_features,
    int in_features
) {
    // Max row bytes for in_features <= 2560, K <= 6 is 4 + 1920 = 1924 bytes (484 uint32_t words).
    constexpr int kMaxRowWords = 512;
    __shared__ uint32_t smem_rows[WARPS_PER_BLOCK][kMaxRowWords];

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    if (row_idx >= out_features) return;

    const size_t stride = ms_row_bytes(in_features, K);
    const uint8_t* g_row = weights + static_cast<size_t>(row_idx) * stride;
    uint32_t* s_row_u32 = smem_rows[warp_in_blk];

    // 100% coalesced 4-byte loads across the 32 warp lanes (128 bytes per warp step)
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

    const uint8_t* s_row = reinterpret_cast<const uint8_t*>(s_row_u32);
    const __half d_h = *reinterpret_cast<const __half*>(s_row);
    const float d = __half2float(d_h);
    const uint16_t s0 = *reinterpret_cast<const uint16_t*>(s_row + 2);
    const uint8_t* s_qs = s_row + 4;

    float acc = 0.0f;
    #pragma unroll 4
    for (int c = lane; c < in_features; c += 32) {
        uint32_t state = decode_state_from_smem<K>(s0, s_qs, c);
        acc = fmaf(decode_1mad_val(state), __ldg(x_rot + c), acc);
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
    ms_1mad_gemv_coalesced_kernel<K, kWarpsPerBlock><<<blocks, kThreads, 0, stream>>>(
        weights, x_rot, out, out_features, in_features
    );
}

} // namespace adamant::kernels::ms_cuda
