#pragma once

// CUDA 1MAD bitshift-trellis GEMV and block FWHT kernels.
// Rows are staged through shared memory and decoded with dp4a.u32.u32; reads of unaligned
// 32-bit windows use two word loads plus a funnel shift.

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

// Grid-stride warp-per-row GEMV. x_rot is staged once per block in shared memory; each warp
// prefetches its next row into registers while it decodes the current row from shared memory.
// Rows must be 4-byte aligned and at most kMaxRowWords words (checked by the launcher).
constexpr int kMaxRowWords = 512;
constexpr int kWordsPerLane = kMaxRowWords / 32;

template <int K, int WARPS_PER_BLOCK>
__global__ void ms_1mad_gemv_fast_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x_rot,
    float* __restrict__ out,
    int out_features,
    int in_features
) {
    __shared__ uint32_t smem_rows[WARPS_PER_BLOCK][kMaxRowWords];
    extern __shared__ float s_x[];

    for (int i = threadIdx.x; i < in_features; i += blockDim.x) s_x[i] = __ldg(x_rot + i);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_stride = gridDim.x * WARPS_PER_BLOCK;
    int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= out_features) return;

    const int stride_words = static_cast<int>(ms_row_bytes(in_features, K) >> 2);
    const uint32_t* g_u32 = reinterpret_cast<const uint32_t*>(weights);
    uint32_t* s_row = smem_rows[threadIdx.x >> 5];

    uint32_t buf[kWordsPerLane];
    auto fetch = [&](int r) {
        const uint32_t* g = g_u32 + static_cast<size_t>(r) * stride_words;
        #pragma unroll
        for (int j = 0; j < kWordsPerLane; ++j) {
            const int w = lane + 32 * j;
            if (w < stride_words) buf[j] = __ldg(g + w);
        }
    };
    fetch(row);

    for (; row < out_features; row += warp_stride) {
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < kWordsPerLane; ++j) {
            const int w = lane + 32 * j;
            if (w < stride_words) s_row[w] = buf[j];
        }
        __syncwarp();
        if (row + warp_stride < out_features) fetch(row + warp_stride);

        const uint32_t hdr = s_row[0];
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr & 0xFFFFu)));
        const uint16_t s0 = static_cast<uint16_t>(hdr >> 16);
        const uint32_t* s_qs_u32 = s_row + 1;

        float acc = 0.0f;
        #pragma unroll 4
        for (int c = lane; c < in_features; c += 32) {
            acc = fmaf(decode_1mad_dp4a(decode_state_smem_u32<K>(s0, s_qs_u32, c)), s_x[c], acc);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
        }
        if (lane == 0) out[row] = acc * d;
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
inline cudaError_t launch_ms_1mad_gemv(
    const uint8_t* weights,
    const float* x_rot,
    float* out,
    int out_features,
    int in_features,
    cudaStream_t stream = nullptr
) {
    constexpr int kWarpsPerBlock = 8;
    constexpr int kThreads = kWarpsPerBlock * 32;
    const size_t row_bytes = ms_row_bytes(in_features, K);
    if ((row_bytes & 3u) != 0u || row_bytes > kMaxRowWords * 4u) return cudaErrorInvalidValue;
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const int max_blocks = sms * 4;
    const int needed = (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int blocks = needed < max_blocks ? needed : max_blocks;
    ms_1mad_gemv_fast_kernel<K, kWarpsPerBlock><<<blocks, kThreads, in_features * sizeof(float), stream>>>(
        weights, x_rot, out, out_features, in_features
    );
    return cudaGetLastError();
}

} // namespace adamant::kernels::ms_cuda
