#pragma once

// CUDA 1MAD bitshift-trellis GEMV and block FWHT kernels.
// Rows are staged through shared memory as a big-endian bit stream and decoded with
// a funnel shift plus dp4a.u32.u32.

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

// Grid-stride warp-per-row GEMV. x_rot is staged once per block in shared memory; each warp
// prefetches its next row into registers while it decodes the current row from shared memory.
// The staged row is a big-endian bit stream: word 0 holds s0 in its low 14 bits, so the state
// of column c is the 14 bits at stream bit c*K + 18. Each lane decodes 4 adjacent columns from
// one 32-bit window (3*K + 14 <= 32); its bit offset inside a word is constant and the word index
// advances by 4*K per step.
// Rows must be 4-byte aligned, at most kMaxRowWords words, and in_features a multiple of 4
// (checked by the launcher).
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
    __shared__ uint32_t smem_rows[WARPS_PER_BLOCK][kMaxRowWords + 1];  // +1: window read past the last word
    static_assert(K >= 1 && 3 * K + 14 <= 32, "4 states must fit one 32-bit window");
    extern __shared__ float4 s_x4[];
    float* s_x = reinterpret_cast<float*>(s_x4);

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

    const int lane_bit = 4 * lane * K + 18;
    const int bit_off = lane_bit & 31;

    for (; row < out_features; row += warp_stride) {
        const uint32_t hdr = __shfl_sync(0xFFFFFFFFu, buf[0], 0);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr & 0xFFFFu)));
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < kWordsPerLane; ++j) {
            const int w = lane + 32 * j;
            if (w < stride_words) s_row[w] = (w == 0) ? (hdr >> 16) & kStateMask : __byte_perm(buf[j], 0, 0x0123);
        }
        __syncwarp();
        if (row + warp_stride < out_features) fetch(row + warp_stride);

        float acc = 0.0f;
        int w = lane_bit >> 5;
        #pragma unroll 2
        for (int c4 = lane; 4 * c4 < in_features; c4 += 32, w += 4 * K) {
            const uint32_t v = __funnelshift_l(s_row[w + 1], s_row[w], bit_off);
            const float4 x = s_x4[c4];
            acc = fmaf(decode_1mad_dp4a((v >> 18) & kStateMask), x.x, acc);
            acc = fmaf(decode_1mad_dp4a((v >> (18 - K)) & kStateMask), x.y, acc);
            acc = fmaf(decode_1mad_dp4a((v >> (18 - 2 * K)) & kStateMask), x.z, acc);
            acc = fmaf(decode_1mad_dp4a((v >> (18 - 3 * K)) & kStateMask), x.w, acc);
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
    if ((row_bytes & 3u) != 0u || row_bytes > kMaxRowWords * 4u || (in_features & 3) != 0) {
        return cudaErrorInvalidValue;
    }
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
