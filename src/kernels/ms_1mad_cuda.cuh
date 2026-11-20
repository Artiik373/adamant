#pragma once

// Ampere (sm_86) CUDA 1MAD bitshift-trellis GEMV & block FWHT kernels for Adamant.
// Matches tools/convert/ms_moe_converter.py & src/kernels/ms_1mad_avx2.h:
//   Row layout (little-endian):
//     __half   d        (2 bytes)
//     uint16_t s0       (2 bytes, 14-bit initial trellis state)
//     uint8_t  qs[...]  ceil((in_features - 1) * K / 8) bytes, MSB-first K-bit symbols

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

// Compute the 14-bit state at column `col` directly from the packed row bitstream.
// Because a bitshift trellis state is simply the 14-bit window ending at column `col`,
// any thread in a warp can decode any column independently with zero serial dependency.
template <int K>
__device__ __forceinline__ uint32_t decode_state_at(
    uint16_t s0,
    const uint8_t* __restrict__ qs,
    int col
) {
    if (col == 0) {
        return static_cast<uint32_t>(s0) & kStateMask;
    }

    // Total bit offset in the virtual stream [s0 (14 bits)][qs (MSB-first)]:
    // weight `col` ends at bit (14 + col * K), so its 14-bit window starts at bit (col * K).
    const int start_bit = col * K;
    if (start_bit < 14) {
        // Window spans both s0 and the first few symbols of qs
        uint32_t hi = (static_cast<uint32_t>(s0) & kStateMask) << start_bit;
        uint32_t lo_bytes = (static_cast<uint32_t>(qs[0]) << 16) |
                            (static_cast<uint32_t>(qs[1]) << 8)  |
                             static_cast<uint32_t>(qs[2]);
        uint32_t lo = lo_bytes >> (24 - start_bit);
        return (hi | lo) & kStateMask;
    }

    const int qs_bit = start_bit - 14;
    const int byte_idx = qs_bit >> 3;
    const int bit_off = qs_bit & 7;
    uint32_t raw = (static_cast<uint32_t>(qs[byte_idx]) << 16) |
                   (static_cast<uint32_t>(qs[byte_idx + 1]) << 8) |
                    static_cast<uint32_t>(qs[byte_idx + 2]);
    return (raw >> (24 - bit_off - 14)) & kStateMask;
}

__device__ __forceinline__ float decode_1mad_val(uint32_t state) {
    uint32_t x = state * kMadMul + kMadAdd;
    uint32_t sum = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24);
    return static_cast<float>(static_cast<int32_t>(sum) - 510);
}

// One warp (32 threads) per output row: embarrassingly parallel register-only 1MAD GEMV.
template <int K>
__global__ void ms_1mad_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x_rot,
    float* __restrict__ out,
    int out_features,
    int in_features
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= out_features) return;

    const size_t stride = ms_row_bytes(in_features, K);
    const uint8_t* row = weights + static_cast<size_t>(warp_id) * stride;

    const __half d_h = *reinterpret_cast<const __half*>(row);
    const float d = __half2float(d_h);
    const uint16_t s0 = *reinterpret_cast<const uint16_t*>(row + 2);
    const uint8_t* qs = row + 4;

    float acc = 0.0f;
    for (int c = lane; c < in_features; c += 32) {
        uint32_t state = decode_state_at<K>(s0, qs, c);
        acc += decode_1mad_val(state) * x_rot[c];
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }

    if (lane == 0) {
        out[warp_id] = acc * d;
    }
}

// Block FWHT + sign flip in shared memory: supports BLOCK_SIZE = 128 or 512.
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
    ms_1mad_gemv_kernel<K><<<blocks, kThreads, 0, stream>>>(
        weights, x_rot, out, out_features, in_features
    );
}

} // namespace adamant::kernels::ms_cuda
