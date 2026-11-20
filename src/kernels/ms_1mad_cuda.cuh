#pragma once

// Peak-throughput Ampere (sm_86) CUDA 1MAD bitshift-trellis GEMV & block FWHT kernels.
// Key sm_86 optimizations:
//   - Single-instruction PTX `dp4a.u32.u32` for the 4-byte unsigned sum + (-510) bias
//     (replaces 7 INT32 shift/mask/add instructions on sm_86's half-rate INT32 pipe).
//   - Contiguous per-thread column blocks (in_features / 32) decoded from register-cached
//     uint32_t words (zero shared memory, zero bank conflicts, 100% occupancy).

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

// Single-instruction unsigned 4-byte sum minus 510 using Ampere native dp4a.u32.u32.
__device__ __forceinline__ float decode_1mad_dp4a(uint32_t state) {
    uint32_t x = state * kMadMul + kMadAdd;
    int32_t sum;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(sum) : "r"(x), "r"(0x01010101u), "r"(-510));
    return __int2float_rn(sum);
}

__device__ __forceinline__ uint32_t load_be24(const uint8_t* __restrict__ p) {
    return (static_cast<uint32_t>(__ldg(p)) << 16) |
           (static_cast<uint32_t>(__ldg(p + 1)) << 8) |
            static_cast<uint32_t>(__ldg(p + 2));
}

template <int K>
__device__ __forceinline__ uint32_t decode_state_at(
    uint16_t s0,
    const uint8_t* __restrict__ qs,
    int col
) {
    if (col == 0) return static_cast<uint32_t>(s0) & kStateMask;
    const int start_bit = col * K;
    if (start_bit < 14) {
        uint32_t hi = (static_cast<uint32_t>(s0) & kStateMask) << start_bit;
        uint32_t lo = load_be24(qs) >> (24 - start_bit);
        return (hi | lo) & kStateMask;
    }
    const int qs_bit = start_bit - 14;
    const int byte_idx = qs_bit >> 3;
    const int bit_off = qs_bit & 7;
    return (load_be24(qs + byte_idx) >> (24 - bit_off - 14)) & kStateMask;
}

// High-occupancy register-only warp-per-row GEMV:
// Each lane processes a contiguous slice of columns, keeping a 64-bit bit-buffer in registers
// and shifting out K-bit symbols with minimal memory traffic + 1-cycle dp4a.u32.u32 decode.
template <int K, int WARPS_PER_BLOCK = 8>
__global__ void ms_1mad_gemv_dp4a_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x_rot,
    float* __restrict__ out,
    int out_features,
    int in_features
) {
    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    if (row_idx >= out_features) return;

    const size_t stride = ms_row_bytes(in_features, K);
    const uint8_t* row = weights + static_cast<size_t>(row_idx) * stride;

    const float d = __half2float(__ldg(reinterpret_cast<const __half*>(row)));
    const uint16_t s0 = __ldg(reinterpret_cast<const uint16_t*>(row + 2));
    const uint8_t* qs = row + 4;

    // Contiguous slice per lane: e.g. for in_features=2560, each lane does 80 contiguous columns
    const int cols_per_lane = (in_features + 31) >> 5;
    const int c_start = lane * cols_per_lane;
    const int c_end = min(c_start + cols_per_lane, in_features);

    float acc = 0.0f;
    if (c_start < c_end) {
        // Initialize 14-bit state at c_start once
        uint32_t state = decode_state_at<K>(s0, qs, c_start);
        acc = fmaf(decode_1mad_dp4a(state), __ldg(x_rot + c_start), acc);

        // Stream remaining symbols in this lane's contiguous slice from a register bit-buffer
        constexpr uint32_t kSymMask = (1u << K) - 1u;
        int c = c_start + 1;
        int bit_pos = c_start * K; // symbol index (c - 1) starts at bit (c - 1) * K
        int byte_idx = bit_pos >> 3;
        int bit_off = bit_pos & 7;

        // Preload 32 bits big-endian into a 64-bit shift register
        uint64_t bitbuf = 0;
        int bits_avail = 0;
        auto refill = [&]() {
            while (bits_avail <= 32 && byte_idx < static_cast<int>(stride - 4)) {
                bitbuf = (bitbuf << 8) | static_cast<uint64_t>(__ldg(qs + byte_idx));
                ++byte_idx;
                bits_avail += 8;
            }
        };

        refill();
        if (bit_off > 0) {
            bits_avail -= bit_off;
        }

        for (; c < c_end; ++c) {
            if (bits_avail < K) refill();
            bits_avail -= K;
            uint32_t sym = static_cast<uint32_t>(bitbuf >> bits_avail) & kSymMask;
            state = ((state << K) | sym) & kStateMask;
            acc = fmaf(decode_1mad_dp4a(state), __ldg(x_rot + c), acc);
        }
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
    constexpr int kWarpsPerBlock = 8;
    constexpr int kThreads = kWarpsPerBlock * 32;
    const int blocks = (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock;
    ms_1mad_gemv_dp4a_kernel<K, kWarpsPerBlock><<<blocks, kThreads, 0, stream>>>(
        weights, x_rot, out, out_features, in_features
    );
}

} // namespace adamant::kernels::ms_cuda
