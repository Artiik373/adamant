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

// Same stride as the CPU header: padded to 4 bytes so every row start is word-aligned.
__host__ __device__ inline size_t ms_row_bytes(int in_features, int k) {
    return (4u + static_cast<size_t>(((in_features - 1) * k + 7) / 8) + 3u) & ~size_t{3};
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

// One launch covers up to kMaxGroup matrices of the same shape and K (blockIdx.y picks one),
// e.g. the active experts of one MoE projection. Passed by value as a kernel parameter.
// x is either the rotated input (float, signs unused) or, with kQ8In, ggml Q8_1 blocks of the
// unrotated input plus its int8 signs; each block then applies signs and the block Hadamard itself.
constexpr int kMaxGroup = 32;
struct MsGemvGroup {
    const uint8_t* weights[kMaxGroup];
    const void* x[kMaxGroup];
    const int8_t* signs[kMaxGroup];
    float* out[kMaxGroup];
};

struct MsQ81 {  // ggml block_q8_1
    __half d, s;
    int8_t qs[32];
};

// x_rot = H(signs * dequant(x_q8)) in shared memory, H the block Hadamard without its 1/sqrt(had_block).
// Blocks of 32..512 values: one warp per Hadamard block, lane l holding values l + 32*i (i < had_block/32);
// butterflies of distance >= 32 inside a lane in registers, shorter ones across lanes by shuffles (the stages
// commute). Smaller blocks use a shared-memory loop.
__device__ __forceinline__ void ms_stage_q8_rotated(
    float* s_x, const MsQ81* __restrict__ xq, const int8_t* __restrict__ signs, int n, int had_block
) {
    auto load = [&](int i) {
        const MsQ81& b = xq[i >> 5];
        return __half2float(b.d) * static_cast<float>(b.qs[i & 31]) * static_cast<float>(__ldg(signs + i));
    };
    if (had_block < 32) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) s_x[i] = load(i);
        __syncthreads();
        for (int h = 1, lg = 0; h < had_block; h <<= 1, ++lg) {
            for (int p = threadIdx.x; p < n / 2; p += blockDim.x) {
                const int lo = ((p >> lg) << (lg + 1)) | (p & (h - 1));
                const float a = s_x[lo], b = s_x[lo + h];
                s_x[lo] = a + b;
                s_x[lo + h] = a - b;
            }
            __syncthreads();
        }
        return;
    }
    const int lane = threadIdx.x & 31;
    const int v = had_block >> 5;
    for (int blk = threadIdx.x >> 5; blk * had_block < n; blk += blockDim.x >> 5) {
        const int base = blk * had_block + lane;
        float r[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) if (i < v) r[i] = load(base + 32 * i);
        #pragma unroll
        for (int h = 1; h < 16; h <<= 1) {
            if (h >= v) break;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < v && (i & h) == 0) {
                    const float a = r[i], b = r[i + h];
                    r[i] = a + b;
                    r[i + h] = a - b;
                }
            }
        }
        #pragma unroll
        for (int m = 1; m < 32; m <<= 1) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < v) {
                    const float o = __shfl_xor_sync(0xFFFFFFFFu, r[i], m);
                    r[i] = (lane & m) ? o - r[i] : r[i] + o;
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) if (i < v) s_x[base + 32 * i] = r[i];
    }
    __syncthreads();
}

template <int K, int WARPS_PER_BLOCK, bool kQ8In>
__global__ void ms_1mad_gemv_fast_kernel(const MsGemvGroup group, int out_features, int in_features, int had_block) {
    const uint8_t* __restrict__ weights = group.weights[blockIdx.y];
    float* __restrict__ out = group.out[blockIdx.y];
    __shared__ uint32_t smem_rows[WARPS_PER_BLOCK][kMaxRowWords + 1];  // +1: window read past the last word
    static_assert(K >= 1 && 3 * K + 14 <= 32, "4 states must fit one 32-bit window");
    extern __shared__ float4 s_x4[];
    float* s_x = reinterpret_cast<float*>(s_x4);

    float out_scale = 1.0f;
    if constexpr (kQ8In) {
        ms_stage_q8_rotated(s_x, static_cast<const MsQ81*>(group.x[blockIdx.y]), group.signs[blockIdx.y],
                            in_features, had_block);
        out_scale = rsqrtf(static_cast<float>(had_block));
    } else {
        const float* __restrict__ x_rot = static_cast<const float*>(group.x[blockIdx.y]);
        for (int i = threadIdx.x; i < in_features; i += blockDim.x) s_x[i] = __ldg(x_rot + i);
        __syncthreads();
    }

    const int lane = threadIdx.x & 31;
    const int warp_stride = gridDim.x * WARPS_PER_BLOCK;  // rows of this matrix per pass
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
        if (lane == 0) out[row] = acc * (d * out_scale);
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

// Largest power-of-two divisor of n, at most 512: the Hadamard block of an n-wide input.
__host__ __device__ inline int ms_hadamard_block(int n) {
    int b = 1;
    while (b < 512 && n % (2 * b) == 0) b *= 2;
    return b;
}

// n same-shape GEMVs in launches of up to kMaxGroup. Pointer arrays are host memory.
template <int K, bool kQ8In>
inline cudaError_t launch_ms_group(
    const uint8_t* const* weights,
    const void* const* x,
    const int8_t* const* signs,
    float* const* out,
    int n,
    int out_features,
    int in_features,
    cudaStream_t stream
) {
    constexpr int kWarpsPerBlock = 8;
    constexpr int kThreads = kWarpsPerBlock * 32;
    const size_t row_bytes = ms_row_bytes(in_features, K);
    if (in_features < 4 || (in_features & 3) != 0 || (kQ8In && (in_features & 31) != 0) ||
        row_bytes > kMaxRowWords * 4u || out_features <= 0 || n < 0) {
        return cudaErrorInvalidValue;
    }
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const int needed = (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int had_block = ms_hadamard_block(in_features);
    for (int base = 0; base < n; base += kMaxGroup) {
        const int count = (n - base < kMaxGroup) ? n - base : kMaxGroup;
        MsGemvGroup group{};
        for (int e = 0; e < count; ++e) {
            group.weights[e] = weights[base + e];
            group.x[e] = x[base + e];
            group.signs[e] = signs ? signs[base + e] : nullptr;
            group.out[e] = out[base + e];
        }
        int per_matrix = (sms * 4 + count - 1) / count;
        if (per_matrix > needed) per_matrix = needed;
        ms_1mad_gemv_fast_kernel<K, kWarpsPerBlock, kQ8In>
            <<<dim3(per_matrix, count), kThreads, in_features * sizeof(float), stream>>>(
                group, out_features, in_features, had_block);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

// Runs n same-shape GEMVs on pre-rotated float inputs: out[e] = W[e] * x_rot[e].
template <int K>
inline cudaError_t launch_ms_1mad_gemv_grouped(
    const uint8_t* const* weights,
    const float* const* x_rot,
    float* const* out,
    int n,
    int out_features,
    int in_features,
    cudaStream_t stream = nullptr
) {
    return launch_ms_group<K, false>(weights, reinterpret_cast<const void* const*>(x_rot), nullptr, out, n,
                                     out_features, in_features, stream);
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
    return launch_ms_1mad_gemv_grouped<K>(&weights, &x_rot, &out, 1, out_features, in_features, stream);
}

// One MS tensor payload (rows, then int8 signs[in_features]) times ncols Q8_1 input columns of the
// unrotated input: y + j*out_features = W * x_j. Column j's Q8_1 blocks start at j*in_features/32.
// shortcut: each column re-reads the weights; fold columns into one pass if verify windows need it.
template <int K>
inline cudaError_t launch_ms_1mad_gemv_q8(
    const uint8_t* payload,
    const void* x_q8_1,
    float* y,
    int out_features,
    int in_features,
    int ncols,
    cudaStream_t stream
) {
    if (ncols < 1 || ncols > kMaxGroup || in_features <= 0 || out_features <= 0) return cudaErrorInvalidValue;
    const int8_t* signs = reinterpret_cast<const int8_t*>(
        payload + static_cast<size_t>(out_features) * ms_row_bytes(in_features, K));
    const uint8_t* w[kMaxGroup];
    const void* x[kMaxGroup];
    const int8_t* sg[kMaxGroup];
    float* out[kMaxGroup];
    for (int j = 0; j < ncols; ++j) {
        w[j] = payload;
        x[j] = static_cast<const MsQ81*>(x_q8_1) + static_cast<size_t>(j) * (in_features / 32);
        sg[j] = signs;
        out[j] = y + static_cast<size_t>(j) * out_features;
    }
    return launch_ms_group<K, true>(w, x, sg, out, ncols, out_features, in_features, stream);
}

} // namespace adamant::kernels::ms_cuda
