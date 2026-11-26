// src/models/inkling2/kernels/qsa.cu - the QSA indexer (qsa_kernels.hpp): its projections, block keys, scores and the
// top-k block selection. The selection is deterministic: ties at the threshold go to the lowest block indices.
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/models/inkling2/qsa_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 qsa ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// sum over a 128-thread block
__device__ __forceinline__ float block128_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    const float s = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    return s;
}

constexpr int kProjWarps = 8;

// A block per (8 outputs, row): the row normed (or decoded from q8_1) into shared memory, a warp an output.
// shortcut: weights re-read per row (from L2); a tiled product would serve long prompt chunks better.
__global__ void __launch_bounds__(kProjWarps * 32) qsa_project_kernel(const float* __restrict__ x, const uint8_t* __restrict__ x8,
                                                                      const float* __restrict__ norm, float eps,
                                                                      const __half* __restrict__ w, int n_out,
                                                                      float* __restrict__ out) {
    __shared__ float xs[kWidth];
    __shared__ float red[kProjWarps];
    const int r = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (x8) {
        const uint8_t* row = x8 + (size_t) r * (kWidth / 32) * 36;
        for (int i = tid; i < kWidth; i += blockDim.x) {
            const uint8_t* b = row + (size_t) (i / 32) * 36;
            const float d = __half2float(*reinterpret_cast<const __half*>(b));
            xs[i] = d * (float) reinterpret_cast<const int8_t*>(b + 4)[i % 32];
        }
        __syncthreads();
    } else {
        const float* xr = x + (size_t) r * kWidth;
        float ss = 0.f;
        for (int i = tid; i < kWidth; i += blockDim.x) {
            const float v = xr[i];
            xs[i] = v;
            ss = fmaf(v, v, ss);
        }
        ss = warp_sum(ss);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = 0.f;
        for (int i = 0; i < kProjWarps; ++i) tot += red[i];
        const float inv = rsqrtf(tot / (float) kWidth + eps);
        for (int i = tid; i < kWidth; i += blockDim.x) xs[i] = xs[i] * inv * norm[i];
        __syncthreads();
    }
    const int n = blockIdx.x * kProjWarps + warp;
    if (n >= n_out) return;
    const __half2* wr = reinterpret_cast<const __half2*>(w + (size_t) n * kWidth);
    float acc = 0.f;
    for (int i = lane; i < kWidth / 2; i += 32) {
        const float2 f = __half22float2(wr[i]);
        acc = fmaf(f.x, xs[2 * i], acc);
        acc = fmaf(f.y, xs[2 * i + 1], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) out[(size_t) r * n_out + n] = acc;
}

// A block per row at a block's last position: the block's key from its four k projections.
__global__ void __launch_bounds__(kQsaDim) qsa_emit_kernel(const float* __restrict__ raw, const int32_t* __restrict__ pos,
                                                           const float* __restrict__ k_norm, const float* __restrict__ ring,
                                                           __half* __restrict__ blocks) {
    __shared__ float red[4];
    const int p0 = pos[0];
    const int p = p0 + blockIdx.x;
    if ((p & 3) != 3) return;
    const int d = threadIdx.x;
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < kQsaBlock; ++i) {
        const int q = p - 3 + i;
        s += q >= p0 ? raw[(size_t) (q - p0) * kQsaDim + d] : ring[(size_t) (q % kQsaRing) * kQsaDim + d];
    }
    s *= 0.25f;
    const float inv = rsqrtf(block128_sum(s * s, red) / (float) kQsaDim + 1e-6f);
    blocks[(size_t) (p >> 2) * kQsaDim + d] = __float2half(s * inv * k_norm[d]);
}

__global__ void __launch_bounds__(kQsaDim) qsa_ring_kernel(const float* __restrict__ raw, const int32_t* __restrict__ pos,
                                                           int first, float* __restrict__ ring) {
    const int r = first + blockIdx.x;
    ring[(size_t) ((pos[0] + r) % kQsaRing) * kQsaDim + threadIdx.x] = raw[(size_t) r * kQsaDim + threadIdx.x];
}

// A block per (row, head): q_h = rms(q_h) * q_norm.
__global__ void __launch_bounds__(kQsaDim) qsa_qnorm_kernel(float* __restrict__ q, const float* __restrict__ q_norm) {
    __shared__ float red[4];
    float* v = q + ((size_t) blockIdx.y * kQsaHeads + blockIdx.x) * kQsaDim;
    const float x = v[threadIdx.x];
    const float inv = rsqrtf(block128_sum(x * x, red) / (float) kQsaDim + 1e-6f);
    v[threadIdx.x] = x * inv * q_norm[threadIdx.x];
}

constexpr float kKeep = 1e6f;   // the sink's and the local window's score: above any indexer score

// A warp per (block, row): the row's score of the block (kKeep for the sink and the local window).
__global__ void __launch_bounds__(128) qsa_score_kernel(const float* __restrict__ q, const __half* __restrict__ blocks,
                                                        const float* __restrict__ w_head, const int32_t* __restrict__ pos,
                                                        int sink, int local, int64_t stride, float* __restrict__ scores) {
    const int r = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int b = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int nb = (pos[r] + 1) >> 2;
    if (b >= nb) return;
    float sc;
    if (b < sink || b >= nb - local) {
        sc = kKeep;
    } else {
        const __half2* kp = reinterpret_cast<const __half2*>(blocks + (size_t) b * kQsaDim) + lane * 2;
        const float2 k0 = __half22float2(kp[0]), k1 = __half22float2(kp[1]);
        const float* qr = q + (size_t) r * kQsaHeads * kQsaDim + lane * 4;
        sc = 0.f;
#pragma unroll
        for (int h = 0; h < kQsaHeads; ++h) {
            const float4 qv = *reinterpret_cast<const float4*>(qr + h * kQsaDim);
            const float dot = warp_sum(qv.x * k0.x + qv.y * k0.y + qv.z * k1.x + qv.w * k1.y);
            sc = fmaf(w_head[h], fmaxf(dot, 0.f), sc);
        }
        sc *= 0.08838834764831845f;   // 1 / sqrt(128)
    }
    if (lane == 0) scores[(size_t) r * stride + b] = sc;
}

constexpr int kSelThreads = 256;

// The non-negative score's bits: their order is the scores' order.
__device__ __forceinline__ uint32_t key_of(float v) { return __float_as_uint(fmaxf(v, 0.f)); }

// Exclusive prefix count of `flag` over the block's threads, and the block's total.
__device__ __forceinline__ int block_scan(bool flag, int* warp_n, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned ballot = __ballot_sync(0xffffffffu, flag);
    if (lane == 0) warp_n[warp] = __popc(ballot);
    __syncthreads();
    int before = __popc(ballot & ((1u << lane) - 1u));
    total = 0;
    for (int i = 0; i < kSelThreads / 32; ++i) {
        if (i < warp) before += warp_n[i];
        total += warp_n[i];
    }
    __syncthreads();
    return before;
}

// A block per row: the k-th largest key by four 8-bit radix passes, then the blocks above it and the first ones equal to
// it in ascending order.
__global__ void __launch_bounds__(kSelThreads) qsa_select_kernel(const float* __restrict__ scores, const int32_t* __restrict__ pos,
                                                                 int k, int64_t stride, int32_t* __restrict__ sel) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_need;
    __shared__ int warp_n[kSelThreads / 32];
    const int r = blockIdx.x, tid = threadIdx.x;
    const int nb = (pos[r] + 1) >> 2;
    int32_t* out = sel + (size_t) r * k;
    if (nb <= k) {
        for (int i = tid; i < k; i += kSelThreads) out[i] = i < nb ? i : -1;
        return;
    }
    const float* sc = scores + (size_t) r * stride;
    uint32_t prefix = 0, need = (uint32_t) k;   // the key's bits found so far, and how many at or above them are wanted
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int b = tid; b < nb; b += kSelThreads) {
            const uint32_t key = key_of(sc[b]);
            if ((key & hi_mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0;
            int bin = 255;
            for (; bin > 0; --bin) {
                if (acc + hist[bin] >= need) break;
                acc += hist[bin];
            }
            s_prefix = prefix | ((uint32_t) bin << shift);
            s_need = need - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
    }
    // prefix: the threshold key; need: the blocks equal to it to take
    int taken = 0, eq_seen = 0;
    for (int b0 = 0; b0 < nb; b0 += kSelThreads) {
        const int b = b0 + tid;
        const uint32_t key = b < nb ? key_of(sc[b]) : 0u;
        const bool eq = b < nb && key == prefix;
        int eq_total = 0;
        const int eq_rank = eq_seen + block_scan(eq, warp_n, eq_total);
        const bool keep = b < nb && (key > prefix || (eq && eq_rank < (int) need));
        int kept = 0;
        const int at = taken + block_scan(keep, warp_n, kept);
        if (keep) out[at] = b;
        taken += kept;
        eq_seen += eq_total;
    }
}

}  // namespace

void qsa_project(const float* x, const void* x8, const float* norm, float eps, const uint16_t* w, int n_out, int rows,
                 float* out, void* stream) {
    if (rows <= 0) return;
    const dim3 grid((unsigned) ((n_out + kProjWarps - 1) / kProjWarps), (unsigned) rows);
    qsa_project_kernel<<<grid, kProjWarps * 32, 0, (cudaStream_t) stream>>>(x, (const uint8_t*) x8, norm, eps,
                                                                            (const __half*) w, n_out, out);
    check("project");
}

void qsa_keys(const float* raw, const int32_t* pos, int rows, const float* k_norm, float* ring, uint16_t* blocks,
              void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    qsa_emit_kernel<<<(unsigned) rows, kQsaDim, 0, s>>>(raw, pos, k_norm, ring, (__half*) blocks);
    const int first = rows > kQsaRing ? rows - kQsaRing : 0;
    qsa_ring_kernel<<<(unsigned) (rows - first), kQsaDim, 0, s>>>(raw, pos, first, ring);
    check("keys");
}

void qsa_select(float* q, const float* q_norm, const float* w_head, const uint16_t* blocks, const int32_t* pos, int rows,
                int sink_blocks, int local_blocks, int k_blocks, int64_t max_blocks, float* scores, int32_t* sel,
                void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    qsa_qnorm_kernel<<<dim3(kQsaHeads, (unsigned) rows), kQsaDim, 0, s>>>(q, q_norm);
    if (max_blocks > 0)
        qsa_score_kernel<<<dim3((unsigned) ((max_blocks + 3) / 4), (unsigned) rows), 128, 0, s>>>(
            q, (const __half*) blocks, w_head, pos, sink_blocks, local_blocks, max_blocks, scores);
    qsa_select_kernel<<<(unsigned) rows, kSelThreads, 0, s>>>(scores, pos, k_blocks, max_blocks, sel);
    check("select");
}

}  // namespace basalt::models::inkling2
