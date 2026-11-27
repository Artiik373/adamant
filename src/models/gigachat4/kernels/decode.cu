// src/models/gigachat4/kernels/decode.cu - the GigaChat4 decode kernels: embedding rows, the Gated DeltaNet's conv and step,
// per-head Q8_0 products, the absorbed MLA attention over an fp16 cache, the indexer's top-k, the MoE plan and combine.
// Launch sizes depend only on the arguments, never on device values, so a captured graph stays valid.
#include "basalt/kernels/mqa_chunk.cuh"
#include "basalt/models/gigachat4/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <stdexcept>

namespace basalt::models::gigachat4 {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets it. red: 32 floats of shared memory.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    v = l < nw ? red[l] : 0.f;
    return warp_sum(v);
}
__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

cudaStream_t S(void* s) { return (cudaStream_t) s; }
void done(void* s) {
    if (!s) cudaStreamSynchronize(nullptr);
}

__global__ void embed_kernel(const uint16_t* __restrict__ table, const int32_t* __restrict__ ids, int n, float* __restrict__ out) {
    const uint2* r = reinterpret_cast<const uint2*>(table + (size_t) ids[blockIdx.x] * n);
    float4* o = reinterpret_cast<float4*>(out + (size_t) blockIdx.x * n);
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        const uint2 v = r[i];
        o[i] = make_float4(bf16f(v.x & 0xffff), bf16f(v.x >> 16), bf16f(v.y & 0xffff), bf16f(v.y >> 16));
    }
}

__global__ void gdn_conv_kernel(const float* __restrict__ x, int xs, float* __restrict__ state, const float* __restrict__ w, int ch, int rows,
                                float* __restrict__ y) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const float4 wc = reinterpret_cast<const float4*>(w)[c];
    float s0 = state[3 * c], s1 = state[3 * c + 1], s2 = state[3 * c + 2];
    for (int t = 0; t < rows; ++t) {
        const float xi = x[(size_t) t * xs + c];
        const float v = wc.x * s0 + wc.y * s1 + wc.z * s2 + wc.w * xi;
        s0 = s1, s1 = s2, s2 = xi;
        y[(size_t) t * ch + c] = v * sigm(v);
    }
    state[3 * c] = s0, state[3 * c + 1] = s1, state[3 * c + 2] = s2;
}

// Block a value head, 512 threads: column j = tid % 128 of the state, key rows [32 p, 32 p + 32), p = tid / 128.
constexpr int kD = 128, kParts = 4, kRowsPer = kD / kParts;
__global__ void __launch_bounds__(kD * kParts) gdn_step_kernel(float* __restrict__ state, const float* __restrict__ qkv,
                                                              const float* __restrict__ ba, const float* __restrict__ a_log,
                                                              const float* __restrict__ dt_bias, const float* __restrict__ z, int zs,
                                                              const float* __restrict__ norm_w, float eps, GdnShape s, int rows,
                                                              float* __restrict__ y) {
    __shared__ float qs[kD], ks[kD], part[kParts][kD], red[32], gb[2];
    const int h = blockIdx.x, tid = threadIdx.x, j = tid % kD, p = tid / kD, kh = h / (s.hv / s.hk);
    const int qkv_n = 2 * s.hk * kD + s.hv * kD;
    float* base = state + (size_t) h * kD * kD + (size_t) p * kRowsPer * kD + j;
    float st[kRowsPer];
#pragma unroll
    for (int i = 0; i < kRowsPer; ++i) st[i] = base[(size_t) i * kD];
    const float qscale = rsqrtf((float) kD);
    for (int t = 0; t < rows; ++t) {
        const float* r = qkv + (size_t) t * qkv_n;
        float v2 = 0.f;
        if (tid < kD) {
            const float q = r[kh * kD + tid];
            qs[tid] = q;
            v2 = q * q;
        } else if (tid < 2 * kD) {
            const float k = r[s.hk * kD + kh * kD + tid - kD];
            ks[tid - kD] = k;
            v2 = k * k;
        }
        // warps 0-3 sum q, 4-7 sum k
        v2 = warp_sum(v2);
        if ((tid & 31) == 0 && tid < 2 * kD) red[tid >> 5] = v2;
        if (tid == 0) {
            const float a = ba[(size_t) t * 2 * s.hv + s.hv + h] + dt_bias[h], b = ba[(size_t) t * 2 * s.hv + h];
            const float sp = a > 20.f ? a : log1pf(expf(a));
            gb[0] = expf(-expf(a_log[h]) * sp);
            gb[1] = sigm(b);
        }
        __syncthreads();
        const float qn = rsqrtf(red[0] + red[1] + red[2] + red[3] + 1e-6f) * qscale;
        const float kn = rsqrtf(red[4] + red[5] + red[6] + red[7] + 1e-6f);
        // fla's l2norm: x / sqrt(sum x^2 + eps)
        const float decay = gb[0], beta = gb[1];
        const float vj = r[2 * s.hk * kD + h * kD + j];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < kRowsPer; ++i) {
            st[i] *= decay;
            kv = fmaf(st[i], ks[p * kRowsPer + i], kv);
        }
        part[p][j] = kv * kn;
        __syncthreads();
        const float delta = (vj - (part[0][j] + part[1][j] + part[2][j] + part[3][j])) * beta;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < kRowsPer; ++i) {
            st[i] = fmaf(ks[p * kRowsPer + i] * kn, delta, st[i]);
            o = fmaf(st[i], qs[p * kRowsPer + i], o);
        }
        __syncthreads();
        part[p][j] = o * qn;
        __syncthreads();
        float ov = 0.f, sq = 0.f;
        if (tid < kD) {
            ov = part[0][j] + part[1][j] + part[2][j] + part[3][j];
            sq = ov * ov;
        }
        sq = warp_sum(sq);
        if ((tid & 31) == 0 && tid < kD) red[8 + (tid >> 5)] = sq;
        __syncthreads();
        if (tid < kD) {
            const float inv = rsqrtf((red[8] + red[9] + red[10] + red[11]) / kD + eps);
            y[((size_t) t * s.hv + h) * kD + j] = ov * inv * norm_w[j] * sigm(z[(size_t) t * zs + h * kD + j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kRowsPer; ++i) base[(size_t) i * kD] = st[i];
}

// Prefill GDN recurrence split across 4 column blocks per value head (128 blocks total), then per-(row, head) RMSNorm+gate.
constexpr int kColDv = 32;
__global__ void __launch_bounds__(kColDv * kParts) gdn_step_col_kernel(float* __restrict__ state, const float* __restrict__ qkv,
                                                                      const float* __restrict__ ba, const float* __restrict__ a_log,
                                                                      const float* __restrict__ dt_bias, GdnShape s, int rows,
                                                                      float* __restrict__ y) {
    __shared__ float qs[kD], ks[kD], part[kParts][kColDv], red[8], gb[2];
    const int h = blockIdx.x, jl = threadIdx.x % kColDv, p = threadIdx.x / kColDv;
    const int j = blockIdx.y * kColDv + jl, tid = threadIdx.x, kh = h / (s.hv / s.hk);
    const int qkv_n = 2 * s.hk * kD + s.hv * kD;
    float* base = state + (size_t) h * kD * kD + (size_t) p * kRowsPer * kD + j;
    float st[kRowsPer];
#pragma unroll
    for (int i = 0; i < kRowsPer; ++i) st[i] = base[(size_t) i * kD];
    const float qscale = rsqrtf((float) kD);
    for (int t = 0; t < rows; ++t) {
        const float* r = qkv + (size_t) t * qkv_n;
        float qv = r[kh * kD + tid], kv_raw = r[s.hk * kD + kh * kD + tid];
        qs[tid] = qv;
        ks[tid] = kv_raw;
        const float q2 = warp_sum(qv * qv);
        const float k2 = warp_sum(kv_raw * kv_raw);
        if ((tid & 31) == 0) {
            red[tid >> 5] = q2;
            red[4 + (tid >> 5)] = k2;
        }
        if (tid == 0) {
            const float a = ba[(size_t) t * 2 * s.hv + s.hv + h] + dt_bias[h], b = ba[(size_t) t * 2 * s.hv + h];
            const float sp = a > 20.f ? a : log1pf(expf(a));
            gb[0] = expf(-expf(a_log[h]) * sp);
            gb[1] = sigm(b);
        }
        __syncthreads();
        const float qn = rsqrtf(red[0] + red[1] + red[2] + red[3] + 1e-6f) * qscale;
        const float kn = rsqrtf(red[4] + red[5] + red[6] + red[7] + 1e-6f);
        const float decay = gb[0], beta = gb[1];
        const float vj = r[2 * s.hk * kD + h * kD + j];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < kRowsPer; ++i) {
            st[i] *= decay;
            kv = fmaf(st[i], ks[p * kRowsPer + i], kv);
        }
        part[p][jl] = kv * kn;
        __syncthreads();
        const float delta = (vj - (part[0][jl] + part[1][jl] + part[2][jl] + part[3][jl])) * beta;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < kRowsPer; ++i) {
            st[i] = fmaf(ks[p * kRowsPer + i] * kn, delta, st[i]);
            o = fmaf(st[i], qs[p * kRowsPer + i], o);
        }
        __syncthreads();
        part[p][jl] = o * qn;
        __syncthreads();
        if (tid < kColDv) {
            const float ov = part[0][jl] + part[1][jl] + part[2][jl] + part[3][jl];
            y[((size_t) t * s.hv + h) * kD + j] = ov;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kRowsPer; ++i) base[(size_t) i * kD] = st[i];
}

__global__ void __launch_bounds__(kD) gdn_out_norm_kernel(float* __restrict__ y, const float* __restrict__ z, int zs,
                                                          const float* __restrict__ norm_w, float eps, int hv) {
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    float* r = y + ((size_t) t * hv + h) * kD;
    const float ov = r[j];
    const float ss = block_sum(ov * ov, red);
    r[j] = ov * rsqrtf(ss / kD + eps) * norm_w[j] * sigm(z[(size_t) t * zs + h * kD + j]);
}

// Fallback per-head Q8_0 GEMV for unaligned pointers or odd row counts.
__global__ void heads_q8_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, int xh, int xt, int heads, int rows, int k,
                                float* __restrict__ y) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), h = blockIdx.y, t = blockIdx.z, l = threadIdx.x & 31;
    if (r >= rows) return;
    const uint8_t* row = w + ((size_t) h * rows + r) * (k / 32) * 34;
    const float* xv = x + (size_t) t * xt + (size_t) h * xh;
    float acc = 0.f;
    for (int c = l; c < k / 4; c += 32) {
        const uint8_t* b = row + (c / 8) * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const int8_t* q = reinterpret_cast<const int8_t*>(b + 2 + (c % 8) * 4);
        const float4 xx = *reinterpret_cast<const float4*>(xv + 4 * c);
        acc = fmaf(d, q[0] * xx.x + q[1] * xx.y + q[2] * xx.z + q[3] * xx.w, acc);
    }
    acc = warp_sum(acc);
    if (l == 0) y[((size_t) t * heads + h) * rows + r] = acc;
}

// Fast shared-x + uint4 __ldcs tiled Q8_0 per-head product (exact FP32 activation accumulation).
template <int TR, int NB>
__global__ void __launch_bounds__(256) heads_q8_tiled_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, int xh, int xt,
                                                             int heads, int rows, int n_tok, float* __restrict__ y) {
    constexpr int K = NB * 32;
    constexpr int kRowBytes = NB * 34;
    constexpr int kWBytes = TR * kRowBytes;
    extern __shared__ __align__(16) unsigned char sm[];
    uint8_t* ws = sm;
    float* xs = reinterpret_cast<float*>(sm + ((kWBytes + 15) & ~15));
    const int tid = threadIdx.x, tiles = rows / TR, h = blockIdx.x / tiles, r0 = (blockIdx.x - h * tiles) * TR;
    const uint4* src = reinterpret_cast<const uint4*>(w + ((size_t) h * rows + r0) * kRowBytes);
    uint4* dst = reinterpret_cast<uint4*>(ws);
    const int n16 = kWBytes / 16;
    for (int i = tid; i < n16; i += 256) dst[i] = __ldcs(src + i);
    for (int i = tid; i < n_tok * (K / 4); i += 256) {
        const int t = i / (K / 4), c4 = i - t * (K / 4);
        reinterpret_cast<float4*>(xs)[i] = *reinterpret_cast<const float4*>(x + (size_t) t * xt + (size_t) h * xh + c4 * 4);
    }
    __syncthreads();
    const int wp = tid >> 5, l = tid & 31, j = l & 7;
    for (int r = wp; r < TR; r += 8) {
        const uint8_t* row = ws + r * kRowBytes;
        for (int t = 0; t < n_tok; ++t) {
            const float4* x4 = reinterpret_cast<const float4*>(xs + t * K);
            float acc = 0.f;
            for (int b = 0; b < NB; b += 4) {
                const int blk = b + (l >> 3);
                const int c4 = blk * 8 + j;
                const uint8_t* bp = row + blk * 34;
                const int odd = blk & 1;
                const uint32_t* wv = reinterpret_cast<const uint32_t*>(row + ((blk * 34 + 2) & ~3));
                const uint32_t wd0 = wv[j];
                const uint32_t q4 = odd ? wd0 : __funnelshift_r(wd0, wv[j + 1], 16);
                const uint16_t dh = *reinterpret_cast<const uint16_t*>(bp);
                const float d = __half2float(__ushort_as_half(dh));
                const int q0 = (int8_t) (q4 & 0xffu), q1 = (int8_t) ((q4 >> 8) & 0xffu),
                          q2 = (int8_t) ((q4 >> 16) & 0xffu), q3 = (int8_t) (q4 >> 24);
                const float4 xv = x4[c4];
                acc = fmaf(d, q0 * xv.x + q1 * xv.y + q2 * xv.z + q3 * xv.w, acc);
            }
            acc = warp_sum(acc);
            if (l == 0) y[((size_t) t * heads + h) * rows + r0 + r] = acc;
        }
    }
}

namespace mqa = basalt::kernels::mqa;
constexpr int kAttTok = 8;
constexpr int kMaxChunks = 128;   // up to 8,192 cells in mqa::attend_kernel
constexpr int kMmaCells = 32, kLd = 584, kPLd = 40;

__device__ __forceinline__ void bf16x2(uint32_t v, float* k) {
    k[0] = __uint_as_float(v << 16);
    k[1] = __uint_as_float(v & 0xffff0000u);
}
__device__ __forceinline__ void f16x2(uint32_t v, float* k) {
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&v));
    k[0] = f.x, k[1] = f.y;
}
__device__ __forceinline__ float fp8_val(uint32_t code, float sc) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) code, __NV_E4M3);
    return __half2float(__half(h)) * sc;
}

struct RowF16 {
    using Raw4 = uint2;
    using Raw2 = uint32_t;
    using RawR = uint32_t;
    using Scales = mqa::NoScales;
    static constexpr int kRowBytes = 576 * 2;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const uint2*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r)[i]; }
    static __device__ __forceinline__ RawR loadr(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r + 1024)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { f16x2(v.x, k); f16x2(v.y, k + 2); }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) { f16x2(v, k); }
    static __device__ __forceinline__ void cvtr(const RawR& v, float k[2]) { f16x2(v, k); }
};

struct RowFp8 {
    using Raw4 = uint32_t;
    using Raw2 = uint2;
    using RawR = uint32_t;
    using Scales = float4;
    static constexpr int kRowBytes = kFp8RowBytes;
    static __device__ __forceinline__ Scales scales(const uint8_t* r) { return *reinterpret_cast<const float4*>(r + 512); }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) {
        return make_uint2(reinterpret_cast<const uint16_t*>(r)[i], __float_as_uint(reinterpret_cast<const float*>(r + 512)[i >> 6]));
    }
    static __device__ __forceinline__ RawR loadr(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r + 528)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales& s, int j, float k[4]) {
        const float sc = j == 0 ? s.x : j == 1 ? s.y : j == 2 ? s.z : s.w;
#pragma unroll
        for (int e = 0; e < 4; ++e) k[e] = fp8_val((v >> (8 * e)) & 0xffu, sc);
    }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) {
        const float sc = __uint_as_float(v.y);
        k[0] = fp8_val(v.x & 0xffu, sc);
        k[1] = fp8_val((v.x >> 8) & 0xffu, sc);
    }
    static __device__ __forceinline__ void cvtr(const RawR& v, float k[2]) { bf16x2(v, k); }
};

template <class R, int HB>
__global__ void __launch_bounds__(mqa::kThreads) attend_kernel(const float* __restrict__ qcur, const uint8_t* __restrict__ cache,
                                                               const int32_t* __restrict__ cells, int cs,
                                                               const int32_t* __restrict__ n_cells, int heads, float scale, int max_ch,
                                                               float* __restrict__ ml, float* __restrict__ op, int* __restrict__ counters,
                                                               float* __restrict__ o_lat) {
    constexpr int W = mqa::kLatent + 64;
    __shared__ float4 qs[HB * W / 4];
    static_assert(mqa::merge_scratch_floats(HB, kMaxChunks) <= HB * W, "merge scratch fits query rows");
    __shared__ __align__(16) float sc[HB][mqa::kChunk];
    __shared__ int rowid[mqa::kChunk];
    __shared__ int last;
    const int c = blockIdx.x, g = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    const int n = n_cells[t], n_ch = max(1, (n + mqa::kChunk - 1) / mqa::kChunk);
    if (c >= n_ch) return;
    const int head0 = g * HB;
    constexpr int kQ4 = HB * W / 4, kQ = (kQ4 + mqa::kThreads - 1) / mqa::kThreads;
    float4 qv[kQ];
    const float4* qg = reinterpret_cast<const float4*>(qcur + ((size_t) t * heads + head0) * W);
#pragma unroll
    for (int k = 0; k < kQ; ++k)
        if (tid + k * mqa::kThreads < kQ4) qv[k] = qg[tid + k * mqa::kThreads];
    int row = -1;
    if (tid < mqa::kChunk) {
        const int j = c * mqa::kChunk + tid;
        if (j < n) row = cells ? cells[(size_t) t * cs + j] : j;
        rowid[tid] = row;
    }
#pragma unroll
    for (int k = 0; k < kQ; ++k)
        if (tid + k * mqa::kThreads < kQ4) qs[tid + k * mqa::kThreads] = qv[k];
    const bool any = __syncthreads_or(tid < mqa::kChunk && row >= 0);
    const size_t base = ((size_t) t * heads + head0) * max_ch;
    if (any) {
        mqa::chunk<R, R, HB, 64>(true, cache, scale, qs, sc, rowid, ml + (base + c) * 2, (size_t) max_ch * 2,
                                 op + (base + c) * mqa::kLatent, (size_t) max_ch * mqa::kLatent);
    } else if (tid < HB) {
        float* m = ml + (base + (size_t) tid * max_ch + c) * 2;
        m[0] = -INFINITY;
        m[1] = 0.f;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        int* cnt = counters + (size_t) t * (heads / HB) + g;
        last = atomicAdd(cnt, 1) == n_ch - 1;
        if (last) *cnt = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    float2 o[HB];
    mqa::merge<HB, true, false>(op + base * mqa::kLatent, (size_t) max_ch * mqa::kLatent, ml + base * 2, (size_t) max_ch * 2, n_ch, nullptr,
                                reinterpret_cast<float*>(qs), o);
#pragma unroll
    for (int h = 0; h < HB; ++h) reinterpret_cast<float2*>(o_lat + ((size_t) t * heads + head0 + h) * mqa::kLatent)[tid] = o[h];
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((uint32_t) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((uint32_t) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void mma16816h(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t v) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
    const __half2 h = __float22half2_rn(f);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint4 fetch8(const __half* cache, size_t cell, int ch) {
    return *reinterpret_cast<const uint4*>(cache + cell * 576 + ch * 8);
}
__device__ __forceinline__ uint4 fetch8(const uint8_t* cache, size_t cell, int ch) {
    const uint8_t* row = cache + cell * kFp8RowBytes;
    if (ch < 512 / 8) {
        const uint2 c = *reinterpret_cast<const uint2*>(row + ch * 8);
        return make_uint4(c.x, c.y, __float_as_uint(reinterpret_cast<const float*>(row + 512)[ch >> 4]), 0);
    }
    return *reinterpret_cast<const uint4*>(row + 528 + (ch - 512 / 8) * 16);
}
__device__ __forceinline__ uint4 half8(const __half*, uint4 v, int) { return v; }
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t codes, __half2 scale) {
    const __half2 h = __hmul2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) codes, __NV_E4M3)), scale);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint4 half8(const uint8_t*, uint4 v, int ch) {
    if (ch >= 512 / 8) return make_uint4(bf2_to_h2(v.x), bf2_to_h2(v.y), bf2_to_h2(v.z), bf2_to_h2(v.w));
    const __half2 s = __float2half2_rn(__uint_as_float(v.z));
    return make_uint4(e4m3x2_to_h2(v.x & 0xffffu, s), e4m3x2_to_h2(v.x >> 16, s),
                      e4m3x2_to_h2(v.y & 0xffffu, s), e4m3x2_to_h2(v.y >> 16, s));
}

template <int G> constexpr size_t mma_smem() {
    return (size_t) (16 * G * kLd + kMmaCells * kLd + 16 * G * kPLd) * 2 + (2 * 16 * G * 32 + 2 * 16 * G) * sizeof(float);
}
template <typename C, int G>
__global__ void __launch_bounds__(256) attend_mma_kernel(const float* __restrict__ qcur, const C* __restrict__ cache,
                                                         const int32_t* __restrict__ cells, int cells_stride,
                                                         const int32_t* __restrict__ n_cells, MlaShape s, float scale,
                                                         float* __restrict__ o_lat) {
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int W = 576, M = 16 * G;
    uint16_t* qh = (uint16_t*) smraw;
    uint16_t* ks = qh + M * kLd;
    uint16_t* ph = ks + kMmaCells * kLd;
    float* sp = (float*) (ph + M * kPLd);
    float* alpha_s = sp + 2 * M * 32;
    float* l_s = alpha_s + M;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int t = blockIdx.y, h0 = blockIdx.x * M;
    const int n = n_cells[t];
    const int32_t* cl = cells ? cells + (size_t) t * cells_stride : nullptr;
    for (int i = tid; i < M * W; i += 256) {
        const int h = i / W, e = i - h * W;
        qh[h * kLd + e] = __half_as_ushort(__float2half_rn(qcur[((size_t) t * s.heads + h0) * W + i] * scale));
    }
    float m[2 * G], l[2 * G];
#pragma unroll
    for (int hh = 0; hh < 2 * G; ++hh) { m[hh] = -INFINITY; l[hh] = 0.f; }
    if (tid < M) l_s[tid] = 0.f;
    float acc[8 * G][4];
#pragma unroll
    for (int j = 0; j < 8 * G; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const int id = lane >> 3, r8 = lane & 7, rg = w % G, dq = w / G;
    const int kh = (w / G) & 1, cp = w / (2 * G);
    constexpr int kPer = kMmaCells * (W / 8) / 256;
    uint4 pre[kPer];
    auto fetch = [&](int j0) {
        const int tn = min(kMmaCells, n - j0);
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = tid + k * 256, c = i / (W / 8), ch = i - c * (W / 8);
            pre[k] = make_uint4(0, 0, 0, 0);
            if (c < tn) {
                const int j = j0 + c;
                const int cell = cl ? cl[j] : j;
                pre[k] = fetch8(cache, (size_t) cell, ch);
            }
        }
    };
    fetch(0);
    for (int j0 = 0; j0 < n; j0 += kMmaCells) {
        const int tn = min(kMmaCells, n - j0);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = tid + k * 256, c = i / (W / 8), ch = i - c * (W / 8);
            *reinterpret_cast<uint4*>(ks + c * kLd + ch * 8) = half8(cache, pre[k], ch);
        }
        __syncthreads();
        if (j0 + kMmaCells < n) fetch(j0 + kMmaCells);
        {
            float c[G][4];
#pragma unroll
            for (int nt = 0; nt < G; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
            for (int kk = 0; kk < 9; ++kk) {
                const int kb = (kh * 18 + kk * 2) * 16;
                uint32_t b[G][4];
#pragma unroll
                for (int nt = 0; nt < G; ++nt) ldsm4(b[nt], ks + ((cp * G + nt) * 8 + r8) * kLd + kb + id * 8);
#pragma unroll
                for (int s2 = 0; s2 < 2; ++s2) {
                    const int kd = kb + s2 * 16;
                    uint32_t ah[4];
                    ldsm4(ah, qh + (rg * 16 + (id & 1) * 8 + r8) * kLd + kd + (id >> 1) * 8);
#pragma unroll
                    for (int nt = 0; nt < G; ++nt) mma16816h(c[nt], ah, b[nt][2 * s2], b[nt][2 * s2 + 1]);
                }
            }
            const int row = rg * 16 + (lane >> 2);
#pragma unroll
            for (int nt = 0; nt < G; ++nt) {
                const int col = (cp * G + nt) * 8 + (lane & 3) * 2;
                sp[(kh * M + row) * 32 + col] = c[nt][0];
                sp[(kh * M + row) * 32 + col + 1] = c[nt][1];
                sp[(kh * M + row + 8) * 32 + col] = c[nt][2];
                sp[(kh * M + row + 8) * 32 + col + 1] = c[nt][3];
            }
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2 * G; ++hh) {
            const int h = w * 2 * G + hh;
            const bool valid = lane < tn;
            const float scv = valid ? sp[h * 32 + lane] + sp[(M + h) * 32 + lane] : -INFINITY;
            float mx = scv;
            for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float nm = fmaxf(m[hh], mx);
            const float alpha = (nm > -INFINITY && m[hh] > -INFINITY) ? expf(m[hh] - nm) : 0.f;
            const float p = (valid && nm > -INFINITY) ? expf(scv - nm) : 0.f;
            float sum = p;
            for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            l[hh] = l[hh] * alpha + sum;
            m[hh] = nm;
            ph[h * kPLd + lane] = __half_as_ushort(__float2half_rn(p));
            if (lane == 0) { alpha_s[h] = alpha; l_s[h] = l[hh]; }
        }
        __syncthreads();
        {
            const float a0 = alpha_s[rg * 16 + (lane >> 2)], a1 = alpha_s[rg * 16 + (lane >> 2) + 8];
#pragma unroll
            for (int j = 0; j < 8 * G; ++j) {
                acc[j][0] *= a0; acc[j][1] *= a0; acc[j][2] *= a1; acc[j][3] *= a1;
            }
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {
                uint32_t ah[4];
                ldsm4(ah, ph + (rg * 16 + (id & 1) * 8 + r8) * kPLd + s2 * 16 + (id >> 1) * 8);
#pragma unroll
                for (int jj = 0; jj < 4 * G; ++jj) {
                    uint32_t b[4];
                    ldsm4t(b, ks + (s2 * 16 + (id & 1) * 8 + r8) * kLd + dq * 64 * G + jj * 16 + (id >> 1) * 8);
                    mma16816h(acc[2 * jj], ah, b[0], b[1]);
                    mma16816h(acc[2 * jj + 1], ah, b[2], b[3]);
                }
            }
        }
    }
    __syncthreads();
    const int row = rg * 16 + (lane >> 2);
    const float i0 = l_s[row] > 0.f ? 1.f / l_s[row] : 0.f;
    const float i1 = l_s[row + 8] > 0.f ? 1.f / l_s[row + 8] : 0.f;
#pragma unroll
    for (int j = 0; j < 8 * G; ++j) {
        const int d = dq * 64 * G + j * 8 + (lane & 3) * 2;
        float* o0 = o_lat + ((size_t) t * s.heads + h0 + row) * s.rank + d;
        float* o1 = o_lat + ((size_t) t * s.heads + h0 + row + 8) * s.rank + d;
        o0[0] = acc[j][0] * i0; o0[1] = acc[j][1] * i0;
        o1[0] = acc[j][2] * i1; o1[1] = acc[j][3] * i1;
    }
}

// Tensor-core indexer scores for 64 heads x 128 dims: each block loads 64 keys into smem once and 4 warps take 16 heads each.
constexpr int kIdxTile = 64, kIdxKsLd = 136;
__global__ void __launch_bounds__(128) idx_score_mma_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                            const __half* __restrict__ keys, const int32_t* __restrict__ n_cells,
                                                            int max_cells, float* __restrict__ scores) {
    __shared__ __align__(16) uint16_t ks[kIdxTile][kIdxKsLd];
    __shared__ float sp[4][kIdxTile];
    const int t = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nc = min(n_cells[t], max_cells);
    const int c0 = blockIdx.x * kIdxTile;
    if (c0 >= max_cells) return;
    if (c0 >= nc) {
        if (tid < kIdxTile && c0 + tid < max_cells) scores[(size_t) t * max_cells + c0 + tid] = -FLT_MAX;
        return;
    }
    for (int i = tid; i < kIdxTile * 16; i += 128) {
        const int row = i >> 4, ch = i & 15;
        const uint4 v = (c0 + row < nc) ? *reinterpret_cast<const uint4*>(keys + (size_t) (c0 + row) * 128 + ch * 8) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&ks[row][ch * 8]) = v;
    }
    const int h0 = warp * 16;
    const float* qt = q + ((size_t) t * 64 + h0) * 128;
    const float* wt = w + (size_t) t * 64 + h0;
    uint32_t qh[8][4], ql[8][4];
#pragma unroll
    for (int kstep = 0; kstep < 8; ++kstep)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = (lane >> 2) + (j & 1) * 8, col = kstep * 16 + (lane & 3) * 2 + (j >> 1) * 8;
            const float2 v = *reinterpret_cast<const float2*>(qt + r * 128 + col);
            const __half2 hi = __floats2half2_rn(v.x, v.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(v.x - hf.x, v.y - hf.y);
            qh[kstep][j] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kstep][j] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    const float wr0 = wt[lane >> 2], wr1 = wt[(lane >> 2) + 8];
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < kIdxTile / 8; ++nt) {
        const int p0 = nt * 8, pn = p0 + (lane >> 2);
        const uint32_t* kr = reinterpret_cast<const uint32_t*>(&ks[pn][0]);
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kstep = 0; kstep < 8; ++kstep) {
            const uint32_t b0 = kr[kstep * 8 + (lane & 3)], b1 = kr[kstep * 8 + 4 + (lane & 3)];
            mma16816h(c, qh[kstep], b0, b1);
            mma16816h(c, ql[kstep], b0, b1);
        }
        float v0 = wr0 * fmaxf(c[0], 0.f) + wr1 * fmaxf(c[2], 0.f);
        float v1 = wr0 * fmaxf(c[1], 0.f) + wr1 * fmaxf(c[3], 0.f);
        for (int o = 4; o < 32; o <<= 1) {
            v0 += __shfl_xor_sync(0xffffffffu, v0, o);
            v1 += __shfl_xor_sync(0xffffffffu, v1, o);
        }
        if (lane < 4) { sp[warp][p0 + lane * 2] = v0; sp[warp][p0 + lane * 2 + 1] = v1; }
    }
    __syncthreads();
    if (tid < kIdxTile && c0 + tid < max_cells) {
        const int c = c0 + tid;
        scores[(size_t) t * max_cells + c] = c < nc ? (sp[0][tid] + sp[1][tid] + sp[2][tid] + sp[3][tid]) : -FLT_MAX;
    }
}

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return u ^ ((u >> 31) ? 0xffffffffu : 0x80000000u);
}

constexpr int kSelThreads = 1024;
__global__ void __launch_bounds__(kSelThreads) idx_select_kernel(const float* __restrict__ scores, const int32_t* __restrict__ n_cells,
                                                                int max_cells, int k, int32_t* __restrict__ cells, int32_t* __restrict__ n_sel) {
    __shared__ int hist[2048];
    __shared__ uint32_t sh_prefix;
    __shared__ int sh_rem, sh_count[kSelThreads + 1], sh_eq[kSelThreads + 1], sh_need, sh_taken[kSelThreads];
    const int t = threadIdx.x, lane = t & 31, row = blockIdx.x;
    const int n = min(n_cells[row], max_cells);
    const float* sr = scores + (size_t) row * max_cells;
    int32_t* out = cells + (size_t) row * k;
    if (n <= k) {
        for (int i = t; i < n; i += kSelThreads) out[i] = i;
        if (t == 0) n_sel[row] = n;
        return;
    }
    const int chunk = (n + kSelThreads - 1) / kSelThreads, lo = min(n, t * chunk), hi = min(n, lo + chunk);
    uint32_t prefix = 0, mask = 0;
    int rem = k;
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = pass == 0 ? 21 : pass == 1 ? 10 : 0, nb = pass == 2 ? 1024 : 2048;
        for (int i = t; i < nb; i += kSelThreads) hist[i] = 0;
        __syncthreads();
        for (int base = 0; base < n; base += kSelThreads) {
            const int i = base + t;
            const uint32_t key = i < n ? order_key(sr[i]) : 0u;
            const int bin = i < n && (key & mask) == prefix ? (int) ((key >> shift) & (uint32_t) (nb - 1)) : -1;
            const unsigned same = __match_any_sync(0xffffffffu, bin);
            if (bin >= 0 && lane == __ffs(same) - 1) atomicAdd(&hist[bin], __popc(same));
        }
        __syncthreads();
        if (t < 32) {
            const int per = nb / 32, top = nb - 1 - lane * per;
            int s = 0;
            for (int j = 0; j < per; ++j) s += hist[top - j];
            int incl = s;
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            const int first = __ffs(__ballot_sync(0xffffffffu, incl >= rem)) - 1;
            if (lane == first) {
                int c = incl - s, b = top;
                for (; c + hist[b] < rem; --b) c += hist[b];
                sh_prefix = prefix | ((uint32_t) b << shift);
                sh_rem = rem - c;
            }
        }
        __syncthreads();
        prefix = sh_prefix;
        rem = sh_rem;
        mask |= (uint32_t) (nb - 1) << shift;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t key = order_key(sr[i]);
        gt += key > thr;
        eq += key == thr;
    }
    sh_count[t] = gt;
    sh_eq[t] = eq;
    __syncthreads();
    if (t == 0) {
        int a = 0, b = 0;
        for (int i = 0; i < kSelThreads; ++i) {
            const int x = sh_count[i], y = sh_eq[i];
            sh_count[i] = a; sh_eq[i] = b; a += x; b += y;
        }
        sh_need = k - a;
    }
    __syncthreads();
    int tie_before = sh_eq[t];
    const int need = sh_need;
    sh_taken[t] = gt + max(0, min(eq, need - tie_before));
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < kSelThreads; ++i) { const int x = sh_taken[i]; sh_taken[i] = a; a += x; }
        n_sel[row] = k;
    }
    __syncthreads();
    int slot = sh_taken[t];
    for (int i = lo; i < hi; ++i) {
        const uint32_t key = order_key(sr[i]);
        bool take = key > thr;
        if (key == thr) take = tie_before++ < need;
        if (take) out[slot++] = i;
    }
}

__global__ void f32_gemv_kernel(const float* __restrict__ w, const float* __restrict__ x, int rows, int k, float* __restrict__ y) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), t = blockIdx.y, l = threadIdx.x & 31;
    if (r >= rows) return;
    const float4* wr = reinterpret_cast<const float4*>(w + (size_t) r * k);
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t) t * k);
    float acc = 0.f;
    for (int i = l; i < k / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = warp_sum(acc);
    if (l == 0) y[(size_t) t * rows + r] = acc;
}

__global__ void swiglu_kernel(const float* __restrict__ gu, int n, int64_t total, float limit, float* __restrict__ h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t t = i / n, c = i % n;
    const float g = fminf(gu[t * 2 * n + c], limit), u = fminf(fmaxf(gu[t * 2 * n + n + c], -limit), limit);
    h[i] = g * sigm(g) * u;
}

__global__ void mul_sigmoid_kernel(float* __restrict__ x, const float* __restrict__ g, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigm(g[i]);
}

// Warp-parallel plan for small windows (n_ent <= 32, e.g. 1..4 decode rows): zero shared memory or barriers.
__global__ void moe_plan_small_kernel(const int32_t* __restrict__ ids, int n_ent, int k,
                                      const unsigned long long* __restrict__ table, unsigned long long* __restrict__ grp_ptr,
                                      int32_t* __restrict__ grp_start, int32_t* __restrict__ n_groups,
                                      int32_t* __restrict__ ent_dst, int32_t* __restrict__ ent_tok, uint32_t* __restrict__ counts) {
    const int i = threadIdx.x;
    const int id = i < n_ent ? ids[i] : -1;
    if (counts && i < n_ent) atomicAdd(&counts[id], 1u);
    int lt = 0, eq_before = 0;
    uint32_t lt_mask = 0;
    for (int j = 0; j < n_ent; ++j) {
        const int oid = __shfl_sync(0xffffffffu, id, j);
        if (i < n_ent) {
            if (oid < id) { ++lt; lt_mask |= 1u << j; }
            else if (oid == id && j < i) ++eq_before;
        }
    }
    const bool is_first = (i < n_ent && eq_before == 0);
    const unsigned first_mask = __ballot_sync(0xffffffffu, is_first);
    if (is_first) {
        const int g = __popc(first_mask & lt_mask);
        grp_ptr[g] = table[id];
        grp_start[g] = lt;
    }
    if (i < n_ent) {
        const int slot = lt + eq_before;
        ent_dst[slot] = i;
        ent_tok[slot] = i / k;
    }
    if (i == 0) {
        const int ng = __popc(first_mask);
        grp_start[ng] = n_ent;
        *n_groups = ng;
    }
}

// One block; thread i an entry.
__global__ void moe_plan_kernel(const int32_t* __restrict__ ids, int n_ent, int k, int n_expert, const unsigned long long* __restrict__ table,
                                unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start, int32_t* __restrict__ n_groups,
                                int32_t* __restrict__ ent_dst, int32_t* __restrict__ ent_tok, uint32_t* __restrict__ counts) {
    __shared__ int cnt[512], first[512];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    const int i = threadIdx.x;
    const int id = i < n_ent ? ids[i] : 0;
    if (i < n_ent) {
        atomicAdd(&cnt[id], 1);
        if (counts) atomicAdd(&counts[id], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {   // exclusive scans: entries before an expert, groups before it
        int ent = 0, g = 0;
        for (int e = 0; e < n_expert; ++e) {
            const int c = cnt[e];
            first[e] = ent;
            if (c) {
                grp_ptr[g] = table[e];
                grp_start[g] = ent;
                cnt[e] = g++;
            } else {
                cnt[e] = -1;
            }
            ent += c;
        }
        grp_start[g] = ent;
        *n_groups = g;
    }
    __syncthreads();
    if (i < n_ent) {
        int before = 0;
        for (int j = 0; j < i; ++j) before += ids[j] == id;
        const int slot = first[id] + before;
        ent_dst[slot] = i;
        ent_tok[slot] = i / k;
    }
}

constexpr int kSortPairs = 1024;
constexpr int kMaxSortExpert = 512;

int sort_blocks(int pairs) { return (pairs + kSortPairs - 1) / kSortPairs; }

__global__ void __launch_bounds__(256) sort_count_kernel(const int32_t* __restrict__ ids, int pairs, int n_expert,
                                                         int32_t* __restrict__ counts) {
    __shared__ int32_t hist[kMaxSortExpert];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) hist[e] = 0;
    __syncthreads();
    const int lo = blockIdx.x * kSortPairs, hi = min(pairs, lo + kSortPairs);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x)
        if (ids[i] >= 0) atomicAdd(&hist[ids[i]], 1);
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) counts[(size_t) blockIdx.x * n_expert + e] = hist[e];
}

__global__ void __launch_bounds__(kMaxSortExpert) sort_scan_kernel(int32_t* __restrict__ counts, int n_blocks, int n_expert,
                                                                   int32_t* __restrict__ bounds) {
    __shared__ int32_t total[kMaxSortExpert];
    __shared__ int32_t all;
    const int e = threadIdx.x;
    int run = 0;
    if (e < n_expert)
        for (int b = 0; b < n_blocks; ++b) {
            const int c = counts[(size_t) b * n_expert + e];
            counts[(size_t) b * n_expert + e] = run;
            run += c;
        }
    total[e] = e < n_expert ? run : 0;
    __syncthreads();
    if (e == 0) {
        int at = 0;
        for (int x = 0; x < n_expert; ++x) {
            const int t = total[x];
            total[x] = at;
            at += t;
        }
        all = at;
    }
    __syncthreads();
    if (e < n_expert) {
        bounds[e] = total[e];
        for (int b = 0; b < n_blocks; ++b) counts[(size_t) b * n_expert + e] += total[e];
    }
    if (e == 0) bounds[n_expert] = all;
}

__global__ void __launch_bounds__(32) sort_scatter_kernel(const int32_t* __restrict__ ids, int pairs, int k, int n_expert,
                                                          const int32_t* __restrict__ counts, int32_t* __restrict__ tok,
                                                          int32_t* __restrict__ dst, int32_t* __restrict__ exp) {
    __shared__ int32_t next[kMaxSortExpert];
    const int lane = threadIdx.x;
    for (int e = lane; e < n_expert; e += 32) next[e] = counts[(size_t) blockIdx.x * n_expert + e];
    __syncwarp();
    const int lo = blockIdx.x * kSortPairs, hi = min(pairs, lo + kSortPairs);
    for (int i0 = lo; i0 < hi; i0 += 32) {
        const int i = i0 + lane;
        const int e = i < hi ? ids[i] : -1;
        const unsigned same = __match_any_sync(0xffffffffu, e);
        const int rank = __popc(same & ((1u << lane) - 1u));
        if (e >= 0) {
            const int at = next[e] + rank;
            tok[at] = i / k;
            dst[at] = i;
            exp[at] = e;
        }
        __syncwarp();
        if (e >= 0 && rank == 0) next[e] += __popc(same);
        __syncwarp();
    }
}

__global__ void moe_combine_kernel(const float* __restrict__ ex, const float* __restrict__ w, const float* __restrict__ sh, int k, int n,
                                   float* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = sh[(size_t) t * n + i];
    for (int j = 0; j < k; ++j) v = fmaf(w[t * k + j], ex[((size_t) t * k + j) * n + i], v);
    out[(size_t) t * n + i] = v;
}

__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ x, int64_t n, int32_t* __restrict__ ids) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* r = x + (size_t) blockIdx.x * n;
    float v = -FLT_MAX;
    int idx = 0;
    for (int64_t i = threadIdx.x; i < n; i += 1024)
        if (r[i] > v) v = r[i], idx = (int) i;
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, m);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, m);
        if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
    }
    if ((threadIdx.x & 31) == 0) bv[threadIdx.x >> 5] = v, bi[threadIdx.x >> 5] = idx;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = bv[threadIdx.x], idx = bi[threadIdx.x];
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, m);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, m);
            if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
        }
        if (threadIdx.x == 0) ids[blockIdx.x] = idx;
    }
}

__global__ void __launch_bounds__(1024) argmax_prob_kernel(const float* __restrict__ x, int64_t n, int32_t* __restrict__ out_id,
                                                            float* __restrict__ out_prob) {
    __shared__ float bv[32], smax;
    __shared__ int bi[32];
    float v = -FLT_MAX;
    int idx = 0;
    for (int64_t i = threadIdx.x; i < n; i += 1024)
        if (x[i] > v) v = x[i], idx = (int) i;
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, m);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, m);
        if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
    }
    if ((threadIdx.x & 31) == 0) bv[threadIdx.x >> 5] = v, bi[threadIdx.x >> 5] = idx;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = bv[threadIdx.x], idx = bi[threadIdx.x];
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, m);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, m);
            if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
        }
        if (threadIdx.x == 0) smax = v, *out_id = idx;
    }
    __syncthreads();
    const float mval = smax;
    float s = 0.f;
    for (int64_t i = threadIdx.x; i < n; i += 1024) s += __expf(x[i] - mval);
    const float denom = block_sum(s, bv);
    if (threadIdx.x == 0) *out_prob = denom > 0.f ? 1.f / denom : 0.f;
}

__global__ void set_rows_kernel(int32_t* pos, int32_t* ncells, int n, int p0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pos[i] = p0 + i, ncells[i] = p0 + i + 1;
}

__global__ void advance_step_kernel(int32_t* pos, int32_t* ncells) {
    pos[0] += 1;
    ncells[0] = pos[0] + 1;
}

__global__ void advance_kernel(const int32_t* tok, int32_t* pos, int32_t* ncells, int32_t* out, int32_t* i, int cap) {
    pos[0] += 1;
    ncells[0] = pos[0] + 1;
    out[*i % cap] = tok[0];
    *i += 1;
}

unsigned blocks(int64_t n, int b) { return (unsigned) ((n + b - 1) / b); }
int chunks_of(int max_cells) { return std::max(1, (max_cells + mqa::kChunk - 1) / mqa::kChunk); }
size_t counters_floats(int heads) { return (((size_t) kAttTok * (heads / 8) * 4 + 255) & ~(size_t) 255) / sizeof(float); }

template <typename C> struct RowOf;
template <> struct RowOf<__half> { using T = RowF16; };
template <> struct RowOf<uint8_t> { using T = RowFp8; };

int sm_count() {
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return sms;
}

}  // namespace

void embed_rows(const uint16_t* table, const int32_t* ids, int n, int rows, float* out, void* stream) {
    if (n % 4) throw std::invalid_argument("embed_rows: n");
    if (rows > 0) embed_kernel<<<rows, 256, 0, S(stream)>>>(table, ids, n, out);
    done(stream);
}

void gdn_conv(const float* x, int x_stride, float* state, const float* w, int ch, int rows, float* y, void* stream) {
    if (rows > 0) gdn_conv_kernel<<<blocks(ch, 256), 256, 0, S(stream)>>>(x, x_stride, state, w, ch, rows, y);
    done(stream);
}

void gdn_step(float* state, const float* qkv, const float* ba, const float* a_log, const float* dt_bias, const float* z, int z_stride,
              const float* norm_w, float eps, const GdnShape& s, int rows, float* y, void* stream) {
    if (s.dk != kD || s.dv != kD || s.hv % s.hk) throw std::invalid_argument("gdn_step: geometry");
    if (rows > 0) {
        cudaStream_t st = S(stream);
        if (rows <= 8) {
            gdn_step_kernel<<<s.hv, kD * kParts, 0, st>>>(state, qkv, ba, a_log, dt_bias, z, z_stride, norm_w, eps, s, rows, y);
        } else {
            gdn_step_col_kernel<<<dim3(s.hv, kD / kColDv), kColDv * kParts, 0, st>>>(state, qkv, ba, a_log, dt_bias, s, rows, y);
            gdn_out_norm_kernel<<<dim3(rows, s.hv), kD, 0, st>>>(y, z, z_stride, norm_w, eps, s.hv);
        }
    }
    done(stream);
}

void heads_q8_gemv(const void* w, const float* x, int x_head, int x_tok, int heads, int rows, int k, int n_tok, float* y, void* stream) {
    if (k % 128 || x_head % 4 || x_tok % 4) throw std::invalid_argument("heads_q8_gemv: geometry");
    if (n_tok > 0) {
        if (n_tok >= 2 && n_tok <= 8 && (k == 128 || k == 512) && ((uintptr_t) w & 15) == 0 && (rows % 16 == 0)) {
            constexpr int TR = 16;
            const size_t sm = ((size_t) TR * (k / 32) * 34 + 15) / 16 * 16 + (size_t) n_tok * k * sizeof(float);
            if (k == 128)
                heads_q8_tiled_kernel<TR, 4><<<heads * (rows / TR), 256, sm, S(stream)>>>(
                    (const uint8_t*) w, x, x_head, x_tok, heads, rows, n_tok, y);
            else
                heads_q8_tiled_kernel<TR, 16><<<heads * (rows / TR), 256, sm, S(stream)>>>(
                    (const uint8_t*) w, x, x_head, x_tok, heads, rows, n_tok, y);
        } else {
            heads_q8_kernel<<<dim3(blocks(rows, 8), heads, n_tok), 256, 0, S(stream)>>>((const uint8_t*) w, x, x_head, x_tok, heads, rows, k, y);
        }
    }
    done(stream);
}

size_t mla_work_floats(int heads, int rank, int max_cells, int n_tok) {
    const int n_t = std::min(n_tok, kAttTok);
    return counters_floats(heads) + (size_t) n_t * heads * chunks_of(max_cells) * (rank + 2);
}

template <typename C>
void mla_attend_impl(const float* qcur, const C* cache, const int32_t* cells, int cell_stride, const int32_t* n_cells, int max_cells,
                     const MlaShape& s, float scale, int n_tok, float* work, float* out, void* stream) {
    if (s.rank != mqa::kLatent || s.rope != 64 || s.heads % 8) throw std::invalid_argument("mla_attend: geometry");
    if (n_tok <= 0) { done(stream); return; }
    cudaStream_t st = S(stream);
    if (n_tok > kAttTok && s.heads % 32 == 0) {
        static bool raised = false;
        if (!raised) {
            cudaFuncSetAttribute(attend_mma_kernel<C, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mma_smem<2>());
            raised = true;
        }
        attend_mma_kernel<C, 2><<<dim3(s.heads / 32, n_tok), 256, mma_smem<2>(), st>>>(qcur, cache, cells, cell_stride, n_cells, s, scale, out);
        done(stream);
        return;
    }
    using R = typename RowOf<C>::T;
    const int max_ch = chunks_of(max_cells);
    if (max_ch > kMaxChunks) throw std::invalid_argument("mla_attend: max_cells exceeds 8192");
    int* counters = reinterpret_cast<int*>(work);
    float* ml = work + counters_floats(s.heads);
    for (int t0 = 0; t0 < n_tok; t0 += kAttTok) {
        const int T = std::min(kAttTok, n_tok - t0);
        float* op = ml + (size_t) T * s.heads * max_ch * 2;
        const int32_t* ct = cells ? cells + (size_t) t0 * cell_stride : nullptr;
        const float* q = qcur + (size_t) t0 * s.heads * (mqa::kLatent + 64);
        float* o = out + (size_t) t0 * s.heads * mqa::kLatent;
        if (max_ch * (s.heads / 8) * T <= sm_count() || s.heads % 16)
            attend_kernel<R, 8><<<dim3(max_ch, s.heads / 8, T), mqa::kThreads, 0, st>>>(
                q, reinterpret_cast<const uint8_t*>(cache), ct, cell_stride, n_cells + t0, s.heads, scale, max_ch, ml, op, counters, o);
        else
            attend_kernel<R, 16><<<dim3(max_ch, s.heads / 16, T), mqa::kThreads, 0, st>>>(
                q, reinterpret_cast<const uint8_t*>(cache), ct, cell_stride, n_cells + t0, s.heads, scale, max_ch, ml, op, counters, o);
    }
    done(stream);
}

void mla_attend(const float* qcur, const uint16_t* cache, const int32_t* cells, int cell_stride, const int32_t* n_cells, int max_cells,
                const MlaShape& s, float scale, int n_tok, float* work, float* out, void* stream) {
    mla_attend_impl(qcur, reinterpret_cast<const __half*>(cache), cells, cell_stride, n_cells, max_cells, s, scale, n_tok, work, out, stream);
}

void mla_attend(const float* qcur, const uint8_t* cache, const int32_t* cells, int cell_stride, const int32_t* n_cells, int max_cells,
                const MlaShape& s, float scale, int n_tok, float* work, float* out, void* stream) {
    mla_attend_impl(qcur, cache, cells, cell_stride, n_cells, max_cells, s, scale, n_tok, work, out, stream);
}

void idx_topk(const float* q, const float* w, const uint16_t* keys, const int32_t* n_cells, int max_cells, int heads, int dim, int k,
              int n_tok, float* scores, int32_t* cells, int32_t* n_sel, void* stream) {
    if (heads != 64 || dim != 128) throw std::invalid_argument("idx_topk: geometry");
    if (n_tok > 0) {
        if (max_cells > k)
            idx_score_mma_kernel<<<dim3(blocks(max_cells, kIdxTile), n_tok), 128, 0, S(stream)>>>(
                q, w, reinterpret_cast<const __half*>(keys), n_cells, max_cells, scores);
        idx_select_kernel<<<n_tok, kSelThreads, 0, S(stream)>>>(scores, n_cells, max_cells, k, cells, n_sel);
    }
    done(stream);
}

void f32_gemv(const float* w, const float* x, int rows, int k, int n_tok, float* y, void* stream) {
    if (k % 4) throw std::invalid_argument("f32_gemv: k");
    if (n_tok > 0) f32_gemv_kernel<<<dim3(blocks(rows, 8), n_tok), 256, 0, S(stream)>>>(w, x, rows, k, y);
    done(stream);
}

void swiglu_clamped(const float* gu, int n, int rows, float limit, float* h, void* stream) {
    const int64_t total = (int64_t) n * rows;
    if (total > 0) swiglu_kernel<<<blocks(total, 256), 256, 0, S(stream)>>>(gu, n, total, limit, h);
    done(stream);
}

void mul_sigmoid(float* x, const float* g, int64_t n, void* stream) {
    if (n > 0) mul_sigmoid_kernel<<<blocks(n, 256), 256, 0, S(stream)>>>(x, g, n);
    done(stream);
}

void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* table, unsigned long long* grp_ptr,
              int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok, uint32_t* counts, void* stream) {
    const int n_ent = rows * k;
    if (n_ent > 256 || n_expert > 512) throw std::invalid_argument("moe_plan: size");
    if (n_ent <= 32)
        moe_plan_small_kernel<<<1, 32, 0, S(stream)>>>(ids, n_ent, k, table, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, counts);
    else
        moe_plan_kernel<<<1, 256, 0, S(stream)>>>(ids, n_ent, k, n_expert, table, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, counts);
    done(stream);
}

size_t moe_sort_scratch(int pairs, int n_expert) { return (size_t) sort_blocks(pairs) * (size_t) n_expert * sizeof(int32_t); }

void moe_sort(const int32_t* ids, int rows, int k, int n_expert, void* scratch, int32_t* bounds, int32_t* tok, int32_t* dst,
              int32_t* exp, void* stream) {
    const int pairs = rows * k;
    if (pairs <= 0) {
        if (bounds) cudaMemsetAsync(bounds, 0, (size_t) (n_expert + 1) * sizeof(int32_t), S(stream));
        done(stream);
        return;
    }
    if (n_expert > kMaxSortExpert) throw std::invalid_argument("moe_sort: too many experts");
    cudaStream_t st = S(stream);
    int32_t* counts = static_cast<int32_t*>(scratch);
    const int nb = sort_blocks(pairs);
    sort_count_kernel<<<nb, 256, 0, st>>>(ids, pairs, n_expert, counts);
    sort_scan_kernel<<<1, kMaxSortExpert, 0, st>>>(counts, nb, n_expert, bounds);
    sort_scatter_kernel<<<nb, 32, 0, st>>>(ids, pairs, k, n_expert, counts, tok, dst, exp);
    done(stream);
}

void moe_combine(const float* experts, const float* w, const float* shared, int k, int n, int rows, float* out, void* stream) {
    if (rows > 0) moe_combine_kernel<<<dim3(blocks(n, 128), rows), 128, 0, S(stream)>>>(experts, w, shared, k, n, out);
    done(stream);
}

void set_rows(int32_t* pos, int32_t* n_cells, int n, int pos0, void* stream) {
    if (n > 0) set_rows_kernel<<<blocks(n, 256), 256, 0, S(stream)>>>(pos, n_cells, n, pos0);
    done(stream);
}

void advance_step(int32_t* pos, int32_t* n_cells, void* stream) {
    advance_step_kernel<<<1, 1, 0, S(stream)>>>(pos, n_cells);
    done(stream);
}

void greedy_advance(const int32_t* tok, int32_t* pos, int32_t* n_cells, int32_t* out, int32_t* count, int cap, void* stream) {
    advance_kernel<<<1, 1, 0, S(stream)>>>(tok, pos, n_cells, out, count, cap);
    done(stream);
}

void argmax_rows(const float* x, int64_t n, int rows, int32_t* ids, void* stream) {
    if (rows > 0) argmax_kernel<<<rows, 1024, 0, S(stream)>>>(x, n, ids);
    done(stream);
}

void argmax_prob_row(const float* x, int64_t n, int32_t* out_id, float* out_prob, void* stream) {
    argmax_prob_kernel<<<1, 1024, 0, S(stream)>>>(x, n, out_id, out_prob);
    done(stream);
}

}  // namespace basalt::models::gigachat4
