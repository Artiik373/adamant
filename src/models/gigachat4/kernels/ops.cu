// src/models/gigachat4/kernels/ops.cu - the GigaChat4 kernels: GatedNorm, the YaRN rope of the MLA and the indexer, and the
// logit softcap. A row's result does not depend on the other rows.
#include "basalt/models/gigachat4/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::models::gigachat4 {
namespace {

constexpr int kThreads = 512;
constexpr int kGate = 16;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

constexpr int kNormThreads = 512;   // 16 warps: one a gate row

__global__ void __launch_bounds__(kNormThreads) gated_norm_kernel(const float* __restrict__ x, int stride, const float* __restrict__ w,
                                                                  const float* __restrict__ up, const float* __restrict__ down, float eps,
                                                                  float gain, int n, float* out, int out_stride, bool add) {
    extern __shared__ __align__(16) float y[];   // [n] then the gate [16] and the reduction [16]
    float* g = y + n;
    float* red = g + kGate;
    const int n4 = n >> 2, wp = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t) blockIdx.x * stride);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* y4 = reinterpret_cast<float4*>(y);
    float ss = 0.f;
    for (int i = threadIdx.x; i < n4; i += kNormThreads) {
        const float4 v = xr[i];
        y4[i] = v;
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = warp_sum(ss);
    if (l == 0) red[wp] = ss;
    __syncthreads();
    ss = l < kNormThreads / 32 ? red[l] : 0.f;
    const float r = rsqrtf(warp_sum(ss) / n + eps);
    for (int i = threadIdx.x; i < n4; i += kNormThreads) {
        float4 v = y4[i];
        const float4 ww = w4[i];
        v.x *= r * ww.x, v.y *= r * ww.y, v.z *= r * ww.z, v.w *= r * ww.w;
        y4[i] = v;
    }
    __syncthreads();
    {
        const float4* ur = reinterpret_cast<const float4*>(up + (size_t) wp * n);
        float d = 0.f;
        for (int i = l; i < n4; i += 32) {
            const float4 u = ur[i], v = y4[i];
            d += u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
        }
        d = warp_sum(d);
        if (l == 0) g[wp] = d / (1.f + __expf(-d));
    }
    __syncthreads();
    float gl[kGate];
#pragma unroll
    for (int j = 0; j < kGate; ++j) gl[j] = g[j];
    float* orow = out + (size_t) blockIdx.x * out_stride;
    for (int i = threadIdx.x; i < n; i += kNormThreads) {
        const float4* dr = reinterpret_cast<const float4*>(down + (size_t) i * kGate);
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < kGate / 4; ++j) {
            const float4 v = __ldg(dr + j);
            d += v.x * gl[4 * j] + v.y * gl[4 * j + 1] + v.z * gl[4 * j + 2] + v.w * gl[4 * j + 3];
        }
        const float v = y[i] * gain / (1.f + __expf(-d));
        orow[i] = add ? orow[i] + v : v;
    }
}

__device__ __forceinline__ void rot(float x0, float x1, float f, int pos, float& o0, float& o1) {
    float s, c;
    sincosf((float) pos * f, &s, &c);
    o0 = x0 * c - x1 * s;
    o1 = x0 * s + x1 * c;
}

__device__ __forceinline__ void store_cache(float* cache, size_t cell, const float* k, const float* f, int p,
                                             const MlaShape& s, float*) {
    const int W = s.rank + s.rope, half = s.rope / 2;
    float* c = cache + cell * W;
    for (int i = threadIdx.x; i < s.rank; i += kThreads) c[i] = k[i];
    if (threadIdx.x < half) {
        rot(k[s.rank + 2 * threadIdx.x], k[s.rank + 2 * threadIdx.x + 1], f[threadIdx.x], p, c[s.rank + 2 * threadIdx.x],
            c[s.rank + 2 * threadIdx.x + 1]);
    }
}
__device__ __forceinline__ void store_cache(__half* cache, size_t cell, const float* k, const float* f, int p,
                                             const MlaShape& s, float*) {
    const int W = s.rank + s.rope, half = s.rope / 2;
    __half* c = cache + cell * W;
    for (int i = threadIdx.x; i < s.rank; i += kThreads) c[i] = __float2half_rn(k[i]);
    if (threadIdx.x < half) {
        float o0, o1;
        rot(k[s.rank + 2 * threadIdx.x], k[s.rank + 2 * threadIdx.x + 1], f[threadIdx.x], p, o0, o1);
        c[s.rank + 2 * threadIdx.x] = __float2half_rn(o0);
        c[s.rank + 2 * threadIdx.x + 1] = __float2half_rn(o1);
    }
}
__device__ __forceinline__ void store_cache(uint8_t* cache, size_t cell, const float* k, const float* f, int p,
                                             const MlaShape& s, float* red) {
    const int i = threadIdx.x, half = s.rope / 2;
    uint8_t* row = cache + cell * kFp8RowBytes;
    const float v = i < s.rank ? k[i] : 0.f;
    const float amax = warp_max(fabsf(v));
    if ((i & 31) == 0) red[i >> 5] = amax;
    __syncthreads();
    if (i < s.rank) {
        const int g = i >> 7;
        const float m = fmaxf(fmaxf(red[g * 4], red[g * 4 + 1]), fmaxf(red[g * 4 + 2], red[g * 4 + 3]));
        const float sc = exp2f(ceilf(log2f(fmaxf(m / 448.f, 1e-4f))));
        row[i] = __nv_cvt_float_to_fp8(v / sc, __NV_SATFINITE, __NV_E4M3);
        if ((i & 127) == 0) reinterpret_cast<float*>(row + 512)[g] = sc;
    }
    if (i < half) {
        float o0, o1;
        rot(k[s.rank + 2 * i], k[s.rank + 2 * i + 1], f[i], p, o0, o1);
        uint16_t* rp = reinterpret_cast<uint16_t*>(row + 528);
        rp[2 * i] = __bfloat16_as_ushort(__float2bfloat16_rn(o0));
        rp[2 * i + 1] = __bfloat16_as_ushort(__float2bfloat16_rn(o1));
    }
}

// Block (head h, row t); head == heads writes the row's cache row.
template <typename C>
__global__ void __launch_bounds__(kThreads) mla_prepare_kernel(const float* __restrict__ q, const float* __restrict__ q_abs,
                                                               const float* __restrict__ kv, const float* __restrict__ f, MlaShape s,
                                                               const int32_t* __restrict__ pos, int ring, float* __restrict__ qcur,
                                                               C* __restrict__ cache) {
    __shared__ float red[16];
    const int h = blockIdx.x, t = blockIdx.y, p = pos[t], W = s.rank + s.rope, half = s.rope / 2;
    if (h < s.heads) {
        const size_t hr = (size_t) t * s.heads + h;
        float* o = qcur + hr * W;
        const float* a = q_abs + hr * s.rank;
        for (int i = threadIdx.x; i < s.rank; i += kThreads) o[i] = a[i];
        if (threadIdx.x < half) {
            const float* r = q + hr * (s.nope + s.rope) + s.nope + 2 * threadIdx.x;
            rot(r[0], r[1], f[threadIdx.x], p, o[s.rank + 2 * threadIdx.x], o[s.rank + 2 * threadIdx.x + 1]);
        }
        return;
    }
    const float* k = kv + (size_t) t * W;
    const size_t cell = (size_t) (ring ? p % ring : p);
    store_cache(cache, cell, k, f, p, s, red);
}

template <typename C>
void prepare_impl(const float* q, const float* q_abs, const float* kv, const float* f, const MlaShape& s, const int32_t* pos, int n_tok,
                  int ring, float* qcur, C* cache, void* stream) {
    if (s.rope / 2 > kThreads || s.rank > kThreads) throw std::invalid_argument("gigachat4 mla_prepare: geometry");
    if (n_tok > 0)
        mla_prepare_kernel<<<dim3(s.heads + 1, n_tok), kThreads, 0, (cudaStream_t) stream>>>(q, q_abs, kv, f, s, pos, ring, qcur, cache);
    if (!stream) cudaStreamSynchronize(nullptr);
}

// One block of `dim` threads a row.
__global__ void idx_keys_kernel(const float* __restrict__ k, const float* __restrict__ w, float eps, const float* __restrict__ f, int dim,
                                int rope, const int32_t* __restrict__ pos, int ring, __half* __restrict__ keys) {
    __shared__ float y[512];
    __shared__ float red[32];
    const int t = blockIdx.x, d = threadIdx.x, p = pos[t], half = rope / 2;
    const float v = k[(size_t) t * dim + d];
    float ss = warp_sum(v * v);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    ss = 0.f;
    for (int i = 0; i < dim / 32; ++i) ss += red[i];
    y[d] = v * rsqrtf(ss / dim + eps) * w[d];
    __syncthreads();
    float o = y[d];
    if (d < half) o = y[d] * cosf(p * f[d]) - y[d + half] * sinf(p * f[d]);
    else if (d < rope) o = y[d] * cosf(p * f[d - half]) + y[d - half] * sinf(p * f[d - half]);
    keys[(size_t) (ring ? p % ring : p) * dim + d] = __float2half_rn(o);
}

// Block (head, row), rope / 2 threads.
__global__ void idx_rope_kernel(float* __restrict__ q, const float* __restrict__ f, int heads, int dim, int rope, const int32_t* __restrict__ pos) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x, half = rope / 2;
    float* r = q + ((size_t) t * heads + h) * dim;
    float o0, o1;
    rot(r[i], r[i + half], f[i], pos[t], o0, o1);
    r[i] = o0;
    r[i + half] = o1;
}

__global__ void softcap_kernel(float* __restrict__ x, int64_t n, float cap) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

__global__ void swiglu_split_clamped_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int n_ff, float limit,
                                            const uint8_t* __restrict__ blobs, size_t stride, size_t tail_off,
                                            const int32_t* __restrict__ bounds, int n, const int32_t* __restrict__ row_expert) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i - r * n_ff;
    float g = gu[r * 2 * n_ff + k], u = gu[r * 2 * n_ff + n_ff + k];
    int q = 0;
    if (row_expert) q = row_expert[r];
    else
        while (q + 1 < n && bounds[q + 1] <= r) ++q;
    const float* s = reinterpret_cast<const float*>(blobs + (size_t) q * stride + tail_off);
    g = fminf(g * s[0], limit);
    u = fminf(fmaxf(u * s[1], -limit), limit);
    h[i] = g / (1.f + __expf(-g)) * u * s[2];
}

__global__ void moe_combine_mapped_kernel(const float* __restrict__ ex, const int32_t* __restrict__ map,
                                          const float* __restrict__ w, const float* __restrict__ sh, int k, int n, float* __restrict__ out) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = sh[(size_t) t * n + i];
        for (int j = 0; j < k; ++j) v = fmaf(w[t * k + j], ex[(size_t) map[t * k + j] * n + i], v);
        out[(size_t) t * n + i] = v;
    }
}

__global__ void head_major_ids_kernel(int32_t* __restrict__ ids, int32_t* __restrict__ bounds, int heads, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < heads * rows) ids[i] = (i % rows) * heads + i / rows;
    if (i <= heads) bounds[i] = i * rows;
}

}  // namespace

void gated_norm(const float* x, int stride, const float* w, const float* up, const float* down, float eps, float gain, int n, int rows,
                float* out, int out_stride, void* stream, bool add) {
    const size_t smem = ((size_t) n + 2 * kGate) * sizeof(float);
    if (smem > 48 * 1024 || n % 4 || stride % 4 || out_stride % 4) throw std::invalid_argument("gated_norm: geometry");
    if (rows > 0)
        gated_norm_kernel<<<rows, kNormThreads, smem, (cudaStream_t) stream>>>(x, stride, w, up, down, eps, gain, n, out, out_stride, add);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void mla_prepare(const float* q, const float* q_abs, const float* kv, const float* inv_freq, const MlaShape& s, const int32_t* pos, int n_tok,
                 int ring, float* qcur, float* cache, void* stream) {
    prepare_impl(q, q_abs, kv, inv_freq, s, pos, n_tok, ring, qcur, cache, stream);
}

void mla_prepare(const float* q, const float* q_abs, const float* kv, const float* inv_freq, const MlaShape& s, const int32_t* pos, int n_tok,
                 int ring, float* qcur, uint16_t* cache, void* stream) {
    prepare_impl(q, q_abs, kv, inv_freq, s, pos, n_tok, ring, qcur, reinterpret_cast<__half*>(cache), stream);
}

void mla_prepare(const float* q, const float* q_abs, const float* kv, const float* inv_freq, const MlaShape& s, const int32_t* pos, int n_tok,
                 int ring, float* qcur, uint8_t* cache, void* stream) {
    prepare_impl(q, q_abs, kv, inv_freq, s, pos, n_tok, ring, qcur, cache, stream);
}

void idx_keys(const float* k, const float* norm_w, float eps, const float* inv_freq, int dim, int rope, const int32_t* pos, int n_tok, int ring,
              uint16_t* keys, void* stream) {
    if (dim % 32 || dim > 512 || rope > dim) throw std::invalid_argument("idx_keys: geometry");
    if (n_tok > 0)
        idx_keys_kernel<<<n_tok, dim, 0, (cudaStream_t) stream>>>(k, norm_w, eps, inv_freq, dim, rope, pos, ring, reinterpret_cast<__half*>(keys));
    if (!stream) cudaStreamSynchronize(nullptr);
}

void idx_rope_queries(float* q, const float* inv_freq, int heads, int dim, int rope, const int32_t* pos, int n_tok, void* stream) {
    if (n_tok > 0) idx_rope_kernel<<<dim3(heads, n_tok), rope / 2, 0, (cudaStream_t) stream>>>(q, inv_freq, heads, dim, rope, pos);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void softcap(float* x, int64_t n, float cap, void* stream) {
    if (n > 0) softcap_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, n, cap);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void swiglu_split_clamped(const float* gu, float* h, int64_t rows, int n_ff, float limit, const uint8_t* blobs, size_t stride,
                          size_t tail_off, const int32_t* bounds, int n, void* stream, const int32_t* row_expert) {
    const int64_t total = rows * n_ff;
    if (total > 0)
        swiglu_split_clamped_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
            gu, h, rows, n_ff, limit, blobs, stride, tail_off, bounds, n, row_expert);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void moe_combine_mapped(const float* experts, const int32_t* map, const float* w, const float* shared, int k, int n, int rows,
                        float* out, void* stream) {
    if (rows > 0)
        moe_combine_mapped_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(experts, map, w, shared, k, n, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void head_major_ids(int32_t* ids, int32_t* bounds, int heads, int rows, void* stream) {
    const int n = std::max(heads * rows, heads + 1);
    if (n > 0) head_major_ids_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(ids, bounds, heads, rows);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::gigachat4
