// src/models/hy4/kernels/decode.cu - see include/basalt/models/hy4/kernels.hpp.
#include "basalt/models/hy4/geometry.hpp"
#include "basalt/models/hy4/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>

#include <cub/block/block_scan.cuh>

namespace basalt::models::hy4 {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sums N values over a block of `threads` threads (a multiple of 32, at most 1024); every thread gets the totals.
template <int N>
__device__ void block_sums(float (&v)[N], float* smem /* [32][N] */) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < N; ++i) smem[warp * N + i] = v[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float s = 0.f;
        for (int w = 0; w < warps; ++w) s += smem[w * N + i];
        v[i] = s;
    }
    __syncthreads();
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

// ---- hyper-connections ----------------------------------------------------------------------------------------

constexpr int kMixThreads = 256;

// block j < m: the dot of mix row j with the token's flat streams; block m: their sum of squares
__global__ void hc_dots_kernel(const float* __restrict__ streams, const float* __restrict__ fn, int m,
                               float* __restrict__ dots) {
    __shared__ float red[32];
    const int j = blockIdx.x, t = blockIdx.y;
    const float4* x = reinterpret_cast<const float4*>(streams + (size_t) t * kHcFlat);
    const float4* w = reinterpret_cast<const float4*>(fn + (size_t) j * kHcFlat);
    float acc[1] = {0.f};
    for (int i = threadIdx.x; i < kHcFlat / 4; i += blockDim.x) {
        const float4 v = x[i];
        if (j < m) {
            const float4 f = w[i];
            acc[0] += f.x * v.x + f.y * v.y + f.z * v.z + f.w * v.w;
        } else {
            acc[0] += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    block_sums<1>(acc, red);
    if (threadIdx.x == 0) dots[t * (kHcMixes + 1) + j] = acc[0];
}

__global__ void hc_collapse_kernel(const float* __restrict__ streams, const float* __restrict__ dots,
                                   const float* __restrict__ base, const float* __restrict__ scale, int m,
                                   float* __restrict__ pre, float* __restrict__ post, float* __restrict__ y) {
    const int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    const float* dt = dots + t * (kHcMixes + 1);
    const float r = rsqrtf(dt[m] / (float) kHcFlat + kRmsEps);
    float w[kHc];
#pragma unroll
    for (int c = 0; c < kHc; ++c) w[c] = sigmoidf_(dt[c] * r * scale[0] + base[c]) + kHcEps;
    if (blockIdx.x == 0 && threadIdx.x < kHc) {
        const int c = threadIdx.x;
        if (pre != nullptr) pre[t * kHc + c] = w[c];
        if (post != nullptr && m == kHcMixes)
            post[t * kHc + c] = kHcMagnitude * sigmoidf_(dt[kHc + c] * r * scale[1] + base[kHc + c]) + kHcEps;
    }
    if (d >= kEmbd) return;
    const float* x = streams + (size_t) t * kHcFlat;
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < kHc; ++c) s += w[c] * x[c * kEmbd + d];
    y[(size_t) t * kEmbd + d] = s;
}

__global__ void hc_apply_kernel(float* __restrict__ streams, const float* __restrict__ y, const float* __restrict__ post,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = i / kHcFlat, s = (i / kEmbd) % kHc, d = i % kEmbd;
    streams[i] += post[t * kHc + s] * y[(size_t) t * kEmbd + d];
}

__global__ void hc_expand_kernel(const float* __restrict__ x, float* __restrict__ streams, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = i / kHcFlat, d = i % kEmbd;
    streams[i] = x[(size_t) t * kEmbd + d];
}

// ---- norms and RoPE ---------------------------------------------------------------------------------------------

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, int width, int stride,
                                float eps, float* __restrict__ out) {
    __shared__ float red[32];
    const float* row = x + (size_t) blockIdx.x * stride;
    float acc[1] = {0.f};
    for (int i = threadIdx.x; i < width; i += blockDim.x) acc[0] += row[i] * row[i];
    block_sums<1>(acc, red);
    const float r = rsqrtf(acc[0] / (float) width + eps);
    float* o = out + (size_t) blockIdx.x * stride;
    for (int i = threadIdx.x; i < width; i += blockDim.x) o[i] = row[i] * r * (w != nullptr ? w[i] : 1.f);
}

struct RopeFreq {
    double inv[kRope / 2];
};

__global__ void rope_gptj_kernel(float* __restrict__ rows, int n_rows, int stride, int offset, int dim, int rows_per_pos,
                                 const int32_t* __restrict__ pos, RopeFreq f) {
    const int pairs = dim / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * pairs) return;
    const int r = i / pairs, k = i % pairs;
    // the angle in fp64, reduced to one turn: positions reach 1M, where an fp32 product loses a milliradian
    const double ang = (double) pos[r / rows_per_pos] * f.inv[k];
    float sn, cs;
    sincosf((float) (ang - 6.283185307179586 * floor(ang / 6.283185307179586)), &sn, &cs);
    float* p = rows + (size_t) r * stride + offset + 2 * k;
    const float a = p[0], b = p[1];
    p[0] = a * cs - b * sn;
    p[1] = a * sn + b * cs;
}

// ---- router -----------------------------------------------------------------------------------------------------

// One thread per expert. Each warp takes the top kExpertsUsed of its 32 experts in order, then warp 0 merges the
// sorted lists: the same picks, in the same order, as a block-wide argmax per pick.
__global__ void route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int32_t* __restrict__ ids,
                             float* __restrict__ weights) {
    constexpr int kWarps = kExperts / 32;
    __shared__ float score[kExperts];
    __shared__ float cand_v[kWarps][kExpertsUsed];
    __shared__ int cand_i[kWarps][kExpertsUsed];
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float s = sigmoidf_(logits[(size_t) t * kExperts + e]);
    score[e] = s;
    float key = s + (bias != nullptr ? bias[e] : 0.f);
    for (int k = 0; k < kExpertsUsed; ++k) {
        float v = key;
        int id = e;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, id, o);
            if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
        }
        if (lane == 0) { cand_v[warp][k] = v; cand_i[warp][k] = id; }
        if (id == e) key = -FLT_MAX;
    }
    __syncthreads();
    if (warp != 0) return;
    int next = 0;   // lane l < kWarps: the head of warp l's list
    float picked_sum = 0.f;
    for (int k = 0; k < kExpertsUsed; ++k) {
        const bool live = lane < kWarps && next < kExpertsUsed;
        float v = live ? cand_v[lane][next] : -FLT_MAX;
        int id = live ? cand_i[lane][next] : INT_MAX;
        const int mine = id;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, id, o);
            if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
        }
        if (mine == id && live) ++next;
        if (lane == 0) {
            ids[t * kExpertsUsed + k] = id;
            weights[t * kExpertsUsed + k] = score[id];
            picked_sum += score[id];
        }
    }
    picked_sum = __shfl_sync(0xffffffffu, picked_sum, 0);
    __syncwarp();   // lane 0's weights are visible to the warp
    if (lane < kExpertsUsed) weights[t * kExpertsUsed + lane] *= kRoutedScale / (picked_sum + 1e-20f);
}

// ---- Engram -----------------------------------------------------------------------------------------------------

// one warp per head
__global__ void engram_rows_kernel(const float* __restrict__ rows, const float* __restrict__ head_norm, float eps,
                                   float* __restrict__ out) {
    const int t = blockIdx.x, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int kPer = kEngramHeadDim / 32;   // 3
    const float* x = rows + ((size_t) t * kEngramHeads + h) * kEngramHeadDim;
    float v[kPer];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        v[i] = x[lane + 32 * i];
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = warp_max(amax);
    const float sc = fmaxf(amax, 1e-12f) / 448.f;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const float q = fminf(fmaxf(v[i] / sc, -448.f), 448.f);
        v[i] = float(__nv_fp8_e4m3(q)) * sc;
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / (float) kEngramHeadDim + eps);
    float* o = out + (size_t) t * kEngramDim + h * kEngramHeadDim;
#pragma unroll
    for (int i = 0; i < kPer; ++i) o[lane + 32 * i] = v[i] * r * head_norm[h * kEngramHeadDim + lane + 32 * i];
}

constexpr int kEngramThreads = 256;

__global__ void engram_apply_kernel(const float* __restrict__ key, const float* __restrict__ value, EngramWeights w,
                                    float* __restrict__ streams) {
    __shared__ float red[32 * 3];
    __shared__ float gate_s[kHc];
    __shared__ float v_r;
    const int t = blockIdx.x;
    float* st = streams + (size_t) t * kHcFlat;
    for (int s = 0; s < kHc; ++s) {
        const float* k = key + ((size_t) t * kHc + s) * kEmbd;
        const float* kb = w.key_bias + s * kEmbd;
        const float* nk = w.norm_key + s * kEmbd;
        const float* nq = w.norm_query + s * kEmbd;
        const float* q = st + s * kEmbd;
        float acc[3] = {};
        for (int d = threadIdx.x; d < kEmbd; d += blockDim.x) {
            const float kv = k[d] + kb[d], qv = q[d];
            acc[0] += kv * kv;
            acc[1] += qv * qv;
            acc[2] += kv * nk[d] * qv * nq[d];
        }
        block_sums<3>(acc, red);
        if (threadIdx.x == 0) {
            const float rk = rsqrtf(acc[0] / (float) kEmbd + w.rms_eps), rq = rsqrtf(acc[1] / (float) kEmbd + w.rms_eps);
            const float g = acc[2] * rk * rq * rsqrtf((float) kEmbd);
            const float sg = g > 0.f ? 1.f : (g < 0.f ? -1.f : 0.f);
            gate_s[s] = sigmoidf_(sqrtf(fmaxf(fabsf(g), 1e-6f)) * sg);
        }
    }
    {
        float acc[1] = {0.f};
        for (int d = threadIdx.x; d < kEmbd; d += blockDim.x) {
            const float v = value[(size_t) t * kEmbd + d] + w.value_bias[d];
            acc[0] += v * v;
        }
        block_sums<1>(acc, red);
        if (threadIdx.x == 0) v_r = rsqrtf(acc[0] / (float) kEmbd + w.engram_eps);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < kEmbd; d += blockDim.x) {
        const float main = w.pre_gamma[d] * (value[(size_t) t * kEmbd + d] + w.value_bias[d]) * v_r;
#pragma unroll
        for (int s = 0; s < kHc; ++s) st[s * kEmbd + d] += gate_s[s] * main + w.per_hc_scale[s * kEmbd + d] * main;
    }
}

// ---- MLA --------------------------------------------------------------------------------------------------------

constexpr int kKvbRows = kNope + kVHead;   // 448 rows of kv_b per head

__global__ void mla_absorb_kernel(const float* __restrict__ q, const __half* __restrict__ kv_b, int heads,
                                  float* __restrict__ q_lat) {
    const int h = blockIdx.x, t = blockIdx.y;
    __shared__ float qn[kNope];
    const float* qh = q + ((size_t) t * heads + h) * kQkHead;
    for (int d = threadIdx.x; d < kNope; d += blockDim.x) qn[d] = qh[d];
    __syncthreads();
    const __half* wk = kv_b + (size_t) h * kKvbRows * kKvRank;
    const int c = blockIdx.z * blockDim.x + threadIdx.x;   // the z blocks split a head's columns: more CTAs in flight
    float s = 0.f;
    for (int d = 0; d < kNope; ++d) s += qn[d] * __half2float(wk[(size_t) d * kKvRank + c]);
    q_lat[((size_t) t * heads + h) * kKvRank + c] = s;
}

constexpr int kAttnWarps = 4;
constexpr int kLatPerLane = kKvRank / 32;   // 16
constexpr int kRopePerLane = kRope / 32;    // 2

// One block per (head, row); each warp walks a quarter of the keys with an online softmax, then the warps merge.
__global__ void mla_attend_kernel(const float* __restrict__ q_lat, const float* __restrict__ q, LatentCache cache,
                                  const int32_t* __restrict__ lo, const int32_t* __restrict__ hi,
                                  const int32_t* __restrict__ sel, const int32_t* __restrict__ sel_n, int sel_stride,
                                  const float* __restrict__ sink, int heads, float scale, float* __restrict__ o_lat) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* ql = q_lat + ((size_t) t * heads + h) * kKvRank;
    const float* qp = q + ((size_t) t * heads + h) * kQkHead + kNope;
    float qv[kLatPerLane], qr[kRopePerLane];
#pragma unroll
    for (int i = 0; i < kLatPerLane; ++i) qv[i] = ql[lane + 32 * i];
#pragma unroll
    for (int i = 0; i < kRopePerLane; ++i) qr[i] = qp[lane + 32 * i];
    const int end = hi[t];
    const bool use_sel = sel != nullptr && sel_n[t] >= 0;
    const int n = use_sel ? sel_n[t] : end - lo[t];
    float m = -FLT_MAX, l = 0.f, acc[kLatPerLane] = {};
    for (int j = warp; j < n; j += kAttnWarps) {
        const int p = use_sel ? sel[(size_t) t * sel_stride + j] : lo[t] + j;
        if (p < 0 || p >= end) continue;   // a selected position past the row's own is masked
        const int slot = p % cache.cap;
        const __half* ck = reinterpret_cast<const __half*>(cache.ckv) + (size_t) slot * kKvRank;
        const __half* kp = reinterpret_cast<const __half*>(cache.kpe) + (size_t) slot * kRope;
        float kv[kLatPerLane];
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < kLatPerLane; ++i) {
            kv[i] = __half2float(ck[lane + 32 * i]);
            s += qv[i] * kv[i];
        }
#pragma unroll
        for (int i = 0; i < kRopePerLane; ++i) s += qr[i] * __half2float(kp[lane + 32 * i]);
        s = warp_sum(s) * scale;
        const float mn = fmaxf(m, s);
        const float a = __expf(m - mn), b = __expf(s - mn);
        l = l * a + b;
#pragma unroll
        for (int i = 0; i < kLatPerLane; ++i) acc[i] = acc[i] * a + b * kv[i];
        m = mn;
    }
    __shared__ float sm_m[kAttnWarps], sm_l[kAttnWarps];
    __shared__ float sm_acc[kAttnWarps][kKvRank];
    if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
#pragma unroll
    for (int i = 0; i < kLatPerLane; ++i) sm_acc[warp][lane + 32 * i] = acc[i];
    __syncthreads();
    const float sk = sink != nullptr ? sink[h] : -FLT_MAX;
    float mm = sk;
    for (int w = 0; w < kAttnWarps; ++w) mm = fmaxf(mm, sm_m[w]);
    float denom = sink != nullptr ? __expf(sk - mm) : 0.f;
    for (int w = 0; w < kAttnWarps; ++w) denom += sm_l[w] * __expf(sm_m[w] - mm);
    const float inv = denom > 0.f ? 1.f / denom : 0.f;
    for (int c = threadIdx.x; c < kKvRank; c += blockDim.x) {
        float s = 0.f;
        for (int w = 0; w < kAttnWarps; ++w) s += sm_acc[w][c] * __expf(sm_m[w] - mm);
        o_lat[((size_t) t * heads + h) * kKvRank + c] = s * inv;
    }
}

constexpr int kOutThreads = 128;
constexpr int kOutRows = 32;   // value rows a CTA; the z blocks split a head's rows, so more CTAs are in flight

__global__ void mla_value_out_kernel(const float* __restrict__ o_lat, const __half* __restrict__ kv_b,
                                     const float* __restrict__ gate, int heads, float* __restrict__ out) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* ol = o_lat + ((size_t) t * heads + h) * kKvRank;
    float ov[kLatPerLane];
#pragma unroll
    for (int i = 0; i < kLatPerLane; ++i) ov[i] = ol[lane + 32 * i];
    const __half* wv = kv_b + ((size_t) h * kKvbRows + kNope) * kKvRank;
    constexpr int kStep = kOutThreads / 32;
    for (int v = blockIdx.z * kOutRows + warp; v < (blockIdx.z + 1) * kOutRows; v += 2 * kStep) {
        // two rows at a time, each summed in the same order as one alone
        const __half* r0 = wv + (size_t) v * kKvRank;
        const __half* r1 = r0 + (size_t) kStep * kKvRank;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int i = 0; i < kLatPerLane; ++i) {
            s0 += ov[i] * __half2float(r0[lane + 32 * i]);
            s1 += ov[i] * __half2float(r1[lane + 32 * i]);
        }
        s0 = warp_sum(s0);
        s1 = warp_sum(s1);
        if (lane == 0) {
            const size_t o = ((size_t) t * heads + h) * kVHead + v;
            out[o] = s0 * sigmoidf_(gate[o]);
            out[o + kStep] = s1 * sigmoidf_(gate[o + kStep]);
        }
    }
}

// ---- helpers ----------------------------------------------------------------------------------------------------

// one warp per output row
__global__ void gemv_f32_kernel(const float* __restrict__ w, const float* __restrict__ x, int n_in, int n_out,
                                float* __restrict__ y) {
    const int lane = threadIdx.x & 31, o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), t = blockIdx.y;
    if (o >= n_out) return;
    const float* row = w + (size_t) o * n_in;
    const float* xt = x + (size_t) t * n_in;
    float s = 0.f;
    for (int i = lane; i < n_in; i += 32) s += row[i] * xt[i];
    s = warp_sum(s);
    if (lane == 0) y[(size_t) t * n_out + o] = s;
}

__global__ void embed_bf16_kernel(const uint16_t* __restrict__ table, const int32_t* __restrict__ ids, int width,
                                  float* __restrict__ out) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    out[(size_t) t * width + i] = __uint_as_float((uint32_t) table[(size_t) ids[t] * width + i] << 16);
}

__global__ void expert_sum_kernel(const float* __restrict__ rows, const float* __restrict__ weights, int width,
                                  float* __restrict__ out) {
    const int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= width) return;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < kExpertsUsed; ++k)
        s += weights[t * kExpertsUsed + k] * rows[((size_t) t * kExpertsUsed + k) * width + d];
    out[(size_t) t * width + d] = s;
}

__global__ void latent_store_kernel(const float* __restrict__ kva, const int32_t* __restrict__ pos, __half* ckv,
                                    __half* kpe, int cap) {
    const int t = blockIdx.x;
    const int slot = pos[t] % cap;
    for (int i = threadIdx.x; i < kKeyDim; i += blockDim.x) {
        const float v = kva[(size_t) t * kKeyDim + i];
        if (i < kKvRank) ckv[(size_t) slot * kKvRank + i] = __float2half(v);
        else kpe[(size_t) slot * kRope + i - kKvRank] = __float2half(v);
    }
}

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, int n, float* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

__global__ void add_scaled_kernel(float* __restrict__ y, const float* __restrict__ g, const float* __restrict__ f, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += g[i] * f[i];
}

constexpr int kArgmaxThreads = 1024;

__global__ void argmax_kernel(const float* __restrict__ x, int n, int32_t* __restrict__ out) {
    __shared__ float sv[kArgmaxThreads / 32];
    __shared__ int si[kArgmaxThreads / 32];
    float bv = -FLT_MAX;
    int bi = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > bv) { bv = x[i]; bi = i; }   // strided: a thread's first maximum is its lowest index
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, o);
        const int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < kArgmaxThreads / 32; ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        *out = bi;
    }
}

__global__ void sel_with_block_kernel(const int32_t* __restrict__ src, const int32_t* __restrict__ src_n,
                                      const int32_t* __restrict__ p0, int block, int32_t* __restrict__ dst,
                                      int32_t* __restrict__ dst_n) {
    const int t = blockIdx.x, n = *src_n;
    int32_t* row = dst + (size_t) t * kDflySelStride;
    if (n < 0) {
        if (threadIdx.x == 0) dst_n[t] = -1;
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] = src[i];
    for (int i = threadIdx.x; i < block; i += blockDim.x) row[n + i] = *p0 + i;
    if (threadIdx.x == 0) dst_n[t] = n + block;
}

// ---- compressed indexer -----------------------------------------------------------------------------------------

// one block per group, one thread per output feature
__device__ void indexer_pool_group(const float* __restrict__ token_keys, const IndexerPool& w, int g,
                                   float* __restrict__ group_keys) {
    __shared__ float key[kIdxRatio][kIdxDim];
    __shared__ float red[32];
    const int o = threadIdx.x;
    for (int i = 0; i < kIdxRatio; ++i) key[i][o] = token_keys[((size_t) g * kIdxRatio + i) * kIdxDim + o];
    __syncthreads();
    float val[kIdxRatio], gate[kIdxRatio], mx = -FLT_MAX;
    for (int i = 0; i < kIdxRatio; ++i) {
        float v = 0.f, gt = 0.f;
        for (int d = 0; d < kIdxDim; ++d) {
            v += w.kv_w[o * kIdxDim + d] * key[i][d];
            gt += w.gate_w[o * kIdxDim + d] * key[i][d];
        }
        val[i] = v;
        gate[i] = gt + w.ape[i * kIdxDim + o];
        mx = fmaxf(mx, gate[i]);
    }
    float den = 0.f, pooled = 0.f;
    for (int i = 0; i < kIdxRatio; ++i) den += __expf(gate[i] - mx);
    for (int i = 0; i < kIdxRatio; ++i) pooled += val[i] * __expf(gate[i] - mx) / den;
    float acc[1] = {pooled * pooled};
    block_sums<1>(acc, red);
    float* out = group_keys + (size_t) g * kIdxDim;
    out[o] = pooled * rsqrtf(acc[0] / (float) kIdxDim + w.eps) * w.key_norm[o];
    __syncthreads();
    // RoPE of the last kRope dims at the group's first position (thread o < kRope / 2 rotates one pair)
    if (o < kRope / 2) {
        const double ang = (double) (g * kIdxRatio) * pow((double) kRopeThetaSparse, -2.0 * o / (double) kRope);
        float sn, cs;
        sincosf((float) (ang - 6.283185307179586 * floor(ang / 6.283185307179586)), &sn, &cs);
        float* p = out + (kIdxDim - kRope) + 2 * o;
        const float a = p[0], b = p[1];
        p[0] = a * cs - b * sn;
        p[1] = a * sn + b * cs;
    }
}

__global__ void indexer_pool_kernel(const float* __restrict__ token_keys, IndexerPool w, int g0,
                                    float* __restrict__ group_keys) {
    indexer_pool_group(token_keys, w, g0 + blockIdx.x, group_keys);
}

// the group position pos[0] completes, if it completes one
__global__ void indexer_pool_at_kernel(const float* __restrict__ token_keys, IndexerPool w, const int32_t* __restrict__ pos,
                                       float* __restrict__ group_keys) {
    if ((pos[0] + 1) % kIdxRatio != 0) return;
    indexer_pool_group(token_keys, w, pos[0] / kIdxRatio, group_keys);
}

__global__ void store_row_kernel(const float* __restrict__ src, float* __restrict__ dst, int width,
                                 const int32_t* __restrict__ pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) dst[(size_t) pos[0] * width + i] = src[i];
}

constexpr int kScoreWarps = 8;

// logits[t][g] for the groups before the row's own; the others are -inf
__global__ void indexer_score_kernel(const float* __restrict__ q, const float* __restrict__ hw,
                                     const float* __restrict__ group_keys, int groups_arg, const int32_t* __restrict__ pos,
                                     float* __restrict__ logits, int stride, bool from_pos) {
    const int groups = from_pos ? (pos[blockIdx.y] + 1) / kIdxRatio : groups_arg;
    if (from_pos && pos[blockIdx.y] + 1 <= kIdxTopk) return;
    __shared__ float qs[kIdxHeads][kIdxDim];
    __shared__ float ws[kIdxHeads];
    const int t = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < kIdxHeads * kIdxDim; i += blockDim.x)
        qs[i / kIdxDim][i % kIdxDim] = q[(size_t) t * kIdxHeads * kIdxDim + i];
    if (threadIdx.x < kIdxHeads) ws[threadIdx.x] = hw[t * kIdxHeads + threadIdx.x] * rsqrtf((float) (kIdxHeads * kIdxDim));
    __syncthreads();
    const int own = pos[t] / kIdxRatio;
    for (int g = blockIdx.x * kScoreWarps + warp; g < groups; g += gridDim.x * kScoreWarps) {
        float k[kIdxDim / 32];
#pragma unroll
        for (int i = 0; i < kIdxDim / 32; ++i) k[i] = group_keys[(size_t) g * kIdxDim + lane + 32 * i];
        float logit = 0.f;
#pragma unroll
        for (int h = 0; h < kIdxHeads; ++h) {
            float s = 0.f;
#pragma unroll
            for (int i = 0; i < kIdxDim / 32; ++i) s += qs[h][lane + 32 * i] * k[i];
            s = warp_sum(s);
            logit += ws[h] * fmaxf(s, 0.f);
        }
        if (lane == 0) logits[(size_t) t * stride + g] = g < own ? logit : -FLT_MAX;
    }
}

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

constexpr int kSelThreads = 1024;

// One block per row: the k-th largest key by a 4 x 8-bit radix select, then the groups above it and the lowest-numbered
// ones equal to it, in ascending group order, as positions.
__global__ void indexer_select_kernel(const float* __restrict__ logits, int groups_arg, const int32_t* __restrict__ pos,
                                      int32_t* __restrict__ sel, int32_t* __restrict__ sel_n, int stride, bool from_pos) {
    const int groups = from_pos ? (pos[blockIdx.x] + 1) / kIdxRatio : groups_arg;
    if (from_pos && pos[blockIdx.x] + 1 <= kIdxTopk) {   // the layer attends every earlier position
        if (threadIdx.x == 0) sel_n[blockIdx.x] = -1;
        return;
    }
    using Scan = cub::BlockScan<int, kSelThreads>;
    __shared__ typename Scan::TempStorage scan_tmp;
    __shared__ unsigned hist[256];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need, s_count;
    const int t = blockIdx.x;
    const float* lg = logits + (size_t) t * stride;
    const int own = pos[t] / kIdxRatio;
    const int valid = min(own, groups);
    const int k = min(kIdxGroupTopk - 1, valid);
    int32_t* out = sel + (size_t) t * kIdxGroupTopk * kIdxRatio;
    if (threadIdx.x == 0) { s_prefix = 0; s_mask = 0; s_need = k; s_count = 0; }
    __syncthreads();
    if (k > 0) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            const uint32_t prefix = s_prefix, mask = s_mask;
            for (int g = threadIdx.x; g < valid; g += blockDim.x) {
                const uint32_t u = order_key(lg[g]);
                if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int need = s_need, bin = 255;
                for (; bin > 0; --bin) {
                    if ((int) hist[bin] >= need) break;
                    need -= (int) hist[bin];
                }
                s_need = need;   // of the keys sharing the prefix so far, how many are still needed
                s_prefix = prefix | ((uint32_t) bin << shift);
                s_mask = mask | (255u << shift);
            }
            __syncthreads();
        }
    }
    const uint32_t thr = s_prefix;   // the k-th key; s_need of the keys equal to it are taken, lowest groups first
    const int need_eq = s_need;
    int base_eq = 0;
    for (int g0 = 0; g0 < valid && k > 0; g0 += kSelThreads) {
        const int g = g0 + (int) threadIdx.x;
        const uint32_t u = g < valid ? order_key(lg[g]) : 0u;
        const int gt = g < valid && u > thr, eq = g < valid && u == thr;
        int eq_rank, eq_total;
        Scan(scan_tmp).ExclusiveSum(eq, eq_rank, eq_total);
        __syncthreads();
        const int take = gt || (eq && base_eq + eq_rank < need_eq);
        int rank, total;
        Scan(scan_tmp).ExclusiveSum(take, rank, total);
        __syncthreads();
        if (take)
            for (int i = 0; i < kIdxRatio; ++i) out[(s_count + rank) * kIdxRatio + i] = g * kIdxRatio + i;
        __syncthreads();
        if (threadIdx.x == 0) s_count += total;
        base_eq += eq_total;
        __syncthreads();
    }
    if (threadIdx.x == 0) {   // the row's own group, up to the row's position
        int n = s_count * kIdxRatio;
        for (int i = 0; i < kIdxRatio; ++i) {
            const int p = own * kIdxRatio + i;
            if (p <= pos[t]) out[n++] = p;
        }
        sel_n[t] = n;
    }
}

}  // namespace

void gemv_f32(const float* w, const float* x, int n_in, int n_out, int n_rows, float* y, void* stream) {
    if (n_rows <= 0 || n_out <= 0) return;
    gemv_f32_kernel<<<dim3((n_out + 7) / 8, n_rows), 256, 0, (cudaStream_t) stream>>>(w, x, n_in, n_out, y);
}

void embed_bf16(const uint16_t* table, const int32_t* ids, int width, int n_rows, float* out, void* stream) {
    if (n_rows <= 0) return;
    embed_bf16_kernel<<<dim3((width + 255) / 256, n_rows), 256, 0, (cudaStream_t) stream>>>(table, ids, width, out);
}

void expert_sum(const float* rows, const float* weights, int width, int n_rows, float* out, void* stream) {
    if (n_rows <= 0) return;
    expert_sum_kernel<<<dim3((width + 255) / 256, n_rows), 256, 0, (cudaStream_t) stream>>>(rows, weights, width, out);
}

void latent_store(const float* kva, const int32_t* pos, int n_rows, uint16_t* ckv, uint16_t* kpe, int cap, void* stream) {
    if (n_rows <= 0) return;
    latent_store_kernel<<<n_rows, 256, 0, (cudaStream_t) stream>>>(kva, pos, reinterpret_cast<__half*>(ckv),
                                                                    reinterpret_cast<__half*>(kpe), cap);
}

void add_rows(const float* a, const float* b, int n, float* y, void* stream) {
    if (n <= 0) return;
    add_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(a, b, n, y);
}

void add_scaled(float* y, const float* g, const float* f, int n, void* stream) {
    if (n <= 0) return;
    add_scaled_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(y, g, f, n);
}

void argmax_row(const float* x, int n, int32_t* out, void* stream) {
    argmax_kernel<<<1, kArgmaxThreads, 0, (cudaStream_t) stream>>>(x, n, out);
}

void sel_with_block(const int32_t* src, const int32_t* src_n, const int32_t* p0, int block, int n_rows, int32_t* dst,
                    int32_t* dst_n, void* stream) {
    if (n_rows <= 0) return;
    sel_with_block_kernel<<<n_rows, 256, 0, (cudaStream_t) stream>>>(src, src_n, p0, block, dst, dst_n);
}

void indexer_pool(const float* token_keys, const IndexerPool& w, int g0, int g1, float* group_keys, void* stream) {
    if (g1 <= g0) return;
    indexer_pool_kernel<<<g1 - g0, kIdxDim, 0, (cudaStream_t) stream>>>(token_keys, w, g0, group_keys);
}

size_t indexer_scratch_floats(int groups, int n_rows) { return (size_t) std::max(groups, 1) * (size_t) n_rows; }

void indexer_select(const float* q, const float* hw, const float* group_keys, int groups, const int32_t* pos,
                    int n_rows, float* scratch, int32_t* sel, int32_t* sel_n, void* stream) {
    if (n_rows <= 0) return;
    if (groups > 0) {
        const int blocks = std::min(1024, (groups + kScoreWarps - 1) / kScoreWarps);
        indexer_score_kernel<<<dim3(blocks, n_rows), kScoreWarps * 32, 0, (cudaStream_t) stream>>>(
            q, hw, group_keys, groups, pos, scratch, std::max(groups, 1), false);
    }
    indexer_select_kernel<<<n_rows, kSelThreads, 0, (cudaStream_t) stream>>>(scratch, groups, pos, sel, sel_n,
                                                                             std::max(groups, 1), false);
}

void indexer_select_at(const float* q, const float* hw, const float* group_keys, int max_groups, const int32_t* pos,
                       int n_rows, float* scratch, int32_t* sel, int32_t* sel_n, void* stream) {
    if (n_rows <= 0 || max_groups <= 0) return;
    const int blocks = std::min(1024, (max_groups + kScoreWarps - 1) / kScoreWarps);
    indexer_score_kernel<<<dim3(blocks, n_rows), kScoreWarps * 32, 0, (cudaStream_t) stream>>>(
        q, hw, group_keys, 0, pos, scratch, max_groups, true);
    indexer_select_kernel<<<n_rows, kSelThreads, 0, (cudaStream_t) stream>>>(scratch, 0, pos, sel, sel_n, max_groups, true);
}

void indexer_pool_at(const float* token_keys, const IndexerPool& w, const int32_t* pos, float* group_keys, void* stream) {
    indexer_pool_at_kernel<<<1, kIdxDim, 0, (cudaStream_t) stream>>>(token_keys, w, pos, group_keys);
}

void store_row(const float* src, float* dst, int width, const int32_t* pos, void* stream) {
    store_row_kernel<<<(width + 255) / 256, 256, 0, (cudaStream_t) stream>>>(src, dst, width, pos);
}

void hc_mix(const float* streams, const HcMix& w, int n_rows, float* pre, float* post, float* y, void* stream) {
    if (n_rows <= 0) return;
    static thread_local float* dots = nullptr;
    static thread_local int cap = 0;
    if (cap < n_rows) {
        if (dots != nullptr) cudaFree(dots);
        cap = std::max(n_rows, kMaxT);
        cudaMalloc(&dots, (size_t) cap * (kHcMixes + 1) * sizeof(float));
    }
    hc_dots_kernel<<<dim3(w.m + 1, n_rows), kMixThreads, 0, (cudaStream_t) stream>>>(streams, w.fn, w.m, dots);
    hc_collapse_kernel<<<dim3((kEmbd + 255) / 256, n_rows), 256, 0, (cudaStream_t) stream>>>(streams, dots, w.base,
                                                                                            w.scale, w.m, pre, post, y);
}

void hc_apply(float* streams, const float* y, const float* post, int n_rows, void* stream) {
    const int n = n_rows * kHcFlat;
    if (n <= 0) return;
    hc_apply_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(streams, y, post, n);
}

void hc_expand(const float* x, float* streams, int n_rows, void* stream) {
    const int n = n_rows * kHcFlat;
    if (n <= 0) return;
    hc_expand_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, streams, n);
}

void rms_norm_rows(const float* x, const float* w, int width, int n_rows, int row_stride, float eps, float* out,
                   void* stream) {
    if (n_rows <= 0) return;
    rms_norm_kernel<<<n_rows, 256, 0, (cudaStream_t) stream>>>(x, w, width, row_stride, eps, out);
}

void rope_gptj(float* rows, int n_rows, int row_stride, int offset, int dim, int rows_per_pos, const int32_t* pos,
               float theta, void* stream) {
    const int n = n_rows * (dim / 2);
    if (n <= 0 || dim > kRope) return;
    RopeFreq f{};
    for (int k = 0; k < dim / 2; ++k) f.inv[k] = std::pow((double) theta, -2.0 * k / (double) dim);
    rope_gptj_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(rows, n_rows, row_stride, offset, dim,
                                                                        rows_per_pos, pos, f);
}

void route_sigmoid_topk(const float* logits, const float* bias, int n_rows, int32_t* ids, float* weights, void* stream) {
    if (n_rows <= 0) return;
    route_kernel<<<n_rows, kExperts, 0, (cudaStream_t) stream>>>(logits, bias, ids, weights);
}

void engram_rows(const float* rows, const float* head_norm, float eps, int n_rows, float* out, void* stream) {
    if (n_rows <= 0) return;
    engram_rows_kernel<<<n_rows, kEngramHeads * 32, 0, (cudaStream_t) stream>>>(rows, head_norm, eps, out);
}

void engram_apply(const float* key, const float* value, const EngramWeights& w, int n_rows, float* streams,
                  void* stream) {
    if (n_rows <= 0) return;
    engram_apply_kernel<<<n_rows, kEngramThreads, 0, (cudaStream_t) stream>>>(key, value, w, streams);
}

void mla_absorb_q(const float* q, const uint16_t* kv_b, int heads, int n_rows, float* q_lat, void* stream) {
    if (n_rows <= 0) return;
    mla_absorb_kernel<<<dim3(heads, n_rows, kKvRank / 128), 128, 0, (cudaStream_t) stream>>>(
        q, reinterpret_cast<const __half*>(kv_b), heads, q_lat);
}

void mla_attend(const float* q_lat, const float* q, const LatentCache& cache, const int32_t* lo, const int32_t* hi,
                const int32_t* sel, const int32_t* sel_n, int sel_stride, const float* sink, int heads, int n_rows,
                float scale, float* o_lat, void* stream) {
    if (n_rows <= 0) return;
    mla_attend_kernel<<<dim3(heads, n_rows), kAttnWarps * 32, 0, (cudaStream_t) stream>>>(
        q_lat, q, cache, lo, hi, sel, sel_n, sel_stride, sink, heads, scale, o_lat);
}

void mla_value_out(const float* o_lat, const uint16_t* kv_b, const float* gate, int heads, int n_rows, float* out,
                   void* stream) {
    if (n_rows <= 0) return;
    mla_value_out_kernel<<<dim3(heads, n_rows, kVHead / kOutRows), kOutThreads, 0, (cudaStream_t) stream>>>(
        o_lat, reinterpret_cast<const __half*>(kv_b), gate, heads, out);
}

}  // namespace basalt::models::hy4
