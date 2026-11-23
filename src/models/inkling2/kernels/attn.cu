// src/models/inkling2/kernels/attn.cu - Inkling 2's decode attention: a query head's scores take a per-query relative bias
// over the last kExtent positions, and a per-query sink logit sits in the softmax's denominator.
//
// A row's keys split in slices whose size comes from the row's own key count alone (whole tiles of the block's four warps,
// at most kMaxSplits slices), so a row's bits do not depend on the other rows of its call or on the launch's bound. A block
// takes one row, one key/value head (its three query heads) and one slice; each warp walks its tiles of 32 keys with a lane
// a key: the lane loads its key's row and scores it in registers, puts its value row in shared memory, and the lanes then
// sum the values with four channels a lane. A second kernel merges a row's slices with the sink, gates the heads and rounds
// them to the q8_1 blocks of the output projection.
#include "basalt/kernels/narrow_bf16.cuh"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "prep.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kWarps = 4;
constexpr int kTile = 32;
constexpr int kUnit = kWarps * kTile;        // keys of one pass of the block's warps
constexpr int kGroup = kHeads / kKvHeads;    // 3 query heads a key/value head
constexpr int kMaxSplits = 64;
constexpr int kVStride = kHeadDim / 2 + 4;   // words a value row takes in shared memory (the lanes' stores do not collide)

struct Q81 {
    half2 ds;
    int8_t qs[32];
};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

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

// The keys a slice of a row with n keys covers: whole passes of the block, at most kMaxSplits slices.
__host__ __device__ __forceinline__ int span_of(int n) {
    const int units = (n + kUnit * kMaxSplits - 1) / (kUnit * kMaxSplits);
    return (units > 1 ? units : 1) * kUnit;
}

__device__ __forceinline__ void row_keys(const AttnArgs& a, int r, int& p, int& lo, int& n) {
    p = a.pos[r];
    lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    n = p + 1 - lo;
}

// 8 cache values as fp32 from 16 bytes of fp16 (or the first 8 of 16 bytes of E4M3, `hi` picking the other 8)
__device__ __forceinline__ void unpack8_f16(const uint4 u, float (&f)[8]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
__device__ __forceinline__ void unpack16_fp8(const uint4 u, float (&f)[16]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 a = bk::e4m3x2_float2(w[i] & 0xFFFFu), b = bk::e4m3x2_float2(w[i] >> 16);
        f[4 * i] = a.x;
        f[4 * i + 1] = a.y;
        f[4 * i + 2] = b.x;
        f[4 * i + 3] = b.y;
    }
}

constexpr int kHalf = kTile / 2;   // value rows a warp stages at once: half the shared memory, more blocks an SM
// blocks an SM the registers and shared memory of a slice block allow
template <int RG> inline constexpr int kSplitBlocks = RG > 1 ? 2 : 3;

// window rows a block scores against each key and value tile it loads: a full layer's rows line up (4), a sliding
// window's do not (1, so they spread over blocks)
constexpr int kRowGroupFull = 4;

template <int RG>
struct Smem {
    float q[RG][kGroup][kHeadDim];
    float rs[RG][kGroup][kRel];
    float m[kWarps][RG][kGroup], l[kWarps][RG][kGroup];
    float p[kWarps][RG][kGroup][kTile];   // a tile's probabilities, read by the P V products (in place of shuffles)
    union {
        uint32_t v[kWarps][kHalf][kVStride];   // half a tile's value rows as fp16 pairs
        float acc[kWarps][kGroup][kHeadDim];   // one row's warp parts, once the tiles are done
    };
};

// A block per (group of RG window rows, key/value head, slice). The rows of a group whose slices start at the same key in
// the same cache (a decode window's rows on a full layer, or before the sliding window fills) are scored together
// against each tile of keys and values the block loads; other rows run one at a time. Each row runs the tiles of its
// own slice in the same order either way, and a tile past a row's end leaves its state as it was (its keys' scores are
// -inf: the correction is 1 and the products add 0), so a row's bits do not depend on the rows beside it.
template <bool FP8, int RG>
__global__ void __launch_bounds__(kWarps * 32, kSplitBlocks<RG>) attn_split_kernel(AttnArgs a, int rows, int z, int mma_from,
                                                                             float* __restrict__ part_ml,
                                                                             float* __restrict__ part_acc) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<RG>& sm = *reinterpret_cast<Smem<RG>*>(smem_raw);
    const int g = blockIdx.y, s = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    constexpr int kRowBytes = FP8 ? kHeadDim : kHeadDim * 2;
    constexpr int kLoads = kRowBytes / 16;
    // the group's rows with a slice s: their positions, slices and caches
    int rr[RG], pp[RG], lo_k[RG], hi_k[RG], seg[RG];
    int nr = 0;
#pragma unroll
    for (int k = 0; k < RG; ++k) {
        const int r = blockIdx.x * RG + k;
        if (r >= rows) break;
        int p, lo, n;
        row_keys(a, r, p, lo, n);
        const int span = span_of(n);
        if (s * span >= n || n >= mma_from) continue;   // past this row's slices, or a row the tensor cores take
        rr[nr] = r;
        pp[nr] = p;
        lo_k[nr] = lo + s * span;
        hi_k[nr] = min(p + 1, lo_k[nr] + span);
        seg[nr] = bk::seg_of(a.seg.at, r);
        ++nr;
    }
    if (nr == 0) return;
    bool together = true;
    for (int k = 1; k < nr; ++k) together = together && lo_k[k] == lo_k[0] && seg[k] == seg[0];
    const int passes = together ? 1 : nr;
    for (int pass = 0; pass < passes; ++pass) {
        const int k0 = together ? 0 : pass, nk = together ? nr : 1;   // the pass's rows: k0 .. k0 + nk - 1
        const int s_lo = lo_k[k0];
        int s_end = hi_k[k0];
        for (int k = 1; k < nk; ++k) s_end = max(s_end, hi_k[k0 + k]);
        const SegCaches& sc = a.seg.c[seg[k0]];   // a batch window's row: its conversation's caches
        const uint8_t* kbase = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.k : a.k_cache) + (size_t) g * kRowBytes;
        const uint8_t* vbase = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.v : a.v_cache) + (size_t) g * kRowBytes;
        // a tile's key rows, the lane's key's (zero past the pass's keys); the warp's first tile is loaded with the
        // queries. The value rows are read when they are staged: held from here they would take the registers of more
        // resident blocks.
        const auto cell_of = [&](int key) { return (size_t) (key % a.cap) * kKvHeads * kRowBytes; };
        const auto load_tile = [&](int t0, uint4 (&kr)[kLoads]) {
            const int key = t0 + lane;
            if (key < s_end) {
                const uint4* kp = reinterpret_cast<const uint4*>(kbase + cell_of(key));
#pragma unroll
                for (int i = 0; i < kLoads; ++i) kr[i] = kp[i];
            } else {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) kr[i] = make_uint4(0, 0, 0, 0);
            }
        };
        const int t_first = s_lo + w * kTile;
        uint4 kr[kLoads];
        if (t_first < s_end) load_tile(t_first, kr);
        if (pass > 0) __syncthreads();   // the last pass's merge read its rows' queries
        for (int i = threadIdx.x; i < nk * kGroup * kHeadDim; i += blockDim.x) {
            const int k = i / (kGroup * kHeadDim), j = i % (kGroup * kHeadDim);
            sm.q[k][j / kHeadDim][j % kHeadDim] = a.q[((size_t) rr[k0 + k] * kHeads + g * kGroup) * kHeadDim + j];
        }
        for (int i = threadIdx.x; i < nk * kGroup * kRel; i += blockDim.x) {
            const int k = i / (kGroup * kRel), j = i % (kGroup * kRel);
            sm.rs[k][j / kRel][j % kRel] = a.r[((size_t) rr[k0 + k] * kHeads + g * kGroup) * kRel + j];
        }
        __syncthreads();
        float m[RG][kGroup], l[RG][kGroup], acc[RG][kGroup][4];
#pragma unroll
        for (int k = 0; k < RG; ++k)
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                m[k][h] = -INFINITY;
                l[k][h] = 0.f;
#pragma unroll
                for (int u = 0; u < 4; ++u) acc[k][h][u] = 0.f;
            }
        for (int t0 = t_first; t0 < s_end; t0 += kUnit) {
            const int key = t0 + lane;
            if (t0 != t_first) load_tile(t0, kr);
            float pr[RG][kGroup];
#pragma unroll
            for (int k = 0; k < RG; ++k) {
                if (k >= nk) break;
                const bool valid = key < hi_k[k0 + k];
                // the lane's key scored for the row's three heads
                float sc[kGroup] = {0.f, 0.f, 0.f};
#pragma unroll
                for (int i = 0; i < kLoads; ++i) {
                    if constexpr (FP8) {
                        float f[16];
                        unpack16_fp8(kr[i], f);
#pragma unroll
                        for (int d = 0; d < 16; ++d)
#pragma unroll
                            for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[k][h][16 * i + d], f[d], sc[h]);
                    } else {
                        float f[8];
                        unpack8_f16(kr[i], f);
#pragma unroll
                        for (int d = 0; d < 8; ++d)
#pragma unroll
                            for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[k][h][8 * i + d], f[d], sc[h]);
                    }
                }
                const int dist = pp[k0 + k] - key;
                float tv[kRel];
#pragma unroll
                for (int i = 0; i < kRel; ++i) tv[i] = valid && dist < kExtent ? a.rel_table[i * kExtent + dist] : 0.f;
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    float b = 0.f;
#pragma unroll
                    for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[k][h][i], tv[i], b);
                    sc[h] = valid ? sc[h] * a.scale + b : -INFINITY;
                }
                // the online softmax over the tile
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    const float mt = warp_max(sc[h]);
                    const float mn = fmaxf(m[k][h], mt);
                    const float corr = m[k][h] == -INFINITY ? 0.f : expf(m[k][h] - mn);
                    pr[k][h] = sc[h] == -INFINITY ? 0.f : expf(sc[h] - mn);
                    sm.p[w][k][h][lane] = pr[k][h];
                    l[k][h] = l[k][h] * corr + warp_sum(pr[k][h]);
#pragma unroll
                    for (int u = 0; u < 4; ++u) acc[k][h][u] *= corr;
                    m[k][h] = mn;
                }
            }
            // P V over the tile's keys in order, their value rows staged half a tile at a time as fp16 pairs (E4M3 is
            // exact in fp16)
            const bool vvalid = key < s_end;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                if ((lane >> 4) == half) {
                    uint32_t* vrow = sm.v[w][lane & (kHalf - 1)];
                    const uint4* vp = reinterpret_cast<const uint4*>(vbase + (vvalid ? cell_of(key) : 0));
                    const auto vr = [&](int i) { return vvalid ? vp[i] : make_uint4(0, 0, 0, 0); };
                    if constexpr (FP8) {
#pragma unroll
                        for (int i = 0; i < kLoads; ++i) {
                            float f[16];
                            unpack16_fp8(vr(i), f);
#pragma unroll
                            for (int d = 0; d < 8; ++d) {
                                const __half2 hv = __floats2half2_rn(f[2 * d], f[2 * d + 1]);
                                vrow[8 * i + d] = *reinterpret_cast<const uint32_t*>(&hv);
                            }
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < kLoads; ++i) *reinterpret_cast<uint4*>(vrow + 4 * i) = vr(i);
                    }
                }
                __syncwarp();
                for (int jj = 0; jj < kHalf; ++jj) {
                    const int j = half * kHalf + jj;
                    const uint2 u = *reinterpret_cast<const uint2*>(&sm.v[w][jj][lane * 2]);
                    const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
                    const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
#pragma unroll
                    for (int k = 0; k < RG; ++k) {
                        if (k >= nk) break;
#pragma unroll
                        for (int h = 0; h < kGroup; ++h) {
                            const float pj = sm.p[w][k][h][j];
                            acc[k][h][0] = fmaf(pj, v01.x, acc[k][h][0]);
                            acc[k][h][1] = fmaf(pj, v01.y, acc[k][h][1]);
                            acc[k][h][2] = fmaf(pj, v23.x, acc[k][h][2]);
                            acc[k][h][3] = fmaf(pj, v23.y, acc[k][h][3]);
                        }
                    }
                }
                __syncwarp();
            }
        }
        // each row's four warp parts of this slice, merged in warp order
#pragma unroll
        for (int k = 0; k < RG; ++k) {
            if (k >= nk) break;
            __syncthreads();   // the value rows, or the last row's parts, read before these take their place
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                if (lane == 0) {
                    sm.m[w][k][h] = m[k][h];
                    sm.l[w][k][h] = l[k][h];
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) sm.acc[w][h][lane * 4 + u] = acc[k][h][u];
            }
            __syncthreads();
            for (int i = threadIdx.x; i < kGroup * kHeadDim; i += blockDim.x) {
                const int h = i / kHeadDim, d = i % kHeadDim;
                float mm = -INFINITY;
                for (int ww = 0; ww < kWarps; ++ww) mm = fmaxf(mm, sm.m[ww][k][h]);
                float ll = 0.f, aa = 0.f;
                for (int ww = 0; ww < kWarps; ++ww) {
                    if (sm.m[ww][k][h] == -INFINITY) continue;
                    const float f = expf(sm.m[ww][k][h] - mm);
                    ll = fmaf(sm.l[ww][k][h], f, ll);
                    aa = fmaf(sm.acc[ww][h][d], f, aa);
                }
                const size_t slot = ((size_t) rr[k0 + k] * kHeads + g * kGroup + h) * z + s;
                part_acc[slot * kHeadDim + d] = aa;
                if (d == 0) {
                    part_ml[slot * 2] = mm;
                    part_ml[slot * 2 + 1] = ll;
                }
            }
        }
    }
}

// ---- full layers on the tensor cores --------------------------------------------------------------------------------
//
// A full layer's decode window with fp16 caches: a block per (key/value head, slice of kMmaSpan keys, conversation of a
// batch window) takes every row of its conversation. Its query rows are the window's rows times the group's three heads (row r, head h at r * kGroup + h), MT
// m16 tiles of them. The block's warps split as MT row tiles x kMmaKw key tiles of kKeysT keys a pass, the pass's K and
// V tiles staged in shared memory once for all its rows. S = Q K^T runs on fp16 tensor cores with fp32 accumulation, then
// scale and the relative bias (fp32, over the last kExtent positions only) and the row's mask; the online softmax is fp32
// (natural base, as the merge reads it) and O += P V runs on fp16 tensor cores with fp32 accumulation. The slices are a
// fixed kMmaSpan keys and every launch splits its warps alike, so a row's bits depend on its own position only (not on
// the rows beside it, their count, or the other conversations of a batch window).
constexpr int kKeysT = 32;                       // keys a tile
constexpr int kMmaSpan = 512;                    // keys a slice
constexpr int kMmaLd = kHeadDim + 8;             // padded fp16 row of a staged tile (halves): ldmatrix without conflicts
constexpr int kMmaMaxRows = 32 / kGroup;         // window rows two m16 tiles hold
// a row with this many keys or more takes the tensor cores; a shorter one the warp-a-tile kernel, whose slices of 128
// keys spread a short row over more blocks. The choice is the row's own (its key count), never its window's.
constexpr int kMmaFrom = 4096;
static_assert(kMmaSpan % (kKeysT * 4) == 0, "a slice is whole passes for every warp split");

template <int MT>
struct MmaSmem {
    static constexpr int kKw = 4 / MT;           // key tiles a pass
    union {
        struct {
            __half k[kKw][kKeysT][kMmaLd];
            __half v[kKw][kKeysT][kMmaLd];
        } t;
        struct {
            float o[kKw][MT * 16][kHeadDim];      // each key group's rows, for the merge
            float m[kKw][MT * 16], l[kKw][MT * 16];
        } x;
    };
};

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

template <int MT>
__global__ void __launch_bounds__(128) attn_mma_kernel(AttnArgs a, int rows, int z, int mma_from, float* __restrict__ part_ml,
                                                       float* __restrict__ part_acc) {
    constexpr int kKw = MmaSmem<MT>::kKw;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    MmaSmem<MT>& sm = *reinterpret_cast<MmaSmem<MT>*>(smem_raw);
    const int g = blockIdx.x, s = blockIdx.y, sg = blockIdx.z;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3;
    const int mt = w % MT, kw = w / MT;           // the warp's row tile and key tile
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    // the block's conversation: rows r0 .. r0 + nr - 1, its caches
    const bool batch = a.seg.at.n > 0;
    const int r0 = batch ? a.seg.at.row0[sg] : 0, nr = (batch ? a.seg.at.row0[sg + 1] : rows) - r0;
    const int p_max = a.pos[r0 + nr - 1];
    const int s_lo = s * kMmaSpan;
    if (s_lo > p_max || p_max + 1 < mma_from) return;   // past every row's keys, or no row of this kernel's
    const int s_hi = min(p_max + 1, s_lo + kMmaSpan);
    // the thread's two query rows (r, head): qa = mt * 16 + gq and qb = qa + 8, as window row / head of the group
    const int qa = mt * 16 + gq, qb = qa + 8;
    const int ra = r0 + qa / kGroup, rb = r0 + qb / kGroup;
    // a row of the conversation with enough keys (a shorter one is the warp-a-tile kernel's)
    const bool va = qa / kGroup < nr && a.pos[ra] + 1 >= mma_from, vb = qb / kGroup < nr && a.pos[rb] + 1 >= mma_from;
    const int ha = g * kGroup + qa % kGroup, hb = g * kGroup + qb % kGroup;
    const int pa = va ? a.pos[ra] : -1, pb = vb ? a.pos[rb] : -1;
    uint32_t qf[kHeadDim / 16][4];
    {
        const float* xa = a.q + ((size_t) (va ? ra : r0) * kHeads + ha) * kHeadDim + 2 * tq;
        const float* xb = a.q + ((size_t) (vb ? rb : r0) * kHeads + hb) * kHeadDim + 2 * tq;
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 16; ++kk) {
            const float2 x0 = *reinterpret_cast<const float2*>(xa + 16 * kk);
            const float2 x1 = *reinterpret_cast<const float2*>(xb + 16 * kk);
            const float2 x2 = *reinterpret_cast<const float2*>(xa + 16 * kk + 8);
            const float2 x3 = *reinterpret_cast<const float2*>(xb + 16 * kk + 8);
            qf[kk][0] = pack_h2(x0.x, x0.y);
            qf[kk][1] = pack_h2(x1.x, x1.y);
            qf[kk][2] = pack_h2(x2.x, x2.y);
            qf[kk][3] = pack_h2(x3.x, x3.y);
        }
    }
    const float* rs_a = a.r + ((size_t) (va ? ra : 0) * kHeads + ha) * kRel;
    const float* rs_b = a.r + ((size_t) (vb ? rb : 0) * kHeads + hb) * kRel;
    const __half* kc = static_cast<const __half*>(batch ? a.seg.c[sg].k : a.k_cache);
    const __half* vc = static_cast<const __half*>(batch ? a.seg.c[sg].v : a.v_cache);

    float o[kHeadDim / 8][4];
#pragma unroll
    for (int n = 0; n < kHeadDim / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;

    // a pass's K (or V) tiles requested into shared memory (zero past the slice's keys)
    constexpr int kPieces = kKeysT * kHeadDim / 8;   // 16-byte pieces of one tile
    const auto request = [&](int t0, bool isv) {
        for (int c = tid; c < kKw * kPieces; c += 128) {
            const int t = c / kPieces, j = (c % kPieces) / (kHeadDim / 8), col = (c % (kHeadDim / 8)) * 8;
            const int key = t0 + t * kKeysT + j;
            const bool ok = key < s_hi;
            const __half* src = (isv ? vc : kc) + ((size_t) (ok ? key % a.cap : 0) * kKvHeads + g) * kHeadDim + col;
            cp16z(isv ? &sm.t.v[t][j][col] : &sm.t.k[t][j][col], src, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    // two stages in the same room: the next pass's keys land during this pass's P V, its values during the next S
    request(s_lo, false);
    request(s_lo, true);
    for (int t0 = s_lo; t0 < s_hi; t0 += kKw * kKeysT) {
        const bool more = t0 + kKw * kKeysT < s_hi;
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");   // this pass's keys (its values may be on the way)
        __syncthreads();
        const int k0 = t0 + kw * kKeysT;          // the warp's tile (past the slice: every score -inf, nothing changes)
        // S = Q K^T
        float sc[kKeysT / 8][4];
#pragma unroll
        for (int n = 0; n < kKeysT / 8; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 16; ++kk) {
#pragma unroll
            for (int n = 0; n < kKeysT / 8; n += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, &sm.t.k[kw][n * 8 + (lane & 7) + ((lane >> 4) << 3)][kk * 16 + ((lane >> 3) & 1) * 8]);
                mma16816(sc[n], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
                mma16816(sc[n + 1], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b2, b3);
            }
        }
        // scale, the bias where a distance reaches below kExtent, and each row's mask (its keys up to its position)
        const bool near = (va ? pa : pb) - (k0 + kKeysT - 1) < kExtent || (vb && pb - (k0 + kKeysT - 1) < kExtent);
#pragma unroll
        for (int n = 0; n < kKeysT / 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = k0 + n * 8 + 2 * tq + (e & 1);
                const bool rowb = e >> 1;
                const int p = rowb ? pb : pa;
                float x = sc[n][e] * a.scale;
                if (near) {
                    const int dist = p - key;
                    if (dist >= 0 && dist < kExtent) {
                        const float* rsv = rowb ? rs_b : rs_a;
                        float b = 0.f;
#pragma unroll
                        for (int i = 0; i < kRel; ++i) b = fmaf(rsv[i], a.rel_table[i * kExtent + dist], b);
                        x += b;
                    }
                }
                sc[n][e] = key <= p && key < s_hi ? x : -INFINITY;
            }
        // the online softmax: row a (sc[n][0..1]) and row b (sc[n][2..3])
        float mx_a = m_a, mx_b = m_b;
#pragma unroll
        for (int n = 0; n < kKeysT / 8; ++n) {
            mx_a = fmaxf(mx_a, fmaxf(sc[n][0], sc[n][1]));
            mx_b = fmaxf(mx_b, fmaxf(sc[n][2], sc[n][3]));
        }
        mx_a = quad_max(mx_a);
        mx_b = quad_max(mx_b);
        const float ca = mx_a == -INFINITY ? 1.f : expf(m_a - mx_a), cb = mx_b == -INFINITY ? 1.f : expf(m_b - mx_b);
        m_a = mx_a;
        m_b = mx_b;
        float ps_a = 0.f, ps_b = 0.f;
#pragma unroll
        for (int n = 0; n < kKeysT / 8; ++n) {
            sc[n][0] = sc[n][0] == -INFINITY ? 0.f : expf(sc[n][0] - m_a);
            sc[n][1] = sc[n][1] == -INFINITY ? 0.f : expf(sc[n][1] - m_a);
            sc[n][2] = sc[n][2] == -INFINITY ? 0.f : expf(sc[n][2] - m_b);
            sc[n][3] = sc[n][3] == -INFINITY ? 0.f : expf(sc[n][3] - m_b);
            ps_a += sc[n][0] + sc[n][1];
            ps_b += sc[n][2] + sc[n][3];
        }
        l_a = l_a * ca + ps_a;
        l_b = l_b * cb + ps_b;
#pragma unroll
        for (int n = 0; n < kHeadDim / 8; ++n) {
            o[n][0] *= ca;
            o[n][1] *= ca;
            o[n][2] *= cb;
            o[n][3] *= cb;
        }
        __syncthreads();   // every warp's S is done: the keys' room takes the next pass's
        if (more) request(t0 + kKw * kKeysT, false);
        if (more) asm volatile("cp.async.wait_group 1;\n" ::: "memory");   // this pass's values
        else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        // O += P V
        unsigned pf[kKeysT / 16][4];
#pragma unroll
        for (int j = 0; j < kKeysT / 16; ++j) {
            pf[j][0] = pack_h2(sc[2 * j][0], sc[2 * j][1]);
            pf[j][1] = pack_h2(sc[2 * j][2], sc[2 * j][3]);
            pf[j][2] = pack_h2(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pf[j][3] = pack_h2(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
#pragma unroll
        for (int n = 0; n < kHeadDim / 8; n += 2)
#pragma unroll
            for (int j = 0; j < kKeysT / 16; ++j) {
                unsigned b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, &sm.t.v[kw][j * 16 + (lane & 15)][n * 8 + (lane >> 4) * 8]);
                mma16816(o[n], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b0, b1);
                mma16816(o[n + 1], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b2, b3);
            }
        __syncthreads();   // every warp's P V is done: the values' room takes the next pass's
        if (more) request(t0 + kKw * kKeysT, true);
    }
    // the key groups' parts of each row merged into the slice's part (the m, l and accumulator the merge kernel reads)
    l_a = quad_sum(l_a);
    l_b = quad_sum(l_b);
    __syncthreads();   // the tiles are read: their room takes the parts
    const int ia = mt * 16 + gq, ib = ia + 8;
#pragma unroll
    for (int n = 0; n < kHeadDim / 8; ++n) {
        const int col = n * 8 + 2 * tq;
        sm.x.o[kw][ia][col] = o[n][0];
        sm.x.o[kw][ia][col + 1] = o[n][1];
        sm.x.o[kw][ib][col] = o[n][2];
        sm.x.o[kw][ib][col + 1] = o[n][3];
    }
    if (tq == 0) {
        sm.x.m[kw][ia] = m_a;
        sm.x.l[kw][ia] = l_a;
        sm.x.m[kw][ib] = m_b;
        sm.x.l[kw][ib] = l_b;
    }
    __syncthreads();
    for (int i = tid; i < MT * 16 * kHeadDim; i += 128) {
        const int q = i / kHeadDim, d = i % kHeadDim;
        const int r = r0 + q / kGroup;
        // a row past the conversation's or left to the other kernel, or a slice past its keys
        if (q / kGroup >= nr || a.pos[r] + 1 < mma_from || s * kMmaSpan > a.pos[r]) continue;
        float mm = -INFINITY;
        for (int k = 0; k < kKw; ++k) mm = fmaxf(mm, sm.x.m[k][q]);
        float ll = 0.f, aa = 0.f;
        for (int k = 0; k < kKw; ++k) {
            if (sm.x.m[k][q] == -INFINITY) continue;
            const float f = expf(sm.x.m[k][q] - mm);
            ll = fmaf(sm.x.l[k][q], f, ll);
            aa = fmaf(sm.x.o[k][q][d], f, aa);
        }
        const size_t slot = ((size_t) r * kHeads + g * kGroup + q % kGroup) * z + s;
        part_acc[slot * kHeadDim + d] = aa;
        if (d == 0) {
            part_ml[slot * 2] = mm;
            part_ml[slot * 2 + 1] = ll;
        }
    }
}

// A block per (row, head), a thread a channel: the row's slices merged with the sink, gated, written in fp32 and/or as
// q8_1 (a warp a block of 32 channels).
__global__ void __launch_bounds__(kHeadDim) attn_combine_kernel(AttnArgs a, int z, int mma_from, const float* __restrict__ part_ml,
                                                                const float* __restrict__ part_acc, float* __restrict__ out) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int p, lo, n;
    row_keys(a, r, p, lo, n);
    const int span = n >= mma_from ? kMmaSpan : span_of(n), splits = (n + span - 1) / span;   // as the slices were cut
    const size_t base = ((size_t) r * kHeads + h) * z;
    const float t = a.sink[(size_t) r * kHeads + h];
    float mm = t;
    for (int s = 0; s < splits; ++s) mm = fmaxf(mm, part_ml[(base + s) * 2]);
    float ll = expf(t - mm), aa = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float f = expf(part_ml[(base + s) * 2] - mm);
        ll = fmaf(part_ml[(base + s) * 2 + 1], f, ll);
        aa = fmaf(part_acc[(base + s) * kHeadDim + d], f, aa);
    }
    float o = aa / ll;
    const int c = h * kHeadDim + d;
    if (a.gate) o *= 1.f / (1.f + expf(-a.gate[(size_t) r * a.gate_ld + c]));
    if (out) out[(size_t) r * kHeads * kHeadDim + c] = o;
    if (a.q8) {
        const float amax = warp_max(fabsf(o));
        const float sum = warp_sum(o);
        const float dq = bk::q8_1_finite(amax / 127.0f);
        Q81& b = reinterpret_cast<Q81*>(a.q8)[(size_t) r * (kHeads * kHeadDim / 32) + c / 32];
        b.qs[c % 32] = bk::q8_1_quant(o, dq, amax);
        if ((c & 31) == 0) b.ds = bk::q8_1_ds(dq, sum);
    }
}

int launch_splits(int64_t max_keys) {
    return (int) std::min<int64_t>(kMaxSplits, std::max<int64_t>(1, (max_keys + kUnit - 1) / kUnit));
}

}  // namespace

int opt_in_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kOptInDevices) throw std::runtime_error("inkling2: device out of range");
    return dev;
}

size_t attention_decode_scratch(int rows, int64_t max_keys) {
    const size_t splits = std::max<size_t>((size_t) launch_splits(max_keys), (size_t) ((max_keys + kMmaSpan - 1) / kMmaSpan));
    const size_t slots = (size_t) rows * kHeads * splits;
    return slots * (2 + kHeadDim) * sizeof(float);
}

void attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    // a full layer's window with fp16 caches on the tensor cores (at most kMmaMaxRows rows a conversation); the rest keeps
    // the warp-a-tile kernel
    bool mma = a.window == 0 && !a.fp8;
    for (int i = 0; i < std::max(1, a.seg.at.n); ++i)
        mma = mma && (a.seg.at.n > 0 ? a.seg.at.row0[i + 1] - a.seg.at.row0[i] : rows) <= kMmaMaxRows;
    const int mma_from = mma ? kMmaFrom : INT_MAX;   // the rows (by key count) the tensor cores take
    const bool any_mma = mma && max_keys >= kMmaFrom;
    const int z = std::max(launch_splits(max_keys), any_mma ? (int) ((max_keys + kMmaSpan - 1) / kMmaSpan) : 1);
    float* part_ml = static_cast<float*>(scratch);
    float* part_acc = part_ml + (size_t) rows * kHeads * z * 2;
    static bool attr[kOptInDevices] = {};   // the opt-in is per device: a layer split launches these on two cards
    if (bool& set = attr[opt_in_device()]; !set) {
        cudaFuncSetAttribute(attn_split_kernel<false, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<1>));
        cudaFuncSetAttribute(attn_split_kernel<true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<1>));
        cudaFuncSetAttribute(attn_split_kernel<false, kRowGroupFull>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) sizeof(Smem<kRowGroupFull>));
        cudaFuncSetAttribute(attn_split_kernel<true, kRowGroupFull>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) sizeof(Smem<kRowGroupFull>));
        cudaFuncSetAttribute(attn_mma_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(MmaSmem<2>));
        set = true;
    }
    const auto launch = [&]<int RG>() {
        const dim3 grid((unsigned) ((rows + RG - 1) / RG), kKvHeads, (unsigned) z);
        if (a.fp8) bk::launch_pdl(attn_split_kernel<true, RG>, grid, dim3(kWarps * 32), sizeof(Smem<RG>), s, a, rows, z, mma_from, part_ml, part_acc);
        else bk::launch_pdl(attn_split_kernel<false, RG>, grid, dim3(kWarps * 32), sizeof(Smem<RG>), s, a, rows, z, mma_from, part_ml, part_acc);
    };
    if (a.window > 0) launch.template operator()<1>();
    else launch.template operator()<kRowGroupFull>();   // the rows with fewer keys
    if (any_mma) {
        // two row tiles whatever the rows: the warps' key split, and with it every row's bits, the same in every launch
        const dim3 grid(kKvHeads, (unsigned) z, (unsigned) std::max(1, a.seg.at.n));
        bk::launch_pdl(attn_mma_kernel<2>, grid, dim3(128), sizeof(MmaSmem<2>), s, a, rows, z, mma_from, part_ml, part_acc);
    }
    check("attention_decode");
    bk::launch_pdl(attn_combine_kernel, dim3((unsigned) rows, kHeads), dim3(kHeadDim), 0, s, a, z, mma_from,
                   (const float*) part_ml, (const float*) part_acc, out);
    check("attention_decode/combine");
}

// A window runs as its preparation, its slices and their merge (no thread-block clusters on this target).
void attention_window(const AttnPrep& pa, const AttnArgs& a, int rows, int64_t max_keys, void* scratch, void* stream) {
    if (rows <= 0) return;
    decode_prepare(pa, rows, stream);
    attention_decode(a, rows, max_keys, scratch, nullptr, stream);
}

}  // namespace basalt::models::inkling2
