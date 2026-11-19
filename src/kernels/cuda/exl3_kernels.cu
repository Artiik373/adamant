// src/kernels/cuda/exl3_kernels.cu - see include/basalt/kernels/exl3_kernels.hpp.
//
// A warp decodes 16x16 tiles: lane l takes ring positions 8l..8l+7 (exl3_lane.hpp), multiplies each decoded value
// with the activation of its tile row for every entry of the group, and keeps one partial sum per tile column pair.
// A CTA covers 128 output columns (8 tile columns, the Hadamard block); its warps split the k blocks, the partial sums
// meet in shared memory in a fixed order, and the output transform runs on the 128 values of one entry in one warp.
#include "basalt/kernels/exl3_kernels.hpp"
#include "basalt/kernels/exl3_lane.hpp"
#include "basalt/kernels/cpu/exl3.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/log.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int kE = 3;   // entries a pass decodes the weights for
constexpr int kWideEntries = 128;   // more entries than a decode window's take the wide kernel

struct Q81 {            // ggml's block_q8_1
    __half d, s;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "q8_1 block");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

// The normalized Hadamard transform of 128 values, lane l holding values 4l..4l+3.
__device__ __forceinline__ void had128(float* v) {
    const int lane = threadIdx.x & 31;
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int s = 0; s < 5; ++s) {
        const int bit = 1 << s;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float o = __shfl_xor_sync(0xffffffffu, v[m], bit);
            v[m] = (lane & bit) ? o - v[m] : v[m] + o;
        }
    }
    const float sc = 0.08838834764831845f;   // 1 / sqrt(128)
#pragma unroll
    for (int m = 0; m < 4; ++m) v[m] *= sc;
}

// The eight windows of a lane at a compile-time rate (KA whole bits a step, HALF: odd steps take one more): the four
// words are shifted left by the lane's bit offset once, then window j sits at the constant bit d(j) of the result, so
// each window is one funnel shift and one shift instead of a data-dependent select chain.
__device__ __forceinline__ void exl3_shift4(const uint32_t* x, int bo, uint32_t* y) {
    y[0] = __funnelshift_l(x[1], x[0], bo);
    y[1] = __funnelshift_l(x[2], x[1], bo);
    y[2] = __funnelshift_l(x[3], x[2], bo);
    y[3] = x[3] << bo;
}
template <int KA, bool HALF>
__device__ __forceinline__ uint32_t exl3_win_ct(const uint32_t* y, int j) {
    const int d = KA * j + (HALF ? (j + 1) >> 1 : 0), q = d >> 5, sh = d & 31;
    const uint32_t hi = q == 0 ? y[0] : q == 1 ? y[1] : q == 2 ? y[2] : y[3];
    const uint32_t lo = q == 0 ? y[1] : q == 1 ? y[2] : q == 2 ? y[3] : 0u;
    return __funnelshift_l(lo, hi, sh) >> 16;
}

// The code of a window as a float: dp4a adds the bits of 2^23 and an offset, so its result is that float exactly and
// one exact subtraction leaves the code (no int to float conversion, a slow instruction).
__device__ __forceinline__ float exl3_code_f(uint32_t win) {
    return __uint_as_float(DP4A(win * 0x83DCD12Du, 0x01010101u, 0x4B000202u)) - 8389632.f;
}

__device__ __forceinline__ void fx_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One projection of every entry of a group, 128 output columns a CTA.
//   KBW k-block groups x NS tile-column splits warps; each warp owns KPW k blocks (K = 128 * KBW * KPW).
//   FROM_Q81: the input row of entry e is token ent_src[e]'s q8_1 activation; otherwise float row e of `xf`.
//   `out_rows`: entry e's output row (null: e itself) in `y` (stride n floats).
//   KA > 0: the rate is known (w = 16 KA + 8 HALF), KA = 0: read from `w` (slow windows).
// A tile is one m16n8k16 product per half of its columns: the lane's 8 decoded weights are its two B fragments (small
// integers, exact in fp16), and the A fragment holds the pass's entries as fp16 pairs hi + lo (rows 0..7 the hi parts,
// rows 8..15 the lo parts), so the products are exact, the sums are fp32 and an entry costs no decode of its own.
//   NC: the output columns (n); with a known rate (KA > 0) the tile addresses are then constants off one base.
//   The second launch bound is the CTAs a multiprocessor's shared memory holds, so the register cap is no tighter.
template <int KBW, int KPW, int NS, int NC, bool FROM_Q81, int KA, bool HALF>
__global__ void __launch_bounds__(KBW * NS * 32, KBW * NS >= 20 ? 2 : KBW * NS >= 12 ? 3 : 4)
exl3_proj_kernel(const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
                    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_src,
                    const int32_t* __restrict__ out_rows, const Q81* __restrict__ xq, const float* __restrict__ xf,
                    float* __restrict__ y0, float* __restrict__ y1, int w_arg, size_t rec0, size_t rec1, int variant) {
    constexpr int K = 128 * KBW * KPW;
    constexpr int n = NC;
    const int w = KA > 0 ? 16 * KA + (HALF ? 8 : 0) : w_arg;
    // xs[entry][k / 2] = {hi(k), hi(k + 1) | lo(k), lo(k + 1)}: two words a pair of activations
    __shared__ __align__(16) uint32_t xs[kE][K];
    __shared__ float part[KBW][kE][128];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int kg = wid % KBW, ns = wid / KBW;
    const int nbk = blockIdx.x;
    const size_t rec = blockIdx.z == 0 ? rec0 : rec1;
    float* y = blockIdx.z == 0 ? y0 : y1;
    const int nw32 = w / 2;
    const Exl3Lane geo = exl3_lane(w, lane);
    int widx[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) widx[i] = geo.wb + i >= nw32 ? geo.wb + i - nw32 : geo.wb + i;
    const size_t trellis_bytes = (size_t) (K / 16) * (size_t) (n / 16) * (size_t) w * 2;
    const int tok = lane >> 2, kq = lane & 3;              // the lane's entry (A row) and pair of tile rows
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g] + rec;
        const __half* suh = (const __half*) (blob + trellis_bytes);
        const __half* svh = suh + K;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int base = e0; base < e1; base += kE) {
            const int ne = e1 - base < kE ? e1 - base : kE;
            // the k group's activations, transformed: its NS warps each take every NS-th k block
#pragma unroll
            for (int kk = 0; kk < KPW; ++kk) {
                if (kk % NS != ns) continue;
                const int kb = kg + kk * KBW;
                for (int ee = 0; ee < ne; ++ee) {
                    float v[4];
                    if (FROM_Q81) {
                        const Q81* blk = xq + (size_t) ent_src[base + ee] * (K / 32) + kb * 4 + (lane >> 3);
                        const int q = *(const int*) (blk->qs + (lane & 7) * 4);
                        const float d = __half2float(blk->d);
                        v[0] = d * (float) (int8_t) (q & 255);
                        v[1] = d * (float) (int8_t) ((q >> 8) & 255);
                        v[2] = d * (float) (int8_t) ((q >> 16) & 255);
                        v[3] = d * (float) (int8_t) ((q >> 24) & 255);
                    } else {
                        const float4 f = *(const float4*) (xf + (size_t) (base + ee) * K + kb * 128 + lane * 4);
                        v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
                    }
                    float sg[4];
#pragma unroll
                    for (int m = 0; m < 4; ++m) sg[m] = __half2float(suh[kb * 128 + lane * 4 + m]);
                    if (!(variant & 1)) {
#pragma unroll
                        for (int m = 0; m < 4; ++m) v[m] *= sg[m];
                    }
                    had128(v);
                    if (variant & 1) {
#pragma unroll
                        for (int m = 0; m < 4; ++m) v[m] *= sg[m];
                    }
                    __half2 hl[4];
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        hl[2 * p] = __floats2half2_rn(v[2 * p], v[2 * p + 1]);
                        const float2 back = __half22float2(hl[2 * p]);
                        hl[2 * p + 1] = __floats2half2_rn(v[2 * p] - back.x, v[2 * p + 1] - back.y);
                    }
                    *(uint4*) (&xs[ee][kb * 128 + lane * 4]) = *(const uint4*) hl;
                }
            }
            __syncthreads();
#pragma unroll 1
            for (int nl = 0; nl < 8 / NS; ++nl) {
                const int nt = ns * (8 / NS) + nl;
                float acc[2][4];
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
#pragma unroll 1
                for (int kk = 0; kk < KPW; ++kk) {
                    const int kb = kg + kk * KBW;
#pragma unroll
                    for (int kt = 0; kt < 8; ++kt) {
                        const uint32_t* tile = (const uint32_t*) (blob + ((size_t) (kb * 8 + kt) * (size_t) (n / 16) +
                                                                          (size_t) (nbk * 8 + nt)) * (size_t) w * 2);
                        uint32_t x[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) x[i] = __ldg(tile + widx[i]);
                        uint32_t yw[4];
                        if constexpr (KA > 0) exl3_shift4(x, geo.pos[0], yw);
                        __half code[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            uint32_t win;
                            if constexpr (KA > 0) win = exl3_win_ct<KA, HALF>(yw, j);
                            else win = exl3_window(x, geo.pos[j]);
                            code[j] = __float2half_rn(exl3_code_f(win));
                        }
                        uint32_t b[2][2];
#pragma unroll
                        for (int h = 0; h < 2; ++h)
#pragma unroll
                            for (int r = 0; r < 2; ++r) {
                                const __half2 pr = __halves2half2(code[h * 4 + r * 2], code[h * 4 + r * 2 + 1]);
                                b[h][r] = *(const uint32_t*) &pr;
                            }
                        uint32_t a[4] = {0u, 0u, 0u, 0u};
                        if (tok < ne) {
                            const uint32_t* xk = &xs[tok][kb * 128 + kt * 16 + kq * 2];
                            const uint2 r0 = *(const uint2*) xk, r8 = *(const uint2*) (xk + 8);
                            a[0] = r0.x; a[1] = r0.y; a[2] = r8.x; a[3] = r8.y;
                        }
                        fx_mma(acc[0], a, b[0][0], b[0][1]);
                        fx_mma(acc[1], a, b[1][0], b[1][1]);
                    }
                }
                // rows tok and tok + 8 of the product are the entry's hi and lo parts
                if (tok < ne) {
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                        *(float2*) (&part[kg][tok][nt * 16 + h * 8 + kq * 2]) =
                            make_float2(acc[h][0] + acc[h][2], acc[h][1] + acc[h][3]);
                }
            }
            __syncthreads();
            // fixed-order sum over the k groups, the output transform, the store
            if (wid < ne) {
                const int ee = wid;
                float v[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    float t = 0.f;
#pragma unroll
                    for (int k2 = 0; k2 < KBW; ++k2) t += part[k2][ee][lane * 4 + m];
                    v[m] = t * exl3_code_scale();
                }
                float sv[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) sv[m] = __half2float(svh[nbk * 128 + lane * 4 + m]);
                if (variant & 2) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) v[m] *= sv[m];
                }
                had128(v);
                if (!(variant & 2)) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) v[m] *= sv[m];
                }
                const int row = out_rows != nullptr ? out_rows[base + ee] : base + ee;
                *(float4*) (y + (size_t) row * n + nbk * 128 + lane * 4) = make_float4(v[0], v[1], v[2], v[3]);
            }
            __syncthreads();
        }
    }
}

// The grouped experts of a wide window (a prompt's): a CTA takes one group's entries kWM at a time and 128 output
// columns, decoding each weight tile once for them. The inputs of a k block are transformed into shared memory as fp16
// hi + lo pairs (the decode kernel's exact products), one m16n8k16 product for each part into the same fp32 sums.
//   GU: gate (rec0) and up (rec1) of q8_1 token rows ent_src[e], the SwiGLU fused: y row e = h. Otherwise down of the
//   float rows xf[e], y row out_rows[e].
constexpr int kWM = 32;     // entries a pass
constexpr int kWPad = 136;  // smem row pitch in halves: conflict-free ldmatrix

template <int K, int N, bool GU, int KA, bool HALF>
__global__ void __launch_bounds__(256) exl3_wide_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_src,
                                                        const int32_t* __restrict__ out_rows, const Q81* __restrict__ xq,
                                                        const float* __restrict__ xf, float* __restrict__ y, int w_arg,
                                                        size_t rec0, size_t rec1, int variant, float limit) {
    constexpr int P = GU ? 2 : 1;
    __shared__ __align__(16) __half As[P][2][kWM * kWPad];   // [projection][hi, lo][row][k]
    float* stage = (float*) &As[0][0][0];                     // [P][kWM][128] after the k loop
    static_assert(sizeof(As) >= (size_t) P * kWM * 128 * 4, "stage fits");
    const int w = KA > 0 ? 16 * KA + (HALF ? 8 : 0) : w_arg;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nb = blockIdx.x;
    const int nw32 = w / 2;
    const Exl3Lane geo = exl3_lane(w, lane);
    int widx[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) widx[i] = geo.wb + i >= nw32 ? geo.wb + i - nw32 : geo.wb + i;
    const size_t tb = (size_t) (K / 16) * (size_t) (N / 16) * (size_t) w * 2;
    const uint32_t as_base = (uint32_t) __cvta_generic_to_shared(&As[0][0][0]);
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* rec[P];
        rec[0] = (const uint8_t*) grp_ptr[g] + rec0;
        if (GU) rec[P - 1] = (const uint8_t*) grp_ptr[g] + rec1;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int base = e0; base < e1; base += kWM) {
            const int ne = e1 - base < kWM ? e1 - base : kWM;
            float acc[P][2][2][4];
#pragma unroll
            for (int p = 0; p < P; ++p)
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int i = 0; i < 4; ++i) acc[p][mt][h][i] = 0.f;
#pragma unroll 1
            for (int kb = 0; kb < K / 128; ++kb) {
                __syncthreads();   // the previous k block's fragments are read
                // rows wid * 4 .. + 3 of the pass: the input transform of this k block, per projection
                for (int i = 0; i < 4; ++i) {
                    const int r = wid * 4 + i;
                    float x0[4] = {0.f, 0.f, 0.f, 0.f};
                    if (r < ne) {
                        if (GU) {
                            const Q81* blk = xq + (size_t) ent_src[base + r] * (K / 32) + kb * 4 + (lane >> 3);
                            const int q = *(const int*) (blk->qs + (lane & 7) * 4);
                            const float d = __half2float(blk->d);
                            x0[0] = d * (float) (int8_t) (q & 255);
                            x0[1] = d * (float) (int8_t) ((q >> 8) & 255);
                            x0[2] = d * (float) (int8_t) ((q >> 16) & 255);
                            x0[3] = d * (float) (int8_t) ((q >> 24) & 255);
                        } else {
                            const float4 f = *(const float4*) (xf + (size_t) (base + r) * K + kb * 128 + lane * 4);
                            x0[0] = f.x; x0[1] = f.y; x0[2] = f.z; x0[3] = f.w;
                        }
                    }
#pragma unroll
                    for (int p = 0; p < P; ++p) {
                        const __half* suh = (const __half*) (rec[p] + tb);
                        float v[4], sg[4];
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            v[m] = x0[m];
                            sg[m] = __half2float(suh[kb * 128 + lane * 4 + m]);
                        }
                        if (!(variant & 1))
#pragma unroll
                            for (int m = 0; m < 4; ++m) v[m] *= sg[m];
                        had128(v);
                        if (variant & 1)
#pragma unroll
                            for (int m = 0; m < 4; ++m) v[m] *= sg[m];
                        __half2 hi[2], lo[2];
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            hi[q] = __floats2half2_rn(v[2 * q], v[2 * q + 1]);
                            const float2 back = __half22float2(hi[q]);
                            lo[q] = __floats2half2_rn(v[2 * q] - back.x, v[2 * q + 1] - back.y);
                        }
                        *(uint2*) &As[p][0][r * kWPad + lane * 4] = *(const uint2*) hi;
                        *(uint2*) &As[p][1][r * kWPad + lane * 4] = *(const uint2*) lo;
                    }
                }
                __syncthreads();
#pragma unroll
                for (int p = 0; p < P; ++p) {
                    const uint32_t* tile0 = (const uint32_t*) (rec[p] + ((size_t) (kb * 8) * (size_t) (N / 16) +
                                                                         (size_t) (nb * 8 + wid)) * (size_t) w * 2);
                    uint32_t xt[8][4];
#pragma unroll
                    for (int kt = 0; kt < 8; ++kt)
#pragma unroll
                        for (int i = 0; i < 4; ++i) xt[kt][i] = __ldg(tile0 + kt * (N / 16) * w / 2 + widx[i]);
#pragma unroll
                    for (int kt = 0; kt < 8; ++kt) {
                        uint32_t yw[4];
                        if constexpr (KA > 0) exl3_shift4(xt[kt], geo.pos[0], yw);
                        __half code[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            uint32_t win;
                            if constexpr (KA > 0) win = exl3_win_ct<KA, HALF>(yw, j);
                            else win = exl3_window(xt[kt], geo.pos[j]);
                            code[j] = __float2half_rn(exl3_code_f(win));
                        }
                        uint32_t b[2][2];
#pragma unroll
                        for (int h = 0; h < 2; ++h)
#pragma unroll
                            for (int r = 0; r < 2; ++r) {
                                const __half2 pr = __halves2half2(code[h * 4 + r * 2], code[h * 4 + r * 2 + 1]);
                                b[h][r] = *(const uint32_t*) &pr;
                            }
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                            for (int hl = 0; hl < 2; ++hl) {
                                uint32_t a[4];
                                const uint32_t addr = as_base + (uint32_t) (((p * 2 + hl) * kWM * kWPad +
                                                                             (mt * 16 + a_row) * kWPad + kt * 16 + a_col) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
                                fx_mma(acc[p][mt][0], a, b[0][0], b[0][1]);
                                fx_mma(acc[p][mt][1], a, b[1][0], b[1][1]);
                            }
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int p = 0; p < P; ++p)
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int c = wid * 16 + h * 8 + 2 * (lane & 3), ra = mt * 16 + (lane >> 2);
                        *(float2*) &stage[(p * kWM + ra) * 128 + c] = make_float2(acc[p][mt][h][0], acc[p][mt][h][1]);
                        *(float2*) &stage[(p * kWM + ra + 8) * 128 + c] = make_float2(acc[p][mt][h][2], acc[p][mt][h][3]);
                    }
            __syncthreads();
            // the output transform of each row, the SwiGLU (GU), the store
            for (int i = 0; i < 4; ++i) {
                const int r = wid * 4 + i;
                if (r >= ne) break;
                float o[P][4];
#pragma unroll
                for (int p = 0; p < P; ++p) {
                    const __half* svh = (const __half*) (rec[p] + tb) + K;
                    float v[4], sv[4];
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        v[m] = stage[(p * kWM + r) * 128 + lane * 4 + m] * exl3_code_scale();
                        sv[m] = __half2float(svh[nb * 128 + lane * 4 + m]);
                    }
                    if (variant & 2)
#pragma unroll
                        for (int m = 0; m < 4; ++m) v[m] *= sv[m];
                    had128(v);
                    if (!(variant & 2))
#pragma unroll
                        for (int m = 0; m < 4; ++m) v[m] *= sv[m];
#pragma unroll
                    for (int m = 0; m < 4; ++m) o[p][m] = v[m];
                }
                if constexpr (GU) {
                    float hv[4];
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        float gt = o[0][m], u = o[P - 1][m];
                        if (limit > 0.0f) {
                            gt = fminf(gt, limit);
                            u = fmaxf(fminf(u, limit), -limit);
                        }
                        hv[m] = __fmul_rn(gt / (1.0f + expf(-gt)), u);
                    }
                    *(float4*) (y + (size_t) (base + r) * N + nb * 128 + lane * 4) = make_float4(hv[0], hv[1], hv[2], hv[3]);
                } else {
                    *(float4*) (y + (size_t) out_rows[base + r] * N + nb * 128 + lane * 4) =
                        make_float4(o[0][0], o[0][1], o[0][2], o[0][3]);
                }
            }
        }
    }
}

template <int K, int N, bool GU>
void launch_wide(int w, dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_src, const int32_t* out_rows, const Q81* xq,
                 const float* xf, float* y, size_t rec0, size_t rec1, int variant, float limit) {
    switch (w) {
#define EXL3_WIDE(W, KA, HALF) case W: exl3_wide_kernel<K, N, GU, KA, HALF><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_src, out_rows, xq, xf, y, w, rec0, rec1, variant, limit); break;
    EXL3_WIDE(32, 2, false)
    EXL3_WIDE(40, 2, true)
    EXL3_WIDE(48, 3, false)
    EXL3_WIDE(56, 3, true)
    EXL3_WIDE(64, 4, false)
#undef EXL3_WIDE
    default:
        exl3_wide_kernel<K, N, GU, 0, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_src, out_rows, xq, xf,
                                                                 y, w, rec0, rec1, variant, limit);
    }
}

// h = silu(gate) * up over the entries of all groups; a positive `limit` caps the gate and clamps the up.
__global__ void __launch_bounds__(256) exl3_swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups, int n_ff, float limit,
                                                          float* __restrict__ h) {
    const long long lo = (long long) grp_start[0] * n_ff, hi = (long long) grp_start[*n_groups] * n_ff;
    for (long long i = lo + (long long) blockIdx.x * blockDim.x + threadIdx.x; i < hi;
         i += (long long) gridDim.x * blockDim.x) {
        float g = gate[i], u = up[i];
        if (limit > 0.0f) {
            g = fminf(g, limit);
            u = fmaxf(fminf(u, limit), -limit);
        }
        h[i] = __fmul_rn(g / (1.0f + expf(-g)), u);
    }
}


// W_full of one projection, one 128 x 128 block a CTA: the block of W_full is diag(suh) H W_inner H diag(svh) on the
// matching 128-blocks of both dimensions, so each block is independent.  Written transposed (output-major, k
// contiguous) into rows `row_mul * n + row_off` of `dst` (stride k).
// A CTA takes one 128 x 128 block in two halves of 64 columns of the output side (smem [128][65] floats, 33 KB, so two CTAs
// share an SM): per half, decode, the input transform down each of its 64 columns, then the 64-point output transform of every
// row; the first half's rows stay in registers, and with the second half's the 128-point transform ends as the sum and the
// difference.  The result is then written out in two 64-row halves (the transposed store).
constexpr int kDqPad = 65;
constexpr int kDqWarps = 16;

// 64-point Hadamard of one vector spread over a warp: element index = lane + 32 m, m = 0..1; unnormalised.
__device__ __forceinline__ void dq_had64_warp(float (&v)[2], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1];
    v[0] = a;
    v[1] = b;
#pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const float x = __shfl_xor_sync(0xffffffffu, v[m], mask);
            v[m] = (lane & mask) ? x - v[m] : v[m] + x;
        }
    }
}

// 128-point Hadamard of one vector spread over a warp: element index = lane + 32 m (v[m]); unnormalised.
__device__ __forceinline__ void dq_had128_warp(float (&v)[4], int lane) {
    float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = a + b; v[1] = a - b; v[2] = c + d; v[3] = c - d;
    a = v[0]; b = v[1]; c = v[2]; d = v[3];
    v[0] = a + c; v[2] = a - c; v[1] = b + d; v[3] = b - d;
#pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float x = __shfl_xor_sync(0xffffffffu, v[m], mask);
            v[m] = (lane & mask) ? x - v[m] : v[m] + x;
        }
    }
}

// W_full of one projection, one 128 x 128 block a CTA: the block of W_full is diag(suh) H W_inner H diag(svh) on the
// matching 128-blocks of both dimensions, so each block is independent.  Written transposed (output-major, k
// contiguous) into rows `row_mul * n + row_off` of `dst` (stride k).
// per-phase cycle sums (thread 0 of every CTA, when `prof` is set): [4] the whole CTA, [5] the CTA count
__device__ unsigned long long g_dq_prof[6];

template <int KA, bool HALF>
__global__ void __launch_bounds__(kDqWarps * 32, 2) exl3_dequant_kernel(const uint8_t* __restrict__ rec0, size_t rec1, int k,
                                                                      int n, int w, int variant, __half* __restrict__ dst,
                                                                      int row_mul, int nb0, int prof) {
    long long c0 = 0;
    if (prof) c0 = clock64();
    extern __shared__ float S[];   // [128][kDqPad]
    const int kb = blockIdx.x, nb = blockIdx.y + nb0;   // nb0: the first 128-block of a row slice
    const int proj = blockIdx.z;
    const uint8_t* rec = rec0 + (proj ? rec1 : 0);
    const size_t trellis_bytes = (size_t) (k / 16) * (size_t) (n / 16) * (size_t) w * 2;
    const __half* suh = (const __half*) (rec + trellis_bytes);
    const __half* svh = suh + k;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw32 = w / 2;
    const Exl3Lane geo = exl3_lane(w, lane);
    int widx[4];
    for (int i = 0; i < 4; ++i) widx[i] = geo.wb + i >= nw32 ? geo.wb + i - nw32 : geo.wb + i;
    constexpr float kHs = 0.08838834764831845f;
    float su[4], sv[4];   // the lane's scales: su over the k rows lane + 32 m, sv over the output columns lane + 32 m
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        su[m] = __half2float(suh[kb * 128 + lane + 32 * m]);
        sv[m] = __half2float(svh[nb * 128 + lane + 32 * m]);
    }
    float held[8][2];   // the first half's rows after the 64-point transform; then the second half's difference
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        // 1. decode this half's 8 x 4 tiles: warp = tile column (wid & 3) and two of the k tiles (wid >> 2)
        {
            const int ntl = wid & 3, i0 = (wid >> 2) * 2, nt = half * 4 + ntl;
            uint32_t xs[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint32_t* tile = (const uint32_t*) (rec + ((size_t) (kb * 8 + i0 + i) * (size_t) (n / 16) +
                                                                (size_t) (nb * 8 + nt)) * (size_t) w * 2);
#pragma unroll
                for (int q = 0; q < 4; ++q) xs[i][q] = __ldg(tile + widx[q]);
            }
            uint32_t ys[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                if constexpr (KA > 0) exl3_shift4(xs[i], geo.pos[0], ys[i]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    uint32_t win;
                    if constexpr (KA > 0) win = exl3_win_ct<KA, HALF>(ys[i], j);
                    else win = exl3_window(xs[i], geo.pos[j]);
                    const float f = (float) (int) DP4A(win * 0x83DCD12Du, 0x01010101u, 0xFFFFFE02u) * exl3_code_scale();
                    S[((i0 + i) * 16 + exl3_row(lane, j)) * kDqPad + ntl * 16 + exl3_col(lane, j)] = f;
                }
            }
        }
        __syncthreads();
        // 2. the input-side transform down each of the half's 64 columns: a warp a column, lane + 32 m holds the k value of
        //    that row (the two high index bits live in registers, the five low ones are shuffled)
#pragma unroll
        for (int ci = 0; ci < 64 / kDqWarps; ++ci) {
            float* col = S + wid + ci * kDqWarps;
            float v[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                v[m] = col[(lane + 32 * m) * kDqPad];
                if (variant & 1) v[m] *= su[m];
            }
            dq_had128_warp(v, lane);
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                float r = v[m] * kHs;
                if (!(variant & 1)) r *= su[m];
                col[(lane + 32 * m) * kDqPad] = r;
            }
        }
        __syncthreads();
        // 3. the output-side transform of every row's 64 columns: a warp a row, lane + 32 m the column
#pragma unroll
        for (int ri = 0; ri < 128 / kDqWarps; ++ri) {
            float* row = S + (wid + ri * kDqWarps) * kDqPad;
            float v[2];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                v[m] = row[lane + 32 * m];
                if (variant & 2) v[m] *= sv[2 * half + m];
            }
            dq_had64_warp(v, lane);
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                if (half == 0) {
                    held[ri][m] = v[m];
                } else {
                    const float lo = held[ri][m] + v[m], hi = held[ri][m] - v[m];
                    float rl = lo * kHs, rh = hi * kHs;
                    if (!(variant & 2)) { rl *= sv[m]; rh *= sv[2 + m]; }
                    row[lane + 32 * m] = rl;   // this warp's own row: no other warp reads it
                    held[ri][m] = rh;
                }
            }
        }
        __syncthreads();
    }
    // 4. the transposed store, output columns 0..63 then 64..127: a warp an output row at a time, a lane two neighbouring k
    //    values twice (4-byte stores, 128 B a warp)
#pragma unroll
    for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int oi = 0; oi < 64 / kDqWarps; ++oi) {
            const int ol = wid + oi * kDqWarps;
            const int orow = row_mul * ((nb - nb0) * 128 + part * 64 + ol) + proj;
            __half* d = dst + (size_t) orow * k + kb * 128;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int kk = h * 64 + 2 * lane;
                *(__half2*) (d + kk) = __floats2half2_rn(S[kk * kDqPad + ol], S[(kk + 1) * kDqPad + ol]);
            }
        }
        if (part == 0) {
            __syncthreads();
#pragma unroll
            for (int ri = 0; ri < 128 / kDqWarps; ++ri) {
                float* row = S + (wid + ri * kDqWarps) * kDqPad;
#pragma unroll
                for (int m = 0; m < 2; ++m) row[lane + 32 * m] = held[ri][m];
            }
            __syncthreads();
        }
    }
    if (prof && threadIdx.x == 0) {
        atomicAdd(&g_dq_prof[4], (unsigned long long) (clock64() - c0));
        atomicAdd(&g_dq_prof[5], 1ull);
    }
}

bool g_dq_prof_on = false;

template <int KA, bool HALF>
void dequant_proj_rate(const uint8_t* rec0, size_t rec1, int k, int n, int w, __half* dst, int row_mul, int nproj,
                       cudaStream_t s, int nb0, int nblocks) {
    constexpr size_t smem = 128 * kDqPad * sizeof(float);
    static const bool attr = [] {
        return cudaFuncSetAttribute(exl3_dequant_kernel<KA, HALF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) == cudaSuccess;
    }();
    if (!attr) { basalt::log::error("basalt", "exl3 dequant: cannot reserve %zu B of shared memory", smem); std::exit(1); }
    exl3_dequant_kernel<KA, HALF><<<dim3((unsigned) (k / 128), (unsigned) (nblocks < 0 ? n / 128 : nblocks), (unsigned) nproj),
                                    kDqWarps * 32, smem, s>>>(rec0, rec1, k, n, w, cpu::exl3_variant(), dst, row_mul, nb0,
                                                              g_dq_prof_on ? 1 : 0);
}

void dequant_proj(int type, const uint8_t* rec0, size_t rec1, int k, int n, __half* dst, int row_mul, int nproj,
                  cudaStream_t s, int nb0 = 0, int nblocks = -1) {
    const int w = cpu::exl3_words(type);
    switch (w) {
#define EXL3_DQ(W, KA, HALF) case W: dequant_proj_rate<KA, HALF>(rec0, rec1, k, n, w, dst, row_mul, nproj, s, nb0, nblocks); break;
    EXL3_DQ(32, 2, false)
    EXL3_DQ(40, 2, true)
    EXL3_DQ(48, 3, false)
    EXL3_DQ(56, 3, true)
    EXL3_DQ(64, 4, false)
    EXL3_DQ(80, 5, false)
    EXL3_DQ(88, 5, true)
    EXL3_DQ(96, 6, false)
    EXL3_DQ(104, 6, true)
    EXL3_DQ(112, 7, false)
    EXL3_DQ(120, 7, true)
    EXL3_DQ(128, 8, false)
#undef EXL3_DQ
    default: dequant_proj_rate<0, false>(rec0, rec1, k, n, w, dst, row_mul, nproj, s, nb0, nblocks);
    }
}

// A dense projection y[c] = x[c] . W for 1..8 activation columns c (q8_1 rows of k values at xq + c * k / 32).  A CTA
// takes G tile columns (16 output columns each), one warp a group of the k blocks (K = 128 * nkg * kpw), the partial
// sums meeting in shared memory in a fixed order.  G = 8 is a whole 128-block and ends with the output transform; a
// smaller G stores y before it and exl3_dense_post finishes (more CTAs for the narrow projections).  exl3_proj_kernel's
// tensor-core arithmetic with a runtime K; a pass takes up to `emax` columns.
template <int KA, bool HALF, int G>
__global__ void __launch_bounds__(512)
exl3_dense_kernel(const uint8_t* __restrict__ blob, const Q81* __restrict__ xq, float* __restrict__ y, int k, int n,
                     int w, int kpw, int ncols, int variant, int emax) {
    constexpr int GW = G * 16;
    extern __shared__ __align__(16) uint32_t dsm_tc[];
    const int nkg = blockDim.x >> 5;
    const int es = ncols < emax ? ncols : emax;            // columns the buffers hold
    // xs[column][k / 2] = {hi(k), hi(k + 1) | lo(k), lo(k + 1)}: two words a pair of activations
    uint32_t* xs = dsm_tc;
    float* part = (float*) (dsm_tc + (size_t) es * k);     // [nkg][es][GW]
    const int lane = threadIdx.x & 31, kg = threadIdx.x >> 5;
    const int cg = blockIdx.x;
    const int nw32 = w / 2;
    const Exl3Lane geo = exl3_lane(w, lane);
    int widx[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) widx[i] = geo.wb + i >= nw32 ? geo.wb + i - nw32 : geo.wb + i;
    const size_t trellis_bytes = (size_t) (k / 16) * (size_t) (n / 16) * (size_t) w * 2;
    const __half* suh = (const __half*) (blob + trellis_bytes);
    const __half* svh = suh + k;
    const int tok = lane >> 2, kq = lane & 3;              // the lane's column (A row) and pair of tile rows
    for (int base = 0; base < ncols; base += emax) {
        const int ne = ncols - base < emax ? ncols - base : emax;
        for (int kk = 0; kk < kpw; ++kk) {
            const int kb = kg + kk * nkg;
            for (int ee = 0; ee < ne; ++ee) {
                const Q81* blk = xq + (size_t) (base + ee) * (size_t) (k / 32) + kb * 4 + (lane >> 3);
                const int q = *(const int*) (blk->qs + (lane & 7) * 4);
                const float d = __half2float(blk->d);
                float v[4];
                v[0] = d * (float) (int8_t) (q & 255);
                v[1] = d * (float) (int8_t) ((q >> 8) & 255);
                v[2] = d * (float) (int8_t) ((q >> 16) & 255);
                v[3] = d * (float) (int8_t) ((q >> 24) & 255);
                float sg[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) sg[m] = __half2float(suh[kb * 128 + lane * 4 + m]);
                if (!(variant & 1)) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) v[m] *= sg[m];
                }
                had128(v);
                if (variant & 1) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) v[m] *= sg[m];
                }
                __half2 hl[4];
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    hl[2 * p] = __floats2half2_rn(v[2 * p], v[2 * p + 1]);
                    const float2 back = __half22float2(hl[2 * p]);
                    hl[2 * p + 1] = __floats2half2_rn(v[2 * p] - back.x, v[2 * p + 1] - back.y);
                }
                *(uint4*) (xs + (size_t) ee * k + kb * 128 + lane * 4) = *(const uint4*) hl;
            }
        }
        __syncwarp();
#pragma unroll 1
        for (int nt = 0; nt < G; ++nt) {
            float acc[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
#pragma unroll 1
            for (int kk = 0; kk < kpw; ++kk) {
                const int kb = kg + kk * nkg;
                const uint32_t* xk = xs + (size_t) tok * k + kb * 128 + kq * 2;
                const uint32_t* tile0 = (const uint32_t*) (blob + ((size_t) (kb * 8) * (size_t) (n / 16) +
                                                                   (size_t) (cg * G + nt)) * (size_t) w * 2);
                const size_t tstride = (size_t) (n / 16) * (size_t) w / 2;   // words between k tiles
                uint32_t x[8][4];
#pragma unroll
                for (int kt = 0; kt < 8; ++kt)
#pragma unroll
                    for (int i = 0; i < 4; ++i) x[kt][i] = __ldg(tile0 + kt * tstride + widx[i]);
#pragma unroll
                for (int kt = 0; kt < 8; ++kt) {
                    uint32_t yw[4];
                    if constexpr (KA > 0) exl3_shift4(x[kt], geo.pos[0], yw);
                    __half code[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        uint32_t win;
                        if constexpr (KA > 0) win = exl3_win_ct<KA, HALF>(yw, j);
                        else win = exl3_window(x[kt], geo.pos[j]);
                        code[j] = __float2half_rn(exl3_code_f(win));
                    }
                    uint32_t b[2][2];
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            const __half2 pr = __halves2half2(code[h * 4 + r * 2], code[h * 4 + r * 2 + 1]);
                            b[h][r] = *(const uint32_t*) &pr;
                        }
                    uint32_t a[4] = {0u, 0u, 0u, 0u};
                    if (tok < ne) {
                        const uint2 r0 = *(const uint2*) (xk + kt * 16), r8 = *(const uint2*) (xk + kt * 16 + 8);
                        a[0] = r0.x; a[1] = r0.y; a[2] = r8.x; a[3] = r8.y;
                    }
                    fx_mma(acc[0], a, b[0][0], b[0][1]);
                    fx_mma(acc[1], a, b[1][0], b[1][1]);
                }
            }
            // rows tok and tok + 8 of the product are the column's hi and lo parts
            if (tok < ne) {
#pragma unroll
                for (int h = 0; h < 2; ++h)
                    *(float2*) (part + (size_t) (kg * es + tok) * GW + nt * 16 + h * 8 + kq * 2) =
                        make_float2(acc[h][0] + acc[h][2], acc[h][1] + acc[h][3]);
            }
        }
        __syncthreads();
        for (int ee = kg; ee < ne; ee += nkg) {
            if constexpr (G == 8) {
                float v[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    float t = 0.f;
                    for (int k2 = 0; k2 < nkg; ++k2) t += part[(size_t) (k2 * es + ee) * GW + lane * 4 + m];
                    v[m] = t * exl3_code_scale();
                }
                float sv[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) sv[m] = __half2float(svh[cg * 128 + lane * 4 + m]);
                if (variant & 2) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) v[m] *= sv[m];
                }
                had128(v);
                if (!(variant & 2)) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) v[m] *= sv[m];
                }
                *(float4*) (y + (size_t) (base + ee) * n + cg * 128 + lane * 4) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
                for (int c = lane; c < GW; c += 32) {
                    float t = 0.f;
                    for (int k2 = 0; k2 < nkg; ++k2) t += part[(size_t) (k2 * es + ee) * GW + c];
                    y[(size_t) (base + ee) * n + cg * GW + c] = t * exl3_code_scale();
                }
            }
        }
        __syncthreads();
    }
}

// The output transform of the y rows the G < 8 kernel left untransformed: a warp a (column, 128-block).
__global__ void __launch_bounds__(128) exl3_dense_post(const uint8_t* __restrict__ blob, float* __restrict__ y, int k, int n,
                                                       int w, int ncols, int variant) {
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int nnb = n / 128;
    if (gw >= ncols * nnb) return;
    const int col = gw / nnb, nb = gw % nnb;
    const __half* svh = (const __half*) (blob + (size_t) (k / 16) * (size_t) (n / 16) * (size_t) w * 2) + k;
    float* p = y + (size_t) col * n + nb * 128 + lane * 4;
    const float4 q = *(const float4*) p;
    float v[4] = {q.x, q.y, q.z, q.w};
    float sv[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) sv[m] = __half2float(svh[nb * 128 + lane * 4 + m]);
    if (variant & 2)
#pragma unroll
        for (int m = 0; m < 4; ++m) v[m] *= sv[m];
    had128(v);
    if (!(variant & 2))
#pragma unroll
        for (int m = 0; m < 4; ++m) v[m] *= sv[m];
    *(float4*) p = make_float4(v[0], v[1], v[2], v[3]);
}

constexpr int kDenseSmemMax = 99 * 1024;

template <int KA, bool HALF, int G>
void launch_dense_g(const uint8_t* w_blob, const Q81* xq, float* y, int k, int n, int w, int nkg, int kpw, int ncols,
                    int variant, cudaStream_t s) {
    static std::atomic<bool> attr[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) dev = 0;
    if (!attr[dev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(exl3_dense_kernel<KA, HALF, G>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kDenseSmemMax) != cudaSuccess) {
            basalt::log::error("basalt", "exl3_mmvq: cannot reserve %d B of shared memory", kDenseSmemMax);
            std::exit(1);
        }
        attr[dev].store(true, std::memory_order_release);
    }
    const size_t per = (size_t) k * 4 + (size_t) nkg * G * 16 * 4;   // bytes a column: its activations, its partial sums
    const int emax = (int) std::min<size_t>(8, (size_t) kDenseSmemMax / per);
    exl3_dense_kernel<KA, HALF, G><<<dim3((unsigned) (n / (16 * G))), (unsigned) (nkg * 32),
                                        per * (size_t) std::min(ncols, emax), s>>>(w_blob, xq, y, k, n, w, kpw, ncols, variant, emax);
    if (G < 8) {
        const unsigned warps = (unsigned) (ncols * (n / 128));
        exl3_dense_post<<<(warps + 3) / 4, 128, 0, s>>>(w_blob, y, k, n, w, ncols, variant);
    }
}

// G: the most tile columns a CTA takes that still leaves it at least 164 CTAs (two a multiprocessor)
template <int KA, bool HALF>
void launch_dense(const uint8_t* w_blob, const Q81* xq, float* y, int k, int n, int w, int nkg, int kpw, int ncols,
                  int variant, cudaStream_t s) {
    const int tiles = n / 16;
    if (tiles / 8 >= 164) launch_dense_g<KA, HALF, 8>(w_blob, xq, y, k, n, w, nkg, kpw, ncols, variant, s);
    else if (tiles / 4 >= 164) launch_dense_g<KA, HALF, 4>(w_blob, xq, y, k, n, w, nkg, kpw, ncols, variant, s);
    else if (tiles / 2 >= 164) launch_dense_g<KA, HALF, 2>(w_blob, xq, y, k, n, w, nkg, kpw, ncols, variant, s);
    else launch_dense_g<KA, HALF, 1>(w_blob, xq, y, k, n, w, nkg, kpw, ncols, variant, s);
}

template <int KBW, int KPW, int NS, int NC, bool FROM_Q81, int KA, bool HALF>
void launch_rate(dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_src, const int32_t* out_rows, const Q81* xq,
                 const float* xf, float* y0, float* y1, int w, size_t rec0, size_t rec1, int variant) {
    // the largest shared-memory split, so that as many CTAs as the registers allow share a multiprocessor
    static std::atomic<bool> attr[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) dev = 0;
    if (!attr[dev].load(std::memory_order_acquire)) {
        cudaFuncSetAttribute(exl3_proj_kernel<KBW, KPW, NS, NC, FROM_Q81, KA, HALF>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        attr[dev].store(true, std::memory_order_release);
    }
    exl3_proj_kernel<KBW, KPW, NS, NC, FROM_Q81, KA, HALF><<<grid, KBW * NS * 32, 0, s>>>(
        grp_ptr, grp_start, n_groups, ent_src, out_rows, xq, xf, y0, y1, w, rec0, rec1, variant);
}

// The common rates (2, 2.5, 3, 3.5 and 4 bits) get their own window arithmetic, any other the generic one.
template <int KBW, int KPW, int NS, int NC, bool FROM_Q81>
void launch_proj(int w, dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_src, const int32_t* out_rows, const Q81* xq,
                 const float* xf, float* y0, float* y1, size_t rec0, size_t rec1, int variant) {
    switch (w) {
#define EXL3_RATE(W, KA, HALF)     case W: launch_rate<KBW, KPW, NS, NC, FROM_Q81, KA, HALF>(grid, s, grp_ptr, grp_start, n_groups, ent_src, out_rows, xq, xf, y0, y1, w, rec0, rec1, variant); break;
    EXL3_RATE(32, 2, false)
    EXL3_RATE(40, 2, true)
    EXL3_RATE(48, 3, false)
    EXL3_RATE(56, 3, true)
    EXL3_RATE(64, 4, false)
#undef EXL3_RATE
    default:
        launch_rate<KBW, KPW, NS, NC, FROM_Q81, 0, false>(grid, s, grp_ptr, grp_start, n_groups, ent_src, out_rows, xq, xf,
                                                         y0, y1, w, rec0, rec1, variant);
    }
}


// ---- The prompt path's expert projections without materializing W_full ------------------------------------------------
// y = x . W_full with W_full = diag(suh) H W_inner H diag(svh) per 128-block, so   y = H(H(x * suh) . W_inner) * svh   and the
// weight never needs its transforms: x1 = H(x * suh) (a pass over the tokens), y1 = x1 . W_inner on the tensor cores with
// the trellis decoded straight into the MMA's B fragments (a lane's 8 decoded weights are exactly its B registers), then
// y = H(y1 * code_scale) * svh (a pass over the outputs).  W_inner's entries are small integers, exact in fp16.
constexpr int kFxMB = 64;       // tokens a CTA
constexpr int kFxPad = 136;     // smem row pitch in halves (128 + 8): conflict-free ldmatrix
constexpr int kFxSlice = 1024;  // tokens a pass; bounds the scratch

// x1[proj][row][k] = fp16(H(x[row] * suh)): a warp a (row, 128-block), lane l holding values 4l..4l+3
__global__ void __launch_bounds__(128) fx_pre_kernel(const uint8_t* __restrict__ rec0, size_t rec1, int trellis_bytes,
                                                     const __half* __restrict__ x, int M, int K, int variant,
                                                     __half* __restrict__ x1) {
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int nkb = K / 128;
    if (gw >= M * nkb) return;
    const int row = gw / nkb, kb = gw % nkb, proj = blockIdx.y;
    const __half* suh = (const __half*) ((proj ? rec0 + rec1 : rec0) + trellis_bytes);
    float v[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        v[m] = __half2float(x[(size_t) row * K + kb * 128 + lane * 4 + m]);
        if (!(variant & 1)) v[m] *= __half2float(suh[kb * 128 + lane * 4 + m]);
    }
    had128(v);
    __half2 o[2];
#pragma unroll
    for (int m = 0; m < 4; ++m)
        if (variant & 1) v[m] *= __half2float(suh[kb * 128 + lane * 4 + m]);
    o[0] = __floats2half2_rn(v[0], v[1]);
    o[1] = __floats2half2_rn(v[2], v[3]);
    *(uint2*) (x1 + ((size_t) proj * M + row) * K + kb * 128 + lane * 4) = *(uint2*) o;
}

// y1[proj][m][n] = sum_k x1[proj][m][k] * W_inner[k][n].  CTA = 128 output columns x 64 tokens; warp w owns tile column
// (n/16) = 8 * blockIdx.x + w and walks every k tile, decoding each once and applying it to the CTA's 4 m16 tiles.
template <int KA, bool HALF>
__global__ void __launch_bounds__(256) fx_gemm_kernel(const uint8_t* __restrict__ rec0, size_t rec1, const __half* __restrict__ x1,
                                                      int M, int K, int N, int w, float* __restrict__ y1) {
    __shared__ __align__(16) __half As[kFxMB * kFxPad];
    const int proj = blockIdx.z, nb = blockIdx.x, m0 = blockIdx.y * kFxMB;
    const uint8_t* rec = rec0 + (proj ? rec1 : 0);
    const __half* xin = x1 + (size_t) proj * M * K;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, tid = threadIdx.x;
    const int nw32 = w / 2;
    const Exl3Lane geo = exl3_lane(w, lane);
    int widx[4];
    for (int i = 0; i < 4; ++i) widx[i] = geo.wb + i >= nw32 ? geo.wb + i - nw32 : geo.wb + i;
    float acc[kFxMB / 16][2][4];
#pragma unroll
    for (int mt = 0; mt < kFxMB / 16; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][h][i] = 0.f;
    const uint32_t as_base = (uint32_t) __cvta_generic_to_shared(As);
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
    for (int kb = 0; kb < K / 128; ++kb) {
        uint32_t xs[8][4];
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const uint32_t* tile = (const uint32_t*) (rec + ((size_t) (kb * 8 + kt) * (size_t) (N / 16) + (size_t) (nb * 8 + wid)) *
                                                              (size_t) w * 2);
#pragma unroll
            for (int q = 0; q < 4; ++q) xs[kt][q] = __ldg(tile + widx[q]);
        }
        __syncthreads();   // the previous k block's A fragments are done with As
#pragma unroll
        for (int p = 0; p < kFxMB / 16; ++p) {
            const int r = p * 16 + (tid >> 4), c = (tid & 15) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m0 + r < M) v = *(const uint4*) (xin + (size_t) (m0 + r) * K + kb * 128 + c);
            *(uint4*) (As + r * kFxPad + c) = v;
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            uint32_t ys[4];
            if constexpr (KA > 0) exl3_shift4(xs[kt], geo.pos[0], ys);
            __half code[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t win;
                if constexpr (KA > 0) win = exl3_win_ct<KA, HALF>(ys, j);
                else win = exl3_window(xs[kt], geo.pos[j]);
                code[j] = __int2half_rn(DP4A(win * 0x83DCD12Du, 0x01010101u, 0xFFFFFE02u));
            }
            uint32_t b[2][2];
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const __half2 pr = __halves2half2(code[h * 4 + r * 2], code[h * 4 + r * 2 + 1]);
                    b[h][r] = *(const uint32_t*) &pr;
                }
#pragma unroll
            for (int mt = 0; mt < kFxMB / 16; ++mt) {
                uint32_t a[4];
                const uint32_t addr = as_base + (uint32_t) (((mt * 16 + a_row) * kFxPad + kt * 16 + a_col) * 2);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
                fx_mma(acc[mt][0], a, b[0][0], b[0][1]);
                fx_mma(acc[mt][1], a, b[1][0], b[1][1]);
            }
        }
    }
    float* y = y1 + (size_t) proj * M * N;
#pragma unroll
    for (int mt = 0; mt < kFxMB / 16; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n = nb * 128 + wid * 16 + h * 8 + 2 * (lane & 3);
            const int ma = m0 + mt * 16 + (lane >> 2), mb = ma + 8;
            if (ma < M) *(float2*) (y + (size_t) ma * N + n) = make_float2(acc[mt][h][0], acc[mt][h][1]);
            if (mb < M) *(float2*) (y + (size_t) mb * N + n) = make_float2(acc[mt][h][2], acc[mt][h][3]);
        }
}

// out[row][row_mul * col + proj] = H(y1[proj][row][block] * code_scale) * svh: a warp a (row, 128-block)
__global__ void __launch_bounds__(128) fx_post_kernel(const uint8_t* __restrict__ rec0, size_t rec1, size_t trellis_bytes, int K,
                                                      const float* __restrict__ y1, int M, int N, int variant, int row_mul,
                                                      float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int nnb = N / 128;
    if (gw >= M * nnb) return;
    const int row = gw / nnb, nb = gw % nnb, proj = blockIdx.y;
    const __half* svh = (const __half*) ((proj ? rec0 + rec1 : rec0) + trellis_bytes) + K;
    const float4 q = *(const float4*) (y1 + ((size_t) proj * M + row) * N + nb * 128 + lane * 4);
    float v[4] = {q.x * 0.0067672729f, q.y * 0.0067672729f, q.z * 0.0067672729f, q.w * 0.0067672729f};
    float sv[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) sv[m] = __half2float(svh[nb * 128 + lane * 4 + m]);
    if (variant & 2)
#pragma unroll
        for (int m = 0; m < 4; ++m) v[m] *= sv[m];
    had128(v);
    if (!(variant & 2))
#pragma unroll
        for (int m = 0; m < 4; ++m) v[m] *= sv[m];
#pragma unroll
    for (int m = 0; m < 4; ++m) out[(size_t) row * N * row_mul + (size_t) row_mul * (nb * 128 + lane * 4 + m) + proj] = v[m];
}

// scratch: x1 and y1 for one slice, per device
struct FxScratch { char* p = nullptr; };
FxScratch g_fx[16];
constexpr size_t kFxX1Bytes = (size_t) 2 * kFxSlice * 2560 * 2;       // two projections of 2560 halves a token
constexpr size_t kFxY1Bytes = (size_t) kFxSlice * 2560 * 4;           // down: 2560 floats a token; gate/up: 2 x 640
char* fx_scratch() {
    int dev = 0;
    cudaGetDevice(&dev);
    FxScratch& f = g_fx[dev & 15];
    if (!f.p && cudaMalloc(&f.p, kFxX1Bytes + kFxY1Bytes) != cudaSuccess) {
        basalt::log::error("basalt", "exl3 fused experts: cannot allocate %zu B of scratch", kFxX1Bytes + kFxY1Bytes);
        std::exit(1);
    }
    return f.p;
}

template <int KA, bool HALF>
void fx_gemm_rate(const uint8_t* rec0, size_t rec1, const __half* x1, int M, int K, int N, int w, int nproj, float* y1,
                  cudaStream_t s) {
    fx_gemm_kernel<KA, HALF><<<dim3((unsigned) (N / 128), (unsigned) ((M + kFxMB - 1) / kFxMB), (unsigned) nproj), 256, 0, s>>>(
        rec0, rec1, x1, M, K, N, w, y1);
}

// the projection(s) of `nproj` records (rec0, rec0 + rec1) on M tokens: x in fp16 [M][K], out in fp32 [M][N * nproj]
void fx_proj(int type, const uint8_t* rec0, size_t rec1, int K, int N, int nproj, const __half* x, int64_t M, float* out,
             cudaStream_t s) {
    const int w = cpu::exl3_words(type), variant = cpu::exl3_variant();
    char* scr = fx_scratch();
    __half* x1 = (__half*) scr;
    float* y1 = (float*) (scr + kFxX1Bytes);
    const size_t trellis_bytes = (size_t) (K / 16) * (size_t) (N / 16) * (size_t) w * 2;
    for (int64_t r0 = 0; r0 < M; r0 += kFxSlice) {
        const int m = (int) std::min<int64_t>(kFxSlice, M - r0);
        fx_pre_kernel<<<dim3((unsigned) ((m * (K / 128) + 3) / 4), (unsigned) nproj), 128, 0, s>>>(
            rec0, rec1, (int) trellis_bytes, x + (size_t) r0 * K, m, K, variant, x1);
        switch (w) {
#define EXL3_FX(W, KA, HALF) case W: fx_gemm_rate<KA, HALF>(rec0, rec1, x1, m, K, N, w, nproj, y1, s); break;
        EXL3_FX(32, 2, false)
        EXL3_FX(40, 2, true)
        EXL3_FX(48, 3, false)
        EXL3_FX(56, 3, true)
        EXL3_FX(64, 4, false)
        EXL3_FX(80, 5, false)
        EXL3_FX(88, 5, true)
        EXL3_FX(96, 6, false)
        EXL3_FX(104, 6, true)
        EXL3_FX(112, 7, false)
        EXL3_FX(120, 7, true)
        EXL3_FX(128, 8, false)
#undef EXL3_FX
        default: fx_gemm_rate<0, false>(rec0, rec1, x1, m, K, N, w, nproj, y1, s);
        }
        fx_post_kernel<<<dim3((unsigned) ((m * (N / 128) + 3) / 4), (unsigned) nproj), 128, 0, s>>>(
            rec0, rec1, trellis_bytes, K, y1, m, N, variant, nproj, out + (size_t) r0 * N * nproj);
    }
}

}  // namespace

bool exl3_expert_supported(int type, int64_t n_embd, int64_t n_ff) noexcept {
    return cpu::exl3_type(type) && ((n_embd == 2560 && n_ff == 640) || (n_embd == 1536 && n_ff == 1536)) &&
           cpu::exl3_words(type) % 8 == 0;
}

void exl3_expert_grouped(const NativeExpertLayout& L, float limit, const unsigned long long* grp_ptr,
                         const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                         const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,
                         void* scratch, float* out, void* stream, int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (!exl3_expert_supported(L.gu_type, L.n_embd, L.n_ff) || !exl3_expert_supported(L.d_type, L.n_embd, L.n_ff) ||
        L.up_type != L.gu_type) {
        basalt::log::error("basalt", "exl3_expert_grouped: types %d/%d/%d, %lld x %lld", L.gu_type, L.up_type, L.d_type,
                           (long long) L.n_embd, (long long) L.n_ff);
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    const int w = cpu::exl3_words(L.gu_type), wd = cpu::exl3_words(L.d_type);
    const int variant = cpu::exl3_variant();
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    const unsigned gy = (unsigned) ((grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups);
    if (L.n_embd == 1536 && cap_entries > kWideEntries) {   // a wide window: each weight tile decoded once a pass
        const unsigned gw = (unsigned) std::min<int64_t>(gy, 1024);
        launch_wide<1536, 1536, true>(w, dim3((unsigned) (L.n_ff / 128), gw), s, grp_ptr, grp_start, n_groups, ent_tok,
                                      nullptr, (const Q81*) x_q8_1, nullptr, h, 0, L.up_off, variant, limit);
        check("exl3_expert_grouped/wide gu");
        launch_wide<1536, 1536, false>(wd, dim3((unsigned) (L.n_embd / 128), gw), s, grp_ptr, grp_start, n_groups,
                                       nullptr, ent_dst, nullptr, h, out, L.down_off, 0, variant, 0.0f);
        check("exl3_expert_grouped/wide down");
        return;
    }
    if (L.n_embd == 1536) {   // 1536 x 1536 experts: K = 12 blocks (6 k groups x 2) for both projections
        launch_proj<6, 2, 2, 1536, true>(w, dim3((unsigned) (L.n_ff / 128), gy, 2), s, grp_ptr, grp_start, n_groups,
                                         ent_tok, nullptr, (const Q81*) x_q8_1, nullptr, gate, up, 0, L.up_off, variant);
        check("exl3_expert_grouped/gu");
        const long long nh = (long long) cap_entries * L.n_ff;
        exl3_swiglu_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, grp_start, n_groups, (int) L.n_ff, limit, h);
        check("exl3_expert_grouped/swiglu");
        launch_proj<6, 2, 2, 1536, false>(wd, dim3((unsigned) (L.n_embd / 128), gy, 1), s, grp_ptr, grp_start, n_groups,
                                          nullptr, ent_dst, nullptr, h, out, out, L.down_off, L.down_off, variant);
        check("exl3_expert_grouped/down");
        return;
    }
    // gate and up: K = n_embd = 20 blocks (10 k groups x 2 column splits), 5 column blocks, z selects the projection
    launch_proj<10, 2, 2, 640, true>(w, dim3((unsigned) (L.n_ff / 128), gy, 2), s, grp_ptr, grp_start, n_groups,
                                     ent_tok, nullptr, (const Q81*) x_q8_1, nullptr, gate, up, 0, L.up_off, variant);
    check("exl3_expert_grouped/gu");
    const long long nh = (long long) cap_entries * L.n_ff;
    exl3_swiglu_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, grp_start, n_groups, (int) L.n_ff, limit, h);
    check("exl3_expert_grouped/swiglu");
    // down: K = n_ff = 5 blocks (5 warps x 2 column halves), 20 column blocks
    launch_proj<5, 1, 2, 2560, false>(wd, dim3((unsigned) (L.n_embd / 128), gy, 1), s, grp_ptr, grp_start, n_groups,
                                      nullptr, ent_dst, nullptr, h, out, out, L.down_off, L.down_off, variant);
    check("exl3_expert_grouped/down");
}

void exl3_mmvq(int type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
               void* stream) {
    if (!cpu::exl3_type(type) || !cpu::exl3_geometry_ok(n_in, n_out, cpu::exl3_words(type)) || ncols < 1 || ncols > 8 ||
        weights == nullptr || x_q8_1 == nullptr || y == nullptr || stream == nullptr ||
        ((uintptr_t) weights & 3) != 0 || ((uintptr_t) x_q8_1 & 3) != 0)
        throw std::invalid_argument("exl3_mmvq: unsupported type, shape, column count or pointer");
    const int nbk = n_in / 128;
    int nkg = 16;
    while (nkg > 1 && nbk % nkg) --nkg;                      // the k groups: the largest divisor of the k blocks up to 16
    const int kpw = nbk / nkg;
    if (((size_t) n_in + (size_t) nkg * 128) * 4 > (size_t) kDenseSmemMax)   // one column and its partial sums
        throw std::invalid_argument("exl3_mmvq: K is too large for the shared-memory plan");
    const int w = cpu::exl3_words(type), variant = cpu::exl3_variant();
    const uint8_t* blob = (const uint8_t*) weights;
    const Q81* xq = (const Q81*) x_q8_1;
    cudaStream_t s = (cudaStream_t) stream;
    switch (w) {
#define EXL3_DENSE(W, KA, HALF) \
    case W: launch_dense<KA, HALF>(blob, xq, y, n_in, n_out, w, nkg, kpw, ncols, variant, s); break;
    EXL3_DENSE(80, 5, false)
    EXL3_DENSE(88, 5, true)
    EXL3_DENSE(96, 6, false)
    EXL3_DENSE(104, 6, true)
    EXL3_DENSE(112, 7, false)
    EXL3_DENSE(120, 7, true)
    EXL3_DENSE(128, 8, false)
#undef EXL3_DENSE
    default: launch_dense<0, false>(blob, xq, y, n_in, n_out, w, nkg, kpw, ncols, variant, s);
    }
    check("exl3_mmvq");
}

void exl3_dequant_dense_f16(int type, const void* weights, int64_t k, int64_t n, int64_t row0, int64_t rows,
                            uint16_t* dst, void* stream) {
    if (!cpu::exl3_type(type) || !cpu::exl3_geometry_ok((int) k, (int) n, cpu::exl3_words(type)) || row0 < 0 ||
        row0 % 128 || rows <= 0 || rows % 128 || row0 + rows > n) {
        basalt::log::error("basalt", "exl3_dequant_dense_f16: type %d, %lld x %lld, rows %lld+%lld", type, (long long) k,
                           (long long) n, (long long) row0, (long long) rows);
        std::exit(1);
    }
    dequant_proj(type, (const uint8_t*) weights, 0, (int) k, (int) n, (__half*) dst, 1, 1, (cudaStream_t) stream,
                 (int) (row0 / 128), (int) (rows / 128));
    check("exl3_dequant_dense_f16");
}

void exl3_dequant_gu_f16(int type, const void* blob, size_t up_off, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                         void* stream) {
    if (!exl3_expert_supported(type, n_embd, n_ff)) { basalt::log::error("basalt", "exl3_dequant_gu_f16: type %d", type); std::exit(1); }
    dequant_proj(type, (const uint8_t*) blob, up_off, (int) n_embd, (int) n_ff, (__half*) dst, 2, 2, (cudaStream_t) stream);
    check("exl3_dequant_gu_f16");
}

void exl3_dequant_down_f16(int type, const void* rec, int64_t n_embd, int64_t n_ff, uint16_t* dst, void* stream) {
    if (!exl3_expert_supported(type, n_embd, n_ff)) { basalt::log::error("basalt", "exl3_dequant_down_f16: type %d", type); std::exit(1); }
    dequant_proj(type, (const uint8_t*) rec, 0, (int) n_ff, (int) n_embd, (__half*) dst, 1, 1, (cudaStream_t) stream);
    check("exl3_dequant_down_f16");
}

// Cycle sums of the dequant kernel's phases (see g_dq_prof); `on` starts the counters, out gets {decode, input transform,
// output transform, store, whole CTA, CTAs} summed over CTAs since the last call with reset
void exl3_dequant_profile(bool on, bool reset, unsigned long long* out6) {
    g_dq_prof_on = on;
    if (out6) {
        cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(out6, g_dq_prof, 6 * sizeof(unsigned long long));
    }
    if (reset) {
        unsigned long long z[6] = {};
        cudaMemcpyToSymbol(g_dq_prof, z, sizeof z);
    }
}


void exl3_expert_gu_fused(int type, const void* blob, size_t up_off, const uint16_t* x, int64_t m, float* y, void* stream) {
    if (m <= 0) return;
    fx_proj(type, (const uint8_t*) blob, up_off, 2560, 640, 2, (const __half*) x, m, y, (cudaStream_t) stream);
    check("exl3_expert_gu_fused");
}

void exl3_expert_down_fused(int type, const void* rec, const uint16_t* x, int64_t m, float* y, void* stream) {
    if (m <= 0) return;
    fx_proj(type, (const uint8_t*) rec, 0, 640, 2560, 1, (const __half*) x, m, y, (cudaStream_t) stream);
    check("exl3_expert_down_fused");
}

}  // namespace basalt::kernels
