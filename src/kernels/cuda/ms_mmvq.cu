// src/kernels/cuda/ms_mmvq.cu - native_mmvq's MS path: one launch of the 1MAD GEMV per call.
#include "basalt/kernels/ms_kernels.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "../ms_1mad_cuda.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace basalt::kernels {
namespace {

namespace mc = adamant::kernels::ms_cuda;

constexpr int kWarps = 8;
constexpr int kMaxCols = 8;
constexpr int kRowWords = 192;               // a warp's staged row chunk, in 32-bit words
constexpr int kWordsPerLane = kRowWords / 32;
constexpr int kMsGroupMax = 16;              // blobs a Q8_0 conversion launch takes
constexpr size_t kSmem = 48 * 1024 - kWarps * (kRowWords + 1) * 4;   // the staged columns: two blocks an SM

// y[c][r] = W[r] . x_c for the columns [c0, c0 + nc) of blockIdx.x's chunk, a warp a row (grid-stride over rows).
// The block stages its columns rotated, H(signs * x_c), in shared memory.  Each warp stages its row in shared memory
// as a big-endian bit stream, prefetching the next row (or chunk of a long row) into registers while it decodes the
// current one; a lane decodes 4 adjacent columns a step from one 32-bit window and takes every staged column from
// the same decoded values.  A column's sum has one order whatever columns ride along.
template <int K, int NC>
__global__ void __launch_bounds__(kWarps * 32, 4) ms_gemv_cols_kernel(const uint8_t* __restrict__ w,
                                                                   const mc::MsQ81* __restrict__ x,
                                                                   float* __restrict__ y, int n_in, int n_out,
                                                                   int col0) {
    __shared__ uint32_t s_rows[kWarps][kRowWords + 1];   // +1: the window reads past the last word
    extern __shared__ float4 s_x4[];
    float* s_x = reinterpret_cast<float*>(s_x4);
    const int c0 = col0 + blockIdx.x * NC;
    const int stride_words = (int) (mc::ms_row_bytes(n_in, K) >> 2);
    const int hb = mc::ms_hadamard_block(n_in);
    const int8_t* signs = reinterpret_cast<const int8_t*>(w + (size_t) n_out * stride_words * 4);
    const float rs = rsqrtf((float) hb);
    const int lane = threadIdx.x & 31, n4 = n_in / 4;
    const int warp_stride = gridDim.y * kWarps;
    int row = blockIdx.y * kWarps + (threadIdx.x >> 5);

    // a row in chunks of kChunkSteps steps of 128 columns: a chunk starts at word ch * kChunkWords with the row
    // start's in-word bit offsets and needs kChunkWords + 1 words
    constexpr int kChunkSteps = (kRowWords - 1) / (4 * K);
    constexpr int kChunkWords = kChunkSteps * 4 * K;
    const int n_steps = (n_in + 127) / 128;
    const int n_chunks = (n_steps + kChunkSteps - 1) / kChunkSteps;
    const uint32_t* g_u32 = reinterpret_cast<const uint32_t*>(w);
    uint32_t* s_row = s_rows[threadIdx.x >> 5];
    uint32_t buf[kWordsPerLane];
    auto fetch = [&](int r, int ch) {
        const int w0 = ch * kChunkWords;
        const uint32_t* g = g_u32 + (size_t) r * stride_words + w0;
        #pragma unroll
        for (int j = 0; j < kWordsPerLane; ++j) {
            const int wi = lane + 32 * j;
            if (wi <= kChunkWords && w0 + wi < stride_words) buf[j] = __ldg(g + wi);
        }
    };
    int ch = 0;
    if (row < n_out) fetch(row, 0);   // in flight while the columns are staged
    // the NC columns' Hadamard blocks spread over all warps, one barrier
    if (hb >= 32) {
        const int per = n_in / hb;
        for (int b = threadIdx.x >> 5; b < NC * per; b += kWarps) {
            const int j = b / per;
            const mc::MsQ81* xq = x + (size_t) (c0 + j) * (n_in / 32);
            mc::ms_rotate_block_warp(s_x + j * n_in, [&](int i) {
                const mc::MsQ81& q = xq[i >> 5];
                return __half2float(q.d) * (float) q.qs[i & 31] * (float) __ldg(signs + i);
            }, (b - j * per) * hb, hb);
        }
        __syncthreads();
    } else {
        for (int j = 0; j < NC; ++j)
            mc::ms_stage_q8_rotated(s_x + j * n_in, x + (size_t) (c0 + j) * (n_in / 32), signs, n_in, hb);
    }
    if (row >= n_out) return;
    const int lane_bit = 4 * lane * K + 18, bit_off = lane_bit & 31;
    const int lane_steps = (n4 - lane + 31) / 32;   // steps where this lane's 4 columns exist
    float acc[NC] = {};
    float d = 0.0f;
    while (row < n_out) {
        const int w0 = ch * kChunkWords;
        uint32_t s0_word = 0;
        if (ch == 0) {
            const uint32_t hdr = __shfl_sync(0xFFFFFFFFu, buf[0], 0);
            d = __half2float(__ushort_as_half((unsigned short) (hdr & 0xFFFFu)));
            s0_word = (hdr >> 16) & mc::kStateMask;
        }
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < kWordsPerLane; ++j) {
            const int wi = lane + 32 * j;
            if (wi <= kChunkWords) {
                s_row[wi] = w0 + wi >= stride_words ? 0u : (w0 + wi == 0) ? s0_word : __byte_perm(buf[j], 0, 0x0123);
            }
        }
        __syncwarp();
        int next_row = row, next_ch = ch + 1;
        if (next_ch == n_chunks) {
            next_ch = 0;
            next_row += warp_stride;
        }
        if (next_row < n_out) fetch(next_row, next_ch);

        const int s_begin = ch * kChunkSteps;
        const int s_end = min(min(s_begin + kChunkSteps, n_steps), lane_steps);
        int wl = lane_bit >> 5;
        for (int st = s_begin; st < s_end; ++st, wl += 4 * K) {
            const uint32_t v = __funnelshift_l(s_row[wl + 1], s_row[wl], bit_off);
            const float d0 = mc::decode_1mad_dp4a((v >> 18) & mc::kStateMask);
            const float d1 = mc::decode_1mad_dp4a((v >> (18 - K)) & mc::kStateMask);
            const float d2 = mc::decode_1mad_dp4a((v >> (18 - 2 * K)) & mc::kStateMask);
            const float d3 = mc::decode_1mad_dp4a((v >> (18 - 3 * K)) & mc::kStateMask);
            const int c4 = lane + 32 * st;
            #pragma unroll
            for (int j = 0; j < NC; ++j) {
                const float4 xv = s_x4[j * n4 + c4];
                acc[j] = fmaf(d0, xv.x, fmaf(d1, xv.y, fmaf(d2, xv.z, fmaf(d3, xv.w, acc[j]))));
            }
        }
        if (next_ch == 0) {
            const float scale = d * rs;
            #pragma unroll
            for (int j = 0; j < NC; ++j) {
                float sum = acc[j];
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xFFFFFFFFu, sum, o);
                if (lane == 0) y[(size_t) (c0 + j) * n_out + row] = sum * scale;
                acc[j] = 0.0f;
            }
        }
        row = next_row;
        ch = next_ch;
    }
}

// ---- The tensor-core GEMV: y[c][r] = W[r] . x_c for up to 8 columns, 16 rows a warp per m16n8k16 product. ----
// A (16 x 16): the rows' 1MAD codes, integers within +-510 and so exact in fp16 (the row scale holds the rest).  B
// (16 x 8): the block's columns, rotated, H(signs * x_c), staged once as fp16, each divided by sqrt(had_block) and by its
// largest |x| (127 times its largest Q8_1 scale), so every value is within +-sqrt(had_block) (the rounding to fp16 is the
// one change against the fp32 GEMV: docs/precision/qwen38-flash-next.md).  The product's k positions 2t, 2t+1, 2t+8, 2t+9 of lane t are the columns 4t .. 4t+3 of the
// step, the same for A and B: a lane decodes 4 adjacent columns of rows g and g + 8 from one 32-bit window each, and
// reads a column's 4 values as one 8-byte load.  A column's sums take one order whatever columns ride along.
constexpr int kTcWarps = 8;
constexpr int kTcCh = 128;                    // input columns a staged row chunk
constexpr size_t kTcSmemMax = 64 * 1024;      // the staged columns' shared memory at most

__device__ __forceinline__ uint32_t tc_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

template <int K>
__global__ void __launch_bounds__(kTcWarps * 32, 2) ms_gemv_tc_kernel(const uint8_t* __restrict__ w,
                                                                    const mc::MsQ81* __restrict__ x,
                                                                    float* __restrict__ y, int n_in, int n_out,
                                                                    int col0, int nc, int ks) {
    static_assert(3 * K + 14 <= 32, "4 states must fit one 32-bit window");
    // a chunk's windows start at bits 18 + c K (c = k0 .. k0 + 124 in steps of 4) and read two words each
    constexpr int kChWords = (31 + (kTcCh - 4) * K) / 32 + 2;
    __shared__ uint32_t s_w[kTcWarps][16][kChWords];
    __shared__ float s_cmax[8];
    extern __shared__ uint2 s_x[];                // column j's values from s_x + j * n_in / 4, 4 fp16 an element
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int cb = col0 + blockIdx.x * nc;
    const int stride_words = (int) (mc::ms_row_bytes(n_in, K) >> 2);
    const int hb = mc::ms_hadamard_block(n_in);
    const float rs = rsqrtf((float) hb);
    const int8_t* signs = reinterpret_cast<const int8_t*>(w + (size_t) n_out * stride_words * 4);
    for (int j = warp; j < nc; j += kTcWarps) {
        const mc::MsQ81* xq = x + (size_t) (cb + j) * (n_in / 32);
        float m = 0.0f;
        for (int b = lane; b < n_in / 32; b += 32) m = fmaxf(m, fabsf(__half2float(xq[b].d)));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
        if (lane == 0) s_cmax[j] = m > 0.0f ? 127.0f * m : 1.0f;
    }
    __syncthreads();
    __half* s_xh = reinterpret_cast<__half*>(s_x);
    const int per = n_in / hb;   // n_in % 32 == 0: the Hadamard blocks are 32 or more, a warp each
    for (int b = warp; b < nc * per; b += kTcWarps) {
        const int j = b / per;
        const mc::MsQ81* xq = x + (size_t) (cb + j) * (n_in / 32);
        mc::ms_rotate_block_warp(s_xh + (size_t) j * n_in, [&](int i) {
            const mc::MsQ81& q = xq[i >> 5];
            return __half2float(q.d) * (float) q.qs[i & 31] * (float) __ldg(signs + i);
        }, (b - j * per) * hb, hb, rs / s_cmax[j]);
    }
    __syncthreads();

    const uint32_t* g32 = reinterpret_cast<const uint32_t*>(w);
    uint32_t(*sw)[kChWords] = s_w[warp];
    const bool col_on = g < nc;
    const int n_tiles = n_out / 16, n_chunks = (n_in + kTcCh - 1) / kTcCh;
    // ks warps share a tile (each every ks-th chunk; partial sums added in shared memory), kTcWarps / ks tiles a block
    const int tpb = kTcWarps / ks, tg = warp / ks, kq = warp % ks;
    __shared__ float4 s_red[kTcWarps][32];
    // lane l holds word l of the 16 rows' chunk: the next chunk is in flight while the warp decodes the staged one
    uint32_t buf[16];
    auto fetch = [&](int tile, int ch) {
        const uint32_t* rows = g32 + (size_t) tile * 16 * stride_words;
        const int gw = ((18 + ch * kTcCh * K) >> 5) + lane;
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr)
            buf[rr] = lane < kChWords && gw < stride_words ? __ldg(rows + (size_t) rr * stride_words + gw) : 0u;
    };
    const int bstride = gridDim.y * tpb;
    if (blockIdx.y * tpb + tg < n_tiles && kq < n_chunks) fetch(blockIdx.y * tpb + tg, kq);
    for (int base = blockIdx.y * tpb; base < n_tiles; base += bstride) {   // block-uniform: the barriers below
        const int tile = base + tg;
        const bool on = tile < n_tiles;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int ch = kq; on && ch < n_chunks; ch += ks) {
            const int k0 = ch * kTcCh, wlo = (18 + k0 * K) >> 5, gw = wlo + lane;
            __syncwarp();
            if (lane < kChWords) {
                #pragma unroll
                for (int rr = 0; rr < 16; ++rr)   // big-endian, word 0 the start state
                    sw[rr][lane] = gw == 0 ? (buf[rr] >> 16) & mc::kStateMask : __byte_perm(buf[rr], 0, 0x0123);
            }
            __syncwarp();
            if (ch + ks < n_chunks) fetch(tile, ch + ks);
            else if (tile + bstride < n_tiles && kq < n_chunks) fetch(tile + bstride, kq);
            const int kend = min(kTcCh, n_in - k0);
            for (int kk = 0; kk < kend; kk += 16) {
                const int c = k0 + kk + 4 * t;
                const int p = 18 + c * K - 32 * wlo, wi = p >> 5, off = p & 31;
                const uint32_t v0 = __funnelshift_l(sw[g][wi + 1], sw[g][wi], off);
                const uint32_t v1 = __funnelshift_l(sw[g + 8][wi + 1], sw[g + 8][wi], off);
                const uint32_t a0 = tc_h2(mc::decode_1mad_dp4a(v0 >> 18), mc::decode_1mad_dp4a((v0 >> (18 - K)) & mc::kStateMask));
                const uint32_t a2 = tc_h2(mc::decode_1mad_dp4a((v0 >> (18 - 2 * K)) & mc::kStateMask),
                                          mc::decode_1mad_dp4a((v0 >> (18 - 3 * K)) & mc::kStateMask));
                const uint32_t a1 = tc_h2(mc::decode_1mad_dp4a(v1 >> 18), mc::decode_1mad_dp4a((v1 >> (18 - K)) & mc::kStateMask));
                const uint32_t a3 = tc_h2(mc::decode_1mad_dp4a((v1 >> (18 - 2 * K)) & mc::kStateMask),
                                          mc::decode_1mad_dp4a((v1 >> (18 - 3 * K)) & mc::kStateMask));
                uint2 b = make_uint2(0u, 0u);
                if (col_on) b = s_x[((size_t) g * n_in + c) >> 2];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y));
            }
        }
        if (ks > 1) {   // the tile's partial sums, in split order
            s_red[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
            __syncthreads();
            if (kq == 0)
                for (int q = 1; q < ks; ++q) {
                    const float4 r = s_red[warp + q][lane];
                    acc[0] += r.x; acc[1] += r.y; acc[2] += r.z; acc[3] += r.w;
                }
            __syncthreads();
        }
        if (!on || kq != 0) continue;
        // acc[0], acc[1]: row g, columns 2t, 2t + 1; acc[2], acc[3]: row g + 8
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = tile * 16 + g + 8 * h;
            const float d = __half2float(__ushort_as_half((unsigned short) (__ldg(g32 + (size_t) row * stride_words) & 0xFFFFu)));
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int j = 2 * t + q;
                if (j < nc) y[(size_t) (cb + j) * n_out + row] = acc[2 * h + q] * d * s_cmax[j];
            }
        }
    }
}

// Q8_0 rows of W = W_rot H diag(signs) for a group of MS expert blobs, a warp a row over all of them: expert e's
// rows are its P payloads (at off[0..P)) of N_EACH rows each, written to dst + e * dst_stride bytes, row r of the
// expert at r * (N_IN / 32) blocks.  The row is staged in shared memory as a big-endian bit stream with the warp's
// next row in flight; each Hadamard block is decoded, transformed and quantized by the warp (quantize_row_q8_0's
// scale, round to nearest).  One launch for a whole group, so the GPU has enough rows to hide each row's latency.
struct MsBlobs {
    const uint8_t* blob[kMsGroupMax];
    int32_t slot[kMsGroupMax];
};

template <int K, int N_IN, int N_EACH, int P>
__global__ void __launch_bounds__(kWarps * 32) ms_q80_rows_kernel(const MsBlobs g, int n_exp, size_t off0, size_t off1,
                                                                  uint8_t* __restrict__ dst, size_t dst_stride) {
    constexpr int kStride = (int) (((4 + ((N_IN - 1) * K + 7) / 8) + 3) & ~3) / 4;
    constexpr int kWords = (kStride + 1 + 31) / 32;
    constexpr int kHb = N_IN % 512 == 0 ? 512 : N_IN % 256 == 0 ? 256 : N_IN % 128 == 0 ? 128 : N_IN % 64 == 0 ? 64 : 32;
    constexpr int kRowsExp = P * N_EACH;
    static_assert(N_IN % 32 == 0 && kHb >= 32, "whole warps");
    __shared__ uint32_t s_rows[kWarps][kStride + 1];
    __shared__ float s_tp[kWarps][kHb + kHb / 16];   // a warp's transposed Hadamard block
    const int lane = threadIdx.x & 31;
    uint32_t* s_row = s_rows[threadIdx.x >> 5];
    const int n_rows = n_exp * kRowsExp, stride_all = gridDim.x * kWarps;
    // the payload of row r: its expert's blob at off0 or off1
    auto payload = [&](int r) {
        const int lr = r % kRowsExp;
        return g.blob[r / kRowsExp] + (P == 2 && lr >= N_EACH ? off1 : off0);
    };
    uint32_t buf[kWords];
    auto fetch = [&](int r) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(payload(r)) + (size_t) (r % kRowsExp % N_EACH) * kStride;
        #pragma unroll
        for (int j = 0; j < kWords; ++j) {
            const int wi = lane + 32 * j;
            buf[j] = wi < kStride ? __ldg(w + wi) : 0u;
        }
    };
    int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row < n_rows) fetch(row);
    const float rs = rsqrtf((float) kHb);
    for (; row < n_rows; row += stride_all) {
        const uint32_t hdr = __shfl_sync(0xFFFFFFFFu, buf[0], 0);
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < kWords; ++j) {
            const int wi = lane + 32 * j;
            if (wi <= kStride) s_row[wi] = wi == 0 ? (hdr >> 16) & mc::kStateMask : __byte_perm(buf[j], 0, 0x0123);
        }
        __syncwarp();
        const uint8_t* p = payload(row);
        const int8_t* signs = reinterpret_cast<const int8_t*>(p + (size_t) N_EACH * kStride * 4);
        mc::MsQ80* o = reinterpret_cast<mc::MsQ80*>(dst + (size_t) g.slot[row / kRowsExp] * dst_stride) +
                       (size_t) (row % kRowsExp) * (N_IN / 32);
        if (row + stride_all < n_rows) fetch(row + stride_all);
        const float scale = __half2float(__ushort_as_half((unsigned short) (hdr & 0xFFFFu))) * rs;
        // a Hadamard block: lane l decodes columns l + 32 i (V of them), the stages of distance >= 32 run in registers;
        // one padded transpose through shared memory gives lane l the V consecutive columns l V + j, so the stages of
        // distance < V run in registers too and only those from V to 16 cross lanes; a Q8_0 block is 32 / V lanes
        constexpr int V = kHb / 32;
        float* tp = s_tp[threadIdx.x >> 5];
        auto at = [](int c) { return c + (c >> 4); };   // a word of padding every 16: conflict-free both ways
        for (int first = 0; first < N_IN; first += kHb) {
            float r[V];
            #pragma unroll
            for (int i = 0; i < V; ++i) {
                const int bit = (first + lane + 32 * i) * K + 18;
                r[i] = mc::decode_1mad_dp4a(__funnelshift_l(s_row[(bit >> 5) + 1], s_row[bit >> 5], bit & 31) >> 18);
            }
            #pragma unroll
            for (int h = 1; h < V; h <<= 1)
                #pragma unroll
                for (int i = 0; i < V; ++i)
                    if ((i & h) == 0) {
                        const float x0 = r[i], x1 = r[i + h];
                        r[i] = x0 + x1;
                        r[i + h] = x0 - x1;
                    }
            #pragma unroll
            for (int i = 0; i < V; ++i) tp[at(lane + 32 * i)] = r[i];
            __syncwarp();
            #pragma unroll
            for (int j = 0; j < V; ++j) r[j] = tp[at(lane * V + j)];
            __syncwarp();
            #pragma unroll
            for (int h = 1; h < V; h <<= 1)
                #pragma unroll
                for (int j = 0; j < V; ++j)
                    if ((j & h) == 0) {
                        const float x0 = r[j], x1 = r[j + h];
                        r[j] = x0 + x1;
                        r[j + h] = x0 - x1;
                    }
            #pragma unroll
            for (int m = 1; m * V < 32; m <<= 1)
                #pragma unroll
                for (int j = 0; j < V; ++j) {
                    const float x1 = __shfl_xor_sync(0xFFFFFFFFu, r[j], m);
                    r[j] = (lane & m) ? x1 - r[j] : r[j] + x1;
                }
            // scale and signs, the block's absolute maximum, quantize_row_q8_0's scale, round to nearest
            const int c0 = first + lane * V;
            float amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < V; ++j) {
                r[j] *= scale * (float) __ldg(signs + c0 + j);
                amax = fmaxf(amax, fabsf(r[j]));
            }
            #pragma unroll
            for (int m = 1; m * V < 32; m <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, m));
            const float d = amax * (1.0f / 127.0f), id = amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f;
            mc::MsQ80* q = o + c0 / 32;
            #pragma unroll
            for (int j = 0; j < V; j += 2) {   // the block's bytes are 2-aligned
                const int lo = __float2int_rn(r[j] * id), hi = __float2int_rn(r[j + 1] * id);
                *reinterpret_cast<uint16_t*>(q->qs + (c0 & 31) + j) = (uint16_t) ((lo & 0xFF) | ((hi & 0xFF) << 8));
            }
            if ((c0 & 31) == 0) q->d = __float2half_rn(d);
        }
    }
}

int sm_count() {
    static const int n = [] {
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        return sms > 0 ? sms : 1;
    }();
    return n;
}

template <int K>
cudaError_t launch_tc(const uint8_t* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    static const bool big = cudaFuncSetAttribute(ms_gemv_tc_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 (int) kTcSmemMax) == cudaSuccess;
    if (!big) return cudaErrorInvalidValue;
    const int fit = (int) (kTcSmemMax / ((size_t) n_in * 2));
    if (fit < 1) return cudaErrorInvalidValue;
    const int cpb = std::min(std::min(fit, ncols), 8), full = ncols / cpb, rest = ncols - full * cpb;
    // the k split: enough warps for the GPU when the rows are few (at most a chunk a warp)
    const int tiles = n_out / 16, n_chunks = (n_in + kTcCh - 1) / kTcCh;
    int ks = 1;
    while (ks < kTcWarps && ks * 2 <= n_chunks && (int64_t) tiles * ks < (int64_t) sm_count() * 16) ks *= 2;
    const int tpb = kTcWarps / ks, need = (tiles + tpb - 1) / tpb;
    auto go = [&](int nc, int col0, int chunks) {
        const int want = sm_count() * 2 / chunks, gy = want < 1 ? 1 : want > need ? need : want;
        ms_gemv_tc_kernel<K><<<dim3(chunks, gy), kTcWarps * 32, (size_t) nc * n_in * 2, s>>>(
            w, static_cast<const mc::MsQ81*>(x), y, n_in, n_out, col0, nc, ks);
    };
    go(cpb, 0, full);
    if (rest > 0) go(rest, full * cpb, 1);
    return cudaGetLastError();
}

template <int K, int NC>
void launch_nc(const uint8_t* w, const void* x, float* y, int n_in, int n_out, int col0, int chunks, cudaStream_t s) {
    // the chunks of a row block are neighbours in the grid, so they read its rows from L2 while they are there
    const int need = (n_out + kWarps - 1) / kWarps, want = sm_count() * 4 / chunks;
    const int gy = want < 1 ? 1 : want > need ? need : want;
    // the most shared memory an SM can carve out: the blocks are limited by it, not by L1
    static const bool carve = cudaFuncSetAttribute(ms_gemv_cols_kernel<K, NC>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                   cudaSharedmemCarveoutMaxShared) == cudaSuccess;
    (void) carve;
    ms_gemv_cols_kernel<K, NC><<<dim3(chunks, gy), kWarps * 32, (size_t) NC * n_in * 4, s>>>(
        w, static_cast<const mc::MsQ81*>(x), y, n_in, n_out, col0);
}

template <int K>
void launch_n(int nc, const uint8_t* w, const void* x, float* y, int n_in, int n_out, int col0, int chunks,
              cudaStream_t s) {
    switch (nc) {
    case 1: launch_nc<K, 1>(w, x, y, n_in, n_out, col0, chunks, s); break;
    case 2: launch_nc<K, 2>(w, x, y, n_in, n_out, col0, chunks, s); break;
    case 3: launch_nc<K, 3>(w, x, y, n_in, n_out, col0, chunks, s); break;
    case 4: launch_nc<K, 4>(w, x, y, n_in, n_out, col0, chunks, s); break;
    case 5: launch_nc<K, 5>(w, x, y, n_in, n_out, col0, chunks, s); break;
    case 6: launch_nc<K, 6>(w, x, y, n_in, n_out, col0, chunks, s); break;
    case 7: launch_nc<K, 7>(w, x, y, n_in, n_out, col0, chunks, s); break;
    default: launch_nc<K, 8>(w, x, y, n_in, n_out, col0, chunks, s); break;
    }
}

// whole chunks of cpb columns (as many as the staged inputs' shared memory takes), then the rest in one launch
template <int K>
cudaError_t launch_cols(const uint8_t* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int fit = (int) (kSmem / ((size_t) n_in * 4));
    if (fit < 1) return cudaErrorInvalidValue;
    const int cpb = fit < ncols ? fit : ncols, full = ncols / cpb, rest = ncols - full * cpb;
    launch_n<K>(cpb, w, x, y, n_in, n_out, 0, full, s);
    if (rest > 0) launch_n<K>(rest, w, x, y, n_in, n_out, full * cpb, 1, s);
    return cudaGetLastError();
}

}  // namespace

void ms_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
             void* stream) {
    if (!basalt::ms_tensor_type((uint32_t) ggml_type) || n_in <= 0 || n_in % 32 || n_in * 4 > (int) kSmem ||
        n_out <= 0 || ncols < 1 || ncols > kMaxCols || weights == nullptr || x_q8_1 == nullptr || y == nullptr ||
        stream == nullptr || ((uintptr_t) weights & 3) != 0)
        throw std::invalid_argument("ms_mmvq: unsupported type, shape, column count or pointer");
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto s = static_cast<cudaStream_t>(stream);
    const bool tc = n_out % 16 == 0;   // whole 16-row tiles: the tensor-core GEMV, every column count alike
    cudaError_t err = cudaSuccess;
    switch (basalt::ms_bits((uint32_t) ggml_type)) {
    case 2: err = tc ? launch_tc<2>(w, x_q8_1, y, n_in, n_out, ncols, s) : launch_cols<2>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
    case 3: err = tc ? launch_tc<3>(w, x_q8_1, y, n_in, n_out, ncols, s) : launch_cols<3>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
    case 4: err = tc ? launch_tc<4>(w, x_q8_1, y, n_in, n_out, ncols, s) : launch_cols<4>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
    case 5: err = tc ? launch_tc<5>(w, x_q8_1, y, n_in, n_out, ncols, s) : launch_cols<5>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
    default: err = tc ? launch_tc<6>(w, x_q8_1, y, n_in, n_out, ncols, s) : launch_cols<6>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
    }
    if (err != cudaSuccess) throw std::runtime_error(std::string("ms_mmvq: ") + cudaGetErrorString(err));
}

void ms_dequant_dense_f16(int ggml_type, const void* weights, int64_t k, int64_t n, int64_t row0, int64_t rows,
                          uint16_t* dst, void* stream) {
    if (!basalt::ms_tensor_type((uint32_t) ggml_type) || k <= 0 || k % 32 || k > (1 << 20) || n <= 0 || n > INT32_MAX ||
        row0 < 0 || rows <= 0 || row0 + rows > n || weights == nullptr || dst == nullptr)
        throw std::invalid_argument("ms_dequant_dense_f16: unsupported type, shape, rows or pointer");
    namespace mc = adamant::kernels::ms_cuda;
    const auto* w = static_cast<const uint8_t*>(weights);
    auto* o = reinterpret_cast<__half*>(dst);
    const auto s = static_cast<cudaStream_t>(stream);
    const int r0 = (int) row0, nr = (int) rows, no = (int) n, ni = (int) k;
    cudaError_t err = cudaSuccess;
    switch (basalt::ms_bits((uint32_t) ggml_type)) {
    case 2: err = mc::launch_ms_dequant_rows<2>(w, r0, nr, no, ni, o, s); break;
    case 3: err = mc::launch_ms_dequant_rows<3>(w, r0, nr, no, ni, o, s); break;
    case 4: err = mc::launch_ms_dequant_rows<4>(w, r0, nr, no, ni, o, s); break;
    case 5: err = mc::launch_ms_dequant_rows<5>(w, r0, nr, no, ni, o, s); break;
    default: err = mc::launch_ms_dequant_rows<6>(w, r0, nr, no, ni, o, s); break;
    }
    if (err != cudaSuccess) throw std::runtime_error(std::string("ms_dequant_dense_f16: ") + cudaGetErrorString(err));
}

void ms_dequant_expert_f16(int ggml_type, const void* blob, int64_t n_embd, int64_t n_ff, size_t up_off,
                           size_t down_off, uint16_t* dq_gu, uint16_t* dq_d, void* stream) {
    if (!basalt::ms_tensor_type((uint32_t) ggml_type) || n_embd <= 0 || n_embd % 32 || n_ff <= 0 || n_ff % 32 ||
        blob == nullptr || dq_gu == nullptr || dq_d == nullptr)
        throw std::invalid_argument("ms_dequant_expert_f16: unsupported type, shape or pointer");
    namespace mc = adamant::kernels::ms_cuda;
    const auto* b = static_cast<const uint8_t*>(blob);
    auto* gu = reinterpret_cast<__half*>(dq_gu);
    auto* d = reinterpret_cast<__half*>(dq_d);
    const auto s = static_cast<cudaStream_t>(stream);
    const int ne = (int) n_embd, nf = (int) n_ff;
    cudaError_t err = cudaSuccess;
    auto run = [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (err == cudaSuccess) err = mc::launch_ms_dequant_rows<K>(b, 0, nf, nf, ne, gu, s, 2 * ne);
        if (err == cudaSuccess) err = mc::launch_ms_dequant_rows<K>(b + up_off, 0, nf, nf, ne, gu + ne, s, 2 * ne);
        if (err == cudaSuccess) err = mc::launch_ms_dequant_rows<K>(b + down_off, 0, ne, ne, nf, d, s);
    };
    switch (basalt::ms_bits((uint32_t) ggml_type)) {
    case 2: run(std::integral_constant<int, 2>{}); break;
    case 3: run(std::integral_constant<int, 3>{}); break;
    case 4: run(std::integral_constant<int, 4>{}); break;
    case 5: run(std::integral_constant<int, 5>{}); break;
    default: run(std::integral_constant<int, 6>{}); break;
    }
    if (err != cudaSuccess) throw std::runtime_error(std::string("ms_dequant_expert_f16: ") + cudaGetErrorString(err));
}

void ms_experts_to_q8_0_slots(int ggml_type, const uint8_t* const* blobs, const int32_t* slots, int n,
                              int64_t n_embd, int64_t n_ff, size_t up_off, size_t down_off, void* gu_dst,
                              size_t gu_stride, void* d_dst, size_t d_stride, void* stream) {
    if (!basalt::ms_tensor_type((uint32_t) ggml_type) || n_embd != 2560 || n_ff != 640 || n < 1 || n > kMsGroupMax ||
        blobs == nullptr || gu_dst == nullptr || d_dst == nullptr)
        throw std::invalid_argument("ms_experts_to_q8_0: unsupported type, shape, count or pointer");
    MsBlobs g{};
    for (int i = 0; i < n; ++i) {
        if (blobs[i] == nullptr) throw std::invalid_argument("ms_experts_to_q8_0: a null blob");
        g.blob[i] = blobs[i];
        g.slot[i] = slots ? slots[i] : i;
    }
    const auto s = static_cast<cudaStream_t>(stream);
    auto* gu = static_cast<uint8_t*>(gu_dst);
    auto* d = static_cast<uint8_t*>(d_dst);
    // about four blocks an SM, fewer when the group has fewer rows than that
    auto grid = [&](int rows) {
        const int want = sm_count() * 4, need = (rows + kWarps - 1) / kWarps;
        return (unsigned) (want < need ? want : need);
    };
    auto run = [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        ms_q80_rows_kernel<K, 2560, 640, 2><<<grid(n * 1280), kWarps * 32, 0, s>>>(g, n, 0, up_off, gu, gu_stride);
        ms_q80_rows_kernel<K, 640, 2560, 1><<<grid(n * 2560), kWarps * 32, 0, s>>>(g, n, down_off, down_off, d, d_stride);
    };
    switch (basalt::ms_bits((uint32_t) ggml_type)) {
    case 2: run(std::integral_constant<int, 2>{}); break;
    case 3: run(std::integral_constant<int, 3>{}); break;
    case 4: run(std::integral_constant<int, 4>{}); break;
    case 5: run(std::integral_constant<int, 5>{}); break;
    default: run(std::integral_constant<int, 6>{}); break;
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("ms_experts_to_q8_0: ") + cudaGetErrorString(err));
}

void ms_experts_to_q8_0(int ggml_type, const uint8_t* const* blobs, int n, int64_t n_embd, int64_t n_ff,
                        size_t up_off, size_t down_off, void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride,
                        void* stream) {
    ms_experts_to_q8_0_slots(ggml_type, blobs, nullptr, n, n_embd, n_ff, up_off, down_off, gu_dst, gu_stride, d_dst,
                             d_stride, stream);
}

void ms_expert_to_q8_0(int ggml_type, const void* blob, int64_t n_embd, int64_t n_ff, size_t up_off, size_t down_off,
                       void* gu_dst, void* d_dst, void* stream) {
    const uint8_t* b = static_cast<const uint8_t*>(blob);
    ms_experts_to_q8_0(ggml_type, &b, 1, n_embd, n_ff, up_off, down_off, gu_dst, 0, d_dst, 0, stream);
}

}  // namespace basalt::kernels
