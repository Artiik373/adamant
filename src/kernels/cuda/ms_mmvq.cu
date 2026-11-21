// src/kernels/cuda/ms_mmvq.cu - native_mmvq's MS path: one launch of the 1MAD GEMV per call.
#include "basalt/kernels/ms_kernels.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "../ms_1mad_cuda.cuh"

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

// Q8_0 rows of W = W_rot H diag(signs) for a group of MS expert blobs, a warp a row over all of them: expert e's
// rows are its P payloads (at off[0..P)) of N_EACH rows each, written to dst + e * dst_stride bytes, row r of the
// expert at r * (N_IN / 32) blocks.  The row is staged in shared memory as a big-endian bit stream with the warp's
// next row in flight; each Hadamard block is decoded, transformed and quantized by the warp (a Q8_0 block of 32
// values is one i across the 32 lanes; quantize_row_q8_0's scale, round to nearest).  One launch for a whole group, so the GPU has
// enough rows to hide each row's latency.
struct MsBlobs {
    const uint8_t* blob[kMsGroupMax];
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
        mc::MsQ80* o = reinterpret_cast<mc::MsQ80*>(dst + (size_t) (row / kRowsExp) * dst_stride) +
                       (size_t) (row % kRowsExp) * (N_IN / 32);
        if (row + stride_all < n_rows) fetch(row + stride_all);
        const float scale = __half2float(__ushort_as_half((unsigned short) (hdr & 0xFFFFu))) * rs;
        for (int first = 0; first < N_IN; first += kHb) {
            mc::ms_hadamard_warp(
                [&](int c) {
                    const int bit = c * K + 18;
                    return mc::decode_1mad_dp4a(__funnelshift_l(s_row[(bit >> 5) + 1], s_row[bit >> 5], bit & 31) >> 18);
                },
                [&](int c, float v) {
                    v *= scale * (float) __ldg(signs + c);
                    float amax = fabsf(v);
                    #pragma unroll
                    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, m));
                    const float d = amax * (1.0f / 127.0f), id = amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f;
                    mc::MsQ80* q = o + c / 32;
                    q->qs[lane] = (int8_t) __float2int_rn(v * id);   // nearest (ties to even), within half a step
                    if (lane == 0) q->d = __float2half_rn(d);
                },
                first, kHb);
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
    cudaError_t err = cudaSuccess;
    switch (basalt::ms_bits((uint32_t) ggml_type)) {
    case 3: err = launch_cols<3>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
    case 4: err = launch_cols<4>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
    default: err = launch_cols<6>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
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
    case 3: err = mc::launch_ms_dequant_rows<3>(w, r0, nr, no, ni, o, s); break;
    case 4: err = mc::launch_ms_dequant_rows<4>(w, r0, nr, no, ni, o, s); break;
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
    case 3: run(std::integral_constant<int, 3>{}); break;
    case 4: run(std::integral_constant<int, 4>{}); break;
    default: run(std::integral_constant<int, 6>{}); break;
    }
    if (err != cudaSuccess) throw std::runtime_error(std::string("ms_dequant_expert_f16: ") + cudaGetErrorString(err));
}

void ms_experts_to_q8_0(int ggml_type, const uint8_t* const* blobs, int n, int64_t n_embd, int64_t n_ff,
                        size_t up_off, size_t down_off, void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride,
                        void* stream) {
    if (!basalt::ms_tensor_type((uint32_t) ggml_type) || n_embd != 2560 || n_ff != 640 || n < 1 || n > kMsGroupMax ||
        blobs == nullptr || gu_dst == nullptr || d_dst == nullptr)
        throw std::invalid_argument("ms_experts_to_q8_0: unsupported type, shape, count or pointer");
    MsBlobs g{};
    for (int i = 0; i < n; ++i) {
        if (blobs[i] == nullptr) throw std::invalid_argument("ms_experts_to_q8_0: a null blob");
        g.blob[i] = blobs[i];
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
    case 3: run(std::integral_constant<int, 3>{}); break;
    case 4: run(std::integral_constant<int, 4>{}); break;
    default: run(std::integral_constant<int, 6>{}); break;
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("ms_experts_to_q8_0: ") + cudaGetErrorString(err));
}

void ms_expert_to_q8_0(int ggml_type, const void* blob, int64_t n_embd, int64_t n_ff, size_t up_off, size_t down_off,
                       void* gu_dst, void* d_dst, void* stream) {
    const uint8_t* b = static_cast<const uint8_t*>(blob);
    ms_experts_to_q8_0(ggml_type, &b, 1, n_embd, n_ff, up_off, down_off, gu_dst, 0, d_dst, 0, stream);
}

}  // namespace basalt::kernels
