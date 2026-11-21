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
constexpr int kSub = 16;                     // steps of 128 columns whose loads are issued together
constexpr size_t kSmem = 48 * 1024;          // the staged columns of a block

// y[c][r] = W[r] . x_c for the columns [c0, c0 + nc) of blockIdx.x's chunk, a warp a row (grid-stride over rows).
// The block stages its columns rotated, H(signs * x_c), in shared memory; a lane decodes 4 columns a step from one
// 32-bit window of the row's big-endian stream (its bit offset is the same every step, its words advance by 4 K) and
// takes every staged column from the same decoded values.  A column's sum has one order whatever columns ride along.
template <int K>
__global__ void __launch_bounds__(kWarps * 32) ms_gemv_cols_kernel(const uint8_t* __restrict__ w,
                                                                   const mc::MsQ81* __restrict__ x,
                                                                   float* __restrict__ y, int n_in, int n_out,
                                                                   int ncols, int cpb) {
    extern __shared__ float4 s_x4[];
    float* s_x = reinterpret_cast<float*>(s_x4);
    const int c0 = blockIdx.x * cpb, nc = min(cpb, ncols - c0);
    const int stride_words = (int) (mc::ms_row_bytes(n_in, K) >> 2);
    const int hb = mc::ms_hadamard_block(n_in);
    const int8_t* signs = reinterpret_cast<const int8_t*>(w + (size_t) n_out * stride_words * 4);
    for (int j = 0; j < nc; ++j)
        mc::ms_stage_q8_rotated(s_x + j * n_in, x + (size_t) (c0 + j) * (n_in / 32), signs, n_in, hb);
    const float rs = rsqrtf((float) hb);
    const int lane = threadIdx.x & 31, n4 = n_in / 4, steps = (n_in + 127) / 128;
    const int lane_bit = 4 * lane * K + 18, w_lane = lane_bit >> 5, b_lane = lane_bit & 31;
    for (int r = blockIdx.y * kWarps + (threadIdx.x >> 5); r < n_out; r += gridDim.y * kWarps) {
        const uint32_t* g32 = reinterpret_cast<const uint32_t*>(w) + (size_t) r * stride_words;
        const uint32_t hdr = __ldg(g32);
        float acc[kMaxCols] = {};
        for (int s0 = 0; s0 < steps; s0 += kSub) {
            // lane i holds word i of each step's 4 K + 2 stream words; a lane's window is two of them
            uint32_t lw[kSub];
            #pragma unroll
            for (int i = 0; i < kSub; ++i) {
                const int wi = 4 * K * (s0 + i) + lane;
                lw[i] = s0 + i < steps && lane < 4 * K + 2 && wi < stride_words
                            ? (wi == 0 ? (hdr >> 16) & mc::kStateMask : __byte_perm(__ldg(g32 + wi), 0, 0x0123))
                            : 0u;
            }
            #pragma unroll
            for (int i = 0; i < kSub; ++i) {
                const int st = s0 + i, c4 = lane + 32 * st;
                if (st >= steps) break;
                const uint32_t a = __shfl_sync(0xFFFFFFFFu, lw[i], w_lane);
                const uint32_t b = __shfl_sync(0xFFFFFFFFu, lw[i], w_lane + 1);
                if (c4 >= n4) continue;
                const uint32_t v = __funnelshift_l(b, a, b_lane);
                const float d0 = mc::decode_1mad_dp4a((v >> 18) & mc::kStateMask);
                const float d1 = mc::decode_1mad_dp4a((v >> (18 - K)) & mc::kStateMask);
                const float d2 = mc::decode_1mad_dp4a((v >> (18 - 2 * K)) & mc::kStateMask);
                const float d3 = mc::decode_1mad_dp4a((v >> (18 - 3 * K)) & mc::kStateMask);
                #pragma unroll
                for (int j = 0; j < kMaxCols; ++j) {
                    if (j < nc) {
                        const float4 xv = s_x4[j * n4 + c4];
                        acc[j] = fmaf(d0, xv.x, fmaf(d1, xv.y, fmaf(d2, xv.z, fmaf(d3, xv.w, acc[j]))));
                    }
                }
            }
        }
        const float scale = __half2float(__ushort_as_half((unsigned short) (hdr & 0xFFFFu))) * rs;
        #pragma unroll
        for (int j = 0; j < kMaxCols; ++j) {
            float s = acc[j];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFFu, s, o);
            if (lane == 0 && j < nc) y[(size_t) (c0 + j) * n_out + r] = s * scale;
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
cudaError_t launch_cols(const uint8_t* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int fit = (int) (kSmem / ((size_t) n_in * 4));
    if (fit < 1) return cudaErrorInvalidValue;
    const int cpb = fit < ncols ? fit : ncols;
    const int chunks = (ncols + cpb - 1) / cpb;
    // the chunks of a row block are neighbours in the grid, so they read its rows from L2 while they are there
    const int need = (n_out + kWarps - 1) / kWarps, want = sm_count() * 4 / chunks;
    const int gy = want < 1 ? 1 : want > need ? need : want;
    ms_gemv_cols_kernel<K><<<dim3(chunks, gy), kWarps * 32, (size_t) cpb * n_in * 4, s>>>(
        w, static_cast<const mc::MsQ81*>(x), y, n_in, n_out, ncols, cpb);
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

}  // namespace basalt::kernels
