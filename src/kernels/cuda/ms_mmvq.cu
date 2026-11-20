// src/kernels/cuda/ms_mmvq.cu - native_mmvq's MS path: one launch of the 1MAD GEMV per call.
#include "basalt/kernels/ms_kernels.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "../ms_1mad_cuda.cuh"

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace basalt::kernels {

void ms_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
             void* stream) {
    if (!basalt::ms_tensor_type((uint32_t) ggml_type) || n_in <= 0 || n_in % 32 || n_out <= 0 || ncols < 1 ||
        ncols > 8 || weights == nullptr || x_q8_1 == nullptr || y == nullptr || stream == nullptr ||
        ((uintptr_t) weights & 3) != 0)
        throw std::invalid_argument("ms_mmvq: unsupported type, shape, column count or pointer");
    namespace mc = adamant::kernels::ms_cuda;
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto s = static_cast<cudaStream_t>(stream);
    cudaError_t err = cudaSuccess;
    switch (basalt::ms_bits((uint32_t) ggml_type)) {
    case 3: err = mc::launch_ms_1mad_gemv_q8<3>(w, x_q8_1, y, n_out, n_in, ncols, s); break;
    case 4: err = mc::launch_ms_1mad_gemv_q8<4>(w, x_q8_1, y, n_out, n_in, ncols, s); break;
    default: err = mc::launch_ms_1mad_gemv_q8<6>(w, x_q8_1, y, n_out, n_in, ncols, s); break;
    }
    if (err != cudaSuccess) throw std::runtime_error(std::string("ms_mmvq: ") + cudaGetErrorString(err));
}

void ms_dequant_dense_f16(int ggml_type, const void* weights, int64_t k, int64_t n, int64_t row0, int64_t rows,
                          uint16_t* dst, void* stream) {
    if (!basalt::ms_tensor_type((uint32_t) ggml_type) || k <= 0 || k % 32 || k > (1 << 20) || n <= 0 || n > INT32_MAX ||
        row0 < 0 || rows <= 0 || row0 + rows > n || weights == nullptr || dst == nullptr || stream == nullptr)
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

}  // namespace basalt::kernels
