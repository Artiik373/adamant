// src/kernels/cuda/ms_mmvq.cu - native_mmvq's MS path: one launch of the 1MAD GEMV per call.
#include "basalt/kernels/ms_kernels.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "../ms_1mad_cuda.cuh"

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

}  // namespace basalt::kernels
