// include/basalt/kernels/ms_kernels.hpp - MS (Mad Shift 1MAD trellis) projections on the GPU.  The payload layout is
// artifact/gguf_reader.hpp's (ms_tensor_type): per-row trellis records, then the int8 signs of the input rotation.
#pragma once

namespace basalt::kernels {

/// native_mmvq for an MS tensor, same arguments: ncols 1..8 Q8_1 columns of the unrotated input, n_in a multiple of 32.
/// The kernel applies the signs and the block Hadamard to the input itself.
void ms_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
             void* stream);

}  // namespace basalt::kernels
