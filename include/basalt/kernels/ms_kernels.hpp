// include/basalt/kernels/ms_kernels.hpp - MS (Mad Shift 1MAD trellis) projections on the GPU.  The payload layout is
// artifact/gguf_reader.hpp's (ms_tensor_type): per-row trellis records, then the int8 signs of the input rotation.
#pragma once

#include "basalt/kernels/iq_kernels.hpp"

#include <cstdint>

namespace basalt::kernels {

/// native_mmvq for an MS tensor, same arguments: ncols 1..8 Q8_1 columns of the unrotated input, n_in a multiple of 32.
/// The kernel applies the signs and the block Hadamard to the input itself.
void ms_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
             void* stream);

/// Rows [row0, row0 + rows) of an MS [k inputs, n outputs] tensor as fp16 W rows (the input rotation undone), into
/// dst[rows][k]: the prompt path's dequantizer.
void ms_dequant_dense_f16(int ggml_type, const void* weights, int64_t k, int64_t n, int64_t row0, int64_t rows,
                          uint16_t* dst, void* stream);

/// The prompt path's fp16 weights of one MS expert blob (device): dq_gu [2 n_ff][n_embd] with gate row r at 2 r and up
/// row r at 2 r + 1, dq_d [n_embd][n_ff].
void ms_dequant_expert_f16(int ggml_type, const void* blob, int64_t n_embd, int64_t n_ff, size_t up_off,
                           size_t down_off, uint16_t* dq_gu, uint16_t* dq_d, void* stream);

/// Whether ms_expert_grouped covers an MS expert layer of this type and geometry.
bool ms_expert_supported(int type, int64_t n_embd, int64_t n_ff) noexcept;

/// native_expert_grouped for an MS layer (blob [gate | up | down] MS payloads): the same arguments and scratch
/// (native_expert_scratch_bytes); `limit` > 0 limits the SwiGLU.
void ms_expert_grouped(const NativeExpertLayout& L, float limit, const unsigned long long* grp_ptr,
                       const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                       const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,
                       void* scratch, float* out, void* stream, int64_t grid_groups);

}  // namespace basalt::kernels
