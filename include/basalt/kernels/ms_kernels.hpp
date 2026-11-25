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
                           size_t down_off, uint16_t* dq_gu, uint16_t* dq_d, void* stream,
                           int up_type = -1, int d_type = -1);

/// One MS expert blob (device) as the prompt path's Q8_0 matrices (ggml block_q8_0, quantize_row_q8_0's rounding of the
/// weights with the input rotation undone): gu_dst = [gate rows | up rows] (2 n_ff rows of n_embd), d_dst = down rows
/// (n_embd rows of n_ff).  The experts' geometry, 2560 x 640.
void ms_expert_to_q8_0(int ggml_type, const void* blob, int64_t n_embd, int64_t n_ff, size_t up_off, size_t down_off,
                       void* gu_dst, void* d_dst, void* stream, int up_type = -1, int d_type = -1);
/// The same for n (1..16) blobs in one launch each for gate/up and down: blob i's matrices at gu_dst + i * gu_stride
/// and d_dst + i * d_stride bytes.
void ms_experts_to_q8_0(int ggml_type, const uint8_t* const* blobs, int n, int64_t n_embd, int64_t n_ff,
                        size_t up_off, size_t down_off, void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride,
                        void* stream, int up_type = -1, int d_type = -1);
/// Same as ms_experts_to_q8_0, writing blob i to slot slots[i] (gu_dst + slots[i] * gu_stride, d_dst + slots[i] * d_stride).
void ms_experts_to_q8_0_slots(int ggml_type, const uint8_t* const* blobs, const int32_t* slots, int n,
                              int64_t n_embd, int64_t n_ff, size_t up_off, size_t down_off, void* gu_dst,
                              size_t gu_stride, void* d_dst, size_t d_stride, void* stream,
                              int up_type = -1, int d_type = -1);

/// Whether ms_expert_grouped covers an MS expert layer of this type and geometry.
bool ms_expert_supported(int type, int64_t n_embd, int64_t n_ff) noexcept;

/// native_expert_grouped for an MS layer (blob [gate | up | down] MS payloads): the same arguments and scratch
/// (native_expert_scratch_bytes); `limit` > 0 limits the SwiGLU.
void ms_expert_grouped(const NativeExpertLayout& L, float limit, const unsigned long long* grp_ptr,
                       const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                       const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,
                       void* scratch, float* out, void* stream, int64_t grid_groups);

}  // namespace basalt::kernels
