// include/basalt/kernels/exl3_kernels.hpp - EXL3 (QTIP trellis) routed experts on the GPU.  The blob layout and the
// arithmetic are kernels/cpu/exl3.hpp's: three projection records [gate | up | down], each evaluated as
// y = H(H(x * suh) . W_inner) * svh with the block-128 Hadamard H.  Activations arrive as q8_1 like every native pack's.
#pragma once

#include "basalt/kernels/iq_kernels.hpp"

namespace basalt::kernels {

/// Whether `exl3_expert_grouped` covers these dimensions (the model's expert geometry only).
bool exl3_expert_supported(int type, int64_t n_embd, int64_t n_ff) noexcept;

/// native_expert_grouped for an EXL3 layer: same arguments and the same scratch (native_expert_scratch_bytes); `out`
/// rows and entry order are the grouped kernels'.  `limit` > 0 applies the SwiGLU limit.
void exl3_expert_grouped(const NativeExpertLayout& L, float limit, const unsigned long long* grp_ptr,
                         const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                         const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1,
                         void* scratch, float* out, void* stream, int64_t grid_groups);

/// The scratch exl3_expert_grouped takes for `cap` entries of a wide window (more than 128): the SwiGLU rows only.
size_t exl3_wide_scratch_bytes(int64_t cap, int64_t n_ff) noexcept;

/// One expert's gate and up into the prompt path's interleaved fp16 layout: row 2r = gate output r, row 2r+1 = up output
/// r, n_embd values each (W_full transposed, the transforms applied).  `blob` is the expert's first byte, `up_off` its
/// up record's offset.
void exl3_dequant_gu_f16(int type, const void* blob, size_t up_off, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                         void* stream);
/// The down record at `rec` into fp16 [n_embd rows][n_ff].
/// Debug: cycle sums of the dequant kernel's phases {decode, input transform, output transform, store, CTA, CTAs}.
void exl3_dequant_profile(bool on, bool reset, unsigned long long* out6);
void exl3_dequant_down_f16(int type, const void* rec, int64_t n_embd, int64_t n_ff, uint16_t* dst, void* stream);

/// The prompt path's expert projections without a dequantized W: x fp16 [m][2560] -> y fp32 [m][1280] interleaved like
/// exl3_dequant_gu_f16 (column 2r gate, 2r+1 up); down: x fp16 [m][640] -> y fp32 [m][2560].  Tensor-core MMA with the
/// trellis decoded into the B fragments; the Hadamard transforms run on the activations, not the weights.
void exl3_expert_gu_fused(int type, const void* blob, size_t up_off, const uint16_t* x, int64_t m, float* y, void* stream);
void exl3_expert_down_fused(int type, const void* rec, const uint16_t* x, int64_t m, float* y, void* stream);

/// A dense EXL3 projection (a record [trellis | suh | svh] of a [n_in x n_out] matrix) for 1..8 q8_1 activation columns:
/// y[c * n_out + r] = sum_i x_c[i] W[i][r].  native_mmvq's contract (n_in, n_out whole 128-blocks).
void exl3_mmvq(int type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// Rows [row0, row0 + rows) (both multiples of 128) of a dense record as fp16 [rows][n_in], the prompt path's GEMM operand.
void exl3_dequant_dense_f16(int type, const void* weights, int64_t k, int64_t n, int64_t row0, int64_t rows,
                            uint16_t* dst, void* stream);

}  // namespace basalt::kernels
