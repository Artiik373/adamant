// include/basalt/kernels/cpu/ms_expert.hpp - MS (Mad Shift 1MAD trellis) projections on the CPU.
//
// A payload is artifact/gguf_reader.hpp's (ms_tensor_type): n_out rows of ms_row_bytes(n_in, K), then int8
// signs[n_in]; y = d_r / sqrt(B) * sum_c v(state_c) * H(signs * x)_c with H the unnormalized block Hadamard of size
// B = ms_hadamard_block(n_in).  The rotated input is quantized to int8 per 128 columns, so a row is integer dot
// products of the decoded bytes against it (vpdpbusd on AVX-VNNI CPUs).
#pragma once

#include <cstdint>
#include <vector>

namespace basalt::kernels::cpu {

/// One token's input to one MS payload, rotated by its signs and quantized: q[c] in every byte of rep[c].
struct MsAct {
    std::vector<int32_t> rep;   ///< n_in words, the int8 q[c] replicated into each byte
    std::vector<float> sc;      ///< per 128 columns: the quantization step
    float corr = 0.f;           ///< -510 * sum_blk sc * sum q: the decoded values' offset, folded in once
    float rscale = 0.f;         ///< 1 / sqrt(B)
};

/// n_in a multiple of 128.  `signs` are the payload's (after its n_out rows).
void ms_act_prepare(const float* x, const int8_t* signs, int n_in, MsAct& a);

/// out[t][r] = row r of the payload's rows times token t, for rows [r0, r1); K is 3, 4 or 6.
void ms_rows(int k, const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0, int r1);

/// The same in plain C (the reference of the parity test).
void ms_rows_ref(int k, const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0, int r1);

}  // namespace basalt::kernels::cpu
