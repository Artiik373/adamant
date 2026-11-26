// include/basalt/models/inkling2/qsa_kernels.hpp - the QSA indexer of Inkling 2's full-attention visits: a block key per
// 4 positions and, for each decode row, the key blocks its attention reads (the sink, the local window and the best
// scored of the rest).
//
// A block's key is rms(mean of its 4 positions' k projections) * k_norm in fp16. A row's score of block b is
// sum_h w_head[h] * relu(q_h . key_b) / sqrt(128), q_h = rms(q projection's head h) * q_norm. No rotary embedding.
// All pointers are device memory; `stream` may be null.
#pragma once

#include <cstdint>

namespace basalt::models::inkling2 {

constexpr int kQsaHeads = 4;     ///< indexer query heads
constexpr int kQsaDim = 128;     ///< indexer head width
constexpr int kQsaBlock = 4;     ///< positions of a key block
constexpr int kQsaRing = 32;     ///< positions of k projections a ring keeps (slot pos % kQsaRing)

/// out[r][n] = W[n] . x_r for n < n_out, W fp16 [n_out][kWidth]. x_r is row r of `x` (fp32) normed by `norm` (eps), or
/// row r of `x8` (q8_1 blocks, 36 bytes per 32 values) when x8 is set.
void qsa_project(const float* x, const void* x8, const float* norm, float eps, const uint16_t* w, int n_out, int rows,
                 float* out, void* stream);

/// The k projections raw [rows][kQsaDim] of rows at positions pos[0] + r: each row at a block's last position writes
/// that block's key to blocks [pos / 4][kQsaDim] (positions before pos[0] from the ring), then the ring takes the rows.
void qsa_keys(const float* raw, const int32_t* pos, int rows, const float* k_norm, float* ring, uint16_t* blocks,
              void* stream);

/// Row r (position pos[r], (pos[r] + 1) / 4 whole blocks) takes all its whole blocks when they are at most k_blocks,
/// else the first sink_blocks, the last local_blocks and the best scored of the rest, k_blocks in all: their indices in
/// ascending order at sel[r][0..k_blocks), -1 past the row's count. q [rows][kQsaHeads][kQsaDim] (normed in place),
/// scores [rows][max_blocks] scratch; max_blocks bounds every row's whole blocks.
void qsa_select(float* q, const float* q_norm, const float* w_head, const uint16_t* blocks, const int32_t* pos, int rows,
                int sink_blocks, int local_blocks, int k_blocks, int64_t max_blocks, float* scores, int32_t* sel,
                void* stream);

}  // namespace basalt::models::inkling2
