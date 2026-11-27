// include/basalt/models/gigachat4/kernels.hpp - the GigaChat4 kernels the shared ones do not cover.
//
// The projections are the shared Q8_0 product (basalt::kernels::native_mmvq), the router basalt::kernels::router (sigmoid,
// weight_scale 2.5) and the experts basalt::kernels::native_expert_grouped_clamped; everything else is here.
//
// All activations fp32, all pointers device memory, n_tok rows a call. `stream` may be null (the call then synchronises).
#pragma once

#include <cstdint>

namespace basalt::models::gigachat4 {

/// GatedNorm of `rows` rows of n values (row r at x + r * stride): y = rms_norm(x) * w, then
/// out = y * gain * sigmoid(down . silu(up . y)), up [16][n], down [n][16]. out may be x (same stride). add: out += that.
void gated_norm(const float* x, int stride, const float* w, const float* up, const float* down, float eps, float gain, int n, int rows,
                float* out, int out_stride, void* stream, bool add = false);

/// The absorbed MLA's inputs. q [t][heads][nope + rope]; q_abs [t][heads][rank] (k_b . q_nope); kv [t][rank + rope], its latent
/// already GatedNormed. Writes qcur [t][heads][rank + rope] = q_abs then the rotated rope part, and the cache row (latent, rotated
/// rope part) of row t at cell pos[t] (modulo `ring` when it is set). Rope pair i is (x[2i], x[2i+1]) at inv_freq[i] (rope / 2
/// values, device).
struct MlaShape {
    int heads = 32, nope = 128, rope = 64, rank = 512;
};
constexpr int kFp8RowBytes = 656;   ///< --kv fp8: 512 E4M3 codes, four fp32 scales (per 128), 64 BF16 rope values
void mla_prepare(const float* q, const float* q_abs, const float* kv, const float* inv_freq, const MlaShape& s, const int32_t* pos,
                 int n_tok, int ring, float* qcur, float* cache, void* stream);
/// The same with an fp16 cache (round to nearest even).
void mla_prepare(const float* q, const float* q_abs, const float* kv, const float* inv_freq, const MlaShape& s, const int32_t* pos,
                 int n_tok, int ring, float* qcur, uint16_t* cache, void* stream);
/// The same with an FP8 cache (kFp8RowBytes per cell).
void mla_prepare(const float* q, const float* q_abs, const float* kv, const float* inv_freq, const MlaShape& s, const int32_t* pos,
                 int n_tok, int ring, float* qcur, uint8_t* cache, void* stream);

/// The indexer's keys: k [t][dim] (wk . x) RMS-normed by norm_w, its first `rope` values rotated in halves (pair i is (x[i],
/// x[i + rope / 2]) at inv_freq[i]), written as fp16 to keys[pos[t] % ring] (ring 0: no ring).
void idx_keys(const float* k, const float* norm_w, float eps, const float* inv_freq, int dim, int rope, const int32_t* pos, int n_tok,
              int ring, uint16_t* keys, void* stream);
/// The indexer's queries q [t][heads][dim]: the first `rope` values of each head rotated in halves at pos[t], in place.
void idx_rope_queries(float* q, const float* inv_freq, int heads, int dim, int rope, const int32_t* pos, int n_tok, void* stream);

/// x[i] = cap * tanh(x[i] / cap).
void softcap(float* x, int64_t n, float cap, void* stream);

// ---- decode (decode.cu)

/// out[t] = the BF16 row ids[t] of `table` (n values a row; host-mapped memory is fine).
void embed_rows(const uint16_t* table, const int32_t* ids, int n, int rows, float* out, void* stream);

/// The GDN's causal conv: for each of `ch` channels, y = w . [state, x] over 4 taps (w [ch][4], the last tap on the current
/// input), then SiLU; state [ch][3] holds the last three inputs and advances by `rows` rows. x row t at x + t * x_stride.
void gdn_conv(const float* x, int x_stride, float* state, const float* w, int ch, int rows, float* y, void* stream);

/// The gated delta rule over `rows` rows, state [hv][dk][dv] fp32 in place. qkv [t][hk * dk | hk * dk | hv * dv] (after the
/// conv); value head h reads key head h / (hv / hk). q and k are L2-normed (eps 1e-6), q scaled by dk^-1/2. ba [t][hv | hv]
/// (b then a): beta = sigmoid(b), g = -exp(a_log) softplus(a + dt_bias). Then the output gate: o = rms_norm(o) * norm_w *
/// sigmoid(z) (z [t] at z + t * z_stride, [hv][dv]); y [t][hv][dv]. dk = dv = 128.
struct GdnShape {
    int hk = 16, hv = 32, dk = 128, dv = 128;
};
void gdn_step(float* state, const float* qkv, const float* ba, const float* a_log, const float* dt_bias, const float* z, int z_stride,
              const float* norm_w, float eps, const GdnShape& s, int rows, float* y, void* stream);

/// Per-head Q8_0 products: y[t][h][r] = W[h][r] . x[t][h] for rows r < `rows`, W [heads][rows][k] Q8_0 (34 bytes a 32
/// values), x[t][h] at x + t * x_tok + h * x_head (k values). k a multiple of 128.
void heads_q8_gemv(const void* w, const float* x, int x_head, int x_tok, int heads, int rows, int k, int n_tok, float* y, void* stream);

/// The absorbed MLA's attention over an fp16 cache of rows [rank + rope]: for row t, cells cells[t * cell_stride + j],
/// j < n_cells[t] (cells null: cells 0 .. n_cells[t] - 1), softmax(scale * qcur[t][h] . cache[cell]) . cache[cell][:rank]
/// -> out [t][heads][rank]. `max_cells` bounds n_cells (the launch's size: fixed for a captured graph). work: mla_work_floats.
size_t mla_work_floats(int heads, int rank, int max_cells, int n_tok);
void mla_attend(const float* qcur, const uint16_t* cache, const int32_t* cells, int cell_stride, const int32_t* n_cells, int max_cells,
                const MlaShape& s, float scale, int n_tok, float* work, float* out, void* stream);
/// The same over an FP8 cache (kFp8RowBytes per cell).
void mla_attend(const float* qcur, const uint8_t* cache, const int32_t* cells, int cell_stride, const int32_t* n_cells, int max_cells,
                const MlaShape& s, float scale, int n_tok, float* work, float* out, void* stream);

/// The indexer's scores: score[t][j] = sum_h w[t][h] relu(q[t][h] . keys[j]) for j < n_cells[t], -inf after (up to max_cells);
/// q [t][heads][dim], keys fp16 [cell][dim]. Then the top `k` cells of each row (k <= max_cells) into cells [t][k] (when
/// n_cells[t] <= k: 0 .. n_cells[t] - 1) and their count into n_sel[t].
void idx_topk(const float* q, const float* w, const uint16_t* keys, const int32_t* n_cells, int max_cells, int heads, int dim, int k,
              int n_tok, float* scores, int32_t* cells, int32_t* n_sel, void* stream);

/// y [t][rows] = W [rows][k] . x[t] (fp32 weights; one warp a row).
void f32_gemv(const float* w, const float* x, int rows, int k, int n_tok, float* y, void* stream);

/// h [t][n] = min(g, limit) * sigmoid(min(g, limit)) * clamp(u, -limit, limit), gu [t][g n | u n].
void swiglu_clamped(const float* gu, int n, int rows, float limit, float* h, void* stream);

/// x[i] *= sigmoid(g[i]).
void mul_sigmoid(float* x, const float* g, int64_t n, void* stream);

/// The expert plan of `rows` rows of k routes: the distinct experts of ids [t][k] (sorted), each a group of the entries that
/// chose it; entry i = t * k + j reads token t and writes row i of the experts' output. grp_ptr[g] = table[expert].
/// rows * k <= 256. counts (may be null): counts[expert] += its entries.
void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* table, unsigned long long* grp_ptr,
              int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok, uint32_t* counts, void* stream);

/// out[t] = shared[t] + sum_j w[t][j] experts[t * k + j] (n values a row).
void moe_combine(const float* experts, const float* w, const float* shared, int k, int n, int rows, float* out, void* stream);

/// Same with indirection map[t * k + j] indexing `experts` rows (for prefill grouped experts).
void moe_combine_mapped(const float* experts, const int32_t* map, const float* w, const float* shared, int k, int n, int rows,
                        float* out, void* stream);

/// Split gate/up SwiGLU for MMQ prefill NVFP4 experts: gu [rows, 2 * n_ff], clamped gate <= limit, up in [-limit, limit],
/// applying each expert's {s_gate, s_up, s_down} tail scales. row_expert (if non-null) gives each row's expert index directly.
void swiglu_split_clamped(const float* gu, float* h, int64_t rows, int n_ff, float limit, const uint8_t* blobs, size_t stride,
                          size_t tail_off, const int32_t* bounds, int n, void* stream,
                          const int32_t* row_expert = nullptr);

/// Stable counting sort of `rows * k` routed pairs by expert [0, n_expert): writes bounds[n_expert + 1], tok[pairs],
/// dst[pairs], exp[pairs]. scratch requires moe_sort_scratch(rows * k, n_expert) bytes.
size_t moe_sort_scratch(int pairs, int n_expert);
void moe_sort(const int32_t* ids, int rows, int k, int n_expert, void* scratch, int32_t* bounds, int32_t* tok, int32_t* dst,
              int32_t* exp, void* stream);

/// Transpose indices ids[h * rows + t] = t * heads + h and bounds[h] = h * rows for head-major batched MMQ.
void head_major_ids(int32_t* ids, int32_t* bounds, int heads, int rows, void* stream);

/// pos[t] = pos0 + t, n_cells[t] = pos0 + t + 1 for t < n.
void set_rows(int32_t* pos, int32_t* n_cells, int n, int pos0, void* stream);
/// Advances pos[0] += 1 and n_cells[0] = pos[0] + 1 on device (for graph-captured MTP drafting).
void advance_step(int32_t* pos, int32_t* n_cells, void* stream);
/// A greedy step's end (one row): out[count % cap] = tok[0], then count, pos and n_cells advance by one.
void greedy_advance(const int32_t* tok, int32_t* pos, int32_t* n_cells, int32_t* out, int32_t* count, int cap, void* stream);

/// ids[t] = argmax of x[t] (n values; the lower index wins a tie).
void argmax_rows(const float* x, int64_t n, int rows, int32_t* ids, void* stream);
/// Greedy argmax token id and its softmax probability for one row of n logits.
void argmax_prob_row(const float* x, int64_t n, int32_t* out_id, float* out_prob, void* stream);

}  // namespace basalt::models::gigachat4
