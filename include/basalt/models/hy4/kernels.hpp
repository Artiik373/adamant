// include/basalt/models/hy4/kernels.hpp - the Hy4-flash decode kernels no shared kernel covers: the hyper-connection
// mix, row norms, GPT-J RoPE, the router's top-k, the Engram post-op and the absorbed MLA attention.
//
// All activations are fp32 device pointers, T rows (one per token of a window). A row's result does not depend on T or
// on the row's place. Every launch enqueues on `stream`.
#pragma once

#include <cstdint>

namespace basalt::models::hy4 {

// ---- hyper-connections (iHC) --------------------------------------------------------------------------------------

/// The weights of one mix: `fn` [m][kHcFlat], `base` [m], `scale` [m / kHc] (pre, then post when m = 8).
struct HcMix {
    const float* fn;
    const float* base;
    const float* scale;
    int m;                                    ///< 8 (a layer's pre + post) or 4 (the head: pre only)
};

/// mixes = (fn . flat) * rsqrt(mean(flat^2) + eps) over the token's 4 x 3072 streams;
/// pre = sigmoid(mixes[0..4) * scale[0] + base) + hc_eps, post = magnitude * sigmoid(mixes[4..8) * scale[1] + base) + hc_eps;
/// y = sum_s pre[s] * streams[s].  `post` may be null (the head).
void hc_mix(const float* streams, const HcMix& w, int n_rows, float* pre, float* post, float* y, void* stream);
/// streams[t][s] += post[t][s] * y[t]
void hc_apply(float* streams, const float* y, const float* post, int n_rows, void* stream);
/// The embedding into the 4 streams: streams[t][s] = x[t].
void hc_expand(const float* x, float* streams, int n_rows, void* stream);

// ---- norms and RoPE -----------------------------------------------------------------------------------------------

/// out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w (w null: unweighted); in place when out == x.
void rms_norm_rows(const float* x, const float* w, int width, int n_rows, int row_stride, float eps, float* out,
                   void* stream);
/// GPT-J RoPE (interleaved pairs) of the `dim` dims at `offset` of every row: rows[n_rows][row_stride], the position of
/// row r is pos[r / rows_per_pos].  theta is the base.
void rope_gptj(float* rows, int n_rows, int row_stride, int offset, int dim, int rows_per_pos, const int32_t* pos,
               float theta, void* stream);

// ---- router -------------------------------------------------------------------------------------------------------

/// scores = sigmoid(logits); the kExpertsUsed experts with the largest score + bias are picked (ties: lower id);
/// weight = score / sum(picked scores) * kRoutedScale.  logits [T][kExperts]; ids, weights [T][kExpertsUsed].
void route_sigmoid_topk(const float* logits, const float* bias, int n_rows, int32_t* ids, float* weights, void* stream);

// ---- Engram -------------------------------------------------------------------------------------------------------

/// The rows a token reads ([T][kEngramHeads][kEngramHeadDim], fp32 from the table) -> the projections' input:
/// each head rounded through FP8 E4M3 with a per-head scale (amax / 448), then RMSNormed with its own weight
/// (`head_norm` [kEngramHeads][kEngramHeadDim], eps `eps`).  out [T][kEngramDim].
void engram_rows(const float* rows, const float* head_norm, float eps, int n_rows, float* out, void* stream);

struct EngramWeights {
    const float* key_bias;                    ///< [kHc][kEmbd]
    const float* norm_key;                    ///< [kHc][kEmbd]
    const float* norm_query;                  ///< [kHc][kEmbd]
    const float* value_bias;                  ///< [kEmbd]
    const float* pre_gamma;                   ///< [kEmbd]
    const float* per_hc_scale;                ///< [kHc][kEmbd]
    float rms_eps;                            ///< the key/query norms
    float engram_eps;                         ///< the unweighted norm of the value
};

/// With key = key_proj . e ([T][kHc][kEmbd], before bias) and value = value_proj . e ([T][kEmbd], before bias):
///   gate_s = sigmoid(signed_sqrt(dot(rms(key_s + kb_s) * nk_s, rms(stream_s) * nq_s) / sqrt(kEmbd)))
///   main = pre_gamma * rms(value + vb)   (unweighted norm, engram_eps)
///   stream_s += gate_s * main + per_hc_scale_s * main
void engram_apply(const float* key, const float* value, const EngramWeights& w, int n_rows, float* streams,
                  void* stream);

// ---- MLA attention (absorbed form) --------------------------------------------------------------------------------

/// The cache of one layer: the normed latent (kKvRank) and the roped shared key (kRope) per position, fp16, in a ring
/// of `cap` positions (position p in slot p % cap).
struct LatentCache {
    const uint16_t* ckv;                      ///< [cap][kKvRank] fp16
    const uint16_t* kpe;                      ///< [cap][kRope] fp16
    int cap;
};

/// q_lat[t][h][c] = sum_d q[t][h][d] * w_uk[h][d][c] for the head's nope dims d < kNope of q ([T][heads][kQkHead]).
/// w_uk: the key rows of kv_b ([heads][kNope + kVHead][kKvRank], fp16), the first kNope rows of each head.
void mla_absorb_q(const float* q, const uint16_t* kv_b, int heads, int n_rows, float* q_lat, void* stream);

/// Attention of each row t over its keys: positions [lo[t], hi[t]) of the cache, or, when `sel` is set, the
/// sel_n[t] positions sel[t][0..) (a negative sel_n[t]: the range). Score = (q_lat . ckv + q_pe . kpe) * scale, with a per-head sink logit `sink`
/// in the softmax that adds nothing to the sum.  o_lat[t][h][kKvRank].  q_pe is the roped dims of q
/// ([T][heads][kQkHead], dims kNope..).
void mla_attend(const float* q_lat, const float* q, const LatentCache& cache, const int32_t* lo, const int32_t* hi,
                const int32_t* sel, const int32_t* sel_n, int sel_stride, const float* sink, int heads, int n_rows,
                float scale, float* o_lat, void* stream);

/// out[t][h][v] = sigmoid(gate[t][h][v]) * sum_c o_lat[t][h][c] * w_uv[h][v][c]; w_uv: the value rows of kv_b
/// (the kVHead rows after the head's kNope key rows).  out [T][heads][kVHead].
void mla_value_out(const float* o_lat, const uint16_t* kv_b, const float* gate, int heads, int n_rows, float* out,
                   void* stream);

// ---- small helpers of the layer loop ------------------------------------------------------------------------------

/// y[t][o] = sum_i w[o][i] * x[t][i] (fp32 weights: the router, whose picks the rounding of a quantized matrix moves).
void gemv_f32(const float* w, const float* x, int n_in, int n_out, int n_rows, float* y, void* stream);
/// out[t] = table[ids[t]] for a BF16 table of `width` columns, as fp32.
void embed_bf16(const uint16_t* table, const int32_t* ids, int width, int n_rows, float* out, void* stream);
/// out[t][d] = sum_k weights[t][k] * rows[t * kExpertsUsed + k][d], d < width.
void expert_sum(const float* rows, const float* weights, int width, int n_rows, float* out, void* stream);
/// The cache rows of positions pos[t]: kva[t] is [normed latent (kKvRank) | roped shared key (kRope)], stored as fp16
/// at slot pos[t] % cap.
void latent_store(const float* kva, const int32_t* pos, int n_rows, uint16_t* ckv, uint16_t* kpe, int cap, void* stream);
/// y[i] = a[i] + b[i]
void add_rows(const float* a, const float* b, int n, float* y, void* stream);
/// y[i] += g[i] * f[i]
void add_scaled(float* y, const float* g, const float* f, int n, void* stream);
/// *out = the index of x's largest value (ties: the lower index), x [n]
void argmax_row(const float* x, int n, int32_t* out, void* stream);
/// The drafter's keys, every row t < n_rows the same: the target's picks src[0..src_n) then positions
/// p0[0] .. p0[0] + block - 1, so dst_n[t] = src_n + block; a negative src_n (every position) stays -1.
/// dst rows are kDflySelStride ints.
void sel_with_block(const int32_t* src, const int32_t* src_n, const int32_t* p0, int block, int n_rows, int32_t* dst,
                    int32_t* dst_n, void* stream);

// ---- compressed indexer (sparse layers) ---------------------------------------------------------------------------

struct IndexerPool {
    const float* kv_w;                        ///< [kIdxDim][kIdxDim]: values = kv_w . key
    const float* gate_w;                      ///< [kIdxDim][kIdxDim]: gates = gate_w . key + ape
    const float* ape;                         ///< [kIdxRatio][kIdxDim]
    const float* key_norm;                    ///< [kIdxDim]
    float eps;
};

/// Groups g in [g0, g1): their kIdxRatio token keys (`token_keys` [positions][kIdxDim], fp32, the normed
/// token_key_projection rows; position p in row p) pooled with a per-feature softmax over the group, RMSNormed with
/// key_norm, and the last kRope dims roped (GPT-J, theta kRopeThetaSparse) at the group's first position.
/// group_keys [groups][kIdxDim].
void indexer_pool(const float* token_keys, const IndexerPool& w, int g0, int g1, float* group_keys, void* stream);

/// Per row t: logits[g] = sum_h hw[t][h] * relu(q[t][h] . group_keys[g]) for the groups before the row's own
/// (g < pos[t] / kIdxRatio), hw scaled by 1 / sqrt(kIdxHeads * kIdxDim); the kIdxGroupTopk - 1 largest (ties: lower g)
/// and the row's own group give the positions g * kIdxRatio + i <= pos[t], written to sel[t][0..sel_n[t]).
/// q [T][kIdxHeads][kIdxDim] (roped at pos[t]), hw [T][kIdxHeads]; `scratch` holds indexer_scratch_floats(groups).
/// sel needs kIdxGroupTopk * kIdxRatio ints per row.
size_t indexer_scratch_floats(int groups, int n_rows);
void indexer_select(const float* q, const float* hw, const float* group_keys, int groups, const int32_t* pos,
                    int n_rows, float* scratch, int32_t* sel, int32_t* sel_n, void* stream);

/// The same per row at a position read on the device: groups = (pos[t] + 1) / kIdxRatio, logits at a row stride of
/// `max_groups`; a row at or below kIdxTopk positions writes sel_n[t] = -1, which mla_attend reads as "every position
/// in [lo, hi)".  Nothing here depends on the host, so it can sit in a graph.
void indexer_select_at(const float* q, const float* hw, const float* group_keys, int max_groups, const int32_t* pos,
                       int n_rows, float* scratch, int32_t* sel, int32_t* sel_n, void* stream);
/// indexer_pool for the group position pos[0] completes, and nothing when it completes none.
void indexer_pool_at(const float* token_keys, const IndexerPool& w, const int32_t* pos, float* group_keys, void* stream);
/// dst[pos[0]][0..width) = src[0..width)
void store_row(const float* src, float* dst, int width, const int32_t* pos, void* stream);

}  // namespace basalt::models::hy4
