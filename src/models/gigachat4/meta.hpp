// src/models/gigachat4/meta.hpp - the GigaChat4 architecture: its geometry, which layers are which, and the host-side constants
// of its attention (YaRN frequencies, the score scale). Free of the runtime, so a CPU test can check it.
//
// A layer: x += post_attn_norm(attn(pre_attn_norm(x))); x += post_ffn_norm(ffn(pre_ffn_norm(x))), every norm a GatedNorm.
// attn is a Gated DeltaNet (sigmoid-gated output norm, times 2) or, on full_attention_layers, an absorbed MLA with a DSA indexer
// and a sigmoid output gate. ffn is a dense SwiGLU on layer 0, then 192 routed experts (sigmoid router, top-8, normalised, times
// 2.5) plus one shared expert; every SwiGLU clamps gate <= 10 and up to [-10, 10]. Logits are softcapped at 30.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace basalt::models::gigachat4 {

inline constexpr const char* kArch = "gigachat4";

struct Geometry {
    int n_layers = 32, n_embd = 2560, n_vocab = 128256;
    int n_ff_dense = 8192, n_ff_exp = 768, n_expert = 192, n_expert_used = 8, n_expert_shared = 1, first_dense = 1;
    float expert_weight_scale = 2.5f, swiglu_limit = 10.f, logit_softcap = 30.f, rms_eps = 1e-6f;
    int norm_gate = 16;           ///< the GatedNorm's bottleneck
    float norm_gain = 2.f;        ///< layernorm_gating_weight
    // MLA
    int heads = 32, q_rank = 1536, kv_rank = 512, nope = 128, rope = 64, v_dim = 128;
    float rope_theta = 1e5f, yarn_factor = 32.f, yarn_beta_fast = 32.f, yarn_beta_slow = 1.f, yarn_mscale_all_dim = 1.f;
    int yarn_orig_ctx = 32768;
    // DSA indexer
    int idx_heads = 64, idx_dim = 128, idx_topk = 2048;
    // GDN
    int gdn_k_heads = 16, gdn_v_heads = 32, gdn_k_dim = 128, gdn_v_dim = 128, gdn_conv = 4;
    float gdn_gate_scale = 2.f;
    int full_attn_every = 4;      ///< layers 3, 7, ..., 31
    /// The recurrent MTP drafter (EAGLE-3 style), layer n_layers of the checkpoint: h' = eh_proj [enorm(embed(tok)); hnorm(h)],
    /// then one layer of full attention (MLA with the indexer) and a dense SwiGLU, then shared_head (norm, head) for the logits.
    /// Each draft step feeds its own h' and token back into the same layer. Its embedding is its own copy.
    int mtp_layers = 1;

    bool is_full_attention(int layer) const { return layer % full_attn_every == full_attn_every - 1; }
    bool is_moe(int layer) const { return layer >= first_dense; }
    /// alpha_q and alpha_kv: the converter folds them into q_b and kv_b (the cache holds the unscaled normed latent).
    float alpha_q() const { return std::sqrt((float) n_embd / q_rank); }
    float alpha_kv() const { return std::sqrt((float) n_embd / kv_rank); }
};

inline double yarn_mscale(double scale, double mscale) { return scale <= 1 ? 1.0 : 0.1 * mscale * std::log(scale) + 1.0; }

/// The MLA's score scale: (nope + rope)^-1/2 times the YaRN mscale squared.
inline float mla_scale(const Geometry& g) {
    const double m = yarn_mscale(g.yarn_factor, g.yarn_mscale_all_dim);
    return (float) (m * m / std::sqrt((double) (g.nope + g.rope)));
}

/// The indexer's head-weight scale: idx_heads^-1/2 * idx_dim^-1/2.
inline float idx_weight_scale(const Geometry& g) { return (float) (1.0 / std::sqrt((double) g.idx_heads * g.idx_dim)); }

/// The YaRN inverse frequencies of the rope/2 pairs (transformers' _compute_yarn_parameters with truncation); the attention
/// factor is mscale/mscale_all_dim = 1 for this model, so the rotations are not scaled.
inline std::vector<float> yarn_inv_freq(const Geometry& g) {
    const int dim = g.rope, half = dim / 2;
    const double base = g.rope_theta, pi = 3.14159265358979323846;
    auto corr = [&](double rot) { return dim * std::log(g.yarn_orig_ctx / (rot * 2 * pi)) / (2 * std::log(base)); };
    double lo = std::max(0.0, std::floor(corr(g.yarn_beta_fast))), hi = std::min(dim - 1.0, std::ceil(corr(g.yarn_beta_slow)));
    if (lo == hi) hi += 0.001;
    std::vector<float> f(half);
    for (int i = 0; i < half; ++i) {
        const double pos = std::pow(base, (2.0 * i) / dim), extra = 1.0 / pos, inter = 1.0 / (g.yarn_factor * pos);
        const double ramp = std::min(1.0, std::max(0.0, (i - lo) / (hi - lo))), keep = 1.0 - ramp;
        f[i] = (float) (inter * (1.0 - keep) + extra * keep);
    }
    return f;
}

}  // namespace basalt::models::gigachat4
