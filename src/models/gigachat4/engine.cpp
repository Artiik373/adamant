// src/models/gigachat4/engine.cpp - the GigaChat4 runtime (see engine.hpp).
#include "models/gigachat4/engine.hpp"

#include "basalt/core/checkpoint_regions.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/log.hpp"
#include "basalt/models/gigachat4/kernels.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>

namespace basalt::models::gigachat4 {
namespace {

namespace K = basalt::kernels;
namespace mmq = basalt::prefill::mmq;
constexpr int kMaxRows = 8;
constexpr int kMaxPrefill = 512;
constexpr int kIdxBatch = 32;
constexpr int kQ80 = 8, kF32 = 0, kBF16 = 30, kNVFP4 = 40;

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            err = std::string(#x) + ": " + cudaGetErrorString(e_);                               \
            return false;                                                                        \
        }                                                                                        \
    } while (0)

struct Norm {
    const float *w = nullptr, *up = nullptr, *down = nullptr;
};
struct Mlp {
    const void *gu = nullptr, *down = nullptr;
    int n_ff = 0;
};
struct Layer {
    Norm attn_norm, attn_post, ffn_norm, ffn_post;
    bool full = false, moe = false;
    // GDN
    const void *qkvz = nullptr, *ba = nullptr, *qkvzba = nullptr, *out = nullptr;
    const float *conv = nullptr, *a_log = nullptr, *dt_bias = nullptr, *gnorm = nullptr;
    float *conv_state = nullptr, *state = nullptr;
    // MLA and its indexer
    const void *q_a = nullptr, *q_b = nullptr, *kv_a = nullptr, *k_b = nullptr, *v_b = nullptr, *o = nullptr, *gate = nullptr;
    Norm q_norm, kv_norm;
    const void *wq_b = nullptr, *wk = nullptr, *mla_in = nullptr;
    const float *k_norm = nullptr, *idx_w = nullptr;
    uint8_t* cache = nullptr;
    uint16_t* keys = nullptr;
    // FFN
    const float *router = nullptr, *router_bias = nullptr;
    Mlp mlp;   // the dense SwiGLU or the shared expert
};

struct Graph {
    cudaGraphExec_t exec = nullptr;
};

int bucket_of(int cells) {
    int b = 256;
    while (b < cells) b *= 2;
    return b;
}
constexpr int kGenCap = 1 << 16;

}  // namespace

struct Engine::Impl {
    Geometry g;
    EngineOptions opt;
    cudaStream_t st = nullptr;
    bool kv_fp8 = false;
    size_t cache_row_bytes = 0;
    // weights
    void* vram = nullptr;
    std::vector<uint8_t*> host_experts;   // per MoE layer, pinned and mapped
    void* slots = nullptr;
    int n_slots = 0;
    unsigned long long* table = nullptr;  // [31][192] device: where each expert's blob is read from
    std::vector<unsigned long long> table_h;
    std::vector<int32_t> slot_of;         // [31 * 192]: slot or -1
    uint32_t* counts = nullptr;           // [31][192] device: reads since load
    uint8_t* host_dense = nullptr;        // the dense section, pinned and mapped (the embedding is read from here)
    const uint16_t* embd = nullptr;
    const void* head = nullptr;
    Norm out_norm;
    std::vector<Layer> L;
    K::NativeExpertLayout xl;
    size_t blob = 0, slot_stride = 0;
    float* inv_freq = nullptr;
    float mla_scale_v = 0.f;
    // GDN contiguous state + snapshot for speculative rollback
    float *gdn_mem = nullptr, *gdn_snap = nullptr, *prev_x = nullptr;
    size_t gdn_state_floats = 0;          // 24 GDN layers (conv_state + state), excluding prev_x
    size_t gdn_total_floats = 0;          // gdn_state_floats + n_embd (prev_x)
    int n_gdn = 0;
    std::vector<int> gdn_layers;
    std::vector<float*> gdn_proj_snap, gdn_bav_snap;
    int64_t state_pos = 0;
    int64_t win_pos = 0;
    int win_len = 0;
    bool win_has_snap = false;
    // MTP drafter (EAGLE-3 style)
    bool has_mtp = false;
    bool mtp_on = false;
    Layer mtp_layer;
    Norm mtp_enorm, mtp_hnorm, mtp_head_norm;
    const void* mtp_eh = nullptr;
    const uint16_t* mtp_embd = nullptr;
    const void* mtp_head = nullptr;
    float *mtp_x = nullptr, *mtp_e = nullptr, *mtp_hin = nullptr, *mtp_cat = nullptr, *mtp_logit = nullptr;
    float* tap_x = nullptr;               // [kMaxRows][E] device: un-normed target hidden states of the last window
    int32_t tap_tok[kMaxRows] = {};
    int64_t tap_pos0 = 0;
    int tap_n = 0;
    int64_t mtp_synced_pos = 0;
    int32_t *d_draft_id = nullptr, *pin_draft_id = nullptr;
    float *d_draft_prob = nullptr, *pin_draft_prob = nullptr;
    int pending_drafts = 0;
    // activations
    float *x, *h, *y, *proj, *bav, *conv_out, *go, *qa, *kva, *q, *qabs, *qcur, *lat, *att, *ag, *iq, *ik, *iw, *scores;
    float *rlog, *rw, *ex, *sgu, *sh, *fgu, *fh, *mla_ffn_pool, *logit, *work;
    int32_t *rid, *cells, *nsel, *pos, *ncells, *tok, *gen_out, *gen_i;
    unsigned long long* grp_ptr;
    int32_t *grp_start, *n_groups, *ent_dst, *ent_tok;
    void *xq, *xscratch;
    // prefill MMQ buffers
    std::unique_ptr<mmq::Context> mmq_ctx;
    std::set<std::pair<int, int>> mmq_loaded;
    std::vector<const uint8_t*> expert_ptrs, layer_contig;
    void *mmq_xq = nullptr, *mmq_xq_h = nullptr, *moe_sort_scr = nullptr;
    int32_t *mmq_ident = nullptr, *mmq_bounds_d = nullptr, *heads_ids = nullptr, *heads_bounds = nullptr;
    uint8_t *moe_grp_gu = nullptr, *moe_grp_d = nullptr;
    void *moe_xq1 = nullptr, *moe_xq2 = nullptr;
    float* moe_scales = nullptr;
    int32_t *moe_tok_d = nullptr, *moe_bounds_d = nullptr, *moe_map_d = nullptr, *moe_exp_d = nullptr, *pin_moe_i = nullptr;
    size_t moe_gub = 0, moe_gb = 0, moe_db = 0, moe_tail_pad = 0;

    int32_t* pin_i = nullptr;             // pinned host staging: tokens, positions
    std::map<int, Graph> graphs;          // Engine::greedy step by attention bucket
    std::map<std::tuple<int, int, int, int>, Graph> dec_graphs;  // decode_forward by (n, bucket, save_gdn, mtp)
    std::map<std::pair<int, int>, Graph> mtp_graphs;             // draft_begin chain by (count, bucket)

    template <typename T>
    bool alloc(T*& p, size_t n, std::string& err) {
        CK(cudaMalloc((void**) &p, std::max<size_t>(n, 1) * sizeof(T)));
        CK(cudaMemsetAsync(p, 0, std::max<size_t>(n, 1) * sizeof(T), st));
        return true;
    }
    void ensure_mmq(int type, int rows) {
        if (mmq_loaded.insert({type, rows}).second) mmq::load(type, rows);
    }
    void quantize_h(int n) {
        if (n > kMaxRows && mmq_xq_h && mmq::built())
            mmq::quantize(h, nullptr, mmq_xq_h, kQ80, g.n_embd, g.n_embd, n, st);
        else
            K::native_quantize_q8_1(h, xq, g.n_embd, n, st);
    }
    void mmvq(const void* w, const float* xin, int n_in, int n_out, int n, float* out) {
        if (n > kMaxRows && mmq_xq && mmq::built() && mmq::fits(kQ80, n_out)) {
            ensure_mmq(kQ80, n_out);
            mmq::quantize(xin, nullptr, mmq_xq, kQ80, n_in, n_in, n, st);
            mmq::Product p;
            p.w = w; p.type = kQ80; p.w_rows = n_out; p.w_cols = n_in;
            p.expert_bytes = mmq::matrix_bytes(kQ80, n_out, n_in);
            p.n = 1; p.xq = mmq_xq; p.bounds = mmq_bounds_d + 2 * n; p.ids = mmq_ident;
            p.total_rows = n; p.max_rows = n; p.dst = out; p.ld_dst = n_out;
            mmq_ctx->run(p, st);
            return;
        }
        for (int off = 0; off < n; off += kMaxRows) {
            const int c = std::min(kMaxRows, n - off);
            K::native_quantize_q8_1(xin + (size_t) off * n_in, xq, n_in, c, st);
            K::native_mmvq(kQ80, w, xq, out + (size_t) off * n_out, n_in, n_out, c, st);
        }
    }
    void mmvq_h(const void* w, int n_out, int n, float* out) {
        if (n > kMaxRows && mmq_xq_h && mmq::built() && mmq::fits(kQ80, n_out)) {
            ensure_mmq(kQ80, n_out);
            mmq::Product p;
            p.w = w; p.type = kQ80; p.w_rows = n_out; p.w_cols = g.n_embd;
            p.expert_bytes = mmq::matrix_bytes(kQ80, n_out, g.n_embd);
            p.n = 1; p.xq = mmq_xq_h; p.bounds = mmq_bounds_d + 2 * n; p.ids = mmq_ident;
            p.total_rows = n; p.max_rows = n; p.dst = out; p.ld_dst = n_out;
            mmq_ctx->run(p, st);
            return;
        }
        if (n <= kMaxRows) { K::native_mmvq(kQ80, w, xq, out, g.n_embd, n_out, n, st); return; }
        mmvq(w, h, g.n_embd, n_out, n, out);
    }
    void heads_product(const void* w, const float* xin, int xh, int xt, int H, int rows, int k, int n, float* out) {
        if (n > kMaxRows && mmq_xq && mmq::built() && mmq::fits(kQ80, rows)) {
            ensure_mmq(kQ80, rows);
            head_major_ids(heads_ids, heads_bounds, H, n, st);
            mmq::quantize(xin, heads_ids, mmq_xq, kQ80, k, xh, (int64_t) H * n, st);
            mmq::Product p;
            p.w = w; p.type = kQ80; p.w_rows = rows; p.w_cols = k;
            p.expert_bytes = mmq::matrix_bytes(kQ80, rows, k);
            p.n = H; p.xq = mmq_xq; p.bounds = heads_bounds; p.ids = heads_ids;
            p.total_rows = (int64_t) H * n; p.max_rows = n; p.dst = out; p.ld_dst = rows;
            mmq_ctx->run(p, st);
            return;
        }
        heads_q8_gemv(w, xin, xh, xt, H, rows, k, n, out, st);
    }
    void norm(const Norm& nm, const float* in, int stride, int n_v, int rows, float* out, int out_stride, bool add = false) {
        gated_norm(in, stride, nm.w, nm.up, nm.down, g.rms_eps, g.norm_gain, n_v, rows, out, out_stride, st, add);
    }
    void attention(Layer& l, int n, int bucket);
    void ffn(Layer& l, int li, int n);
    void moe_prompt(Layer& l, int li, int n);
    void step(int n, int bucket, bool save_gdn = false, bool need_logits = true);
    void mtp_step_batch(const int32_t* h_tokens, const float* h_prev, const float* h_rest, int n, int pos0);
    void mtp_step_one(const int32_t* tok_d, const float* h_in, int bucket, int out_idx);
};

Engine::Engine() : d_(new Impl) {}
Engine::~Engine() {
    if (!d_) return;
    for (auto& [k, gr] : d_->graphs)
        if (gr.exec) cudaGraphExecDestroy(gr.exec);
    for (auto& [k, gr] : d_->dec_graphs)
        if (gr.exec) cudaGraphExecDestroy(gr.exec);
    for (auto& [k, gr] : d_->mtp_graphs)
        if (gr.exec) cudaGraphExecDestroy(gr.exec);
    // the process exit releases the device and pinned memory
}

const Geometry& Engine::geometry() const { return d_->g; }
const float* Engine::logits() const { return d_->logit; }
int Engine::cached_experts() const { return d_->n_slots; }
bool Engine::kv_fp8() const { return d_->kv_fp8; }
bool Engine::has_mtp() const { return d_->has_mtp; }
bool Engine::mtp_active() const { return d_->mtp_on; }
int Engine::max_ctx() const { return d_->opt.max_ctx; }
int Engine::prefill_chunk_size() const { return kMaxPrefill; }
void* Engine::cuda_stream() const { return d_->st; }
void Engine::sync() const { cudaStreamSynchronize(d_->st); }

int64_t Engine::kv_bytes() const {
    const Impl& D = *d_;
    const int full_layers = D.g.n_layers / D.g.full_attn_every + (D.mtp_on ? 1 : 0);
    return (int64_t) full_layers * D.opt.max_ctx * ((int64_t) D.cache_row_bytes + (int64_t) D.g.idx_dim * 2);
}

bool Engine::load(const EngineOptions& o, std::string& err) {
    Impl& D = *d_;
    D.opt = o;
    D.kv_fp8 = o.kv == "fp8";
    const Geometry& g = D.g;
    const int H = g.heads, W_ = g.kv_rank + g.rope;
    D.cache_row_bytes = D.kv_fp8 ? (size_t) kFp8RowBytes : (size_t) W_ * 2;
    CK(cudaStreamCreateWithFlags(&D.st, cudaStreamNonBlocking));
    core::ModelFile mf;
    if (!mf.open(o.pack, err) || !mf.load_meta(err)) return false;
    const core::Json& meta = mf.meta();
    const core::Json* arch = meta.get("arch");
    if (!arch || arch->s != kArch) {
        err = o.pack + ": not a gigachat4 pack";
        return false;
    }
    const core::ModelFile::Section* dsec = mf.find("dense");
    const core::ModelFile::Section* esec = mf.find("experts");
    const core::Json* tens = meta.get("tensors");
    if (!dsec || !esec || !tens || !tens->is_object()) {
        err = o.pack + ": no dense or experts section";
        return false;
    }
    D.xl = K::native_expert_layout(kNVFP4, kNVFP4, g.n_embd, g.n_ff_exp, g.swiglu_limit);
    D.blob = D.xl.bytes;
    if (!K::native_expert_supported(kNVFP4, kNVFP4, g.n_embd, g.n_ff_exp) || (uint64_t) meta.get("expert_blob")->i64() != D.blob ||
        esec->bytes != (uint64_t) (g.n_layers - 1) * g.n_expert * D.blob) {
        err = o.pack + ": the expert blobs do not match the NVFP4 layout";
        return false;
    }
    // dense: read whole into pinned host memory, then everything but the embeddings into VRAM
    CK(cudaHostAlloc((void**) &D.host_dense, dsec->bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    if (!core::read_range_at(mf.range("dense"), 0, D.host_dense, dsec->bytes, err)) return false;
    struct T {
        int type;
        uint64_t off, bytes;
    };
    std::map<std::string, T> tab;
    uint64_t vram_bytes = 0;
    const size_t q8_past = mmq::built() ? mmq::read_past_bytes(kQ80, 16384) : 0;
    auto is_bundled_tail = [](const std::string& n) {
        auto ends = [&](const char* suf) {
            const size_t sl = std::strlen(suf);
            return n.size() >= sl && n.compare(n.size() - sl, sl, suf) == 0;
        };
        return ends(".gdn.ba") || ends(".mla.kv_a") || ends(".idx.wk");
    };
    for (const auto& [name, t] : tens->members) {
        tab[name] = {(int) t.get("type")->i64(), (uint64_t) t.get("offset")->i64(), (uint64_t) t.get("bytes")->i64()};
    }
    for (const auto& [name, t] : tab) {
        if (name.find("token_embd") != std::string::npos || is_bundled_tail(name)) continue;
        uint64_t sz = t.bytes;
        if (name.size() >= 9 && name.compare(name.size() - 9, 9, ".gdn.qkvz") == 0) {
            const std::string ba_n = name.substr(0, name.size() - 4) + "ba";
            if (tab.count(ba_n)) sz += tab[ba_n].bytes;
        } else if (name.size() >= 8 && name.compare(name.size() - 8, 8, ".mla.q_a") == 0) {
            const std::string pref = name.substr(0, name.size() - 7);
            if (tab.count(pref + "mla.kv_a")) sz += tab[pref + "mla.kv_a"].bytes;
            if (tab.count(pref + "idx.wk")) sz += tab[pref + "idx.wk"].bytes;
        }
        vram_bytes += (sz + (t.type == kQ80 ? q8_past : 0) + 255) / 256 * 256;
    }
    CK(cudaMalloc(&D.vram, vram_bytes));
    CK(cudaMemsetAsync(D.vram, 0, vram_bytes, D.st));
    std::map<std::string, const void*> dev;
    uint64_t at = 0;
    for (const auto& [name, t] : tab) {
        if (name.find("token_embd") != std::string::npos || is_bundled_tail(name)) continue;
        uint8_t* dst = (uint8_t*) D.vram + at;
        CK(cudaMemcpyAsync(dst, D.host_dense + t.off, t.bytes, cudaMemcpyHostToDevice, D.st));
        dev[name] = dst;
        uint64_t sz = t.bytes;
        if (name.size() >= 9 && name.compare(name.size() - 9, 9, ".gdn.qkvz") == 0) {
            const std::string ba_n = name.substr(0, name.size() - 4) + "ba";
            if (auto it = tab.find(ba_n); it != tab.end()) {
                CK(cudaMemcpyAsync(dst + sz, D.host_dense + it->second.off, it->second.bytes, cudaMemcpyHostToDevice, D.st));
                dev[ba_n] = dst + sz;
                sz += it->second.bytes;
            }
        } else if (name.size() >= 8 && name.compare(name.size() - 8, 8, ".mla.q_a") == 0) {
            const std::string pref = name.substr(0, name.size() - 7);
            for (const std::string& sn : {pref + "mla.kv_a", pref + "idx.wk"}) {
                if (auto it = tab.find(sn); it != tab.end()) {
                    CK(cudaMemcpyAsync(dst + sz, D.host_dense + it->second.off, it->second.bytes, cudaMemcpyHostToDevice, D.st));
                    dev[sn] = dst + sz;
                    sz += it->second.bytes;
                }
            }
        }
        at += (sz + (t.type == kQ80 ? q8_past : 0) + 255) / 256 * 256;
    }
    bool missing = false;
    auto W = [&](const std::string& n, int type) -> const void* {
        auto it = dev.find(n);
        if (it == dev.end() || tab[n].type != type) {
            if (!missing) err = o.pack + ": tensor " + n + (it == dev.end() ? " missing" : " has the wrong type");
            missing = true;
            return nullptr;
        }
        return it->second;
    };
    auto F = [&](const std::string& n) { return (const float*) W(n, kF32); };
    auto N = [&](const std::string& n) { return Norm{F(n + ".w"), F(n + ".up"), F(n + ".down")}; };
    {
        auto it = tab.find("token_embd");
        if (it == tab.end() || it->second.type != kBF16) {
            err = o.pack + ": no BF16 token_embd";
            return false;
        }
        CK(cudaHostGetDevicePointer((void**) &D.embd, D.host_dense + it->second.off, 0));
    }
    D.head = W("output", kQ80);
    D.out_norm = N("output_norm");
    D.L.resize(g.n_layers);
    D.n_gdn = 0;
    D.gdn_layers.assign(g.n_layers, -1);
    for (int li = 0; li < g.n_layers; ++li)
        if (!g.is_full_attention(li)) D.gdn_layers[li] = D.n_gdn++;
    const size_t conv_floats = (size_t) 3 * (2 * g.gdn_k_heads * g.gdn_k_dim + g.gdn_v_heads * g.gdn_v_dim);
    const size_t state_floats = (size_t) g.gdn_v_heads * g.gdn_k_dim * g.gdn_v_dim;
    const size_t gdn_layer_floats = conv_floats + state_floats;
    D.gdn_state_floats = (size_t) D.n_gdn * gdn_layer_floats;
    D.gdn_total_floats = D.gdn_state_floats + (size_t) g.n_embd;
    if (!D.alloc(D.gdn_mem, D.gdn_total_floats, err)) return false;
    D.prev_x = D.gdn_mem + D.gdn_state_floats;
    const int NP = 2 * g.gdn_k_heads * g.gdn_k_dim + 2 * g.gdn_v_heads * g.gdn_v_dim;
    if (o.spec >= 2) {
        if (!D.alloc(D.gdn_snap, D.gdn_total_floats, err)) return false;
        D.gdn_proj_snap.resize(D.n_gdn, nullptr);
        D.gdn_bav_snap.resize(D.n_gdn, nullptr);
        for (int gi = 0; gi < D.n_gdn; ++gi)
            if (!D.alloc(D.gdn_proj_snap[gi], (size_t) kMaxRows * NP, err) ||
                !D.alloc(D.gdn_bav_snap[gi], (size_t) kMaxRows * 2 * g.gdn_v_heads, err))
                return false;
    }

    auto load_mla = [&](Layer& l, const std::string& b, bool alloc_kv) -> bool {
        l.q_a = W(b + "mla.q_a", kQ80), l.q_b = W(b + "mla.q_b", kQ80), l.kv_a = W(b + "mla.kv_a", kQ80);
        l.k_b = W(b + "mla.k_b", kQ80), l.v_b = W(b + "mla.v_b", kQ80), l.o = W(b + "mla.o", kQ80), l.gate = W(b + "mla.gate", kQ80);
        l.q_norm = N(b + "mla.q_a_norm"), l.kv_norm = N(b + "mla.kv_norm");
        l.wq_b = W(b + "idx.wq_b", kQ80), l.wk = W(b + "idx.wk", kQ80), l.k_norm = F(b + "idx.k_norm"), l.idx_w = F(b + "idx.w");
        l.mla_in = l.q_a;
        if (alloc_kv) {
            CK(cudaMalloc((void**) &l.cache, (size_t) o.max_ctx * D.cache_row_bytes));
            CK(cudaMemsetAsync(l.cache, 0, (size_t) o.max_ctx * D.cache_row_bytes, D.st));
            CK(cudaMalloc((void**) &l.keys, (size_t) o.max_ctx * g.idx_dim * 2));
            CK(cudaMemsetAsync(l.keys, 0, (size_t) o.max_ctx * g.idx_dim * 2, D.st));
        }
        return true;
    };

    for (int li = 0; li < g.n_layers; ++li) {
        Layer& l = D.L[li];
        const std::string b = "blk." + std::to_string(li) + ".";
        l.full = g.is_full_attention(li);
        l.moe = g.is_moe(li);
        l.attn_norm = N(b + "attn_norm"), l.attn_post = N(b + "attn_post_norm"), l.ffn_norm = N(b + "ffn_norm"), l.ffn_post = N(b + "ffn_post_norm");
        if (l.full) {
            if (!load_mla(l, b, true)) return false;
        } else {
            l.qkvz = W(b + "gdn.qkvz", kQ80), l.ba = W(b + "gdn.ba", kQ80), l.out = W(b + "gdn.out", kQ80);
            l.qkvzba = l.qkvz;
            l.conv = F(b + "gdn.conv"), l.a_log = F(b + "gdn.a_log"), l.dt_bias = F(b + "gdn.dt_bias"), l.gnorm = F(b + "gdn.norm");
            float* base = D.gdn_mem + (size_t) D.gdn_layers[li] * gdn_layer_floats;
            l.conv_state = base;
            l.state = base + conv_floats;
        }
        if (l.moe) {
            l.router = F(b + "router"), l.router_bias = F(b + "router_bias");
            l.mlp = {W(b + "shexp.gu", kQ80), W(b + "shexp.down", kQ80), g.n_ff_exp * g.n_expert_shared};
        } else {
            l.mlp = {W(b + "ffn.gu", kQ80), W(b + "ffn.down", kQ80), g.n_ff_dense};
        }
    }
    D.has_mtp = tab.count("mtp.eh") > 0;
    D.mtp_on = D.has_mtp && o.spec >= 2;
    if (D.mtp_on) {
        D.mtp_enorm = N("mtp.enorm");
        D.mtp_hnorm = N("mtp.hnorm");
        D.mtp_head_norm = N("mtp.head_norm");
        D.mtp_eh = W("mtp.eh", kQ80);
        if (auto it = tab.find("mtp.token_embd"); it != tab.end()) {
            if (it->second.type != kBF16) { err = o.pack + ": mtp.token_embd not BF16"; return false; }
            CK(cudaHostGetDevicePointer((void**) &D.mtp_embd, D.host_dense + it->second.off, 0));
        } else {
            D.mtp_embd = D.embd;
        }
        D.mtp_head = tab.count("mtp.head") ? W("mtp.head", kQ80) : D.head;
        D.mtp_layer.full = true;
        D.mtp_layer.moe = false;
        D.mtp_layer.attn_norm = N("mtp.attn_norm");
        D.mtp_layer.attn_post = N("mtp.attn_post_norm");
        D.mtp_layer.ffn_norm = N("mtp.ffn_norm");
        D.mtp_layer.ffn_post = N("mtp.ffn_post_norm");
        if (!load_mla(D.mtp_layer, "mtp.", true)) return false;
        D.mtp_layer.mlp = {W("mtp.ffn.gu", kQ80), W("mtp.ffn.down", kQ80), g.n_ff_dense};
    }
    if (missing) return false;
    {
        const auto f = yarn_inv_freq(g);
        CK(cudaMalloc((void**) &D.inv_freq, f.size() * 4));
        CK(cudaMemcpyAsync(D.inv_freq, f.data(), f.size() * 4, cudaMemcpyHostToDevice, D.st));
        D.mla_scale_v = mla_scale(g);
    }
    // activations (sized for kMaxPrefill chunks; logit stays kMaxRows, scores stays kIdxBatch)
    const int R = kMaxPrefill, RD = kMaxRows, E = g.n_embd;
    const int RK = R * g.n_expert_used, RKD = RD * g.n_expert_used;
    const int work_cells = std::min(o.max_ctx, g.idx_topk);
    const size_t n_q = (size_t) R * H * (g.nope + g.rope), n_qabs = (size_t) R * H * g.kv_rank,
                 n_qcur = (size_t) R * H * W_, n_lat = (size_t) R * H * g.kv_rank,
                 n_att = (size_t) R * H * g.v_dim, n_ag = (size_t) R * H * g.v_dim,
                 n_iq = (size_t) R * g.idx_heads * g.idx_dim, n_ik = (size_t) R * g.idx_dim,
                 n_iw = (size_t) R * g.idx_heads;
    const size_t n_rlog = (size_t) R * g.n_expert, n_rw = (size_t) RK,
                 n_ex = (size_t) RK * E, n_sgu = (size_t) R * 2 * g.n_ff_exp,
                 n_sh = (size_t) R * E,
                 n_fgu = std::max<size_t>((size_t) R * 2 * g.n_ff_dense, (size_t) RK * 2 * g.n_ff_exp),
                 n_fh = std::max<size_t>((size_t) R * g.n_ff_dense, (size_t) RK * g.n_ff_exp);
    auto a64 = [](size_t nv) { return (nv + 63) / 64 * 64; };
    const size_t mla_pool = a64(n_q) + a64(n_qabs) + a64(n_qcur) + a64(n_lat) + a64(n_att) + a64(n_ag) + a64(n_iq) + a64(n_ik) + a64(n_iw);
    const size_t ffn_pool = a64(n_rlog) + a64(n_rw) + a64(n_ex) + a64(n_sgu) + a64(n_sh) + a64(n_fgu) + a64(n_fh);
    bool ok = D.alloc(D.x, R * E, err) && D.alloc(D.h, R * E, err) && D.alloc(D.y, R * E, err) && D.alloc(D.proj, R * NP, err) &&
              D.alloc(D.bav, R * 2 * g.gdn_v_heads, err) && D.alloc(D.conv_out, R * (NP - g.gdn_v_heads * g.gdn_v_dim), err) &&
              D.alloc(D.go, R * g.gdn_v_heads * g.gdn_v_dim, err) && D.alloc(D.qa, R * g.q_rank, err) && D.alloc(D.kva, R * W_, err) &&
              D.alloc(D.mla_ffn_pool, std::max(mla_pool, ffn_pool), err) &&
              D.alloc(D.scores, (size_t) kIdxBatch * o.max_ctx, err) &&
              D.alloc(D.logit, (size_t) RD * g.n_vocab, err) && D.alloc(D.work, mla_work_floats(H, g.kv_rank, work_cells, RD), err) &&
              D.alloc(D.rid, RK, err) && D.alloc(D.cells, R * g.idx_topk, err) && D.alloc(D.nsel, R, err) && D.alloc(D.pos, R, err) &&
              D.alloc(D.ncells, R, err) && D.alloc(D.tok, R, err) && D.alloc(D.gen_out, 1 << 16, err) && D.alloc(D.gen_i, 1, err) &&
              D.alloc(D.grp_ptr, RKD, err) && D.alloc(D.grp_start, RKD + 1, err) && D.alloc(D.n_groups, 1, err) &&
              D.alloc(D.ent_dst, RKD, err) && D.alloc(D.ent_tok, RKD, err) &&
              D.alloc(D.counts, (size_t) (g.n_layers - 1) * g.n_expert, err) && D.alloc(D.table, (size_t) (g.n_layers - 1) * g.n_expert, err);
    if (ok) {
        size_t o1 = 0, o2 = 0;
        auto sub = [&](size_t& off, size_t nv) { float* r = D.mla_ffn_pool + off; off += a64(nv); return r; };
        D.q = sub(o1, n_q); D.qabs = sub(o1, n_qabs); D.qcur = sub(o1, n_qcur);
        D.lat = sub(o1, n_lat); D.att = sub(o1, n_att); D.ag = sub(o1, n_ag);
        D.iq = sub(o1, n_iq); D.ik = sub(o1, n_ik); D.iw = sub(o1, n_iw);
        D.rlog = sub(o2, n_rlog); D.rw = sub(o2, n_rw); D.ex = sub(o2, n_ex);
        D.sgu = sub(o2, n_sgu); D.sh = sub(o2, n_sh); D.fgu = sub(o2, n_fgu); D.fh = sub(o2, n_fh);
    }
    if (ok && D.mtp_on) {
        ok = D.alloc(D.mtp_x, R * E, err) && D.alloc(D.mtp_e, R * E, err) && D.alloc(D.mtp_hin, R * E, err) &&
             D.alloc(D.mtp_cat, R * 2 * E, err) && D.alloc(D.mtp_logit, g.n_vocab, err) &&
             D.alloc(D.tap_x, RD * E, err) && D.alloc(D.d_draft_id, RD + 1, err) && D.alloc(D.d_draft_prob, RD, err);
        if (ok) {
            CK(cudaHostAlloc((void**) &D.pin_draft_id, RD * sizeof(int32_t), cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &D.pin_draft_prob, RD * sizeof(float), cudaHostAllocPortable));
        }
    }
    if (!ok) return false;
    CK(cudaMalloc(&D.xq, K::native_q8_1_bytes(std::max(g.n_ff_dense, g.heads * g.v_dim), RD)));
    CK(cudaMalloc(&D.xscratch, K::native_expert_scratch_bytes(RKD, g.n_ff_exp)));
    CK(cudaHostAlloc((void**) &D.pin_i, (R + 64) * sizeof(int32_t), cudaHostAllocPortable));
    if (mmq::built()) {
        D.mmq_ctx = std::make_unique<mmq::Context>();
        const size_t xq_bytes = std::max({mmq::q8_bytes(R, g.n_ff_dense), mmq::q8_bytes(R, H * g.v_dim),
                                          mmq::q8_bytes((int64_t) H * R, g.kv_rank), mmq::q8_bytes(R, 2 * E)});
        CK(cudaMalloc(&D.mmq_xq, xq_bytes));
        CK(cudaMalloc(&D.mmq_xq_h, mmq::q8_bytes(R, E)));
        if (!D.alloc(D.mmq_ident, RK + 64, err) || !D.alloc(D.mmq_bounds_d, 2 * (R + 2), err) ||
            !D.alloc(D.heads_ids, (size_t) H * R, err) || !D.alloc(D.heads_bounds, H + 1, err))
            return false;
        mmq::iota(D.mmq_ident, RK + 64, D.st);
        std::vector<int32_t> bh(2 * (R + 2), 0);
        for (int b = 0; b <= R; ++b) bh[2 * b + 1] = b;
        CK(cudaMemcpyAsync(D.mmq_bounds_d, bh.data(), bh.size() * 4, cudaMemcpyHostToDevice, D.st));
        D.moe_gub = mmq::gu_slot_bytes(kNVFP4, kNVFP4, g.n_ff_exp, E);
        D.moe_gb = mmq::matrix_bytes(kNVFP4, g.n_ff_exp, E);
        D.moe_db = mmq::matrix_bytes(kNVFP4, E, g.n_ff_exp);
        D.moe_tail_pad = std::max<size_t>(mmq::read_past_bytes(kNVFP4, E), 4096);
        CK(cudaMalloc(&D.moe_xq1, mmq::q8_bytes(RK, E)));
        CK(cudaMalloc(&D.moe_xq2, mmq::q8_bytes(RK, g.n_ff_exp)));
        CK(cudaMalloc(&D.moe_sort_scr, moe_sort_scratch(RK, g.n_expert)));
        if (!D.alloc(D.moe_tok_d, RK, err) || !D.alloc(D.moe_bounds_d, RK + 64, err) ||
            !D.alloc(D.moe_map_d, RK, err) || !D.alloc(D.moe_exp_d, RK, err))
            return false;
    }
    // experts: every layer in pinned mapped host memory, then as many as fit into VRAM slots
    const int NL = g.n_layers - 1, NE = g.n_expert;
    D.host_experts.resize(NL);
    D.table_h.resize((size_t) NL * NE);
    D.slot_of.assign((size_t) NL * NE, -1);
    for (int l = 0; l < NL; ++l) {
        const size_t bytes = (size_t) NE * D.blob;
        CK(cudaHostAlloc((void**) &D.host_experts[l], bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        if (!core::read_range_at(mf.range("experts"), (uint64_t) l * bytes, D.host_experts[l], bytes, err)) return false;
        uint8_t* dp = nullptr;
        CK(cudaHostGetDevicePointer((void**) &dp, D.host_experts[l], 0));
        for (int e = 0; e < NE; ++e) D.table_h[(size_t) l * NE + e] = (unsigned long long) (dp + (size_t) e * D.blob);
    }
    CK(cudaStreamSynchronize(D.st));
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    D.slot_stride = (D.blob + 143) / 144 * 144;
    const size_t ctx_room = free_b > o.vram_reserve ? free_b - o.vram_reserve : 0;
    D.n_slots = (int) std::min<size_t>((size_t) NL * NE, ctx_room / D.slot_stride);
    if (D.n_slots > 0) {
        const size_t slot_pad = std::max<size_t>(D.moe_tail_pad, 4096);
        CK(cudaMalloc(&D.slots, (size_t) D.n_slots * D.slot_stride + slot_pad));
        CK(cudaMemsetAsync((uint8_t*) D.slots + (size_t) D.n_slots * D.slot_stride, 0, slot_pad, D.st));
    }
    // which experts: the profile's most used first, keeping full in-VRAM layers contiguous for zero-copy prefill
    std::vector<uint32_t> prof((size_t) NL * NE, 0);
    if (!o.profile.empty()) {
        std::string perr;
        std::vector<uint8_t> raw;
        if (core::read_range(core::FileRange(o.profile), raw, perr) && raw.size() == prof.size() * 4)
            std::memcpy(prof.data(), raw.data(), raw.size());
        else
            log::warn("gigachat4", "profile %s not used: %s", o.profile.c_str(), perr.empty() ? "wrong size" : perr.c_str());
    }
    std::vector<int> order((size_t) NL * NE);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return prof[a] > prof[b]; });
    std::vector<int> lcnt(NL, 0);
    for (int s = 0; s < D.n_slots; ++s) lcnt[order[s] / NE] += 1;
    if (D.n_slots < NL * NE) {
        std::vector<uint64_t> lscore(NL, 0);
        for (int l = 0; l < NL; ++l)
            for (int e = 0; e < NE; ++e) lscore[l] += prof[(size_t) l * NE + e];
        std::vector<int> lo(NL);
        std::iota(lo.begin(), lo.end(), 0);
        std::stable_sort(lo.begin(), lo.end(), [&](int a, int b) {
            return lcnt[a] != lcnt[b] ? lcnt[a] > lcnt[b] : lscore[a] > lscore[b];
        });
        const int n_full = D.n_slots / NE;
        lcnt.assign(NL, 0);
        for (int i = 0; i < n_full; ++i) lcnt[lo[i]] = NE;
    }
    std::vector<unsigned long long> tab_dev = D.table_h;
    D.layer_contig.assign(NL, nullptr);
    int next_slot = 0;
    for (int l = 0; l < NL; ++l) {
        if (lcnt[l] != NE) continue;
        uint8_t* dst = (uint8_t*) D.slots + (size_t) next_slot * D.slot_stride;
        CK(cudaMemcpy2DAsync(dst, D.slot_stride, D.host_experts[l], D.blob, D.blob, NE, cudaMemcpyHostToDevice, D.st));
        D.layer_contig[l] = dst;
        for (int e = 0; e < NE; ++e) {
            const int id = l * NE + e;
            tab_dev[id] = (unsigned long long) (dst + (size_t) e * D.slot_stride);
            D.slot_of[id] = next_slot + e;
        }
        next_slot += NE;
    }
    for (int s = 0; s < NL * NE && next_slot < D.n_slots; ++s) {
        const int id = order[s], l = id / NE;
        if (lcnt[l] == NE) continue;
        uint8_t* dst = (uint8_t*) D.slots + (size_t) next_slot * D.slot_stride;
        CK(cudaMemcpyAsync(dst, (const void*) (D.host_experts[l] + (size_t) (id % NE) * D.blob), D.blob, cudaMemcpyHostToDevice, D.st));
        tab_dev[id] = (unsigned long long) dst;
        D.slot_of[id] = next_slot++;
    }
    bool has_sparse = false;
    for (int l = 0; l < NL; ++l)
        if (!D.layer_contig[l]) has_sparse = true;
    if (mmq::built() && has_sparse) {
        constexpr int GMAX = mmq::kGatherGroupMax;
        CK(cudaMalloc(&D.moe_grp_gu, (size_t) GMAX * D.moe_gub + D.moe_tail_pad));
        CK(cudaMalloc(&D.moe_grp_d, (size_t) GMAX * D.moe_db + D.moe_tail_pad));
        if (!D.alloc(D.moe_scales, GMAX * 4, err)) return false;
        CK(cudaHostAlloc((void**) &D.pin_moe_i, RK * 4 * sizeof(int32_t), cudaHostAllocPortable));
    }
    D.expert_ptrs.resize((size_t) NL * NE);
    for (int i = 0; i < NL * NE; ++i) D.expert_ptrs[i] = reinterpret_cast<const uint8_t*>(tab_dev[i]);
    CK(cudaMemcpyAsync(D.table, tab_dev.data(), tab_dev.size() * 8, cudaMemcpyHostToDevice, D.st));
    CK(cudaStreamSynchronize(D.st));
    log::info("gigachat4", "experts: %d of %d in VRAM (%.2f GiB), kv=%s, mtp=%s",
              D.n_slots, NL * NE, (double) D.n_slots * D.blob / (1 << 30),
              D.kv_fp8 ? "fp8" : "fp16", D.mtp_on ? "on" : "off");
    return true;
}

void Engine::Impl::attention(Layer& l, int n, int bucket) {
    const int H = g.heads, W_ = g.kv_rank + g.rope, NQ = g.nope + g.rope;
    MlaShape s{H, g.nope, g.rope, g.kv_rank};
    quantize_h(n);
    if (n <= kMaxRows && l.mla_in) {
        K::Q80Segments seg;
        seg.n = 3;
        seg.row0[0] = 0;             seg.n_out[0] = g.q_rank;  seg.x[0] = xq; seg.y[0] = qa;
        seg.row0[1] = g.q_rank;      seg.n_out[1] = W_;        seg.x[1] = xq; seg.y[1] = kva;
        seg.row0[2] = g.q_rank + W_; seg.n_out[2] = g.idx_dim; seg.x[2] = xq; seg.y[2] = ik;
        K::native_q8_0_mmvq_segments(l.mla_in, seg, g.n_embd, n, st);
    } else {
        mmvq_h(l.q_a, g.q_rank, n, qa);
        mmvq_h(l.kv_a, W_, n, kva);
        mmvq_h(l.wk, g.idx_dim, n, ik);
    }
    norm(l.q_norm, qa, g.q_rank, g.q_rank, n, qa, g.q_rank);
    norm(l.kv_norm, kva, W_, g.kv_rank, n, kva, W_);
    mmvq(l.q_b, qa, g.q_rank, H * NQ, n, q);
    heads_product(l.k_b, q, NQ, H * NQ, H, g.kv_rank, g.nope, n, qabs);
    if (kv_fp8) mla_prepare(q, qabs, kva, inv_freq, s, pos, n, 0, qcur, l.cache, st);
    else mla_prepare(q, qabs, kva, inv_freq, s, pos, n, 0, qcur, reinterpret_cast<uint16_t*>(l.cache), st);
    idx_keys(ik, l.k_norm, g.rms_eps, inv_freq, g.idx_dim, g.rope, pos, n, 0, l.keys, st);
    const int32_t* sel = nullptr;
    const int32_t* nc = ncells;
    if (bucket > g.idx_topk) {
        mmvq(l.wq_b, qa, g.q_rank, g.idx_heads * g.idx_dim, n, iq);
        idx_rope_queries(iq, inv_freq, g.idx_heads, g.idx_dim, g.rope, pos, n, st);
        f32_gemv(l.idx_w, h, g.idx_heads, g.n_embd, n, iw, st);
        for (int t0 = 0; t0 < n; t0 += kIdxBatch) {
            const int nt = std::min(kIdxBatch, n - t0);
            idx_topk(iq + (size_t) t0 * g.idx_heads * g.idx_dim, iw + (size_t) t0 * g.idx_heads, l.keys, ncells + t0, bucket,
                     g.idx_heads, g.idx_dim, g.idx_topk, nt, scores, cells + (size_t) t0 * g.idx_topk, nsel + t0, st);
        }
        sel = cells;
        nc = nsel;
    }
    if (kv_fp8) mla_attend(qcur, l.cache, sel, g.idx_topk, nc, std::min(bucket, g.idx_topk), s, mla_scale_v, n, work, lat, st);
    else mla_attend(qcur, reinterpret_cast<const uint16_t*>(l.cache), sel, g.idx_topk, nc, std::min(bucket, g.idx_topk), s, mla_scale_v, n, work, lat, st);
    heads_product(l.v_b, lat, g.kv_rank, H * g.kv_rank, H, g.v_dim, g.kv_rank, n, att);
    mmvq_h(l.gate, H * g.v_dim, n, ag);
    mul_sigmoid(att, ag, (int64_t) n * H * g.v_dim, st);
    mmvq(l.o, att, H * g.v_dim, g.n_embd, n, y);
}

void Engine::Impl::moe_prompt(Layer& l, int li, int n) {
    const int E = g.n_embd, K_ = g.n_expert_used, F = g.n_ff_exp, NE = g.n_expert;
    const int le = li - g.first_dense;
    const int total = n * K_;
    ensure_mmq(kNVFP4, 2 * F);
    ensure_mmq(kNVFP4, E);
    if (layer_contig[le] != nullptr) {
        const uint8_t* w = layer_contig[le];
        moe_sort(rid, n, K_, NE, moe_sort_scr, moe_bounds_d, moe_tok_d, moe_map_d, moe_exp_d, st);
        mmq::quantize(h, moe_tok_d, moe_xq1, kNVFP4, E, E, total, st);
        mmq::Product gu;
        gu.w = w; gu.type = kNVFP4; gu.w_rows = 2 * F; gu.w_cols = E; gu.expert_bytes = slot_stride;
        gu.n = NE; gu.xq = moe_xq1; gu.bounds = moe_bounds_d; gu.ids = mmq_ident;
        gu.total_rows = total; gu.max_rows = n; gu.opt_rows = (total + NE - 1) / NE;
        gu.dst = fgu; gu.ld_dst = 2 * F;
        mmq_ctx->run(gu, st);
        swiglu_split_clamped(fgu, fh, total, F, g.swiglu_limit, w, slot_stride, blob - 16, moe_bounds_d, NE, st, moe_exp_d);
        mmq::quantize(fh, nullptr, moe_xq2, kNVFP4, F, F, total, st);
        mmq::Product dn;
        dn.w = w + xl.down_off; dn.type = kNVFP4; dn.w_rows = E; dn.w_cols = F; dn.expert_bytes = slot_stride;
        dn.n = NE; dn.xq = moe_xq2; dn.bounds = moe_bounds_d; dn.ids = moe_map_d;
        dn.total_rows = total; dn.max_rows = n; dn.opt_rows = (total + NE - 1) / NE;
        dn.dst = ex; dn.ld_dst = E;
        mmq_ctx->run(dn, st);
        quantize_h(n);
        mmvq_h(l.mlp.gu, 2 * l.mlp.n_ff, n, sgu);
        swiglu_clamped(sgu, l.mlp.n_ff, n, g.swiglu_limit, fh, st);
        mmvq(l.mlp.down, fh, l.mlp.n_ff, E, n, sh);
        moe_combine(ex, rw, sh, K_, E, n, y, st);
        return;
    }
    cudaMemcpyAsync(pin_moe_i, rid, (size_t) total * sizeof(int32_t), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    int head[512];
    std::fill_n(head, NE, -1);
    std::vector<int32_t> next(total), map_h(total);
    std::vector<int> cnt(NE, 0), used;
    used.reserve(NE);
    for (int p = total - 1; p >= 0; --p) {
        const int e = pin_moe_i[p];
        next[p] = head[e];
        head[e] = p;
        cnt[e] += 1;
    }
    for (int e = 0; e < NE; ++e)
        if (cnt[e] > 0) used.push_back(e);
    int32_t* tok_stage = pin_moe_i;
    int32_t* bnd_stage = pin_moe_i + total;
    int32_t* map_stage = bnd_stage + (NE + 64);
    int pos_row = 0, bnd_off = 0;
    for (size_t b0 = 0; b0 < used.size();) {
        const int nb = (int) std::min<size_t>(mmq::kGatherGroupMax, used.size() - b0);
        mmq::GatherGroup gg;
        gg.first = 0;
        gg.n = nb;
        int rows = 0, max_rows = 0;
        int32_t* tok_cur = tok_stage + pos_row;
        int32_t* bnd_cur = bnd_stage + bnd_off;
        bnd_cur[0] = 0;
        for (int qi = 0; qi < nb; ++qi) {
            const int e = used[b0 + (size_t) qi];
            gg.blob[qi] = expert_ptrs[(size_t) le * NE + e];
            max_rows = std::max(max_rows, cnt[e]);
            for (int p = head[e]; p >= 0; p = next[p]) {
                map_h[p] = pos_row + rows;
                tok_cur[rows++] = p / K_;
            }
            bnd_cur[qi + 1] = rows;
        }
        int32_t* d_tok = moe_tok_d + pos_row;
        int32_t* d_bnd = moe_bounds_d + bnd_off;
        cudaMemcpyAsync(d_tok, tok_cur, (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_bnd, bnd_cur, (size_t) (nb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, st);
        if (!mmq::gather_native_group(gg, xl.up_off, moe_gb, xl.down_off, moe_db, moe_grp_gu, moe_gub, moe_grp_d, moe_db, st,
                                      blob - 16, moe_scales, moe_gb)) {
            for (int qi = 0; qi < nb; ++qi) {
                const uint8_t* b = gg.blob[qi];
                mmq::gather_native(b, b + xl.up_off, moe_gb, b + xl.down_off, moe_db,
                                   moe_grp_gu + (size_t) qi * moe_gub, moe_grp_d + (size_t) qi * moe_db, st,
                                   b + blob - 16, moe_scales + 4 * qi, moe_gb);
            }
        }
        cudaMemsetAsync(moe_grp_gu + (size_t) nb * moe_gub, 0, moe_tail_pad, st);
        cudaMemsetAsync(moe_grp_d + (size_t) nb * moe_db, 0, moe_tail_pad, st);
        mmq::quantize(h, d_tok, moe_xq1, kNVFP4, E, E, rows, st);
        mmq::Product gu;
        gu.w = moe_grp_gu; gu.type = kNVFP4; gu.w_rows = 2 * F; gu.w_cols = E; gu.expert_bytes = moe_gub;
        gu.n = nb; gu.xq = moe_xq1; gu.bounds = d_bnd; gu.ids = nullptr;
        gu.total_rows = rows; gu.max_rows = max_rows; gu.dst = fgu; gu.ld_dst = 2 * F;
        mmq_ctx->run(gu, st);
        swiglu_split_clamped(fgu, fh, rows, F, g.swiglu_limit, reinterpret_cast<const uint8_t*>(moe_scales), 16, 0, d_bnd, nb, st);
        mmq::quantize(fh, nullptr, moe_xq2, kNVFP4, F, F, rows, st);
        mmq::Product dn;
        dn.w = moe_grp_d; dn.type = kNVFP4; dn.w_rows = E; dn.w_cols = F; dn.expert_bytes = moe_db;
        dn.n = nb; dn.xq = moe_xq2; dn.bounds = d_bnd; dn.ids = nullptr;
        dn.total_rows = rows; dn.max_rows = max_rows; dn.dst = ex + (size_t) pos_row * E; dn.ld_dst = E;
        mmq_ctx->run(dn, st);
        pos_row += rows;
        bnd_off += nb + 1;
        b0 += (size_t) nb;
    }
    std::memcpy(map_stage, map_h.data(), (size_t) total * sizeof(int32_t));
    cudaMemcpyAsync(moe_map_d, map_stage, (size_t) total * sizeof(int32_t), cudaMemcpyHostToDevice, st);
    quantize_h(n);
    mmvq_h(l.mlp.gu, 2 * l.mlp.n_ff, n, sgu);
    swiglu_clamped(sgu, l.mlp.n_ff, n, g.swiglu_limit, fh, st);
    mmvq(l.mlp.down, fh, l.mlp.n_ff, E, n, sh);
    moe_combine_mapped(ex, moe_map_d, rw, sh, K_, E, n, y, st);
}

void Engine::Impl::ffn(Layer& l, int li, int n) {
    const int E = g.n_embd;
    if (!l.moe) {
        mmvq(l.mlp.gu, h, E, 2 * l.mlp.n_ff, n, fgu);
        swiglu_clamped(fgu, l.mlp.n_ff, n, g.swiglu_limit, fh, st);
        mmvq(l.mlp.down, fh, l.mlp.n_ff, E, n, y);
        return;
    }
    K::RouterArgs ra;
    ra.n_expert = g.n_expert, ra.n_embd = E, ra.k = g.n_expert_used, ra.weight_scale = g.expert_weight_scale, ra.gate = K::RouterGate::Sigmoid;
    K::router(h, l.router, l.router_bias, n, ra, rlog, rid, rw, st);
    const int le = li - g.first_dense;
    if (n > kMaxRows && (layer_contig[le] != nullptr || moe_grp_gu) && mmq::built() &&
        mmq::fits(kNVFP4, 2 * g.n_ff_exp) && mmq::fits(kNVFP4, E)) {
        moe_prompt(l, li, n);
        return;
    }
    moe_plan(rid, n, g.n_expert_used, g.n_expert, table + (size_t) le * g.n_expert, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
             counts + (size_t) le * g.n_expert, st);
    K::native_quantize_q8_1(h, xq, E, n, st);
    const int RK = n * g.n_expert_used;
    K::native_expert_grouped_clamped(xl, g.swiglu_limit, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, RK, RK, xq, xscratch, ex, st);
    K::native_mmvq(kQ80, l.mlp.gu, xq, sgu, E, 2 * l.mlp.n_ff, n, st);
    swiglu_clamped(sgu, l.mlp.n_ff, n, g.swiglu_limit, fh, st);
    mmvq(l.mlp.down, fh, l.mlp.n_ff, E, n, sh);
    moe_combine(ex, rw, sh, g.n_expert_used, E, n, y, st);
}

void Engine::Impl::step(int n, int bucket, bool save_gdn, bool need_logits) {
    const int E = g.n_embd, NK = g.gdn_k_heads * g.gdn_k_dim, NV = g.gdn_v_heads * g.gdn_v_dim, NP = 2 * NK + 2 * NV;
    embed_rows(embd, tok, E, n, x, st);
    for (int li = 0; li < g.n_layers; ++li) {
        Layer& l = L[li];
        norm(l.attn_norm, x, E, E, n, h, E);
        if (l.full) {
            attention(l, n, bucket);
        } else {
            quantize_h(n);
            if (n <= kMaxRows && l.qkvzba) {
                K::Q80Segments seg;
                seg.n = 2;
                seg.row0[0] = 0;  seg.n_out[0] = NP;                seg.x[0] = xq; seg.y[0] = proj;
                seg.row0[1] = NP; seg.n_out[1] = 2 * g.gdn_v_heads; seg.x[1] = xq; seg.y[1] = bav;
                K::native_q8_0_mmvq_segments(l.qkvzba, seg, E, n, st);
            } else {
                mmvq_h(l.qkvz, NP, n, proj);
                mmvq_h(l.ba, 2 * g.gdn_v_heads, n, bav);
            }
            if (save_gdn) {
                const int gi = gdn_layers[li];
                cudaMemcpyAsync(gdn_proj_snap[gi], proj, (size_t) n * NP * sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(gdn_bav_snap[gi], bav, (size_t) n * 2 * g.gdn_v_heads * sizeof(float), cudaMemcpyDeviceToDevice, st);
            }
            gdn_conv(proj, NP, l.conv_state, l.conv, 2 * NK + NV, n, conv_out, st);
            gdn_step(l.state, conv_out, bav, l.a_log, l.dt_bias, proj + 2 * NK + NV, NP, l.gnorm, g.rms_eps,
                     GdnShape{g.gdn_k_heads, g.gdn_v_heads, g.gdn_k_dim, g.gdn_v_dim}, n, go, st);
            mmvq(l.out, go, NV, E, n, y);
        }
        norm(l.attn_post, y, E, E, n, x, E, true);
        norm(l.ffn_norm, x, E, E, n, h, E);
        ffn(l, li, n);
        norm(l.ffn_post, y, E, E, n, x, E, true);
    }
    if (!need_logits) return;
    if (n <= kMaxRows) {
        norm(out_norm, x, E, E, n, h, E);
        K::native_quantize_q8_1(h, xq, E, n, st);
        if (n == 1) K::native_q8_0_mmvq_stream(head, xq, logit, E, g.n_vocab, st);
        else K::native_mmvq(kQ80, head, xq, logit, E, g.n_vocab, n, st);
        softcap(logit, (int64_t) n * g.n_vocab, g.logit_softcap, st);
    } else {
        norm(out_norm, x + (size_t) (n - 1) * E, E, E, 1, h, E);
        K::native_quantize_q8_1(h, xq, E, 1, st);
        K::native_q8_0_mmvq_stream(head, xq, logit, E, g.n_vocab, st);
        softcap(logit, g.n_vocab, g.logit_softcap, st);
    }
}

void Engine::Impl::mtp_step_batch(const int32_t* h_tokens, const float* h_prev, const float* h_rest, int n, int pos0) {
    const int E = g.n_embd;
    std::memcpy(pin_i, h_tokens, (size_t) n * sizeof(int32_t));
    cudaMemcpyAsync(tok, pin_i, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, st);
    set_rows(pos, ncells, n, pos0, st);
    embed_rows(mtp_embd, tok, E, n, mtp_e, st);
    norm(mtp_enorm, mtp_e, E, E, n, mtp_cat, 2 * E);
    cudaMemcpyAsync(mtp_hin, h_prev, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, st);
    if (n > 1)
        cudaMemcpyAsync(mtp_hin + E, h_rest, (size_t) (n - 1) * E * sizeof(float), cudaMemcpyDeviceToDevice, st);
    norm(mtp_hnorm, mtp_hin, E, E, n, mtp_cat + E, 2 * E);
    mmvq(mtp_eh, mtp_cat, 2 * E, E, n, mtp_x);
    const int bucket = std::min(bucket_of(pos0 + n), opt.max_ctx);
    norm(mtp_layer.attn_norm, mtp_x, E, E, n, h, E);
    attention(mtp_layer, n, bucket);
    norm(mtp_layer.attn_post, y, E, E, n, mtp_x, E, true);
    norm(mtp_layer.ffn_norm, mtp_x, E, E, n, h, E);
    ffn(mtp_layer, -1, n);
    norm(mtp_layer.ffn_post, y, E, E, n, mtp_x, E, true);
}

void Engine::Impl::mtp_step_one(const int32_t* tok_d, const float* h_in, int bucket, int out_idx) {
    const int E = g.n_embd;
    embed_rows(mtp_embd, tok_d, E, 1, mtp_e, st);
    norm(mtp_enorm, mtp_e, E, E, 1, mtp_cat, 2 * E);
    norm(mtp_hnorm, h_in, E, E, 1, mtp_cat + E, 2 * E);
    mmvq(mtp_eh, mtp_cat, 2 * E, E, 1, mtp_x);
    norm(mtp_layer.attn_norm, mtp_x, E, E, 1, h, E);
    attention(mtp_layer, 1, bucket);
    norm(mtp_layer.attn_post, y, E, E, 1, mtp_x, E, true);
    norm(mtp_layer.ffn_norm, mtp_x, E, E, 1, h, E);
    ffn(mtp_layer, -1, 1);
    norm(mtp_layer.ffn_post, y, E, E, 1, mtp_x, E, true);
    norm(mtp_head_norm, mtp_x, E, E, 1, h, E);
    K::native_quantize_q8_1(h, xq, E, 1, st);
    K::native_q8_0_mmvq_stream(mtp_head, xq, mtp_logit, E, g.n_vocab, st);
    softcap(mtp_logit, (int64_t) g.n_vocab, g.logit_softcap, st);
    argmax_prob_row(mtp_logit, g.n_vocab, d_draft_id + 1 + out_idx, d_draft_prob + out_idx, st);
}

void Engine::forward(const int32_t* tokens, int n, int pos, bool need_logits) {
    Impl& D = *d_;
    std::memcpy(D.pin_i, tokens, (size_t) n * sizeof(int32_t));
    cudaMemcpyAsync(D.tok, D.pin_i, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, D.st);
    set_rows(D.pos, D.ncells, n, pos, D.st);
    D.step(n, std::min(bucket_of(pos + n), D.opt.max_ctx), false, need_logits);
    if (D.mtp_on) {
        D.mtp_step_batch(tokens, D.prev_x, D.x, n, pos);
        D.mtp_synced_pos = pos + n;
    }
    cudaMemcpyAsync(D.prev_x, D.x + (size_t) (n - 1) * D.g.n_embd, (size_t) D.g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, D.st);
    D.state_pos = pos + n;
    D.win_has_snap = false;
    D.tap_n = 0;
    cudaStreamSynchronize(D.st);
}

bool Engine::decode_forward(const int32_t* tokens, int n, int pos, std::string& err) {
    Impl& D = *d_;
    if (n <= 0 || n > kMaxRows) {
        err = "gigachat4 decode_forward expects 1..8 tokens";
        return false;
    }
    if (pos + n > D.opt.max_ctx) {
        err = "gigachat4: position past max_ctx";
        return false;
    }
    if (D.win_has_snap && pos < D.state_pos) {
        const int keep = (int) (pos - D.win_pos);
        if (keep < 0 || keep > D.win_len) {
            err = "gigachat4: decode position out of window range";
            return false;
        }
        commit_kept(keep);
    }
    const bool save_gdn = (n > 1 && D.gdn_snap != nullptr);
    const int bucket = std::min(bucket_of(pos + n), D.opt.max_ctx);
    std::memcpy(D.pin_i, tokens, (size_t) n * sizeof(int32_t));
    cudaMemcpyAsync(D.tok, D.pin_i, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, D.st);
    set_rows(D.pos, D.ncells, n, pos, D.st);
    Graph& gr = D.dec_graphs[{n, bucket, save_gdn ? 1 : 0, D.mtp_on ? 1 : 0}];
    if (!gr.exec) {
        cudaGraph_t graph = nullptr;
        const cudaError_t cb = cudaStreamBeginCapture(D.st, cudaStreamCaptureModeRelaxed);
        if (cb != cudaSuccess) {
            err = std::string("gigachat4 graph capture: ") + cudaGetErrorString(cb);
            return false;
        }
        if (save_gdn)
            cudaMemcpyAsync(D.gdn_snap, D.gdn_mem, D.gdn_total_floats * sizeof(float), cudaMemcpyDeviceToDevice, D.st);
        D.step(n, bucket, save_gdn, true);
        if (D.mtp_on)
            cudaMemcpyAsync(D.tap_x, D.x, (size_t) n * D.g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, D.st);
        else
            cudaMemcpyAsync(D.prev_x, D.x + (size_t) (n - 1) * D.g.n_embd, (size_t) D.g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, D.st);
        const cudaError_t ce = cudaStreamEndCapture(D.st, &graph);
        const cudaError_t ci = ce == cudaSuccess ? cudaGraphInstantiateWithFlags(&gr.exec, graph, 0) : ce;
        if (graph) cudaGraphDestroy(graph);
        if (ci != cudaSuccess) {
            gr.exec = nullptr;
            err = std::string("gigachat4 graph instantiate: ") + cudaGetErrorString(ci);
            return false;
        }
    }
    if (cudaGraphLaunch(gr.exec, D.st) != cudaSuccess) {
        err = std::string("gigachat4 graph launch: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    D.win_pos = pos;
    D.win_len = n;
    D.win_has_snap = save_gdn;
    D.state_pos = pos + n;
    if (D.mtp_on) {
        std::memcpy(D.tap_tok, tokens, (size_t) n * sizeof(int32_t));
        D.tap_pos0 = pos;
        D.tap_n = n;
    }
    return cudaGetLastError() == cudaSuccess || ((err = cudaGetErrorString(cudaGetLastError())), false);
}

void Engine::commit_kept(int keep) {
    Impl& D = *d_;
    const Geometry& g = D.g;
    if (D.win_has_snap && keep < D.win_len) {
        const int NK = g.gdn_k_heads * g.gdn_k_dim, NV = g.gdn_v_heads * g.gdn_v_dim, NP = 2 * NK + 2 * NV;
        cudaMemcpyAsync(D.gdn_mem, D.gdn_snap, D.gdn_total_floats * sizeof(float), cudaMemcpyDeviceToDevice, D.st);
        if (keep > 0) {
            for (int li = 0; li < g.n_layers; ++li) {
                Layer& l = D.L[li];
                if (l.full) continue;
                const int gi = D.gdn_layers[li];
                gdn_conv(D.gdn_proj_snap[gi], NP, l.conv_state, l.conv, 2 * NK + NV, keep, D.conv_out, D.st);
                gdn_step(l.state, D.conv_out, D.gdn_bav_snap[gi], l.a_log, l.dt_bias,
                         D.gdn_proj_snap[gi] + 2 * NK + NV, NP, l.gnorm, g.rms_eps,
                         GdnShape{g.gdn_k_heads, g.gdn_v_heads, g.gdn_k_dim, g.gdn_v_dim}, keep, D.go, D.st);
            }
        }
        D.state_pos = D.win_pos + keep;
    }
    D.win_has_snap = false;
    if (D.mtp_on && D.tap_n > 0 && keep >= 1 && keep <= D.tap_n) {
        cudaMemcpyAsync(D.prev_x, D.tap_x + (size_t) (keep - 1) * g.n_embd, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, D.st);
    }
}

bool Engine::draft_begin(int32_t anchor, int pos, int n, std::string& err) {
    Impl& D = *d_;
    D.pending_drafts = 0;
    if (!D.mtp_on) {
        err = "gigachat4 has no MTP drafter active";
        return false;
    }
    const int room = D.opt.max_ctx - pos;
    const int count = std::min({n, kMaxRows - 1, room});
    if (count <= 0) return true;
    const int E = D.g.n_embd;
    // Sync MTP cache up to `pos` from the last target window's tapped hidden states.
    if (D.tap_n > 0 && pos >= D.tap_pos0 && pos <= D.tap_pos0 + D.tap_n) {
        const int keep = (int) (pos - D.tap_pos0);
        if (D.mtp_synced_pos < pos) {
            const int64_t from = std::max(D.mtp_synced_pos, D.tap_pos0);
            const int sync_n = (int) (pos - from);
            if (sync_n > 0 && sync_n <= kMaxRows) {
                const int off = (int) (from - D.tap_pos0);
                const float* h0 = off == 0 ? D.prev_x : (D.tap_x + (size_t) (off - 1) * E);
                const float* hr = D.tap_x + (size_t) off * E;
                D.mtp_step_batch(D.tap_tok + off, h0, hr, sync_n, (int) from);
            }
            D.mtp_synced_pos = pos;
        }
        if (keep > 0)
            cudaMemcpyAsync(D.prev_x, D.tap_x + (size_t) (keep - 1) * E, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, D.st);
    }
    // Step 0 uses the target's true hidden state (prev_x); subsequent steps feed mtp_x recurrently on device.
    D.pin_i[0] = anchor;
    cudaMemcpyAsync(D.d_draft_id, D.pin_i, sizeof(int32_t), cudaMemcpyHostToDevice, D.st);
    set_rows(D.pos, D.ncells, 1, pos, D.st);
    cudaMemcpyAsync(D.mtp_x, D.prev_x, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, D.st);
    const int b0 = std::min(bucket_of(pos + 1), D.opt.max_ctx);
    const int b1 = std::min(bucket_of(pos + count), D.opt.max_ctx);
    if (b0 == b1) {
        Graph& gr = D.mtp_graphs[{count, b0}];
        if (!gr.exec) {
            cudaGraph_t graph = nullptr;
            const cudaError_t cb = cudaStreamBeginCapture(D.st, cudaStreamCaptureModeRelaxed);
            if (cb != cudaSuccess) {
                err = std::string("gigachat4 mtp graph capture: ") + cudaGetErrorString(cb);
                return false;
            }
            for (int j = 0; j < count; ++j) {
                D.mtp_step_one(D.d_draft_id + j, D.mtp_x, b0, j);
                if (j + 1 < count) advance_step(D.pos, D.ncells, D.st);
            }
            const cudaError_t ce = cudaStreamEndCapture(D.st, &graph);
            const cudaError_t ci = ce == cudaSuccess ? cudaGraphInstantiateWithFlags(&gr.exec, graph, 0) : ce;
            if (graph) cudaGraphDestroy(graph);
            if (ci != cudaSuccess) {
                gr.exec = nullptr;
                err = std::string("gigachat4 mtp graph instantiate: ") + cudaGetErrorString(ci);
                return false;
            }
        }
        if (cudaGraphLaunch(gr.exec, D.st) != cudaSuccess) {
            err = std::string("gigachat4 mtp graph launch: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    } else {
        for (int j = 0; j < count; ++j) {
            set_rows(D.pos, D.ncells, 1, pos + j, D.st);
            const int b = std::min(bucket_of(pos + j + 1), D.opt.max_ctx);
            D.mtp_step_one(D.d_draft_id + j, D.mtp_x, b, j);
        }
    }
    if (D.mtp_synced_pos == pos) D.mtp_synced_pos = pos + 1;
    cudaMemcpyAsync(D.pin_draft_id, D.d_draft_id + 1, (size_t) count * sizeof(int32_t), cudaMemcpyDeviceToHost, D.st);
    cudaMemcpyAsync(D.pin_draft_prob, D.d_draft_prob, (size_t) count * sizeof(float), cudaMemcpyDeviceToHost, D.st);
    D.pending_drafts = count;
    return cudaGetLastError() == cudaSuccess || ((err = cudaGetErrorString(cudaGetLastError())), false);
}

bool Engine::draft_end(int32_t* ids, float* conf, int max_n, int& n,
                       const std::function<bool(const float*, int, int)>& need, std::string& err) {
    Impl& D = *d_;
    const int avail = std::min(D.pending_drafts, max_n);
    D.pending_drafts = 0;
    n = 0;
    if (avail <= 0) return true;
    if (cudaStreamSynchronize(D.st) != cudaSuccess) {
        err = "gigachat4: draft stream sync failed";
        return false;
    }
    for (int j = 0; j < avail; ++j) {
        ids[j] = D.pin_draft_id[j];
        conf[j] = D.pin_draft_prob[j];
        if (need && !need(conf, j, avail)) break;
        ++n;
    }
    return true;
}

void Engine::greedy(int32_t first, int pos, int n, std::vector<int32_t>& out) {
    Impl& D = *d_;
    D.pin_i[0] = first;
    cudaMemcpyAsync(D.tok, D.pin_i, 4, cudaMemcpyHostToDevice, D.st);
    set_rows(D.pos, D.ncells, 1, pos, D.st);
    cudaMemsetAsync(D.gen_i, 0, 4, D.st);
    for (int i = 0; i < n; ++i) {
        const int b = std::min(bucket_of(pos + i + 1), D.opt.max_ctx);
        Graph& gr = D.graphs[b];
        if (!gr.exec) {
            cudaGraph_t graph = nullptr;
            cudaStreamBeginCapture(D.st, cudaStreamCaptureModeThreadLocal);
            D.step(1, b);
            argmax_rows(D.logit, D.g.n_vocab, 1, D.tok, D.st);
            greedy_advance(D.tok, D.pos, D.ncells, D.gen_out, D.gen_i, kGenCap, D.st);
            cudaStreamEndCapture(D.st, &graph);
            cudaGraphInstantiate(&gr.exec, graph, 0);
            cudaGraphDestroy(graph);
        }
        cudaGraphLaunch(gr.exec, D.st);
    }
    cudaStreamSynchronize(D.st);
    out.resize(std::min(n, kGenCap));
    cudaMemcpy(out.data(), D.gen_out, out.size() * 4, cudaMemcpyDeviceToHost);
}

void Engine::reset() {
    Impl& D = *d_;
    cudaMemsetAsync(D.gdn_mem, 0, D.gdn_total_floats * sizeof(float), D.st);
    D.state_pos = 0;
    D.win_pos = 0;
    D.win_len = 0;
    D.win_has_snap = false;
    D.tap_pos0 = 0;
    D.tap_n = 0;
    D.mtp_synced_pos = 0;
    D.pending_drafts = 0;
    cudaStreamSynchronize(D.st);
}

void Engine::expert_stats(uint64_t& host_reads, uint64_t& reads) const {
    const Impl& D = *d_;
    std::vector<uint32_t> c(D.slot_of.size());
    cudaMemcpy(c.data(), D.counts, c.size() * 4, cudaMemcpyDeviceToHost);
    host_reads = reads = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        reads += c[i];
        if (D.slot_of[i] < 0) host_reads += c[i];
    }
}

bool Engine::save_profile(const std::string& path, std::string& err) const {
    const Impl& D = *d_;
    std::vector<uint32_t> c(D.slot_of.size());
    cudaMemcpy(c.data(), D.counts, c.size() * 4, cudaMemcpyDeviceToHost);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(c.data(), 4, c.size(), f) != c.size()) {
        if (f) std::fclose(f);
        err = "cannot write " + path;
        return false;
    }
    std::fclose(f);
    return true;
}

size_t Engine::running_bytes() const { return d_->gdn_total_floats * sizeof(float); }

bool Engine::running_state(int64_t pos, bool save, core::RunningState& st) {
    Impl& D = *d_;
    if (save && D.win_has_snap) commit_kept((int) (pos - D.win_pos));
    st.regions.push_back({D.gdn_mem, D.gdn_total_floats * sizeof(float)});
    st.stream = D.st;
    return true;
}

void Engine::on_running_restored(int64_t pos) {
    Impl& D = *d_;
    D.state_pos = pos;
    D.win_has_snap = false;
    D.tap_n = 0;
    D.mtp_synced_pos = pos;
    D.pending_drafts = 0;
}

size_t Engine::cells_bytes(int64_t positions) const {
    const Impl& D = *d_;
    const int full_layers = D.g.n_layers / D.g.full_attn_every + (D.mtp_on ? 1 : 0);
    return (size_t) full_layers * (size_t) positions * (D.cache_row_bytes + (size_t) D.g.idx_dim * 2);
}

bool Engine::save_cells(std::vector<core::ConversationKv>& kv, int64_t positions, std::string& err) const {
    const Impl& D = *d_;
    if (positions > D.opt.max_ctx) { err = "gigachat4: positions past max_ctx"; return false; }
    cudaStreamSynchronize(D.st);
    kv.clear();
    const int fmt = D.kv_fp8 ? 1 : 3;
    const size_t kb = (size_t) positions * D.cache_row_bytes;
    const size_t pb = (size_t) positions * D.g.idx_dim * 2;
    auto push_layer = [&](const Layer& l) -> bool {
        core::ConversationKv c;
        c.format = fmt;
        c.page_size = 1;
        c.cells = positions;
        c.pooled_rows = positions;
        c.idx_dim = D.g.idx_dim;
        c.k.resize_for_overwrite(kb);
        c.pooled.resize_for_overwrite(pb);
        bool ok = c.k.visit(0, kb, [&](uint8_t* p, size_t n, size_t at) {
            return cudaMemcpy(p, l.cache + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
        });
        ok = ok && c.pooled.visit(0, pb, [&](uint8_t* p, size_t n, size_t at) {
            return cudaMemcpy(p, reinterpret_cast<const uint8_t*>(l.keys) + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
        });
        if (!ok) { err = "gigachat4: reading KV cache failed"; return false; }
        kv.push_back(std::move(c));
        return true;
    };
    for (const Layer& l : D.L)
        if (l.full && !push_layer(l)) return false;
    if (D.mtp_on && !push_layer(D.mtp_layer)) return false;
    return true;
}

bool Engine::restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions, std::string& err) {
    Impl& D = *d_;
    if (positions > D.opt.max_ctx) { err = "gigachat4: parked conversation exceeds max_ctx"; return false; }
    cudaStreamSynchronize(D.st);
    const int fmt = D.kv_fp8 ? 1 : 3;
    const size_t want = (size_t) (D.g.n_layers / D.g.full_attn_every + (D.mtp_on ? 1 : 0));
    if (kv.size() != want) { err = "gigachat4: parked conversation has incompatible KV layout"; return false; }
    const size_t kb = (size_t) positions * D.cache_row_bytes;
    const size_t pb = (size_t) positions * D.g.idx_dim * 2;
    size_t i = 0;
    auto pull_layer = [&](Layer& l) -> bool {
        const core::ConversationKv& c = kv[i++];
        if (c.format != fmt || c.page_size != 1 || c.cells != positions || c.pooled_rows != positions ||
            c.k.size() != kb || c.pooled.size() != pb) {
            err = "gigachat4: parked conversation has incompatible KV format";
            return false;
        }
        bool ok = c.k.visit(0, kb, [&](const uint8_t* p, size_t n, size_t at) {
            return cudaMemcpy(l.cache + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
        });
        ok = ok && c.pooled.visit(0, pb, [&](const uint8_t* p, size_t n, size_t at) {
            return cudaMemcpy(reinterpret_cast<uint8_t*>(l.keys) + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
        });
        if (!ok) { err = "gigachat4: restoring KV cache failed"; return false; }
        return true;
    };
    for (Layer& l : D.L)
        if (l.full && !pull_layer(l)) return false;
    if (D.mtp_on && !pull_layer(D.mtp_layer)) return false;
    D.mtp_synced_pos = positions;
    return true;
}

}  // namespace basalt::models::gigachat4
