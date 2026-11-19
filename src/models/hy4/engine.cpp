// src/models/hy4/engine.cpp - see engine.hpp.
#include "models/hy4/engine.hpp"

#include "basalt/core/model_file.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/log.hpp"
#include "basalt/models/hy4/kernels.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <future>
#include <map>
#include <sstream>

namespace basalt::models::hy4 {
namespace {

namespace k = basalt::kernels;

struct Mat {
    void* w = nullptr;
    int type = 0, n_in = 0, n_out = 0;
};

constexpr float kEngramEps = 0.0078125f;   // bfloat16's eps: the checkpoint's dtype

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool cuda_ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

template <typename T>
T* dalloc(size_t n, std::vector<void*>& owned) {
    void* p = nullptr;
    if (cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T));
    owned.push_back(p);
    return (T*) p;
}

}  // namespace

struct Engine::Impl {
    core::ModelFile file;
    core::FileRange trunk;
    const core::Json* index = nullptr;
    EngineOptions o;
    cudaStream_t s = nullptr;
    std::vector<void*> owned;
    uint64_t trunk_bytes_gpu = 0;

    struct Layer {
        bool sliding = false;
        int heads = 0;
        float theta = 0.f;
        Mat q_a, q_b, kv_a, gate, o;
        float *in_norm = nullptr, *post_attn_norm = nullptr, *q_a_norm = nullptr, *kv_a_norm = nullptr,
              *post_proj_norm = nullptr, *sink = nullptr;
        uint16_t* kv_b = nullptr;
        HcMix attn_hc{}, mlp_hc{};
        bool indexer = false;
        Mat idx_q, idx_tk, idx_hw;
        float *idx_tk_norm = nullptr;
        IndexerPool pool{};
        bool dense = false;
        Mat d_gate, d_up, d_down;
        float* post_fc2_norm = nullptr;
        float *router = nullptr, *router_bias = nullptr, *combine_norm = nullptr;
        Mat lat_in, lat_out, sh_gate, sh_up, sh_down;
        int engram = -1;
        int dfly = -1;                  // the draft layer that reads this layer's cache and picks
        uint16_t *ckv = nullptr, *kpe = nullptr;
        int cap = 0;
        float *tok_keys = nullptr, *grp_keys = nullptr;
    };
    Layer L[kLayers];

    struct Engram {
        int fd = -1;
        uint64_t off = 0, rows = 0;
        std::vector<int64_t> sizes, offsets;
        int64_t mult[kEngramMaxNgram] = {};
        int32_t pad = 0;
        std::vector<int32_t> lookup;
        Mat key, value;   // key: the kHc key projections as one matrix, so its rows come out [T][kHc][kEmbd]
        EngramWeights w{};
        float* head_norm = nullptr;
    };
    std::vector<Engram> eg;

    uint16_t* embed = nullptr;
    Mat head;
    float* final_norm = nullptr;
    HcMix head_hc{};

    uint8_t* experts = nullptr;
    uint64_t experts_mapped = 0, blob_bytes = 0;
    std::vector<uint64_t> layer_off;   // per model layer (0 for the dense one)
    k::NativeExpertLayout xl{};

    // buffers (kMaxT rows)
    float *streams, *y, *xn, *qr, *q, *kva, *gate_b, *q_lat, *o_lat, *o_heads, *a, *logits, *router_logits, *route_w,
        *lat, *exp_rows, *lat_sum, *sh_g, *sh_u, *ffn_a, *ffn_b, *idx_q, *idx_hw, *idx_tk, *idx_scr, *eg_rows, *eg_e,
        *eg_key, *eg_val, *hc_pre, *hc_post, *dense_g, *dense_u;
    void *xq = nullptr, *xq2 = nullptr, *exp_scratch = nullptr;
    int32_t *route_ids, *sel, *sel_n, *small;
    // the grouped-experts tables of a layer, built on the host from the router's picks and copied in one piece
    static constexpr int kMaxEnt = kMaxT * kExpertsUsed;
    struct Groups {
        unsigned long long ptr[kMaxEnt];
        int32_t start[kMaxEnt + 1];
        int32_t n;
        int32_t dst[kMaxEnt];
        int32_t tok[kMaxEnt];
    };
    Groups* d_grp = nullptr;
    Groups* h_grp = nullptr;      // pinned
    int32_t* h_small = nullptr;   // pinned, kMaxT each: token, pos, lo_window, hi, lo_full (0)
    float* h_eg_rows = nullptr;   // pinned: [engram][kMaxT][kEngramHeads][kEngramHeadDim]
    std::vector<uint16_t> eg_raw;
    std::vector<int32_t> history;
    // ---- the VRAM expert cache: hits read their slot, a miss is copied in by DMA over the least-routed slot ----
    uint8_t* slots = nullptr;
    int n_slots = 0;
    std::vector<int32_t> slot_of;   // [kLayers * kExperts]: the expert's slot, -1 when not cached
    std::vector<int32_t> owner;     // [n_slots]: the slot's expert, -1 free
    std::vector<float> score;       // [kLayers * kExperts]: routings, halved every kHalfLife tokens
    std::vector<float> counts;      // [kLayers * kExperts]: all routings (the saved profile)
    int32_t* h_ids = nullptr;
    uint64_t hits = 0, misses = 0;
    static constexpr int kHalfLife = 256;

    bool cache_init(std::string& err) {
        const size_t n = (size_t) kLayers * kExperts;
        slot_of.assign(n, -1);
        score.assign(n, 0.f);
        counts.assign(n, 0.f);
        if (!o.expert_profile.empty())
            if (std::FILE* f = platform::open_file_utf8(o.expert_profile, "rb")) {
                std::vector<float> c(n);
                if (std::fread(c.data(), 4, n, f) == n) counts = c;
                std::fclose(f);
            }
        g_a.assign((size_t) kMaxT * kLayers, nullptr);
        g_b.assign((size_t) kMaxT * kLayers, nullptr);
        g_head.assign(kMaxT, nullptr);
        warm.assign(kMaxT + 1, false);
        if (cudaMallocHost((void**) &h_logits, (size_t) kMaxT * kVocab * 4) != cudaSuccess ||
            cudaMallocHost((void**) &h_ids, kMaxEnt * 4) != cudaSuccess ||
            cudaMallocHost((void**) &h_grp, sizeof(Groups)) != cudaSuccess) {
            err = "pinned host buffers";
            return false;
        }
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t reserve = (size_t) o.vram_reserve_mib << 20;
        n_slots = free_b > reserve ? (int) std::min<size_t>((free_b - reserve) / blob_bytes, (size_t) (kLayers - 1) * kExperts) : 0;
        while (n_slots > 0 && cudaMalloc((void**) &slots, (size_t) n_slots * blob_bytes) != cudaSuccess) {
            cudaGetLastError();
            n_slots -= n_slots / 16 + 1;
        }
        if (n_slots <= 0) {
            n_slots = 0;
            basalt::log::warn("basalt hy4", "no VRAM for an expert cache: every expert is read over PCIe");
            return true;
        }
        owner.assign((size_t) n_slots, -1);
        // the first fill: the most routed experts of the profile, else the same share of every layer
        std::vector<int32_t> order;
        for (int l = 1; l < kLayers; ++l)
            for (int e = 0; e < kExperts; ++e) order.push_back(l * kExperts + e);
        const bool have = std::any_of(counts.begin(), counts.end(), [](float c) { return c > 0.f; });
        if (have)
            std::stable_sort(order.begin(), order.end(), [&](int32_t a2, int32_t b2) { return counts[a2] > counts[b2]; });
        else
            std::stable_sort(order.begin(), order.end(), [](int32_t a2, int32_t b2) { return a2 % kExperts < b2 % kExperts; });
        for (int i = 0; i < n_slots; ++i) {
            const int32_t idx = order[(size_t) i];
            const int l = idx / kExperts, e = idx % kExperts;
            cudaMemcpyAsync(slots + (size_t) i * blob_bytes, experts + layer_off[(size_t) l] + (size_t) e * blob_bytes,
                            blob_bytes, cudaMemcpyHostToDevice, s);
            slot_of[(size_t) idx] = i;
            owner[(size_t) i] = idx;
            score[(size_t) idx] = have ? counts[(size_t) idx] : 0.f;
        }
        if (!cuda_ok(cudaStreamSynchronize(s), "the expert cache first fill", err)) return false;
        basalt::log::info("basalt hy4", "expert cache: %d slots in VRAM (%.2f GiB, %.0f%% of the experts), filled from %s",
                          n_slots, (double) n_slots * blob_bytes / 1073741824.0, 100.0 * n_slots / ((kLayers - 1) * kExperts),
                          have ? o.expert_profile.c_str() : "no profile (the same share of every layer)");
        return true;
    }
    // the device address of layer l's expert e, copying it into a slot on a miss; `picked` ([n_picked]) are this
    // window's experts, whose slots are not given up
    unsigned long long expert_place(int l, int e, const int32_t* picked, int n_picked) {
        const size_t idx = (size_t) l * kExperts + (size_t) e;
        counts[idx] += 1.f;
        score[idx] += 1.f;
        const uint8_t* host = experts + layer_off[(size_t) l] + (size_t) e * blob_bytes;
        if (n_slots == 0) return (unsigned long long) host;
        if (slot_of[idx] >= 0) {
            ++hits;
            return (unsigned long long) (slots + (size_t) slot_of[idx] * blob_bytes);
        }
        ++misses;
        int v = -1;
        float best = 3.4e38f;
        for (int i = 0; i < n_slots; ++i) {
            const int32_t ow = owner[(size_t) i];
            if (ow < 0) { v = i; break; }
            if (score[(size_t) ow] >= best) continue;
            bool in_use = ow / kExperts == l;
            if (in_use) {
                in_use = false;
                for (int k2 = 0; k2 < n_picked; ++k2) in_use = in_use || picked[k2] == ow % kExperts;
            }
            if (!in_use) { best = score[(size_t) ow]; v = i; }
        }
        if (v < 0) return (unsigned long long) host;   // every slot holds one of this window's experts
        if (owner[(size_t) v] >= 0) slot_of[(size_t) owner[(size_t) v]] = -1;
        owner[(size_t) v] = (int32_t) idx;
        slot_of[idx] = v;
        uint8_t* dst = slots + (size_t) v * blob_bytes;
        cudaMemcpyAsync(dst, host, blob_bytes, cudaMemcpyHostToDevice, s);
        return (unsigned long long) dst;
    }

    // the window's experts as groups: entries of the same expert share one group, so a window reads each once
    void expert_groups_host(int l, int T) {
        const int ne = T * kExpertsUsed;
        int ord[kMaxEnt];
        for (int i = 0; i < ne; ++i) ord[i] = i;
        std::stable_sort(ord, ord + ne, [&](int a2, int b2) { return h_ids[a2] < h_ids[b2]; });
        Groups& g = *h_grp;
        int n = 0;
        for (int i = 0; i < ne; ++i) {
            const int en = ord[i];
            const unsigned long long at = expert_place(l, h_ids[en], h_ids, ne);
            if (i == 0 || h_ids[en] != h_ids[ord[i - 1]]) {
                g.ptr[n] = at;
                g.start[n++] = i;
            }
            g.dst[i] = en;
            g.tok[i] = en / kExpertsUsed;
        }
        g.start[n] = ne;
        g.n = n;
    }

    // ---- graphs per window size: per layer the launches before the expert placement (A) and after it (B), the head ----
    std::vector<cudaGraphExec_t> g_a, g_b, g_head;
    std::vector<bool> warm;   // [T]: a window of T rows ran every launch once, so its graphs may be captured
    float* h_logits = nullptr;
    bool graphs() const { return !o.no_graphs && !o.profile && o.dump_dir.empty() && o.feed_dir.empty(); }
    // runs `f` as the graph `g`, captured on its first use once `capture` holds (a warm-up step ran every launch once)
    template <class F>
    bool segment(cudaGraphExec_t& g, bool capture, F&& f, std::string& err) {
        if (!capture) {
            f();
            return true;
        }
        if (g == nullptr) {
            cudaGraph_t gr = nullptr;
            if (!cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "graph capture", err)) return false;
            f();
            if (!cuda_ok(cudaStreamEndCapture(s, &gr), "graph capture", err)) return false;
            const cudaError_t e = cudaGraphInstantiate(&g, gr, 0);
            cudaGraphDestroy(gr);
            if (!cuda_ok(e, "graph instantiate", err)) return false;
        }
        return cuda_ok(cudaGraphLaunch(g, s), "graph launch", err);
    }

    // diagnostic (--profile): GPU time by phase, from events recorded after each phase's launches
    enum Phase { kPhAttn, kPhIdx, kPhExperts, kPhFfn, kPhEngram, kPhHead, kPhCopy, kPhCount };
    std::vector<cudaEvent_t> ev_pool;
    std::vector<int> ev_phase;
    double ph_ms[kPhCount] = {}, host_ms = 0;
    int ph_steps = 0;
    void mark(int phase) {
        if (!o.profile) return;
        if (ev_phase.size() == ev_pool.size()) {
            cudaEvent_t e;
            cudaEventCreate(&e);
            ev_pool.push_back(e);
        }
        cudaEventRecord(ev_pool[ev_phase.size()], s);
        ev_phase.push_back(phase);
    }
    std::vector<std::vector<float>> dumps;
    std::vector<std::string> dump_names;

    // ---- loading ----
    const core::Json* meta_of(const std::string& name, std::string& err) {
        const core::Json* t = index->get(name);
        if (t == nullptr) err = "the pack has no tensor " + name;
        return t;
    }

    bool read_f32(const std::string& name, std::vector<float>& out, std::vector<int64_t>& shape, std::string& err) {
        const core::Json* t = meta_of(name, err);
        if (t == nullptr) return false;
        const std::string dt = t->get("dtype")->s;
        const uint64_t off = t->get("offset")->u64(), bytes = t->get("bytes")->u64();
        shape.clear();
        uint64_t n = 1;
        for (const core::Json& d : t->get("shape")->items) {
            shape.push_back(d.i64());
            n *= (uint64_t) d.i64();
        }
        std::vector<uint8_t> raw(bytes);
        if (!core::read_range_at(trunk, off, raw.data(), bytes, err)) return false;
        out.resize(n);
        if (dt == "BF16") {
            const uint16_t* p = (const uint16_t*) raw.data();
            for (uint64_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
        } else if (dt == "F32") {
            std::memcpy(out.data(), raw.data(), n * 4);
        } else if (dt == "I32") {
            const int32_t* p = (const int32_t*) raw.data();
            for (uint64_t i = 0; i < n; ++i) out[i] = (float) p[i];
        } else {
            err = name + ": dtype " + dt + " is not read";
            return false;
        }
        return true;
    }

    float* up_f32(const std::string& name, std::string& err, size_t want = 0) {
        std::vector<float> v;
        std::vector<int64_t> sh;
        if (!read_f32(name, v, sh, err)) return nullptr;
        if (want && v.size() != want) {
            err = name + ": " + std::to_string(v.size()) + " values, expected " + std::to_string(want);
            return nullptr;
        }
        float* d = dalloc<float>(v.size(), owned);
        if (d == nullptr || cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = name + ": upload failed";
            return nullptr;
        }
        trunk_bytes_gpu += v.size() * 4;
        return d;
    }

    // concatenated fp32 copies of `names` (each `each` values)
    float* up_f32_cat(const std::vector<std::string>& names, size_t each, std::string& err) {
        std::vector<float> all;
        for (const std::string& n : names) {
            std::vector<float> v;
            std::vector<int64_t> sh;
            if (!read_f32(n, v, sh, err)) return nullptr;
            if (v.size() != each) { err = n + ": unexpected size"; return nullptr; }
            all.insert(all.end(), v.begin(), v.end());
        }
        float* d = dalloc<float>(all.size(), owned);
        if (d == nullptr || cudaMemcpy(d, all.data(), all.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "upload failed";
            return nullptr;
        }
        trunk_bytes_gpu += all.size() * 4;
        return d;
    }

    bool up_mat(const std::string& name, Mat& m, std::string& err, int type = -1) {
        return up_mat_rows({name}, m, err, type);
    }

    // the matrices `names` (one input width) stacked by rows into one
    bool up_mat_rows(const std::vector<std::string>& names, Mat& m, std::string& err, int type = -1) {
        std::vector<float> v;
        m.n_out = 0;
        for (const std::string& part_name : names) {
            std::vector<float> part;
            std::vector<int64_t> sh;
            if (!read_f32(part_name, part, sh, err)) return false;
            if (sh.size() != 2 || (m.n_out > 0 && (int) sh[1] != m.n_in)) {
                err = part_name + ": not a matrix of the stack's width";
                return false;
            }
            m.n_out += (int) sh[0];
            m.n_in = (int) sh[1];
            v.insert(v.end(), part.begin(), part.end());
        }
        const std::string& name = names[0];
        m.type = type >= 0 ? type : o.trunk_type;
        if (m.n_in % ggml_blck_size((ggml_type) m.type) != 0) m.type = 8;   // Q6_K needs 256-wide rows: Q8_0 then
        const size_t bytes = ggml_row_size((ggml_type) m.type, m.n_in) * (size_t) m.n_out;
        std::vector<uint8_t> q(bytes);
        ggml_quantize_chunk((ggml_type) m.type, v.data(), q.data(), 0, m.n_out, m.n_in);
        m.w = dalloc<uint8_t>(bytes, owned);
        if (m.w == nullptr || cudaMemcpy(m.w, q.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = name + ": upload failed";
            return false;
        }
        trunk_bytes_gpu += bytes;
        return true;
    }

    uint16_t* up_f16(const std::string& name, std::string& err) {
        std::vector<float> v;
        std::vector<int64_t> sh;
        if (!read_f32(name, v, sh, err)) return nullptr;
        std::vector<__half> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
        uint16_t* d = dalloc<uint16_t>(h.size(), owned);
        if (d == nullptr || cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = name + ": upload failed";
            return nullptr;
        }
        trunk_bytes_gpu += h.size() * 2;
        return d;
    }

    // a BF16 tensor as stored (the embedding: embed_bf16 reads its rows)
    uint16_t* up_bf16(const std::string& name, std::string& err) {
        const core::Json* t = meta_of(name, err);
        if (t == nullptr) return nullptr;
        if (t->get("dtype")->s != "BF16") { err = name + ": not BF16"; return nullptr; }
        const uint64_t off = t->get("offset")->u64(), bytes = t->get("bytes")->u64();
        std::vector<uint8_t> raw(bytes);
        if (!core::read_range_at(trunk, off, raw.data(), bytes, err)) return nullptr;
        uint16_t* d = dalloc<uint16_t>(bytes / 2, owned);
        if (d == nullptr || cudaMemcpy(d, raw.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = name + ": upload failed";
            return nullptr;
        }
        trunk_bytes_gpu += bytes;
        return d;
    }

    bool hc(const std::string& pre, HcMix& m, int mixes, std::string& err) {
        const bool head = mixes == kHc;
        m.fn = up_f32(pre + (head ? ".hc_head_fn" : ".hc_fn"), err, (size_t) mixes * kHcFlat);
        m.base = up_f32(pre + (head ? ".hc_head_base" : ".hc_base"), err, (size_t) mixes);
        m.scale = up_f32(pre + (head ? ".hc_head_scale" : ".hc_scale"), err, (size_t) (mixes / kHc));
        m.m = mixes;
        return m.fn && m.base && m.scale;
    }

    bool load_layer(int l, std::string& err) {
        Layer& y = L[l];
        const std::string p = "model.layers." + std::to_string(l);
        y.sliding = sliding_layer(l);
        y.heads = layer_heads(l);
        y.theta = y.sliding ? kRopeThetaSliding : kRopeThetaSparse;
        if (!(y.in_norm = up_f32(p + ".input_layernorm.weight", err, kEmbd))) return false;
        if (!(y.post_attn_norm = up_f32(p + ".post_attention_layernorm.weight", err, kEmbd))) return false;
        const std::string a = p + ".self_attn";
        if (!up_mat(a + ".q_a_proj.weight", y.q_a, err) || !up_mat(a + ".q_b_proj.weight", y.q_b, err) ||
            !up_mat(a + ".kv_a_proj_with_mqa.weight", y.kv_a, err) || !up_mat(a + ".linear_gate.weight", y.gate, err) ||
            !up_mat(a + ".o_proj.weight", y.o, err))
            return false;
        if (!(y.q_a_norm = up_f32(a + ".q_a_layernorm.weight", err, kQRank)) ||
            !(y.kv_a_norm = up_f32(a + ".kv_a_layernorm.weight", err, kKvRank)) ||
            !(y.post_proj_norm = up_f32(a + ".post_proj_norm.weight", err, kEmbd)) ||
            !(y.sink = up_f32(a + ".learnable_sink_param", err, (size_t) y.heads)) ||
            !(y.kv_b = up_f16(a + ".kv_b_proj.weight", err)))
            return false;
        if (y.q_b.n_out != y.heads * kQkHead || y.gate.n_out != y.heads * kVHead || y.o.n_in != y.heads * kVHead) {
            err = a + ": the head count does not match the layer type";
            return false;
        }
        if (!hc(p + ".hc_attn_layer.hc_pre", y.attn_hc, kHcMixes, err) || !hc(p + ".hc_mlp_layer.hc_pre", y.mlp_hc, kHcMixes, err))
            return false;
        y.indexer = !y.sliding;
        if (y.indexer) {
            const std::string x = a + ".compressed_indexer";
            if (!up_mat(x + ".query_projection.weight", y.idx_q, err) || !up_mat(x + ".token_key_projection.weight", y.idx_tk, err) ||
                !up_mat(x + ".head_weight_projection.weight", y.idx_hw, err, 8))
                return false;
            if (!(y.idx_tk_norm = up_f32(x + ".token_key_norm.weight", err, kIdxDim)) ||
                !(y.pool.kv_w = up_f32(x + ".key_value_projection.weight", err, kIdxDim * kIdxDim)) ||
                !(y.pool.gate_w = up_f32(x + ".key_gate_projection.weight", err, kIdxDim * kIdxDim)) ||
                !(y.pool.ape = up_f32(x + ".intra_group_ape", err, kIdxRatio * kIdxDim)) ||
                !(y.pool.key_norm = up_f32(x + ".key_norm.weight", err, kIdxDim)))
                return false;
            y.pool.eps = kRmsEps;
        }
        const std::string f = p + ".mlp";
        y.dense = l == 0;
        if (y.dense) {
            if (!up_mat(f + ".gate_proj.weight", y.d_gate, err) || !up_mat(f + ".up_proj.weight", y.d_up, err) ||
                !up_mat(f + ".down_proj.weight", y.d_down, err) || !(y.post_fc2_norm = up_f32(f + ".post_fc2_norm.weight", err, kEmbd)))
                return false;
        } else {
            if (!(y.router = up_f32(f + ".gate.weight", err, (size_t) kExperts * kEmbd)) ||
                !(y.router_bias = up_f32(f + ".gate.e_score_correction_bias", err, kExperts)) ||
                !(y.combine_norm = up_f32(f + ".combine_output_norm.weight", err, kEmbd)))
                return false;
            if (!up_mat(f + ".routed_expert_down_proj.weight", y.lat_in, err) || !up_mat(f + ".routed_expert_up_proj.weight", y.lat_out, err) ||
                !up_mat(f + ".shared_experts.gate_proj.weight", y.sh_gate, err) || !up_mat(f + ".shared_experts.up_proj.weight", y.sh_up, err) ||
                !up_mat(f + ".shared_experts.down_proj.weight", y.sh_down, err))
                return false;
        }
        y.cap = y.sliding ? kWindow + kMaxT : o.max_context;
        y.ckv = dalloc<uint16_t>((size_t) y.cap * kKvRank, owned);
        y.kpe = dalloc<uint16_t>((size_t) y.cap * kRope, owned);
        if (y.indexer) {
            y.tok_keys = dalloc<float>((size_t) o.max_context * kIdxDim, owned);
            y.grp_keys = dalloc<float>((size_t) (o.max_context / kIdxRatio + 1) * kIdxDim, owned);
        }
        return y.ckv && y.kpe && (!y.indexer || (y.tok_keys && y.grp_keys));
    }

    bool load_engram(const core::Json& e, std::string& err) {
        Engram g;
        const int layer = (int) e.get("layer")->i64();
        const std::string path = e.get("file")->s;
        g.fd = platform::open_fd_utf8(path);
        if (g.fd < 0) { err = "cannot open the Engram table file " + path; return false; }
        g.off = e.get("offset")->u64();
        g.rows = e.get("rows")->u64();
        const core::Json* sz = e.get("head_sizes");
        const core::Json* mu = e.get("multipliers");
        if (sz == nullptr || mu == nullptr || sz->count != kEngramHeads || mu->count != kEngramMaxNgram) {
            err = "meta.engram: head_sizes / multipliers missing or of the wrong length";
            return false;
        }
        int64_t acc = 0;
        for (const core::Json& v : sz->items) {
            g.sizes.push_back(v.i64());
            g.offsets.push_back(acc);
            acc += v.i64();
        }
        for (int i = 0; i < kEngramMaxNgram; ++i) g.mult[i] = mu->items[(size_t) i].i64();
        g.pad = (int32_t) e.get("pad")->i64();
        const std::string p = "model.layers." + std::to_string(layer) + ".engram_layer";
        std::vector<float> lk;
        std::vector<int64_t> sh;
        if (!read_f32(p + ".lookup_table", lk, sh, err)) return false;
        g.lookup.assign(lk.begin(), lk.end());
        std::vector<std::string> kw, kb, nk, nq, ps, hn;
        for (int s = 0; s < kHc; ++s) kw.push_back(p + ".key_projs." + std::to_string(s) + ".weight");
        if (!up_mat_rows(kw, g.key, err) || !up_mat(p + ".value_proj.weight", g.value, err)) return false;
        for (int s = 0; s < kHc; ++s) {
            kb.push_back(p + ".key_projs." + std::to_string(s) + ".bias");
            nk.push_back(p + ".norm_key." + std::to_string(s) + ".weight");
            nq.push_back(p + ".norm_query." + std::to_string(s) + ".weight");
            ps.push_back(p + ".unified_per_hc_scales." + std::to_string(s));
        }
        for (int h = 0; h < kEngramHeads; ++h) hn.push_back(p + ".lookup_per_head_norms." + std::to_string(h) + ".weight");
        g.w.key_bias = up_f32_cat(kb, kEmbd, err);
        g.w.norm_key = up_f32_cat(nk, kEmbd, err);
        g.w.norm_query = up_f32_cat(nq, kEmbd, err);
        g.w.per_hc_scale = up_f32_cat(ps, kEmbd, err);
        g.head_norm = up_f32_cat(hn, kEngramHeadDim, err);
        g.w.value_bias = up_f32(p + ".value_proj.bias", err, kEmbd);
        g.w.pre_gamma = up_f32(p + ".unified_pre_gamma", err, kEmbd);
        g.w.rms_eps = kRmsEps;
        g.w.engram_eps = kEngramEps;
        if (!g.w.key_bias || !g.w.norm_key || !g.w.norm_query || !g.w.per_hc_scale || !g.head_norm || !g.w.value_bias ||
            !g.w.pre_gamma)
            return false;
        L[layer].engram = (int) eg.size();
        eg.push_back(std::move(g));
        return true;
    }

    bool load_experts(std::string& err) {
        std::vector<uint8_t> text;
        if (!file.read("experts.layout", text, err)) return false;
        std::istringstream in(std::string(text.begin(), text.end()));
        std::string line;
        layer_off.assign(kLayers, 0);
        int lines = 0, type = -1;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            int el = 0, gu = 0, d = 0, up = 0;
            uint64_t off = 0, blob = 0;
            if (!(ls >> el >> gu >> d >> off >> blob >> up)) { err = "experts.layout: a bad line"; return false; }
            if (el < 0 || el + 1 >= kLayers) { err = "experts.layout: layer out of range"; return false; }
            if (blob_bytes != 0 && blob != blob_bytes) { err = "experts.layout: the layers differ in blob size"; return false; }
            blob_bytes = blob;
            type = gu;
            layer_off[(size_t) el + 1] = off;
            ++lines;
        }
        if (lines != kLayers - 1) { err = "experts.layout: " + std::to_string(lines) + " layers, expected 41"; return false; }
        if (!k::native_expert_supported(type, type, kLatent, kExpertFf)) {
            err = "the experts' format (type " + std::to_string(type) + ") has no GPU kernels at 1536 x 1536";
            return false;
        }
        xl = k::native_expert_layout(type, type, kLatent, kExpertFf, kSwigluLimit);
        if (xl.bytes != blob_bytes) { err = "the expert blob size does not match its format"; return false; }
        const core::ModelFile::Section* sec = file.find("experts");
        if (sec == nullptr) { err = "the pack has no experts section"; return false; }
        experts = core::vmm::alloc_host(sec->bytes, experts_mapped);
        if (experts == nullptr) { err = "cannot pin " + std::to_string(sec->bytes >> 20) + " MiB of RAM for the experts"; return false; }
        const int fd = platform::open_fd_utf8(file.path());
        if (fd < 0) { err = "cannot open " + file.path(); return false; }
        constexpr uint64_t kChunk = 256ull << 20;
        for (uint64_t at = 0; at < sec->bytes; at += kChunk) {
            const uint64_t n = std::min(kChunk, sec->bytes - at);
            if (!platform::pread_all(fd, experts + at, n, sec->offset + at)) {
                platform::close_fd(fd);
                err = "reading the experts failed";
                return false;
            }
        }
        platform::close_fd(fd);
        return true;
    }

    bool alloc_buffers(std::string& err) {
        const int max_in = kHeadsSliding * kVHead;   // 16384: the widest GEMV input (o_proj of a sliding layer)
        constexpr size_t M = kMaxT;
        streams = dalloc<float>(M * kHcFlat, owned);
        y = dalloc<float>(M * kEmbd, owned);
        xn = dalloc<float>(M * kEmbd, owned);
        qr = dalloc<float>(M * kQRank, owned);
        q = dalloc<float>(M * kMaxHeads * kQkHead, owned);
        kva = dalloc<float>(M * kKeyDim, owned);
        gate_b = dalloc<float>(M * kMaxHeads * kVHead, owned);
        q_lat = dalloc<float>(M * kMaxHeads * kKvRank, owned);
        o_lat = dalloc<float>(M * kMaxHeads * kKvRank, owned);
        o_heads = dalloc<float>(M * kMaxHeads * kVHead, owned);
        a = dalloc<float>(M * kEmbd, owned);
        logits = dalloc<float>(M * kVocab, owned);
        router_logits = dalloc<float>(M * kExperts, owned);
        route_w = dalloc<float>(M * kExpertsUsed, owned);
        lat = dalloc<float>(M * kLatent, owned);
        exp_rows = dalloc<float>(M * kExpertsUsed * kLatent, owned);
        lat_sum = dalloc<float>(M * kLatent, owned);
        sh_g = dalloc<float>(M * kSharedFf, owned);
        sh_u = dalloc<float>(M * kSharedFf, owned);
        ffn_a = dalloc<float>(M * kEmbd, owned);
        ffn_b = dalloc<float>(M * kEmbd, owned);
        dense_g = dalloc<float>(M * kDenseFf, owned);
        dense_u = dalloc<float>(M * kDenseFf, owned);
        idx_q = dalloc<float>(M * kIdxHeads * kIdxDim, owned);
        idx_hw = dalloc<float>(M * kIdxHeads, owned);
        idx_tk = dalloc<float>(M * kIdxDim, owned);
        idx_scr = dalloc<float>(indexer_scratch_floats(o.max_context / kIdxRatio + 1, kMaxT), owned);
        eg_rows = dalloc<float>(M * kEngramLayers * kEngramHeads * kEngramHeadDim, owned);
        eg_e = dalloc<float>(M * kEngramDim, owned);
        eg_key = dalloc<float>(M * kHc * kEmbd, owned);
        eg_val = dalloc<float>(M * kEmbd, owned);
        hc_pre = dalloc<float>(M * kHc, owned);
        hc_post = dalloc<float>(M * kHc, owned);
        xq = dalloc<uint8_t>(k::native_q8_1_bytes(std::max(max_in, kDenseFf), kMaxT), owned);
        xq2 = dalloc<uint8_t>(k::native_q8_1_bytes(std::max(max_in, kDenseFf), kMaxT), owned);
        exp_scratch = dalloc<uint8_t>(k::native_expert_scratch_bytes(kMaxEnt, kExpertFf), owned);
        route_ids = dalloc<int32_t>(kMaxEnt, owned);
        sel = dalloc<int32_t>(M * kIdxGroupTopk * kIdxRatio, owned);
        sel_n = dalloc<int32_t>(M, owned);
        small = dalloc<int32_t>(5 * M, owned);
        d_grp = dalloc<Groups>(1, owned);
        if (cudaMallocHost((void**) &h_small, 5 * M * 4) != cudaSuccess ||
            cudaMallocHost((void**) &h_eg_rows, M * kEngramLayers * kEngramHeads * kEngramHeadDim * 4) != cudaSuccess) {
            err = "pinned host buffers";
            return false;
        }
        eg_raw.resize((size_t) kEngramHeads * kEngramHeadDim);
        for (void* p : {(void*) streams, (void*) xq, (void*) exp_scratch, (void*) small, (void*) idx_scr, (void*) d_grp})
            if (p == nullptr) { err = "out of VRAM for the decode buffers"; return false; }
        return true;
    }

    // ---- decode: T rows, row t at position pos + t ----
    int T = 1;
    int32_t* P() const { return small + kMaxT; }           // pos[T]
    int32_t* LoWin() const { return small + 2 * kMaxT; }   // the sliding window's first position [T]
    int32_t* Hi() const { return small + 3 * kMaxT; }      // pos + 1 [T]
    int32_t* Zero() const { return small + 4 * kMaxT; }    // 0 [T]

    void mm(const Mat& m, const void* x_q8_1, float* out) {
        k::native_mmvq(m.type, m.w, x_q8_1, out, m.n_in, m.n_out, T, s);
    }
    void quant(const float* x, int n, void* out) { k::native_quantize_q8_1(x, out, n, T, s); }

    // the 16 rows of Engram layer e for row t, the token at `pos` (the history holds tokens 0..pos)
    bool engram_rows_one(size_t e, int t, int64_t pos, std::string& err) {
        std::vector<uint16_t> raw((size_t) kEngramHeads * kEngramHeadDim);
        {
            Engram& g = eg[e];
            auto cid = [&](int64_t p) -> int64_t {
                const int32_t tk = history[(size_t) p];
                return tk >= 0 && (size_t) tk < g.lookup.size() ? g.lookup[(size_t) tk] : g.pad;
            };
            int64_t prev_pad = -1;   // the last pad before pos
            for (int64_t p = pos - 1; p >= 0 && p >= pos - kEngramMaxNgram; --p)
                if (cid(p) == g.pad) { prev_pad = p; break; }
            const int64_t in_seg = pos - (prev_pad + 1);
            int64_t sh[kEngramMaxNgram];
            for (int s2 = 0; s2 < kEngramMaxNgram; ++s2)
                sh[s2] = (in_seg >= s2 && pos - s2 >= 0) ? cid(pos - s2) : g.pad;
            for (int n = 2; n <= kEngramMaxNgram; ++n) {
                uint64_t mixed = (uint64_t) sh[0] * (uint64_t) g.mult[0];
                for (int j = 1; j < n; ++j) mixed ^= (uint64_t) sh[j] * (uint64_t) g.mult[j];
                const int64_t m = (int64_t) mixed;
                for (int h = (n - 2) * 8; h < (n - 1) * 8; ++h) {
                    int64_t r = m % g.sizes[(size_t) h];
                    if (r < 0) r += g.sizes[(size_t) h];
                    const uint64_t row = (uint64_t) (r + g.offsets[(size_t) h]);
                    if (!platform::pread_all(g.fd, raw.data() + (size_t) h * kEngramHeadDim, kEngramHeadDim * 2,
                                             g.off + row * kEngramHeadDim * 2)) {
                        err = "reading an Engram row failed";
                        return false;
                    }
                }
            }
            float* out = h_eg_rows + (e * kMaxT + (size_t) t) * kEngramHeads * kEngramHeadDim;
            for (size_t i = 0; i < raw.size(); ++i) out[i] = bf16_to_f32(raw[i]);
        }
        return true;
    }

    // diagnostic: the streams of the reference's module `name` at `pos` replace the engine's own (row 0)
    std::map<std::string, std::vector<float>> feed_cache;
    void feed(const std::string& name, int64_t pos) {
        if (o.feed_dir.empty()) return;
        auto it = feed_cache.find(name);
        if (it == feed_cache.end()) {
            std::vector<float> v;
            if (std::FILE* f = platform::open_file_utf8(o.feed_dir + "/" + name + ".npy", "rb")) {
                char magic[8];
                uint16_t hl = 0;
                if (std::fread(magic, 1, 8, f) == 8 && std::fread(&hl, 2, 1, f) == 1 && std::fseek(f, hl, SEEK_CUR) == 0) {
                    float b[4096];
                    size_t n;
                    while ((n = std::fread(b, 4, 4096, f)) > 0) v.insert(v.end(), b, b + n);
                }
                std::fclose(f);
            }
            it = feed_cache.emplace(name, std::move(v)).first;
        }
        if ((size_t) (pos + 1) * kHcFlat <= it->second.size())
            cudaMemcpyAsync(streams, it->second.data() + (size_t) pos * kHcFlat, kHcFlat * 4, cudaMemcpyHostToDevice, s);
    }

    // diagnostic: the streams of every row
    void dump(const std::string& name) {
        if (o.dump_dir.empty()) return;
        size_t idx = dumps.size();
        for (size_t i = 0; i < dump_names.size(); ++i)
            if (dump_names[i] == name) idx = i;
        if (idx == dumps.size()) {
            dumps.emplace_back();
            dump_names.push_back(name);
        }
        std::vector<float> h((size_t) T * kHcFlat);
        cudaMemcpyAsync(h.data(), streams, h.size() * 4, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        dumps[idx].insert(dumps[idx].end(), h.begin(), h.end());
    }

    void attention(int l) {
        Layer& y_ = L[l];
        hc_mix(streams, y_.attn_hc, T, hc_pre, hc_post, y, s);
        rms_norm_rows(y, y_.in_norm, kEmbd, T, kEmbd, kRmsEps, xn, s);
        quant(xn, kEmbd, xq);
        mm(y_.q_a, xq, qr);
        rms_norm_rows(qr, y_.q_a_norm, kQRank, T, kQRank, kRmsEps, qr, s);
        quant(qr, kQRank, xq2);
        mm(y_.q_b, xq2, q);
        mm(y_.kv_a, xq, kva);
        rms_norm_rows(kva, y_.kv_a_norm, kKvRank, T, kKeyDim, kRmsEps, kva, s);
        rope_gptj(kva, T, kKeyDim, kKvRank, kRope, 1, P(), y_.theta, s);
        rope_gptj(q, y_.heads * T, kQkHead, kNope, kRope, y_.heads, P(), y_.theta, s);
        latent_store(kva, P(), T, y_.ckv, y_.kpe, y_.cap, s);   // every row's key before any row attends
        mark(kPhAttn);
        mm(y_.gate, xq, gate_b);
        mla_absorb_q(q, y_.kv_b, y_.heads, T, q_lat, s);
        const float scale = 1.f / std::sqrt((float) kQkHead);
        const LatentCache c{y_.ckv, y_.kpe, y_.cap};
        if (y_.indexer) {   // every switch on the device: the positions a token adds, the group it completes, past kIdxTopk
            mm(y_.idx_tk, xq, idx_tk);
            rms_norm_rows(idx_tk, y_.idx_tk_norm, kIdxDim, T, kIdxDim, kRmsEps, idx_tk, s);
            for (int t = 0; t < T; ++t) store_row(idx_tk + (size_t) t * kIdxDim, y_.tok_keys, kIdxDim, P() + t, s);
            for (int t = 0; t < T; ++t) indexer_pool_at(y_.tok_keys, y_.pool, P() + t, y_.grp_keys, s);
            mm(y_.idx_q, xq2, idx_q);
            rope_gptj(idx_q, kIdxHeads * T, kIdxDim, kIdxDim - kRope, kRope, kIdxHeads, P(), kRopeThetaSparse, s);
            mm(y_.idx_hw, xq, idx_hw);
            indexer_select_at(idx_q, idx_hw, y_.grp_keys, o.max_context / kIdxRatio, P(), T, idx_scr, sel, sel_n, s);
            if (y_.dfly >= 0) {   // the drafter reads the last accepted row's picks
                cudaMemcpyAsync(df.tsel[y_.dfly], sel, (size_t) T * kIdxGroupTopk * kIdxRatio * 4, cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(df.tsel_n[y_.dfly], sel_n, (size_t) T * 4, cudaMemcpyDeviceToDevice, s);
            }
        }
        mark(kPhIdx);
        mla_attend(q_lat, q, c, y_.sliding ? LoWin() : Zero(), Hi(), y_.indexer ? sel : nullptr, sel_n,
                   kIdxGroupTopk * kIdxRatio, y_.sink, y_.heads, T, scale, o_lat, s);
        mla_value_out(o_lat, y_.kv_b, gate_b, y_.heads, T, o_heads, s);
        quant(o_heads, y_.heads * kVHead, xq);
        mm(y_.o, xq, a);
        rms_norm_rows(a, y_.post_proj_norm, kEmbd, T, kEmbd, kRmsEps, a, s);
        hc_apply(streams, a, hc_post, T, s);
        mark(kPhAttn);
    }

    void ffn_pre(int l) {
        Layer& y_ = L[l];
        hc_mix(streams, y_.mlp_hc, T, hc_pre, hc_post, y, s);
        rms_norm_rows(y, y_.post_attn_norm, kEmbd, T, kEmbd, kRmsEps, xn, s);
        quant(xn, kEmbd, xq);
        if (y_.dense) {
            mm(y_.d_gate, xq, dense_g);
            mm(y_.d_up, xq, dense_u);
            k::native_swiglu_quantize_q8_1(dense_g, dense_u, xq2, kDenseFf, T, s);
            mm(y_.d_down, xq2, ffn_a);
            rms_norm_rows(ffn_a, y_.post_fc2_norm, kEmbd, T, kEmbd, kRmsEps, a, s);
        } else {
            gemv_f32(y_.router, xn, kEmbd, kExperts, T, router_logits, s);
            route_sigmoid_topk(router_logits, y_.router_bias, T, route_ids, route_w, s);
            mm(y_.lat_in, xq, lat);
            quant(lat, kLatent, xq2);
            cudaMemcpyAsync(h_ids, route_ids, (size_t) T * kExpertsUsed * 4, cudaMemcpyDeviceToHost, s);
            mark(kPhFfn);
            return;
        }
        hc_apply(streams, a, hc_post, T, s);
        mark(kPhFfn);
    }

    // after the host grouped and placed this window's experts (h_grp)
    void ffn_post(int l) {
        Layer& y_ = L[l];
        const int ne = T * kExpertsUsed;
        cudaMemcpyAsync(d_grp, h_grp, sizeof(Groups), cudaMemcpyHostToDevice, s);
        mark(kPhCopy);   // the cache misses' copies, queued before this segment
        k::native_expert_grouped_clamped(xl, kSwigluLimit, d_grp->ptr, d_grp->start, &d_grp->n, d_grp->dst, d_grp->tok,
                                         ne, ne, xq2, exp_scratch, exp_rows, s);
        mark(kPhExperts);
        expert_sum(exp_rows, route_w, kLatent, T, lat_sum, s);
        quant(lat_sum, kLatent, xq2);
        mm(y_.lat_out, xq2, ffn_a);
        mm(y_.sh_gate, xq, sh_g);
        mm(y_.sh_up, xq, sh_u);
        k::native_swiglu_quantize_q8_1(sh_g, sh_u, xq2, kSharedFf, T, s);
        mm(y_.sh_down, xq2, ffn_b);
        add_rows(ffn_a, ffn_b, T * kEmbd, a, s);
        rms_norm_rows(a, y_.combine_norm, kEmbd, T, kEmbd, kRmsEps, a, s);
        hc_apply(streams, a, hc_post, T, s);
        mark(kPhFfn);
    }

    void engram(int e) {
        Engram& g = eg[(size_t) e];
        const size_t rows = (size_t) kEngramHeads * kEngramHeadDim;
        cudaMemcpyAsync(eg_rows + (size_t) e * kMaxT * rows, h_eg_rows + (size_t) e * kMaxT * rows, (size_t) T * rows * 4,
                        cudaMemcpyHostToDevice, s);
        engram_rows(eg_rows + (size_t) e * kMaxT * rows, g.head_norm, kEngramEps, T, eg_e, s);
        quant(eg_e, kEngramDim, xq2);
        mm(g.key, xq2, eg_key);
        mm(g.value, xq2, eg_val);
        engram_apply(eg_key, eg_val, g.w, T, streams, s);
        mark(kPhEngram);
    }

    // ---- DFly2, the block drafter ----
    struct Dfly {
        bool on = false;
        Layer L[kDflyLayers];                 // the attention fields, in_norm / post_attn_norm and d_gate/d_up/d_down
        float *final_norm = nullptr, *enorm = nullptr, *hnorm = nullptr, *gate = nullptr;
        Mat eh_proj;                          // the MTP fusion: enorm(E(anchor)) :: hnorm(target hidden) -> kEmbd
        float *hc_hnorm = nullptr, *hc_enorm = nullptr;
        Mat hc_gate, hc_up, hc_down;          // the hidden correction: hidden :: previous token -> a residual
        float* conf_w = nullptr;              // the confidence head: a slot's accept chance from its corrected hidden
        float conf_b = 0.f;
        float* conf = nullptr;                // [kDflyBlock] its logits, without the bias
        float* h_conf = nullptr;              // pinned
        int32_t* tsel[kDflyLayers] = {};      // the target layers' picks of the last window, [kMaxT][topk * ratio]
        int32_t* tsel_n[kDflyLayers] = {};    // [kMaxT]
        int32_t* rsel[kDflyLayers] = {};      // the last accepted row's picks [topk * ratio], and [1] its count
        int32_t* rsel_n[kDflyLayers] = {};
        int32_t* bsel = nullptr;              // the block's keys [kDflyBlock][kDflySelStride]
        int32_t* bsel_n = nullptr;
        float *dh = nullptr, *dx = nullptr, *hn = nullptr, *cat = nullptr, *f = nullptr, *e = nullptr, *corr = nullptr;
        int32_t* dsmall = nullptr;            // kMaxT each: pos, hi, zero, then the block's ids
        int32_t* h_dsmall = nullptr;          // pinned
        int32_t* toks = nullptr;              // [kDflyBlock]: the anchor, then the drafts
        int32_t* h_toks = nullptr;            // pinned
        cudaGraphExec_t g = nullptr;
        bool warm = false;
    } df;

    bool load_dfly(std::string& err) {
        if (!o.dfly || index->get("model.dfly_layers.final_norm.weight") == nullptr) return true;
        for (int i = 0; i < kDflyLayers; ++i) {
            Layer& y = df.L[i];
            const std::string p = "model.dfly_layers." + std::to_string(i), a = p + ".self_attn";
            y.heads = kDflyHeads;
            y.theta = kRopeThetaSparse;
            if (!(y.in_norm = up_f32(p + ".input_layernorm.weight", err, kEmbd)) ||
                !(y.post_attn_norm = up_f32(p + ".post_attention_layernorm.weight", err, kEmbd)) ||
                !up_mat(a + ".q_a_proj.weight", y.q_a, err) || !up_mat(a + ".q_b_proj.weight", y.q_b, err) ||
                !up_mat(a + ".kv_a_proj_with_mqa.weight", y.kv_a, err) || !up_mat(a + ".linear_gate.weight", y.gate, err) ||
                !up_mat(a + ".o_proj.weight", y.o, err) || !(y.q_a_norm = up_f32(a + ".q_a_layernorm.weight", err, kQRank)) ||
                !(y.kv_a_norm = up_f32(a + ".kv_a_layernorm.weight", err, kKvRank)) ||
                !(y.post_proj_norm = up_f32(a + ".post_proj_norm.weight", err, kEmbd)) ||
                !(y.sink = up_f32(a + ".learnable_sink_param", err, kDflyHeads)) || !(y.kv_b = up_f16(a + ".kv_b_proj.weight", err)) ||
                !up_mat(p + ".mlp.gate_proj.weight", y.d_gate, err) || !up_mat(p + ".mlp.up_proj.weight", y.d_up, err) ||
                !up_mat(p + ".mlp.down_proj.weight", y.d_down, err))
                return false;
            if (y.q_b.n_out != kDflyHeads * kQkHead || y.d_gate.n_out != kDflyFf) { err = p + ": unexpected shapes"; return false; }
            L[kDflyTarget[i]].dfly = i;
            df.tsel[i] = dalloc<int32_t>((size_t) kMaxT * kIdxGroupTopk * kIdxRatio, owned);
            df.tsel_n[i] = dalloc<int32_t>(kMaxT, owned);
            df.rsel[i] = dalloc<int32_t>((size_t) kIdxGroupTopk * kIdxRatio, owned);
            df.rsel_n[i] = dalloc<int32_t>(1, owned);
        }
        const std::string p = "model.dfly_layers.";
        if (!(df.final_norm = up_f32(p + "final_norm.weight", err, kEmbd)) || !(df.enorm = up_f32(p + "mtp_fusion.enorm.weight", err, kEmbd)) ||
            !(df.hnorm = up_f32(p + "mtp_fusion.hnorm.weight", err, kEmbd)) || !(df.gate = up_f32(p + "mtp_fusion.gate", err, kEmbd)) ||
            !up_mat(p + "mtp_fusion.eh_proj.weight", df.eh_proj, err) ||
            !(df.hc_hnorm = up_f32(p + "hidden_correction.hidden_norm.weight", err, kEmbd)) ||
            !(df.hc_enorm = up_f32(p + "hidden_correction.embed_norm.weight", err, kEmbd)) ||
            !up_mat(p + "hidden_correction.gate_proj.weight", df.hc_gate, err) ||
            !up_mat(p + "hidden_correction.up_proj.weight", df.hc_up, err) ||
            !up_mat(p + "hidden_correction.down_proj.weight", df.hc_down, err) ||
            !(df.conf_w = up_f32(p + "confidence_head.proj.weight", err, kEmbd)))
            return false;
        {
            std::vector<float> b;
            std::vector<int64_t> sh;
            if (!read_f32(p + "confidence_head.proj.bias", b, sh, err) || b.size() != 1) return false;
            df.conf_b = b[0];
        }
        df.bsel = dalloc<int32_t>((size_t) kDflyBlock * kDflySelStride, owned);
        df.bsel_n = dalloc<int32_t>(kDflyBlock, owned);
        df.dh = dalloc<float>((size_t) kDflyBlock * kEmbd, owned);
        df.dx = dalloc<float>((size_t) kDflyBlock * kEmbd, owned);
        df.hn = dalloc<float>((size_t) kDflyBlock * kEmbd, owned);
        df.cat = dalloc<float>(2 * kEmbd, owned);
        df.f = dalloc<float>(kEmbd, owned);
        df.e = dalloc<float>(kEmbd, owned);
        df.corr = dalloc<float>(kEmbd, owned);
        df.dsmall = dalloc<int32_t>(4 * kMaxT, owned);
        df.toks = dalloc<int32_t>(kDflyBlock, owned);
        df.conf = dalloc<float>(kDflyBlock, owned);
        if (df.toks == nullptr || cudaMallocHost((void**) &df.h_dsmall, 4 * kMaxT * 4) != cudaSuccess ||
            cudaMallocHost((void**) &df.h_toks, kDflyBlock * 4) != cudaSuccess ||
            cudaMallocHost((void**) &df.h_conf, kDflyBlock * 4) != cudaSuccess) {
            err = "the drafter's buffers";
            return false;
        }
        df.on = true;
        return true;
    }

    // the block from the device inputs (dsmall, rsel, the target hidden in df.f's place): kDflyBlock - 1 drafts in toks
    void dfly_block() {
        int32_t *pos = df.dsmall, *hi = df.dsmall + kMaxT, *zero = df.dsmall + 2 * kMaxT, *ids = df.dsmall + 3 * kMaxT;
        const float scale = 1.f / std::sqrt((float) kQkHead);
        T = kDflyBlock;
        embed_bf16(embed, ids, kEmbd, T, df.dh, s);
        // slot 0 += gate * eh_proj(enorm(E(anchor)) :: hnorm(h)), h already in cat's second half
        T = 1;
        rms_norm_rows(df.dh, df.enorm, kEmbd, 1, kEmbd, kRmsEps, df.cat, s);
        rms_norm_rows(df.cat + kEmbd, df.hnorm, kEmbd, 1, kEmbd, kRmsEps, df.cat + kEmbd, s);
        quant(df.cat, 2 * kEmbd, xq);
        mm(df.eh_proj, xq, df.f);
        add_scaled(df.dh, df.gate, df.f, kEmbd, s);
        T = kDflyBlock;
        for (int i = 0; i < kDflyLayers; ++i) {
            const Layer& y_ = df.L[i];
            Layer& tl = L[kDflyTarget[i]];
            rms_norm_rows(df.dh, y_.in_norm, kEmbd, T, kEmbd, kRmsEps, df.dx, s);
            quant(df.dx, kEmbd, xq);
            mm(y_.q_a, xq, qr);
            rms_norm_rows(qr, y_.q_a_norm, kQRank, T, kQRank, kRmsEps, qr, s);
            quant(qr, kQRank, xq2);
            mm(y_.q_b, xq2, q);
            mm(y_.kv_a, xq, kva);
            rms_norm_rows(kva, y_.kv_a_norm, kKvRank, T, kKeyDim, kRmsEps, kva, s);
            rope_gptj(kva, T, kKeyDim, kKvRank, kRope, 1, pos, y_.theta, s);
            rope_gptj(q, y_.heads * T, kQkHead, kNope, kRope, y_.heads, pos, y_.theta, s);
            // the block's keys go where the target's own rows for these positions will be written before any read
            latent_store(kva, pos, T, tl.ckv, tl.kpe, tl.cap, s);
            mm(y_.gate, xq, gate_b);
            mla_absorb_q(q, y_.kv_b, y_.heads, T, q_lat, s);
            sel_with_block(df.rsel[i], df.rsel_n[i], pos, kDflyBlock, T, df.bsel, df.bsel_n, s);
            mla_attend(q_lat, q, {tl.ckv, tl.kpe, tl.cap}, zero, hi, df.bsel, df.bsel_n, kDflySelStride, y_.sink, y_.heads,
                       T, scale, o_lat, s);
            mla_value_out(o_lat, y_.kv_b, gate_b, y_.heads, T, o_heads, s);
            quant(o_heads, y_.heads * kVHead, xq);
            mm(y_.o, xq, a);
            rms_norm_rows(a, y_.post_proj_norm, kEmbd, T, kEmbd, kRmsEps, a, s);
            add_rows(df.dh, a, T * kEmbd, df.dh, s);
            rms_norm_rows(df.dh, y_.post_attn_norm, kEmbd, T, kEmbd, kRmsEps, df.dx, s);
            quant(df.dx, kEmbd, xq);
            mm(y_.d_gate, xq, dense_g);
            mm(y_.d_up, xq, dense_u);
            k::native_swiglu_quantize_q8_1(dense_g, dense_u, xq2, kDflyFf, T, s);
            mm(y_.d_down, xq2, a);
            add_rows(df.dh, a, T * kEmbd, df.dh, s);
        }
        rms_norm_rows(df.dh, df.final_norm, kEmbd, T, kEmbd, kRmsEps, df.hn, s);
        // slot j drafts the token after slot j - 1's: its hidden corrected with that token, then the target's head
        T = 1;
        cudaMemcpyAsync(df.toks, ids, 4, cudaMemcpyDeviceToDevice, s);
        for (int j = 1; j < kDflyBlock; ++j) {
            const float* h = df.hn + (size_t) j * kEmbd;
            embed_bf16(embed, df.toks + j - 1, kEmbd, 1, df.e, s);
            rms_norm_rows(h, df.hc_hnorm, kEmbd, 1, kEmbd, kRmsEps, df.cat, s);
            rms_norm_rows(df.e, df.hc_enorm, kEmbd, 1, kEmbd, kRmsEps, df.cat + kEmbd, s);
            quant(df.cat, 2 * kEmbd, xq);
            mm(df.hc_gate, xq, dense_g);
            mm(df.hc_up, xq, dense_u);
            k::native_swiglu_quantize_q8_1(dense_g, dense_u, xq2, kEmbd, 1, s);
            mm(df.hc_down, xq2, df.corr);
            add_rows(h, df.corr, kEmbd, df.corr, s);
            quant(df.corr, kEmbd, xq);
            mm(head, xq, logits);
            argmax_row(logits, kVocab, df.toks + j, s);
            gemv_f32(df.conf_w, df.corr, kEmbd, 1, 1, df.conf + j, s);
        }
        cudaMemcpyAsync(df.h_toks, df.toks, kDflyBlock * 4, cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(df.h_conf, df.conf, kDflyBlock * 4, cudaMemcpyDeviceToHost, s);
    }
};

Engine::Engine() : m_(new Impl) {}

Engine::~Engine() {
    if (m_ == nullptr) return;
    if (m_->hits + m_->misses > 0)
        basalt::log::info("basalt hy4", "expert cache: %.1f%% hits (%llu of %llu routings)",
                          100.0 * m_->hits / (m_->hits + m_->misses), (unsigned long long) m_->hits,
                          (unsigned long long) (m_->hits + m_->misses));
    if (!m_->o.expert_profile.empty() && !m_->counts.empty())
        if (std::FILE* f = platform::open_file_utf8(m_->o.expert_profile, "wb")) {
            std::fwrite(m_->counts.data(), 4, m_->counts.size(), f);
            std::fclose(f);
        }
    if (m_->o.profile && m_->ph_steps > 0) {
        const char* names[] = {"attention", "indexer + attend", "routed experts", "ffn other", "engram", "embed + head",
                               "expert copies"};
        double total = 0;
        for (double v : m_->ph_ms) total += v;
        for (int i = 0; i < Impl::kPhCount; ++i)
            basalt::log::info("basalt hy4", "profile %-18s %7.2f ms/token (%4.1f%%)", names[i], m_->ph_ms[i] / m_->ph_steps,
                              100.0 * m_->ph_ms[i] / total);
        basalt::log::info("basalt hy4", "profile %-18s %7.2f ms/token GPU, %.2f ms/token Engram reads on the host",
                          "total", total / m_->ph_steps, m_->host_ms / m_->ph_steps);
    }
    cudaDeviceSynchronize();
    for (void* p : m_->owned) cudaFree(p);
    if (m_->experts != nullptr) core::vmm::free_host(m_->experts, m_->experts_mapped);
    for (auto& g : m_->eg)
        if (g.fd >= 0) platform::close_fd(g.fd);
    if (m_->h_small) cudaFreeHost(m_->h_small);
    if (m_->h_ids) cudaFreeHost(m_->h_ids);
    if (m_->h_logits) cudaFreeHost(m_->h_logits);
    for (cudaGraphExec_t g : m_->g_a)
        if (g) cudaGraphExecDestroy(g);
    for (cudaGraphExec_t g : m_->g_b)
        if (g) cudaGraphExecDestroy(g);
    for (cudaGraphExec_t g : m_->g_head)
        if (g) cudaGraphExecDestroy(g);
    if (m_->h_grp) cudaFreeHost(m_->h_grp);
    if (m_->df.g) cudaGraphExecDestroy(m_->df.g);
    if (m_->df.h_dsmall) cudaFreeHost(m_->df.h_dsmall);
    if (m_->df.h_toks) cudaFreeHost(m_->df.h_toks);
    if (m_->df.h_conf) cudaFreeHost(m_->df.h_conf);
    if (m_->slots) cudaFree(m_->slots);
    if (m_->h_eg_rows) cudaFreeHost(m_->h_eg_rows);
    if (m_->s) cudaStreamDestroy(m_->s);
    if (!m_->o.dump_dir.empty())
        for (size_t i = 0; i < m_->dumps.size(); ++i) {
            const std::string path = m_->o.dump_dir + "/" + m_->dump_names[i] + ".npy";
            std::FILE* f = platform::open_file_utf8(path, "wb");
            if (f == nullptr) continue;
            const size_t rows = m_->dumps[i].size() / kHcFlat;
            std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                              std::to_string(kHc) + ", " + std::to_string(kEmbd) + "), }";
            while ((10 + hdr.size() + 1) % 64) hdr += ' ';
            hdr += '\n';
            const uint16_t hl = (uint16_t) hdr.size();
            std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
            std::fwrite(&hl, 2, 1, f);
            std::fwrite(hdr.data(), 1, hdr.size(), f);
            std::fwrite(m_->dumps[i].data(), 4, m_->dumps[i].size(), f);
            std::fclose(f);
        }
    delete m_;
}

bool Engine::load(const std::string& pack, const EngineOptions& o, std::string& err) {
    Impl& m = *m_;
    m.o = o;
    if (!m.file.open(pack, err) || !m.file.load_meta(err)) return false;
    const core::Json& meta = m.file.meta();
    const core::Json* arch = meta.get("arch");
    if (arch == nullptr || arch->s != "hy4") { err = pack + ": not a Hy4-flash pack"; return false; }
    m.index = meta.get("trunk");
    if (m.index == nullptr || !m.index->is_object()) { err = "meta.trunk is missing"; return false; }
    m.trunk = m.file.range("trunk");
    if (!cuda_ok(cudaStreamCreateWithFlags(&m.s, cudaStreamNonBlocking), "stream", err)) return false;
    for (int l = 0; l < kLayers; ++l)
        if (!m.load_layer(l, err)) return false;
    const core::Json* egs = meta.get("engram");
    if (egs == nullptr || egs->count != kEngramLayers) { err = "meta.engram: expected 4 tables"; return false; }
    for (const core::Json& e : egs->items)
        if (!m.load_engram(e, err)) return false;
    if (!(m.embed = m.up_bf16("model.embed_tokens.weight", err)) || !m.up_mat("lm_head.weight", m.head, err) ||
        !(m.final_norm = m.up_f32("model.norm.weight", err, kEmbd)) || !m.hc("model.hc_head", m.head_hc, kHc, err) ||
        !m.load_dfly(err))
        return false;
    if (m.df.on) basalt::log::info("basalt hy4", "DFly2 drafter: %d layers, %d drafts per window", kDflyLayers, kDflyBlock - 1);
    basalt::log::info("basalt hy4", "trunk on the GPU: %.2f GiB", (double) m.trunk_bytes_gpu / 1073741824.0);
    if (!m.load_experts(err)) return false;
    basalt::log::info("basalt hy4", "experts pinned in RAM: %.2f GiB", (double) m.experts_mapped / 1073741824.0);
    return m.alloc_buffers(err) && m.cache_init(err);
}

bool Engine::step_window(const int32_t* tokens, int n, int64_t pos, int32_t* picks, float* logits, std::string& err) {
    Impl& m = *m_;
    if (n < 1 || n > kMaxT || pos < 0 || pos > (int64_t) m.history.size() || pos + n > m.o.max_context) {
        err = "a window is 1 to 8 tokens, starts at or before the next position and stays below the context";
        return false;
    }
    // a start before the history drops the later positions: their cache rows are written again before anything reads them
    m.history.resize((size_t) pos);
    m.history.insert(m.history.end(), tokens, tokens + n);
    const int T = n;
    m.T = T;
    for (int t = 0; t < T; ++t) {
        const int64_t p = pos + t;
        m.h_small[t] = tokens[t];
        m.h_small[kMaxT + t] = (int32_t) p;
        m.h_small[2 * kMaxT + t] = (int32_t) std::max<int64_t>(0, p + 1 - kWindow);
        m.h_small[3 * kMaxT + t] = (int32_t) p + 1;
        m.h_small[4 * kMaxT + t] = 0;
    }
    // the Engram rows: one task per layer, read while the layers before it run
    std::vector<std::string> eg_err(m.eg.size());
    std::vector<std::future<bool>> rows;
    for (size_t e = 0; e < m.eg.size(); ++e)
        rows.push_back(std::async(std::launch::async, [&m, e, pos, T, &eg_err] {
            for (int t = 0; t < T; ++t)
                if (!m.engram_rows_one(e, t, pos + t, eg_err[e])) return false;
            return true;
        }));
    const bool cap = m.graphs() && m.warm[(size_t) T];   // a window size runs every launch once before it is captured
    const size_t gi = (size_t) (T - 1) * kLayers;
    m.ev_phase.clear();
    m.mark(Impl::kPhHead);
    cudaMemcpyAsync(m.small, m.h_small, 5 * kMaxT * 4, cudaMemcpyHostToDevice, m.s);
    embed_bf16(m.embed, m.small, kEmbd, T, m.xn, m.s);
    hc_expand(m.xn, m.streams, T, m.s);
    m.dump("001_hc_expand");
    std::string prev = "001_hc_expand";
    int mod = 2;
    for (int l = 0; l < kLayers; ++l) {
        char nm[64];
        const int e = m.L[l].engram;
        if (e >= 0) {
            const auto h0 = std::chrono::steady_clock::now();
            if (!rows[(size_t) e].get()) { err = eg_err[(size_t) e]; return false; }
            m.host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
        }
        if (!cap && e >= 0) {   // the dumps and the feed sit between the Engram module and the block
            m.feed(prev, pos);
            m.engram(e);
            std::snprintf(nm, sizeof nm, "%03d_model.layers.%d.engram_layer", mod++, l);
            m.dump(nm);
            prev = nm;
        }
        if (!cap) m.feed(prev, pos);
        if (!m.segment(m.g_a[gi + (size_t) l], cap, [&] {
                if (cap && e >= 0) m.engram(e);
                m.attention(l);
                m.ffn_pre(l);
            }, err))
            return false;
        if (!m.L[l].dense) {
            if (!cuda_ok(cudaStreamSynchronize(m.s), "the router", err)) return false;
            m.expert_groups_host(l, T);
            if (!m.segment(m.g_b[gi + (size_t) l], cap, [&] { m.ffn_post(l); }, err)) return false;
        }
        if (cap && e >= 0) ++mod;
        std::snprintf(nm, sizeof nm, "%03d_model.layers.%d", mod++, l);
        m.dump(nm);
        prev = nm;
    }
    if (!m.segment(m.g_head[(size_t) T - 1], cap, [&] {
            hc_mix(m.streams, m.head_hc, T, m.hc_pre, nullptr, m.y, m.s);
            rms_norm_rows(m.y, m.final_norm, kEmbd, T, kEmbd, kRmsEps, m.xn, m.s);
            m.quant(m.xn, kEmbd, m.xq);
            m.mm(m.head, m.xq, m.logits);
            cudaMemcpyAsync(m.h_logits, m.logits, (size_t) T * kVocab * 4, cudaMemcpyDeviceToHost, m.s);
        }, err))
        return false;
    m.mark(Impl::kPhHead);
    if (!cuda_ok(cudaStreamSynchronize(m.s), "decode step", err)) return false;
    m.warm[(size_t) T] = true;
    if ((pos + T) / Impl::kHalfLife > pos / Impl::kHalfLife)
        for (float& v : m.score) v *= 0.5f;
    if (m.o.profile) {
        for (size_t i = 1; i < m.ev_phase.size(); ++i) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, m.ev_pool[i - 1], m.ev_pool[i]);
            m.ph_ms[m.ev_phase[i]] += ms;
        }
        m.ph_steps += T;
    }
    if (logits != nullptr) std::memcpy(logits, m.h_logits, (size_t) T * kVocab * 4);
    for (int t = 0; t < T; ++t) {
        const float* lg = m.h_logits + (size_t) t * kVocab;
        picks[t] = (int32_t) (std::max_element(lg, lg + kVocab) - lg);
    }
    return true;
}

int Engine::draft(int32_t anchor, int64_t pos, int32_t* out, float* conf, std::string& err) {
    Impl& m = *m_;
    if (!m.df.on || pos + kDflyBlock > m.o.max_context) return 0;
    // the last accepted token is the row before `pos` of the last window (its later rows are rejected drafts)
    const int64_t last_row = pos - 1 - ((int64_t) m.history.size() - m.T);
    if (last_row < 0 || last_row >= m.T) {
        err = "a draft follows a window, from the position after one of its rows";
        return -1;
    }
    // the last accepted row's final hidden (the head's normed rows) and its picks in each target layer
    cudaMemcpyAsync(m.df.cat + kEmbd, m.xn + (size_t) last_row * kEmbd, kEmbd * 4, cudaMemcpyDeviceToDevice, m.s);
    for (int i = 0; i < kDflyLayers; ++i) {
        cudaMemcpyAsync(m.df.rsel[i], m.df.tsel[i] + (size_t) last_row * kIdxGroupTopk * kIdxRatio,
                        (size_t) kIdxGroupTopk * kIdxRatio * 4, cudaMemcpyDeviceToDevice, m.s);
        cudaMemcpyAsync(m.df.rsel_n[i], m.df.tsel_n[i] + last_row, 4, cudaMemcpyDeviceToDevice, m.s);
    }
    int32_t* h = m.df.h_dsmall;
    for (int j = 0; j < kDflyBlock; ++j) {
        h[j] = (int32_t) (pos + j);
        h[kMaxT + j] = (int32_t) (pos + kDflyBlock);
        h[2 * kMaxT + j] = 0;
        h[3 * kMaxT + j] = j == 0 ? anchor : kDflyMaskToken;
    }
    cudaMemcpyAsync(m.df.dsmall, h, 4 * kMaxT * 4, cudaMemcpyHostToDevice, m.s);
    if (!m.segment(m.df.g, m.graphs() && m.df.warm, [&] { m.dfly_block(); }, err)) return -1;
    if (!cuda_ok(cudaStreamSynchronize(m.s), "the draft", err)) return -1;
    m.df.warm = true;
    std::memcpy(out, m.df.h_toks + 1, (kDflyBlock - 1) * 4);
    if (conf != nullptr)
        for (int j = 1; j < kDflyBlock; ++j) conf[j - 1] = 1.f / (1.f + std::exp(-(m.df.h_conf[j] + m.df.conf_b)));
    return kDflyBlock - 1;
}

int32_t Engine::step(int32_t token, int64_t pos, float* logits, std::string& err) {
    int32_t pick = -1;
    return step_window(&token, 1, pos, &pick, logits, err) ? pick : -1;
}

}  // namespace basalt::models::hy4
