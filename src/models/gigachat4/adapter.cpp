// src/models/gigachat4/adapter.cpp - GigaChat4 adapter: registers the architecture, validates and describes the pack,
// parses model flags (--kv, --spec, ...), and wraps gigachat4::Engine as a LoadedModel and ModelSession.
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/checkpoint_regions.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"
#include "basalt/program/pretok_o200k.hpp"
#include "models/gigachat4/engine.hpp"
#include "models/gigachat4/meta.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::gigachat4 {
namespace {

namespace bk = basalt::kernels;
constexpr int kMaxWindow = 8;

bool pack_has_mtp(const std::string& path) {
    if (path.empty()) return false;
    core::ModelFile file;
    std::string err;
    if (!file.open(path, err) || !file.load_meta(err)) return false;
    const core::Json* tens = file.meta().get("tensors");
    return tens != nullptr && tens->is_object() && tens->get("mtp.eh") != nullptr;
}

class Session final : public model::ModelSession {
public:
    Session(Engine& eng, int spec) : eng_(eng), spec_(spec) {
        st_ = (cudaStream_t) eng_.cuda_stream();
        const int vocab = eng_.geometry().n_vocab;
        cudaMalloc((void**) &pick_, 32 * sizeof(int32_t));
        cudaMallocHost((void**) &pick_h_, kMaxWindow * sizeof(int32_t));
        lp_floats_ = (size_t) bk::row_logprobs_scratch_floats(kMaxWindow, vocab) +
                     (size_t) kMaxWindow * (bk::kLogprobStride + 1);
        cudaMalloc((void**) &lp_, lp_floats_ * sizeof(float));
    }

    ~Session() override {
        if (pick_) cudaFree(pick_);
        if (pick_h_) cudaFreeHost(pick_h_);
        if (lp_) cudaFree(lp_);
        if (d_bias_) cudaFree(d_bias_);
        if (d_mask_) cudaFree(d_mask_);
        if (d_hist_) cudaFree(d_hist_);
    }

    int max_window() const override { return spec_ >= 2 ? std::clamp(spec_, 1, kMaxWindow) : 1; }
    bool copy_drafts() const override { return true; }

    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        if (!eng_.mtp_active()) return {model::Status::NotImplemented, "gigachat4: no drafter active"};
        std::string err;
        if (!eng_.draft_begin(anchor, (int) pos, n, err)) return {model::Status::Error, err};
        return {};
    }

    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed& need) override {
        if (!eng_.mtp_active()) return {model::Status::NotImplemented, "gigachat4: no drafter active"};
        std::string err;
        if (!eng_.draft_end(ids, conf, max_n, n, need, err)) return {model::Status::Error, err};
        return {};
    }

    model::Result decode_kept(ModelSession* const*, const int* kept, int n) override {
        if (n >= 1 && kept) eng_.commit_kept(kept[0]);
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos + r.n_tokens > eng_.max_ctx())
            return {model::Status::Refused, "gigachat4: prompt exceeds max_context"};
        if (r.pos == 0) eng_.reset();
        const int max_chunk = eng_.prefill_chunk_size();
        for (int at = 0; at < r.n_tokens; at += max_chunk) {
            if (r.stop && r.stop()) return {model::Status::Error, "cancelled"};
            const int chunk = std::min(max_chunk, r.n_tokens - at);
            eng_.forward(r.tokens + at, chunk, (int) (r.pos + at), false);
            if (r.progress) r.progress(at + chunk);
        }
        return {};
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens <= 0 || r.n_tokens > kMaxWindow)
            return {model::Status::Error, "gigachat4 decode_window expects 1..8 tokens"};
        if (r.pos + r.n_tokens > eng_.max_ctx())
            return {model::Status::Refused, "gigachat4: context full"};
        std::string err;
        if (!eng_.decode_forward(r.tokens, r.n_tokens, (int) r.pos, err))
            return {model::Status::Error, err};
        last_rows_ = r.n_tokens;
        const int vocab = eng_.geometry().n_vocab;
        if (r.logits) {
            const float* src = eng_.logits() + (size_t) (r.n_tokens - 1) * vocab;
            if (cudaMemcpyAsync(r.logits, src, (size_t) vocab * sizeof(float), cudaMemcpyDeviceToHost, st_) != cudaSuccess)
                return {model::Status::Error, "gigachat4 cudaMemcpy logits failed"};
        }
        if (r.next_token || r.row_tokens) {
            bk::SamplerParams sp;
            if (r.sampling) sp = *r.sampling;
            else { sp.greedy = true; sp.temperature = 0.f; }
            sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
            history_len_ = r.history != nullptr ? r.history_len : 0;
            if (history_len_ > 0) {
                const size_t need = (size_t) r.n_tokens * history_len_;
                if (d_hist_len_ < need) {
                    if (d_hist_) cudaFree(d_hist_);
                    if (cudaMalloc((void**) &d_hist_, need * sizeof(int32_t)) != cudaSuccess)
                        return {model::Status::Error, "gigachat4: out of VRAM for sampling history"};
                    d_hist_len_ = need;
                }
                if (cudaMemcpyAsync(d_hist_, r.history, need * sizeof(int32_t), cudaMemcpyHostToDevice, st_) != cudaSuccess)
                    return {model::Status::Error, "gigachat4: sampling history upload failed"};
            }
            sampling_ = sp;
            bk::sample_tokens(eng_.logits(), r.n_tokens, vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sp, pick_, st_);
            if (cudaMemcpyAsync(pick_h_, pick_, (size_t) r.n_tokens * sizeof(int32_t), cudaMemcpyDeviceToHost, st_) != cudaSuccess)
                return {model::Status::Error, "gigachat4: pick download failed"};
        }
        if (cudaStreamSynchronize(st_) != cudaSuccess)
            return {model::Status::Error, "gigachat4: decode_window sync failed"};
        if (r.row_tokens)
            for (int row = 0; row < r.n_tokens; ++row) r.row_tokens[row] = pick_h_[row];
        if (r.next_token) *r.next_token = pick_h_[r.n_tokens - 1];
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string err;
        if (!upload_bias(d_bias_, bias, err)) return {model::Status::Error, err};
        return {};
    }

    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        if (row < 0 || row >= last_rows_)
            return {model::Status::Error, "gigachat4: resample_row out of last window"};
        std::string err;
        if (!upload_bias(d_mask_, bias, err)) return {model::Status::Error, err};
        const int vocab = eng_.geometry().n_vocab;
        bk::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        bk::sample_tokens(eng_.logits() + (size_t) row * vocab, 1, vocab,
                          history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr,
                          history_len_, sp, pick_ + 24, st_);
        int32_t t = 0;
        if (cudaMemcpyAsync(&t, pick_ + 24, sizeof(int32_t), cudaMemcpyDeviceToHost, st_) != cudaSuccess ||
            cudaStreamSynchronize(st_) != cudaSuccess)
            return {model::Status::Error, "gigachat4: masked resample failed"};
        token = t;
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= last_rows_)
            return {model::Status::Error, "gigachat4: copy_logits out of last window"};
        const int vocab = eng_.geometry().n_vocab;
        const float* src = eng_.logits() + (size_t) row * vocab;
        if (cudaMemcpy(out, src, (size_t) vocab * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return {model::Status::Error, "gigachat4 cudaMemcpy logits failed"};
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n < 1 || row0 + n > last_rows_)
            return {model::Status::Error, "gigachat4: row_logprobs out of last window"};
        const int vocab = eng_.geometry().n_vocab;
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(n, vocab);
        int32_t* d_pick = pick_ + 16;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, st_);
        bk::row_logprobs(eng_.logits() + (size_t) row0 * vocab, n, vocab, d_pick, top, d_out, lp_, st_);
        cudaMemcpyAsync(out, d_out, (size_t) n * bk::kLogprobStride * sizeof(float), cudaMemcpyDeviceToHost, st_);
        return cudaStreamSynchronize(st_) == cudaSuccess
                   ? model::Result{}
                   : model::Result{model::Status::Error, "gigachat4: row_logprobs failed"};
    }

    size_t running_bytes() const override { return eng_.running_bytes(); }

    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (!eng_.running_state(pos, save, st))
            return {model::Status::Error, "gigachat4: running_state failed"};
        return {};
    }

    model::Result running_restored(int64_t pos, const uint8_t*) override {
        eng_.on_running_restored(pos);
        return {};
    }

    size_t cells_bytes(int64_t positions) const override { return eng_.cells_bytes(positions); }

    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        std::string err;
        if (!eng_.save_cells(kv, positions, err)) return {model::Status::Error, err};
        return {};
    }

    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        std::string err;
        if (!eng_.restore_cells(kv, positions, err)) return {model::Status::Error, err};
        return {};
    }

    ExpertCounts expert_counts() const override {
        uint64_t host = 0, reads = 0;
        eng_.expert_stats(host, reads);
        const int64_t hits = (int64_t) (reads - host);
        return {hits, (int64_t) reads, (int64_t) host, 0};
    }

    std::vector<CardExperts> experts_by_card() const override {
        const Geometry& g = eng_.geometry();
        const uint64_t blob = (uint64_t) 3 * g.n_ff_exp * g.n_embd / 64 * 36 + 16;
        const int64_t slots = eng_.cached_experts();
        return {{0, slots, (uint64_t) slots * blob}};
    }

    int64_t kv_bytes() const override { return eng_.kv_bytes(); }

    std::string info_words() const override {
        return "expert_slots=" + std::to_string(eng_.cached_experts()) +
               " kv=" + std::string(eng_.kv_fp8() ? "fp8" : "fp16");
    }

private:
    bool upload_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) eng_.geometry().n_vocab * sizeof(float);
        if (!dev && cudaMalloc((void**) &dev, bytes) != cudaSuccess) {
            err = "gigachat4: out of VRAM for logit bias";
            return false;
        }
        if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, st_) != cudaSuccess) {
            err = "gigachat4: logit bias upload failed";
            return false;
        }
        return true;
    }

    Engine& eng_;
    int spec_ = 0;
    cudaStream_t st_ = nullptr;
    int32_t *pick_ = nullptr, *pick_h_ = nullptr, *d_hist_ = nullptr;
    float *lp_ = nullptr, *d_bias_ = nullptr, *d_mask_ = nullptr;
    size_t lp_floats_ = 0, d_hist_len_ = 0;
    int history_len_ = 0, last_rows_ = 0;
    bool bias_on_ = false;
    bk::SamplerParams sampling_;
};

class GigaChat4Model final : public model::LoadedModel {
public:
    const model::ModelDesc& desc() const override { return desc_; }

    model::Result plan(const model::LoadRequest& req, core::LoadPlan&, std::vector<std::unique_ptr<model::ModelSession>>&) override {
        Geometry g;
        desc_.arch = kArch;
        desc_.n_layers = g.n_layers;
        desc_.n_embd = g.n_embd;
        desc_.n_vocab = g.n_vocab;
        desc_.n_expert = g.n_expert;
        desc_.n_expert_used = g.n_expert_used;
        desc_.n_ff_exp = g.n_ff_exp;
        desc_.n_expert_shared = g.n_expert_shared;
        desc_.context_length = req.max_context > 0 ? req.max_context : g.yarn_orig_ctx;
        return {};
    }

    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override {
        EngineOptions opts;
        opts.pack = req.paths.empty() ? std::string() : req.paths[0];
        opts.max_ctx = (int) desc_.context_length;
        if (req.vram_reserve_mib > 0) opts.vram_reserve = (size_t) req.vram_reserve_mib << 20;
        opts.profile = req.expert_profile;
        opts.kv = req.kv.empty() ? "fp16" : req.kv;
        opts.spec = req.spec;
        std::string err;
        if (!eng_.load(opts, err)) return {model::Status::Error, err};
        sessions.clear();
        for (int i = 0; i < std::max(1, req.sessions); ++i) sessions.push_back(std::make_unique<Session>(eng_, opts.spec));
        return {};
    }

private:
    model::ModelDesc desc_;
    Engine eng_;
};

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return kArch; }

    std::string validate(const GgufFile&) const override {
        return {};
    }

    model::ModelDesc describe(const GgufFile&) const override {
        Geometry g;
        model::ModelDesc d;
        d.arch = kArch;
        d.n_layers = g.n_layers;
        d.n_embd = g.n_embd;
        d.n_vocab = g.n_vocab;
        d.n_expert = g.n_expert;
        d.n_expert_used = g.n_expert_used;
        d.n_ff_exp = g.n_ff_exp;
        d.n_expert_shared = g.n_expert_shared;
        d.context_length = g.yarn_orig_ctx;
        return d;
    }

    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<GigaChat4Model>(); }
    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "gigachat4"}; }
    PreTokenizer pretokenizer() const override { return {"gigachat4", program::tokenizer::pretok_o200k, true}; }

    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<\xef\xbd\x9c" "GCML\xef\xbd\x9c" "tool_calls>\n";
        f.call_close = "\n</\xef\xbd\x9c" "GCML\xef\xbd\x9c" "tool_calls>";
        f.fn_open = "<\xef\xbd\x9c" "GCML\xef\xbd\x9c" "invoke name=\"";
        f.fn_mid = "\">\n";
        f.fn_close = "</\xef\xbd\x9c" "GCML\xef\xbd\x9c" "invoke>";
        f.param_open = "<\xef\xbd\x9c" "GCML\xef\xbd\x9c" "parameter name=\"";
        f.param_mid_text = "\" string=\"true\">";
        f.param_mid_value = "\" string=\"false\">";
        f.param_close = "</\xef\xbd\x9c" "GCML\xef\xbd\x9c" "parameter>\n";
        f.text_end = "</\xef\xbd\x9c" "GCML\xef\xbd\x9c" "parameter>";
        f.calls_share_block = true;
        f.trigger = "<\xef\xbd\x9c" "GCML\xef\xbd\x9c" "tool_calls>";
        return f;
    }

    program::GpuModes gpu_modes() const override { return {1, 0, 0}; }
    const char* model_name() const override { return "GigaChat4-37B-A4B"; }
    const char* community_quant() const override { return "NVFP4"; }
    model::TokenNames token_names() const override { return {{"<|message_sep|>\n\n", "</s>"}, "<|role_sep|>\n", "", "", ""}; }

    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int&) const override {
        if (a == "--kv") {
            o.kv = args.next("--kv");
            return true;
        }
        if (a == "--spec") {
            o.spec = std::atoi(args.next("--spec"));
            o.spec_given = true;
            return true;
        }
        if (a == "--spec-oracle") {
            o.spec_oracle = args.next("--spec-oracle");
            return true;
        }
        if (a == "--diag-teacher") {
            o.diag_teacher = true;
            return true;
        }
        if (a == "--draft-log") {
            core::diagnostics().draft_log = args.next("--draft-log");
            return true;
        }
        return false;
    }

    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (o.kv.empty()) o.kv = "fp16";
        if (o.kv != "fp16" && o.kv != "fp8") {
            basalt::log::error("basalt gigachat4", "--kv %s: the latent caches are fp16 (the default) or fp8", o.kv.c_str());
            return 2;
        }
        const bool mtp = pack_has_mtp(o.model);
        if (!o.spec_given) o.spec = mtp ? kMaxWindow : 0;
        if (o.spec > kMaxWindow) {
            basalt::log::warn("basalt gigachat4", "--spec %d capped at %d", o.spec, kMaxWindow);
            o.spec = kMaxWindow;
        }
        if (o.spec >= 2 && o.spec_oracle.empty() && !mtp) {
            basalt::log::warn("basalt gigachat4", "--spec %d off: the pack has no mtp drafter", o.spec);
            o.spec = 0;
        }
        return std::nullopt;
    }

    int planned_spec(const program::CommonOptions& o) const override {
        if (o.spec_given) return o.spec >= 2 ? std::min(o.spec, kMaxWindow) : 0;
        return pack_has_mtp(o.model) ? kMaxWindow : 0;
    }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::gigachat4
