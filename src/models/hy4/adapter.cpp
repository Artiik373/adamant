// src/models/hy4/adapter.cpp - the Hy4-flash adapter and its serving: one request at a time over the --serve line
// protocol's solo subset (GEN, STOP, VRAM, QUIT). A request reuses the prompt prefix the engine already holds; the
// prompt runs in windows of kMaxP, decode windows carry the DFly2 drafts while the product of their accept chances
// stays at kDflyMin or above, and each row's pick (greedy, or sampled on the host) accepts drafts up to the first
// that differs or falls below the relaxed-drafting threshold.
#include "models/hy4/adapter.hpp"
#include "models/hy4/engine.hpp"

#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"
#include "basalt/protocol.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace basalt::models::hy4 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float kDflyMin = 0.4f;
constexpr const char* kTag = "basalt serve";

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

struct Hy4Options : public program::CommonOptions {
    int trunk_type = 8;
    bool dfly = true;
    float relaxed_drafting = 0.0f;
};

/// stdin lines, read on their own thread so STOP reaches a running request.
class Input {
public:
    Input() {
        std::thread([this] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                push(line);
            }
            push("QUIT");
        }).detach();
    }
    std::string next() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return !q_.empty(); });
        std::string s = std::move(q_.front());
        q_.pop_front();
        return s;
    }
    /// Takes a waiting STOP; other lines stay queued.
    bool stopped() {
        std::lock_guard<std::mutex> lk(m_);
        const auto it = std::find(q_.begin(), q_.end(), std::string("STOP"));
        if (it == q_.end()) return false;
        q_.erase(it);
        return true;
    }

private:
    void push(const std::string& s) {
        { std::lock_guard<std::mutex> lk(m_); q_.push_back(s); }
        cv_.notify_one();
    }
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
};

struct Sampling {
    float temperature = 0, top_p = 1, min_p = 0;
    int top_k = 0;
    bool seeded = false;
    uint64_t seed = 0;
    float relaxed_drafting = 0.0f;
};

struct SampleResult {
    int32_t pick = -1;
    float draft_prob = 0.0f;
};

/// One draw from a row of logits after temperature, top-k, min-p and top-p.
/// When draft_id >= 0, returns the normalized probability of draft_id in draft_prob.
SampleResult sample_row(const float* logits, const Sampling& s, std::mt19937_64& rng, int32_t draft_id = -1) {
    const float mx = *std::max_element(logits, logits + kVocab);
    // tokens more than 30 nats below the best carry no probability a float draw can reach
    const float floor = mx - 30.f * s.temperature;
    std::vector<std::pair<float, int32_t>> c;
    for (int32_t i = 0; i < kVocab; ++i)
        if (logits[i] >= floor) c.push_back({(logits[i] - mx) / s.temperature, i});
    auto desc = [](const auto& a, const auto& b) { return a.first > b.first; };
    if (s.top_k > 0 && (size_t) s.top_k < c.size()) {
        std::nth_element(c.begin(), c.begin() + s.top_k, c.end(), desc);
        c.resize((size_t) s.top_k);
    }
    // the order matters only to the min-p and top-p cuts
    if (s.top_p < 1.f || s.min_p > 0.f) std::sort(c.begin(), c.end(), desc);
    double sum = 0;
    for (auto& e : c) sum += (e.first = std::exp(e.first));
    size_t keep = c.size();
    double acc = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        if (c[i].first < s.min_p * c[0].first) { keep = i; break; }
        acc += c[i].first;
        if (acc >= s.top_p * sum) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(keep, 1);
    double total = 0;
    float draft_p = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        total += c[i].first;
        if (draft_id >= 0 && c[i].second == draft_id) draft_p = (float) c[i].first;
    }
    if (total > 0.0) draft_p = (float) (draft_p / total);
    double u = std::uniform_real_distribution<double>(0.0, total)(rng);
    int32_t pick = c[keep - 1].second;
    for (size_t i = 0; i < keep; ++i) {
        if ((u -= c[i].first) <= 0) { pick = c[i].second; break; }
    }
    return {pick, draft_p};
}

/// The drafts a decode window carries: none unless --spec N (N >= 2) asks for them. Each draft row routes its own
/// experts, so on a card that holds a small share of them plain decode is faster.
int drafts(const Hy4Options& o) {
    return o.dfly && o.spec_given && o.spec >= 2 ? std::min(o.spec - 1, (int) kDflyBlock - 1) : 0;
}

class Server {
public:
    Server(Engine& eng, const Hy4Options& o) : eng_(eng), o_(o) {}

    int run() {
        std::string eos;
        for (int64_t id : o_.eos_ids) eos += (eos.empty() ? "" : ",") + std::to_string(id);
        basalt::protocol::emit("INFO eos_ids=%s kv=%s\n", eos.c_str(), o_.kv.empty() ? "fp16" : o_.kv.c_str());
        basalt::protocol::emit("READY %lld stop\n", (long long) o_.max_context);
        for (;;) {
            const std::string line = in_.next();
            if (line == "QUIT") return 0;
            if (line.empty() || line == "STOP") continue;
            if (line.rfind("GEN ", 0) == 0) {
                if (!request(line)) return 1;
            } else if (line.rfind("VRAM", 0) == 0) {
                size_t free_b = 0, total_b = 0;
                cudaMemGetInfo(&free_b, &total_b);
                basalt::protocol::emit("VRAM vram_free_mib=%lld vram_total_mib=%lld\n", (long long) (free_b >> 20),
                                       (long long) (total_b >> 20));
            } else {
                const std::string word = line.substr(0, line.find(' '));
                basalt::protocol::emit("ERR bad request: this engine has no %s command\n", word.c_str());
            }
        }
    }

private:
    /// One GEN request; false on an engine failure, which ends serving.
    bool request(const std::string& line) {
        std::istringstream ws(line);
        std::vector<std::string> w;
        for (std::string x; ws >> x;) w.push_back(x);
        if (w.size() < 3) {
            basalt::protocol::emit("ERR bad request: GEN <max_new> [k=v ...] <ids>\n");
            return true;
        }
        int64_t max_new = std::atoll(w[1].c_str());
        Sampling s;
        s.relaxed_drafting = o_.relaxed_drafting;
        for (size_t i = 2; i + 1 < w.size(); ++i) {
            const size_t eq = w[i].find('=');
            if (eq == std::string::npos) continue;
            const std::string k = w[i].substr(0, eq), v = w[i].substr(eq + 1);
            if (k == "temperature") s.temperature = (float) std::atof(v.c_str());
            else if (k == "top_p") s.top_p = (float) std::atof(v.c_str());
            else if (k == "top_k") s.top_k = std::atoi(v.c_str());
            else if (k == "min_p") s.min_p = (float) std::atof(v.c_str());
            else if (k == "seed") { s.seeded = true; s.seed = std::strtoull(v.c_str(), nullptr, 10); }
            else if (k == "relax" || k == "relaxed_drafting") s.relaxed_drafting = (float) std::atof(v.c_str());
            // penalties, logit bias and grammar are not carried by this engine: they are ignored
        }
        std::vector<int32_t> ids;
        std::stringstream is(w.back());
        for (std::string t; std::getline(is, t, ',');) ids.push_back(std::atoi(t.c_str()));
        const int64_t n = (int64_t) ids.size();
        for (int32_t t : ids)
            if (t < 0 || t >= kVocab) {
                basalt::protocol::emit("ERR a token id is outside the vocabulary (%d)\n", (int) t);
                return true;
            }
        if (n == 0 || n >= o_.max_context) {
            basalt::protocol::emit("ERR prompt (%lld tokens) leaves no room to answer in the context (%lld)\n",
                                   (long long) n, (long long) o_.max_context);
            return true;
        }
        max_new = std::min<int64_t>(max_new, o_.max_context - n);
        const bool greedy = s.temperature <= 0;
        std::mt19937_64 rng(s.seeded ? s.seed : std::random_device{}());
        if (!greedy) logits_.resize((size_t) kMaxP * kVocab);
        std::string err;
        int32_t picks[kMaxP];

        // the prompt: the held prefix stays, at least the last token runs again
        int64_t reuse = 0;
        while (reuse < (int64_t) held_.size() && reuse < n - 1 && held_[(size_t) reuse] == ids[(size_t) reuse]) ++reuse;
        held_.resize((size_t) reuse);
        if (reuse > 0) basalt::protocol::emit("RESUME %lld\n", (long long) reuse);
        const auto t0 = Clock::now();
        int64_t pos = reuse;
        int32_t next = -1;
        bool stopped = false;
        while (pos < n) {
            const int rows = (int) std::min<int64_t>(kMaxP, n - pos);
            const bool last = pos + rows == n;
            if (!eng_.step_window(ids.data() + pos, rows, pos, picks, last && !greedy ? logits_.data() : nullptr, err))
                return fail(err);
            held_.insert(held_.end(), ids.begin() + pos, ids.begin() + pos + rows);
            pos += rows;
            const double ms = ms_since(t0);
            basalt::protocol::emit("PP %lld %lld %.1f %.1f\n", (long long) pos, (long long) n, ms,
                                   ms > 0 ? 1000.0 * (double) (pos - reuse) / ms : 0.0);
            if (last)
                next = greedy ? picks[rows - 1] : sample_row(logits_.data() + (size_t) (rows - 1) * kVocab, s, rng).pick;
            if (in_.stopped()) { stopped = true; break; }
        }
        const double prompt_ms = ms_since(t0);

        // decode: `next` is produced but not yet run; it heads each window
        const auto d0 = Clock::now();
        int64_t produced = 0, offered = 0, accepted = 0, relaxed_kept = 0;
        const char* finish = "length";
        int32_t window[kMaxT];
        float conf[kMaxT];
        auto is_eos = [&](int32_t t) {
            return std::find(o_.eos_ids.begin(), o_.eos_ids.end(), (int64_t) t) != o_.eos_ids.end();
        };
        auto produce = [&](int32_t t) {
            basalt::protocol::emit("T %d\n", (int) t);
            ++produced;
            if (is_eos(t)) finish = "stop";
            return is_eos(t) || produced >= max_new;
        };
        if (stopped) finish = "cancel";
        else if (max_new > 0) {
            const int max_draft = drafts(o_);
            for (;;) {
                if (produce(next)) break;
                if (in_.stopped()) { finish = "cancel"; break; }
                window[0] = next;
                const int room = (int) std::min<int64_t>(
                    {(int64_t) kMaxT - 1, max_new - produced, o_.max_context - pos - 1});
                int k = (room > 0 && max_draft > 0) ? eng_.draft(next, pos, window + 1, conf, err) : 0;
                if (k < 0) return fail(err);
                if (k > max_draft) k = max_draft;
                float all = 1.f;   // the chance that every draft kept so far is accepted
                for (int j = 0; j < k; ++j)
                    if ((all *= conf[j]) < kDflyMin) { k = j; break; }
                const int rows = 1 + std::min(k, room);
                if (!eng_.step_window(window, rows, pos, picks, greedy ? nullptr : logits_.data(), err))
                    return fail(err);
                int a = 0;
                if (greedy) {
                    while (a < rows - 1 && window[a + 1] == picks[a]) ++a;
                } else {
                    while (a < rows - 1) {
                        const int32_t d = window[a + 1];
                        const SampleResult sr = sample_row(logits_.data() + (size_t) a * kVocab, s, rng, d);
                        if (d == sr.pick) {
                            picks[a] = d;
                            ++a;
                        } else if (s.relaxed_drafting > 0.0f && sr.draft_prob >= s.relaxed_drafting) {
                            picks[a] = d;
                            ++relaxed_kept;
                            ++a;
                        } else {
                            picks[a] = sr.pick;
                            break;
                        }
                    }
                    if (a == rows - 1) {
                        picks[a] = sample_row(logits_.data() + (size_t) a * kVocab, s, rng).pick;
                    }
                }
                offered += rows - 1;
                accepted += a;
                held_.insert(held_.end(), window, window + a + 1);
                pos += a + 1;
                bool end = false;
                for (int i = 1; i <= a && !end; ++i) end = produce(window[i]);
                if (end) break;
                next = picks[a];
            }
        }
        const double decode_ms = ms_since(d0);
        if (relaxed_kept > 0) {
            basalt::log::info(kTag, "prompt %lld tokens = %lld reused + %lld read in %.0f ms (%.1f tok/s), "
                              "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld (relaxed kept %lld)",
                              (long long) n, (long long) reuse, (long long) (n - reuse), prompt_ms,
                              prompt_ms > 0 ? 1000.0 * (double) (n - reuse) / prompt_ms : 0.0, (long long) produced,
                              decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0,
                              (long long) accepted, (long long) offered, (long long) relaxed_kept);
        } else {
            basalt::log::info(kTag, "prompt %lld tokens = %lld reused + %lld read in %.0f ms (%.1f tok/s), "
                              "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld",
                              (long long) n, (long long) reuse, (long long) (n - reuse), prompt_ms,
                              prompt_ms > 0 ? 1000.0 * (double) (n - reuse) / prompt_ms : 0.0, (long long) produced,
                              decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0,
                              (long long) accepted, (long long) offered);
        }
        basalt::protocol::emit("DONE %lld %lld %.1f %.1f %s %lld %lld %lld\n", (long long) produced, (long long) n,
                               prompt_ms, decode_ms, finish, (long long) accepted, (long long) offered,
                               (long long) reuse);
        return true;
    }

    bool fail(const std::string& err) {
        basalt::log::error(kTag, "%s", err.c_str());
        basalt::protocol::emit("ERR %s\n", err.c_str());
        return false;
    }

    Engine& eng_;
    const Hy4Options& o_;
    Input in_;
    std::vector<int32_t> held_;   // the tokens whose positions the engine holds
    std::vector<float> logits_;
};

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "hy4"; }
    std::string validate(const GgufFile&) const override { return {}; }
    model::ModelDesc describe(const GgufFile&) const override {
        model::ModelDesc d;
        d.arch = "hy4";
        d.n_vocab = kVocab;
        return d;
    }
    std::unique_ptr<program::CommonOptions> make_options() const override {
        return std::make_unique<Hy4Options>();
    }
    std::optional<int> run(program::CommonOptions& o, std::vector<program::GpuCard>&) const override {
        if (!o.serve) {
            basalt::log::error("basalt", "Hy4-flash packs run only with --serve here (hy4_generate decodes ids)");
            return 2;
        }
        const auto& ho = static_cast<const Hy4Options&>(o);
        EngineOptions eo;
        eo.max_context = (int) ho.max_context;
        eo.vram_reserve_mib = ho.vram_reserve_mib;
        eo.trunk_type = ho.trunk_type;
        eo.dfly = drafts(ho) > 0;   // the drafter's VRAM goes to the expert cache when nothing drafts
        // its own routing counts beside the pack: a shared --expert-profile names another model's file
        eo.expert_profile = ho.model + ".prof";
        Engine eng;
        std::string err;
        const auto t0 = Clock::now();
        if (!eng.load(ho.model, eo, err)) {
            basalt::log::error("basalt", "%s", err.c_str());
            return 1;
        }
        basalt::log::info(kTag, "Hy4-flash loaded in %.1f s, context %lld, one request at a time",
                          ms_since(t0) / 1000.0, (long long) ho.max_context);
        return Server(eng, ho).run();
    }
    int max_expert_cards() const override { return 0; }
    bool parse_flag(const std::string& name, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        auto& ho = static_cast<Hy4Options&>(o);
        if (name == "--kv") {
            const std::string v = args.next("--kv");
            if (v != "fp16" && v != "int8" && v != "q4_0" && v != "k8v4" && v != "fp8") {
                basalt::log::error("basalt hy4", "--kv %s: expected fp16, int8, q4_0, k8v4 or fp8", v.c_str());
                code = 2;
                return true;
            }
            if (v != "fp16") {
                basalt::log::warn("basalt hy4", "--kv %s: this model's caches stay fp16", v.c_str());
            }
            ho.kv = v;
            return true;
        } else if (name == "--trunk") {
            const std::string v = args.next("--trunk");
            if (v == "q8_0" || v == "q8") {
                ho.trunk_type = 8;
            } else if (v == "q6_k" || v == "q6" || v == "q6_0") {
                ho.trunk_type = 14;
            } else {
                basalt::log::error("basalt hy4", "--trunk %s: expected q8_0 or q6_k", v.c_str());
                code = 2;
                return true;
            }
            return true;
        } else if (name == "--no-dfly") {
            ho.dfly = false;
            return true;
        } else if (name == "--spec") {
            ho.spec = std::atoi(args.next("--spec"));
            ho.spec_given = true;
            return true;
        } else if (name == "--relaxed-drafting") {
            ho.relaxed_drafting = 0.5f;
            if (args.peek() != nullptr) {
                char* e = nullptr;
                const float v = std::strtof(args.peek(), &e);
                if (e != args.peek() && *e == '\0') {
                    args.take();
                    if (!(v > 0.0f && v <= 1.0f)) {
                        basalt::log::error("basalt hy4", "--relaxed-drafting: P must be in (0, 1]");
                        code = 2;
                        return true;
                    }
                    ho.relaxed_drafting = v;
                }
            }
            return true;
        }
        return false;
    }
    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        auto& ho = static_cast<Hy4Options&>(o);
        if (ho.kv.empty()) ho.kv = "fp16";
        if (ho.relaxed_drafting > 0.0f) {
            basalt::log::warn("basalt hy4", "--relaxed-drafting %g accepts draft tokens the model would not pick",
                              (double) ho.relaxed_drafting);
        }
        return std::nullopt;
    }
    int planned_spec(const program::CommonOptions& o) const override {
        const auto& ho = static_cast<const Hy4Options&>(o);
        const int d = drafts(ho);
        return d > 0 ? d + 1 : 0;
    }
    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.json", "hy4"}; }
    model::TokenNames token_names() const override { return {{"<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"}, {}, {}}; }
    PreTokenizer pretokenizer() const override { return {}; }
    program::grammar::ToolForm tool_form() const override { return {}; }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::hy4
