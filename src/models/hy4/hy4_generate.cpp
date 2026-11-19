// src/models/hy4/hy4_generate.cpp - greedy decode of a Hy4-flash pack (a bring-up tool).
//
//     hy4_generate --model PACK.basalt --tokens 120000,1,2,3 [--max-new 16] [--ctx 8192] [--trunk q8_0|q6_k]
//                  [--no-dfly] [--dfly-min P] [--suffix-draft N] [--prompt-window N] [--dump DIR]
//
// The prompt runs in windows of --prompt-window tokens (kMaxT; up to kMaxP, whose windows give only their last row's
// pick); the greedy pick after each token is printed beside the next prompt token (the teacher-forced top-1), then
// --max-new tokens are generated. Each decode window is the last token plus drafts: the pack's DFly2 drafter's, kept while the product of their accept chances (its confidence head) stays at P or
// above, or with --no-dfly (or no drafter in the pack) the suffix drafter's (a match of at least N tokens; 0: one token
// per window); the picks accept the drafts up to the first that differs. --dump writes the residual streams after each
// module, per position, named as hy4_reference.py names its module outputs, to compare the two.
#include "models/hy4/engine.hpp"

#include "basalt/log.hpp"
#include "basalt/spec/suffix_drafter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    namespace hy4 = basalt::models::hy4;
    std::string model, tokens, trunk = "q8_0";
    hy4::EngineOptions o;
    int max_new = 16, suffix_draft = 3, prompt_rows = hy4::kMaxT;
    float dfly_min = 0.4f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens = next();
        else if (a == "--max-new") max_new = std::atoi(next().c_str());
        else if (a == "--ctx") o.max_context = std::atoi(next().c_str());
        else if (a == "--trunk") trunk = next();
        else if (a == "--no-dfly") o.dfly = false;
        else if (a == "--dfly-min") dfly_min = (float) std::atof(next().c_str());
        else if (a == "--suffix-draft") suffix_draft = std::max(0, std::atoi(next().c_str()));
        else if (a == "--prompt-window") prompt_rows = std::clamp(std::atoi(next().c_str()), 1, hy4::kMaxP);
        else if (a == "--dump") o.dump_dir = next();
        else if (a == "--feed") o.feed_dir = next();
        else if (a == "--profile") o.profile = true;
        else if (a == "--no-graphs") o.no_graphs = true;
        else if (a == "--expert-profile") o.expert_profile = next();
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoi(next().c_str());
        else {
            std::fprintf(stderr, "hy4_generate: unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty() || tokens.empty()) {
        std::fprintf(stderr, "usage: hy4_generate --model PACK --tokens ids [--max-new N] [--ctx N] [--trunk q8_0|q6_k] "
                             "[--no-dfly] [--dfly-min P] [--suffix-draft N] [--dump DIR]\n");
        return 2;
    }
    o.trunk_type = trunk == "q6_k" ? 14 : 8;
    std::vector<int32_t> ids;
    std::stringstream ss(tokens);
    for (std::string t; std::getline(ss, t, ',');) ids.push_back(std::atoi(t.c_str()));
    if (ids.empty()) {
        std::fprintf(stderr, "hy4_generate: no prompt tokens\n");
        return 2;
    }
    // the diagnostics compare one position per step
    const int prompt_window = o.dump_dir.empty() && o.feed_dir.empty() ? prompt_rows : 1;
    hy4::Engine eng;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!eng.load(model, o, err)) {
        basalt::log::error("hy4_generate", "%s", err.c_str());
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("loaded in %.1f s\n", load_s);
    int hits = 0;
    int32_t picks[hy4::kMaxP];
    int32_t next = -1;
    int64_t pos = 0;
    const auto p0 = std::chrono::steady_clock::now();
    while (pos < (int64_t) ids.size()) {
        const int n = (int) std::min<int64_t>(prompt_window, (int64_t) ids.size() - pos);
        if (!eng.step_window(ids.data() + pos, n, pos, picks, nullptr, err)) {
            basalt::log::error("hy4_generate", "%s", err.c_str());
            return 1;
        }
        for (int t = 0; t < n; ++t, ++pos)
            if ((size_t) pos + 1 < ids.size()) {
                hits += picks[t] == ids[(size_t) pos + 1];
                std::printf("pos %4lld  next %6d  pick %6d %s\n", (long long) pos, ids[(size_t) pos + 1], picks[t],
                            picks[t] == ids[(size_t) pos + 1] ? "HIT" : "");
            }
        next = picks[n - 1];
    }
    const double prompt_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
    std::printf("prompt %zu tokens in %.2f s -> %.1f tok/s\n", ids.size(), prompt_s, (double) ids.size() / prompt_s);
    std::printf("teacher-forced top-1: %d of %zu\n", hits, ids.size() - 1);
    std::printf("output:");
    basalt::spec::SuffixDrafter sfx(std::max(1, suffix_draft), 64, (size_t) o.max_context + 4096);
    if (suffix_draft > 0) sfx.append(ids.data(), ids.size());
    int32_t window[hy4::kMaxT];
    int produced = 0;
    long long windows = 0, drafted = 0, accepted = 0, by_accepted[hy4::kMaxT] = {};
    double draft_ms = 0;
    const auto d0 = std::chrono::steady_clock::now();
    while (produced < max_new) {
        // `next` is produced but not yet run: it heads the window
        std::printf(" %d", next);
        std::fflush(stdout);
        ++produced;
        if (suffix_draft > 0) sfx.append(next);
        if (produced >= max_new) break;
        window[0] = next;
        const int room = (int) std::min<int64_t>(
            {(int64_t) hy4::kMaxT - 1, (int64_t) max_new - produced, (int64_t) o.max_context - pos - 1});
        const auto q0 = std::chrono::steady_clock::now();
        float conf[hy4::kMaxT];
        int k = eng.draft(next, pos, window + 1, conf, err);
        float cum = 1.f;   // the chance that every draft kept so far is accepted
        for (int j = 0; j < k; ++j)
            if ((cum *= conf[j]) < dfly_min) {
                k = j;
                break;
            }
        draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - q0).count();
        if (k < 0) {
            basalt::log::error("hy4_generate", "%s", err.c_str());
            return 1;
        }
        if (k == 0 && suffix_draft > 0) k = sfx.propose(room, window + 1);
        const int n = 1 + std::min(k, room);
        if (!eng.step_window(window, n, pos, picks, nullptr, err)) {
            basalt::log::error("hy4_generate", "%s", err.c_str());
            return 1;
        }
        int a = 0;
        while (a < n - 1 && window[a + 1] == picks[a]) ++a;
        ++windows;
        drafted += n - 1;
        accepted += a;
        ++by_accepted[a];
        for (int i = 1; i <= a; ++i) {   // the accepted drafts are produced and already run
            std::printf(" %d", window[i]);
            ++produced;
            if (suffix_draft > 0) sfx.append(window[i]);
        }
        pos += a + 1;
        next = picks[a];
    }
    const double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
    std::printf("\ndecode %d tokens in %.2f s -> %.2f tok/s\n", produced, dec_s, produced > 0 ? produced / dec_s : 0.0);
    if (windows > 0) {
        std::printf("windows %lld, drafts %lld, accepted %lld (%.0f%%), %.2f tokens per window; drafting %.1f ms\n",
                    windows, drafted, accepted, drafted > 0 ? 100.0 * accepted / drafted : 0.0,
                    (double) (windows + accepted) / windows, draft_ms);
        std::printf("windows by accepted drafts:");
        for (int i = 0; i < hy4::kMaxT; ++i) std::printf(" %d:%lld", i, by_accepted[i]);
        std::printf("\n");
    }
    return 0;
}
