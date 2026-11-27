// src/models/gigachat4/cli.cpp - gigachat4-run: the GigaChat4 runtime on a file of token ids (raw little-endian int32).
//   --model PACK --tokens FILE         the prompt
//   --ppl                              perplexity of the tokens (prefill rows of 8); --logits OUT writes the last 16 rows
//   --gen N                            greedy decoding of N tokens after the prompt, timed; --out FILE writes them
//   --profile FILE / --save-profile F  expert counts that choose the VRAM-cached experts / write this run's counts
//   --max-ctx N
#include "models/gigachat4/engine.hpp"

#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

using namespace basalt::models::gigachat4;

static int run(int argc, char** argv) {
    EngineOptions o;
    std::string tokens_path, logits_path, out_path, save_profile;
    bool ppl = false;
    int gen = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                basalt::log::error("gigachat4", "%s needs a value", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") o.pack = next();
        else if (a == "--tokens") tokens_path = next();
        else if (a == "--ppl") ppl = true;
        else if (a == "--logits") logits_path = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--out") out_path = next();
        else if (a == "--profile") o.profile = next();
        else if (a == "--save-profile") save_profile = next();
        else if (a == "--max-ctx") o.max_ctx = std::stoi(next());
        else {
            basalt::log::error("gigachat4", "unknown flag %s", a.c_str());
            return 2;
        }
    }
    std::vector<uint8_t> raw;
    std::string err;
    if (o.pack.empty() || tokens_path.empty() || !basalt::core::read_range(basalt::core::FileRange(tokens_path), raw, err)) {
        basalt::log::error("gigachat4", "need --model and --tokens: %s", err.c_str());
        return 2;
    }
    std::vector<int32_t> toks(raw.size() / 4);
    std::memcpy(toks.data(), raw.data(), toks.size() * 4);
    Engine eng;
    const auto t0 = std::chrono::steady_clock::now();
    if (!eng.load(o, err)) {
        basalt::log::error("gigachat4", "%s", err.c_str());
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    basalt::log::info("gigachat4", "loaded in %.1f s", load_s);
    const Geometry& g = eng.geometry();
    const int n = (int) toks.size();
    const int prompt = n;
    std::vector<float> lg((size_t) 8 * g.n_vocab), last;
    double nll = 0;
    int scored = 0;
    std::vector<int32_t> argmax;
    const auto p0 = std::chrono::steady_clock::now();
    const int chunk_sz = ppl ? 8 : eng.prefill_chunk_size();
    for (int at = 0; at < prompt; at += chunk_sz) {
        const int r = std::min(chunk_sz, prompt - at);
        const bool need_logits = ppl || (at + r == prompt);
        eng.forward(toks.data() + at, r, at, need_logits);
        if (!ppl) continue;
        cudaMemcpy(lg.data(), eng.logits(), (size_t) r * g.n_vocab * 4, cudaMemcpyDeviceToHost);
        for (int j = 0; j < r; ++j) {
            const float* row = lg.data() + (size_t) j * g.n_vocab;
            float m = row[0];
            int am = 0;
            for (int v = 1; v < g.n_vocab; ++v)
                if (row[v] > m) m = row[v], am = v;
            argmax.push_back(am);
            if (at + j + 1 < n) {
                double s = 0;
                for (int v = 0; v < g.n_vocab; ++v) s += std::exp((double) row[v] - m);
                nll += m + std::log(s) - row[toks[at + j + 1]];
                ++scored;
            }
            if (at + j >= n - 16) last.insert(last.end(), row, row + g.n_vocab);
        }
    }
    const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
    basalt::log::info("gigachat4", "prompt %d tokens in %.3f s (%.1f tok/s)", prompt, prefill_s, prompt / prefill_s);
    if (ppl && scored) {
        basalt::log::info("gigachat4", "ppl %.4f over %d tokens (mean nll %.5f)", std::exp(nll / scored), scored, nll / scored);
        std::string am;
        for (int i = 0; i < std::min<int>(40, (int) argmax.size()); ++i) am += std::to_string(argmax[i]) + " ";
        basalt::log::info("gigachat4", "argmax %s", am.c_str());
        if (!logits_path.empty()) {
            FILE* f = std::fopen(logits_path.c_str(), "wb");
            if (f) {
                std::fwrite(last.data(), 4, last.size(), f);
                std::fwrite(argmax.data(), 4, argmax.size(), f);
                std::fclose(f);
            }
        }
    }
    if (gen > 0) {
        // the prompt's last row already predicted the first token: take its argmax, then decode
        cudaMemcpy(lg.data(), eng.logits() + (size_t) (chunk_sz <= 8 ? (prompt - 1) % 8 : 0) * g.n_vocab, (size_t) g.n_vocab * 4, cudaMemcpyDeviceToHost);
        const int first = (int) (std::max_element(lg.begin(), lg.begin() + g.n_vocab) - lg.begin());
        std::vector<int32_t> out;
        eng.greedy(first, prompt, 8, out);   // warm-up: captures the graphs of the first buckets
        eng.reset();
        for (int at = 0; at < prompt; at += chunk_sz) {
            const int r = std::min(chunk_sz, prompt - at);
            eng.forward(toks.data() + at, r, at, false);
        }
        const auto g0 = std::chrono::steady_clock::now();
        eng.greedy(first, prompt, gen, out);
        const double gs = std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count();
        uint64_t host = 0, all = 0;
        eng.expert_stats(host, all);
        basalt::log::info("gigachat4", "decode %d tokens in %.3f s: %.1f tok/s; expert reads from host %.2f%%", gen, gs, gen / gs,
                          all ? 100.0 * host / all : 0.0);
        std::string s = std::to_string(first) + " ";
        for (int i = 0; i < std::min(gen, 48); ++i) s += std::to_string(out[i]) + " ";
        basalt::log::info("gigachat4", "tokens %s", s.c_str());
        if (!out_path.empty()) {
            FILE* f = std::fopen(out_path.c_str(), "wb");
            if (f) {
                std::fwrite(&first, 4, 1, f);
                std::fwrite(out.data(), 4, out.size(), f);
                std::fclose(f);
            }
        }
    }
    if (!save_profile.empty() && !eng.save_profile(save_profile, err)) basalt::log::error("gigachat4", "%s", err.c_str());
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        basalt::log::error("gigachat4", "%s", e.what());
        return 1;
    }
}
