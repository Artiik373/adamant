// src/models/hy4/engine.hpp - a loaded Hy4-flash pack and its decode of windows of up to kMaxT tokens.
//
// The trunk lives on the GPU: its matrices quantized at load (Q8_0 or Q6_K), kv_b as fp16 for the absorbed MLA, the
// router, norms, iHC and small indexer weights as fp32, the embedding as BF16. The routed experts are one pinned host
// range (host VMM), read by the GPU in place. The Engram tables stay in the source shards and their rows are read per
// token. KV caches: a ring of kWindow + kMaxT positions on the sliding layers, the whole context on the sparse ones.
#pragma once

#include "basalt/models/hy4/geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::hy4 {

struct EngineOptions {
    int max_context = 8192;
    int trunk_type = 8;                        ///< ggml type of the trunk matrices: 8 Q8_0, 14 Q6_K
    std::string dump_dir;                      ///< diagnostic: the streams after each module, per position
    int vram_reserve_mib = 1536;               ///< VRAM left free beside the expert cache
    std::string expert_profile;                ///< routing counts: the first fill, saved again at exit
    bool profile = false;                      ///< diagnostic: GPU time by phase, logged at exit
    bool no_graphs = false;                    ///< run every launch eagerly (diagnostics turn graphs off too)
    bool dfly = true;                          ///< load the pack's DFly2 drafter, when it has one
    std::string feed_dir;                      ///< diagnostic: each module starts from these dumped streams instead
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool load(const std::string& pack, const EngineOptions& o, std::string& err);
    /// Runs tokens[0..n) (n <= kMaxP) at positions pos .. pos + n - 1. `pos` may be below the positions run so far:
    /// the later ones are dropped (a rejected draft). picks[t]: the greedy next token after tokens[t]; `logits`
    /// ([n][kVocab] floats, host) optional. A window wider than kMaxT (a prompt's) gives its last row only: the
    /// other picks are -1 and their logits rows are left as they were.
    bool step_window(const int32_t* tokens, int n, int64_t pos, int32_t* picks, float* logits, std::string& err);
    /// step_window of one token; returns the greedy next token, -1 on failure.
    int32_t step(int32_t token, int64_t pos, float* logits, std::string& err);
    /// DFly2's drafts after `anchor`, the next token, at `pos`, right after the window whose row for pos - 1 holds the
    /// last accepted token. Writes kDflyBlock - 1 tokens to `out` and returns their count; 0 without a drafter or room
    /// in the context, -1 on failure. `conf` (optional, kDflyBlock - 1 floats): each draft's accept chance, given the
    /// ones before it, from the drafter's confidence head.
    int draft(int32_t anchor, int64_t pos, int32_t* out, float* conf, std::string& err);

    struct Impl;

private:
    Impl* m_;
};

}  // namespace basalt::models::hy4
