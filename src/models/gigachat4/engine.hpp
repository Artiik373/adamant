// src/models/gigachat4/engine.hpp - the GigaChat4 runtime: the pack's dense weights in VRAM, the routed experts in pinned host
// memory with as many as fit cached in VRAM (the rest read in place over the bus), the recurrent and attention state of one
// sequence, the EAGLE-3 style MTP drafter, and a forward of 1 to 8 rows.
#pragma once

#include "models/gigachat4/meta.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core {
struct ConversationKv;
struct RunningState;
}

namespace basalt::models::gigachat4 {

struct EngineOptions {
    std::string pack;
    int max_ctx = 32768;
    size_t vram_reserve = 512ull << 20;   ///< VRAM left free after the expert cache
    std::string profile;                  ///< expert counts ([31][192] uint32) that choose the cached experts; empty: none
    std::string kv = "fp16";              ///< "fp16" or "fp8"
    int spec = 0;                         ///< speculative verify window (0/1: off, 2..8: MTP drafter)
};

class Engine {
public:
    Engine();
    ~Engine();
    bool load(const EngineOptions& o, std::string& err);

    /// Rows tokens[0 .. n) at positions pos .. pos + n - 1 (n <= prefill_chunk_size(), pos + n <= max_ctx).
    void forward(const int32_t* tokens, int n, int pos, bool need_logits = true);
    /// Decode window of 1..8 rows with GDN rollback support and hidden-state tap for MTP drafting.
    bool decode_forward(const int32_t* tokens, int n, int pos, std::string& err);
    /// Commit the first `keep` rows of the last decode_forward window (rolls back GDN state when keep < n).
    void commit_kept(int keep);
    /// Enqueue `n` speculative drafts (1..7) after `anchor` at `pos`.
    bool draft_begin(int32_t anchor, int pos, int n, std::string& err);
    /// Wait for the drafts started by draft_begin; `need(conf, j, max_n)` may stop early.
    bool draft_end(int32_t* ids, float* conf, int max_n, int& n,
                   const std::function<bool(const float*, int, int)>& need, std::string& err);

    const float* logits() const;          ///< device
    /// Greedy decoding: feeds `first` at `pos`, then each argmax, for n steps; out gets the n argmax tokens.
    void greedy(int32_t first, int pos, int n, std::vector<int32_t>& out);
    void reset();                         ///< empties the sequence state
    /// Experts read from host memory so far, of all expert reads (counted on the device).
    void expert_stats(uint64_t& host_reads, uint64_t& reads) const;
    bool save_profile(const std::string& path, std::string& err) const;
    void sync() const;
    const Geometry& geometry() const;
    int cached_experts() const;
    bool kv_fp8() const;
    bool has_mtp() const;
    bool mtp_active() const;
    int max_ctx() const;
    int prefill_chunk_size() const;
    int64_t kv_bytes() const;
    size_t running_bytes() const;
    bool running_state(int64_t pos, bool save, core::RunningState& st);
    void on_running_restored(int64_t pos);
    size_t cells_bytes(int64_t positions) const;
    bool save_cells(std::vector<core::ConversationKv>& kv, int64_t positions, std::string& err) const;
    bool restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions, std::string& err);
    void* cuda_stream() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace basalt::models::gigachat4
