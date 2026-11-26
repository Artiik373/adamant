// src/models/inkling2/qsa.hpp - the QSA indexer file (--qsa-indexer): a pretrained block-sparse key selector for the
// full-attention visits (qsa_kernels.hpp). Its format: a header of seven int32 (magic "QSAW", version 1, visits, width,
// heads, head width, block) and a float32 rotary theta, the visit indices (int32), then per visit wq fp16 [heads * 128]
// [width], wk fp16 [128][width], q_norm, k_norm fp32 [128] and w_head fp32 [heads].
#pragma once

#include "models/inkling2/pack.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

struct QsaVisit {
    uint16_t* wq = nullptr;
    uint16_t* wk = nullptr;
    float* q_norm = nullptr;
    float* k_norm = nullptr;
    float* w_head = nullptr;
};

struct QsaIndexer {
    int budget = 0, local = 0, sink = 0;   ///< tokens: the keys a row reads in all, its local window, the first ones
    std::vector<int> index;                ///< per visit: its entry in `visits`, -1 without one
    std::vector<QsaVisit> visits;
    bool on() const { return !visits.empty(); }
    const QsaVisit* of(int visit) const {
        return visit < (int) index.size() && index[(size_t) visit] >= 0 ? &visits[(size_t) index[(size_t) visit]] : nullptr;
    }
    int k_blocks() const { return budget / 4; }
};

/// Reads `path` into `out` for the pack's geometry; `alloc(visit, bytes)` gives device memory on that visit's card.
bool qsa_load(const std::string& path, const Geometry& g, const std::function<void*(int, size_t)>& alloc, QsaIndexer& out,
              std::string& err);

}  // namespace basalt::models::inkling2
