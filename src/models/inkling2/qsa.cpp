// src/models/inkling2/qsa.cpp - reads the QSA indexer file (qsa.hpp) onto the cards.
#include "models/inkling2/qsa.hpp"

#include "basalt/core/device_copy.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/models/inkling2/qsa_kernels.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <memory>

namespace basalt::models::inkling2 {

bool qsa_load(const std::string& path, const Geometry& g, const std::function<void*(int, size_t)>& alloc, QsaIndexer& out,
              std::string& err) {
    const std::unique_ptr<std::FILE, int (*)(std::FILE*)> f(platform::open_file_utf8(path, "rb"), std::fclose);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    int32_t h[7] = {};
    float theta = 0.f;
    if (std::fread(h, 4, 7, f.get()) != 7 || std::fread(&theta, 4, 1, f.get()) != 1) {
        err = path + ": short header";
        return false;
    }
    if ((uint32_t) h[0] != 0x51534157u || h[1] != 1) {
        err = path + ": not a QSA indexer file (version 1)";
        return false;
    }
    if (h[3] != kWidth || h[4] != kQsaHeads || h[5] != kQsaDim || h[6] != kQsaBlock || theta > 0.f) {
        err = path + ": width " + std::to_string(h[3]) + ", heads " + std::to_string(h[4]) + "x" + std::to_string(h[5]) + ", block " +
              std::to_string(h[6]) + ", theta " + std::to_string(theta) + "; this engine takes 2304, 4x128, 4 and no rotary";
        return false;
    }
    const int n = h[2];
    std::vector<int32_t> vl((size_t) std::max(n, 0));
    if (n <= 0 || std::fread(vl.data(), 4, (size_t) n, f.get()) != (size_t) n) {
        err = path + ": short visit list";
        return false;
    }
    out.index.assign(g.visits.size(), -1);
    out.visits.assign((size_t) n, QsaVisit{});
    const size_t nq = (size_t) kQsaHeads * kQsaDim * kWidth, nk = (size_t) kQsaDim * kWidth;
    std::vector<uint8_t> buf;
    for (int i = 0; i < n; ++i) {
        const int v = vl[(size_t) i];
        if (v < 0 || v >= (int) g.visits.size() || g.swa[(size_t) g.visits[(size_t) v].layer] || out.index[(size_t) v] >= 0) {
            err = path + ": visit " + std::to_string(v) + " is not a full-attention visit of this pack";
            return false;
        }
        out.index[(size_t) v] = i;
        QsaVisit& q = out.visits[(size_t) i];
        const size_t sizes[5] = {nq * 2, nk * 2, kQsaDim * 4, kQsaDim * 4, kQsaHeads * 4};
        void** dst[5] = {(void**) &q.wq, (void**) &q.wk, (void**) &q.q_norm, (void**) &q.k_norm, (void**) &q.w_head};
        for (int t = 0; t < 5; ++t) {
            buf.resize(sizes[t]);
            if (std::fread(buf.data(), 1, sizes[t], f.get()) != sizes[t]) {
                err = path + ": truncated at visit " + std::to_string(v);
                return false;
            }
            *dst[t] = alloc(v, sizes[t]);
            if (!*dst[t] || basalt::core::device_copy(*dst[t], buf.data(), sizes[t], cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "out of device memory for the QSA indexer";
                return false;
            }
        }
    }
    if (std::fgetc(f.get()) != EOF) {
        err = path + ": bytes past the last visit";
        return false;
    }
    return true;
}

}  // namespace basalt::models::inkling2
