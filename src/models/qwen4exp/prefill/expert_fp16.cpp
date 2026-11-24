#include "experts.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/exl3_kernels.hpp"
#include "basalt/kernels/ms_kernels.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/models/qwen4exp/prefill/kernels.hpp"
#include "basalt/log.hpp"
#include <cstdlib>
#include <vector>

namespace basalt::prefill {
using namespace detail;

// BASALT_PREFILL_TIMING=1: GPU time of the fp16 path's dequant and GEMM stages, summed over a layer's experts and
// printed once per layer (the events are read back after the layer, which costs a stall: diagnostics only)
namespace {
struct Fp16Timing {
    bool on = std::getenv("BASALT_PREFILL_TIMING") != nullptr;
    std::vector<cudaEvent_t> ev;   // 3 per expert: before dequant, after dequant, after the products
    double dq_ms = 0, gemm_ms = 0, span_ms = 0;
    long long n = 0, layers = 0;
    cudaEvent_t take() {
        cudaEvent_t e;
        cudaEventCreate(&e);
        ev.push_back(e);
        return e;
    }
};
Fp16Timing& timing() { static Fp16Timing t; return t; }
}  // namespace

void fp16_timing_flush(int layer) {
    Fp16Timing& t = timing();
    if (!t.on || t.ev.empty()) return;
    cudaEventSynchronize(t.ev.back());
    for (size_t i = 0; i + 2 < t.ev.size(); i += 3) {
        float a = 0, b = 0;
        cudaEventElapsedTime(&a, t.ev[i], t.ev[i + 1]);
        cudaEventElapsedTime(&b, t.ev[i + 1], t.ev[i + 2]);
        t.dq_ms += a;
        t.gemm_ms += b;
        ++t.n;
    }
    float span = 0;
    cudaEventElapsedTime(&span, t.ev.front(), t.ev.back());
    t.span_ms += span;
    for (cudaEvent_t e : t.ev) cudaEventDestroy(e);
    t.ev.clear();
    if (++t.layers % 48 == 0)
        basalt::log::info("basalt prefill", "fp16 experts, %d layers: %lld experts, dequant %.0f ms, swiglu+GEMMs %.0f ms, "
                          "first->last event %.0f ms (layer %d)", 48, t.n, t.dq_ms, t.gemm_ms, t.span_ms, layer);
}

bool Prefill::Run::Chunk::Experts::Group::compute_fp16(size_t j, const uint8_t* blob_dev, int slot) {
    const int32_t e = order[j];
    const int q = (int) (j % DQ);
    Fp16Timing& tm = timing();
    cudaEvent_t t0 = nullptr, t1 = nullptr, t2 = nullptr;
    // the dequant runs on its own stream so it overlaps the previous expert's GEMMs (BASALT_PREFILL_DQ_OVERLAP=1: on; measured no gain, so off by default)
    static const bool overlap = [] { const char* v = std::getenv("BASALT_PREFILL_DQ_OVERLAP"); return v && v[0] == '1'; }();
    if (overlap && m.ds == nullptr) {
        cudaStreamCreateWithFlags(&m.ds, cudaStreamNonBlocking);
        for (int i = 0; i < DQ; ++i) {
            cudaEventCreateWithFlags(&m.dq_done[i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&m.dq_free[i], cudaEventDisableTiming);
        }
    }
    const bool ov = overlap && m.ds != nullptr;
    cudaStream_t dqs = ov ? m.ds : m.cs;
    if (ov) {
        if (slot >= 0) cudaStreamWaitEvent(dqs, m.copied[slot], 0);   // the blob is on the device
        cudaStreamWaitEvent(dqs, m.dq_free[q], 0);                     // the GEMMs of the expert before gave buffer q back
    }
    if (tm.on) { t0 = tm.take(); t1 = tm.take(); t2 = tm.take(); cudaEventRecord(t0, dqs); }
    static const bool fused = [] { const char* v = std::getenv("BASALT_EXL3_FUSED"); return !v || v[0] != '0'; }();
    if (lay.native && fused && lay.fmt[(size_t) l].exl3_w) {
        // EXL3: no dequantized W - the projections run on the trellis directly (BASALT_EXL3_FUSED=0: the dequant path)
        const auto& f = lay.fmt[(size_t) l];
        const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
        if (tm.on) cudaEventRecord(t1, m.cs);
        basalt::kernels::exl3_expert_gu_fused(f.gu_type, blob_dev, f.up_off, m.Xs + o0 * N, ne, m.GU + o0 * 1280, m.cs);
        swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
        basalt::kernels::exl3_expert_down_fused(f.d_type, blob_dev + f.down_off, m.Hh + o0 * 640, ne, m.Dm + o0 * N, m.cs);
        if (slot >= 0) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
        if (tm.on) cudaEventRecord(t2, m.cs);
        return true;
    }
    if (lay.native) {
        // a native pack's layer, dequantized by llama.cpp's own formulas
        const auto& f = lay.fmt[(size_t) l];
        if (f.ms_k) {   // MS: each payload's rows, the input rotation undone (a mixed layer: the expert's own K)
            const auto& g = f.ms_mix ? basalt::kernels::cpu::ms_fmt_k(f, lay.ek[(size_t) l][(size_t) e]) : f;
            basalt::kernels::ms_dequant_expert_f16(g.gu_type, blob_dev + (f.ms_mix ? basalt::kernels::cpu::kMsMixHead : 0),
                                                   g.n_embd, g.n_ff, g.up_off, g.down_off, m.dq_gu[q], m.dq_d[q], dqs);
        } else if (f.exl3_w) {   // EXL3: W_full of the three records, transforms applied
            basalt::kernels::exl3_dequant_gu_f16(f.gu_type, blob_dev, f.up_off, f.n_ff, f.n_embd, m.dq_gu[q], dqs);
            basalt::kernels::exl3_dequant_down_f16(f.d_type, blob_dev + f.down_off, f.n_embd, f.n_ff, m.dq_d[q], dqs);
        } else {
            // NVFP4: the tail's FP32 scales {s_gate, s_up, s_down} go into the fp16 weights
            const float* tail = f.tail ? (const float*) (blob_dev + f.bytes - f.tail) : nullptr;
            basalt::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                               m.dq_gu[q], dqs, tail, f.up_type);
            basalt::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], dqs,
                                            tail ? tail + 2 : nullptr);
        }
    } else {
        blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], dqs);
    }
    if (tm.on) cudaEventRecord(t1, dqs);
    if (slot >= 0) { cudaEventRecord(m.used[slot], dqs); m.used_of[slot] = slot; }
    if (ov) {
        cudaEventRecord(m.dq_done[q], dqs);
        cudaStreamWaitEvent(m.cs, m.dq_done[q], 0);
    }
    const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
    m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
    swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
    m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
    if (ov) cudaEventRecord(m.dq_free[q], m.cs);
    if (tm.on) cudaEventRecord(t2, m.cs);
    return true;
}

}  // namespace basalt::prefill
