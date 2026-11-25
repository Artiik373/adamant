#include "experts.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/kernels/ms_kernels.hpp"
#include "basalt/models/qwen4exp/prefill/kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cstring>

namespace basalt::prefill {
using namespace detail;

void Prefill::Run::Chunk::Experts::Group::prepare_products() {
    if (use_mmq) {
        // the layer's activations as q8_1 rows in expert order, straight from `mixed`
        mmq::quantize(m.mixed, m.src_dev, m.Xq, mmq_gt, N, N, T * K, m.cs);
        // each group's rows: absolute bounds (gate/up reads the layer's rows), relative ones (down
        // reads the group's own quantized H)
        const size_t n = order.size(), ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
        m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
        for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
        m.bounds_host[n] = (int32_t) rows_local;
        for (size_t g = 0; g < ng; ++g)
            for (size_t i = 0; i <= MMQ_GROUP; ++i)
                m.bounds_host[n + 1 + g * (MMQ_GROUP + 1) + i] =
                    m.bounds_host[std::min(n, g * MMQ_GROUP + i)] - m.bounds_host[g * MMQ_GROUP];
        if (grp_mapped) {
            int32_t* bh = m.grp_host + 3 * m.grp_tk;
            std::memcpy(bh, m.bounds_host.data(), m.bounds_host.size() * 4);
            copy_i32(m.bounds_dev, m.grp_dev + 3 * m.grp_tk, (int64_t) m.bounds_host.size(), m.cs);
        } else {
            cudaMemcpyAsync(m.bounds_dev, m.bounds_host.data(), m.bounds_host.size() * 4,
                            cudaMemcpyHostToDevice, m.cs);
        }
    } else {
        gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
    }
    // multi-GPU: the peer's share, enqueued before the primary's own experts so both cards work at once
}

void Prefill::Run::Chunk::Experts::Group::flush() {
    if (gg.n <= gg.first) return;
    const auto& f = lay.fmt[(size_t) l];
    // Copy-ahead slots need not follow group order.
    for (int i = 0; i < gg_nslots; ++i)
        cudaStreamWaitEvent(m.cs, m.copied[gg_slots[i]], 0);
    float* sc = f.tail ? grp_scales(m.grp_d) : nullptr;
    if (f.ms_mix) {   // group by (k_gate, k_up, k_down) so each distinct tuple converts in one launch
        const uint8_t* blobs[MMQ_GROUP];
        int32_t slots[MMQ_GROUP];
        bool done[MMQ_GROUP]{};
        for (int i = gg.first; i < gg.n; ++i) {
            if (done[i]) continue;
            const int key = gg_k[i];
            int cnt = 0;
            for (int j = i; j < gg.n; ++j) {
                if (gg_k[j] != key) continue;
                done[j] = true;
                blobs[cnt] = gg.blob[j] + basalt::kernels::cpu::kMsMixHead;
                slots[cnt] = j - gg.first;
                ++cnt;
            }
            const auto& g = basalt::kernels::cpu::ms_fmt_k(f, key);
            basalt::kernels::ms_experts_to_q8_0_slots(g.gu_type, blobs, slots, cnt, g.n_embd, g.n_ff, g.up_off,
                                                      g.down_off, m.grp_gu + gg.first * mmq_gub, mmq_gub,
                                                      m.grp_d + gg.first * mmq_db, mmq_db, m.cs,
                                                      g.up_type, g.d_type);
        }
    } else if (f.ms_k) {   // MS: the group's Q8_0 slots from the trellis, one launch
        basalt::kernels::ms_experts_to_q8_0(f.gu_type, gg.blob + gg.first, gg.n - gg.first, f.n_embd, f.n_ff, f.up_off,
                                            f.down_off, m.grp_gu + gg.first * mmq_gub, mmq_gub,
                                            m.grp_d + gg.first * mmq_db, mmq_db, m.cs);
    } else if (!mmq::gather_native_group(gg, f.up_off, mmq_gb, f.down_off, mmq_db, m.grp_gu, mmq_gub,
                                  m.grp_d, mmq_db, m.cs, sc ? f.bytes - f.tail : 0, sc, mmq_ub)) {
        for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
            const uint8_t* b = gg.blob[i];
            mmq::gather_native(b, b + f.up_off, mmq_gb, b + f.down_off, mmq_db,
                               m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs,
                               sc ? b + f.bytes - f.tail : nullptr, sc ? sc + 4 * i : nullptr, mmq_ub);
        }
    }
    if (gg_nslots > 0) {
        const int rel = gg_slots[gg_nslots - 1];
        cudaEventRecord(m.used[rel], m.cs);
        for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
    }
    gg_nslots = 0;
    gg.first = gg.n;
}

bool Prefill::Run::Chunk::Experts::Group::compute_mmq(size_t j, const uint8_t* blob_dev, int slot) {
    // gather the expert into its group slot (GGUF blocks, unchanged or converted)
    const size_t q = j % MMQ_GROUP;
    if (group_gather) {
        gg.blob[q] = blob_dev;
        if (lay.native && lay.fmt[(size_t) l].ms_mix) gg_k[q] = lay.ek[(size_t) l][(size_t) order[j]];
        gg.n = (int) q + 1;
        if (slot >= 0) gg_slots[gg_nslots++] = slot;
        if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
        flush();
        gg = mmq::GatherGroup{};
    } else if (lay.native && lay.fmt[(size_t) l].ms_k) {   // MS: the slot's Q8_0 rows from the trellis
        const auto& fl = lay.fmt[(size_t) l];
        const auto& f = fl.ms_mix ? basalt::kernels::cpu::ms_fmt_k(fl, lay.ek[(size_t) l][(size_t) order[j]]) : fl;
        basalt::kernels::ms_expert_to_q8_0(f.gu_type, blob_dev + (fl.ms_mix ? basalt::kernels::cpu::kMsMixHead : 0),
                                           f.n_embd, f.n_ff, f.up_off, f.down_off,
                                           m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs,
                                           f.up_type, f.d_type);
    } else if (lay.native) {
        const auto& f = lay.fmt[(size_t) l];
        mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gb, blob_dev + f.down_off,
                           mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs,
                           f.tail ? blob_dev + f.bytes - f.tail : nullptr,
                           grp_scales(m.grp_d) + 4 * q, mmq_ub);
    } else {
        mmq::gather_q2(blob_dev, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
    }
    if (slot >= 0 && !group_gather) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
    if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
    // the group's products: gate/up, swiglu, the group's H to q8_1, down
    const size_t j0 = j - q, g = j0 / MMQ_GROUP, n = order.size();
    const int ngx = (int) (q + 1);
    const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
    int64_t maxr = 0;
    for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
    // the zeroed tail after the group's last expert (see MMQ_TAIL)
    cudaMemsetAsync(m.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, m.cs);
    cudaMemsetAsync(m.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, m.cs);
    mmq::Product gu;
    gu.w = m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
    gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = nullptr;
    gu.total_rows = T * K; gu.max_rows = maxr; gu.dst = m.GU; gu.ld_dst = 1280;
    if (mmq_ut != mmq_gt) {   // gate rows and up rows of two types: a product each, into the two halves of GU
        gu.w_rows = 640;
        m.mmq_ctx->run(gu, m.cs);
        gu.w = m.grp_gu + mmq_gb; gu.type = mmq_ut; gu.dst = m.GU + 640;
    }
    m.mmq_ctx->run(gu, m.cs);
    const bool scaled = lay.native && lay.fmt[(size_t) l].tail != 0;
    mmq::swiglu(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, !lay.native, m.cs,
                scaled ? grp_scales(m.grp_d) : nullptr,
                m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1), ngx);
    mmq::quantize(m.H + r0 * 640, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs);
    mmq::Product dn;
    dn.w = m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
    dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
    dn.ids = nullptr; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
    dn.ld_dst = N;
    m.mmq_ctx->run(dn, m.cs);
    return true;

}

}  // namespace basalt::prefill
