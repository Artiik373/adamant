// src/kernels/cpu/native_expert.cpp - native (GGUF-form) experts on the CPU through the extracted ggml dot products.
// See the header.  The activation quantizers and the row dot products here are ggml's (src/kernels/cpu/ggml), so an
// IQ expert computes what llama.cpp's CPU backend computes for it.
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/exl3.hpp"
#include "basalt/kernels/cpu/ms_expert.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/kernels/cpu/swiglu.hpp"
#include "basalt/log.hpp"

#include "basalt/ggml/ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace basalt::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// The prepared inputs of one projection record: x2[t * k ...], and on AVX2 their column-kernel form.
struct Exl3In {
    std::vector<float> x2, store, bias, sc;
    std::vector<int8_t> qstore;
    float* xp = nullptr;    // 32-byte aligned, 2 k floats a token
    int8_t* xq = nullptr;   // q8: 32-byte aligned, 8 k bytes a token
    bool fast = false, q8 = false;
};

void exl3_inputs(const Exl3Proj& p, const void* const* act, int nt, Exl3In& in) {
    static const bool avx2 = cpu_avx2_ok();
    const int variant = exl3_variant();
    thread_local std::vector<float> xs;
    xs.resize((size_t) p.k);
    in.x2.resize((size_t) p.k * (size_t) nt);
    in.fast = avx2 && exl3_cols_avx2_ok(p.w);
    in.q8 = in.fast && exl3_q8();
    const size_t per = (size_t) p.k * 2;
    if (in.q8) {
        in.qstore.resize((size_t) p.k * 8 * (size_t) nt + 32);
        in.sc.resize((size_t) (p.k / 128) * (size_t) nt);
        in.bias.resize((size_t) nt);
        in.xq = (int8_t*) (((uintptr_t) in.qstore.data() + 31) & ~(uintptr_t) 31);
    } else if (in.fast) {
        in.store.resize(per * (size_t) nt + 8);
        in.bias.resize((size_t) nt);
        in.xp = (float*) (((uintptr_t) in.store.data() + 31) & ~(uintptr_t) 31);
    }
    for (int t = 0; t < nt; ++t) {
        float* x2 = in.x2.data() + (size_t) t * p.k;
        exl3_unpack_act((const uint16_t*) act[t], p.k, xs.data());
        if (avx2) exl3_prepare_input_avx2(xs.data(), p.suh, p.k, variant, x2);
        else exl3_prepare_input(xs.data(), p.suh, p.k, variant, x2);
        if (in.q8)
            exl3_expand_input_q8(x2, p.k, in.xq + (size_t) t * p.k * 8, in.sc.data() + (size_t) t * (p.k / 128),
                                 &in.bias[(size_t) t]);
        else if (in.fast) exl3_expand_input(x2, p.k, in.xp + (size_t) t * per, &in.bias[(size_t) t]);
    }
}

// y1[t][c] for columns [c0, c1) of one record from prepared inputs.
void exl3_cols(const Exl3Proj& p, const Exl3In& in, int nt, float* const* y1, int c0, int c1) {
    if (!in.fast) {
        for (int t = 0; t < nt; ++t) exl3_cols_ref(p, in.x2.data() + (size_t) t * p.k, y1[t], c0, c1);
        return;
    }
    if (in.q8) {
        const int8_t* xq[64];
        const float* sc[64];
        for (int t0 = 0; t0 < nt; t0 += 64) {
            const int n = (std::min)(64, nt - t0);
            for (int t = 0; t < n; ++t) {
                xq[t] = in.xq + (size_t) (t0 + t) * p.k * 8;
                sc[t] = in.sc.data() + (size_t) (t0 + t) * (p.k / 128);
            }
            exl3_cols_q8(p, xq, sc, in.bias.data() + t0, n, y1 + t0, c0, c1);
        }
        return;
    }
    const float* xp[64];
    for (int t0 = 0; t0 < nt; t0 += 64) {
        const int n = (std::min)(64, nt - t0);
        for (int t = 0; t < n; ++t) xp[t] = in.xp + (size_t) (t0 + t) * p.k * 2;
        exl3_cols_expanded_avx2(p, xp, in.bias.data() + t0, n, y1 + t0, c0, c1);
    }
}

void exl3_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out, int r0,
                    int r1) {
    const Exl3Proj p = exl3_proj_at(blob + f.down_off, (int) f.n_ff, (int) f.n_embd, f.exl3_w);
    thread_local Exl3In in;
    exl3_inputs(p, hq, nt, in);
    exl3_cols(p, in, nt, out, r0, r1);
    for (int t = 0; t < nt; ++t) exl3_finish(p, out[t], r0 / 128, r1 / 128, exl3_variant());
}

// The rotated inputs of one MS payload's signs for nt tokens, kept while the same bytes come again (a job's parts).
struct MsIn {
    std::vector<int8_t> signs;   // by value: a ring slot can hold another expert at the same address
    std::vector<std::vector<uint8_t>> raw;
    std::vector<MsAct> act;
    std::vector<const MsAct*> ptr;
};

const MsAct* const* ms_inputs(MsIn& c, const int8_t* signs, const void* const* act, int nt, int n) {
    const size_t bytes = (size_t) n * 4;
    bool same = (int) c.signs.size() == n && std::memcmp(c.signs.data(), signs, (size_t) n) == 0 &&
                (int) c.raw.size() == nt;
    for (int t = 0; same && t < nt; ++t) same = std::memcmp(c.raw[(size_t) t].data(), act[t], bytes) == 0;
    if (same) return c.ptr.data();
    c.signs.assign(signs, signs + n);
    c.raw.resize((size_t) nt);
    c.act.resize((size_t) nt);
    c.ptr.resize((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        c.raw[(size_t) t].assign((const uint8_t*) act[t], (const uint8_t*) act[t] + bytes);
        ms_act_prepare((const float*) act[t], signs, n, c.act[(size_t) t]);
        c.ptr[(size_t) t] = &c.act[(size_t) t];
    }
    return c.ptr.data();
}

void ms_gu_rows(const NativeFmt& fl, const uint8_t* blob, const void* const* act, int nt, float* const* ff, int r0,
                int r1) {
    const NativeFmt& f = ms_blob_fmt(fl, blob);
    const int n = (int) f.n_embd;
    const int8_t* sg = (const int8_t*) blob + f.up_off - n;   // each payload's signs follow its rows
    const int8_t* su = (const int8_t*) blob + f.down_off - n;
    thread_local MsIn cg, cu;
    const MsAct* const* ag = ms_inputs(cg, sg, act, nt, n);
    const MsAct* const* au = std::memcmp(sg, su, (size_t) n) == 0 ? ag : ms_inputs(cu, su, act, nt, n);
    thread_local std::vector<float> u;
    thread_local std::vector<float*> up;
    u.resize((size_t) nt * (size_t) f.n_ff);
    up.resize((size_t) nt);
    for (int t = 0; t < nt; ++t) up[(size_t) t] = u.data() + (size_t) t * f.n_ff;
    ms_rows(f.ms_k, blob, n, ag, nt, ff, r0, r1);
    ms_rows(f.ms_k, blob + f.up_off, n, au, nt, up.data(), r0, r1);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            float a = ff[t][r], b = up[(size_t) t][r];
            if (f.swiglu_limit > 0.f) {
                a = std::fmin(a, f.swiglu_limit);
                b = std::fmax(std::fmin(b, f.swiglu_limit), -f.swiglu_limit);
            }
            ff[t][r] = (a / (1.f + std::exp(-a))) * b;
        }
}

void ms_down_rows(const NativeFmt& fl, const uint8_t* blob, const void* const* hq, int nt, float* const* out, int r0,
                  int r1) {
    const NativeFmt& f = ms_blob_fmt(fl, blob);
    const int n = (int) f.n_ff;
    const uint8_t* rows = blob + f.down_off;
    thread_local MsIn cd;
    const MsAct* const* a = ms_inputs(cd, (const int8_t*) rows + (size_t) f.n_embd * f.d_row, hq, nt, n);
    ms_rows(f.ms_k, rows, n, a, nt, out, r0, r1);
}

}  // namespace

void exl3_gu_cols(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* g1,
                  float* const* u1, int c0, int c1, uint64_t stamp) {
    const Exl3Proj pg = exl3_proj_at(blob, (int) f.n_embd, (int) f.n_ff, f.exl3_w);
    const Exl3Proj pu = exl3_proj_at(blob + f.up_off, (int) f.n_embd, (int) f.n_ff, f.exl3_w);
    struct Cache {
        uint64_t stamp = 0;
        const uint8_t* blob = nullptr;
        Exl3In xg, xu;
    };
    thread_local Cache c;
    if (stamp == 0 || c.stamp != stamp || c.blob != blob) {
        exl3_inputs(pg, act, nt, c.xg);
        exl3_inputs(pu, act, nt, c.xu);
        c.stamp = stamp;
        c.blob = blob;
    }
    exl3_cols(pg, c.xg, nt, g1, c0, c1);
    exl3_cols(pu, c.xu, nt, u1, c0, c1);
}

void exl3_gu_finish(const NativeFmt& f, const uint8_t* blob, int nt, float* const* ff, float* const* u1, int b0, int b1) {
    const Exl3Proj pg = exl3_proj_at(blob, (int) f.n_embd, (int) f.n_ff, f.exl3_w);
    const Exl3Proj pu = exl3_proj_at(blob + f.up_off, (int) f.n_embd, (int) f.n_ff, f.exl3_w);
    const int variant = exl3_variant();
    for (int t = 0; t < nt; ++t) {
        exl3_finish(pg, ff[t], b0, b1, variant);
        exl3_finish(pu, u1[t], b0, b1, variant);
        for (int r = b0 * 128; r < b1 * 128; ++r) {
            float a = ff[t][r], b = u1[t][r];
            if (f.swiglu_limit > 0.f) {
                a = std::fmin(a, f.swiglu_limit);
                b = std::fmax(std::fmin(b, f.swiglu_limit), -f.swiglu_limit);
            }
            ff[t][r] = (a / (1.f + std::exp(-a))) * b;
        }
    }
}

namespace {

SwiGlu glu_of(const NativeFmt& f) { return {f.swiglu_limit, f.gate_softcap, f.up_softcap}; }

// The gate/up rows through ggml's single-token dot, then the layer's SwiGLU.
void gu_rows_ggml(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff, int r0,
                  int r1, float sg, float su) {
    const SwiGlu glu = glu_of(f);
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot, dot_u = traits(f.up_type)->vec_dot;
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.up_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot_u(n, &u, 0, ur, 0, act[t], 0, 1);
            if (f.tail) {
                g *= sg;
                u *= su;
            }
            ff[t][r] = swiglu(glu, g, u);
        }
    }
}

// A layer whose gate and up types differ: each half through a multi-token row kernel (`rows`: iq512_rows or
// iq256_rows), then native_gu_rows' SwiGLU expression.  A token's dots are the uniform layers' bits for each type.
template<typename Rows>
void gu_rows_mixed(Rows rows, const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                   int r0, int r1) {
    constexpr int kTile = 64, kTok = 8;
    float ub[kTok][kTile];
    for (int t0 = 0; t0 < nt; t0 += kTok) {
        const int m = nt - t0 < kTok ? nt - t0 : kTok;
        for (int a = r0; a < r1; a += kTile) {
            const int b = a + kTile < r1 ? a + kTile : r1;
            float* up[kTok];
            for (int t = 0; t < m; ++t) up[t] = ub[t];
            rows(f.gu_type, blob, f.gu_row, (int) f.n_embd, act + t0, m, ff + t0, a, b);
            rows(f.up_type, blob + f.up_off + (size_t) a * f.up_row, f.up_row, (int) f.n_embd, act + t0, m, up, 0, b - a);
            for (int t = 0; t < m; ++t)
                for (int r = a; r < b; ++r) {
                    const float g = ff[t0 + t][r], u = ub[t][r - a];
                    ff[t0 + t][r] = (g / (1.f + std::exp(-g))) * u;
                }
        }
    }
}

// MXFP8 (type 46) rows: each token's activation is converted once, a weight block decoded once for every token of the
// call (the vec_dot of the table is single-token), and each token's dot is that vec_dot's bits.  `out[t][r]`, r in [r0, r1).
constexpr int kMx8Tok = 8;
struct Mx8Act {
    float* p[kMx8Tok];
    Mx8Act(const void* const* act, int m, int n) {
        thread_local std::vector<float> buf;   // one set per pool worker, grown to the widest row it has met
        const size_t per = (size_t) (n / 128) * GGML_MXFP8_PREP_BLOCK;
        if (buf.size() < (size_t) m * per) buf.resize((size_t) kMx8Tok * per);
        for (int t = 0; t < m; ++t) {
            p[t] = buf.data() + (size_t) t * per;
            ggml_mxfp8_q8_0_prep(n, act[t], p[t]);
        }
    }
};

void mx8_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff, int r0,
                 int r1) {
    const SwiGlu glu = glu_of(f);
    const int n = (int) f.n_embd;
    thread_local std::vector<float> ub((size_t) kMx8Tok * (size_t) kNativeMaxFF);
    for (int t0 = 0; t0 < nt; t0 += kMx8Tok) {
        const int m = nt - t0 < kMx8Tok ? nt - t0 : kMx8Tok;
        Mx8Act a(act + t0, m, n);
        float* up[kMx8Tok];
        for (int t = 0; t < m; ++t) up[t] = ub.data() + (size_t) t * kNativeMaxFF;
        ggml_mxfp8_q8_0_rows(n, blob, f.gu_row, a.p, m, ff + t0, r0, r1);
        ggml_mxfp8_q8_0_rows(n, blob + f.up_off, f.up_row, a.p, m, up, r0, r1);
        for (int t = 0; t < m; ++t)
            for (int r = r0; r < r1; ++r) ff[t0 + t][r] = swiglu(glu, ff[t0 + t][r], up[t][r]);
    }
}

void mx8_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out, int r0,
                   int r1) {
    const int n = (int) f.n_ff;
    for (int t0 = 0; t0 < nt; t0 += kMx8Tok) {
        const int m = nt - t0 < kMx8Tok ? nt - t0 : kMx8Tok;
        Mx8Act a(hq + t0, m, n);
        ggml_mxfp8_q8_0_rows(n, blob + f.down_off, f.d_row, a.p, m, out + t0, r0, r1);
    }
}

}  // namespace

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err,
                float swiglu_limit, int up_type) {
    if (up_type < 0) up_type = gu_type;
    if (basalt::ms_tensor_type((uint32_t) gu_type) || basalt::ms_tensor_type((uint32_t) d_type) ||
        basalt::ms_tensor_type((uint32_t) up_type)) {
        if (gu_type != d_type || gu_type != up_type) {
            err = "native experts: an MS layer has one K for gate, up and down";
            return false;
        }
        if (!cpu_avx2_ok()) {
            err = "native experts: MS experts need an AVX2 CPU";
            return false;
        }
        if (n_embd % 128 || n_ff % 128) {
            err = "native experts: MS geometry is not whole 128-column blocks";
            return false;
        }
        const int k = basalt::ms_bits((uint32_t) gu_type);
        f.gu_type = f.d_type = f.up_type = gu_type;
        f.ms_k = k;
        f.n_embd = n_embd;
        f.n_ff = n_ff;
        f.swiglu_limit = swiglu_limit;
        f.gu_act = f.d_act = -1;
        f.gu_row = f.up_row = (size_t) basalt::ms_row_bytes((uint64_t) n_embd, k);
        f.d_row = (size_t) basalt::ms_row_bytes((uint64_t) n_ff, k);
        f.up_off = f.gu_row * (size_t) n_ff + (size_t) n_embd;
        f.down_off = 2 * f.up_off;
        f.bytes = f.down_off + f.d_row * (size_t) n_embd + (size_t) n_ff;
        f.tail = 0;
        f.act_bytes = (size_t) n_embd * 4;   // fp32: no range limit before the int8 rotated input
        f.h_bytes = (size_t) n_ff * 4;
        if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes || n_ff > kNativeMaxFF) {
            err = "native experts: activation larger than the pool's buffers";
            return false;
        }
        return true;
    }
    if (exl3_type(gu_type) || exl3_type(d_type) || exl3_type(up_type)) {
        if (gu_type != d_type || gu_type != up_type) {
            err = "native experts: an EXL3 layer has one bit rate for gate, up and down";
            return false;
        }
        const int w = exl3_words(gu_type);
        if (!exl3_geometry_ok((int) n_embd, (int) n_ff, w) || !exl3_geometry_ok((int) n_ff, (int) n_embd, w)) {
            err = "native experts: EXL3 geometry is not whole 128-blocks";
            return false;
        }
        f.gu_type = f.d_type = f.up_type = gu_type;
        f.exl3_w = w;
        f.n_embd = n_embd;
        f.n_ff = n_ff;
        f.swiglu_limit = swiglu_limit;
        f.gu_act = f.d_act = -1;
        // bytes of trellis per output row: what a prefetch of a row range covers
        f.gu_row = f.up_row = (size_t) n_embd * (size_t) w / 128;
        f.d_row = (size_t) n_ff * (size_t) w / 128;
        f.up_off = exl3_proj_bytes((int) n_embd, (int) n_ff, w);
        f.down_off = f.up_off + exl3_proj_bytes((int) n_embd, (int) n_ff, w);
        f.bytes = f.down_off + exl3_proj_bytes((int) n_ff, (int) n_embd, w);
        f.tail = 0;
        f.act_bytes = (size_t) n_embd * 2;
        f.h_bytes = (size_t) n_ff * 2;
        if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes || n_ff > kNativeMaxFF) {
            err = "native experts: activation larger than the pool's buffers";
            return false;
        }
        return true;
    }
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* tu = traits(up_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr || tu == nullptr ||
        tu->vec_dot == nullptr) {
        err = "native experts: no CPU dot product for type " +
              std::to_string(!(tg && tg->vec_dot) ? gu_type : !(tu && tu->vec_dot) ? up_type : d_type);
        return false;
    }
    if (tu->vec_dot_type != tg->vec_dot_type || n_embd % ggml_blck_size((ggml_type) up_type) ||
        (up_type != gu_type && (gu_type == GGML_TYPE_NVFP4 || up_type == GGML_TYPE_NVFP4 || gu_type == GGML_TYPE_Q2_0 ||
                               up_type == GGML_TYPE_Q2_0))) {
        // NVFP4's scale tail and the Q2_0 kernels' own activations are whole-layer
        err = "native experts: gate type " + std::to_string(gu_type) + " and up type " + std::to_string(up_type) +
              " cannot share one layer";
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.swiglu_limit = swiglu_limit;
    f.gate_softcap = f.up_softcap = 0.f;
    f.up_type = up_type;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.up_row = ggml_row_size((ggml_type) up_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = f.up_off + f.up_row * (size_t) n_ff;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    // NVFP4 keeps its per-expert FP32 scales in a tail after down, so they move with the blob through every tier.
    // The scales cover all three projections: the two kinds cannot mix in one expert.
    if ((gu_type == GGML_TYPE_NVFP4) != (d_type == GGML_TYPE_NVFP4)) {
        err = "native experts: NVFP4 gate/up and down come together (the scale tail covers all three)";
        return false;
    }
    f.tail = gu_type == GGML_TYPE_NVFP4 ? kNvfp4Tail : 0;
    f.bytes += f.tail;
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

const NativeFmt& ms_blob_fmt(const NativeFmt& f, const uint8_t*& blob) {
    if (!f.ms_mix) return f;
    const NativeFmt& g = ms_fmt_k(f, blob[0]);
    blob += kMsMixHead;
    return g;
}

const NativeFmt& ms_fmt_k(const NativeFmt& f, int k) {
    thread_local NativeFmt by_k[7];
    NativeFmt& g = by_k[k < 2 || k > 6 ? 0 : k];
    if (g.ms_k != k || g.n_embd != f.n_embd || g.n_ff != f.n_ff || g.swiglu_limit != f.swiglu_limit) {
        std::string err;
        if (k < 2 || k > 6 || !native_fmt(120 + k, 120 + k, f.n_embd, f.n_ff, g, err, f.swiglu_limit)) {
            basalt::log::error("basalt", "mixed MS expert: bad K %d", k);
            std::exit(1);
        }
    }
    return g;
}

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (f.ms_k) {
        std::memcpy(dst, x, (size_t) f.n_embd * 4);
        return;
    }
    if (f.exl3_w) {
        exl3_pack_act(x, (int) f.n_embd, (uint16_t*) dst);
        return;
    }
    static const bool avx512 = cpu_avx512_ok();
    if (f.gu_act == GGML_TYPE_Q8_K) {   // ggml's reference quantizer is scalar
        if (avx512) {
            quantize_q8_K_avx512(x, dst, f.n_embd);
            return;
        }
        quantize_q8_K_avx2(x, dst, f.n_embd);   // the engine runs on AVX2 CPUs at least
        return;
    }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (f.ms_k) {
        std::memcpy(dst, h, (size_t) f.n_ff * 4);
        return;
    }
    if (f.exl3_w) {
        exl3_pack_act(h, (int) f.n_ff, (uint16_t*) dst);
        return;
    }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    if (f.ms_k) {
        ms_gu_rows(f, blob, act, nt, ff, r0, r1);
        return;
    }
    if (f.exl3_w) {   // whole 128-blocks
        thread_local std::vector<float> u;
        u.resize((size_t) kNativeMaxFF * (size_t) nt);
        std::vector<float*> up((size_t) nt);
        for (int t = 0; t < nt; ++t) up[(size_t) t] = u.data() + (size_t) t * kNativeMaxFF;
        exl3_gu_cols(f, blob, act, nt, ff, up.data(), r0, r1, 0);
        exl3_gu_finish(f, blob, nt, ff, up.data(), r0 / 128, r1 / 128);
        return;
    }
    // The multi-token kernels decode the weights once for all tokens; the single-token ggml dot below serves the
    // formats without one.  A token's rows round the same alone as in a group, so greedy output does not depend on
    // the number of drafts in a window.  A format with only an AVX-2 kernel (IQ4_XS) takes it on AVX-2 CPUs
    // only: an AVX-512 CPU keeps the ggml dot for it.
    static const bool avx512 = cpu_avx512_ok();
    static const bool avx2 = cpu_avx2_ok();
    static const bool cpu512 = avx512;
    // only the NVFP4 rows below take the softcapped SwiGLU: a softcapped layer of another type takes the ggml dot
    const bool soft = f.gate_softcap > 0.f;
    if (f.up_type != f.gu_type) {   // gate and up of two types: each half on its type's row kernel
        if (f.swiglu_limit <= 0.f && f.tail == 0 && !soft) {
            if (avx512 && iq512_supported(f.gu_type) && iq512_supported(f.up_type)) {
                gu_rows_mixed(iq512_rows, f, blob, act, nt, ff, r0, r1);
                return;
            }
            if (!cpu512 && avx2 && iq256_supported(f.gu_type) && iq256_supported(f.up_type)) {
                gu_rows_mixed(iq256_rows, f, blob, act, nt, ff, r0, r1);
                return;
            }
        }
        gu_rows_ggml(f, blob, act, nt, ff, r0, r1, 1.f, 1.f);
        return;
    }
    // the multi-token kernels have the plain SwiGLU built in: a limited layer takes the ggml dot below
    if (f.swiglu_limit <= 0.f && !soft && (iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (avx512 && iq512_supported(f.gu_type)) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2 && iq256_supported(f.gu_type)) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    if (f.gu_type == GGML_TYPE_MXFP8 && f.up_type == GGML_TYPE_MXFP8 && f.n_ff <= kNativeMaxFF) {
        mx8_gu_rows(f, blob, act, nt, ff, r0, r1);
        return;
    }
    const int n = (int) f.n_embd;
    float sg = 1.f, su = 1.f;
    if (f.tail) {
        std::memcpy(&sg, blob + f.bytes - f.tail, 4);
        std::memcpy(&su, blob + f.bytes - f.tail + 4, 4);
    }
    // IQ1_S on AVX-512: bitwise equal per token to the ggml dot, the SwiGLU limit included.
    if (avx512 && !soft && f.gu_type == GGML_TYPE_IQ1_S && iq1s_512_fits(n)) {
        iq1s_512_gu_rows(blob, f.gu_row, f.up_off, n, sg, su, f.tail != 0, f.swiglu_limit, act, nt, ff, r0, r1);
        return;
    }
    if (avx512 && !soft && f.swiglu_limit > 0.f && iq_exact_512_supported(f.gu_type) && iq1s_512_fits(n)) {
        iq_exact_512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, n, sg, su, f.tail != 0, f.swiglu_limit, act, nt, ff, r0, r1);
        return;
    }
    // NVFP4 on AVX-512: the multi-token kernel, bitwise equal per token to the ggml dot it replaces (the layer's SwiGLU
    // included).
    if (avx512 && f.gu_type == GGML_TYPE_NVFP4 && nvfp4_512_fits((int) f.n_embd, (int) f.n_ff)) {
        nvfp4_512_gu_rows(blob, f.gu_row, f.up_off, n, sg, su, glu_of(f), act, nt, ff, r0, r1);
        return;
    }
    // The same on AVX2 CPUs, bitwise equal per token to the AVX2 ggml dot.
    if (!avx512 && avx2 && f.gu_type == GGML_TYPE_NVFP4 && nvfp4_256_fits((int) f.n_embd, (int) f.n_ff)) {
        nvfp4_256_gu_rows(blob, f.gu_row, f.up_off, n, sg, su, glu_of(f), act, nt, ff, r0, r1);
        return;
    }
    gu_rows_ggml(f, blob, act, nt, ff, r0, r1, sg, su);
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    if (f.ms_k) {
        ms_down_rows(f, blob, hq, nt, out, r0, r1);
        return;
    }
    if (f.exl3_w) {
        exl3_down_rows(f, blob, hq, nt, out, r0, r1);
        return;
    }
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles once per block instead of once per token;
    // the ggml dot is single-token.
    if (cpu_avx2_ok() && f.d_type == 20 && iq4nl256_fits((int) f.n_ff)) {   // the same rule as the gate/up rows
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (f.d_type == GGML_TYPE_MXFP8) {
        mx8_down_rows(f, blob, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    float sd = 1.f;
    if (f.tail) std::memcpy(&sd, blob + f.bytes - f.tail + 8, 4);
    static const bool avx512 = cpu_avx512_ok();
    if (avx512 && f.d_type == GGML_TYPE_NVFP4 && nvfp4_512_fits((int) f.n_embd, (int) f.n_ff)) {
        nvfp4_512_down_rows(blob + f.down_off, f.d_row, n, sd, hq, nt, out, r0, r1);
        return;
    }
    if (!avx512 && cpu_avx2_ok() && f.d_type == GGML_TYPE_NVFP4 && nvfp4_256_fits((int) f.n_embd, (int) f.n_ff)) {
        nvfp4_256_down_rows(blob + f.down_off, f.d_row, n, sd, hq, nt, out, r0, r1);
        return;
    }
    if (avx512 && f.d_type == GGML_TYPE_IQ1_S && !f.tail && iq1s_512_fits(n)) {
        iq1s_512_rows(blob + f.down_off, f.d_row, n, hq, nt, out, r0, r1);
        return;
    }
    if (avx512 && f.swiglu_limit > 0.f && !f.tail && iq_exact_512_supported(f.d_type) && iq1s_512_fits(n)) {
        iq_exact_512_rows(f.d_type, blob + f.down_off, f.d_row, n, hq, nt, out, r0, r1);
        return;
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = f.tail ? s * sd : s;
        }
    }
}

}  // namespace basalt::kernels::cpu
