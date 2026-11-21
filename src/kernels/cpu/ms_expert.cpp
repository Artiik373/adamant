// src/kernels/cpu/ms_expert.cpp - MS projections on the CPU.  See the header.  Built with the AVX2 flags; reached
// only behind cpu_avx2_ok() (native_fmt refuses MS experts on other CPUs).
#include "basalt/kernels/cpu/ms_expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/artifact/gguf_reader.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstring>

#define BASALT_MS_VNNI 0
#define BASALT_MS_ROWS rows_avx2
#include "ms_rows.inc"

namespace basalt::kernels::cpu {
namespace ms_detail {
void rows_vnni(int k, const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0, int r1);
}

void ms_act_prepare(const float* x, const int8_t* signs, int n_in, MsAct& a) {
    const int hb = (int) basalt::ms_hadamard_block((uint64_t) n_in);
    thread_local std::vector<float> v;
    v.resize((size_t) n_in);
    for (int i = 0; i < n_in; ++i) v[(size_t) i] = x[i] * (float) signs[i];
    for (int h = 1; h < hb; h <<= 1)
        for (int i = 0; i < n_in; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float p = v[(size_t) j], q = v[(size_t) (j + h)];
                v[(size_t) j] = p + q;
                v[(size_t) (j + h)] = p - q;
            }
    a.rep.resize((size_t) n_in);
    a.sc.resize((size_t) (n_in / 128));
    a.rscale = 1.f / std::sqrt((float) hb);
    double corr = 0.0;
    for (int b = 0; b < n_in / 128; ++b) {
        float amax = 0.f;
        for (int i = b * 128; i < b * 128 + 128; ++i) amax = std::fmax(amax, std::fabs(v[(size_t) i]));
        const float s = amax / 127.f, inv = amax > 0.f ? 127.f / amax : 0.f;
        int sum = 0;
        for (int i = b * 128; i < b * 128 + 128; ++i) {
            const int q = (int) std::nearbyint(v[(size_t) i] * inv);
            sum += q;
            a.rep[(size_t) i] = (int32_t) ((uint32_t) (uint8_t) (int8_t) q * 0x01010101u);
        }
        a.sc[(size_t) b] = s;
        corr += (double) s * sum;
    }
    a.corr = (float) (-510.0 * corr);
}

void ms_rows(int k, const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) ms_detail::rows_vnni(k, rows, n_in, a, nt, out, r0, r1);
    else ms_detail::rows_avx2(k, rows, n_in, a, nt, out, r0, r1);
}

void ms_rows_ref(int k, const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0,
                 int r1) {
    const size_t stride = basalt::ms_row_bytes((uint64_t) n_in, k);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = rows + (size_t) r * stride;
        uint16_t d16, s0;
        std::memcpy(&d16, row, 2);
        std::memcpy(&s0, row + 2, 2);
        uint32_t state = s0 & 0x3FFFu;
        for (int t = 0; t < nt; ++t) {
            double acc = 0.0;
            uint32_t st = state;
            for (int c = 0; c < n_in; ++c) {
                if (c > 0) {
                    const int bit = (c - 1) * k;
                    const uint32_t w = ((uint32_t) row[4 + (bit >> 3)] << 8) | row[4 + (bit >> 3) + 1];
                    st = ((st << k) | ((w >> (16 - (bit & 7) - k)) & ((1u << k) - 1))) & 0x3FFFu;
                }
                const uint32_t x = st * 34038481u + 76625530u;
                const int v = (int) ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24));
                const int8_t q = (int8_t) (a[t]->rep[(size_t) c] & 0xFF);
                acc += (double) a[t]->sc[(size_t) (c / 128)] * v * q;
            }
            out[t][r] = (float) ((double) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(d16))) * a[t]->rscale * (acc + a[t]->corr));
        }
    }
}

}  // namespace basalt::kernels::cpu
