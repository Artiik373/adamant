// src/kernels/cpu/ms_rows.inc - the MS row kernel, built once per ISA (ms_expert.cpp: AVX2, ms_expert_vnni.cpp:
// AVX-VNNI).  BASALT_MS_VNNI picks the byte dot; BASALT_MS_ROWS names the entry point.

namespace basalt::kernels::cpu::ms_detail {
namespace {

// 8 states of columns c..c+7 (c a multiple of 8) from the 16 stream bytes at (c * K + 18) / 8: lane i takes the
// big-endian word at byte (i * K + 2) / 8 and shifts its 14 bits down.
template<int K>
struct Lanes {
    __m256i shuf, shuf0, shift;
    Lanes() {
        alignas(32) int8_t s[32], s0[32];
        alignas(32) int32_t sh[8];
        for (int i = 0; i < 8; ++i) {
            const int j = (i * K + 2) >> 3;
            for (int b = 0; b < 4; ++b) {
                const int8_t idx = (int8_t) (j + 3 - b);
                const int off = (i & 3) * 4 + b + (i >> 2) * 16;
                s[off] = idx;
                s0[off] = idx == 0 ? 1 : (idx == 1 ? 0 : idx);
            }
            sh[i] = 32 - 14 - ((i * K + 2) & 7);
        }
        shuf = _mm256_load_si256((const __m256i*) s);
        shuf0 = _mm256_load_si256((const __m256i*) s0);
        shift = _mm256_load_si256((const __m256i*) sh);
    }
};

template<int K>
inline __m256i decode8_s(const uint8_t* p, const __m256i shuf, const __m256i shift) {
    const __m256i w = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) p));
    const __m256i st = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(w, shuf), shift),
                                        _mm256_set1_epi32(0x3FFF));
    return _mm256_add_epi32(_mm256_mullo_epi32(st, _mm256_set1_epi32((int) 34038481u)), _mm256_set1_epi32(76625530));
}

inline __m256i dot8(__m256i acc, __m256i x, __m256i rep) {
#if BASALT_MS_VNNI
    return _mm256_dpbusd_avx_epi32(acc, x, rep);
#else
    const __m256i b = _mm256_madd_epi16(_mm256_maddubs_epi16(x, _mm256_set1_epi8(1)), _mm256_set1_epi16(1));
    return _mm256_add_epi32(acc, _mm256_mullo_epi32(b, _mm256_srai_epi32(rep, 24)));
#endif
}

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}

inline float half_f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

// NT tokens of one row: the states are decoded once per 8 columns for all of them.
template<int K, int NT>
inline void row_nt(const uint8_t* row, int n_in, const MsAct* const* a, float* const* out, int r, const Lanes<K>& L) {
    uint16_t d16;
    std::memcpy(&d16, row, 2);
    const int32_t* rp[NT];
    __m256i ia[NT];
    alignas(32) float fb[NT][8]{};
    __m256 fa[NT <= 4 ? NT : 1];
    for (int t = 0; t < NT; ++t) {
        rp[t] = a[t]->rep.data();
        ia[t] = _mm256_setzero_si256();
        if constexpr (NT <= 4) fa[t] = _mm256_setzero_ps();
    }
    const __m256i shuf = L.shuf, shuf0 = L.shuf0, shift = L.shift;
    const uint8_t* src = row + 2;   // group c's 16 stream bytes start at src + c * K / 8
    {
        const __m256i x = decode8_s<K>(src, shuf0, shift);
        for (int t = 0; t < NT; ++t) ia[t] = dot8(ia[t], x, _mm256_loadu_si256((const __m256i*) rp[t]));
    }
    for (int b = 0; b < n_in / 128; ++b) {
        for (int c = b * 128 + (b == 0 ? 8 : 0); c < b * 128 + 128; c += 8) {
            const __m256i x = decode8_s<K>(src + c * K / 8, shuf, shift);
            for (int t = 0; t < NT; ++t) ia[t] = dot8(ia[t], x, _mm256_loadu_si256((const __m256i*) (rp[t] + c)));
        }
        for (int t = 0; t < NT; ++t) {
            const __m256 vs = _mm256_set1_ps(a[t]->sc[b]);
            if constexpr (NT <= 4) {
                fa[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia[t]), vs, fa[t]);
            } else {
                const __m256 f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia[t]), vs, _mm256_load_ps(fb[t]));
                _mm256_store_ps(fb[t], f);
            }
            ia[t] = _mm256_setzero_si256();
        }
    }
    const float d = half_f(d16);
    for (int t = 0; t < NT; ++t) {
        const __m256 v = NT <= 4 ? fa[t] : _mm256_load_ps(fb[t]);
        out[t][r] = d * a[t]->rscale * (hsum(v) + a[t]->corr);
    }
}

// Two rows simultaneously for NT <= 4: shares activation loads and breaks decode dependency chains.
template<int K, int NT>
inline void row2_nt(const uint8_t* row0, const uint8_t* row1, int n_in, const MsAct* const* a, float* const* out, int r,
                    const Lanes<K>& L) {
    uint16_t d0, d1;
    std::memcpy(&d0, row0, 2);
    std::memcpy(&d1, row1, 2);
    const int32_t* rp[NT];
    __m256i ia0[NT], ia1[NT];
    alignas(32) float fb0[NT][8]{}, fb1[NT][8]{};
    __m256 fa0[NT <= 2 ? NT : 1], fa1[NT <= 2 ? NT : 1];
    for (int t = 0; t < NT; ++t) {
        rp[t] = a[t]->rep.data();
        ia0[t] = _mm256_setzero_si256();
        ia1[t] = _mm256_setzero_si256();
        if constexpr (NT <= 2) {
            fa0[t] = _mm256_setzero_ps();
            fa1[t] = _mm256_setzero_ps();
        }
    }
    const __m256i shuf = L.shuf, shuf0 = L.shuf0, shift = L.shift;
    const uint8_t* src0 = row0 + 2;
    const uint8_t* src1 = row1 + 2;
    {
        const __m256i x0 = decode8_s<K>(src0, shuf0, shift);
        const __m256i x1 = decode8_s<K>(src1, shuf0, shift);
        for (int t = 0; t < NT; ++t) {
            const __m256i rv = _mm256_loadu_si256((const __m256i*) rp[t]);
            ia0[t] = dot8(ia0[t], x0, rv);
            ia1[t] = dot8(ia1[t], x1, rv);
        }
    }
    for (int b = 0; b < n_in / 128; ++b) {
        for (int c = b * 128 + (b == 0 ? 8 : 0); c < b * 128 + 128; c += 8) {
            const __m256i x0 = decode8_s<K>(src0 + c * K / 8, shuf, shift);
            const __m256i x1 = decode8_s<K>(src1 + c * K / 8, shuf, shift);
            for (int t = 0; t < NT; ++t) {
                const __m256i rv = _mm256_loadu_si256((const __m256i*) (rp[t] + c));
                ia0[t] = dot8(ia0[t], x0, rv);
                ia1[t] = dot8(ia1[t], x1, rv);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const __m256 vs = _mm256_set1_ps(a[t]->sc[b]);
            if constexpr (NT <= 2) {
                fa0[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia0[t]), vs, fa0[t]);
                fa1[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia1[t]), vs, fa1[t]);
            } else {
                _mm256_store_ps(fb0[t], _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia0[t]), vs, _mm256_load_ps(fb0[t])));
                _mm256_store_ps(fb1[t], _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia1[t]), vs, _mm256_load_ps(fb1[t])));
            }
            ia0[t] = _mm256_setzero_si256();
            ia1[t] = _mm256_setzero_si256();
        }
    }
    const float h0 = half_f(d0), h1 = half_f(d1);
    for (int t = 0; t < NT; ++t) {
        const __m256 v0 = NT <= 2 ? fa0[t] : _mm256_load_ps(fb0[t]);
        const __m256 v1 = NT <= 2 ? fa1[t] : _mm256_load_ps(fb1[t]);
        out[t][r] = h0 * a[t]->rscale * (hsum(v0) + a[t]->corr);
        out[t][r + 1] = h1 * a[t]->rscale * (hsum(v1) + a[t]->corr);
    }
}

// Four rows simultaneously for NT = 1..2: 4x ILP hides vpmulld latency and cuts activation loads 4x.
template<int K, int NT>
inline void row4_nt(const uint8_t* row0, const uint8_t* row1, const uint8_t* row2, const uint8_t* row3, int n_in,
                    const MsAct* const* a, float* const* out, int r, const Lanes<K>& L) {
    uint16_t d0, d1, d2, d3;
    std::memcpy(&d0, row0, 2);
    std::memcpy(&d1, row1, 2);
    std::memcpy(&d2, row2, 2);
    std::memcpy(&d3, row3, 2);
    const int32_t* rp[NT];
    __m256i ia0[NT], ia1[NT], ia2[NT], ia3[NT];
    alignas(32) float fb0[NT][8]{}, fb1[NT][8]{}, fb2[NT][8]{}, fb3[NT][8]{};
    for (int t = 0; t < NT; ++t) {
        rp[t] = a[t]->rep.data();
        ia0[t] = _mm256_setzero_si256();
        ia1[t] = _mm256_setzero_si256();
        ia2[t] = _mm256_setzero_si256();
        ia3[t] = _mm256_setzero_si256();
    }
    const __m256i shuf = L.shuf, shuf0 = L.shuf0, shift = L.shift;
    const uint8_t* s0 = row0 + 2;
    const uint8_t* s1 = row1 + 2;
    const uint8_t* s2 = row2 + 2;
    const uint8_t* s3 = row3 + 2;
    {
        const __m256i x0 = decode8_s<K>(s0, shuf0, shift);
        const __m256i x1 = decode8_s<K>(s1, shuf0, shift);
        const __m256i x2 = decode8_s<K>(s2, shuf0, shift);
        const __m256i x3 = decode8_s<K>(s3, shuf0, shift);
        for (int t = 0; t < NT; ++t) {
            const __m256i rv = _mm256_loadu_si256((const __m256i*) rp[t]);
            ia0[t] = dot8(ia0[t], x0, rv);
            ia1[t] = dot8(ia1[t], x1, rv);
            ia2[t] = dot8(ia2[t], x2, rv);
            ia3[t] = dot8(ia3[t], x3, rv);
        }
    }
    for (int b = 0; b < n_in / 128; ++b) {
        for (int c = b * 128 + (b == 0 ? 8 : 0); c < b * 128 + 128; c += 8) {
            const __m256i x0 = decode8_s<K>(s0 + c * K / 8, shuf, shift);
            const __m256i x1 = decode8_s<K>(s1 + c * K / 8, shuf, shift);
            const __m256i x2 = decode8_s<K>(s2 + c * K / 8, shuf, shift);
            const __m256i x3 = decode8_s<K>(s3 + c * K / 8, shuf, shift);
            for (int t = 0; t < NT; ++t) {
                const __m256i rv = _mm256_loadu_si256((const __m256i*) (rp[t] + c));
                ia0[t] = dot8(ia0[t], x0, rv);
                ia1[t] = dot8(ia1[t], x1, rv);
                ia2[t] = dot8(ia2[t], x2, rv);
                ia3[t] = dot8(ia3[t], x3, rv);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const __m256 vs = _mm256_set1_ps(a[t]->sc[b]);
            _mm256_store_ps(fb0[t], _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia0[t]), vs, _mm256_load_ps(fb0[t])));
            _mm256_store_ps(fb1[t], _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia1[t]), vs, _mm256_load_ps(fb1[t])));
            _mm256_store_ps(fb2[t], _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia2[t]), vs, _mm256_load_ps(fb2[t])));
            _mm256_store_ps(fb3[t], _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia3[t]), vs, _mm256_load_ps(fb3[t])));
            ia0[t] = _mm256_setzero_si256();
            ia1[t] = _mm256_setzero_si256();
            ia2[t] = _mm256_setzero_si256();
            ia3[t] = _mm256_setzero_si256();
        }
    }
    const float h0 = half_f(d0), h1 = half_f(d1), h2 = half_f(d2), h3 = half_f(d3);
    for (int t = 0; t < NT; ++t) {
        const float rs = a[t]->rscale, cr = a[t]->corr;
        out[t][r]     = h0 * rs * (hsum(_mm256_load_ps(fb0[t])) + cr);
        out[t][r + 1] = h1 * rs * (hsum(_mm256_load_ps(fb1[t])) + cr);
        out[t][r + 2] = h2 * rs * (hsum(_mm256_load_ps(fb2[t])) + cr);
        out[t][r + 3] = h3 * rs * (hsum(_mm256_load_ps(fb3[t])) + cr);
    }
}

template<int K>
void rows_k(const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0, int r1) {
    static const Lanes<K> L;
    const size_t stride = (4 + ((size_t) (n_in - 1) * K + 7) / 8 + 3) & ~(size_t) 3;
    int r = r0;
    if (nt == 1 || nt == 2) {
        for (; r + 3 < r1; r += 4) {
            const uint8_t* row0 = rows + (size_t) r * stride;
            const uint8_t* row1 = row0 + stride;
            const uint8_t* row2 = row1 + stride;
            const uint8_t* row3 = row2 + stride;
            if (r + 7 < r1) {
                _mm_prefetch((const char*) (row3 + stride), _MM_HINT_T0);
                _mm_prefetch((const char*) (row3 + 2 * stride), _MM_HINT_T0);
                _mm_prefetch((const char*) (row3 + 3 * stride), _MM_HINT_T0);
                _mm_prefetch((const char*) (row3 + 4 * stride), _MM_HINT_T0);
            }
            if (nt == 1) row4_nt<K, 1>(row0, row1, row2, row3, n_in, a, out, r, L);
            else row4_nt<K, 2>(row0, row1, row2, row3, n_in, a, out, r, L);
        }
    }
    for (; r + 1 < r1; r += 2) {
        const uint8_t* row0 = rows + (size_t) r * stride;
        const uint8_t* row1 = row0 + stride;
        if (r + 3 < r1) {
            _mm_prefetch((const char*) (row1 + stride), _MM_HINT_T0);
            _mm_prefetch((const char*) (row1 + 2 * stride), _MM_HINT_T0);
        }
        for (int t0 = 0; t0 < nt;) {
            const int m = nt - t0;
            if (m >= 8) {
                row_nt<K, 8>(row0, n_in, a + t0, out + t0, r, L);
                row_nt<K, 8>(row1, n_in, a + t0, out + t0, r + 1, L);
                t0 += 8;
            } else if (m == 7) {
                row_nt<K, 7>(row0, n_in, a + t0, out + t0, r, L);
                row_nt<K, 7>(row1, n_in, a + t0, out + t0, r + 1, L);
                t0 += 7;
            } else if (m == 6) {
                row_nt<K, 6>(row0, n_in, a + t0, out + t0, r, L);
                row_nt<K, 6>(row1, n_in, a + t0, out + t0, r + 1, L);
                t0 += 6;
            } else if (m == 5) {
                row_nt<K, 5>(row0, n_in, a + t0, out + t0, r, L);
                row_nt<K, 5>(row1, n_in, a + t0, out + t0, r + 1, L);
                t0 += 5;
            } else if (m == 4) { row2_nt<K, 4>(row0, row1, n_in, a + t0, out + t0, r, L); t0 += 4; }
            else if (m == 3) { row2_nt<K, 3>(row0, row1, n_in, a + t0, out + t0, r, L); t0 += 3; }
            else if (m == 2) { row2_nt<K, 2>(row0, row1, n_in, a + t0, out + t0, r, L); t0 += 2; }
            else { row2_nt<K, 1>(row0, row1, n_in, a + t0, out + t0, r, L); t0 += 1; }
        }
    }
    if (r < r1) {
        const uint8_t* row = rows + (size_t) r * stride;
        for (int t0 = 0; t0 < nt;) {
            const int m = nt - t0;
            if (m >= 8) { row_nt<K, 8>(row, n_in, a + t0, out + t0, r, L); t0 += 8; }
            else if (m == 7) { row_nt<K, 7>(row, n_in, a + t0, out + t0, r, L); t0 += 7; }
            else if (m == 6) { row_nt<K, 6>(row, n_in, a + t0, out + t0, r, L); t0 += 6; }
            else if (m == 5) { row_nt<K, 5>(row, n_in, a + t0, out + t0, r, L); t0 += 5; }
            else if (m == 4) { row_nt<K, 4>(row, n_in, a + t0, out + t0, r, L); t0 += 4; }
            else if (m == 3) { row_nt<K, 3>(row, n_in, a + t0, out + t0, r, L); t0 += 3; }
            else if (m == 2) { row_nt<K, 2>(row, n_in, a + t0, out + t0, r, L); t0 += 2; }
            else { row_nt<K, 1>(row, n_in, a + t0, out + t0, r, L); t0 += 1; }
        }
    }
}

}  // namespace

void BASALT_MS_ROWS(int k, const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0,
                    int r1) {
    if (k == 2) rows_k<2>(rows, n_in, a, nt, out, r0, r1);
    else if (k == 3) rows_k<3>(rows, n_in, a, nt, out, r0, r1);
    else if (k == 4) rows_k<4>(rows, n_in, a, nt, out, r0, r1);
    else if (k == 5) rows_k<5>(rows, n_in, a, nt, out, r0, r1);
    else rows_k<6>(rows, n_in, a, nt, out, r0, r1);
}

}  // namespace basalt::kernels::cpu::ms_detail
