// src/kernels/cpu/ms_rows.inc - the MS row kernel, built once per ISA (ms_expert.cpp: AVX2, ms_expert_vnni.cpp:
// AVX-VNNI).  BASALT_MS_VNNI picks the byte dot; BASALT_MS_ROWS names the entry point.

namespace basalt::kernels::cpu::ms_detail {
namespace {

// 8 states of columns c..c+7 (c a multiple of 8) from the 16 stream bytes at (c * K + 18) / 8: lane i takes the
// big-endian word at byte (i * K + 2) / 8 and shifts its 14 bits down.
template<int K>
struct Lanes {
    __m256i shuf, shift;
    Lanes() {
        alignas(32) int8_t s[32];
        alignas(32) int32_t sh[8];
        for (int i = 0; i < 8; ++i) {
            const int j = (i * K + 2) >> 3;
            for (int b = 0; b < 4; ++b) s[(i & 3) * 4 + b + (i >> 2) * 16] = (int8_t) (j + 3 - b);
            sh[i] = 32 - 14 - ((i * K + 2) & 7);
        }
        shuf = _mm256_load_si256((const __m256i*) s);
        shift = _mm256_load_si256((const __m256i*) sh);
    }
};

template<int K>
inline __m256i decode8(const uint8_t* p, const Lanes<K>& L) {
    const __m256i w = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) p));
    const __m256i st = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(w, L.shuf), L.shift),
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
    uint16_t d16, s0;
    std::memcpy(&d16, row, 2);
    std::memcpy(&s0, row + 2, 2);
    // stream bytes 0..3 are s0 as a big-endian word; the first group reads bytes 2..17
    alignas(16) uint8_t head[16];
    head[0] = (uint8_t) ((s0 & 0x3FFF) >> 8);
    head[1] = (uint8_t) (s0 & 0xFF);
    std::memcpy(head + 2, row + 4, 14);
    const int32_t* rp[NT];
    __m256 fa[NT];
    __m256i ia[NT];
    for (int t = 0; t < NT; ++t) {
        rp[t] = a[t]->rep.data();
        fa[t] = _mm256_setzero_ps();
        ia[t] = _mm256_setzero_si256();
    }
    const uint8_t* src = row + 2;   // group c's 16 stream bytes start at src + c * K / 8
    {
        const __m256i x = decode8<K>(head, L);
        for (int t = 0; t < NT; ++t) ia[t] = dot8(ia[t], x, _mm256_loadu_si256((const __m256i*) rp[t]));
    }
    for (int b = 0; b < n_in / 128; ++b) {
        for (int c = b * 128 + (b == 0 ? 8 : 0); c < b * 128 + 128; c += 8) {
            const __m256i x = decode8<K>(src + c * K / 8, L);
            for (int t = 0; t < NT; ++t) ia[t] = dot8(ia[t], x, _mm256_loadu_si256((const __m256i*) (rp[t] + c)));
        }
        for (int t = 0; t < NT; ++t) {
            fa[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia[t]), _mm256_set1_ps(a[t]->sc[b]), fa[t]);
            ia[t] = _mm256_setzero_si256();
        }
    }
    const float d = half_f(d16);
    for (int t = 0; t < NT; ++t) out[t][r] = d * a[t]->rscale * (hsum(fa[t]) + a[t]->corr);
}

template<int K>
void rows_k(const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0, int r1) {
    static const Lanes<K> L;
    const size_t stride = (4 + ((size_t) (n_in - 1) * K + 7) / 8 + 3) & ~(size_t) 3;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = rows + (size_t) r * stride;
        if (r + 1 < r1) _mm_prefetch((const char*) (row + stride), _MM_HINT_T0);
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int m = nt - t0;
            if (m >= 4) row_nt<K, 4>(row, n_in, a + t0, out + t0, r, L);
            else if (m == 3) row_nt<K, 3>(row, n_in, a + t0, out + t0, r, L);
            else if (m == 2) row_nt<K, 2>(row, n_in, a + t0, out + t0, r, L);
            else row_nt<K, 1>(row, n_in, a + t0, out + t0, r, L);
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
