#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace {

// overridable to probe other 1MAD constants; the format uses these defaults
#ifndef MS_MAD_MUL
#define MS_MAD_MUL 34038481u
#define MS_MAD_ADD 76625530u
#endif
constexpr uint32_t MAD_MUL = MS_MAD_MUL;
constexpr uint32_t MAD_ADD = MS_MAD_ADD;
constexpr int STATE_BITS = 14;
constexpr int NUM_STATES = 1 << STATE_BITS;
constexpr int STATE_MASK = NUM_STATES - 1;

__device__ __forceinline__ float decode_1mad(int s) {
    uint32_t x = static_cast<uint32_t>(s) * MAD_MUL + MAD_ADD;
    int v = static_cast<int>((x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu)) - 510;
#ifdef MS_FP4_C
    // probe: the value snapped to the nearest E2M1 level times MS_FP4_C
    const float a = fabsf(static_cast<float>(v)) / MS_FP4_C;
    const float l = a < 0.25f ? 0.f : a < 0.75f ? 0.5f : a < 1.25f ? 1.f : a < 1.75f ? 1.5f
                  : a < 2.5f ? 2.f : a < 3.5f ? 3.f : a < 5.f ? 4.f : 6.f;
    return copysignf(l * MS_FP4_C, static_cast<float>(v));
#else
    return static_cast<float>(v);
#endif
}

// n consecutive elements written as one aligned vector store where n fits one
template <int N, typename E>
__device__ __forceinline__ void store_run(E* dst, const E (&x)[N]) {
    constexpr int bytes = N * static_cast<int>(sizeof(E));
    if constexpr (bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16) {
        using V = std::conditional_t<bytes == 2, uint16_t, std::conditional_t<bytes == 4, uint32_t,
                  std::conditional_t<bytes == 8, uint2, uint4>>>;
        V u;
        memcpy(&u, x, bytes);
        *reinterpret_cast<V*>(dst) = u;
    } else {
        #pragma unroll
        for (int i = 0; i < N; ++i) dst[i] = x[i];
    }
}

// One block per row runs the whole window DP. Thread tid owns the states of its remainders j = s mod 2^(14-k), with
// every predecessor of each, so costs live in registers and only the per-j minimum crosses threads (double-buffered: one barrier per step). All states with the same j share one
// predecessor, so the back-pointer is stored once per j as the predecessor's top-k bits.
template <int PT, int K>
__global__ void __launch_bounds__(NUM_STATES / PT) viterbi_window_kernel(
    const float* __restrict__ w_win,      // [rc, w_len]
    const float* __restrict__ scales,     // [rc]
    const float* __restrict__ inv_piv2,   // [n_groups, w_len]
    int rows_per_group,
    const int32_t* __restrict__ prev_s,   // [rc] or nullptr
    int w_len,
    int step_commit,
    uint8_t* __restrict__ backptr,        // [rc, w_len, n_rem]
    int32_t* __restrict__ out_states      // [rc, step_commit]
) {
    constexpr int T = NUM_STATES / PT;
    extern __shared__ float min_c[];      // [2, n_rem]
    __shared__ float red_c[T];
    __shared__ int red_s[T];

    constexpr int k = K;
    constexpr int shift_r = STATE_BITS - K;
    constexpr int n_rem = 1 << shift_r;
    constexpr int n_sub = PT >> K;        // remainders owned per thread
    static_assert(PT >= (1 << K), "a thread must own all predecessors of its remainders");
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const float d = scales[r];
    const float d2 = d * d;
    const float* w_row = w_win + static_cast<size_t>(r) * w_len;
    const float* piv_row = inv_piv2 + static_cast<size_t>(r / rows_per_group) * w_len;
    uint8_t* bp_row = backptr + static_cast<size_t>(r) * w_len * n_rem;

    // state m = a * n_sub + sub is s = (a << shift_r) | j with j = j0 + sub: a thread owns n_sub consecutive
    // remainders, so their minima go out as one vector store and, when n_sub <= 2^k, one loaded minimum serves
    // the n_sub states of each predecessor top a
    const int j0 = tid * n_sub;
    float v[PT], c[PT];
    #pragma unroll
    for (int m = 0; m < PT; ++m) v[m] = decode_1mad(((m / n_sub) << shift_r) | (j0 + m % n_sub));

    {
        const float a = piv_row[0] * d2;
        const float b = -piv_row[0] * 2.0f * w_row[0] * d;
        const int mask = ~((1 << k) - 1);
        const int base = prev_s ? ((prev_s[r] << k) & STATE_MASK) : 0;
        #pragma unroll
        for (int m = 0; m < PT; ++m) {
            int s = ((m / n_sub) << shift_r) | (j0 + m % n_sub);
            float branch = v[m] * (a * v[m] + b);
            c[m] = (prev_s == nullptr || (s & mask) == base) ? branch : INFINITY;
        }
    }

    for (int t = 1; t < w_len; ++t) {
        float* mc = min_c + (t & 1) * n_rem;
        uint8_t* bp_t = bp_row + static_cast<size_t>(t) * n_rem;
        float best[n_sub];
        uint8_t best_a[n_sub];
        #pragma unroll
        for (int sub = 0; sub < n_sub; ++sub) {
            // pairwise tree over the 2^k predecessors; the right side wins only when strictly smaller, so ties
            // keep the lowest index
            constexpr int A = 1 << K;
            float tv[A];
            int ta[A];
            #pragma unroll
            for (int a = 0; a < A; ++a) { tv[a] = c[a * n_sub + sub]; ta[a] = a; }
            #pragma unroll
            for (int w = 1; w < A; w *= 2) {
                #pragma unroll
                for (int a = 0; a + w < A; a += 2 * w) {
                    if (tv[a + w] < tv[a]) { tv[a] = tv[a + w]; ta[a] = ta[a + w]; }
                }
            }
            best[sub] = tv[0];
            best_a[sub] = static_cast<uint8_t>(ta[0]);
        }
        store_run<n_sub>(mc + j0, best);
        store_run<n_sub>(bp_t + j0, best_a);
        __syncthreads();
        const float a = piv_row[t] * d2;
        const float b = -piv_row[t] * 2.0f * w_row[t] * d;
        #pragma unroll
        for (int ap = 0; ap < (1 << K); ++ap) {
            constexpr int per_load = n_sub <= (1 << K) ? n_sub : (1 << K);
            #pragma unroll
            for (int sub0 = 0; sub0 < n_sub; sub0 += per_load) {
                const float prev = mc[(ap << (shift_r - k)) | ((j0 + sub0) >> k)];
                #pragma unroll
                for (int sub = sub0; sub < sub0 + per_load; ++sub) {
                    const int m = ap * n_sub + sub;
                    c[m] = prev + v[m] * (a * v[m] + b);
                }
            }
        }
    }

    float best_c = INFINITY;
    int best_s = 0;
    #pragma unroll
    for (int m = 0; m < PT; ++m) {   // m runs in increasing s, so the first strict minimum is the lowest index
        if (c[m] < best_c) { best_c = c[m]; best_s = ((m / n_sub) << shift_r) | (j0 + m % n_sub); }
    }
    red_c[tid] = best_c;
    red_s[tid] = best_s;
    __syncthreads();
    for (int stride = T / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float oc = red_c[tid + stride];
            int os = red_s[tid + stride];
            if (oc < red_c[tid] || (oc == red_c[tid] && os < red_s[tid])) {
                red_c[tid] = oc;
                red_s[tid] = os;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int cur = red_s[0];
        for (int t = w_len - 1; t >= 0; --t) {
            if (t < step_commit) out_states[r * step_commit + t] = cur;
            if (t > 0) {
                int j = cur >> k;
                cur = (static_cast<int>(bp_row[static_cast<size_t>(t) * n_rem + j]) << shift_r) | j;
            }
        }
    }
}

}  // namespace

torch::Tensor viterbi_window_cuda(
    torch::Tensor w_win,        // [rc, w_len] float32 contiguous
    torch::Tensor scales,       // [rc] float32 contiguous
    torch::Tensor inv_piv2,     // [n_groups, w_len] float32 CUDA contiguous; rows split evenly into groups
    c10::optional<torch::Tensor> prev_state,  // [rc] int32 or None
    int64_t k,
    int64_t step_commit,
    torch::Tensor backptr       // [>= rc * w_len * n_rem] uint8
) {
    TORCH_CHECK(k >= 1 && k <= 6, "k must be in [1, 6]");
    TORCH_CHECK(w_win.is_contiguous() && inv_piv2.is_cuda(), "bad inputs");
    int rc = static_cast<int>(w_win.size(0));
    int w_len = static_cast<int>(w_win.size(1));
    int n_rem = 1 << (STATE_BITS - k);
    TORCH_CHECK(inv_piv2.numel() % w_len == 0 && rc % (inv_piv2.numel() / w_len) == 0, "bad inv_piv2 groups");
    int rows_per_group = rc / static_cast<int>(inv_piv2.numel() / w_len);
    TORCH_CHECK(backptr.numel() >= static_cast<int64_t>(rc) * w_len * n_rem, "backptr too small");

    size_t smem = 2 * static_cast<size_t>(n_rem) * sizeof(float);
    auto out_states = torch::empty({rc, step_commit}, torch::dtype(torch::kInt32).device(w_win.device()));
    auto launch = [&](auto kern, int threads) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        kern<<<rc, threads, smem>>>(
            w_win.data_ptr<float>(),
            scales.data_ptr<float>(),
            inv_piv2.data_ptr<float>(),
            rows_per_group,
            prev_state.has_value() ? prev_state.value().data_ptr<int32_t>() : nullptr,
            w_len,
            static_cast<int>(step_commit),
            backptr.data_ptr<uint8_t>(),
            out_states.data_ptr<int32_t>()
        );
    };
    switch (k) {
        case 1: launch(viterbi_window_kernel<16, 1>, NUM_STATES / 16); break;
        case 2: launch(viterbi_window_kernel<32, 2>, NUM_STATES / 32); break;
        case 3: launch(viterbi_window_kernel<32, 3>, NUM_STATES / 32); break;
        case 4: launch(viterbi_window_kernel<32, 4>, NUM_STATES / 32); break;
        case 5: launch(viterbi_window_kernel<64, 5>, NUM_STATES / 64); break;
        default: launch(viterbi_window_kernel<64, 6>, NUM_STATES / 64); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out_states;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("viterbi_window_cuda", &viterbi_window_cuda, "Fused 14-bit 1MAD Viterbi window DP (CUDA)");
}
