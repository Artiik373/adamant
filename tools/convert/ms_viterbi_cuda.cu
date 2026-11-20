#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {

constexpr uint32_t MAD_MUL = 34038481u;
constexpr uint32_t MAD_ADD = 76625530u;
constexpr int STATE_BITS = 14;
constexpr int NUM_STATES = 1 << STATE_BITS;
constexpr int STATE_MASK = NUM_STATES - 1;

__device__ __forceinline__ float decode_1mad(int s) {
    uint32_t x = static_cast<uint32_t>(s) * MAD_MUL + MAD_ADD;
    int v = static_cast<int>((x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu)) - 510;
    return static_cast<float>(v);
}

// Step t == 0: initialize costs from either uniform 0 (first window) or prev_state
__global__ void viterbi_step0_kernel(
    const float* __restrict__ w_col,      // [rc]
    const float* __restrict__ scales,     // [rc]
    const int32_t* __restrict__ prev_s,   // [rc] or nullptr
    float piv,
    int rc,
    int k,
    float* __restrict__ costs_out,        // [rc, NUM_STATES]
    int16_t* __restrict__ backptr0        // [rc, NUM_STATES]
) {
    int r = blockIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rc || s >= NUM_STATES) return;

    float d = scales[r];
    float x = w_col[r];
    float v = decode_1mad(s);
    float branch = piv * (d * d * v * v - 2.0f * x * d * v);

    int idx = r * NUM_STATES + s;
    if (prev_s == nullptr) {
        costs_out[idx] = branch;
        backptr0[idx] = 0;
    } else {
        int ps = prev_s[r];
        int base = (ps << k) & STATE_MASK;
        int mask = ~((1 << k) - 1);
        costs_out[idx] = ((s & mask) == base) ? branch : INFINITY;
        backptr0[idx] = static_cast<int16_t>(ps);
    }
}

// Step t > 0: one thread per (row, j) where j in [0, n_rem)
__global__ void viterbi_step_kernel(
    const float* __restrict__ costs_in,   // [rc, NUM_STATES]
    const float* __restrict__ w_col,      // [rc]
    const float* __restrict__ scales,     // [rc]
    float piv,
    int rc,
    int k,
    float* __restrict__ costs_out,        // [rc, NUM_STATES]
    int16_t* __restrict__ backptr_t       // [rc, NUM_STATES]
) {
    int r = blockIdx.y;
    int shift_r = STATE_BITS - k;
    int n_pred = 1 << k;
    int n_rem = 1 << shift_r;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rc || j >= n_rem) return;

    const float* row_in = costs_in + r * NUM_STATES;
    float min_c = INFINITY;
    int best_a = 0;

    #pragma unroll 4
    for (int a = 0; a < n_pred; ++a) {
        int pred = (a << shift_r) | j;
        float c = row_in[pred];
        if (c < min_c) {
            min_c = c;
            best_a = a;
        }
    }

    int16_t best_pred = static_cast<int16_t>((best_a << shift_r) | j);
    float d = scales[r];
    float d2 = d * d;
    float two_xd = 2.0f * w_col[r] * d;

    float* row_out = costs_out + r * NUM_STATES;
    int16_t* bp_out = backptr_t + r * NUM_STATES;
    int s_base = j << k;

    #pragma unroll 4
    for (int b = 0; b < n_pred; ++b) {
        int s_next = s_base | b;
        float v = decode_1mad(s_next);
        float branch = piv * (d2 * v * v - two_xd * v);
        row_out[s_next] = min_c + branch;
        bp_out[s_next] = best_pred;
    }
}

// Coalesced 256-thread block per row: parallel argmin across 16,384 states + backtrack
__global__ void viterbi_backtrack_kernel(
    const float* __restrict__ costs_final,  // [rc, NUM_STATES]
    const int16_t* __restrict__ backptr,    // [w_len, rc, NUM_STATES]
    int rc,
    int w_len,
    int step_commit,
    int32_t* __restrict__ out_states        // [rc, step_commit]
) {
    int r = blockIdx.x;
    if (r >= rc) return;

    int tid = threadIdx.x;
    const float* row_c = costs_final + static_cast<size_t>(r) * NUM_STATES;

    float best_c = INFINITY;
    int best_s = 0;
    for (int s = tid; s < NUM_STATES; s += 256) {
        float c = row_c[s];
        if (c < best_c) {
            best_c = c;
            best_s = s;
        }
    }

    __shared__ float smem_c[256];
    __shared__ int smem_s[256];
    smem_c[tid] = best_c;
    smem_s[tid] = best_s;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (smem_c[tid + stride] < smem_c[tid]) {
                smem_c[tid] = smem_c[tid + stride];
                smem_s[tid] = smem_s[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int cur = smem_s[0];
        for (int t = w_len - 1; t >= 0; --t) {
            if (t < step_commit) {
                out_states[r * step_commit + t] = cur;
            }
            cur = static_cast<int>(backptr[(static_cast<size_t>(t) * rc + r) * NUM_STATES + cur]);
        }
    }
}

}  // namespace

torch::Tensor viterbi_window_cuda(
    torch::Tensor w_win,        // [rc, w_len] float32 contiguous
    torch::Tensor scales,       // [rc] float32 contiguous
    torch::Tensor inv_piv2,     // [w_len] float32 CPU or CUDA
    c10::optional<torch::Tensor> prev_state,  // [rc] int32 or None
    int k,
    int step_commit,
    torch::Tensor costs_a,      // [rc, NUM_STATES] float32
    torch::Tensor costs_b,      // [rc, NUM_STATES] float32
    torch::Tensor backptr       // [w_len, rc, NUM_STATES] int16
) {
    int rc = static_cast<int>(w_win.size(0));
    int w_len = static_cast<int>(w_win.size(1));
    auto inv_piv2_cpu = inv_piv2.to(torch::kCPU).contiguous();
    const float* piv_ptr = inv_piv2_cpu.data_ptr<float>();

    auto w_t = w_win.transpose(0, 1).contiguous();  // [w_len, rc]
    const float* scales_ptr = scales.data_ptr<float>();
    const int32_t* prev_ptr = prev_state.has_value() ? prev_state.value().data_ptr<int32_t>() : nullptr;

    float* c_in = costs_a.data_ptr<float>();
    float* c_out = costs_b.data_ptr<float>();
    int16_t* bp_base = backptr.data_ptr<int16_t>();

    {
        dim3 block(256);
        dim3 grid((NUM_STATES + 255) / 256, rc);
        viterbi_step0_kernel<<<grid, block>>>(
            w_t.data_ptr<float>(),
            scales_ptr,
            prev_ptr,
            piv_ptr[0],
            rc,
            k,
            c_in,
            bp_base
        );
    }

    int n_rem = 1 << (STATE_BITS - k);
    dim3 block_step(256);
    dim3 grid_step((n_rem + 255) / 256, rc);
    for (int t = 1; t < w_len; ++t) {
        viterbi_step_kernel<<<grid_step, block_step>>>(
            c_in,
            w_t.data_ptr<float>() + t * rc,
            scales_ptr,
            piv_ptr[t],
            rc,
            k,
            c_out,
            bp_base + static_cast<size_t>(t) * rc * NUM_STATES
        );
        std::swap(c_in, c_out);
    }

    auto out_states = torch::empty({rc, step_commit}, torch::dtype(torch::kInt32).device(w_win.device()));
    viterbi_backtrack_kernel<<<rc, 256>>>(
        c_in,
        bp_base,
        rc,
        w_len,
        step_commit,
        out_states.data_ptr<int32_t>()
    );
    return out_states;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("viterbi_window_cuda", &viterbi_window_cuda, "Fused 14-bit 1MAD Viterbi window DP (CUDA)");
}
