#pragma once

// Multi-threaded AVX2 MoE SwiGLU FFN dispatcher for DDR5-offloaded experts (i5-13600KF).
// Parallelizes across the active routed experts (e.g. top-10) using std::thread
// (or OpenMP when compiled with -fopenmp) to saturate dual-channel DDR5 bandwidth.

#include "kernels/ms_1mad_avx2.h"
#include "kernels/ms_dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace adamant::kernels::ms {

struct MsExpertView {
    const uint8_t* gate_w;
    const uint8_t* up_w;
    const uint8_t* down_w;
    const int8_t*  down_signs; // length = moe_intermediate_size (640)
    float          router_weight;
    MsQuantType    quant_type;
};

// Execute one routed expert FFN: y += router_weight * (W_down (R_down^T (silu(W_gate x_rot) * (W_up x_rot))))
inline void run_single_ms_expert(
    const MsExpertView& ex,
    const float* __restrict x_rot,   // already rotated with R_in^T (length = hidden_size = 2560)
    float* __restrict scratch_gate,  // length = inter_size (640)
    float* __restrict scratch_up,    // length = inter_size (640)
    float* __restrict scratch_mid,   // length = inter_size (640)
    float* __restrict scratch_out,   // length = hidden_size (2560)
    int hidden_size,
    int inter_size,
    int down_hadamard_block
) noexcept {
    dispatch_ms_cpu_gemv(ex.quant_type, ex.gate_w, x_rot, scratch_gate, inter_size, hidden_size);
    dispatch_ms_cpu_gemv(ex.quant_type, ex.up_w,   x_rot, scratch_up,   inter_size, hidden_size);

    for (int i = 0; i < inter_size; ++i) {
        float g = scratch_gate[i];
        float silu = g / (1.0f + std::exp(-g));
        scratch_gate[i] = silu * scratch_up[i];
    }

    fwht_rotate_input(scratch_gate, ex.down_signs, scratch_mid, inter_size, down_hadamard_block);
    dispatch_ms_cpu_gemv(ex.quant_type, ex.down_w, scratch_mid, scratch_out, hidden_size, inter_size);
}

// Parallel top-K MoE routed FFN over DDR5-resident experts.
inline void run_ms_moe_layer_cpu(
    const MsExpertView* experts,
    int num_active,
    const float* __restrict x_in,
    const int8_t* __restrict in_signs,
    float* __restrict y_out,
    int hidden_size,
    int inter_size,
    int in_hadamard_block = 512,
    int down_hadamard_block = 128,
    int num_threads = 0
) {
    std::vector<float> x_rot(hidden_size);
    fwht_rotate_input(x_in, in_signs, x_rot.data(), hidden_size, in_hadamard_block);

    if (num_threads <= 0) {
        num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    num_threads = std::min(num_threads, num_active);

    std::vector<std::vector<float>> partial_outs(num_active, std::vector<float>(hidden_size, 0.0f));

    auto worker = [&](int tid) {
        std::vector<float> s_gate(inter_size);
        std::vector<float> s_up(inter_size);
        std::vector<float> s_mid(inter_size);
        for (int idx = tid; idx < num_active; idx += num_threads) {
            run_single_ms_expert(
                experts[idx],
                x_rot.data(),
                s_gate.data(),
                s_up.data(),
                s_mid.data(),
                partial_outs[idx].data(),
                hidden_size,
                inter_size,
                down_hadamard_block
            );
        }
    };

    if (num_threads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& th : threads) th.join();
    }

    std::fill(y_out, y_out + hidden_size, 0.0f);
    for (int idx = 0; idx < num_active; ++idx) {
        const float w = experts[idx].router_weight;
        const float* p = partial_outs[idx].data();
        for (int i = 0; i < hidden_size; ++i) {
            y_out[i] += w * p[i];
        }
    }
}

} // namespace adamant::kernels::ms
