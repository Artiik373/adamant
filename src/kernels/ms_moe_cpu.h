#pragma once

// Spin-barrier persistent-thread-pool AVX2 MoE SwiGLU FFN dispatcher for DDR5-offloaded experts.
// Eliminates Windows std::condition_variable wakeup latency (~2 ms/layer) across 48 layers/token.

#include "kernels/ms_1mad_avx2.h"
#include "kernels/ms_dispatch.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "basalt/platform/compiler.hpp"

namespace adamant::kernels::ms {

struct MsExpertView {
    const uint8_t* gate_w;
    const uint8_t* up_w;
    const uint8_t* down_w;
    const int8_t*  down_signs; // length = moe_intermediate_size (640)
    float          router_weight;
    MsQuantType    quant_type;
};

inline void cpu_pause() noexcept { basalt::platform::cpu_relax(); }

inline void run_single_ms_expert(
    const MsExpertView& ex,
    const float* __restrict x_rot,
    float* __restrict scratch_gate,
    float* __restrict scratch_up,
    float* __restrict scratch_mid,
    float* __restrict scratch_out,
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

// Low-latency spin-barrier worker pool with pre-allocated per-thread scratch buffers.
class MsCpuMoEExecutor {
public:
    MsCpuMoEExecutor(int num_threads, int hidden_size, int inter_size, int max_active = 16)
        : num_threads_(std::max(1, num_threads)),
          hidden_size_(hidden_size),
          inter_size_(inter_size),
          x_rot_(hidden_size),
          partial_outs_(max_active, std::vector<float>(hidden_size, 0.0f)),
          s_gate_(num_threads_, std::vector<float>(inter_size)),
          s_up_(num_threads_, std::vector<float>(inter_size)),
          s_mid_(num_threads_, std::vector<float>(inter_size)) {
        workers_.reserve(num_threads_ - 1);
        for (int t = 1; t < num_threads_; ++t) {
            workers_.emplace_back([this, t]() { worker_loop(t); });
        }
    }

    ~MsCpuMoEExecutor() {
        stop_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& w : workers_) w.join();
    }

    void run_layer(
        const MsExpertView* experts,
        int num_active,
        const float* __restrict x_in,
        const int8_t* __restrict in_signs,
        float* __restrict y_out,
        int in_hadamard_block = 512,
        int down_hadamard_block = 128
    ) {
        fwht_rotate_input(x_in, in_signs, x_rot_.data(), hidden_size_, in_hadamard_block);

        if (num_threads_ == 1) {
            run_slice(0, experts, num_active, down_hadamard_block);
        } else {
            cur_experts_ = experts;
            cur_active_ = num_active;
            cur_down_blk_ = down_hadamard_block;
            remaining_.store(num_threads_ - 1, std::memory_order_release);
            gen_.fetch_add(1, std::memory_order_release);

            run_slice(0, experts, num_active, down_hadamard_block);
            while (remaining_.load(std::memory_order_acquire) > 0) {
                cpu_pause();
            }
        }

        std::fill(y_out, y_out + hidden_size_, 0.0f);
        for (int idx = 0; idx < num_active; ++idx) {
            const float w = experts[idx].router_weight;
            const float* p = partial_outs_[idx].data();
            for (int i = 0; i < hidden_size_; ++i) {
                y_out[i] += w * p[i];
            }
        }
    }

private:
    void run_slice(int tid, const MsExpertView* experts, int num_active, int down_blk) noexcept {
        for (int idx = tid; idx < num_active; idx += num_threads_) {
            run_single_ms_expert(
                experts[idx],
                x_rot_.data(),
                s_gate_[tid].data(),
                s_up_[tid].data(),
                s_mid_[tid].data(),
                partial_outs_[idx].data(),
                hidden_size_,
                inter_size_,
                down_blk
            );
        }
    }

    void worker_loop(int tid) {
        uint64_t seen = 0;
        while (true) {
            while (gen_.load(std::memory_order_acquire) == seen) {
                cpu_pause();
            }
            seen = gen_.load(std::memory_order_relaxed);
            if (stop_.load(std::memory_order_acquire)) return;

            run_slice(tid, cur_experts_, cur_active_, cur_down_blk_);
            remaining_.fetch_sub(1, std::memory_order_release);
        }
    }

    int num_threads_;
    int hidden_size_;
    int inter_size_;
    std::vector<float> x_rot_;
    std::vector<std::vector<float>> partial_outs_;
    std::vector<std::vector<float>> s_gate_;
    std::vector<std::vector<float>> s_up_;
    std::vector<std::vector<float>> s_mid_;

    std::vector<std::thread> workers_;
    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int> remaining_{0};
    alignas(64) std::atomic<bool> stop_{false};
    const MsExpertView* cur_experts_ = nullptr;
    int cur_active_ = 0;
    int cur_down_blk_ = 128;
};

} // namespace adamant::kernels::ms
