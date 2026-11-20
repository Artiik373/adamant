// Pre-conversion DDR5 tok/s micro-benchmark for Qwen3.8-Flash-Next MS MoE (48 layers, top-10 active).
// Uses persistent MsCpuMoEExecutor (zero per-layer thread spawn / malloc overhead).
// Build: g++ -O3 -march=native -std=c++20 -pthread -Isrc tests/bench_ms_1mad.cpp -o build/bench_ms_1mad
// Run:   ./build/bench_ms_1mad [num_threads]

#include "kernels/ms_moe_cpu.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

using namespace adamant::kernels::ms;

int main(int argc, char** argv) {
    constexpr int kHidden = 2560;
    constexpr int kInter  = 640;
    constexpr int kActive = 10;
    constexpr int kPool   = 32;
    constexpr int kLayersC3 = 37;
    constexpr int kLayersC4 = 11;
    constexpr int kIters = 10;

    int threads = (argc > 1) ? std::atoi(argv[1]) : 10;
    if (threads <= 0) threads = static_cast<int>(std::thread::hardware_concurrency());

    auto alloc_pool = [&](int k_bits) {
        size_t gate_bytes = static_cast<size_t>(kInter) * ms_row_bytes(kHidden, k_bits);
        size_t down_bytes = static_cast<size_t>(kHidden) * ms_row_bytes(kInter, k_bits);
        size_t per_expert = 2 * gate_bytes + down_bytes;
        std::vector<uint8_t> buf(kPool * per_expert, 0x5Au);
        return std::make_tuple(std::move(buf), gate_bytes, down_bytes, per_expert);
    };

    auto [pool_c3, g3, d3, pe3] = alloc_pool(3);
    auto [pool_c4, g4, d4, pe4] = alloc_pool(4);

    std::vector<float> x_in(kHidden, 0.25f), y_out(kHidden, 0.0f);
    std::vector<int8_t> in_signs(kHidden, 1), down_signs(kInter, 1);

    std::vector<MsExpertView> active_c3(kActive), active_c4(kActive);
    for (int i = 0; i < kActive; ++i) {
        size_t off3 = static_cast<size_t>((i * 3) % kPool) * pe3;
        active_c3[i] = {pool_c3.data() + off3, pool_c3.data() + off3 + g3,
                        pool_c3.data() + off3 + 2 * g3, down_signs.data(), 0.1f, MsQuantType::MS_C3};
        size_t off4 = static_cast<size_t>((i * 3) % kPool) * pe4;
        active_c4[i] = {pool_c4.data() + off4, pool_c4.data() + off4 + g4,
                        pool_c4.data() + off4 + 2 * g4, down_signs.data(), 0.1f, MsQuantType::MS_C4};
    }

    MsCpuMoEExecutor exec(threads, kHidden, kInter, kActive);

    // Warmup
    exec.run_layer(active_c3.data(), kActive, x_in.data(), in_signs.data(), y_out.data());

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < kIters; ++it) {
        for (int l = 0; l < kLayersC3; ++l) {
            exec.run_layer(active_c3.data(), kActive, x_in.data(), in_signs.data(), y_out.data());
        }
        for (int l = 0; l < kLayersC4; ++l) {
            exec.run_layer(active_c4.data(), kActive, x_in.data(), in_signs.data(), y_out.data());
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ms_per_tok = (sec / kIters) * 1000.0;
    double tok_per_s = kIters / sec;
    double bytes_per_tok = static_cast<double>(kLayersC3) * kActive * pe3 +
                           static_cast<double>(kLayersC4) * kActive * pe4;
    double gb_per_s = (bytes_per_tok * kIters) / (sec * 1e9);

    std::printf("Threads: %d | Routed MoE bytes/tok: %.2f MB\n", threads, bytes_per_tok / 1e6);
    std::printf("Projected 48-layer CPU MoE decode: %.2f ms/tok (%.1f tok/s) | Effective BW: %.1f GB/s\n",
                ms_per_tok, tok_per_s, gb_per_s);
    return 0;
}
