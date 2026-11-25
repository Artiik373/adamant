#pragma once

// Unified MS (Mad Shift) quant type IDs and CPU/CUDA dispatch helpers for Amperite (.basalt).
// Type IDs 43..48 match ms-llama.cpp's ggml type IDs (MS_C1..MS_C6) to avoid collisions.

#include "kernels/ms_1mad_avx2.h"
#if defined(__CUDACC__)
#include "kernels/ms_1mad_cuda.cuh"
#endif

#include <cstdint>

namespace adamant::kernels::ms {

enum class MsQuantType : uint32_t {
    MS_C1 = 43,
    MS_C2 = 44,
    MS_C3 = 45,
    MS_C4 = 46,
    MS_C5 = 47,
    MS_C6 = 48,
};

inline int ms_quant_bits(MsQuantType t) noexcept {
    switch (t) {
        case MsQuantType::MS_C1: return 1;
        case MsQuantType::MS_C2: return 2;
        case MsQuantType::MS_C3: return 3;
        case MsQuantType::MS_C4: return 4;
        case MsQuantType::MS_C5: return 5;
        case MsQuantType::MS_C6: return 6;
    }
    return 0;
}

inline bool is_ms_quant_type(uint32_t type_id) noexcept {
    return type_id >= static_cast<uint32_t>(MsQuantType::MS_C1) &&
           type_id <= static_cast<uint32_t>(MsQuantType::MS_C6);
}

// CPU AVX2 dispatch for DDR5-resident MoE experts (e.g. i5-13600KF).
inline bool dispatch_ms_cpu_gemv(
    MsQuantType type,
    const uint8_t* __restrict weights,
    const float* __restrict x_rot,
    float* __restrict out,
    int out_features,
    int in_features
) noexcept {
    switch (type) {
        case MsQuantType::MS_C3:
            ms_1mad_gemv<3>(weights, x_rot, out, out_features, in_features);
            return true;
        case MsQuantType::MS_C4:
            ms_1mad_gemv<4>(weights, x_rot, out, out_features, in_features);
            return true;
        case MsQuantType::MS_C6:
            ms_1mad_gemv<6>(weights, x_rot, out, out_features, in_features);
            return true;
        default:
            return false;
    }
}

#if defined(__CUDACC__)
// Ampere (sm_86) CUDA dispatch for VRAM-resident dense spine & cached hot experts.
inline bool dispatch_ms_cuda_gemv(
    MsQuantType type,
    const uint8_t* weights,
    const float* x_rot,
    float* out,
    int out_features,
    int in_features,
    cudaStream_t stream = nullptr
) {
    switch (type) {
        case MsQuantType::MS_C3:
            ms_cuda::launch_ms_1mad_gemv<3>(weights, x_rot, out, out_features, in_features, stream);
            return true;
        case MsQuantType::MS_C4:
            ms_cuda::launch_ms_1mad_gemv<4>(weights, x_rot, out, out_features, in_features, stream);
            return true;
        case MsQuantType::MS_C6:
            ms_cuda::launch_ms_1mad_gemv<6>(weights, x_rot, out, out_features, in_features, stream);
            return true;
        default:
            return false;
    }
}
#endif

} // namespace adamant::kernels::ms
