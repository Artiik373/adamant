// include/basalt/core/cuda_cc.hpp - the compute capability this build's kernels are compiled for, by name.
//
// include/basalt/cuda_target.hpp decides which capability that is; every "is this card ours" check reads these names,
// so no other file spells the number.
#pragma once

#include <string>

#include "basalt/cuda_target.hpp"

namespace basalt::core {

inline constexpr int kCudaCCMajor = 8;
inline constexpr int kCudaCCMinor = 6;
inline constexpr int kCudaCC = kCudaCCMajor * 10 + kCudaCCMinor;
static_assert(basalt::cuda_target::supports(kCudaCCMajor, kCudaCCMinor));

/// Whether a card of compute capability major.minor runs this build's kernels.
inline constexpr bool cuda_cc_is_ours(int major, int minor) { return basalt::cuda_target::supports(major, minor); }

/// "sm_86", for messages.
inline std::string cuda_cc_name() { return "sm_" + std::to_string(kCudaCC); }
/// "8.6", for messages.
inline std::string cuda_cc_dotted() { return std::to_string(kCudaCCMajor) + "." + std::to_string(kCudaCCMinor); }

}  // namespace basalt::core
