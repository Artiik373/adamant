// src/models/inkling2/adapter.hpp - the Inkling 2 adapter's entry points.
#pragma once

#include <cstddef>
#include <vector>

namespace basalt::models::inkling2 {

/// The checkpoint's tokenizer.json sets ignore_merges: a piece that is a vocabulary entry is that one token (337 entries
/// cannot be built from the merges).
inline constexpr bool kIgnoreMerges = true;

}  // namespace basalt::models::inkling2
