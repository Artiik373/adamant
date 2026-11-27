// include/basalt/program/pretok_o200k.hpp - the o200k pre-tokenizer (OpenAI's o200k split): the text cut into the pieces BPE
// merges, each piece's end offset into `ends`.
#pragma once

#include <cstddef>
#include <vector>

namespace basalt::program::tokenizer {

void pretok_o200k(const char* text, size_t size, std::vector<size_t>& ends);

}  // namespace basalt::program::tokenizer
