// include/basalt/models/hy4/geometry.hpp - the fixed geometry of Hy4-flash (HYV4FlashForCausalLM).
#pragma once

#include <cstdint>

namespace basalt::models::hy4 {

constexpr int kLayers = 42;
constexpr int kEmbd = 3072;
constexpr int kVocab = 120832;
constexpr int kHc = 4;                       ///< residual streams
constexpr int kHcFlat = kHc * kEmbd;         ///< 12288: a token's streams as one row
constexpr int kHcMixes = 2 * kHc;            ///< pre (4) + post (4) logits of a layer's mix
constexpr float kHcMagnitude = 2.0f;
constexpr float kHcEps = 1e-6f;
constexpr float kRmsEps = 1e-5f;

// attention: MLA in every layer; sparse layers (with the compressed indexer) and sliding-window layers
constexpr int kQRank = 1024;
constexpr int kKvRank = 512;                 ///< the latent: keys and values of the absorbed form
constexpr int kRope = 64;                    ///< rotated dims of q and of the shared key, GPT-J pairs
constexpr int kNope = 192;
constexpr int kQkHead = kNope + kRope;       ///< 256
constexpr int kVHead = 256;
constexpr int kKeyDim = kKvRank + kRope;     ///< 576: the absorbed key row
constexpr int kHeadsSparse = 32;
constexpr int kHeadsSliding = 64;
constexpr int kMaxHeads = kHeadsSliding;
constexpr int kWindow = 512;                 ///< sliding layers attend positions (p - 511 .. p)
constexpr float kRopeThetaSliding = 10000.0f;
constexpr float kRopeThetaSparse = 10000000.0f;
constexpr int kMaxT = 8;                     ///< rows of a decode window
constexpr int kMaxP = 2048;                  ///< rows of a prompt window: the routed experts run on all of them at once
constexpr int kPassRows = 128;               ///< rows of a prompt window the trunk runs at a time

// compressed indexer of the sparse layers
constexpr int kIdxHeads = 16;
constexpr int kIdxDim = 128;
constexpr int kIdxRatio = 4;                 ///< token keys pooled per group
constexpr int kIdxGroupTopk = 512;           ///< groups a query reads (its own included)
constexpr int kIdxTopk = 2048;               ///< up to this many keys a sparse layer attends all of them

// FFN: layer 0 dense, the rest LatentMoE
constexpr int kDenseFf = 13824;
constexpr int kLatent = 1536;                ///< the routed experts' width
constexpr int kExpertFf = 1536;
constexpr int kExperts = 512;
constexpr int kExpertsUsed = 16;
constexpr float kRoutedScale = 3.998f;
constexpr float kSwigluLimit = 10.0f;
constexpr int kSharedFf = 1536;

// Engram: before the blocks of these (0-based) layers
constexpr int kEngramLayers = 4;
constexpr int kEngramLayer[kEngramLayers] = {1, 11, 21, 31};
constexpr int kEngramHeads = 16;             ///< 2-grams and 3-grams x 8 heads
constexpr int kEngramHeadDim = 96;
constexpr int kEngramDim = kEngramHeads * kEngramHeadDim;   ///< 1536
constexpr int kEngramMaxNgram = 3;

// DFly2, the block drafter: draft layer i attends over target layer kDflyTarget[i]'s latent cache, at the positions
// that layer's indexer picked for the last accepted token, and over its own block
constexpr int kDflyLayers = 5;
constexpr int kDflyBlock = 6;                ///< the anchor and kDflyBlock - 1 mask slots, one draft each
constexpr int kDflyTarget[kDflyLayers] = {25, 29, 33, 37, 41};
constexpr int kDflyFf = 9216;
constexpr int kDflyHeads = 32;
constexpr int32_t kDflyMaskToken = 120009;
constexpr int kDflySelStride = kIdxGroupTopk * kIdxRatio + kDflyBlock;

/// Whether layer `l` is a sliding-window layer (else a sparse one), from the published swa_pattern.
constexpr bool sliding_layer(int l) { return l >= 2 && (l - 1) % 4 != 0; }
constexpr int layer_heads(int l) { return sliding_layer(l) ? kHeadsSliding : kHeadsSparse; }

}  // namespace basalt::models::hy4
