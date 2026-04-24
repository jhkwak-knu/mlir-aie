//===- mlp.h - Forward-pass orchestration for the padded MLP ----*- C++ -*-===//
//
// The MLP is executed one fully-connected layer at a time via the Dispatcher:
//
//   fc1: input [batch, K1] x W1 [K1, N1] -> H1 [batch, N1];   ReLU
//   fc2: H1    [batch, K2] x W2 [K2, N2] -> H2 [batch, N2];   ReLU
//   fc3: H2    [batch, K3] x W3 [K3, N3] -> logits_raw [batch, N3_padded]
//
// The final column count N3_padded matches the NPU-facing padded shape
// (16 for our MNIST config). After fc3 we slice the first
// ExperimentConfig::output_classes columns and run a host-side softmax, so
// the produced probability vector still has dimension 10 for MNIST.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "config_loader.h"
#include "npu_dispatch.h"

namespace mlp_runner {

/// Host-visible weights. One entry per GEMM layer, in forward order.
/// Weights are row-major [K, N]; biases currently unused (omitted) to match
/// the generate_configs tc_entry which carries no bias term.
struct MlpWeights {
  std::vector<std::vector<float>> W;  // W[layer] is a flat [K, N] buffer
};

/// Deterministic Xavier-ish init (uniform in [-scale, scale] with
/// scale = sqrt(6 / (fan_in + fan_out))) using std::mt19937 seeded by `seed`.
MlpWeights initRandomWeights(const std::vector<LayerKernelEntry>& layers,
                             uint32_t seed);

/// Seeded random input of shape [batch, layer_sizes[0]] in [0, 1).
std::vector<float> makeRandomInput(int batch, int input_dim, uint32_t seed);

/// Forward pass. Returns the softmaxed [batch, output_classes] buffer.
/// Intermediate activations are owned internally and freed on return.
std::vector<float> forward(const ExperimentConfig& cfg,
                           const std::vector<LayerKernelEntry>& layers,
                           const MlpWeights& weights,
                           const std::vector<float>& input,
                           Dispatcher& dispatcher);

}  // namespace mlp_runner
