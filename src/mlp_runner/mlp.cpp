//===- mlp.cpp - Forward-pass implementation --------------------*- C++ -*-===//

#include "mlp.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "gemm_cpu.h"

namespace mlp_runner {

static void _assertLayerChain(const ExperimentConfig& cfg,
                              const std::vector<LayerKernelEntry>& layers) {
  const int expected = static_cast<int>(cfg.layer_sizes.size()) - 1;
  if (static_cast<int>(layers.size()) != expected) {
    throw std::runtime_error("forward: layer count mismatch between "
                             "ExperimentConfig and LayerKernelEntry list");
  }
  for (int i = 0; i < expected; ++i) {
    if (layers[i].M != cfg.batch_size)
      throw std::runtime_error("forward: layer M must equal batch_size");
    if (layers[i].K != cfg.layer_sizes[i])
      throw std::runtime_error("forward: layer K must equal layer_sizes[i]");
    if (layers[i].N != cfg.layer_sizes[i + 1])
      throw std::runtime_error("forward: layer N must equal layer_sizes[i+1]");
  }
}

MlpWeights initRandomWeights(const std::vector<LayerKernelEntry>& layers,
                             uint32_t seed) {
  std::mt19937 rng(seed);
  MlpWeights out;
  out.W.reserve(layers.size());
  for (const auto& L : layers) {
    const float scale = std::sqrt(6.0f / static_cast<float>(L.K + L.N));
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> w(static_cast<size_t>(L.K) * L.N);
    for (auto& v : w) v = dist(rng);
    out.W.push_back(std::move(w));
  }
  return out;
}

std::vector<float> makeRandomInput(int batch, int input_dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(static_cast<size_t>(batch) * input_dim);
  for (auto& v : x) v = dist(rng);
  return x;
}

std::vector<float> forward(const ExperimentConfig& cfg,
                           const std::vector<LayerKernelEntry>& layers,
                           const MlpWeights& weights,
                           const std::vector<float>& input,
                           Dispatcher& dispatcher) {
  _assertLayerChain(cfg, layers);
  if (weights.W.size() != layers.size())
    throw std::runtime_error("forward: weights.W count must match layers");

  std::vector<float> act = input;  // current activation [batch, K_current]
  for (size_t i = 0; i < layers.size(); ++i) {
    const auto& L = layers[i];
    std::vector<float> next(static_cast<size_t>(L.M) * L.N);

    dispatcher.dispatch(L, act.data(), weights.W[i].data(), next.data());

    const bool is_last = (i + 1 == layers.size());
    if (!is_last) {
      const std::string& activ = cfg.activations.at(i);
      if (activ == "relu") {
        relu(next.data(), static_cast<int>(next.size()));
      } else {
        throw std::runtime_error("forward: unsupported activation '" + activ + "'");
      }
    }
    act = std::move(next);
  }

  // Post-process the output: slice padded columns -> softmax.
  const int n_padded = layers.back().N;
  const int n_classes = cfg.output_classes;
  std::vector<float> probs(static_cast<size_t>(cfg.batch_size) * n_classes);
  sliceColumns(act.data(), probs.data(), cfg.batch_size, n_padded, n_classes);
  softmaxRowWise(probs.data(), cfg.batch_size, n_classes);
  return probs;
}

}  // namespace mlp_runner
