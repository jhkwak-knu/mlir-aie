//===- mlp_runner.cpp - Host entry point ------------------------*- C++ -*-===//
//
// Notion step 21, Step 6-3 scope:
//   * Parse experiment config + configurations.json
//   * Initialize random weights + input deterministically by seed
//   * Run one forward pass via the CPU dispatcher (NPU via stub; Step 6-6
//     replaces it)
//   * Emit a one-line summary + per-sample argmax prediction to stdout
//
// Measurement instrumentation (warmup / outer_batches / RAPL) is deferred to
// Step 6-4.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "cxxopts.hpp"

#include "config_loader.h"
#include "mlp.h"
#include "npu_dispatch.h"

using namespace mlp_runner;

int main(int argc, char** argv) try {
  cxxopts::Options opts("mlp_runner",
                        "MLP End-to-End EDP runner (Notion step 21)");
  opts.add_options()
      ("config", "experiment config.json",
       cxxopts::value<std::string>())
      ("configurations",
       "configurations.json (defaults to <results_dir>/configurations.json)",
       cxxopts::value<std::string>()->default_value(""))
      ("kernel-binaries-dir",
       "root of per-(layer, setter) xclbin cache "
       "(defaults to <results_dir>/kernel_binaries)",
       cxxopts::value<std::string>()->default_value(""))
      ("setter",
       "which setter's kernels to consume (star_map / max_p / charm_cdse / timeloop)",
       cxxopts::value<std::string>())
      ("backend",
       "dispatcher backend: 'cpu' (reference) or 'npu' (XRT, stub in Step 6-3)",
       cxxopts::value<std::string>()->default_value("cpu"))
      ("seed", "PRNG seed for weights + input",
       cxxopts::value<uint32_t>()->default_value("42"))
      ("n-samples",
       "how many forward passes to execute (ignored in Step 6-3 except for "
       "loop-safety; always >=1)",
       cxxopts::value<int>()->default_value("1"))
      ("h,help", "print help");
  auto args = opts.parse(argc, argv);
  if (args.count("help") || !args.count("config") || !args.count("setter")) {
    std::cout << opts.help() << "\n";
    return args.count("help") ? 0 : 2;
  }

  const auto cfg = loadExperimentConfig(args["config"].as<std::string>());
  const std::string setter = args["setter"].as<std::string>();

  auto join = [](const std::string& a, const std::string& b) {
    if (a.empty()) return std::string();
    return a.back() == '/' ? a + b : a + "/" + b;
  };
  std::string configurations_path =
      args["configurations"].as<std::string>();
  if (configurations_path.empty())
    configurations_path = join(cfg.results_dir, "configurations.json");

  std::string kernel_binaries_dir =
      args["kernel-binaries-dir"].as<std::string>();
  if (kernel_binaries_dir.empty())
    kernel_binaries_dir = join(cfg.results_dir, "kernel_binaries");

  const auto layers =
      loadLayerKernels(configurations_path, setter, kernel_binaries_dir);
  const uint32_t seed = args["seed"].as<uint32_t>();
  const int n_samples = std::max(1, args["n-samples"].as<int>());

  const auto weights = initRandomWeights(layers, seed);
  const auto input =
      makeRandomInput(cfg.batch_size, cfg.layer_sizes.front(), seed ^ 0xA5A5u);

  auto dispatcher = makeDispatcher(args["backend"].as<std::string>());

  std::vector<float> last_probs;
  for (int it = 0; it < n_samples; ++it) {
    last_probs = forward(cfg, layers, weights, input, *dispatcher);
  }

  // Summary line + per-sample argmax to confirm end-to-end wiring.
  std::cout << "setter=" << setter
            << " backend=" << dispatcher->name()
            << " batch=" << cfg.batch_size
            << " layers=";
  for (size_t i = 0; i < cfg.layer_sizes.size(); ++i) {
    if (i) std::cout << "-";
    std::cout << cfg.layer_sizes[i];
  }
  std::cout << " output_classes=" << cfg.output_classes
            << " iterations=" << n_samples << "\n";

  for (int b = 0; b < cfg.batch_size; ++b) {
    const float* row = last_probs.data() + b * cfg.output_classes;
    int argmax = static_cast<int>(
        std::max_element(row, row + cfg.output_classes) - row);
    float row_sum = std::accumulate(row, row + cfg.output_classes, 0.0f);
    std::cout << "sample=" << b << " argmax=" << argmax
              << " prob_sum=" << row_sum << "\n";
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "mlp_runner: " << e.what() << "\n";
  return 1;
}
