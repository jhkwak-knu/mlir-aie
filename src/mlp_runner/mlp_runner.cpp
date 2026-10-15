//===- mlp_runner.cpp - Host entry point ------------------------*- C++ -*-===//
//
// Notion step 21, Steps 6-3 + 6-4:
//   * Parse experiment config + configurations.json
//   * Initialize random weights + input deterministically by seed
//   * Either:
//     - run a single forward pass (--mode forward, Step 6-3 sanity), or
//     - run the full batch-mode measurement protocol (--mode measure,
//       Step 6-4): warmup + outer_batches x inner-loop-to-target-wall,
//       emit measurements.csv / measurements.json.
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
#include "measurement.h"
#include "mlp.h"
#include "npu_dispatch.h"

using namespace mlp_runner;

static std::string joinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return std::string();
  return a.back() == '/' ? a + b : a + "/" + b;
}

static int runForward(const ExperimentConfig& cfg,
                      const std::vector<LayerKernelEntry>& layers,
                      const MlpWeights& weights,
                      const std::vector<float>& input,
                      Dispatcher& dispatcher,
                      const std::string& setter,
                      int n_samples) {
  std::vector<float> last_probs;
  for (int it = 0; it < n_samples; ++it)
    last_probs = forward(cfg, layers, weights, input, dispatcher);

  std::cout << "setter=" << setter << " backend=" << dispatcher.name()
            << " batch=" << cfg.batch_size << " layers=";
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
}

static int runMeasure(const ExperimentConfig& cfg,
                      const std::vector<LayerKernelEntry>& layers,
                      const MlpWeights& weights,
                      const std::vector<float>& input,
                      Dispatcher& dispatcher,
                      const std::string& setter,
                      const std::string& output_dir) {
  std::cout << "[measure] setter=" << setter
            << " backend=" << dispatcher.name()
            << " warmup=" << cfg.warmup_iterations
            << " outer_batches=" << cfg.outer_batches
            << " inner_target=" << cfg.inner_target_seconds << "s\n";

  auto stats = runMeasurement(cfg, layers, weights, input, dispatcher);

  writeMeasurements(output_dir, setter, dispatcher.name(), cfg, layers, stats);

  std::cout << "[measure] idle_baseline_mw=" << stats.idle_baseline_mw
            << " idle_post_mw=" << stats.idle_post_mw
            << " batches=" << stats.batches.size() << "\n";
  std::cout << "[measure] batch_energy_cv_pct="
            << stats.batch_energy_cv_pct
            << " batch_time_cv_pct=" << stats.batch_time_cv_pct << "\n";
  std::cout << "[measure] batch_min_model_time_us="
            << stats.batch_min_model_time_us
            << " batch_min_energy_per_inference_uj="
            << stats.batch_min_energy_per_inference_uj << "\n";
  return 0;
}

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
       "dispatcher backend: 'cpu' (reference) or 'npu' (XRT, stub until Step 6-6)",
       cxxopts::value<std::string>()->default_value("cpu"))
      ("mode",
       "'forward' for a single-pass sanity, 'measure' for the full batch-mode protocol",
       cxxopts::value<std::string>()->default_value("forward"))
      ("seed", "PRNG seed for weights + input",
       cxxopts::value<uint32_t>()->default_value("42"))
      ("n-samples",
       "number of forward passes to execute in --mode forward (>=1)",
       cxxopts::value<int>()->default_value("1"))
      ("output-dir",
       "where to write measurements.{csv,json} (defaults to <results_dir>/)",
       cxxopts::value<std::string>()->default_value(""))
      ("h,help", "print help");
  auto args = opts.parse(argc, argv);
  if (args.count("help") || !args.count("config") || !args.count("setter")) {
    std::cout << opts.help() << "\n";
    return args.count("help") ? 0 : 2;
  }

  const auto cfg = loadExperimentConfig(args["config"].as<std::string>());
  const std::string setter = args["setter"].as<std::string>();
  const std::string mode = args["mode"].as<std::string>();

  std::string configurations_path = args["configurations"].as<std::string>();
  if (configurations_path.empty())
    configurations_path = joinPath(cfg.results_dir, "configurations.json");

  std::string kernel_binaries_dir = args["kernel-binaries-dir"].as<std::string>();
  if (kernel_binaries_dir.empty())
    kernel_binaries_dir = joinPath(cfg.results_dir, "kernel_binaries");

  const auto layers =
      loadLayerKernels(configurations_path, setter, kernel_binaries_dir);

  const uint32_t seed = args["seed"].as<uint32_t>();
  const auto weights = initRandomWeights(layers, seed);
  const auto input =
      makeRandomInput(cfg.batch_size, cfg.layer_sizes.front(), seed ^ 0xA5A5u);

  auto dispatcher = makeDispatcher(args["backend"].as<std::string>());

  if (mode == "forward") {
    const int n_samples = std::max(1, args["n-samples"].as<int>());
    return runForward(cfg, layers, weights, input, *dispatcher, setter,
                      n_samples);
  }
  if (mode == "measure") {
    std::string output_dir = args["output-dir"].as<std::string>();
    if (output_dir.empty()) output_dir = cfg.results_dir;
    if (output_dir.empty()) {
      throw std::runtime_error(
          "--output-dir not set and config.output.results_dir is empty");
    }
    return runMeasure(cfg, layers, weights, input, *dispatcher, setter,
                      output_dir);
  }
  throw std::runtime_error("unknown --mode: " + mode);
} catch (const std::exception& e) {
  std::cerr << "mlp_runner: " << e.what() << "\n";
  return 1;
}
