//===- config_loader.h - Experiment / configurations.json loader -*- C++ -*-===//
//
// Parses the MLP E2E experiment config (Notion step 21) plus the
// per-(layer, setter) tiling configurations emitted by generate_configs.py.
// The loader materializes flat, POD-ish structs for the forward-pass
// runtime so hot paths do not touch JSON.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlp_runner {

/// Canonical experiment config: model shape, measurement knobs, output dir.
struct ExperimentConfig {
  std::vector<int> layer_sizes;           // e.g. {784, 512, 512, 16}
  std::vector<std::string> activations;   // e.g. {"relu", "relu"} for hidden gaps
  int output_classes = 0;                 // pre-padding class count (10 for MNIST)
  int batch_size = 0;
  int warmup_iterations = 0;
  int outer_batches = 0;
  double inner_target_seconds = 0.0;
  int bracket_idle_count = 0;
  double bracket_idle_seconds = 0.0;
  std::vector<std::string> setters;       // requested setter list
  std::string results_dir;                // absolute or relative path
  std::string config_path;                // where this ExperimentConfig was loaded from
};

/// Per-layer tiling + binary locations for a single setter.
struct LayerKernelEntry {
  std::string layer;                      // "fc1" / "fc2" / "fc3"
  std::string setter;                     // "star_map" / "max_p" / ...
  int M = 0;
  int K = 0;
  int N = 0;
  int num_cores = 0;                      // P
  std::string xclbin_path;                // kernel_binaries/<layer>_<setter>/final.xclbin
  std::string insts_path;                 // ..../insts.bin
};

/// Load the experiment config JSON; validates top-level structure.
ExperimentConfig loadExperimentConfig(const std::string& path);

/// Load configurations.json and return the subset entries for `setter`,
/// in forward order (fc1, fc2, ...). `kernel_binaries_dir` is used to
/// resolve xclbin / insts paths. Throws if any layer's entry is missing
/// or carries a generate-time error.
std::vector<LayerKernelEntry> loadLayerKernels(
    const std::string& configurations_path,
    const std::string& setter,
    const std::string& kernel_binaries_dir);

}  // namespace mlp_runner
