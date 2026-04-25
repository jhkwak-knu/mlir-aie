//===- measurement.h - Batch-mode measurement orchestrator ------*- C++ -*-===//
//
// Implements the measurement protocol approved for Notion step 21 Step 6-4:
//
//   warmup_iterations x full model inference  (not measured)
//   for batch in 1..outer_batches:
//     pre-bracket idle bracket
//     inner loop: run full model inferences until target_wall_s reached,
//                 timestamp + RAPL per dispatch + per inference
//     post-bracket idle bracket
//     record (wall_s, active_uj, idle_power, per-layer stats)
//   compute CV across outer_batches
//
// One inner iteration == one full model inference (not a per-layer dispatch)
// so cross-batch CV reflects inference-level variability.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "config_loader.h"
#include "mlp.h"
#include "npu_dispatch.h"

namespace mlp_runner {

/// Optional richer metadata for a per-layer sample. Populated by runners that
/// know the ggml call's role (DistilBERT step 4-4); mlp_runner leaves the
/// defaults so existing fc1/fc2/fc3 rows stay schema-equivalent.
struct LayerMeta {
  std::string gemm_type = "";   // e.g. "attention_qkv", "ffn_expand"
  int layer_idx = -1;           // 0..num_layers-1, -1 when N/A
  std::string sub_type = "";    // e.g. "Q"/"K"/"V" within attention_qkv
};

/// Per-layer timing sample within a single inference.
struct PerLayerSample {
  std::string layer;
  LayerMeta meta;
  double time_us;
  int64_t energy_uj_package;
  int64_t energy_uj_core;
};

/// Per-inference (whole-model) sample.
struct InferenceSample {
  double time_us;
  int64_t energy_uj_package;
  int64_t energy_uj_core;
  std::vector<PerLayerSample> layers;
};

/// Summary of a single outer batch.
struct BatchStats {
  int index = 0;
  int n_inner = 0;                   // inferences executed in the batch
  double wall_s = 0.0;
  int64_t active_uj_package = 0;     // total package energy over wall_s
  int64_t active_uj_core = 0;
  double idle_power_mw = 0.0;        // mean(pre,post) bracket idle
  double idle_power_pre_mw = 0.0;
  double idle_power_post_mw = 0.0;
  int64_t npu_uj_package = 0;        // active - idle_power * wall
  int64_t npu_uj_core = 0;
  double energy_per_inference_uj = 0.0;
  double model_time_us_min = 0.0;
  double model_time_us_mean = 0.0;
  std::map<std::string, double> layer_time_us_min;
  std::map<std::string, double> layer_time_us_mean;
  // Per-batch sum of per-inference RAPL deltas for this layer. Used instead of
  // min because short layers (fc3 ~680us) can sit below the RAPL update tick
  // (~1ms) and return delta=0 on individual inferences; summing across
  // n_inner iterations always captures multiple ticks and remains positive.
  std::map<std::string, int64_t> layer_energy_uj_sum;
  // Optional richer per-layer metadata (gemm_type / layer_idx / sub_type).
  // Keyed by the same `layer` string used by the maps above. Empty/default
  // when the runner does not provide it (e.g. mlp_runner's fc1/fc2/fc3).
  std::map<std::string, LayerMeta> layer_meta;
};

/// Top-level run stats collected across outer_batches.
struct RunStats {
  int warmup_iterations = 0;
  int outer_batches = 0;
  double inner_target_seconds = 0.0;
  double idle_baseline_mw = 0.0;     // pre-experiment median
  double idle_post_mw = 0.0;         // post-experiment drift check
  std::vector<BatchStats> batches;

  // Aggregates
  double batch_energy_cv_pct = 0.0;  // across batches
  double batch_time_cv_pct = 0.0;
  double batch_energy_mean_uj = 0.0;
  double batch_time_mean_s = 0.0;
  double batch_min_energy_per_inference_uj = 0.0;
  double batch_min_model_time_us = 0.0;

  // Global per-layer min across all inner iterations in all batches.
  std::map<std::string, double> layer_time_us_min_global;
  // Global per-layer sum-of-deltas minimum across batches (batch with the
  // lowest cumulative layer energy — robust to RAPL tick quantization).
  std::map<std::string, int64_t> layer_energy_uj_sum_min_global;
};

/// One instrumented forward pass: records per-layer + whole-inference
/// timestamps and RAPL deltas. Output probabilities are stored in `probs`.
InferenceSample instrumentedForward(const ExperimentConfig& cfg,
                                    const std::vector<LayerKernelEntry>& layers,
                                    const MlpWeights& weights,
                                    const std::vector<float>& input,
                                    Dispatcher& dispatcher,
                                    std::vector<float>& probs);

/// Run the full measurement protocol. Returns complete RunStats; mlp_runner
/// can later serialize these to measurements.{csv,json}.
RunStats runMeasurement(const ExperimentConfig& cfg,
                        const std::vector<LayerKernelEntry>& layers,
                        const MlpWeights& weights,
                        const std::vector<float>& input,
                        Dispatcher& dispatcher);

/// Emit measurements.csv + measurements.json under `output_dir`. `setter` is
/// stamped into every row so multiple setter runs can concatenate into one CSV.
void writeMeasurements(const std::string& output_dir,
                       const std::string& setter,
                       const std::string& backend,
                       const ExperimentConfig& cfg,
                       const std::vector<LayerKernelEntry>& layers,
                       const RunStats& stats);

}  // namespace mlp_runner
