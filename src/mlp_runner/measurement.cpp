//===- measurement.cpp - Implementations ------------------------*- C++ -*-===//

#include "measurement.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

#include "gemm_cpu.h"
#include "rapl.h"

namespace mlp_runner {

using clock_t_ = std::chrono::steady_clock;
using json = nlohmann::json;
namespace fs = std::filesystem;

static double _elapsed_us(clock_t_::time_point t0, clock_t_::time_point t1) {
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// ------------- Instrumented forward ----------------------------------------

InferenceSample instrumentedForward(
    const ExperimentConfig& cfg,
    const std::vector<LayerKernelEntry>& layers,
    const MlpWeights& weights,
    const std::vector<float>& input,
    Dispatcher& dispatcher,
    std::vector<float>& probs_out) {
  InferenceSample sample;
  sample.layers.reserve(layers.size());

  const int64_t e0_pkg = readRaplEnergyUj(kRaplPackagePath);
  const int64_t e0_core = readRaplEnergyUj(kRaplCorePath);
  const auto t0 = clock_t_::now();

  std::vector<float> act = input;
  for (size_t i = 0; i < layers.size(); ++i) {
    const auto& L = layers[i];
    std::vector<float> next(static_cast<size_t>(L.M) * L.N);

    const int64_t l0_pkg = readRaplEnergyUj(kRaplPackagePath);
    const int64_t l0_core = readRaplEnergyUj(kRaplCorePath);
    const auto lt0 = clock_t_::now();

    dispatcher.dispatch(L, act.data(), weights.W[i].data(), next.data());

    const auto lt1 = clock_t_::now();
    const int64_t l1_pkg = readRaplEnergyUj(kRaplPackagePath);
    const int64_t l1_core = readRaplEnergyUj(kRaplCorePath);

    PerLayerSample ps;
    ps.layer = L.layer;
    ps.time_us = _elapsed_us(lt0, lt1);
    ps.energy_uj_package = raplDelta(l1_pkg, l0_pkg);
    ps.energy_uj_core = raplDelta(l1_core, l0_core);
    sample.layers.push_back(std::move(ps));

    const bool is_last = (i + 1 == layers.size());
    if (!is_last) {
      const std::string& activ = cfg.activations.at(i);
      if (activ == "relu") relu(next.data(), static_cast<int>(next.size()));
      else throw std::runtime_error("unsupported activation: " + activ);
    }
    act = std::move(next);
  }

  // Post-process output (slice padded cols -> softmax) but do NOT include in
  // layer timing — it's a host-side formatting step, not an NPU dispatch.
  const int n_padded = layers.back().N;
  probs_out.assign(static_cast<size_t>(cfg.batch_size) * cfg.output_classes,
                   0.0f);
  sliceColumns(act.data(), probs_out.data(), cfg.batch_size, n_padded,
               cfg.output_classes);
  softmaxRowWise(probs_out.data(), cfg.batch_size, cfg.output_classes);

  const auto t1 = clock_t_::now();
  const int64_t e1_pkg = readRaplEnergyUj(kRaplPackagePath);
  const int64_t e1_core = readRaplEnergyUj(kRaplCorePath);
  sample.time_us = _elapsed_us(t0, t1);
  sample.energy_uj_package = raplDelta(e1_pkg, e0_pkg);
  sample.energy_uj_core = raplDelta(e1_core, e0_core);
  return sample;
}

// ------------- Run loop -----------------------------------------------------

RunStats runMeasurement(const ExperimentConfig& cfg,
                        const std::vector<LayerKernelEntry>& layers,
                        const MlpWeights& weights,
                        const std::vector<float>& input,
                        Dispatcher& dispatcher) {
  RunStats stats;
  stats.warmup_iterations = cfg.warmup_iterations;
  stats.outer_batches = cfg.outer_batches;
  stats.inner_target_seconds = cfg.inner_target_seconds;

  // 1. Idle baseline (median of N_IDLE_SAMPLES).
  stats.idle_baseline_mw = measureIdlePowerMw(kRaplPackagePath);

  // 2. Warmup.
  std::vector<float> scratch_probs;
  for (int w = 0; w < cfg.warmup_iterations; ++w) {
    (void)instrumentedForward(cfg, layers, weights, input, dispatcher,
                              scratch_probs);
  }

  // 3. Per-batch sweep. Inner loop runs until target wall reached.
  std::vector<double> batch_energies_per_inf;
  std::vector<double> batch_times;

  for (int b = 0; b < cfg.outer_batches; ++b) {
    BatchStats bs;
    bs.index = b;
    bs.idle_power_pre_mw = measureBracketIdlePowerMw(
        kRaplPackagePath, cfg.bracket_idle_count, cfg.bracket_idle_seconds);

    const int64_t batch_e0_pkg = readRaplEnergyUj(kRaplPackagePath);
    const int64_t batch_e0_core = readRaplEnergyUj(kRaplCorePath);
    const auto batch_t0 = clock_t_::now();

    std::vector<double> inner_times;
    std::map<std::string, std::vector<double>> layer_times;
    std::map<std::string, std::vector<int64_t>> layer_energies_pkg;

    int n_inner = 0;
    while (true) {
      InferenceSample s = instrumentedForward(cfg, layers, weights, input,
                                              dispatcher, scratch_probs);
      inner_times.push_back(s.time_us);
      for (const auto& ps : s.layers) {
        layer_times[ps.layer].push_back(ps.time_us);
        layer_energies_pkg[ps.layer].push_back(ps.energy_uj_package);
      }
      ++n_inner;
      double elapsed_s = std::chrono::duration<double>(
                             clock_t_::now() - batch_t0)
                             .count();
      if (elapsed_s >= cfg.inner_target_seconds) break;
    }

    const auto batch_t1 = clock_t_::now();
    const int64_t batch_e1_pkg = readRaplEnergyUj(kRaplPackagePath);
    const int64_t batch_e1_core = readRaplEnergyUj(kRaplCorePath);

    bs.idle_power_post_mw = measureBracketIdlePowerMw(
        kRaplPackagePath, cfg.bracket_idle_count, cfg.bracket_idle_seconds);

    bs.n_inner = n_inner;
    bs.wall_s = std::chrono::duration<double>(batch_t1 - batch_t0).count();
    bs.active_uj_package = raplDelta(batch_e1_pkg, batch_e0_pkg);
    bs.active_uj_core = raplDelta(batch_e1_core, batch_e0_core);
    bs.idle_power_mw = 0.5 * (bs.idle_power_pre_mw + bs.idle_power_post_mw);

    const double idle_uj =
        bs.idle_power_mw * bs.wall_s * 1000.0;  // mW * s * 1000 -> uJ
    bs.npu_uj_package = static_cast<int64_t>(
        std::max<double>(0.0, bs.active_uj_package - idle_uj));
    bs.npu_uj_core = bs.active_uj_core;  // core idle subtraction optional
    bs.energy_per_inference_uj = (n_inner > 0)
        ? static_cast<double>(bs.npu_uj_package) / n_inner
        : 0.0;

    if (!inner_times.empty()) {
      auto [mn, mx] = std::minmax_element(inner_times.begin(), inner_times.end());
      double sum = 0.0;
      for (double v : inner_times) sum += v;
      bs.model_time_us_min = *mn;
      bs.model_time_us_mean = sum / inner_times.size();
    }

    for (const auto& [layer, ts] : layer_times) {
      auto [mn, mx] = std::minmax_element(ts.begin(), ts.end());
      double sum = 0.0;
      for (double v : ts) sum += v;
      bs.layer_time_us_min[layer] = *mn;
      bs.layer_time_us_mean[layer] = sum / ts.size();

      auto& global_min = stats.layer_time_us_min_global[layer];
      if (global_min == 0.0 || *mn < global_min) global_min = *mn;
    }
    for (const auto& [layer, es] : layer_energies_pkg) {
      int64_t mn = *std::min_element(es.begin(), es.end());
      bs.layer_energy_uj_min[layer] = mn;
      auto& g = stats.layer_energy_uj_min_global[layer];
      if (g == 0 || mn < g) g = mn;
    }

    batch_energies_per_inf.push_back(bs.energy_per_inference_uj);
    batch_times.push_back(bs.wall_s);
    stats.batches.push_back(std::move(bs));
  }

  // 4. Post-experiment idle (drift check).
  stats.idle_post_mw = measureIdlePowerMw(kRaplPackagePath);

  auto energy_cv = computeMeanCv(batch_energies_per_inf);
  auto time_cv = computeMeanCv(batch_times);
  stats.batch_energy_mean_uj = energy_cv.mean;
  stats.batch_energy_cv_pct = energy_cv.cv_pct;
  stats.batch_time_mean_s = time_cv.mean;
  stats.batch_time_cv_pct = time_cv.cv_pct;

  if (!stats.batches.empty()) {
    stats.batch_min_energy_per_inference_uj = std::min_element(
        stats.batches.begin(), stats.batches.end(),
        [](const BatchStats& a, const BatchStats& b) {
          return a.energy_per_inference_uj < b.energy_per_inference_uj;
        })->energy_per_inference_uj;
    stats.batch_min_model_time_us = std::min_element(
        stats.batches.begin(), stats.batches.end(),
        [](const BatchStats& a, const BatchStats& b) {
          return a.model_time_us_min < b.model_time_us_min;
        })->model_time_us_min;
  }

  return stats;
}

// ------------- Output serialization ----------------------------------------

static std::string _headerCsv() {
  return "measurement_type,setter,backend,layer,batch_idx,n_inner,"
         "time_us_min,time_us_mean,"
         "energy_uj_min_package,wall_s,"
         "idle_power_pre_mw,idle_power_post_mw,"
         "batch_energy_cv_pct,batch_time_cv_pct";
}

void writeMeasurements(const std::string& output_dir,
                       const std::string& setter,
                       const std::string& backend,
                       const ExperimentConfig& cfg,
                       const std::vector<LayerKernelEntry>& layers,
                       const RunStats& stats) {
  (void)layers;  // reserved for Step 6-6 when we stamp xclbin hashes per row
  fs::create_directories(output_dir);

  const fs::path csv_path = fs::path(output_dir) / "measurements.csv";
  const bool csv_exists = fs::exists(csv_path);
  std::ofstream csv(csv_path, std::ios::app);
  if (!csv) throw std::runtime_error("cannot open " + csv_path.string());

  if (!csv_exists) csv << _headerCsv() << "\n";

  auto write_row = [&](const std::string& mtype, const std::string& layer,
                       int batch_idx, int n_inner,
                       double t_min, double t_mean, int64_t e_min,
                       double wall_s, double idle_pre, double idle_post,
                       double e_cv, double t_cv) {
    csv << mtype << "," << setter << "," << backend << ","
        << (layer.empty() ? "_" : layer) << "," << batch_idx << "," << n_inner
        << "," << std::fixed << std::setprecision(3) << t_min << "," << t_mean
        << "," << e_min << "," << std::setprecision(6) << wall_s << ","
        << std::setprecision(2) << idle_pre << "," << idle_post << ","
        << std::setprecision(3) << e_cv << "," << t_cv << "\n";
  };

  for (const auto& bs : stats.batches) {
    write_row("model", "", bs.index, bs.n_inner,
              bs.model_time_us_min, bs.model_time_us_mean, bs.npu_uj_package,
              bs.wall_s, bs.idle_power_pre_mw, bs.idle_power_post_mw,
              stats.batch_energy_cv_pct, stats.batch_time_cv_pct);
    for (const auto& [layer, t_min] : bs.layer_time_us_min) {
      double t_mean = bs.layer_time_us_mean.at(layer);
      int64_t e_min = bs.layer_energy_uj_min.at(layer);
      write_row("kernel", layer, bs.index, bs.n_inner,
                t_min, t_mean, e_min,
                bs.wall_s, bs.idle_power_pre_mw, bs.idle_power_post_mw,
                stats.batch_energy_cv_pct, stats.batch_time_cv_pct);
    }
  }
  csv.close();

  // JSON
  json doc;
  doc["setter"] = setter;
  doc["backend"] = backend;
  doc["config_path"] = cfg.config_path;
  doc["layer_sizes"] = cfg.layer_sizes;
  doc["batch_size"] = cfg.batch_size;
  doc["output_classes"] = cfg.output_classes;
  doc["warmup_iterations"] = stats.warmup_iterations;
  doc["outer_batches"] = stats.outer_batches;
  doc["inner_target_seconds"] = stats.inner_target_seconds;
  doc["idle_baseline_mw"] = stats.idle_baseline_mw;
  doc["idle_post_mw"] = stats.idle_post_mw;
  doc["batch_energy_cv_pct"] = stats.batch_energy_cv_pct;
  doc["batch_time_cv_pct"] = stats.batch_time_cv_pct;
  doc["batch_min_energy_per_inference_uj"] =
      stats.batch_min_energy_per_inference_uj;
  doc["batch_min_model_time_us"] = stats.batch_min_model_time_us;
  json layer_mins = json::object();
  for (const auto& [l, v] : stats.layer_time_us_min_global)
    layer_mins[l] = v;
  doc["layer_time_us_min_global"] = layer_mins;

  json batches = json::array();
  for (const auto& bs : stats.batches) {
    json b;
    b["index"] = bs.index;
    b["n_inner"] = bs.n_inner;
    b["wall_s"] = bs.wall_s;
    b["active_uj_package"] = bs.active_uj_package;
    b["idle_power_pre_mw"] = bs.idle_power_pre_mw;
    b["idle_power_post_mw"] = bs.idle_power_post_mw;
    b["npu_uj_package"] = bs.npu_uj_package;
    b["energy_per_inference_uj"] = bs.energy_per_inference_uj;
    b["model_time_us_min"] = bs.model_time_us_min;
    b["model_time_us_mean"] = bs.model_time_us_mean;
    json lmin = json::object();
    json lmean = json::object();
    for (const auto& [l, v] : bs.layer_time_us_min) lmin[l] = v;
    for (const auto& [l, v] : bs.layer_time_us_mean) lmean[l] = v;
    b["layer_time_us_min"] = lmin;
    b["layer_time_us_mean"] = lmean;
    batches.push_back(b);
  }
  doc["batches"] = batches;

  // measurements.json: one doc per (setter, backend) run. Append to an array
  // so multiple setters accumulate.
  const fs::path json_path = fs::path(output_dir) / "measurements.json";
  json root;
  if (fs::exists(json_path)) {
    std::ifstream ifs(json_path);
    try {
      ifs >> root;
    } catch (...) {
      root = json::object();
    }
  }
  if (!root.contains("runs") || !root["runs"].is_array())
    root["runs"] = json::array();
  root["runs"].push_back(doc);
  std::ofstream ofs(json_path);
  ofs << root.dump(2) << "\n";
}

}  // namespace mlp_runner
