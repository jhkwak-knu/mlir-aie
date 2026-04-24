//===- config_loader.cpp - Implementation -----------------------*- C++ -*-===//

#include "config_loader.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace mlp_runner {

using json = nlohmann::json;
namespace fs = std::filesystem;

static json _readJson(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("cannot open: " + path);
  json doc;
  ifs >> doc;
  return doc;
}

static int _outputClassesFromModel(const json& model,
                                   const std::vector<int>& layer_sizes) {
  if (model.contains("output_classes")) {
    int oc = model.at("output_classes").get<int>();
    if (oc <= 0 || oc > layer_sizes.back())
      throw std::runtime_error("model.output_classes must be in (0, output_dim]");
    return oc;
  }
  return layer_sizes.back();
}

ExperimentConfig loadExperimentConfig(const std::string& path) {
  json doc = _readJson(path);

  ExperimentConfig cfg;
  cfg.config_path = path;

  const auto& model = doc.at("model");
  if (model.value("type", "") != "mlp")
    throw std::runtime_error("config.model.type must be 'mlp'");

  cfg.layer_sizes = model.at("layer_sizes").get<std::vector<int>>();
  if (cfg.layer_sizes.size() < 2)
    throw std::runtime_error("config.model.layer_sizes must have at least 2 entries");

  cfg.activations = model.value("activations", std::vector<std::string>{});
  if (static_cast<int>(cfg.activations.size()) !=
      static_cast<int>(cfg.layer_sizes.size()) - 2) {
    throw std::runtime_error(
        "config.model.activations length must equal layer_sizes.size() - 2");
  }

  cfg.output_classes = _outputClassesFromModel(model, cfg.layer_sizes);

  const auto& dataset = doc.at("dataset");
  cfg.batch_size = dataset.at("batch_size").get<int>();
  if (cfg.batch_size <= 0)
    throw std::runtime_error("config.dataset.batch_size must be positive");

  const auto& meas = doc.at("measurement");
  cfg.warmup_iterations    = meas.value("warmup_iterations", 10);
  cfg.outer_batches        = meas.value("outer_batches", 5);
  cfg.inner_target_seconds = meas.value("inner_target_seconds", 0.5);
  cfg.bracket_idle_count   = meas.value("bracket_idle_count", 5);
  cfg.bracket_idle_seconds = meas.value("bracket_idle_seconds", 0.3);

  cfg.setters = doc.at("setters").get<std::vector<std::string>>();
  if (cfg.setters.empty())
    throw std::runtime_error("config.setters must be non-empty");

  cfg.results_dir = doc.value("/output/results_dir"_json_pointer, std::string{});
  return cfg;
}

std::vector<LayerKernelEntry> loadLayerKernels(
    const std::string& configurations_path,
    const std::string& setter,
    const std::string& kernel_binaries_dir) {
  json doc = _readJson(configurations_path);
  const auto& entries = doc.at("configurations");

  std::vector<LayerKernelEntry> picked;
  for (const auto& e : entries) {
    const std::string& e_setter = e.at("setter").get_ref<const std::string&>();
    if (e_setter != setter) continue;
    if (e.contains("error")) {
      throw std::runtime_error(
          "configurations.json: entry for setter '" + setter + "' layer '" +
          e.at("shape").at("layer").get<std::string>() +
          "' has generate-time error: " + e.at("error").get<std::string>());
    }

    LayerKernelEntry k;
    k.layer  = e.at("shape").at("layer").get<std::string>();
    k.setter = setter;
    k.M      = e.at("shape").at("M").get<int>();
    k.K      = e.at("shape").at("K").get<int>();
    k.N      = e.at("shape").at("N").get<int>();

    const auto& cfg = e.at("config");
    k.num_cores = cfg.at("P").get<int>();

    fs::path dir = fs::path(kernel_binaries_dir) / (k.layer + "_" + k.setter);
    k.xclbin_path = (dir / "final.xclbin").string();
    k.insts_path  = (dir / "insts.bin").string();
    picked.push_back(std::move(k));
  }

  if (picked.empty()) {
    throw std::runtime_error(
        "configurations.json: no entries matched setter '" + setter + "'");
  }

  // Layer name ordering (fc1, fc2, ...) — lexicographic on the numeric suffix.
  std::sort(picked.begin(), picked.end(),
            [](const LayerKernelEntry& a, const LayerKernelEntry& b) {
              // "fc" + int; strip prefix.
              auto suffix = [](const std::string& s) {
                return std::stoi(s.substr(2));
              };
              return suffix(a.layer) < suffix(b.layer);
            });
  return picked;
}

}  // namespace mlp_runner
