//===- config_loader_distilbert.cpp - DistilBERT config JSON parser ---------===//

#include "config_loader_distilbert.h"

#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace distilbert_runner {

using json = nlohmann::json;

static int _require_positive_int(const json& obj, const char* key,
                                 const char* where) {
  if (!obj.contains(key) || !obj[key].is_number_integer()) {
    throw std::runtime_error(std::string(where) + "." + key +
                             " must be an integer");
  }
  int v = obj[key].get<int>();
  if (v <= 0) {
    throw std::runtime_error(std::string(where) + "." + key +
                             " must be > 0");
  }
  return v;
}

ExperimentConfig loadExperimentConfig(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("cannot open config: " + path);
  }
  json doc = json::parse(f);

  if (!doc.contains("model") || !doc["model"].is_object()) {
    throw std::runtime_error(path + ": missing 'model' object");
  }
  const auto& m = doc["model"];
  const std::string mtype = m.value("type", "");
  // distilbert_runner is a misnomer at this point — the same encoder graph
  // covers any standard BERT-base architecture (DistilBERT, BERT, prajjwal1
  // compact BERTs). Accept either label.
  if (mtype != "distilbert" && mtype != "bert") {
    throw std::runtime_error(path + ": model.type must be 'distilbert' or "
                             "'bert' (got '" + mtype + "')");
  }

  if (!doc.contains("dataset") || !doc["dataset"].is_object()) {
    throw std::runtime_error(path + ": missing 'dataset' object");
  }
  const auto& ds = doc["dataset"];

  if (!doc.contains("measurement") || !doc["measurement"].is_object()) {
    throw std::runtime_error(path + ": missing 'measurement' object");
  }
  const auto& meas = doc["measurement"];

  ExperimentConfig cfg;
  cfg.config_path = path;

  cfg.pretrained_source     = m.value("pretrained_source", "");
  cfg.num_layers            = _require_positive_int(m, "num_layers", "model");
  cfg.hidden_dim            = _require_positive_int(m, "hidden_dim", "model");
  cfg.ffn_dim               = _require_positive_int(m, "ffn_dim", "model");
  cfg.num_heads             = _require_positive_int(m, "num_heads", "model");
  cfg.sequence_length       = _require_positive_int(m, "sequence_length", "model");
  cfg.multi_head_strategy   = m.value("multi_head_strategy", "batched");
  if (cfg.hidden_dim % cfg.num_heads != 0) {
    throw std::runtime_error(
        "model.hidden_dim must be divisible by model.num_heads");
  }

  cfg.dataset_name           = ds.value("name", "sst2");
  cfg.batch_size             = _require_positive_int(ds, "batch_size", "dataset");
  cfg.num_inference_samples  = ds.value("num_inference_samples", 100);

  cfg.warmup_iterations      = meas.value("warmup_iterations", 10);
  cfg.outer_batches          = meas.value("outer_batches", 5);
  cfg.inner_target_seconds   = meas.value("inner_target_seconds", 1.0);
  cfg.bracket_idle_count     = meas.value("bracket_idle_count", 5);
  cfg.bracket_idle_seconds   = meas.value("bracket_idle_seconds", 0.3);

  if (doc.contains("setters") && doc["setters"].is_array()) {
    for (const auto& s : doc["setters"]) cfg.setters.push_back(s.get<std::string>());
  }
  if (cfg.setters.empty()) {
    throw std::runtime_error(path + ": 'setters' must be a non-empty list");
  }

  if (doc.contains("output") && doc["output"].is_object()) {
    cfg.results_dir = doc["output"].value("results_dir", "");
  }
  return cfg;
}

}  // namespace distilbert_runner
