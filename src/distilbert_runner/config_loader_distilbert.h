//===- config_loader_distilbert.h - DistilBERT experiment config -*- C++ -*-===//
//
// Notion task #22 §4. Parses experiments/configs/distilbert_*.json into a
// flat struct used by main.cpp / model.cpp / measurement so the hot path
// never touches JSON. We deliberately avoid reusing mlp_runner::Experiment
// Config because the MLP fields (layer_sizes / activations / output_classes)
// have no DistilBERT analogue and DistilBERT brings its own knobs
// (num_layers / hidden_dim / ffn_dim / num_heads / sequence_length).
//
// LayerKernelEntry / loadLayerKernels are reused from mlp_runner: the
// configurations.json schema is model-agnostic — a list of (shape, setter,
// tc_entry, xclbin) records keyed by `shape.layer`. distilbert config
// generation labels each shape with a gemm_type ("attention_qkv",
// "ffn_expand", ...) so the existing mlp_runner::loadLayerKernels(setter)
// returns the entries we need for the body's 48 GEMM dispatches.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <vector>

namespace distilbert_runner {

/// Canonical DistilBERT experiment config. Mirrors Notion #22 §4 plus the
/// measurement knobs and setter list (shared schema with the MLP run).
struct ExperimentConfig {
  // model.*
  std::string pretrained_source;            // e.g. "distilbert-base-uncased-finetuned-sst-2-english"
  int num_layers = 0;                       // typically 6
  int hidden_dim = 0;                       // typically 768
  int ffn_dim = 0;                          // typically 3072
  int num_heads = 0;                        // typically 12
  int sequence_length = 0;                  // typically 128
  std::string multi_head_strategy;          // typically "batched"

  // dataset.*
  std::string dataset_name;                 // typically "sst2"
  int batch_size = 0;                       // typically 1
  int num_inference_samples = 0;            // typically 100

  // measurement.*
  int warmup_iterations = 0;
  int outer_batches = 0;
  double inner_target_seconds = 0.0;
  int bracket_idle_count = 0;
  double bracket_idle_seconds = 0.0;

  // setters / output
  std::vector<std::string> setters;
  std::string results_dir;                  // absolute or relative
  std::string config_path;                  // bookkeeping
};

/// Parse a DistilBERT experiment config JSON. Throws on schema errors.
ExperimentConfig loadExperimentConfig(const std::string& path);

}  // namespace distilbert_runner
