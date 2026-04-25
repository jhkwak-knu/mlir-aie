//===- main.cpp - DistilBERT runner host entry ----------------------*- C++ -*-===//
//
// Notion task #22 §4-3. Mirrors mlp_runner.cpp's CLI shape so measure.py /
// run_experiment.sh can invoke either runner uniformly:
//
//   distilbert_runner --config <experiment.json>
//                     --gguf   <distilbert-sst2-f16.gguf>
//                     --setter <star_map | max_p | charm_cdse | timeloop>
//                     --backend <cpu | npu>
//                     --mode <forward | measure>
//                     [--n-samples N]
//                     [--seed S]
//                     [--output-dir DIR]
//
// Step 4-3-C-1 (this file's first incarnation):
//   * Parse the experiment config and a `--gguf` path
//   * bert_load_from_file the GGUF
//   * Tokenize a fixed probe sentence so we can sanity-check the
//     vocabulary / max-tokens hookup before adding the forward pass.
//
// The full encoder forward + classifier head + measurement protocol arrive
// in 4-3-C-2 / 4-3-D.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"

#include "config_loader_distilbert.h"
#include "model.h"

// bert.cpp public API
#include "bert.h"

namespace {

constexpr const char* kDefaultProbe = "Hello world";

}  // namespace

int main(int argc, char** argv) try {
  cxxopts::Options opts(
      "distilbert_runner",
      "DistilBERT End-to-End EDP runner (Notion task #22 §4-3)"
  );
  opts.add_options()
      ("config", "experiment config.json",
       cxxopts::value<std::string>())
      ("gguf",   "DistilBERT GGUF weights (with classifier head)",
       cxxopts::value<std::string>())
      ("setter", "star_map / max_p / charm_cdse / timeloop",
       cxxopts::value<std::string>())
      ("backend", "dispatcher backend: 'cpu' or 'npu'",
       cxxopts::value<std::string>()->default_value("cpu"))
      ("mode", "'forward' for sanity, 'measure' for the full protocol",
       cxxopts::value<std::string>()->default_value("forward"))
      ("seed", "PRNG seed",
       cxxopts::value<uint32_t>()->default_value("42"))
      ("n-samples",
       "number of inferences in --mode forward (>=1)",
       cxxopts::value<int>()->default_value("1"))
      ("output-dir",
       "where to write measurements.{csv,json} (defaults to "
       "<results_dir>)",
       cxxopts::value<std::string>()->default_value(""))
      ("probe",
       "probe sentence used by --mode forward (defaults to 'Hello world')",
       cxxopts::value<std::string>()->default_value(kDefaultProbe))
      ("h,help", "print help");
  auto args = opts.parse(argc, argv);
  if (args.count("help") || !args.count("config") || !args.count("gguf")
      || !args.count("setter")) {
    std::cout << opts.help() << "\n";
    return args.count("help") ? 0 : 2;
  }

  const auto cfg = distilbert_runner::loadExperimentConfig(
      args["config"].as<std::string>());
  const std::string gguf_path = args["gguf"].as<std::string>();
  const std::string setter    = args["setter"].as<std::string>();
  const std::string backend   = args["backend"].as<std::string>();
  const std::string mode      = args["mode"].as<std::string>();

  std::cout << "[distilbert_runner] config=" << cfg.config_path
            << " gguf=" << gguf_path
            << " setter=" << setter
            << " backend=" << backend
            << " mode=" << mode << "\n"
            << "  num_layers=" << cfg.num_layers
            << " hidden_dim=" << cfg.hidden_dim
            << " ffn_dim=" << cfg.ffn_dim
            << " num_heads=" << cfg.num_heads
            << " sequence_length=" << cfg.sequence_length
            << " batch_size=" << cfg.batch_size << "\n";

  // Load GGUF. `use_cpu = (backend != "npu")` lets ggml-cuda / ggml-metal
  // fall back to CPU when our backend hook isn't installed yet.
  const bool use_cpu = (backend != "npu");
  bert_ctx* bctx = bert_load_from_file(gguf_path.c_str(), use_cpu);
  if (!bctx) {
    std::cerr << "bert_load_from_file failed: " << gguf_path << "\n";
    return 1;
  }
  const int32_t n_embd = bert_n_embd(bctx);
  const int32_t n_max_tokens = bert_n_max_tokens(bctx);
  std::cout << "  bert: n_embd=" << n_embd
            << " n_max_tokens=" << n_max_tokens << "\n";

  // Allocate compute buffers for batch_size = 1 (Notion §4 fixes this).
  // Keep n_max_tokens at the GGUF default so probe sentences shorter than
  // sequence_length don't get truncated even if the user passed something
  // unusual.
  bert_allocate_buffers(bctx, n_max_tokens, /*batch_size=*/1);

  if (mode == "forward") {
    const std::string probe = args["probe"].as<std::string>();
    bert_tokens tokens = bert_tokenize(bctx, probe, n_max_tokens);
    std::cout << "  probe='" << probe << "' tokens=[";
    for (size_t i = 0; i < tokens.size(); ++i) {
      std::cout << tokens[i];
      if (i + 1 < tokens.size()) std::cout << ", ";
    }
    std::cout << "] (" << tokens.size() << ")\n";

    distilbert_runner::ClassifierHead head =
        distilbert_runner::loadClassifierHead(bctx);
    std::cout << "  classifier: num_labels=" << head.num_labels << "\n";

    std::vector<float> logits;
    distilbert_runner::runForwardClassify(bctx, head, tokens,
                                          /*n_threads=*/4, logits);

    std::cout << "  logits=[";
    for (size_t i = 0; i < logits.size(); ++i) {
      std::cout << logits[i];
      if (i + 1 < logits.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // Argmax + softmax probability for the predicted label.
    int argmax = 0;
    for (int i = 1; i < head.num_labels; ++i) {
      if (logits[i] > logits[argmax]) argmax = i;
    }
    double max_logit = logits[argmax];
    double denom = 0.0;
    for (int i = 0; i < head.num_labels; ++i) {
      denom += std::exp(static_cast<double>(logits[i]) - max_logit);
    }
    const double prob_argmax = 1.0 / denom;
    const char* label_str = (argmax == 1) ? "POSITIVE" : "NEGATIVE";
    std::cout << "  predicted_label=" << argmax
              << " (" << label_str << ") prob=" << prob_argmax << "\n";

    bert_free(bctx);
    return 0;
  }

  if (mode == "measure") {
    std::cerr << "distilbert_runner: --mode measure not yet implemented "
                 "(arrives in Step 4-3-D)\n";
    bert_free(bctx);
    return 2;
  }

  std::cerr << "distilbert_runner: unknown --mode '" << mode << "'\n";
  bert_free(bctx);
  return 2;
} catch (const std::exception& e) {
  std::cerr << "distilbert_runner: " << e.what() << "\n";
  return 1;
}
