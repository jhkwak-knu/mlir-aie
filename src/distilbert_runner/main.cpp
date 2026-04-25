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
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "cxxopts.hpp"

#include "config_loader_distilbert.h"
#include "model.h"

// mlp_runner shared primitives: LayerKernelEntry table + dispatcher factory.
#include "config_loader.h"
#include "npu_dispatch.h"

// bert.cpp public API
#include "bert.h"
#include "ggml.h"

namespace {

constexpr const char* kDefaultProbe = "Hello world";

// Step 4-3-D-2 dispatch table: holds the XRT-backed dispatcher and a
// (M, K, N) -> LayerKernelEntry index built once at startup. The hook
// uses this to decide whether to claim a mul_mat for the NPU. Entries
// not in the table (classifier head M=1, attention_score / attention_
// context per-head batched, etc.) fall back to ggml's CPU kernel.
struct HookContext {
  std::unique_ptr<mlp_runner::Dispatcher> dispatcher;
  std::vector<mlp_runner::LayerKernelEntry> entries;
  std::map<std::tuple<int, int, int>, const mlp_runner::LayerKernelEntry*>
      by_shape;
  long long calls_dispatched = 0;
  long long calls_fallback = 0;
};

bool distilbert_mul_mat_hook(const ggml_tensor* src0, const ggml_tensor* src1,
                             ggml_tensor* dst, int ith, int /*nth*/,
                             void* user_data) {
  auto* ctx = static_cast<HookContext*>(user_data);

  // No dispatcher attached (--backend cpu, or context not yet populated):
  // let ggml's CPU kernel handle every mul_mat.
  if (!ctx || !ctx->dispatcher) return false;

  // Type / rank guard. Hook must only claim 2D f32-activation ops whose
  // weight is f16 or f32. Batched per-head ops (ne[2] > 1) and any other
  // dtype combo fall through to ggml.
  const bool weight_ok =
      (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
  const bool act_ok  = (src1->type == GGML_TYPE_F32);
  const bool dst_ok  = (dst->type  == GGML_TYPE_F32);
  const bool rank_ok = (src0->ne[2] == 1 && src0->ne[3] == 1 &&
                        src1->ne[2] == 1 && src1->ne[3] == 1);
  if (!(weight_ok && act_ok && dst_ok && rank_ok)) {
    if (ith == 0) ++ctx->calls_fallback;
    return false;
  }

  // ggml_mul_mat semantics with ne ordering:
  //   src0 = (K, N)   -> weight matrix, logically [N rows of K]
  //   src1 = (K, M)   -> activation matrix, logically [M rows of K]
  //   dst  = (N, M)   -> output, logically [M rows of N]
  // i.e. dst[m, n] = sum_k src1[m, k] * src0[n, k].
  const int K = static_cast<int>(src0->ne[0]);
  const int N = static_cast<int>(src0->ne[1]);
  const int M = static_cast<int>(src1->ne[1]);
  if (src1->ne[0] != src0->ne[0]) {
    if (ith == 0) ++ctx->calls_fallback;
    return false;
  }

  auto it = ctx->by_shape.find(std::make_tuple(M, K, N));
  if (it == ctx->by_shape.end()) {
    if (ith == 0) ++ctx->calls_fallback;
    return false;
  }

  // Single-source the dispatch on ith == 0; ggml's per-node barrier
  // makes the other workers wait until we return true here. Returning
  // true everywhere keeps every worker out of the CPU kernel.
  if (ith != 0) return true;

  ++ctx->calls_dispatched;

  // A buffer [M, K] f32 row-major from src1.
  std::vector<float> A(static_cast<size_t>(M) * K);
  if (ggml_is_contiguous(src1)) {
    std::memcpy(A.data(), src1->data,
                static_cast<size_t>(M) * K * sizeof(float));
  } else {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        const auto* p = static_cast<const char*>(src1->data) +
                        m * src1->nb[1] + k * src1->nb[0];
        A[static_cast<size_t>(m) * K + k] =
            *reinterpret_cast<const float*>(p);
      }
    }
  }

  // B buffer [K, N] f32 row-major from src0 (logically [N, K]).
  // XrtDispatcher::dispatch transposes B -> NPU layout [N, K] internally
  // and matches matB[n*K + k] = W[n, k]. We therefore need
  // B[k*N + n] = W[n, k] = src0_data[n*K + k].
  std::vector<float> B(static_cast<size_t>(K) * N);
  if (src0->type == GGML_TYPE_F16) {
    for (int n = 0; n < N; ++n) {
      const auto* row = reinterpret_cast<const ggml_fp16_t*>(
          static_cast<const char*>(src0->data) + n * src0->nb[1]);
      for (int k = 0; k < K; ++k) {
        B[static_cast<size_t>(k) * N + n] = ggml_fp16_to_fp32(row[k]);
      }
    }
  } else {  // GGML_TYPE_F32
    for (int n = 0; n < N; ++n) {
      const auto* row = reinterpret_cast<const float*>(
          static_cast<const char*>(src0->data) + n * src0->nb[1]);
      for (int k = 0; k < K; ++k) {
        B[static_cast<size_t>(k) * N + n] = row[k];
      }
    }
  }

  // C buffer [M, N] f32 — XrtDispatcher applies bf16 truncation + tile
  // staging internally and writes the fp32 result back into C.
  std::vector<float> C(static_cast<size_t>(M) * N);
  ctx->dispatcher->dispatch(*it->second, A.data(), B.data(), C.data());

  // dst (N, M) contiguous: data[m*N + n] = result[m, n] — same layout as C.
  if (ggml_is_contiguous(dst)) {
    std::memcpy(dst->data, C.data(),
                static_cast<size_t>(M) * N * sizeof(float));
  } else {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        auto* p = static_cast<char*>(dst->data) +
                  m * dst->nb[1] + n * dst->nb[0];
        *reinterpret_cast<float*>(p) = C[static_cast<size_t>(m) * N + n];
      }
    }
  }
  return true;
}

static std::string joinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  return a.back() == '/' ? a + b : a + "/" + b;
}

// Build the HookContext from configurations.json + kernel_binaries dir
// resolved off cfg.results_dir. Throws on missing files / empty entries
// so misconfigured runs fail loud rather than silently CPU-falling-back.
std::unique_ptr<HookContext> buildHookContext(
    const distilbert_runner::ExperimentConfig& cfg,
    const std::string& setter,
    const std::string& backend) {
  if (backend != "npu") return nullptr;

  const std::string configurations_path =
      joinPath(cfg.results_dir, "configurations.json");
  const std::string kernel_binaries_dir =
      joinPath(cfg.results_dir, "kernel_binaries");

  auto ctx = std::make_unique<HookContext>();
  ctx->dispatcher = mlp_runner::makeDispatcher(backend);
  ctx->entries = mlp_runner::loadLayerKernels(
      configurations_path, setter, kernel_binaries_dir);
  for (const auto& e : ctx->entries) {
    auto key = std::make_tuple(e.M, e.K, e.N);
    if (!ctx->by_shape.count(key)) {
      ctx->by_shape.emplace(key, &e);
    }
  }
  std::cout << "[hook] loaded " << ctx->entries.size()
            << " kernel entries for setter=" << setter
            << " (unique shapes=" << ctx->by_shape.size() << ")\n";
  return ctx;
}

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
      ("sentences-file",
       "newline-separated input sentences for batched inference; when set, "
       "the runner ignores --probe and writes one logits row per line to "
       "--output-logits",
       cxxopts::value<std::string>()->default_value(""))
      ("output-logits",
       "destination for batched logits (one space-separated row per "
       "sentence). Defaults to stdout when --sentences-file is set.",
       cxxopts::value<std::string>()->default_value(""))
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

  // Always force CPU compute backend: our NPU dispatch lives in ggml's
  // mul_mat hook, which only fires from the CPU code path. Picking a
  // GPU backend here would route mul_mat around the hook entirely.
  bert_ctx* bctx = bert_load_from_file(gguf_path.c_str(), /*use_cpu=*/true);
  if (!bctx) {
    std::cerr << "bert_load_from_file failed: " << gguf_path << "\n";
    return 1;
  }

  // Build the dispatch table (only for --backend npu) and register the
  // mul_mat hook. ggml stores the hook as a global, so a single call
  // covers every ggml_backend_graph_compute issued below.
  auto hook_ctx = buildHookContext(cfg, setter, backend);
  ggml_set_mul_mat_hook(distilbert_mul_mat_hook, hook_ctx.get());
  // ggml replicates each op across all worker threads. Our hook trusts
  // the per-node barrier to keep ith != 0 idle while ith == 0 runs the
  // NPU dispatch, but we serialize with n_threads = 1 on the NPU path
  // until we have a thread-safety audit of XrtDispatcher's xrt::bo
  // re-use. CPU mode keeps the historical 4 threads.
  const int n_threads = hook_ctx ? 1 : 4;
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
    distilbert_runner::ClassifierHead head_for_batch;
    const std::string sfile = args["sentences-file"].as<std::string>();
    if (!sfile.empty()) {
      // Batch mode: SST-2 sweep. Load the head once, stream sentences.
      head_for_batch = distilbert_runner::loadClassifierHead(bctx);
      std::ifstream sin(sfile);
      if (!sin.is_open()) {
        throw std::runtime_error("cannot open --sentences-file: " + sfile);
      }
      std::ostream* logits_sink = &std::cout;
      std::ofstream lout;
      const std::string lpath = args["output-logits"].as<std::string>();
      if (!lpath.empty()) {
        lout.open(lpath);
        if (!lout.is_open()) {
          throw std::runtime_error("cannot open --output-logits: " + lpath);
        }
        logits_sink = &lout;
      }
      *logits_sink << std::setprecision(7);
      std::string line;
      int n = 0;
      std::vector<float> logits;
      while (std::getline(sin, line)) {
        if (line.empty()) continue;
        auto tokens = bert_tokenize(bctx, line, n_max_tokens);
        distilbert_runner::runForwardClassify(bctx, head_for_batch, tokens,
                                              n_threads, logits);
        for (size_t i = 0; i < logits.size(); ++i) {
          if (i) *logits_sink << ' ';
          *logits_sink << logits[i];
        }
        *logits_sink << '\n';
        n += 1;
      }
      const long long dispatched =
          hook_ctx ? hook_ctx->calls_dispatched : 0;
      const long long fellback =
          hook_ctx ? hook_ctx->calls_fallback   : 0;
      std::cout << "  batch sentences processed: " << n
                << " dispatched_mul_mat=" << dispatched
                << " cpu_fallback_mul_mat=" << fellback << "\n";
      bert_free(bctx);
      return 0;
    }

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
