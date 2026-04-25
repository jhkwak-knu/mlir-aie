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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "cxxopts.hpp"

#include "config_loader_distilbert.h"
#include "model.h"

// mlp_runner shared primitives: LayerKernelEntry table + dispatcher factory,
// RAPL helpers, and the measurement RunStats / writeMeasurementsCore that
// distilbert reuses verbatim (Step 4-4-B).
#include "config_loader.h"
#include "measurement.h"
#include "npu_dispatch.h"
#include "rapl.h"

// bert.cpp public API
#include "bert.h"
#include "ggml.h"

namespace {

constexpr const char* kDefaultProbe = "Hello world";

// One slot in the per-layer NPU dispatch sequence. iamlemec/bert.cpp's
// encoder block fires mul_mat in a deterministic order — Q, K, V, attention
// output, ffn_expand, ffn_compress — and the hook's recording path uses the
// observed call ordinal modulo |expected_seq| to label each sample without
// needing string-matching against the GGUF tensor names.
struct ExpectedCall {
  std::string gemm_type;
  std::string sub_type;  // "Q"/"K"/"V" for qkv, "" otherwise
  int M;
  int K;
  int N;
};

// Build the per-layer NPU dispatch sequence from the experiment config.
// Shapes are derived from cfg (sequence_length, hidden_dim, ffn_dim) so a
// non-DistilBERT-base config (e.g. larger hidden) still labels correctly.
std::vector<ExpectedCall> buildExpectedSeq(
    const distilbert_runner::ExperimentConfig& cfg) {
  const int M = cfg.sequence_length;
  const int K = cfg.hidden_dim;
  const int N_ffn = cfg.ffn_dim;
  return {
      {"attention_qkv",    "Q", M, K, K},
      {"attention_qkv",    "K", M, K, K},
      {"attention_qkv",    "V", M, K, K},
      {"attention_output", "",  M, K, K},
      {"ffn_expand",       "",  M, K, N_ffn},
      {"ffn_compress",     "",  M, N_ffn, K},
  };
}

// Step 4-3-D-2 dispatch table: holds the XRT-backed dispatcher and a
// (M, K, N) -> LayerKernelEntry index built once at startup. The hook
// uses this to decide whether to claim a mul_mat for the NPU. Entries
// not in the table (classifier head M=1, attention_score / attention_
// context per-head batched, etc.) fall back to ggml's CPU kernel.
//
// Step 4-4-B additions: optional per-call recorder. When `recording=true`
// the hook brackets each NPU dispatch with steady_clock + RAPL and pushes
// a labeled mlp_runner::PerLayerSample onto current_inference_samples.
// expected_seq drives the (gemm_type, layer_idx, sub_type) labeling.
struct HookContext {
  std::unique_ptr<mlp_runner::Dispatcher> dispatcher;
  std::vector<mlp_runner::LayerKernelEntry> entries;
  std::map<std::tuple<int, int, int>, const mlp_runner::LayerKernelEntry*>
      by_shape;
  long long calls_dispatched = 0;
  long long calls_fallback = 0;

  // Recording state — only consulted on the ith==0 path.
  bool recording = false;
  int call_idx_in_inference = 0;
  std::vector<mlp_runner::PerLayerSample> current_inference_samples;
  std::vector<ExpectedCall> expected_seq;
  int total_npu_dispatches_per_inference = 0;
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

  // Optional per-call instrumentation (Step 4-4-B). Only wraps the dispatch
  // itself — the surrounding A/B copies are workload-independent host work
  // shared by every backend, so charging them to a kernel sample would
  // overstate the NPU's energy footprint.
  using clock_t_ = std::chrono::steady_clock;
  const bool record = ctx->recording;
  clock_t_::time_point t0;
  int64_t e0_pkg = 0, e0_core = 0;
  if (record) {
    e0_pkg  = mlp_runner::readRaplEnergyUj(mlp_runner::kRaplPackagePath);
    e0_core = mlp_runner::readRaplEnergyUj(mlp_runner::kRaplCorePath);
    t0 = clock_t_::now();
  }

  ctx->dispatcher->dispatch(*it->second, A.data(), B.data(), C.data());

  if (record) {
    const auto t1 = clock_t_::now();
    const int64_t e1_pkg  =
        mlp_runner::readRaplEnergyUj(mlp_runner::kRaplPackagePath);
    const int64_t e1_core =
        mlp_runner::readRaplEnergyUj(mlp_runner::kRaplCorePath);

    const int call_idx = ctx->call_idx_in_inference;
    const int per_layer_n = static_cast<int>(ctx->expected_seq.size());
    if (per_layer_n == 0 ||
        call_idx >= ctx->total_npu_dispatches_per_inference) {
      std::ostringstream oss;
      oss << "hook: extra NPU dispatch beyond expected ("
          << ctx->total_npu_dispatches_per_inference << "); call_idx="
          << call_idx;
      throw std::runtime_error(oss.str());
    }
    const int per_layer_idx = call_idx % per_layer_n;
    const int layer_idx = call_idx / per_layer_n;
    const auto& exp = ctx->expected_seq[per_layer_idx];
    if (M != exp.M || K != exp.K || N != exp.N) {
      std::ostringstream oss;
      oss << "hook: shape mismatch at call_idx=" << call_idx
          << " per_layer_idx=" << per_layer_idx
          << " expected (" << exp.M << "," << exp.K << "," << exp.N
          << ") got (" << M << "," << K << "," << N << ")";
      throw std::runtime_error(oss.str());
    }

    mlp_runner::PerLayerSample ps;
    std::ostringstream key;
    key << "L" << layer_idx << "/" << exp.gemm_type;
    if (!exp.sub_type.empty()) key << "/" << exp.sub_type;
    ps.layer = key.str();
    ps.meta.gemm_type = exp.gemm_type;
    ps.meta.layer_idx = layer_idx;
    ps.meta.sub_type  = exp.sub_type;
    ps.time_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    ps.energy_uj_package = mlp_runner::raplDelta(e1_pkg, e0_pkg);
    ps.energy_uj_core    = mlp_runner::raplDelta(e1_core, e0_core);
    ctx->current_inference_samples.push_back(std::move(ps));
    ++ctx->call_idx_in_inference;
  }

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
    if (!hook_ctx) {
      std::cerr << "distilbert_runner: --mode measure requires "
                   "--backend npu (CPU baseline measurement is "
                   "covered by --mode forward)\n";
      bert_free(bctx);
      return 2;
    }
    if (cfg.warmup_iterations <= 0 || cfg.outer_batches <= 0 ||
        cfg.inner_target_seconds <= 0.0 || cfg.bracket_idle_count <= 0 ||
        cfg.bracket_idle_seconds <= 0.0) {
      std::cerr << "distilbert_runner: measurement.* config fields must all "
                   "be positive (warmup_iterations, outer_batches, "
                   "inner_target_seconds, bracket_idle_count, "
                   "bracket_idle_seconds)\n";
      bert_free(bctx);
      return 2;
    }

    using clock_t_ = std::chrono::steady_clock;
    using dseconds = std::chrono::duration<double>;
    using dmicro   = std::chrono::duration<double, std::micro>;

    // Hook recorder setup. Per-layer NPU dispatches arrive in fixed order
    // (Q, K, V, attention_output, ffn_expand, ffn_compress) per encoder
    // block; total = num_layers * 6 calls per inference.
    hook_ctx->expected_seq = buildExpectedSeq(cfg);
    hook_ctx->total_npu_dispatches_per_inference =
        cfg.num_layers * static_cast<int>(hook_ctx->expected_seq.size());

    // Workload: a single fixed sentence, tokenized once. Drives identical
    // graph structure on every inference so per-iteration variability comes
    // from execution noise rather than payload differences.
    distilbert_runner::ClassifierHead head =
        distilbert_runner::loadClassifierHead(bctx);
    const std::string probe = args["probe"].as<std::string>();
    bert_tokens tokens = bert_tokenize(bctx, probe, n_max_tokens);
    std::vector<float> scratch_logits;

    auto run_one_inference = [&]() {
      hook_ctx->recording = true;
      hook_ctx->call_idx_in_inference = 0;
      hook_ctx->current_inference_samples.clear();
      distilbert_runner::runForwardClassify(bctx, head, tokens, n_threads,
                                            scratch_logits);
      hook_ctx->recording = false;
      const int got = static_cast<int>(
          hook_ctx->current_inference_samples.size());
      if (got != hook_ctx->total_npu_dispatches_per_inference) {
        std::ostringstream oss;
        oss << "measure: expected "
            << hook_ctx->total_npu_dispatches_per_inference
            << " NPU dispatches per inference, observed " << got
            << " — call ordering changed (verify bert.cpp encoder block)";
        throw std::runtime_error(oss.str());
      }
    };

    mlp_runner::RunStats stats;
    stats.warmup_iterations = cfg.warmup_iterations;
    stats.outer_batches = cfg.outer_batches;
    stats.inner_target_seconds = cfg.inner_target_seconds;
    stats.idle_baseline_mw =
        mlp_runner::measureIdlePowerMw(mlp_runner::kRaplPackagePath);

    std::cout << "[measure] idle_baseline=" << stats.idle_baseline_mw
              << " mW, warmup=" << cfg.warmup_iterations
              << ", outer_batches=" << cfg.outer_batches << "\n";

    for (int w = 0; w < cfg.warmup_iterations; ++w) {
      run_one_inference();
    }

    std::vector<double> batch_energies_per_inf;
    std::vector<double> batch_times;

    for (int b = 0; b < cfg.outer_batches; ++b) {
      mlp_runner::BatchStats bs;
      bs.index = b;
      bs.idle_power_pre_mw = mlp_runner::measureBracketIdlePowerMw(
          mlp_runner::kRaplPackagePath, cfg.bracket_idle_count,
          cfg.bracket_idle_seconds);

      const int64_t batch_e0_pkg =
          mlp_runner::readRaplEnergyUj(mlp_runner::kRaplPackagePath);
      const int64_t batch_e0_core =
          mlp_runner::readRaplEnergyUj(mlp_runner::kRaplCorePath);
      const auto batch_t0 = clock_t_::now();

      std::vector<double> inner_times;
      std::map<std::string, std::vector<double>> layer_times;
      std::map<std::string, std::vector<int64_t>> layer_energies_pkg;

      int n_inner = 0;
      while (true) {
        const auto inf_t0 = clock_t_::now();
        run_one_inference();
        const auto inf_t1 = clock_t_::now();

        inner_times.push_back(dmicro(inf_t1 - inf_t0).count());
        for (const auto& ps : hook_ctx->current_inference_samples) {
          layer_times[ps.layer].push_back(ps.time_us);
          layer_energies_pkg[ps.layer].push_back(ps.energy_uj_package);
          if (!bs.layer_meta.count(ps.layer)) bs.layer_meta[ps.layer] = ps.meta;
        }
        ++n_inner;

        const double elapsed_s = dseconds(clock_t_::now() - batch_t0).count();
        if (elapsed_s >= cfg.inner_target_seconds) break;
      }

      const auto batch_t1 = clock_t_::now();
      const int64_t batch_e1_pkg =
          mlp_runner::readRaplEnergyUj(mlp_runner::kRaplPackagePath);
      const int64_t batch_e1_core =
          mlp_runner::readRaplEnergyUj(mlp_runner::kRaplCorePath);

      bs.idle_power_post_mw = mlp_runner::measureBracketIdlePowerMw(
          mlp_runner::kRaplPackagePath, cfg.bracket_idle_count,
          cfg.bracket_idle_seconds);

      bs.n_inner = n_inner;
      bs.wall_s = dseconds(batch_t1 - batch_t0).count();
      bs.active_uj_package = mlp_runner::raplDelta(batch_e1_pkg, batch_e0_pkg);
      bs.active_uj_core    = mlp_runner::raplDelta(batch_e1_core, batch_e0_core);
      bs.idle_power_mw = 0.5 * (bs.idle_power_pre_mw + bs.idle_power_post_mw);
      const double idle_uj =
          bs.idle_power_mw * bs.wall_s * 1000.0;  // mW * s * 1000 -> uJ
      bs.npu_uj_package = static_cast<int64_t>(
          std::max<double>(0.0, bs.active_uj_package - idle_uj));
      bs.npu_uj_core = bs.active_uj_core;
      bs.energy_per_inference_uj = (n_inner > 0)
          ? static_cast<double>(bs.npu_uj_package) / n_inner
          : 0.0;

      if (!inner_times.empty()) {
        auto mn = std::min_element(inner_times.begin(), inner_times.end());
        const double sum = std::accumulate(inner_times.begin(),
                                           inner_times.end(), 0.0);
        bs.model_time_us_min = *mn;
        bs.model_time_us_mean = sum / inner_times.size();
      }
      for (const auto& [layer, ts] : layer_times) {
        auto mn = std::min_element(ts.begin(), ts.end());
        const double sum = std::accumulate(ts.begin(), ts.end(), 0.0);
        bs.layer_time_us_min[layer] = *mn;
        bs.layer_time_us_mean[layer] = sum / ts.size();
        auto& g = stats.layer_time_us_min_global[layer];
        if (g == 0.0 || *mn < g) g = *mn;
      }
      for (const auto& [layer, es] : layer_energies_pkg) {
        const int64_t s = std::accumulate(es.begin(), es.end(), int64_t{0});
        bs.layer_energy_uj_sum[layer] = s;
        auto& g = stats.layer_energy_uj_sum_min_global[layer];
        if (g == 0 || s < g) g = s;
      }

      batch_energies_per_inf.push_back(bs.energy_per_inference_uj);
      batch_times.push_back(bs.wall_s);

      std::cout << "[measure] batch " << b << ": n_inner=" << n_inner
                << " wall_s=" << bs.wall_s
                << " model_time_us_min=" << bs.model_time_us_min
                << " npu_uj=" << bs.npu_uj_package
                << " idle_pre=" << bs.idle_power_pre_mw
                << " idle_post=" << bs.idle_power_post_mw << "\n";

      stats.batches.push_back(std::move(bs));
    }

    stats.idle_post_mw =
        mlp_runner::measureIdlePowerMw(mlp_runner::kRaplPackagePath);

    const auto e_cv = mlp_runner::computeMeanCv(batch_energies_per_inf);
    const auto t_cv = mlp_runner::computeMeanCv(batch_times);
    stats.batch_energy_mean_uj = e_cv.mean;
    stats.batch_energy_cv_pct = e_cv.cv_pct;
    stats.batch_time_mean_s = t_cv.mean;
    stats.batch_time_cv_pct = t_cv.cv_pct;

    if (!stats.batches.empty()) {
      stats.batch_min_energy_per_inference_uj = std::min_element(
          stats.batches.begin(), stats.batches.end(),
          [](const auto& a, const auto& b) {
            return a.energy_per_inference_uj < b.energy_per_inference_uj;
          })->energy_per_inference_uj;
      stats.batch_min_model_time_us = std::min_element(
          stats.batches.begin(), stats.batches.end(),
          [](const auto& a, const auto& b) {
            return a.model_time_us_min < b.model_time_us_min;
          })->model_time_us_min;
    }

    nlohmann::json model_summary;
    model_summary["config_path"] = cfg.config_path;
    model_summary["model_type"] = "distilbert";
    model_summary["pretrained_source"] = cfg.pretrained_source;
    model_summary["num_layers"] = cfg.num_layers;
    model_summary["hidden_dim"] = cfg.hidden_dim;
    model_summary["ffn_dim"] = cfg.ffn_dim;
    model_summary["num_heads"] = cfg.num_heads;
    model_summary["sequence_length"] = cfg.sequence_length;
    model_summary["batch_size"] = cfg.batch_size;
    model_summary["multi_head_strategy"] = cfg.multi_head_strategy;

    std::string out_dir = args["output-dir"].as<std::string>();
    if (out_dir.empty()) out_dir = cfg.results_dir;
    mlp_runner::writeMeasurementsCore(out_dir, setter, backend, stats,
                                      model_summary);

    std::cout << "[measure] done — energy_cv=" << stats.batch_energy_cv_pct
              << "%  time_cv=" << stats.batch_time_cv_pct << "%  "
              << "min_model_us=" << stats.batch_min_model_time_us << "  "
              << "min_energy_per_inf_uj="
              << stats.batch_min_energy_per_inference_uj << "  "
              << "wrote -> " << out_dir << "\n";

    bert_free(bctx);
    return 0;
  }

  std::cerr << "distilbert_runner: unknown --mode '" << mode << "'\n";
  bert_free(bctx);
  return 2;
} catch (const std::exception& e) {
  std::cerr << "distilbert_runner: " << e.what() << "\n";
  return 1;
}
