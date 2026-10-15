//===- host_reconfig.cpp ----------------------------------------*- C++ -*-===//
//
// Differential microbenchmark for NPU runtime reconfiguration (layer-switch)
// cost. Loads K distinct xclbins (different tiling configs of the SAME GEMM
// shape) simultaneously, then measures dispatch+wait time/energy under two
// schedules:
//   - baseline:    dispatch config i repeatedly (i -> i, no config switch)
//   - alternating: round-robin over d configs (every dispatch switches config)
// per-switch cost = alternating_per_dispatch - mean(baseline_per_dispatch).
//
// Timing of a matmul dispatch is data-independent (fixed-function, no
// data-dependent branching), so input buffers are zero-filled and synced once;
// no tile staging / decode / verification is needed. The dispatch+wait window
// (t0..t1 around runKernel) captures any driver column reprogram that happens
// inside run.wait() when the resident config changes.
//
// Methodology (batch-min + idle-bracketed RAPL) mirrors host.cpp:507-649.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "tiling_param.h"  // loadTilingParam, DATATYPE, PKT_HDR_ELEMS, AXIS_*

// ---------------------------------------------------------------------------
// RAPL energy helpers (copied verbatim from host.cpp:38-94)
// ---------------------------------------------------------------------------
static const char *RAPL_PKG_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj";
static const char *RAPL_CORE_PATH =
    "/sys/class/powercap/intel-rapl:0:0/energy_uj";
static constexpr int64_t RAPL_MAX_ENERGY_RANGE_UJ = 65532610987LL;

static int64_t raplDelta(int64_t after, int64_t before) {
  int64_t d = after - before;
  if (d < 0) d += RAPL_MAX_ENERGY_RANGE_UJ;
  return d;
}

static int64_t readRaplEnergyUj(const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) return 0;
  int64_t val = 0;
  if (fscanf(fp, "%ld", &val) != 1) val = 0;
  fclose(fp);
  return val;
}

static constexpr double IDLE_SAMPLE_WINDOW_S = 0.2;
static constexpr int N_IDLE_SAMPLES = 10;
static constexpr double BRACKET_IDLE_WINDOW_S = 0.3;
static constexpr int N_BRACKET_IDLE_SAMPLES = 5;

static double medianSorted(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static double measureIdlePowerMw(const char *raplPath, int nSamples,
                                 double windowS) {
  std::vector<double> samples(nSamples);
  for (int s = 0; s < nSamples; ++s) {
    int64_t s0 = readRaplEnergyUj(raplPath);
    auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(windowS));
    auto t1 = std::chrono::steady_clock::now();
    int64_t s1 = readRaplEnergyUj(raplPath);
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    samples[s] = static_cast<double>(raplDelta(s1, s0)) / elapsed / 1000.0;
  }
  return medianSorted(samples);
}

static double coeffVarPct(const std::vector<double> &v) {
  if (v.size() < 2) return -1.0;
  double sum = 0.0;
  for (double x : v) sum += x;
  double mean = sum / v.size();
  if (mean == 0.0) return -1.0;
  double var = 0.0;
  for (double x : v) var += (x - mean) * (x - mean);
  var /= (v.size() - 1);
  return 100.0 * std::sqrt(var) / mean;
}

// ---------------------------------------------------------------------------
// Per-config resident context
// ---------------------------------------------------------------------------
struct Ctx {
  xrt::kernel kernel;
  xrt::bo bo_instr, bo_inA, bo_inB, bo_outC, bo_inC;
  bool useInC = false;
  size_t instrSize = 0;
  // metadata for reporting
  uint32_t M = 0, K = 0, N = 0, numCores = 0, SPm = 0, SPn = 0;
};

// Compute per-dispatch chunk element counts (host.cpp:131-139 formulas).
static void chunkSizes(const tilingParam &tp, int &a, int &b, int &c,
                       int &outc, bool &useInC) {
  int chunkTPm = (tp.tpOrder[0] == AXIS_M) ? tp.TPm : 1;
  int chunkTPn = (tp.tpOrder[0] == AXIS_N) ? tp.TPn : 1;
  int chunkTPk = (tp.tpOrder[0] == AXIS_K) ? tp.TPk : 1;
  a = (tp.TM * tp.TK) * tp.SPm * chunkTPm * chunkTPk;
  b = (tp.TN * tp.TK) * tp.SPn * chunkTPn * chunkTPk;
  c = (tp.TM * tp.TN) * tp.SPm * tp.SPn * chunkTPm * chunkTPn;
  outc = (tp.TM * tp.TN + PKT_HDR_ELEMS) * tp.SPm * tp.SPn * chunkTPm * chunkTPn;
  useInC = (tp.TPk > 1) && (tp.tpOrder[0] != AXIS_K);
}

// Load one config directory (final.xclbin + insts.bin + tc.json) into a Ctx.
static void loadConfig(xrt::device &device, const std::string &cfgDir,
                       int verbosity, Ctx &ctx) {
  const std::string xclbin = cfgDir + "/final.xclbin";
  const std::string insts = cfgDir + "/insts.bin";
  const std::string tcjson = cfgDir + "/tc.json";

  tilingParam tp = loadTilingParam(tcjson);
  int a, b, c, outc;
  bool useInC;
  chunkSizes(tp, a, b, c, outc, useInC);
  ctx.useInC = useInC;
  ctx.M = tp.M; ctx.K = tp.K; ctx.N = tp.N;
  ctx.SPm = tp.SPm; ctx.SPn = tp.SPn;
  ctx.numCores = tp.SPm * tp.SPn;

  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(insts);
  ctx.instrSize = instr_v.size();

  // device(0) is reopened inside init each call; all contexts share device(0)
  // but keep their own hw_context alive via the kernel handle (see
  // xrt_dispatcher.cpp:128-137 for the proven multi-context pattern).
  test_utils::init_xrt_load_kernel(device, ctx.kernel, verbosity, xclbin,
                                   "MLIR_AIE");

  ctx.bo_instr = xrt::bo(device, ctx.instrSize * sizeof(int),
                         XCL_BO_FLAGS_CACHEABLE, ctx.kernel.group_id(1));
  ctx.bo_inA = xrt::bo(device, a * sizeof(DATATYPE), XRT_BO_FLAGS_HOST_ONLY,
                       ctx.kernel.group_id(3));
  ctx.bo_inB = xrt::bo(device, b * sizeof(DATATYPE), XRT_BO_FLAGS_HOST_ONLY,
                       ctx.kernel.group_id(4));
  ctx.bo_outC = xrt::bo(device, outc * sizeof(DATATYPE), XRT_BO_FLAGS_HOST_ONLY,
                        ctx.kernel.group_id(5));
  if (useInC)
    ctx.bo_inC = xrt::bo(device, c * sizeof(DATATYPE), XRT_BO_FLAGS_HOST_ONLY,
                         ctx.kernel.group_id(6));

  // Values are irrelevant for timing; write zeros once and sync.
  std::memcpy(ctx.bo_instr.map<void *>(), instr_v.data(),
              ctx.instrSize * sizeof(int));
  std::memset(ctx.bo_inA.map<void *>(), 0, a * sizeof(DATATYPE));
  std::memset(ctx.bo_inB.map<void *>(), 0, b * sizeof(DATATYPE));
  std::memset(ctx.bo_outC.map<void *>(), 0, outc * sizeof(DATATYPE));
  ctx.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ctx.bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ctx.bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (useInC) {
    std::memset(ctx.bo_inC.map<void *>(), 0, c * sizeof(DATATYPE));
    ctx.bo_inC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
}

static void runKernel(Ctx &c) {
  xrt::run run;
  if (c.useInC)
    run = c.kernel(3, c.bo_instr, c.instrSize, c.bo_inA, c.bo_inB, c.bo_outC,
                   c.bo_inC);
  else
    run = c.kernel(3, c.bo_instr, c.instrSize, c.bo_inA, c.bo_inB, c.bo_outC);
  run.wait();
}

// ---------------------------------------------------------------------------
// Batch-min measurement over a dispatch schedule
// ---------------------------------------------------------------------------
struct BatchResult {
  double per_dispatch_us_min = -1.0;   // min over batches of mean per-dispatch t
  double per_dispatch_us_mean = -1.0;  // grand mean per-dispatch t
  double per_dispatch_energy_uj = -1.0;  // min-energy batch, npu_uj / n_inner
  double wall_s = -1.0;
  double idle_pre_mw = -1.0;
  double idle_post_mw = -1.0;
  double energy_cv_pct = -1.0;
  double time_cv_pct = -1.0;
  int n_batches = 0;
  int n_inner = 0;
};

// `pick(iter)` returns the Ctx to dispatch on global iteration `iter`.
static BatchResult measureSchedule(const std::function<Ctx &(int)> &pick,
                                   int n_batches, int n_warmup,
                                   double target_batch_wall_s,
                                   bool rapl_available, bool rapl_core_available,
                                   double session_idle_mw) {
  BatchResult r;
  r.n_batches = n_batches;

  // Warmup + size n_inner from min warmup dispatch time.
  double single_us = 1e18;
  int wi = 0;
  for (int w = 0; w < n_warmup; ++w, ++wi) {
    auto t0 = std::chrono::high_resolution_clock::now();
    runKernel(pick(wi));
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (us < single_us) single_us = us;
  }
  if (single_us <= 0 || single_us >= 1e18) single_us = 1.0;
  int n_inner = std::max(10, static_cast<int>(std::ceil(
                                 target_batch_wall_s / (single_us / 1e6))));
  r.n_inner = n_inner;

  int iter = wi;  // continue the rotation counter past warmup
  double best_time = 1e18, best_energy = 1e18;
  double time_sum = 0.0;
  long time_cnt = 0;
  std::vector<double> batch_time_means, batch_energies;

  for (int batch = 0; batch < n_batches; ++batch) {
    double idle_before = session_idle_mw;
    if (rapl_available)
      idle_before = measureIdlePowerMw(RAPL_PKG_PATH, N_BRACKET_IDLE_SAMPLES,
                                       BRACKET_IDLE_WINDOW_S);
    int64_t pkg_before = rapl_available ? readRaplEnergyUj(RAPL_PKG_PATH) : 0;
    (void)rapl_core_available;
    auto wall0 = std::chrono::steady_clock::now();

    double batch_time_total = 0.0;
    for (int i = 0; i < n_inner; ++i, ++iter) {
      auto t0 = std::chrono::high_resolution_clock::now();
      runKernel(pick(iter));
      auto t1 = std::chrono::high_resolution_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      batch_time_total += us;
      time_sum += us;
      ++time_cnt;
    }

    auto wall1 = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(wall1 - wall0).count();
    double idle_after = idle_before;
    if (rapl_available)
      idle_after = measureIdlePowerMw(RAPL_PKG_PATH, N_BRACKET_IDLE_SAMPLES,
                                      BRACKET_IDLE_WINDOW_S);
    double idle_mw = 0.5 * (idle_before + idle_after);

    double batch_time_mean = batch_time_total / n_inner;
    batch_time_means.push_back(batch_time_mean);
    if (batch_time_mean < best_time) {
      best_time = batch_time_mean;
      r.wall_s = wall_s;
      r.idle_pre_mw = idle_before;
      r.idle_post_mw = idle_after;
    }

    if (rapl_available) {
      int64_t pkg_after = readRaplEnergyUj(RAPL_PKG_PATH);
      double active_uj = static_cast<double>(raplDelta(pkg_after, pkg_before));
      double npu_uj = active_uj - (idle_mw * wall_s * 1000.0);
      double e_per = npu_uj / n_inner;
      batch_energies.push_back(e_per);
      if (e_per >= 0 && e_per < best_energy) best_energy = e_per;
    }
  }

  r.per_dispatch_us_min = best_time;
  r.per_dispatch_us_mean = (time_cnt > 0) ? time_sum / time_cnt : -1.0;
  r.per_dispatch_energy_uj = (best_energy < 1e18) ? best_energy : -1.0;
  r.time_cv_pct = coeffVarPct(batch_time_means);
  r.energy_cv_pct = coeffVarPct(batch_energies);
  return r;
}

// ---------------------------------------------------------------------------
int main(int argc, const char *argv[]) {
  cxxopts::Options options("reconfig_bench",
                           "NPU reconfiguration differential microbenchmark");
  options.add_options()
      ("config-dir", "Directory holding cfg0..cfg{depth-1} subdirs",
       cxxopts::value<std::string>())
      ("depth", "Rotation depth d (number of distinct configs)",
       cxxopts::value<int>()->default_value("2"))
      ("n-warmup", "Warmup dispatches", cxxopts::value<int>()->default_value("20"))
      ("n-batches", "Outer batches", cxxopts::value<int>()->default_value("5"))
      ("target-batch-wall-s", "Target wall per batch (s)",
       cxxopts::value<double>()->default_value("0.5"))
      ("out-csv", "Append results to this CSV",
       cxxopts::value<std::string>()->default_value("out/reports/reconfig_v1.csv"))
      ("shape-tag", "Shape label for CSV",
       cxxopts::value<std::string>()->default_value(""))
      ("run-tag", "Run label for CSV (e.g. timestamp)",
       cxxopts::value<std::string>()->default_value("run"))
      ("verbosity", "Verbosity", cxxopts::value<int>()->default_value("0"))
      ("h,help", "Print help");

  auto vm = options.parse(argc, argv);
  if (vm.count("help") || !vm.count("config-dir")) {
    std::cout << options.help() << std::endl;
    return vm.count("config-dir") ? 0 : 1;
  }

  const std::string config_dir = vm["config-dir"].as<std::string>();
  const int depth = vm["depth"].as<int>();
  const int n_warmup = vm["n-warmup"].as<int>();
  const int n_batches = vm["n-batches"].as<int>();
  const double target_wall = vm["target-batch-wall-s"].as<double>();
  const std::string out_csv = vm["out-csv"].as<std::string>();
  const std::string shape_tag = vm["shape-tag"].as<std::string>();
  const std::string run_tag = vm["run-tag"].as<std::string>();
  const int verbosity = vm["verbosity"].as<int>();

  if (depth < 2) {
    std::cerr << "depth must be >= 2\n";
    return 1;
  }

  // Load `depth` configs (device shared across all).
  xrt::device device;
  std::vector<Ctx> ctxs(depth);
  for (int i = 0; i < depth; ++i) {
    std::string dir = config_dir + "/cfg" + std::to_string(i);
    std::cout << "Loading config " << i << " from " << dir << std::endl;
    loadConfig(device, dir, verbosity, ctxs[i]);
  }

  // Session idle baseline (median).
  bool rapl_available = readRaplEnergyUj(RAPL_PKG_PATH) > 0;
  bool rapl_core_available = readRaplEnergyUj(RAPL_CORE_PATH) > 0;
  double session_idle_mw = -1.0;
  if (rapl_available)
    session_idle_mw =
        measureIdlePowerMw(RAPL_PKG_PATH, N_IDLE_SAMPLES, IDLE_SAMPLE_WINDOW_S);

  // Open CSV (write header if new).
  bool exists = std::ifstream(out_csv).good();
  std::ofstream csv(out_csv, std::ios::app);
  if (!csv) {
    std::cerr << "cannot open csv: " << out_csv << std::endl;
    return 1;
  }
  if (!exists)
    csv << "run_tag,shape_tag,depth,schedule,config_index,numCores,SPm,SPn,"
           "n_batches,n_inner,per_dispatch_us_min,per_dispatch_us_mean,"
           "per_dispatch_energy_uj,wall_s,idle_pre_mw,idle_post_mw,"
           "energy_cv_pct,time_cv_pct\n";

  auto write_row = [&](const std::string &sched, int cfg_idx, const Ctx &c,
                       const BatchResult &r) {
    csv << run_tag << "," << shape_tag << "," << depth << "," << sched << ","
        << cfg_idx << "," << c.numCores << "," << c.SPm << "," << c.SPn << ","
        << r.n_batches << "," << r.n_inner << "," << std::fixed
        << std::setprecision(3) << r.per_dispatch_us_min << ","
        << r.per_dispatch_us_mean << "," << r.per_dispatch_energy_uj << ","
        << std::setprecision(6) << r.wall_s << "," << std::setprecision(2)
        << r.idle_pre_mw << "," << r.idle_post_mw << "," << std::setprecision(3)
        << r.energy_cv_pct << "," << r.time_cv_pct << "\n";
    csv.flush();
  };

  // Phase 1: baseline for each config (i -> i, no switch).
  for (int i = 0; i < depth; ++i) {
    auto pick = [&, i](int) -> Ctx & { return ctxs[i]; };
    BatchResult r = measureSchedule(pick, n_batches, n_warmup, target_wall,
                                    rapl_available, rapl_core_available,
                                    session_idle_mw);
    std::cout << "baseline cfg" << i << " (P=" << ctxs[i].numCores
              << "): " << r.per_dispatch_us_min << " us, "
              << r.per_dispatch_energy_uj << " uJ" << std::endl;
    write_row("baseline", i, ctxs[i], r);
  }

  // Phase 2: alternating round-robin over the d configs (every dispatch a switch).
  {
    auto pick = [&](int iter) -> Ctx & { return ctxs[iter % depth]; };
    BatchResult r = measureSchedule(pick, n_batches, n_warmup, target_wall,
                                    rapl_available, rapl_core_available,
                                    session_idle_mw);
    std::cout << "alternating d=" << depth << ": " << r.per_dispatch_us_min
              << " us, " << r.per_dispatch_energy_uj << " uJ" << std::endl;
    write_row("alternating", -1, ctxs[0], r);
  }

  std::cout << "Wrote results to " << out_csv << std::endl;
  return 0;
}
