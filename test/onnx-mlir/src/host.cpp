//===- host.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "tiling_param.h"
#include "tile_ops.h"
#include "tile_order.h"

// ---------------------------------------------------------------------------
// RAPL energy measurement helpers
// ---------------------------------------------------------------------------
// sysfs path for package-level RAPL counter (Intel/AMD).
// NPU contribution = active_pkg - idle_pkg_rate * elapsed.
// idle-with-IO baseline runs the same memcpy+sync without kernel execution,
// so CPU/bus activity cancels out in the difference.
static const char *RAPL_PKG_PATH =
    "/sys/class/powercap/intel-rapl:0/energy_uj";
static const char *RAPL_CORE_PATH =
    "/sys/class/powercap/intel-rapl:0:0/energy_uj";

// RAPL counter wrap-around threshold (read from max_energy_range_uj).
// Fallback: 65,532,610,987 uJ (~65.5 J) for AMD Ryzen AI package-0.
static constexpr int64_t RAPL_MAX_ENERGY_RANGE_UJ = 65532610987LL;

// Compute RAPL delta with wrap-around handling.
static int64_t raplDelta(int64_t after, int64_t before) {
  int64_t d = after - before;
  if (d < 0) d += RAPL_MAX_ENERGY_RANGE_UJ;
  return d;
}

// Read a single RAPL energy counter.  Returns 0 on failure so callers
// can detect unavailability without exceptions.
static int64_t readRaplEnergyUj(const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) return 0;
  int64_t val = 0;
  if (fscanf(fp, "%ld", &val) != 1) val = 0;
  fclose(fp);
  return val;
}

// Per-sample idle measurement window (seconds).
static constexpr double IDLE_SAMPLE_WINDOW_S = 0.2;
// Number of idle samples to collect; median is used as the idle baseline.
static constexpr int N_IDLE_SAMPLES = 10;

// Per-batch idle bracketing: enough samples to average out OS scheduling noise.
static constexpr double BRACKET_IDLE_WINDOW_S = 0.3;
static constexpr int N_BRACKET_IDLE_SAMPLES = 5;

// Compute median of a small sorted vector (caller must ensure non-empty).
static double medianSorted(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Measure idle power (mW) over a short window.  Returns median of N samples.
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

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("onnx_matmul");
  options.add_options()
      ("tc-json", "Path to tc.json tiling config",
       cxxopts::value<std::string>()->default_value("out/tc.json"))
      ("json-output", "Write structured JSON result to this file",
       cxxopts::value<std::string>()->default_value(""))
      ("strict-verify", "Use exact float comparison instead of epsilon tolerance",
       cxxopts::value<bool>()->default_value("false"))
      ("n-iterations", "Number of measured iterations (default: 10)",
       cxxopts::value<int>()->default_value("10"))
      ("n-warmup", "Number of warmup iterations (default: 10)",
       cxxopts::value<int>()->default_value("10"))
      ("n-batches", "Number of outer batches for double-loop min (0=legacy mode)",
       cxxopts::value<int>()->default_value("5"))
      ("target-batch-wall-s", "Target wall-clock per batch in seconds (inner N auto-determined)",
       cxxopts::value<double>()->default_value("1.0"))
      ("diag-json", "Optional path for per-case diagnostic JSON (idle samples, per-batch stats)",
       cxxopts::value<std::string>()->default_value(""));
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  bool verify = vm["verify"].as<bool>();
  bool strictVerify = vm["strict-verify"].as<bool>();
  std::string jsonOutputPath = vm["json-output"].as<std::string>();

  auto tp = loadTilingParam(vm["tc-json"].as<std::string>());
  int matCSize = tp.M * tp.N;

  // Chunk sizes determine how many elements are transferred per DMA batch.
  // Only the reuse axis contributes multiple temporal iterations to a single chunk;
  // other axes are handled by the host outer/inner loop.
  int chunkTPm = (tp.tpOrder[0] == AXIS_M) ? tp.TPm : 1;
  int chunkTPn = (tp.tpOrder[0] == AXIS_N) ? tp.TPn : 1;
  int chunkTPk = (tp.tpOrder[0] == AXIS_K) ? tp.TPk : 1;

  int chunkASize = ((tp.TM * tp.TK) * tp.SPm) * chunkTPm * chunkTPk;
  int chunkBSize = ((tp.TN * tp.TK) * tp.SPn) * chunkTPn * chunkTPk;
  int chunkCSize = ((tp.TM * tp.TN) * tp.SPm * tp.SPn) * chunkTPm * chunkTPn;
  // PKT_HDR_ELEMS accounts for the 4-byte packet header prepended to each output tile
  int chunkOutCSize = ((tp.TM * tp.TN + PKT_HDR_ELEMS) * tp.SPm * tp.SPn) * chunkTPm * chunkTPn;

  // Partial sum input (pres) needed when K is split across temporal iterations
  // AND K is not the innermost (reuse) axis (otherwise tiles accumulate locally).
  bool useInC = (tp.TPk > 1) && (tp.tpOrder[0] != AXIS_K);

  auto ms = initMatrices(tp, verbosity);
  auto &matA = ms.A;
  auto &matB = ms.B;
  auto &matC = ms.C;
  auto &matCRef = ms.CRef;

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, chunkASize * sizeof(DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB = xrt::bo(device, chunkBSize * sizeof(DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_outC = xrt::bo(device, chunkOutCSize * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  xrt::bo bo_inC;
  if (useInC) {
    bo_inC = xrt::bo(device, chunkCSize * sizeof(DATATYPE),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  }

  int traceSz = vm["trace_sz"].as<int>();
  std::string traceFile = vm["trace_file"].as<std::string>();

  xrt::bo bo_trace;
  if (traceSz > 0) {
    // trace arg is last: group_id(6) without pres, group_id(7) with pres
    int traceGroupId = useInC ? 7 : 6;
    bo_trace = xrt::bo(device, traceSz, XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(traceGroupId));
    memset(bo_trace.map<char*>(), 0, traceSz);
  }

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer bo_inA
  DATATYPE *bufInA = bo_inA.map<DATATYPE *>();
  memset(bufInA, 0, chunkASize * sizeof(DATATYPE));

  // Initialize buffer bo_inB
  DATATYPE *bufInB = bo_inB.map<DATATYPE *>();
  memset(bufInB, 0, chunkBSize * sizeof(DATATYPE));

  // Zero out buffer bo_outC
  DATATYPE *bufOut = bo_outC.map<DATATYPE *>();
  memset(bufOut, 0, chunkOutCSize * sizeof(DATATYPE));

  // Initialize buffer bo_inC
  DATATYPE *bufInC;
  if (useInC) {
    bufInC = bo_inC.map<DATATYPE *>();
    memset(bufInC, 0, chunkCSize * sizeof(DATATYPE));
  }

  // ------------------------------------------------------
  // Run configuration
  // ------------------------------------------------------
  int n_iterations = vm["n-iterations"].as<int>();
  int n_warmup_iterations = vm["n-warmup"].as<int>();
  unsigned num_iter = n_iterations + n_warmup_iterations;
  int errors = 0;

  auto toc = buildTileOrders(tp);
  int totalSteps = toc.outerTP * toc.innerTP;

  // Kernel dispatch helper — avoids repeating BO argument permutations.
  unsigned int opcode = 3;
  auto runKernel = [&]() {
    xrt::run run;
    if (useInC && traceSz > 0)
      run = kernel(opcode, bo_instr, instr_v.size(),
                   bo_inA, bo_inB, bo_outC, bo_inC, bo_trace);
    else if (useInC)
      run = kernel(opcode, bo_instr, instr_v.size(),
                   bo_inA, bo_inB, bo_outC, bo_inC);
    else if (traceSz > 0)
      run = kernel(opcode, bo_instr, instr_v.size(),
                   bo_inA, bo_inB, bo_outC, bo_trace);
    else
      run = kernel(opcode, bo_instr, instr_v.size(),
                   bo_inA, bo_inB, bo_outC);
    run.wait();
  };

  // ------------------------------------------------------
  // Pre-stage input tile data
  // ------------------------------------------------------
  // A and B tiles depend only on tiling config and constant source matrices.
  // Extract + convert to mmul layout once; memcpy into BO during measurement.
  struct StagedStep {
    std::vector<DATATYPE> a;
    std::vector<DATATYPE> b;
    std::vector<std::pair<int,int>> cTileOrder;
  };
  std::vector<StagedStep> staged(totalSteps);

  for (int oi = 0; oi < toc.outerTP; ++oi) {
    for (int ij = 0; ij < toc.innerTP; ++ij) {
      int step = oi * toc.innerTP + ij;
      auto &ss = staged[step];

      // Stage A: tile extraction + mmul layout conversion
      std::vector<std::pair<int,int>> aTileOrder;
      for (const auto &base : toc.chunkATileOrderBase) {
        aTileOrder.emplace_back(
            base.first  + toc.matAInnerOffset.first  * ij + toc.matAOuterOffset.first  * oi,
            base.second + toc.matAInnerOffset.second * ij + toc.matAOuterOffset.second * oi);
      }
      ss.a.resize(chunkASize, static_cast<DATATYPE>(0));
      int aCount = 0;
      for (const auto &[tr, tc] : aTileOrder) {
        auto tile = extract_tile_1d_strict<DATATYPE>(matA, tp.M, tp.K, tp.TM, tp.TK, tr, tc);
        auto tiled = tile_to_mmul_layout<DATATYPE, MMUL_R, MMUL_S>(tile.data(), tp.TM, tp.TK);
        std::copy(tiled.begin(), tiled.end(), ss.a.data() + aCount);
        aCount += tiled.size();
      }

      // Stage B: tile extraction + mmul layout conversion
      std::vector<std::pair<int,int>> bTileOrder;
      for (const auto &base : toc.chunkBTileOrderBase) {
        bTileOrder.emplace_back(
            base.first  + toc.matBInnerOffset.first  * ij + toc.matBOuterOffset.first  * oi,
            base.second + toc.matBInnerOffset.second * ij + toc.matBOuterOffset.second * oi);
      }
      ss.b.resize(chunkBSize, static_cast<DATATYPE>(0));
      int bCount = 0;
      for (const auto &[tr, tc] : bTileOrder) {
        auto tile = extract_tile_1d_strict<DATATYPE>(matB, tp.N, tp.K, tp.TN, tp.TK, tr, tc);
        auto tiled = tile_to_mmul_layout<DATATYPE, MMUL_T, MMUL_S>(tile.data(), tp.TN, tp.TK);
        std::copy(tiled.begin(), tiled.end(), ss.b.data() + bCount);
        bCount += tiled.size();
      }

      // Build C tile order (used by verification for output writeback)
      for (const auto &base : toc.chunkCTileOrderBase) {
        ss.cTileOrder.emplace_back(
            base.first  + toc.matCInnerOffset.first  * ij + toc.matCOuterOffset.first  * oi,
            base.second + toc.matCInnerOffset.second * ij + toc.matCOuterOffset.second * oi);
      }

      if (verbosity >= 2) {
        printMatrix("Staged A [step " + std::to_string(step) + "]",
                    ss.a, chunkASize / tp.TK, tp.TK);
        printMatrix("Staged B [step " + std::to_string(step) + "]",
                    ss.b, chunkBSize / tp.TK, tp.TK);
      }
    }
  }

  // Sync instruction buffer once (constant across all invocations)
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Verification (one complete iteration with correct data)
  // ------------------------------------------------------
  if (verify) {
    if (verbosity >= 1)
      std::cout << "Running verification iteration...\n";

    std::fill(matC.begin(), matC.end(), static_cast<DATATYPE>(0));

    for (int step = 0; step < totalSteps; ++step) {
      const auto &ss = staged[step];
      memcpy(bufInA, ss.a.data(), chunkASize * sizeof(DATATYPE));
      memcpy(bufInB, ss.b.data(), chunkBSize * sizeof(DATATYPE));
      memset(bufOut, 0, chunkOutCSize * sizeof(DATATYPE));

      if (useInC) {
        // Extract partial sums from current matC (accumulated from prior steps)
        int cCount = 0;
        for (const auto &[tr, tc] : ss.cTileOrder) {
          auto tile = extract_tile_1d_strict<DATATYPE>(matC, tp.M, tp.N, tp.TM, tp.TN, tr, tc);
          auto tiled = tile_to_mmul_layout<DATATYPE, MMUL_R, MMUL_T>(tile.data(), tp.TM, tp.TN);
          std::copy(tiled.begin(), tiled.end(), bufInC + cCount);
          cCount += tiled.size();
        }
      }

      bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      if (useInC) bo_inC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      runKernel();
      bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

      // Decode packet headers and write output tiles back to matC.
      int repeatCount = (toc.reuseTPAxis != AXIS_K) ? toc.reuseTP : 1;
      for (int ri = 0; ri < repeatCount; ++ri) {
        int outerOff = tp.SPm * tp.SPn * ri;
        for (int rj = 0; rj < static_cast<int>((tp.SPm * tp.SPn) / 4); ++rj) {
          int innerOff = 4 * rj;
          for (int rk = 0; rk < 4; ++rk) {
            int idx = outerOff + innerOff + rk;
            uint32_t hdr;
            std::memcpy(&hdr, &bufOut[(tp.TM * tp.TN + PKT_HDR_ELEMS) * idx], sizeof(hdr));
            uint32_t rawId = hdr & 0x1F, pid = 0;
            while (rawId > 1) { rawId >>= 1; ++pid; }
            auto tilePos = ss.cTileOrder[outerOff + innerOff + pid];
            const DATATYPE *tiledOut =
                &bufOut[(tp.TM * tp.TN + PKT_HDR_ELEMS) * idx + PKT_HDR_ELEMS];
            auto val = untile_from_mmul_layout<DATATYPE, MMUL_R, MMUL_T>(
                tiledOut, tp.TM, tp.TN);
            write_tile_1d_strict<DATATYPE>(
                matC, tp.M, tp.N, tp.TM, tp.TN, tilePos.first, tilePos.second, val);
          }
        }
      }
    }

    // Compare against reference
    if (verbosity >= 1)
      std::cout << "Verifying results..." << std::endl;
    constexpr float REL_TOL = 1e-2f;
    constexpr float ABS_TOL = 5e-1f;
    for (int vi = 0; vi < static_cast<int>(tp.M); ++vi) {
      for (int vj = 0; vj < static_cast<int>(tp.N); ++vj) {
        float ref = matCRef[vi * tp.N + vj];
        float out = matC[vi * tp.N + vj];
        bool match = strictVerify
            ? (out == ref)
            : (std::fabs(out - ref) <= ABS_TOL + REL_TOL * std::fabs(ref));
        if (!match) {
          if (verbosity >= 1)
            std::cout << "Error in output " << out << " != " << ref << std::endl;
          errors++;
        } else if (verbosity >= 1) {
          std::cout << "Correct output " << out << " == " << ref << std::endl;
        }
      }
    }
  }

  // ------------------------------------------------------
  // Idle (sleep) RAPL baseline — multi-sample median
  // ------------------------------------------------------
  // Take N_IDLE_SAMPLES measurements of IDLE_SAMPLE_WINDOW_S each.
  // Use the median to reduce sensitivity to OS scheduling noise and
  // transient system activity that plagued the single-sample approach.
  double idle_pkg_mw = -1.0;
  bool rapl_available = false;
  bool rapl_core_available = false;
  // Diagnostic captures
  std::vector<double> diag_idle_pre;
  std::vector<double> diag_idle_post;
  std::vector<double> diag_batch_wall_s;
  std::vector<double> diag_batch_active_uj;
  std::vector<double> diag_batch_core_uj;
  std::vector<double> diag_batch_e_per_iter;
  std::vector<double> diag_warmup_t_step;
  std::vector<double> diag_bracket_idle_before;
  std::vector<double> diag_bracket_idle_after;
  {
    int64_t pkg0 = readRaplEnergyUj(RAPL_PKG_PATH);
    if (pkg0 > 0) {
      rapl_available = true;
      int64_t core0 = readRaplEnergyUj(RAPL_CORE_PATH);
      if (core0 > 0) rapl_core_available = true;

      std::vector<double> idle_samples(N_IDLE_SAMPLES);
      for (int s = 0; s < N_IDLE_SAMPLES; s++) {
        int64_t s0 = readRaplEnergyUj(RAPL_PKG_PATH);
        auto t0 = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(
            std::chrono::duration<double>(IDLE_SAMPLE_WINDOW_S));
        auto t1 = std::chrono::steady_clock::now();
        int64_t s1 = readRaplEnergyUj(RAPL_PKG_PATH);
        double elapsed =
            std::chrono::duration<double>(t1 - t0).count();
        idle_samples[s] =
            static_cast<double>(raplDelta(s1, s0)) / elapsed / 1000.0;
      }
      // Save unsorted samples for diagnostics (before sorting for median).
      diag_idle_pre = idle_samples;
      std::sort(idle_samples.begin(), idle_samples.end());
      idle_pkg_mw = idle_samples[N_IDLE_SAMPLES / 2];
    }
  }

  // ------------------------------------------------------
  // Measurement: double-loop min or legacy mode
  // ------------------------------------------------------
  int n_batches = vm["n-batches"].as<int>();
  double target_batch_wall_s = vm["target-batch-wall-s"].as<double>();
  bool use_batch_mode = (n_batches > 0);

  // Legacy per-iteration stats (always populated for backward compat)
  double npu_time_total = 0;
  double npu_time_min = 1e18;
  double npu_time_max = 0;
  double step_time_total = 0;
  double step_time_min = 1e18;
  double step_time_max = 0;

  // Batch-mode stats
  double batch_min_avg_us = -1.0;
  double batch_min_step_avg_us = -1.0;
  double batch_min_energy_per_iter_uj = -1.0;
  int batch_n_inner = 0;

  // Extended diagnostics for CSV (populated in batch mode)
  double idle_post_mw = -1.0;
  double bracket_idle_mean_mw = -1.0;
  double batch_energy_cv_pct = -1.0;
  double batch_step_cv_pct = -1.0;
  double batch_best_wall_s = -1.0;
  double batch_best_active_uj = -1.0;
  double core_energy_per_iter_uj = -1.0;
  double active_pkg_mw = -1.0;
  double npu_power_mw = -1.0;
  double npu_energy_uj = -1.0;
  double npu_energy_per_iter_uj = -1.0;
  double wall_elapsed_s = -1.0;

  // Pre-zero pres buffer once — value is irrelevant for performance measurement.
  if (useInC) {
    memset(bufInC, 0, chunkCSize * sizeof(DATATYPE));
    bo_inC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // Helper lambda: run one complete matmul iteration (all temporal steps).
  // Returns {npu_time_us, step_time_us}.
  auto runOneIteration = [&]() -> std::pair<double, double> {
    double npu_time = 0;
    double step_time = 0;
    for (int step = 0; step < totalSteps; ++step) {
      const auto &ss = staged[step];
      auto t_step_0 = std::chrono::high_resolution_clock::now();
      memcpy(bufInA, ss.a.data(), chunkASize * sizeof(DATATYPE));
      memcpy(bufInB, ss.b.data(), chunkBSize * sizeof(DATATYPE));
      bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto t0 = std::chrono::high_resolution_clock::now();
      runKernel();
      auto t1 = std::chrono::high_resolution_clock::now();
      npu_time += std::chrono::duration<double, std::micro>(t1 - t0).count();
      step_time += std::chrono::duration<double, std::micro>(t1 - t_step_0).count();
    }
    return {npu_time, step_time};
  };

  if (use_batch_mode) {
    // ---- Double-loop min measurement ----
    // Warmup outside outer loop; use MIN of warmup t_step to size n_inner.
    // Why t_step: batch wall time is driven by per-iter full step (memcpy
    // +sync+dispatch+wait), not NPU-only time. Why min: first iteration
    // carries cold-start overhead that overestimates single_iter_us and
    // undersizes n_inner, shrinking batch wall below target.
    double single_iter_us = 1e18;
    for (int w = 0; w < n_warmup_iterations; ++w) {
      auto [t_npu, t_step] = runOneIteration();
      diag_warmup_t_step.push_back(t_step);
      if (t_step < single_iter_us) single_iter_us = t_step;
    }
    if (single_iter_us <= 0 || single_iter_us >= 1e18) single_iter_us = 1.0;

    // Dynamically determine inner iteration count
    double single_iter_s = single_iter_us / 1e6;
    int n_inner = std::max(10, static_cast<int>(std::ceil(target_batch_wall_s / single_iter_s)));
    batch_n_inner = n_inner;

    std::cout << "Batch mode: K=" << n_batches
              << ", N=" << n_inner
              << " (single_iter=" << std::fixed << std::setprecision(1)
              << single_iter_us << "us, target_wall="
              << target_batch_wall_s << "s)" << std::endl;

    // Clear trace buffer before batches (use last batch for trace)
    if (traceSz > 0) {
      memset(bo_trace.map<char *>(), 0, traceSz);
      bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    double best_batch_npu_avg = 1e18;
    double best_batch_step_avg = 1e18;
    double best_batch_energy = 1e18;
    // Accumulators for legacy compat (aggregate across all batches)
    int total_measured_iters = 0;

    for (int batch = 0; batch < n_batches; ++batch) {
      // Per-batch idle bracket: measure idle BEFORE this batch
      double bracket_idle_before_mw = idle_pkg_mw;  // fallback to session-level
      if (rapl_available) {
        bracket_idle_before_mw = measureIdlePowerMw(
            RAPL_PKG_PATH, N_BRACKET_IDLE_SAMPLES, BRACKET_IDLE_WINDOW_S);
      }

      // RAPL start for this batch
      int64_t rapl_pkg_before = 0;
      int64_t rapl_core_before = 0;
      if (rapl_available)
        rapl_pkg_before = readRaplEnergyUj(RAPL_PKG_PATH);
      if (rapl_core_available)
        rapl_core_before = readRaplEnergyUj(RAPL_CORE_PATH);
      auto batch_wall_start = std::chrono::steady_clock::now();

      double batch_npu_total = 0;
      double batch_step_total = 0;

      for (int iter = 0; iter < n_inner; ++iter) {
        auto [t_npu, t_step] = runOneIteration();
        batch_npu_total += t_npu;
        batch_step_total += t_step;

        // Legacy per-iteration stats
        npu_time_total += t_npu;
        npu_time_min = std::min(npu_time_min, t_npu);
        npu_time_max = std::max(npu_time_max, t_npu);
        step_time_total += t_step;
        step_time_min = std::min(step_time_min, t_step);
        step_time_max = std::max(step_time_max, t_step);
      }

      // RAPL end for this batch
      auto batch_wall_stop = std::chrono::steady_clock::now();
      double batch_wall_s = std::chrono::duration<double>(batch_wall_stop - batch_wall_start).count();

      // Per-batch idle bracket: measure idle AFTER this batch
      double bracket_idle_after_mw = bracket_idle_before_mw;
      if (rapl_available) {
        bracket_idle_after_mw = measureIdlePowerMw(
            RAPL_PKG_PATH, N_BRACKET_IDLE_SAMPLES, BRACKET_IDLE_WINDOW_S);
      }

      // Use mean of before/after bracket as this batch's idle baseline
      double batch_idle_mw = (bracket_idle_before_mw + bracket_idle_after_mw) / 2.0;
      diag_bracket_idle_before.push_back(bracket_idle_before_mw);
      diag_bracket_idle_after.push_back(bracket_idle_after_mw);

      double batch_npu_avg = batch_npu_total / n_inner;
      double batch_step_avg = batch_step_total / n_inner;
      double batch_energy_per_iter = -1.0;

      if (rapl_available) {
        int64_t rapl_pkg_after = readRaplEnergyUj(RAPL_PKG_PATH);
        double active_uj = static_cast<double>(raplDelta(rapl_pkg_after, rapl_pkg_before));
        double core_uj = 0.0;
        if (rapl_core_available) {
          int64_t rapl_core_after = readRaplEnergyUj(RAPL_CORE_PATH);
          core_uj = static_cast<double>(raplDelta(rapl_core_after, rapl_core_before));
        }
        double npu_uj = active_uj - (batch_idle_mw * batch_wall_s * 1000.0);
        batch_energy_per_iter = npu_uj / n_inner;
        diag_batch_wall_s.push_back(batch_wall_s);
        diag_batch_active_uj.push_back(active_uj);
        diag_batch_core_uj.push_back(core_uj);
        diag_batch_e_per_iter.push_back(batch_energy_per_iter);
      }

      // Independent min selection: time and energy separately
      if (batch_npu_avg < best_batch_npu_avg) {
        best_batch_npu_avg = batch_npu_avg;
        best_batch_step_avg = batch_step_avg;
      }
      if (batch_energy_per_iter >= 0 && batch_energy_per_iter < best_batch_energy) {
        best_batch_energy = batch_energy_per_iter;
        batch_best_wall_s = batch_wall_s;
        if (!diag_batch_active_uj.empty())
          batch_best_active_uj = diag_batch_active_uj.back();
        if (!diag_batch_core_uj.empty())
          core_energy_per_iter_uj = diag_batch_core_uj.back() / n_inner;
      }

      total_measured_iters += n_inner;
    }

    // Post-batch idle measurement for drift detection.
    if (rapl_available) {
      diag_idle_post.resize(N_IDLE_SAMPLES);
      for (int s = 0; s < N_IDLE_SAMPLES; ++s) {
        int64_t s0 = readRaplEnergyUj(RAPL_PKG_PATH);
        auto t0 = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(
            std::chrono::duration<double>(IDLE_SAMPLE_WINDOW_S));
        auto t1 = std::chrono::steady_clock::now();
        int64_t s1 = readRaplEnergyUj(RAPL_PKG_PATH);
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        diag_idle_post[s] = static_cast<double>(raplDelta(s1, s0)) / elapsed / 1000.0;
      }
    }

    batch_min_avg_us = best_batch_npu_avg;
    batch_min_step_avg_us = best_batch_step_avg;
    batch_min_energy_per_iter_uj = (best_batch_energy < 1e18) ? best_batch_energy : -1.0;

    // Compute extended diagnostics for CSV
    // idle_post median
    if (!diag_idle_post.empty()) {
      auto sorted_post = diag_idle_post;
      std::sort(sorted_post.begin(), sorted_post.end());
      idle_post_mw = sorted_post[sorted_post.size() / 2];
    }
    // bracket idle mean across all batches
    if (!diag_bracket_idle_before.empty()) {
      double sum = 0;
      for (size_t i = 0; i < diag_bracket_idle_before.size(); ++i)
        sum += (diag_bracket_idle_before[i] + diag_bracket_idle_after[i]) / 2.0;
      bracket_idle_mean_mw = sum / diag_bracket_idle_before.size();
    }
    // batch energy CV (coefficient of variation %)
    if (diag_batch_e_per_iter.size() >= 2) {
      double mean = 0;
      for (double v : diag_batch_e_per_iter) mean += v;
      mean /= diag_batch_e_per_iter.size();
      double var = 0;
      for (double v : diag_batch_e_per_iter) var += (v - mean) * (v - mean);
      var /= (diag_batch_e_per_iter.size() - 1);
      if (mean > 0) batch_energy_cv_pct = std::sqrt(var) / mean * 100.0;
    }
    // batch step time CV: compute per-batch step avg, then CV across batches
    {
      // diag_batch_wall_s has one entry per batch; step avg per batch = wall/n_inner*1e6
      // but more precisely, use the actual batch step totals:
      // We already have per-iteration step times accumulated in step_time_total.
      // For CV, recompute per-batch step avgs from the iteration data.
      // Simpler: use diag_batch_wall_s as a proxy (wall time ~ step time * n_inner).
      if (diag_batch_wall_s.size() >= 2) {
        double mean = 0;
        for (double v : diag_batch_wall_s) mean += v;
        mean /= diag_batch_wall_s.size();
        double var = 0;
        for (double v : diag_batch_wall_s) var += (v - mean) * (v - mean);
        var /= (diag_batch_wall_s.size() - 1);
        if (mean > 0) batch_step_cv_pct = std::sqrt(var) / mean * 100.0;
      }
    }

    // Populate legacy fields from batch aggregates
    n_iterations = total_measured_iters;
    npu_energy_per_iter_uj = batch_min_energy_per_iter_uj;
    wall_elapsed_s = -1.0;  // not meaningful in batch mode (multiple RAPL brackets)

  } else {
    // ---- Legacy single-loop measurement ----
    bool rapl_started = false;
    int64_t rapl_pkg_before = 0;
    std::chrono::steady_clock::time_point wall_start;

    for (unsigned iter = 0; iter < num_iter; iter++) {
      bool is_measured = (iter >= static_cast<unsigned>(n_warmup_iterations));

      if (!rapl_started && is_measured) {
        if (traceSz > 0) {
          memset(bo_trace.map<char *>(), 0, traceSz);
          bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (rapl_available)
          rapl_pkg_before = readRaplEnergyUj(RAPL_PKG_PATH);
        wall_start = std::chrono::steady_clock::now();
        rapl_started = true;
      }

      auto [t_npu, t_step] = runOneIteration();

      if (is_measured) {
        npu_time_total += t_npu;
        npu_time_min = std::min(npu_time_min, t_npu);
        npu_time_max = std::max(npu_time_max, t_npu);
        step_time_total += t_step;
        step_time_min = std::min(step_time_min, t_step);
        step_time_max = std::max(step_time_max, t_step);
      }
    }

    if (rapl_started && rapl_available) {
      auto wall_stop = std::chrono::steady_clock::now();
      int64_t rapl_pkg_after = readRaplEnergyUj(RAPL_PKG_PATH);
      wall_elapsed_s = std::chrono::duration<double>(wall_stop - wall_start).count();
      double active_pkg_uj = static_cast<double>(raplDelta(rapl_pkg_after, rapl_pkg_before));
      active_pkg_mw = active_pkg_uj / wall_elapsed_s / 1000.0;
      npu_energy_uj = active_pkg_uj - (idle_pkg_mw * wall_elapsed_s * 1000.0);
      npu_power_mw = npu_energy_uj / wall_elapsed_s / 1000.0;
      npu_energy_per_iter_uj = npu_energy_uj / n_iterations;
    } else if (rapl_started) {
      auto wall_stop = std::chrono::steady_clock::now();
      wall_elapsed_s = std::chrono::duration<double>(wall_stop - wall_start).count();
    }
  }

  // ------------------------------------------------------
  // Save trace data (last measurement iteration)
  // ------------------------------------------------------
  if (traceSz > 0) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint32_t *traceOut = reinterpret_cast<uint32_t *>(bo_trace.map<char *>());
    size_t totalWords = static_cast<size_t>(traceSz) / sizeof(uint32_t);

    size_t tStart = 0;
    while (tStart < totalWords && traceOut[tStart] == 0) ++tStart;
    size_t tEnd = totalWords;
    while (tEnd > tStart && traceOut[tEnd - 1] == 0) --tEnd;

    FILE *fp = fopen(traceFile.c_str(), "w");
    if (fp) {
      for (size_t ti = tStart; ti < tEnd; ti++)
        fprintf(fp, "%08x\n", traceOut[ti]);
      fclose(fp);
    }
    std::cout << "Trace data written to " << traceFile
              << " (" << traceSz << " bytes)" << std::endl;
  }

  // ------------------------------------------------------
  // Report results
  // ------------------------------------------------------
  std::cout << std::endl
            << "Number of iterations: " << n_iterations
            << " (warmup iterations: " << n_warmup_iterations << ")"
            << std::endl;

  std::ios oldState(nullptr);
  oldState.copyfmt(std::cout);

  double avgUs = npu_time_total / n_iterations;
  double stepAvgUs = step_time_total / n_iterations;

  std::cout << std::endl
            << "Avg NPU time: " << std::fixed << std::setprecision(2)
            << avgUs << "us." << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << std::fixed << std::setprecision(2)
            << npu_time_min << "us." << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << std::fixed << std::setprecision(2)
            << npu_time_max << "us." << std::endl;

  std::cout << std::endl
            << "Step time (memcpy+sync+dispatch):" << std::endl
            << "  Avg: " << std::fixed << std::setprecision(2) << stepAvgUs << "us"
            << "  Min: " << step_time_min << "us"
            << "  Max: " << step_time_max << "us" << std::endl;

  std::cout.copyfmt(oldState);

  if (rapl_available) {
    std::cout << std::endl
              << std::fixed << std::setprecision(1)
              << "Idle pkg (median): " << idle_pkg_mw << " mW" << std::endl;
    if (!use_batch_mode) {
      std::cout << "Active pkg: " << active_pkg_mw << " mW" << std::endl
                << "NPU power: " << npu_power_mw << " mW" << std::endl
                << "NPU energy: " << npu_energy_uj << " uJ ("
                << npu_power_mw << " mW, "
                << std::setprecision(2) << wall_elapsed_s << "s)" << std::endl;
    }
    std::cout.copyfmt(oldState);
  }

  // Batch-mode summary
  if (use_batch_mode) {
    std::cout << std::endl
              << "Batch min avg NPU time: " << std::fixed << std::setprecision(2)
              << batch_min_avg_us << "us" << std::endl
              << "Batch min avg step time: " << batch_min_step_avg_us << "us" << std::endl
              << "Batch min energy/iter: " << std::setprecision(1)
              << batch_min_energy_per_iter_uj << " uJ" << std::endl
              << "Batches: " << n_batches << ", inner: " << batch_n_inner << std::endl;
    std::cout.copyfmt(oldState);
  }

  std::string diagJsonPath = vm["diag-json"].as<std::string>();
  if (!diagJsonPath.empty() && use_batch_mode) {
    std::ofstream df(diagJsonPath);
    auto dump_vec = [&](const std::vector<double>& v) {
      df << "[";
      for (size_t i = 0; i < v.size(); ++i) {
        if (i) df << ",";
        df << v[i];
      }
      df << "]";
    };
    df << std::fixed << std::setprecision(3);
    df << "{\n";
    df << "  \"idle_pre_samples_mw\": ";    dump_vec(diag_idle_pre);   df << ",\n";
    df << "  \"idle_post_samples_mw\": ";   dump_vec(diag_idle_post);  df << ",\n";
    df << "  \"idle_pkg_mw_used\": " << idle_pkg_mw << ",\n";
    df << "  \"warmup_t_step_us\": ";       dump_vec(diag_warmup_t_step); df << ",\n";
    df << "  \"n_inner\": " << batch_n_inner << ",\n";
    df << "  \"batch_wall_s\": ";           dump_vec(diag_batch_wall_s); df << ",\n";
    df << "  \"batch_active_uj\": ";        dump_vec(diag_batch_active_uj); df << ",\n";
    df << "  \"batch_e_per_iter_uj\": ";    dump_vec(diag_batch_e_per_iter); df << ",\n";
    df << "  \"batch_core_uj\": ";           dump_vec(diag_batch_core_uj); df << ",\n";
    df << "  \"bracket_idle_before_mw\": "; dump_vec(diag_bracket_idle_before); df << ",\n";
    df << "  \"bracket_idle_after_mw\": ";  dump_vec(diag_bracket_idle_after); df << "\n";
    df << "}\n";
  }

  if (!jsonOutputPath.empty()) {
    ExtendedDiag ediag;
    ediag.idle_post_mw = idle_post_mw;
    ediag.bracket_idle_mean_mw = bracket_idle_mean_mw;
    ediag.batch_energy_cv_pct = batch_energy_cv_pct;
    ediag.batch_step_cv_pct = batch_step_cv_pct;
    ediag.batch_best_wall_s = batch_best_wall_s;
    ediag.batch_best_active_uj = batch_best_active_uj;
    ediag.core_energy_per_iter_uj = core_energy_per_iter_uj;
    writeJsonResult(jsonOutputPath, errors ? "FAIL" : "PASS",
                    errors, n_iterations, n_warmup_iterations,
                    avgUs, npu_time_min, npu_time_max,
                    stepAvgUs, step_time_min, step_time_max,
                    idle_pkg_mw, active_pkg_mw, npu_power_mw,
                    npu_energy_uj, npu_energy_per_iter_uj, wall_elapsed_s,
                    use_batch_mode ? n_batches : 0,
                    batch_n_inner,
                    batch_min_avg_us, batch_min_step_avg_us,
                    batch_min_energy_per_iter_uj,
                    ediag);
  }

  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
