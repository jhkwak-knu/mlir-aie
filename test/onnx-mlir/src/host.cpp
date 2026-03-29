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

// Per-sample idle measurement window (seconds).
static constexpr double IDLE_SAMPLE_WINDOW_S = 0.2;
// Number of idle samples to collect; median is used as the idle baseline.
static constexpr int N_IDLE_SAMPLES = 5;

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
      ("n-warmup", "Number of warmup iterations (default: 3)",
       cxxopts::value<int>()->default_value("3"));
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
  {
    int64_t pkg0 = readRaplEnergyUj(RAPL_PKG_PATH);
    if (pkg0 > 0) {
      rapl_available = true;

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
            static_cast<double>(s1 - s0) / elapsed / 1000.0;
      }
      std::sort(idle_samples.begin(), idle_samples.end());
      idle_pkg_mw = idle_samples[N_IDLE_SAMPLES / 2];
    }
  }

  // ------------------------------------------------------
  // Measurement loop (warmup + measured, unified body)
  // ------------------------------------------------------
  // Every iteration executes the same tight sequence:
  //   memcpy staged→BO → sync(TO_DEVICE) → kernel → wait
  // Warmup iterations warm the NPU pipeline but are not counted
  // toward timing or energy statistics.
  double npu_time_total = 0;
  double npu_time_min = 1e18;
  double npu_time_max = 0;
  // step_time: memcpy + sync + dispatch (matches energy measurement scope)
  double step_time_total = 0;
  double step_time_min = 1e18;
  double step_time_max = 0;

  bool rapl_started = false;
  int64_t rapl_pkg_before = 0;
  std::chrono::steady_clock::time_point wall_start;

  // Pre-zero pres buffer once before the loop — value is irrelevant for
  // performance measurement (verified above), so skip per-step memset+sync.
  if (useInC) {
    memset(bufInC, 0, chunkCSize * sizeof(DATATYPE));
    bo_inC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  for (unsigned iter = 0; iter < num_iter; iter++) {
    bool is_measured = (iter >= static_cast<unsigned>(n_warmup_iterations));

    // At the first measured iteration: clear trace buffer and start RAPL.
    // Trace clear here ensures warmup iterations do not pollute trace data.
    if (!rapl_started && is_measured) {
      if (traceSz > 0) {
        memset(bo_trace.map<char *>(), 0, traceSz);
        bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
      if (rapl_available) {
        rapl_pkg_before = readRaplEnergyUj(RAPL_PKG_PATH);
      }
      wall_start = std::chrono::steady_clock::now();
      rapl_started = true;
    }

    double npu_time = 0;
    double step_time = 0;

    for (int step = 0; step < totalSteps; ++step) {
      const auto &ss = staged[step];

      // step_time bracket: includes memcpy + sync + dispatch
      auto t_step_0 = std::chrono::high_resolution_clock::now();

      // Fill BO buffers from pre-staged data
      memcpy(bufInA, ss.a.data(), chunkASize * sizeof(DATATYPE));
      memcpy(bufInB, ss.b.data(), chunkBSize * sizeof(DATATYPE));
      // Note: memset(bufOut) removed -- kernel zeros C internally when acc=false,
      // and accumulates in device-local buffer when acc=true (no host→device sync needed).

      bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      // npu_time bracket: dispatch + wait only
      auto t0 = std::chrono::high_resolution_clock::now();
      runKernel();
      auto t1 = std::chrono::high_resolution_clock::now();

      npu_time += std::chrono::duration<double, std::micro>(t1 - t0).count();
      step_time += std::chrono::duration<double, std::micro>(t1 - t_step_0).count();
    }

    if (is_measured) {
      npu_time_total += npu_time;
      npu_time_min = std::min(npu_time_min, npu_time);
      npu_time_max = std::max(npu_time_max, npu_time);
      step_time_total += step_time;
      step_time_min = std::min(step_time_min, step_time);
      step_time_max = std::max(step_time_max, step_time);
    }
  }

  // ------------------------------------------------------
  // End RAPL bracket and compute energy
  // ------------------------------------------------------
  double active_pkg_mw = -1.0;
  double npu_power_mw = -1.0;
  double npu_energy_uj = -1.0;
  double npu_energy_per_iter_uj = -1.0;
  double wall_elapsed_s = -1.0;

  if (rapl_started && rapl_available) {
    auto wall_stop = std::chrono::steady_clock::now();
    int64_t rapl_pkg_after = readRaplEnergyUj(RAPL_PKG_PATH);

    wall_elapsed_s = std::chrono::duration<double>(wall_stop - wall_start).count();
    double active_pkg_uj = static_cast<double>(rapl_pkg_after - rapl_pkg_before);
    active_pkg_mw = active_pkg_uj / wall_elapsed_s / 1000.0;

    // NPU contribution = active pkg minus idle-with-IO baseline rate.
    // idle-with-IO runs the same memcpy+sync, so CPU/bus activity cancels out.
    npu_energy_uj = active_pkg_uj - (idle_pkg_mw * wall_elapsed_s * 1000.0);
    npu_power_mw = npu_energy_uj / wall_elapsed_s / 1000.0;
    npu_energy_per_iter_uj = npu_energy_uj / n_iterations;
  } else if (rapl_started) {
    // RAPL unavailable but wall time still meaningful
    auto wall_stop = std::chrono::steady_clock::now();
    wall_elapsed_s = std::chrono::duration<double>(wall_stop - wall_start).count();
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

  if (rapl_available && rapl_started) {
    std::cout << std::endl
              << std::fixed << std::setprecision(1)
              << "Idle pkg (median): " << idle_pkg_mw << " mW" << std::endl
              << "Active pkg: " << active_pkg_mw << " mW" << std::endl
              << "NPU power: " << npu_power_mw << " mW" << std::endl
              << "NPU energy: " << npu_energy_uj << " uJ ("
              << npu_power_mw << " mW, "
              << std::setprecision(2) << wall_elapsed_s << "s)" << std::endl;
    std::cout.copyfmt(oldState);
  }

  if (!jsonOutputPath.empty()) {
    writeJsonResult(jsonOutputPath, errors ? "FAIL" : "PASS",
                    errors, n_iterations, n_warmup_iterations,
                    avgUs, npu_time_min, npu_time_max,
                    stepAvgUs, step_time_min, step_time_max,
                    idle_pkg_mw, active_pkg_mw, npu_power_mw,
                    npu_energy_uj, npu_energy_per_iter_uj, wall_elapsed_s);
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
