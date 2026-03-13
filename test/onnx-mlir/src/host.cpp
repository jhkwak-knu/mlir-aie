//===- host.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "tiling_param.h"
#include "tile_ops.h"
#include "tile_order.h"

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("onnx_matmul");
  options.add_options()
      ("tc-json", "Path to tc.json tiling config",
       cxxopts::value<std::string>()->default_value("out/tc.json"))
      ("json-output", "Write structured JSON result to this file",
       cxxopts::value<std::string>()->default_value(""))
      ("strict-verify", "Use exact float comparison instead of epsilon tolerance",
       cxxopts::value<bool>()->default_value("false"));
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
  // Initialize run configs
  // ------------------------------------------------------
  int n_iterations = 10;
  int n_warmup_iterations = 3;
  unsigned num_iter = n_iterations + n_warmup_iterations;
  double npu_time_total = 0;
  double npu_time_min = 99999999;
  double npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  // toc holds pre-computed tile iteration orders and per-axis offsets.
  // outerTP/innerTP = temporal loop trip counts for the 2nd/1st non-reuse axes.
  // i iterates outerTP, j iterates innerTP; reuse axis is implicit (no host loop).
  auto toc = buildTileOrders(tp);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    double npu_time = 0;

    // Reset matC so previous iteration's partial sums don't leak into
    // the next iteration's pres input (inC is populated from matC).
    std::fill(matC.begin(), matC.end(), static_cast<DATATYPE>(0));

    for (int i = 0; i < toc.outerTP; ++i) {
      for (int j = 0; j < toc.innerTP; ++j) {
        // set chunk data
        if (verbosity >= 2)
          std::cout << "Set Data (" << (i * toc.innerTP) + j << "):\n";

        // Build per-iteration tile order by applying outer/inner offsets to
        // the base order. Offsets encode which axis each loop variable advances.
        std::vector<std::pair<int,int>> chunkATileOrder;
        for (auto &tileA : toc.chunkATileOrderBase) {
          int row = tileA.first + (toc.matAInnerOffset.first * j) + (toc.matAOuterOffset.first * i);
          int col = tileA.second + (toc.matAInnerOffset.second * j) + (toc.matAOuterOffset.second * i);
          chunkATileOrder.emplace_back(row, col);
        }

        if (verbosity >= 2) {
          std::cout << "chunkATileOrder: ";
          for (const auto& [r,c] : chunkATileOrder) {
            std::cout << "(" << r << "," << c << ") ";
          }
          std::cout << "\n";
        }

        int chunkACount = 0;
        for (const auto& [tileRow, tileCol] : chunkATileOrder) {
          auto tileVec = extract_tile_1d_strict<DATATYPE>(
              matA, tp.M, tp.K, tp.TM, tp.TK, tileRow, tileCol);

          // Convert row-major tile to mmul-tiled layout: [TM/4][TK/8][4*8]
          auto tiledA = tile_to_mmul_layout<DATATYPE, MMUL_R, MMUL_S>(
              tileVec.data(), tp.TM, tp.TK);

          if (chunkACount + tiledA.size() > chunkASize) {
            std::cerr << "Overflow while writing bufInA\n";
            std::exit(EXIT_FAILURE);
          }

          std::copy(tiledA.begin(), tiledA.end(), bufInA + chunkACount);
          chunkACount += tiledA.size();
        }
        if (verbosity >= 2) printMatrix("Chunk A", std::vector<DATATYPE>(bufInA, bufInA + chunkASize), chunkASize / tp.TK, tp.TK);

        std::vector<std::pair<int,int>> chunkBTileOrder;
        for (auto &tileB : toc.chunkBTileOrderBase) {
          int row = tileB.first + (toc.matBInnerOffset.first * j) + (toc.matBOuterOffset.first * i);
          int col = tileB.second + (toc.matBInnerOffset.second * j) + (toc.matBOuterOffset.second * i);
          chunkBTileOrder.emplace_back(row, col);
        }

        if (verbosity >= 2) {
          std::cout << "chunkBTileOrder: ";
          for (const auto& [r,c] : chunkBTileOrder) {
            std::cout << "(" << r << "," << c << ") ";
          }
          std::cout << "\n";
        }

        int chunkBCount = 0;
        for (const auto& [tileRow, tileCol] : chunkBTileOrder) {
          auto tileVec = extract_tile_1d_strict<DATATYPE>(
              matB, tp.N, tp.K, tp.TN, tp.TK, tileRow, tileCol);

          // Convert row-major tile to mmul-tiled layout: [TN/8][TK/8][8*8]
          // B is [N x K] row-major (pre-transposed); subtile is MMUL_T x MMUL_S.
          auto tiledB = tile_to_mmul_layout<DATATYPE, MMUL_T, MMUL_S>(
              tileVec.data(), tp.TN, tp.TK);

          if (chunkBCount + tiledB.size() > chunkBSize) {
            std::cerr << "Overflow while writing bufInB\n";
            std::exit(EXIT_FAILURE);
          }

          std::copy(tiledB.begin(), tiledB.end(), bufInB + chunkBCount);
          chunkBCount += tiledB.size();
        }
        if (verbosity >= 2) printMatrix("Chunk B", std::vector<DATATYPE>(bufInB, bufInB + chunkBSize), chunkBSize / tp.TK, tp.TK);

        // matrix C
        memset(bufOut, 0, chunkOutCSize * sizeof(DATATYPE));

        std::vector<std::pair<int,int>> chunkCTileOrder;
        for (auto &tileC : toc.chunkCTileOrderBase) {
          int row = tileC.first + (toc.matCInnerOffset.first * j) + (toc.matCOuterOffset.first * i);
          int col = tileC.second + (toc.matCInnerOffset.second * j) + (toc.matCOuterOffset.second * i);
          chunkCTileOrder.emplace_back(row, col);
        }

        if (verbosity >= 2) {
          std::cout << "chunkCTileOrder: ";
          for (const auto& [r,c] : chunkCTileOrder) {
            std::cout << "(" << r << "," << c << ") ";
          }
          std::cout << "\n";
        }

        if (useInC) {
          int chunkCCount = 0;
          for (const auto& [tileRow, tileCol] : chunkCTileOrder) {
            auto tileVec = extract_tile_1d_strict<DATATYPE>(
                matC, tp.M, tp.N, tp.TM, tp.TN, tileRow, tileCol);

            // Convert partial-sum C to mmul-tiled layout: [TM/4][TN/8][4*8]
            auto tiledC = tile_to_mmul_layout<DATATYPE, MMUL_R, MMUL_T>(
                tileVec.data(), tp.TM, tp.TN);

            if (chunkCCount + tiledC.size() > chunkCSize) {
              std::cerr << "Overflow while writing bufInC\n";
              std::exit(EXIT_FAILURE);
            }

            std::copy(tiledC.begin(), tiledC.end(), bufInC + chunkCCount);
            chunkCCount += tiledC.size();
          }
          if (verbosity >= 2) printMatrix("Chunk C", std::vector<DATATYPE>(bufInC, bufInC + chunkCSize), chunkCSize / tp.TN, tp.TN);
        }

        // sync host to device memories
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        if (useInC) {
          bo_inC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // Run kernel
        if (verbosity >= 1)
          std::cout << "Running Kernel.\n";

        auto start = std::chrono::high_resolution_clock::now();
        unsigned int opcode = 3;
        xrt::run run;
        if (useInC && traceSz > 0) {
          run = kernel(opcode, bo_instr, instr_v.size(),
                       bo_inA, bo_inB, bo_outC, bo_inC, bo_trace);
        } else if (useInC) {
          run = kernel(opcode, bo_instr, instr_v.size(),
                       bo_inA, bo_inB, bo_outC, bo_inC);
        } else if (traceSz > 0) {
          run = kernel(opcode, bo_instr, instr_v.size(),
                       bo_inA, bo_inB, bo_outC, bo_trace);
        } else {
          run = kernel(opcode, bo_instr, instr_v.size(),
                       bo_inA, bo_inB, bo_outC);
        }
        run.wait();
        auto stop = std::chrono::high_resolution_clock::now();

        npu_time += std::chrono::duration<double, std::micro>(stop - start).count();

        // Sync device to host memories
        bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        if (verbosity >= 2) {
          // When reuse axis != K, each reuse iteration produces separate output blocks
          int repeatCount = (toc.reuseTPAxis != AXIS_K) ? toc.reuseTP : 1;
          for (int i = 0; i < repeatCount; ++i) {
            for (int j = 0; j < (tp.SPm * tp.SPn); ++j) {
              uint32_t pkt_header, pkt_id;
              std::memcpy(&pkt_header, &bufOut[(tp.TM * tp.TN + PKT_HDR_ELEMS) * ((tp.SPm * tp.SPn) * i + j) + 0], sizeof(pkt_header));
              pkt_id = pkt_header & 0x1F;

              std::cout << "OutC[" << ((tp.SPm * tp.SPn) * i + j) << "] (packet id = " << pkt_id << "): ";
              for (int k = 0; k < (tp.TM * tp.TN); ++k) {
                std::cout << static_cast<float>(bufOut[(tp.TM * tp.TN + PKT_HDR_ELEMS) * ((tp.SPm * tp.SPn) * i + j) + k + PKT_HDR_ELEMS]) << " ";
              }
              std::cout << "\n";
            }
          }
        }

        // Store partial sums to matC.
        // When reuse axis is K, all tiles accumulate locally so repeatCount=1.
        // Otherwise, each reuse iteration produces distinct output tiles.
        int repeatCount = (toc.reuseTPAxis != AXIS_K) ? toc.reuseTP : 1;
        for (int i = 0; i < repeatCount; ++i) {
          int outerOffset = tp.SPm * tp.SPn * i;

          for (int j = 0; j < ((tp.SPm * tp.SPn) / 4); ++j) {
            int innerOffset =  4 * j;

            for (int k = 0; k < 4; ++k) {
              int idx = outerOffset + innerOffset + k;

              uint32_t packetHeader, packetId;
              std::memcpy(&packetHeader, &bufOut[(tp.TM * tp.TN + PKT_HDR_ELEMS) * idx], sizeof(packetHeader));
              // RES packet IDs are power-of-2: {1,2,4,8} → tile index {0,1,2,3}.
              // Extract 5-bit ID, then find the set bit position (log2).
              uint32_t rawId = packetHeader & 0x1F;
              packetId = 0;
              while (rawId > 1) { rawId >>= 1; ++packetId; }
              int matCIdx = outerOffset + innerOffset + packetId;

              std::pair<int,int> tilePos = chunkCTileOrder[matCIdx];

              const size_t tileElems  = static_cast<size_t>(tp.TM) * tp.TN;
              const size_t blockElems = tileElems + PKT_HDR_ELEMS;
              const size_t start      = blockElems * static_cast<size_t>(idx) + 1;
              const size_t end        = start + tileElems;

              if (verbosity >= 2) {
                std::cout
                  << "[k=" << k << "] "
                  << " idx=" << idx
                  << " packetId=" << packetId
                  << " matCIdx=" << matCIdx
                  << " tilePos=(" << tilePos.first << "," << tilePos.second << ") "
                  << "tileElems=" << static_cast<unsigned long long>(tileElems)
                  << " blockElems=" << static_cast<unsigned long long>(blockElems)
                  << " start=" << static_cast<unsigned long long>(start)
                  << " end=" << static_cast<unsigned long long>(end)
                  << " outerOffset=" << outerOffset
                  << " innerOffset=" << innerOffset
                  << "\n";
              }

              // Output tile is in mmul-tiled layout; convert back to row-major.
              const DATATYPE* tiledOut = &bufOut[(tp.TM * tp.TN + PKT_HDR_ELEMS) * idx + PKT_HDR_ELEMS];
              auto tileValue = untile_from_mmul_layout<DATATYPE, MMUL_R, MMUL_T>(
                  tiledOut, tp.TM, tp.TN);
              write_tile_1d_strict<DATATYPE>(matC, tp.M, tp.N, tp.TM, tp.TN, tilePos.first, tilePos.second, tileValue);
            }
          }
        }
      }
    }

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      // Clear trace buffer after last warmup so only measurement data remains.
      if (traceSz > 0 && iter == n_warmup_iterations - 1) {
        memset(bo_trace.map<char*>(), 0, traceSz);
        bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
      continue;
    }

    // Compare out to ref.
    // Default: epsilon tolerance to handle accumulation order differences
    // between host ref and AIE kernel.  bf16 has ~2 decimal digits of
    // precision, so wider tolerances are needed than for f32.
    // --strict-verify reverts to exact bitwise comparison for debugging.
    if(verify) {
      if (verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      constexpr float REL_TOL = 1e-2f;
      constexpr float ABS_TOL = 5e-1f;
      for (int i = 0; i < tp.M; ++i) {
        for (int j = 0; j < tp.N; ++j) {
          float ref = matCRef[(i * tp.N) + j];
          float out = matC[(i * tp.N) + j];

          bool match = strictVerify
              ? (out == ref)
              : (std::fabs(out - ref) <= ABS_TOL + REL_TOL * std::fabs(ref));

          if (!match) {
            if (verbosity >= 1)
              std::cout << "Error in output " << out << " != " << ref << std::endl;
            errors++;
          } else {
            if (verbosity >= 1)
              std::cout << "Correct output " << out << " == " << ref << std::endl;
          }
        }
      }
    }

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // Save trace data after all iterations (last iteration's trace is captured)
  if (traceSz > 0) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // Write trace buffer as hex words, matching parse_trace.py expected format
    {
      uint32_t *traceOut = reinterpret_cast<uint32_t*>(bo_trace.map<char*>());
      size_t totalWords = static_cast<size_t>(traceSz) / sizeof(uint32_t);

      // Skip leading zeros (DMA write pointer may not start at offset 0).
      size_t start = 0;
      while (start < totalWords && traceOut[start] == 0)
        ++start;

      // Find last non-zero word to avoid trailing zeros.
      size_t end = totalWords;
      while (end > start && traceOut[end - 1] == 0)
        --end;

      FILE *fp = fopen(traceFile.c_str(), "w");
      if (fp) {
        for (size_t i = start; i < end; i++) {
          fprintf(fp, "%08x\n", traceOut[i]);
        }
        fclose(fp);
      }
    }
    std::cout << "Trace data written to " << traceFile
              << " (" << traceSz << " bytes)" << std::endl;
  }

  // ------------------------------------------------------
  // Print verification and timing results
  // ------------------------------------------------------
  std::cout << std::endl
            << "Number of iterations: " << n_iterations
            << " (warmup iterations: " << n_warmup_iterations << ")"
            << std::endl;

  std::ios oldState(nullptr);
  oldState.copyfmt(std::cout);

  std::cout << std::endl
            << "Avg NPU time: " << std::fixed << std::setprecision(2) << npu_time_total / n_iterations << "us."
            << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << std::fixed << std::setprecision(2) << npu_time_min << "us." << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << std::fixed << std::setprecision(2) << npu_time_max << "us." << std::endl;

  std::cout.copyfmt(oldState);

  double avgUs = npu_time_total / n_iterations;

  // Write machine-readable JSON if --json-output was specified.
  // run_tc_all.sh can parse this instead of fragile grep-based log scraping.
  if (!jsonOutputPath.empty()) {
    writeJsonResult(jsonOutputPath, errors ? "FAIL" : "PASS",
                    errors, n_iterations, n_warmup_iterations,
                    avgUs, npu_time_min, npu_time_max);
  }

  // Print Pass/Fail result of our test
  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
