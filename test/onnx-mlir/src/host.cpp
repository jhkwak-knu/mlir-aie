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
#include <iomanip>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE = float; // Configure this to match your buffer data type
#endif

#include "nlohmann/json.hpp"
using json = nlohmann::json;

struct tilingParam {
  uint32_t M, K, N;
  uint32_t SPm, SPn, TPm, TPk, TPn;
  uint32_t TM, TK, TN;
  std::vector<uint32_t> tpOrder; // 0:M, 1:N, 2:K
};

static tilingParam loadTilingParam(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("cannot open: " + path);
  json j; ifs >> j;

  tilingParam tp{};
  tp.M  = j.at("M").get<uint32_t>();
  tp.K  = j.at("K").get<uint32_t>();
  tp.N  = j.at("N").get<uint32_t>();

  const auto& L0 = j.at("levels").at(0);
  tp.SPm = L0.at("SPm").get<uint32_t>();
  tp.SPn = L0.at("SPn").get<uint32_t>();
  tp.TPm = L0.at("TPm").get<uint32_t>();
  tp.TPk = L0.at("TPk").get<uint32_t>();
  tp.TPn = L0.at("TPn").get<uint32_t>();
  tp.TM = L0.at("TM").get<uint32_t>();
  tp.TK = L0.at("TK").get<uint32_t>();
  tp.TN = L0.at("TN").get<uint32_t>();

  if (L0.contains("tpOrder")) {
    for (auto& v : L0["tpOrder"]) tp.tpOrder.push_back(v.get<uint32_t>());
  }
  return tp;
}

static void printMatrix(const std::string name, const std::vector<DATATYPE> &mat, const int rows, const int cols) {
  std::cout << "Matrix " << name << "[" << rows << "][" << cols << "]:\n";
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << mat[(i * cols) + j] << " ";
    }
    std::cout << "\n";
  }
}

std::vector<std::pair<int,int>>
makeTileOrder(const std::array<int,3>& sizes,
              const std::array<std::pair<int,int>,3>& steps)
{
  if (sizes[0] < 0 || sizes[1] < 0 || sizes[2] < 0)
    throw std::invalid_argument("sizes must be non-negative");

  std::vector<std::pair<int,int>> tileOrder;
  tileOrder.reserve(static_cast<size_t>(sizes[0]) *
                    static_cast<size_t>(sizes[1]) *
                    static_cast<size_t>(sizes[2]));

  for (int i = 0; i < sizes[2]; ++i) {
    for (int j = 0; j < sizes[1]; ++j) {
      for (int k = 0; k < sizes[0]; ++k) {
        int tr = steps[0].first  * k + steps[1].first  * j + steps[2].first  * i;
        int tc = steps[0].second * k + steps[1].second * j + steps[2].second * i;
        tileOrder.emplace_back(tr, tc);
      }
    }
  }
  return tileOrder;
}

template <typename T>
std::vector<T> extract_tile_1d_strict(const std::vector<T>& mat,
                                      size_t M, size_t N,
                                      size_t TM, size_t TN,
                                      size_t tileRow, size_t tileCol)
{
  const size_t r0 = tileRow * TM;
  const size_t c0 = tileCol * TN;
  if (r0 + TM > M || c0 + TN > N) {
    throw std::out_of_range("tile out of bounds");
  }

  std::vector<T> out;
  out.reserve(TM * TN);
  for (size_t r = 0; r < TM; ++r) {
    const T* rowptr = mat.data() + (r0 + r) * N + c0;
    out.insert(out.end(), rowptr, rowptr + TN);
  }
  return out;
}

template <typename T>
void write_tile_1d_strict(std::vector<T>& mat,
                          size_t M, size_t N,
                          size_t TM, size_t TN,
                          size_t tileRow, size_t tileCol,
                          const std::vector<T>& tile
                          )
{
  const size_t r0 = tileRow * TM;
  const size_t c0 = tileCol * TN;

  if (tile.size() != TM * TN)
    throw std::invalid_argument("tile size mismatch");

  if (r0 + TM > M || c0 + TN > N)
    throw std::out_of_range("tile out of bounds");

  const T* src = tile.data();
  for (size_t r = 0; r < TM; ++r) {
    T* dstRow = mat.data() + (r0 + r) * N + c0;
    std::copy(src + r * TN, src + (r + 1) * TN, dstRow);
  }
}

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("onnx_matmul");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  bool verify = vm["verify"].as<bool>();

  // Declaring design constants
  auto tp = loadTilingParam("/home/ace/ryzen_ai/mlir-aie-dev/mlir-aie/test/onnx-mlir/out/tc.json");
  int matASize = tp.M * tp.K;
  int matBSize = tp.N * tp.K;
  int matCSize = tp.M * tp.N;

  int chunkTPm = (tp.tpOrder[0] == 0) ? tp.TPm : 1;
  int chunkTPn = (tp.tpOrder[0] == 1) ? tp.TPn : 1;
  int chunkTPk = (tp.tpOrder[0] == 2) ? tp.TPk : 1;

  int chunkASize = ((tp.TM * tp.TK) * tp.SPm) * chunkTPm * chunkTPk;
  int chunkBSize = ((tp.TN * tp.TK) * tp.SPn) * chunkTPn * chunkTPk;
  int chunkCSize = ((tp.TM * tp.TN) * tp.SPm * tp.SPn) * chunkTPm * chunkTPn;
  int chunkOutCSize = ((tp.TM * tp.TN + 1) * tp.SPm * tp.SPn) * chunkTPm * chunkTPn;

  bool useInC= (tp.TPk > 1) && (tp.tpOrder[0] != 2);

  // Initialize matrix A
  std::vector<DATATYPE> matA(matASize);
  for (int i = 0; i < matASize; ++i) {
    matA[i] = i / tp.M;
  }
  if (verbosity >= 2) printMatrix("A", matA, tp.M, tp.K);

  // Initialize matrix B (transposed)
  std::vector<DATATYPE> matB(matBSize);
  for (int i = 0; i < matBSize; ++i) {
    matB[i] = i / tp.N;
  }
  if (verbosity >= 2) printMatrix("B", matB, tp.N, tp.K);

  // Initialize matrix C
  std::vector<DATATYPE> matC(matCSize, 0);
  if (verbosity >= 2) printMatrix("C", matC, tp.M, tp.N);
  
  // Initialize matrix C (ref)
  std::vector<DATATYPE> matCRef(matCSize);
  for (int i = 0; i < tp.M; ++i) {
    for (int j = 0; j < tp.N; ++j) {
      int idx = (i * tp.N) + j;
      matCRef[idx] = 0;

      for (int k = 0; k < tp.K; ++k) {
        matCRef[idx] += matA[(i * tp.K) + k] * matB[(j * tp.K) + k];
      }
    }
  }
  if (verbosity >= 2) printMatrix("CRef", matCRef, tp.M, tp.N);

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
  int n_iterations = 1;
  int n_warmup_iterations = 0;
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 99999999;
  float npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  std::array<int,3> tpValues{static_cast<int>(tp.TPm), static_cast<int>(tp.TPn), static_cast<int>(tp.TPk)};
  int baseStepforSPm = static_cast<int>((tp.M / tp.TM) / tp.SPm);
  int baseStepforSPn = static_cast<int>((tp.N / tp.TN) / tp.SPn);
  int defaultSize = 1;
  std::pair<int,int> defaultStep{0,0};

  std::array<int,3> matASizeBase{static_cast<int>(tp.SPm), static_cast<int>(tp.TPm), static_cast<int>(tp.TPk)};
  std::array<std::pair<int,int>,3> matAStepBase{std::pair<int,int>{baseStepforSPm,0},
                                                std::pair<int,int>{1,0},
                                                std::pair<int,int>{0,1}};
  
  std::array<int,3> matBSizeBase{static_cast<int>(tp.SPn), static_cast<int>(tp.TPn), static_cast<int>(tp.TPk)};
  std::array<std::pair<int,int>,3> matBStepBase{std::pair<int,int>{baseStepforSPn,0},
                                                std::pair<int,int>{1,0},
                                                std::pair<int,int>{0,1}};
  
  std::array<int,4> matCSizeBase{static_cast<int>(tp.SPm), static_cast<int>(tp.SPn), static_cast<int>(tp.TPm), static_cast<int>(tp.TPn)};
  std::array<std::pair<int,int>,4> matCStepBase{std::pair<int,int>{baseStepforSPm,0},
                                                std::pair<int,int>{0,baseStepforSPn},
                                                std::pair<int,int>{1,0},
                                                std::pair<int,int>{0,1}};
  
  int reuseTPAxis = tp.tpOrder[0];
  int innerTPAxis = tp.tpOrder[1];
  int outerTPAxis = tp.tpOrder[2];
  int reuseTP = tpValues[reuseTPAxis];
  int innerTP = tpValues[innerTPAxis];
  int outerTP = tpValues[outerTPAxis];

  std::pair<int,int> matAOuterOffset, matAInnerOffset;
  std::array<int,3> matASizes;
  std::array<std::pair<int,int>,3> matASteps;
  std::pair<int,int> matBOuterOffset, matBInnerOffset;
  std::array<int,3> matBSizes;
  std::array<std::pair<int,int>,3> matBSteps;
  std::pair<int,int> matCOuterOffset, matCInnerOffset;
  std::array<int,3> matCSizes;
  std::array<std::pair<int,int>,3> matCSteps;

  if (reuseTPAxis == 0) {
    matAOuterOffset = matAStepBase[2];
    matAInnerOffset = defaultStep;
    matASizes = std::array<int,3>{matASizeBase[1], matASizeBase[0], defaultSize};
    matASteps = std::array<std::pair<int,int>,3>{matAStepBase[1], matAStepBase[0], defaultStep};

    matBOuterOffset = matBStepBase[2];
    matBInnerOffset = matBStepBase[1];
    matBSizes = std::array<int,3>{matBSizeBase[0], defaultSize, defaultSize};
    matBSteps = std::array<std::pair<int,int>,3>{matBStepBase[0], defaultStep, defaultStep};

    matCOuterOffset = defaultStep;
    matCInnerOffset = matCStepBase[3];
    matCSizes = std::array<int,3>{matCSizeBase[2], matCSizeBase[0], matCSizeBase[1]};
    matCSteps = std::array<std::pair<int,int>,3>{matCStepBase[2], matCStepBase[0], matCStepBase[1]};
  } else if (reuseTPAxis == 1) {
    matAOuterOffset = matAStepBase[2];
    matAInnerOffset = matAStepBase[1];
    matASizes = std::array<int,3>{matASizeBase[0], defaultSize, defaultSize};
    matASteps = std::array<std::pair<int,int>,3>{matAStepBase[0], defaultStep, defaultStep};

    matBOuterOffset = matBStepBase[2];
    matBInnerOffset = defaultStep;
    matBSizes = std::array<int,3>{matBSizeBase[1], matBSizeBase[0], defaultSize};
    matBSteps = std::array<std::pair<int,int>,3>{matBStepBase[1], matBStepBase[0], defaultStep};

    matCOuterOffset = defaultStep;
    matCInnerOffset = matCStepBase[2];
    matCSizes = std::array<int,3>{matCSizeBase[0], matCSizeBase[3], matCSizeBase[1]};
    matCSteps = std::array<std::pair<int,int>,3>{matCStepBase[0], matCStepBase[3], matCStepBase[1]};
  } else { // reuseTPAxis == 2
    matAOuterOffset = defaultStep;
    matAInnerOffset = matAStepBase[1];
    matASizes = std::array<int,3>{matASizeBase[2], matASizeBase[0], defaultSize};
    matASteps = std::array<std::pair<int,int>,3>{matAStepBase[2], matAStepBase[0], defaultStep};

    matBOuterOffset = matBStepBase[1];
    matBInnerOffset = defaultStep;
    matBSizes = std::array<int,3>{matBSizeBase[2], matBSizeBase[0], defaultSize};
    matBSteps = std::array<std::pair<int,int>,3>{matBStepBase[2], matBStepBase[0], defaultStep};

    matCOuterOffset = matCStepBase[3];
    matCInnerOffset = matCStepBase[2];
    matCSizes = std::array<int,3>{matCSizeBase[0], matCSizeBase[1], defaultSize};
    matCSteps = std::array<std::pair<int,int>,3>{matCStepBase[0], matCStepBase[1], defaultStep};
  }

  auto chunkATileOrderBase = makeTileOrder(matASizes, matASteps);
  auto chunkBTileOrderBase = makeTileOrder(matBSizes, matBSteps);
  auto chunkCTileOrderBase = makeTileOrder(matCSizes, matCSteps);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    float npu_time = 0;

    for (int i = 0; i < outerTP; ++i) {
      for (int j = 0; j < innerTP; ++j) {
        // set chunk data
        if (verbosity >= 2)
          std::cout << "Set Data (" << (i * innerTP) + j << "):\n";

        // matrix A
        std::vector<std::pair<int,int>> chunkATileOrder;
        for (auto &tileA : chunkATileOrderBase) {
          int row = tileA.first + (matAInnerOffset.first * j) + (matAOuterOffset.first * i);
          int col = tileA.second + (matAInnerOffset.second * j) + (matAOuterOffset.second * i);
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

          if (chunkACount + tileVec.size() > chunkASize) {
            std::cerr << "Overflow while writing bufInA\n";
            std::exit(EXIT_FAILURE);
          }

          std::copy(tileVec.begin(), tileVec.end(), bufInA + chunkACount);
          chunkACount += tileVec.size();
        }
        if (verbosity >= 2) printMatrix("Chunk A", std::vector<DATATYPE>(bufInA, bufInA + chunkASize), chunkASize / tp.TK, tp.TK);

        // matrix B
        std::vector<std::pair<int,int>> chunkBTileOrder;
        for (auto &tileB : chunkBTileOrderBase) {
          int row = tileB.first + (matBInnerOffset.first * j) + (matBOuterOffset.first * i);
          int col = tileB.second + (matBInnerOffset.second * j) + (matBOuterOffset.second * i);
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

          if (chunkBCount + tileVec.size() > chunkBSize) {
            std::cerr << "Overflow while writing bufInB\n";
            std::exit(EXIT_FAILURE);
          }

          std::copy(tileVec.begin(), tileVec.end(), bufInB + chunkBCount);
          chunkBCount += tileVec.size();
        }
        if (verbosity >= 2) printMatrix("Chunk B", std::vector<DATATYPE>(bufInB, bufInB + chunkBSize), chunkBSize / tp.TK, tp.TK);

        // matrix C
        memset(bufOut, 0, chunkOutCSize * sizeof(DATATYPE));

        std::vector<std::pair<int,int>> chunkCTileOrder;
        for (auto &tileC : chunkCTileOrderBase) {
          int row = tileC.first + (matCInnerOffset.first * j) + (matCOuterOffset.first * i);
          int col = tileC.second + (matCInnerOffset.second * j) + (matCOuterOffset.second * i);
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

            if (chunkCCount + tileVec.size() > chunkCSize) {
              std::cerr << "Overflow while writing bufInC\n";
              std::exit(EXIT_FAILURE);
            }

            std::copy(tileVec.begin(), tileVec.end(), bufInC + chunkCCount);
            chunkCCount += tileVec.size();
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
        if (useInC) {
          run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC, bo_inC);
        } else {
          run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC);
        }
        run.wait();
        auto stop = std::chrono::high_resolution_clock::now();

        npu_time += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

        // Sync device to host memories
        bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        if (verbosity >= 2) {
          for (int i = 0; i < (tp.SPm * tp.SPn); ++i) {
            uint32_t pkt_header, pkt_id;
            std::memcpy(&pkt_header, &bufOut[(tp.TM * tp.TN + 1) * i + 0], sizeof(pkt_header));
            pkt_id = pkt_header & 0x1F;

            std::cout << "OutC[" << i << "] (packet id = " << pkt_id << "): ";
            for (int j = 0; j < (tp.TM * tp.TN); ++j) {
              std::cout << bufOut[(tp.TM * tp.TN + 1) * i + j + 1] << " ";
            }
            std::cout << "\n";
          }
        }

        // store partial sums to matC
        for (int i = 0; i < ((tp.SPm * tp.SPn) / 4); ++i) {
          int repeatCount = (reuseTPAxis != 2) ? reuseTP : 1;
          int outerOffset = (repeatCount * 4) * i;

          for (int j = 0; j < repeatCount; ++j) {
            int innerFactor = (reuseTPAxis != 1) ? 1 : tp.SPm;
            int innerOffset = innerFactor * j;

            for (int k = 0; k < 4; ++k) {
              int baseOffset = (reuseTPAxis != 1) ? (repeatCount * k) : ((k / tp.SPm) * tp.SPm * tp.TPn + (k % tp.SPm));
              int idx = baseOffset + innerOffset + outerOffset;

              int packetHeader, packetId;
              std::memcpy(&packetHeader, &bufOut[(tp.TM * tp.TN + 1) * idx], sizeof(packetHeader));
              packetId = packetHeader & 0x1F;
              if (packetId == 0) packetId = 0;
              else if (packetId == 2) packetId = 1;
              else if (packetId == 6) packetId = 2;
              else if (packetId == 14) packetId = 3;
              int matCOffset = (reuseTPAxis != 1) ? (repeatCount * packetId) : ((packetId / tp.SPm) * tp.SPm * tp.TPn + (packetId % tp.SPm));
              int matCIdx = matCOffset + innerOffset + outerOffset;

              std::pair<int,int> tilePos = chunkCTileOrder[matCIdx];

              const size_t tileElems  = static_cast<size_t>(tp.TM) * tp.TN;
              const size_t blockElems = tileElems + 1;
              const size_t start      = blockElems * static_cast<size_t>(idx) + 1;
              const size_t end        = start + tileElems;

              if (verbosity >= 2) {
                std::cout
                  << "[k=" << k << "] "
                  << "baseOffset=" << baseOffset
                  << " idx=" << idx
                  << " packetId=" << packetId
                  << " matCOffset=" << matCOffset
                  << " matCIdx=" << matCIdx
                  << " tilePos=(" << tilePos.first << "," << tilePos.second << ") "
                  << "tileElems=" << static_cast<unsigned long long>(tileElems)
                  << " blockElems=" << static_cast<unsigned long long>(blockElems)
                  << " start=" << static_cast<unsigned long long>(start)
                  << " end=" << static_cast<unsigned long long>(end)
                  << " innerOffset=" << innerOffset
                  << " outerOffset=" << outerOffset
                  << "\n";
              }
              
              std::vector<DATATYPE> tileValue(&bufOut[(tp.TM * tp.TN + 1) * idx + 1], &bufOut[(tp.TM * tp.TN + 1) * (idx + 1)]);
              write_tile_1d_strict<DATATYPE>(matC, tp.M, tp.N, tp.TM, tp.TN, tilePos.first, tilePos.second, tileValue);
            }
          }
        }
      }
    }

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    // Compare out to ref
    if(verify) {
      if (verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      for (int i = 0; i < tp.M; ++i) {
        for (int j = 0; j < tp.N; ++j) {
          float ref = matCRef[(i * tp.N) + j];
          float out = matC[(i * tp.N) + j];

          if (out != ref) {
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
            << "Avg NPU time: " << std::fixed << std::setprecision(1) << npu_time_total / n_iterations << "us."
            << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << std::fixed << std::setprecision(0) << npu_time_min << "us." << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << std::fixed << std::setprecision(0) << npu_time_max << "us." << std::endl;

  std::cout.copyfmt(oldState);

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

  return 0;
}
