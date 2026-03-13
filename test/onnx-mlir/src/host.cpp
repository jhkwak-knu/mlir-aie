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

#include <stdfloat>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE = std::bfloat16_t;
#endif

// Packet header is 4 bytes; number of DATATYPE elements it occupies.
static constexpr size_t PKT_HDR_ELEMS = 4 / sizeof(DATATYPE);

// AIE2 to_vector<bfloat16>() uses truncation (round-toward-zero), not
// round-to-nearest-even.  This helper replicates that behaviour so the
// host reference matches the NPU's intermediate bf16 precision.
static inline float bf16_trunc(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  bits &= 0xFFFF0000u;          // zero the lower 16 bits (truncate)
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

#include "nlohmann/json.hpp"
using json = nlohmann::json;

// Axis indices used in tpOrder to select the innermost temporal loop axis.
// tpOrder[0] determines which axis is reused across iterations:
//   AXIS_M (0) -> RHS reuse, AXIS_N (1) -> LHS reuse, AXIS_K (2) -> local accumulation.
enum AxisId : uint32_t { AXIS_M = 0, AXIS_N = 1, AXIS_K = 2 };

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
  tp.TM  = L0.at("TM").get<uint32_t>();
  tp.TK  = L0.at("TK").get<uint32_t>();
  tp.TN  = L0.at("TN").get<uint32_t>();

  if (L0.contains("tpOrder")) {
    for (auto& v : L0["tpOrder"]) tp.tpOrder.push_back(v.get<uint32_t>());
  }
  return tp;
}

static void printMatrix(const std::string name, const std::vector<DATATYPE> &mat, const int rows, const int cols) {
  std::cout << "Matrix " << name << "[" << rows << "][" << cols << "]:\n";
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << static_cast<float>(mat[(i * cols) + j]) << " ";
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
    const T* srcRow = src + (r * TN);
    T* dstRow = mat.data() + (r0 + r) * N + c0;

    for (size_t i = 0; i < TN; ++i) {
      dstRow[i] = srcRow[i];
    }
  }
}

// ---------------------------------------------------------------------------
// mmul-tiled layout conversion for bf16 mmul<4,8,8>.
//
// Row-major tile data is reordered into subtile-contiguous layout so the AIE
// kernel can use sequential vector loads (aie::load_v) instead of scattered
// scalar accesses.
// ---------------------------------------------------------------------------

// bf16 mmul<4,8,8> subtile dimensions (optimal for aie2p).
constexpr int MMUL_R = 4;  // A/C row subtile height
constexpr int MMUL_S = 8;  // A/B K subtile width
constexpr int MMUL_T = 8;  // B/C column subtile width

// Convert row-major data [rows x cols] to mmul-tiled layout where each
// (sub_rows x sub_cols) subtile is stored contiguously.
// Output order: for each M-subtile, for each N-subtile, for each row in
// the subtile, sub_cols contiguous elements.
template <typename T, int sub_rows, int sub_cols>
std::vector<T> tile_to_mmul_layout(const T* rowmaj,
                                   size_t rows, size_t cols) {
  std::vector<T> tiled(rows * cols);
  size_t idx = 0;
  for (size_t mt = 0; mt < rows / sub_rows; ++mt) {
    for (size_t nt = 0; nt < cols / sub_cols; ++nt) {
      for (size_t ii = 0; ii < static_cast<size_t>(sub_rows); ++ii) {
        const T* src = rowmaj + (mt * sub_rows + ii) * cols + nt * sub_cols;
        std::copy(src, src + sub_cols, tiled.data() + idx);
        idx += sub_cols;
      }
    }
  }
  return tiled;
}

// Convert mmul-tiled layout back to row-major [rows x cols].
// Inverse of tile_to_mmul_layout.
template <typename T, int sub_rows, int sub_cols>
std::vector<T> untile_from_mmul_layout(const T* tiled,
                                       size_t rows, size_t cols) {
  std::vector<T> rowmaj(rows * cols);
  size_t idx = 0;
  for (size_t mt = 0; mt < rows / sub_rows; ++mt) {
    for (size_t nt = 0; nt < cols / sub_cols; ++nt) {
      for (size_t ii = 0; ii < static_cast<size_t>(sub_rows); ++ii) {
        T* dst = rowmaj.data() + (mt * sub_rows + ii) * cols + nt * sub_cols;
        std::copy(tiled + idx, tiled + idx + sub_cols, dst);
        idx += sub_cols;
      }
    }
  }
  return rowmaj;
}

// Holds test matrices: input A/B, output C, and CPU reference CRef.
struct MatrixSet {
  std::vector<DATATYPE> A, B, C, CRef;
};

// Initialize test matrices and compute CPU reference result.
// A is row-major [M x K], B is stored transposed [N x K] (matching AIE kernel
// convention where B is pre-transposed), C is zeroed [M x N].
// CRef = A * B^T computed on host for verification.
static MatrixSet initMatrices(const tilingParam &tp, int verbosity) {
  int matASize = tp.M * tp.K;
  int matBSize = tp.N * tp.K;
  int matCSize = tp.M * tp.N;

  MatrixSet ms;
  ms.A.resize(matASize);
  for (int i = 0; i < matASize; ++i) ms.A[i] = i / tp.M;

  ms.B.resize(matBSize);
  for (int i = 0; i < matBSize; ++i) ms.B[i] = i / tp.N;

  ms.C.assign(matCSize, 0);

  // Host reference: C[i][j] = sum_k A[i][k] * B[j][k]  (B transposed layout)
  //
  // The NPU kernel accumulates in float32 within each TK-sized chunk, then
  // stores the result as bf16 via to_vector<bfloat16>() which uses truncation
  // (round-toward-zero).  When TPk > 1, the next iteration loads that bf16
  // value back into the float32 accumulator and continues.  We replicate this
  // bf16 truncation at each temporal boundary so the reference matches the
  // hardware's intermediate precision.
  ms.CRef.resize(matCSize);
  for (int i = 0; i < static_cast<int>(tp.M); ++i) {
    for (int j = 0; j < static_cast<int>(tp.N); ++j) {
      float acc = 0.0f;
      for (int tk = 0; tk < static_cast<int>(tp.TPk); ++tk) {
        int kStart = tk * static_cast<int>(tp.TK);
        int kEnd   = kStart + static_cast<int>(tp.TK);
        for (int k = kStart; k < kEnd; ++k) {
          acc += static_cast<float>(ms.A[(i * tp.K) + k]) *
                 static_cast<float>(ms.B[(j * tp.K) + k]);
        }
        // Match AIE2 to_vector<bfloat16>() truncation at temporal boundary
        acc = bf16_trunc(acc);
      }
      ms.CRef[(i * tp.N) + j] = static_cast<DATATYPE>(acc);
    }
  }

  if (verbosity >= 2) {
    printMatrix("A", ms.A, tp.M, tp.K);
    printMatrix("B", ms.B, tp.N, tp.K);
    printMatrix("C", ms.C, tp.M, tp.N);
    printMatrix("CRef", ms.CRef, tp.M, tp.N);
  }
  return ms;
}

// Pre-computed tile iteration order and per-iteration offsets for A/B/C chunks,
// determined by tpOrder (which axis is innermost/reused).
struct TileOrderConfig {
  int reuseTPAxis, innerTPAxis, outerTPAxis;
  int reuseTP, innerTP, outerTP;

  std::pair<int,int> matAOuterOffset, matAInnerOffset;
  std::array<int,3> matASizes;
  std::array<std::pair<int,int>,3> matASteps;

  std::pair<int,int> matBOuterOffset, matBInnerOffset;
  std::array<int,3> matBSizes;
  std::array<std::pair<int,int>,3> matBSteps;

  std::pair<int,int> matCOuterOffset, matCInnerOffset;
  std::array<int,3> matCSizes;
  std::array<std::pair<int,int>,3> matCSteps;

  std::vector<std::pair<int,int>> chunkATileOrderBase;
  std::vector<std::pair<int,int>> chunkBTileOrderBase;
  std::vector<std::pair<int,int>> chunkCTileOrderBase;
};

static TileOrderConfig buildTileOrders(const tilingParam &tp) {
  TileOrderConfig cfg;

  std::array<int,3> tpValues{static_cast<int>(tp.TPm), static_cast<int>(tp.TPn), static_cast<int>(tp.TPk)};
  int baseStepforSPm = static_cast<int>((tp.M / tp.TM) / tp.TPm);
  int baseStepforSPn = static_cast<int>((tp.N / tp.TN) / tp.TPn);
  int defaultSize = 1;
  std::pair<int,int> defaultStep{0,0};

  std::array<int,3> matASizeBase{static_cast<int>(tp.SPm), static_cast<int>(tp.TPm), static_cast<int>(tp.TPk)};
  std::array<std::pair<int,int>,3> matAStepBase{std::pair<int,int>{1,0},
                                                std::pair<int,int>{baseStepforSPm,0},
                                                std::pair<int,int>{0,1}};

  std::array<int,3> matBSizeBase{static_cast<int>(tp.SPn), static_cast<int>(tp.TPn), static_cast<int>(tp.TPk)};
  std::array<std::pair<int,int>,3> matBStepBase{std::pair<int,int>{1,0},
                                                std::pair<int,int>{baseStepforSPn,0},
                                                std::pair<int,int>{0,1}};

  std::array<int,4> matCSizeBase{static_cast<int>(tp.SPm), static_cast<int>(tp.SPn), static_cast<int>(tp.TPm), static_cast<int>(tp.TPn)};
  std::array<std::pair<int,int>,4> matCStepBase{std::pair<int,int>{1,0},
                                                std::pair<int,int>{0,1},
                                                std::pair<int,int>{baseStepforSPm,0},
                                                std::pair<int,int>{0,baseStepforSPn}};

  cfg.reuseTPAxis = tp.tpOrder[0];
  cfg.innerTPAxis = tp.tpOrder[1];
  cfg.outerTPAxis = tp.tpOrder[2];
  cfg.reuseTP = tpValues[cfg.reuseTPAxis];
  cfg.innerTP = tpValues[cfg.innerTPAxis];
  cfg.outerTP = tpValues[cfg.outerTPAxis];

  // Inner/outer offsets are determined by which axis each host loop variable
  // advances.  A is M×K, B is N×K, C is M×N — an axis that does not index
  // the matrix yields a zero offset.
  auto matAOffsetFor = [&](int axis) -> std::pair<int,int> {
    if (axis == AXIS_M) return matAStepBase[1]; // TPm step
    if (axis == AXIS_K) return matAStepBase[2]; // TPk step
    return defaultStep;                         // N: A is independent of N
  };
  auto matBOffsetFor = [&](int axis) -> std::pair<int,int> {
    if (axis == AXIS_N) return matBStepBase[1]; // TPn step
    if (axis == AXIS_K) return matBStepBase[2]; // TPk step
    return defaultStep;                         // M: B is independent of M
  };
  auto matCOffsetFor = [&](int axis) -> std::pair<int,int> {
    if (axis == AXIS_M) return matCStepBase[2]; // TPm step
    if (axis == AXIS_N) return matCStepBase[3]; // TPn step
    return defaultStep;                         // K: C is independent of K
  };

  cfg.matAInnerOffset = matAOffsetFor(cfg.innerTPAxis);
  cfg.matAOuterOffset = matAOffsetFor(cfg.outerTPAxis);
  cfg.matBInnerOffset = matBOffsetFor(cfg.innerTPAxis);
  cfg.matBOuterOffset = matBOffsetFor(cfg.outerTPAxis);
  cfg.matCInnerOffset = matCOffsetFor(cfg.innerTPAxis);
  cfg.matCOuterOffset = matCOffsetFor(cfg.outerTPAxis);

  // Chunk sizes and tile-order steps depend on the reuse axis: the reuse
  // dimension is folded into the chunk, while the other two are host loops.
  if (cfg.reuseTPAxis == AXIS_M) {
    cfg.matASizes = {matASizeBase[0], matASizeBase[1], defaultSize};
    cfg.matASteps = {matAStepBase[0], matAStepBase[1], defaultStep};

    cfg.matBSizes = {matBSizeBase[0], defaultSize, defaultSize};
    cfg.matBSteps = {matBStepBase[0], defaultStep, defaultStep};

    cfg.matCSizes = {matCSizeBase[0], matCSizeBase[1], matCSizeBase[2]};
    cfg.matCSteps = {matCStepBase[0], matCStepBase[1], matCStepBase[2]};
  } else if (cfg.reuseTPAxis == AXIS_N) {
    cfg.matASizes = {matASizeBase[0], defaultSize, defaultSize};
    cfg.matASteps = {matAStepBase[0], defaultStep, defaultStep};

    cfg.matBSizes = {matBSizeBase[0], matBSizeBase[1], defaultSize};
    cfg.matBSteps = {matBStepBase[0], matBStepBase[1], defaultStep};

    cfg.matCSizes = {matCSizeBase[0], matCSizeBase[1], matCSizeBase[3]};
    cfg.matCSteps = {matCStepBase[0], matCStepBase[1], matCStepBase[3]};
  } else { // AXIS_K
    cfg.matASizes = {matASizeBase[0], matASizeBase[2], defaultSize};
    cfg.matASteps = {matAStepBase[0], matAStepBase[2], defaultStep};

    cfg.matBSizes = {matBSizeBase[0], matBSizeBase[2], defaultSize};
    cfg.matBSteps = {matBStepBase[0], matBStepBase[2], defaultStep};

    cfg.matCSizes = {matCSizeBase[0], matCSizeBase[1], defaultSize};
    cfg.matCSteps = {matCStepBase[0], matCStepBase[1], defaultStep};
  }

  cfg.chunkATileOrderBase = makeTileOrder(cfg.matASizes, cfg.matASteps);
  cfg.chunkBTileOrderBase = makeTileOrder(cfg.matBSizes, cfg.matBSteps);
  cfg.chunkCTileOrderBase = makeTileOrder(cfg.matCSizes, cfg.matCSteps);

  return cfg;
}

// Write structured JSON result to a file for machine-readable log parsing.
static void writeJsonResult(const std::string &path, const std::string &status,
                            int errors, int iterations, int warmup,
                            double avgUs, double minUs, double maxUs) {
  json j;
  j["status"] = status;
  j["errors"] = errors;
  j["iterations"] = iterations;
  j["warmup"] = warmup;
  j["avg_us"] = avgUs;
  j["min_us"] = minUs;
  j["max_us"] = maxUs;

  std::ofstream ofs(path);
  if (!ofs) {
    std::cerr << "Warning: cannot write JSON result to " << path << "\n";
    return;
  }
  ofs << j.dump(2) << "\n";
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
