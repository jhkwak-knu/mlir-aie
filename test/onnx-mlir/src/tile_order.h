//===- tile_order.h - Tile iteration order & matrix init --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_ORDER_H
#define TILE_ORDER_H

#include "tiling_param.h"

#include <array>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

static inline void printMatrix(const std::string name, const std::vector<DATATYPE> &mat, const int rows, const int cols) {
  std::cout << "Matrix " << name << "[" << rows << "][" << cols << "]:\n";
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << static_cast<float>(mat[(i * cols) + j]) << " ";
    }
    std::cout << "\n";
  }
}

static inline std::vector<std::pair<int,int>>
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

// Holds test matrices: input A/B, output C, and CPU reference CRef.
struct MatrixSet {
  std::vector<DATATYPE> A, B, C, CRef;
};

// Initialize test matrices and compute CPU reference result.
// A is row-major [M x K], B is stored transposed [N x K] (matching AIE kernel
// convention where B is pre-transposed), C is zeroed [M x N].
// CRef = A * B^T computed on host for verification.
static inline MatrixSet initMatrices(const tilingParam &tp, int verbosity) {
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

static inline TileOrderConfig buildTileOrders(const tilingParam &tp) {
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

#endif // TILE_ORDER_H
