//===- tile_ops.h - Tile extract/insert & mmul layout conversion -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_OPS_H
#define TILE_OPS_H

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Tile extract / insert helpers
// ---------------------------------------------------------------------------

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

#endif // TILE_OPS_H
