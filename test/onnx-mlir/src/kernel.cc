//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

// bf16 mmul shape supported by AIE2 hardware vector unit.
// Tile dimensions must satisfy: N_ROW % r == 0, N_DEP % s == 0, N_COL % t == 0.
constexpr int r = 4, s = 8, t = 4;
using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16>;

extern "C" {

void extern_kernel(bfloat16 *restrict A, bfloat16 *restrict B,
                   bfloat16 *restrict C, uint32_t N_ROW, uint32_t N_COL,
                   uint32_t N_DEP, bool acc) {

  // A: [N_ROW x N_DEP] row-major
  // B: [N_COL x N_DEP] row-major (pre-transposed)
  // C: [N_ROW x N_COL] row-major

  for (unsigned i = 0; i < N_ROW; i += r) {
    for (unsigned j = 0; j < N_COL; j += t) {

      // Load or zero-init the 4x4 C accumulator tile.
      // C is row-major so the 4 elements per row are non-contiguous
      // when t < N_COL; gather them with scalar loads.
      aie::vector<bfloat16, MMUL::size_C> c_vec;
      if (acc) {
        for (int ii = 0; ii < r; ii++)
          for (int jj = 0; jj < t; jj++)
            c_vec[ii * t + jj] = C[(i + ii) * N_COL + (j + jj)];
      } else {
        c_vec = aie::zeros<bfloat16, MMUL::size_C>();
      }
      MMUL C_acc(c_vec);

      for (unsigned k = 0; k < N_DEP; k += s) {
        // Gather 4 rows x 8 cols from A (each row is contiguous in memory).
        aie::vector<bfloat16, MMUL::size_A> a_vec;
        for (int ii = 0; ii < r; ii++)
          a_vec.insert(ii, aie::load_v<s>(&A[(i + ii) * N_DEP + k]));

        // B is [N_COL x N_DEP] row-major (transposed).
        // Load t=4 rows of s=8 elements → 4x8 layout, then transpose
        // to the 8x4 layout that mmul expects for the B operand.
        aie::vector<bfloat16, MMUL::size_B> b_raw;
        for (int jj = 0; jj < t; jj++)
          b_raw.insert(jj, aie::load_v<s>(&B[(j + jj) * N_DEP + k]));
        auto b_vec = aie::transpose(b_raw, t, s);

        C_acc.mac(a_vec, b_vec);
      }

      // Scatter the 4x4 result tile back to row-major C.
      auto result = C_acc.template to_vector<bfloat16>();
      for (int ii = 0; ii < r; ii++)
        for (int jj = 0; jj < t; jj++)
          C[(i + ii) * N_COL + (j + jj)] = result[ii * t + jj];
    }
  }
}

} // extern "C"
