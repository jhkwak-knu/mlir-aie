//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Optimized bf16 matmul kernel using 2x2 mmul expansion pattern.
// Expects data in mmul-tiled layout (host converts row-major before DMA).
//
// Memory layout (bf16 mmul<4,8,8> for aie2p):
//   A: [M_TILE/4][K_TILE/8][4*8]  — each 4x8 subtile contiguous
//   B: [N_TILE/8][K_TILE/8][8*8]  — col-major; transposed 8x8 at load
//   C: [M_TILE/4][N_TILE/8][4*8]  — each 4x8 subtile contiguous
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

// Tile dimensions injected by Makefile from tc.json.
#ifndef M_TILE
#error "M_TILE must be defined via -DM_TILE=<value>"
#endif
#ifndef K_TILE
#error "K_TILE must be defined via -DK_TILE=<value>"
#endif
#ifndef N_TILE
#error "N_TILE must be defined via -DN_TILE=<value>"
#endif

// bf16 mmul shape optimized for AIE2P hardware vector unit.
constexpr int r = 4, s = 8, t = 8;
using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16>;

// Number of mmul tiles in each dimension.
constexpr unsigned rowA = M_TILE / r;
constexpr unsigned colA = K_TILE / s;
constexpr unsigned colB = N_TILE / t;

// 2x2 expansion requires at least 2 mmul tiles in M and N directions.
static_assert(M_TILE % (2 * r) == 0, "M_TILE must be a multiple of 8");
static_assert(N_TILE % (2 * t) == 0, "N_TILE must be a multiple of 16");
static_assert(K_TILE % s == 0, "K_TILE must be a multiple of 8");

extern "C" {

void extern_kernel(bfloat16 *restrict A, bfloat16 *restrict B,
                   bfloat16 *restrict C, uint32_t N_ROW, uint32_t N_COL,
                   uint32_t N_DEP, bool acc) {

  // Zero C when not accumulating partial sums.
  if (!acc) {
    auto zvec = aie::zeros<bfloat16, MMUL::size_C>();
    for (unsigned i = 0; i < rowA * colB; ++i)
      aie::store_v(&C[i * MMUL::size_C], zvec);
  }

  // 2x2 expansion: 2 M-tiles x 2 N-tiles per iteration (4 accumulators).
  // 2x2 is the standard expansion for aie2p (matches upstream aie_kernels/aie2p/mm.cc).
  // aie2p mmul<4,8,8>: size_A=32, size_B=64, size_C=32.
  //
  // chess_prepare_for_pipelining: enable loop pipelining for the M-loop.
  // chess_loop_range: hint minimum trip count to the scheduler.
  for (unsigned m = 0; m < rowA; m += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
    for (unsigned n = 0; n < colB; n += 2) {
      unsigned c00_off = (m * colB + n) * MMUL::size_C;
      unsigned c01_off = (m * colB + n + 1) * MMUL::size_C;
      unsigned c10_off = ((m + 1) * colB + n) * MMUL::size_C;
      unsigned c11_off = ((m + 1) * colB + n + 1) * MMUL::size_C;

      MMUL C00(aie::load_v<MMUL::size_C>(&C[c00_off]));
      MMUL C01(aie::load_v<MMUL::size_C>(&C[c01_off]));
      MMUL C10(aie::load_v<MMUL::size_C>(&C[c10_off]));
      MMUL C11(aie::load_v<MMUL::size_C>(&C[c11_off]));

      for (unsigned k = 0; k < colA; ++k)
        chess_prepare_for_pipelining chess_loop_range(2, ) {
        auto a0 = aie::load_v<MMUL::size_A>(&A[(m * colA + k) * MMUL::size_A]);
        auto a1 = aie::load_v<MMUL::size_A>(&A[((m + 1) * colA + k) * MMUL::size_A]);
        auto b0 = aie::transpose(
            aie::load_v<MMUL::size_B>(&B[(n * colA + k) * MMUL::size_B]), t, s);
        auto b1 = aie::transpose(
            aie::load_v<MMUL::size_B>(&B[((n + 1) * colA + k) * MMUL::size_B]), t, s);

        C00.mac(a0, b0); C01.mac(a0, b1);
        C10.mac(a1, b0); C11.mac(a1, b1);
      }

      aie::store_v(&C[c00_off], C00.template to_vector<bfloat16>());
      aie::store_v(&C[c01_off], C01.template to_vector<bfloat16>());
      aie::store_v(&C[c10_off], C10.template to_vector<bfloat16>());
      aie::store_v(&C[c11_off], C11.template to_vector<bfloat16>());
    }
  }
}

} // extern "C"
