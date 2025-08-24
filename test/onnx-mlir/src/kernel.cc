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

extern "C" {

void extern_kernel(float *restrict A, float *restrict B, float *restrict C,
                   uint32_t N_ROW, uint32_t N_COL, uint32_t N_DEP) {

  for (int row = 0; row < N_ROW; row++) {
    for (int col = 0; col < N_COL; col++) {
      for (int dep = 0; dep < N_DEP; dep++) {
        C[row * N_COL + col] += A[row * N_DEP + dep] * B[col * N_DEP + dep];
      }
    }
  }
}

} // extern "C"
