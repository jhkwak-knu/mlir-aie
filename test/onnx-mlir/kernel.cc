//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define N_ROW 32
#define N_COL 32
#define N_DEP 32

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

extern "C" {

void extern_kernel(float *restrict A, float *restrict B, float *restrict C) {

  for (int row = 0; row < N_ROW; row++) {
    for (int col = 0; col < N_COL; col++) {
      float running_sum = 0.0f;
      for (int i = 0; i < N_DEP; i++) {
        running_sum += A[row * N_DEP + i] * B[col * N_DEP + i];
      }
      C[row * N_COL + col] = running_sum;
    }
  }
}

} // extern "C"
