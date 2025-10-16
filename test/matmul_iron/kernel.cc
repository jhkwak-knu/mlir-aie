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
                   int32_t N_ROW, int32_t N_COL, int32_t N_DEP, int8_t acc) {
  event0(); // event to mark start of function
  for (int32_t row = 0; row < N_ROW; row++) {
    for (int32_t col = 0; col < N_COL; col++) {
      if (acc == 0) {
        C[row * N_COL + col] = 0;
      }
      
      for (int32_t dep = 0; dep < N_DEP; dep++) {
        C[row * N_COL + col] += A[row * N_DEP + dep] * B[col * N_DEP + dep];
      }
    }
  }
  event1(); // event to mark end of function
}

} // extern "C"
