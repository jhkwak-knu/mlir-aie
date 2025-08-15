//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "memory_allocator.h"
#include "test_library.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <xaiengine.h>

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  u32 sleep_u = 100000;
  usleep(sleep_u);
  printf("before configure cores.\n");

  // mlir_aie_clear_tile_memory(_xaie, 0, 1);
  // mlir_aie_clear_tile_memory(_xaie, 0, 2);
  // mlir_aie_clear_tile_memory(_xaie, 0, 3);
  // mlir_aie_clear_tile_memory(_xaie, 0, 4);
  // mlir_aie_clear_tile_memory(_xaie, 0, 5);
  // mlir_aie_clear_tile_memory(_xaie, 1, 1);
  // mlir_aie_clear_tile_memory(_xaie, 1, 2);
  // mlir_aie_clear_tile_memory(_xaie, 1, 3);
  // mlir_aie_clear_tile_memory(_xaie, 1, 4);
  // mlir_aie_clear_tile_memory(_xaie, 1, 5);
  // mlir_aie_configure_cores(_xaie);

  // usleep(sleep_u);
  // printf("before configure switchboxes.\n");
  // mlir_aie_configure_switchboxes(_xaie);
  // mlir_aie_initialize_locks(_xaie);

  // usleep(sleep_u);
  // printf("before configure DMA\n");
  // mlir_aie_configure_dmas(_xaie);
  // int errors = 0;

  // printf("Finish configure\n");
  // ext_mem_model_t buf0, buf1, buf2, buf3, buf4, buf5;
  // int *mem_ptr0 = mlir_aie_mem_alloc(_xaie, buf0, 32768);
  // int *mem_ptr1 = mlir_aie_mem_alloc(_xaie, buf1, 32768);
  // int *mem_ptr2 = mlir_aie_mem_alloc(_xaie, buf2, 8192);
  // int *mem_ptr3 = mlir_aie_mem_alloc(_xaie, buf3, 8192);
  // int *mem_ptr4 = mlir_aie_mem_alloc(_xaie, buf4, 4096);
  // int *mem_ptr5 = mlir_aie_mem_alloc(_xaie, buf5, 4096);

  // // initialize the external buffers
  // for (int i = 0; i < 4096; i++) {
  //   *(mem_ptr0 + i) = 1;  // LHS_tile0
  //   *(mem_ptr1 + i) = 2;  // LHS_tile1
  //   *(mem_ptr2 + i) = 3;  // RHS_tile0
  //   *(mem_ptr3 + i) = 4;  // RHS_tile1
  //   *(mem_ptr4 + i) = 99; // Out_tile0
  //   *(mem_ptr5 + i) = 99; // Out_tile1
  // }

  // mlir_aie_sync_mem_dev(buf0); // only used in libaiev2
  // mlir_aie_sync_mem_dev(buf1); // only used in libaiev2
  // mlir_aie_sync_mem_dev(buf2); // only used in libaiev2
  // mlir_aie_sync_mem_dev(buf3); // only used in libaiev2
  // mlir_aie_sync_mem_dev(buf4); // only used in libaiev2
  // mlir_aie_sync_mem_dev(buf5); // only used in libaiev2

  // mlir_aie_external_set_addr_ext_buf_lhs_0_0(_xaie, (u64)mem_ptr0);
  // mlir_aie_external_set_addr_ext_buf_lhs_1_0(_xaie, (u64)mem_ptr1);
  // mlir_aie_external_set_addr_ext_buf_rhs_0_0(_xaie, (u64)mem_ptr2);
  // mlir_aie_external_set_addr_ext_buf_rhs_1_0(_xaie, (u64)mem_ptr3);
  // mlir_aie_external_set_addr_ext_buf_res_0_0(_xaie, (u64)mem_ptr4);
  // mlir_aie_external_set_addr_ext_buf_res_1_0(_xaie, (u64)mem_ptr5);
  // mlir_aie_configure_shimdma_00(_xaie);
  // mlir_aie_configure_shimdma_10(_xaie);

  // printf("before core start\n");

  // mlir_aie_release_lock_0_0_0(_xaie, 1, 0);
  // mlir_aie_release_lock_0_0_1(_xaie, 1, 0);
  // mlir_aie_release_lock_0_0_2(_xaie, 1, 0);
  // mlir_aie_release_lock_1_0_0(_xaie, 1, 0);
  // mlir_aie_release_lock_1_0_1(_xaie, 1, 0);
  // mlir_aie_release_lock_1_0_2(_xaie, 1, 0);

  // mlir_aie_start_cores(_xaie);

  // usleep(sleep_u);
  // // // Check if the local buffer contain the correct data
  // // for (int bd = 0; bd < 1; bd++) {
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_lhs_0_5(_xaie, bd), 1, errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_rhs_0_5(_xaie, bd), 3, errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_res_0_5(_xaie, bd), 96, // Sub_sum0
  // //                  errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_lhs_1_4(_xaie, bd), 2, errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_rhs_1_4(_xaie, bd), 4, errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_res_1_4(_xaie, bd), 352, // Out_tile0
  // //                  errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_lhs_1_5(_xaie, bd), 1, errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_rhs_1_5(_xaie, bd), 5, errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_res_1_5(_xaie, bd), 160, // Sub_sum1
  // //                  errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_lhs_0_4(_xaie, bd), 2, errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_rhs_0_4(_xaie, bd), 6, errors);
  // //   mlir_aie_check("Before release lock:",
  // //                  mlir_aie_read_buffer_buf_res_0_4(_xaie, bd), 544, // Out_tile1
  // //                  errors);
  // // }

  // mlir_aie_sync_mem_cpu(buf4); // only used in libaiev2
  // mlir_aie_sync_mem_cpu(buf5); // only used in libaiev2

  // for (int idx0 = 0; idx0 < 1; ++idx0) {
  //   if (mem_ptr4[idx0] != 352) {
  //     printf("Out_tile0[%d]=%d\n", idx0, mem_ptr4[idx0]);
  //     errors++;
  //   }
  //   if (mem_ptr5[idx0] != 544) {
  //     printf("Out_tile1[%d]=%d\n", idx0, mem_ptr5[idx0]);
  //     errors++;
  //   }
  // }

  int res = 0;
  // if (!errors) {
  //   printf("PASS!\n");
  //   res = 0;
  // } else {
  //   printf("Fail!\n");
  //   res = -1;
  // }
  // mlir_aie_deinit_libxaie(_xaie);

  printf("test done.\n");

  return res;
}
