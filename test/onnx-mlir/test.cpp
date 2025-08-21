//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

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

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("onnx_matmul");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  bool verify = vm["verify"].as<bool>();

  // Declaring design constants
  int M = 512;
  int K = 32;
  int N = 32;
  int A_SIZE = M * K;
  int B_SIZE = K * N;
  int C_SIZE = M * N;

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
  auto bo_inA = xrt::bo(device, A_SIZE * sizeof(DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB = xrt::bo(device, B_SIZE * sizeof(DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_outC = xrt::bo(device, C_SIZE * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer bo_inA
  DATATYPE *bufInA = bo_inA.map<DATATYPE *>();
  for (int i = 0; i < A_SIZE; i++)
    bufInA[i] = i % 100;

  // Initialize buffer bo_inB
  DATATYPE *bufInB = bo_inB.map<DATATYPE *>();
  for (int i = 0; i < B_SIZE; i++)
    bufInB[i] = i % 100;

  // Zero out buffer bo_outC
  DATATYPE *bufOut = bo_outC.map<DATATYPE *>();
  memset(bufOut, 0, C_SIZE * sizeof(DATATYPE));

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel and wait to finish
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC);
  run.wait();

  // Sync device to host memories
  bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Compare out to golden
  if(verify) {
    int errors = 0;
    if (verbosity >= 1) {
      std::cout << "Verifying results ..." << std::endl;
    }
    for (uint32_t i = 0; i < M; i++) {
      for (uint32_t j = 0; j < N; j++) {
        int32_t ref = 0;
        int32_t test = bufOut[(i * N) + j];

        for (uint32_t k = 0; k < K; k++) {
          ref += bufInA[(i * K) + k] * bufInB[(j * K) + k];
        }

        if (test != ref) {
          if (verbosity >= 1)
            std::cout << "Error in output " << test << " != " << ref << std::endl;
          errors++;
        } else {
          if (verbosity >= 1)
            std::cout << "Correct output " << test << " == " << ref << std::endl;
        }
      }
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

  return 0;
}
