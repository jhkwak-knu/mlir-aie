//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdint>

//*****************************************************************************
// Modify this section to customize buffer datatypes, initialization functions,
// and verify function. The other place to reconfigure your design is the
// Makefile.
//*****************************************************************************

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
using DATATYPE_IN1 = std::int32_t;
using DATATYPE_IN2 = std::int32_t;
using DATATYPE_OUT = std::int32_t;
#endif

// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn1[i] = i + 1;
}

// Initialize Input buffer 2
void initialize_bufIn2(DATATYPE_IN2 *bufIn2, int SIZE) {
  bufIn2[0] = 3; // scaleFactor
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

// Functional correctness verifyer
int verify_vector_scalar_mul(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                             DATATYPE_OUT *bufOut, int SIZE, int verbosity) {
  int errors = 0;

  for (int i = 0; i < SIZE; i++) {
    int32_t ref = bufIn1[i] * bufIn2[0];
    int32_t test = bufOut[i];
    if (test != ref) {
      if (verbosity >= 1)
        std::cout << "Error in output " << test << " != " << ref << std::endl;
      errors++;
    } else {
      if (verbosity >= 1)
        std::cout << "Correct output " << test << " == " << ref << std::endl;
    }
  }
  return errors;
}

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

int main(int argc, const char *argv[]) {

  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE_IN1);
  constexpr int IN2_VOLUME = IN2_SIZE / sizeof(DATATYPE_IN2);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE_OUT);

  args myargs = parse_args(argc, argv);

  // Load instruction sequence
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(myargs.instr);
  if (myargs.verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, myargs.verbosity,
                                   myargs.xclbin, myargs.kernel);

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in1 = xrt::bo(device, IN1_VOLUME * sizeof(DATATYPE_IN1), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_in2 = xrt::bo(device, IN2_VOLUME * sizeof(DATATYPE_IN2), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_VOLUME * sizeof(DATATYPE_OUT), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

  // Placeholder dummy buffer objects because 0 causes seg faults
  auto bo_tmp1 = xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Workaround so we declare a really small trace buffer when one is not used
  // Second workaround for driver issue. Allocate large trace buffer *4
  int tmp_trace_size = (myargs.trace_size > 0) ? myargs.trace_size * 4 : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (myargs.verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer objects
  DATATYPE_IN1 *bufIn1 = bo_in1.map<DATATYPE_IN1 *>();
  DATATYPE_IN2 *bufIn2 = bo_in2.map<DATATYPE_IN2 *>();
  DATATYPE_OUT *bufOut = bo_out.map<DATATYPE_OUT *>();
  char *bufTrace = bo_trace.map<char *>();

  initialize_bufIn1(bufIn1, IN1_VOLUME);
  initialize_bufIn2(bufIn2, IN2_VOLUME);
  initialize_bufOut(bufOut, OUT_VOLUME); // <<< what size do I pass it?

  char *bufTmp1 = bo_tmp1.map<char *>();
  memset(bufTmp1, 0, 4);

  if (myargs.trace_size > 0)
    memset(bufTrace, 0, myargs.trace_size);

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_tmp1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (myargs.trace_size > 0)
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Initialize run configs
  // ------------------------------------------------------
  unsigned num_iter = myargs.n_iterations + myargs.n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";

    // Run kernel
    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_in2, bo_out,
                      bo_tmp1, bo_trace);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (myargs.trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < myargs.n_warmup_iterations)
      /* Warmup iterations do not count towards average runtime. */
      continue;

    // Copy output results and verify they are correct
    if (myargs.do_verify) {
      if (myargs.verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::high_resolution_clock::now();

      errors +=
          verify_vector_scalar_mul(bufIn1, bufIn2, bufOut, IN1_VOLUME, myargs.verbosity);

      auto vstop = std::chrono::high_resolution_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::microseconds>(vstop - vstart)
              .count();
      if (myargs.verbosity >= 1)
        std::cout << "Verify time: " << vtime << " us." << std::endl;
    } else {
      if (myargs.verbosity >= 1)
        std::cout << "WARNING: results not verified." << std::endl;
    }

    // Write trace values if trace_size > 0 and first iteration
    if (myargs.trace_size > 0 && iter == myargs.n_warmup_iterations) {
      test_utils::write_out_trace(((char *)bufTrace), myargs.trace_size,
                                  myargs.trace_file);
    }

    // Accumulate run times
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // ------------------------------------------------------
  // Print verification and timing results
  // ------------------------------------------------------

  // TODO - Mac count to guide gflops
  float macs = 0;

  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / myargs.n_iterations << " us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / myargs.n_iterations)
              << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << " us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
              << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << " us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max)
              << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }

  return 0;
}
