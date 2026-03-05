# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023-2025 Advanced Micro Devices, Inc.

# parameters
# -DXRT_INC_DIR: Full path to XRT headers (default: /opt/xilinx/xrt/include)
# -DXRT_LIB_DIR: Path to XRT libs (default: /opt/xilinx/xrt/lib)
# -DTARGET_NAME: Target name to be built

cmake_minimum_required(VERSION 3.1)

project(onnx_mlir_host CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Toolchain (override if needed)
set(CMAKE_C_COMPILER gcc-13)
set(CMAKE_CXX_COMPILER g++-13)

# XRT paths (can be overridden from CLI)
set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT headers")
set(XRT_LIB_DIR /opt/xilinx/xrt/lib     CACHE STRING "Path to XRT libraries")

# Target name (Makefile passes -DTARGET_NAME=onnx_matmul)
set(TARGET_NAME onnx_matmul CACHE STRING "Target to be built")
set(ProjectName ${TARGET_NAME})
set(currentTarget ${TARGET_NAME})

# ---- Source layout (new tree) ----
# CMakeLists.txt is at repo root: onnx-mlir/CMakeLists.txt
# Host source lives in src/host.cpp
# Runtime helper remains in ../../runtime_lib relative to this file (unchanged)
set(RUNTIME_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime_lib)

add_executable(${currentTarget}
  ${RUNTIME_LIB_DIR}/test_lib/test_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/host.cpp
)

target_compile_definitions(${currentTarget} PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories(${currentTarget} PUBLIC
  ${XRT_INC_DIR}
  ${RUNTIME_LIB_DIR}/test_lib
  ${RUNTIME_LIB_DIR}
)

target_link_directories(${currentTarget} PUBLIC
  ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
  xrt_coreutil
)

# (Optional) Helpful status
message(STATUS "TARGET_NAME=${TARGET_NAME}")
message(STATUS "XRT_INC_DIR=${XRT_INC_DIR}")
message(STATUS "XRT_LIB_DIR=${XRT_LIB_DIR}")
message(STATUS "RUNTIME_LIB_DIR=${RUNTIME_LIB_DIR}")
