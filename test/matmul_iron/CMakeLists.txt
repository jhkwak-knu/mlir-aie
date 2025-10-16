# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

# cmake needs this line
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_C_COMPILER gcc-13)
set(CMAKE_CXX_COMPILER g++-13)
set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT cloned repo")
set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to xrt_coreutil.lib")

set(IN1_SIZE 8192 CACHE STRING "in1 buffer size")
set(IN2_SIZE 4 CACHE STRING "in2 buffer size")
set(OUT_SIZE 8192 CACHE STRING "out buffer size")

SET (ProjectName ${TARGET_NAME})
SET (currentTarget ${TARGET_NAME})

project(${ProjectName})

add_executable(${currentTarget}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime_lib/test_lib/test_utils.cpp
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC
    IN1_SIZE=${IN1_SIZE}
    IN2_SIZE=${IN2_SIZE}
    OUT_SIZE=${OUT_SIZE}
    DISABLE_ABI_CHECK=1
    )

target_include_directories (${currentTarget} PUBLIC
    ${XRT_INC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime_lib/test_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime_lib
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)
