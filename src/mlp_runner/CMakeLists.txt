# Build file for src/mlp_runner/. Invoked standalone via
# experiments/scripts/build_runner.sh.
#
# Notion step 21 Step 6-3 / 6-4 / 6-6: produces
#   mlp_runner  - main host entry (CPU dispatcher + XRT dispatcher when
#                 XRT headers/libs are found)
#   mlp_smoke   - standalone test harness (forward pass numerical smoke,
#                 RAPL / instrumentation unit tests)

cmake_minimum_required(VERSION 3.16)
project(mlp_runner CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

# Match the toolchain used elsewhere in this repo for stdfloat bf16.
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER g++-13)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# XRT paths (can be overridden from CLI; defaults match host.cpp build).
set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT headers")
set(XRT_LIB_DIR /opt/xilinx/xrt/lib     CACHE STRING "Path to XRT libraries")

option(MLP_RUNNER_WITH_XRT "Build with the XRT-backed NPU dispatcher" ON)

set(MLP_RUNNER_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${REPO_ROOT}/runtime_lib/test_lib
  # tile_ops.h / tile_order.h / tiling_param.h live with the onnx_matmul host.
  ${REPO_ROOT}/test/onnx-mlir/src
)

set(MLP_RUNNER_CORE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/config_loader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mlp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/npu_dispatch.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rapl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/measurement.cpp
)

set(XRT_SOURCES "")
if(MLP_RUNNER_WITH_XRT)
  # Fail loud if XRT is missing. This mirrors the expectation that measurement
  # hosts already source test/onnx-mlir/scripts/setup_env.sh before building.
  if(NOT EXISTS "${XRT_INC_DIR}/xrt/xrt_bo.h")
    message(FATAL_ERROR "XRT headers not found at ${XRT_INC_DIR}. "
            "Configure with -DMLP_RUNNER_WITH_XRT=OFF to build CPU-only.")
  endif()
  list(APPEND XRT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/xrt_dispatcher.cpp
    ${REPO_ROOT}/runtime_lib/test_lib/test_utils.cpp
  )
  add_definitions(-DMLP_RUNNER_HAS_XRT=1 -DDISABLE_ABI_CHECK=1)
endif()

add_library(mlp_runner_core STATIC ${MLP_RUNNER_CORE_SOURCES} ${XRT_SOURCES})
target_include_directories(mlp_runner_core PUBLIC ${MLP_RUNNER_INCLUDE_DIRS})

if(MLP_RUNNER_WITH_XRT)
  target_include_directories(mlp_runner_core PUBLIC ${XRT_INC_DIR})
  target_link_directories(mlp_runner_core PUBLIC ${XRT_LIB_DIR})
  target_link_libraries(mlp_runner_core PUBLIC xrt_coreutil)
endif()

add_executable(mlp_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/mlp_runner.cpp
)
target_link_libraries(mlp_runner PRIVATE mlp_runner_core)

add_executable(mlp_smoke
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlp_smoke.cpp
)
target_link_libraries(mlp_smoke PRIVATE mlp_runner_core)

enable_testing()
add_test(NAME mlp_smoke COMMAND mlp_smoke)
