# Build file for src/distilbert_runner/. Notion task #22 §4-3.
#
# Mirrors src/mlp_runner/CMakeLists.txt as much as possible. The shared
# C++ primitives (gemm_cpu, npu_dispatch, rapl, xrt_dispatcher) live in
# src/mlp_runner/ and we just compile their .cpp files into our static
# library — copying them into src/distilbert_runner/ would force every
# bug fix to land twice. config_loader.cpp is also reused for its
# loadLayerKernels() (the configurations.json schema is model-agnostic).
#
# bert.cpp + ggml are pulled in as a sub-project from external/bert.cpp.
# The vendored copy declines all GPU backends by default; we further
# turn off ggml's tests / examples so we only build what we link.

cmake_minimum_required(VERSION 3.16)
project(distilbert_runner CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER g++-13)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

set(REPO_ROOT     "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(MLP_RUNNER_DIR "${REPO_ROOT}/src/mlp_runner")
set(BERT_CPP_DIR   "${REPO_ROOT}/external/bert.cpp")

# bert.cpp options (overrideable from the top-level CLI). Off by default
# so no extra deps get pulled in.
option(GGML_CUBLAS "ggml: use cuBLAS" OFF)
option(GGML_METAL  "ggml: use Metal"  OFF)

# Pull bert.cpp + ggml into the build as a sub-project. EXCLUDE_FROM_ALL
# keeps bert.cpp's bin/main / bin/quantize from being built unless
# something here actually depends on them.
add_subdirectory(${BERT_CPP_DIR} ${CMAKE_BINARY_DIR}/bert_cpp_build EXCLUDE_FROM_ALL)

# XRT (optional)
set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT headers")
set(XRT_LIB_DIR /opt/xilinx/xrt/lib     CACHE STRING "Path to XRT libraries")
option(DISTILBERT_RUNNER_WITH_XRT
       "Build with the XRT-backed NPU dispatcher" ON)

# Sources shared verbatim with mlp_runner. config_loader.cpp brings the
# model-agnostic loadLayerKernels() function used to wire configurations.json
# into LayerKernelEntry records.
set(SHARED_SOURCES
  ${MLP_RUNNER_DIR}/config_loader.cpp
  ${MLP_RUNNER_DIR}/gemm_cpu.cpp
  ${MLP_RUNNER_DIR}/npu_dispatch.cpp
  ${MLP_RUNNER_DIR}/rapl.cpp
)

if(DISTILBERT_RUNNER_WITH_XRT)
  if(NOT EXISTS "${XRT_INC_DIR}/xrt/xrt_bo.h")
    message(FATAL_ERROR "XRT headers not found at ${XRT_INC_DIR}. "
            "Configure with -DDISTILBERT_RUNNER_WITH_XRT=OFF to build CPU-only.")
  endif()
  list(APPEND SHARED_SOURCES
    ${MLP_RUNNER_DIR}/xrt_dispatcher.cpp
    ${REPO_ROOT}/runtime_lib/test_lib/test_utils.cpp
  )
  add_definitions(-DMLP_RUNNER_HAS_XRT=1 -DDISABLE_ABI_CHECK=1)
endif()

set(DISTILBERT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/config_loader_distilbert.cpp
)

add_library(distilbert_runner_core STATIC ${SHARED_SOURCES} ${DISTILBERT_SOURCES})
target_include_directories(distilbert_runner_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MLP_RUNNER_DIR}
  ${REPO_ROOT}/runtime_lib/test_lib   # cxxopts.hpp + test_utils
  ${REPO_ROOT}/test/onnx-mlir/src     # tile_ops.h / tile_order.h / tiling_param.h
  ${BERT_CPP_DIR}/src                 # bert.h
  ${BERT_CPP_DIR}/ggml/include        # ggml.h / ggml-backend.h
)
if(DISTILBERT_RUNNER_WITH_XRT)
  target_include_directories(distilbert_runner_core PUBLIC ${XRT_INC_DIR})
  target_link_directories(distilbert_runner_core PUBLIC ${XRT_LIB_DIR})
  target_link_libraries(distilbert_runner_core PUBLIC xrt_coreutil)
endif()
# bert / ggml come from add_subdirectory targets.
target_link_libraries(distilbert_runner_core PUBLIC bert ggml)

add_executable(distilbert_runner ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(distilbert_runner PRIVATE distilbert_runner_core)
