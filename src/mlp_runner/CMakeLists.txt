# Build file for src/mlp_runner/. Invoked standalone via experiments/scripts/build_runner.sh.
#
# Notion step 21 Step 6-3: produces two binaries
#   mlp_runner  : main host entry (CPU dispatcher now, NPU dispatcher in Step 6-6)
#   mlp_smoke   : standalone test harness (forward pass numerical smoke)

cmake_minimum_required(VERSION 3.16)
project(mlp_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# cxxopts lives under <repo>/runtime_lib/test_lib. The mlp_runner CMakeLists
# sits at <repo>/src/mlp_runner/, so walk two levels up.
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

set(MLP_RUNNER_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${REPO_ROOT}/runtime_lib/test_lib
)

set(MLP_RUNNER_CORE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/config_loader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mlp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/npu_dispatch.cpp
)

add_library(mlp_runner_core STATIC ${MLP_RUNNER_CORE_SOURCES})
target_include_directories(mlp_runner_core PUBLIC ${MLP_RUNNER_INCLUDE_DIRS})

add_executable(mlp_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/mlp_runner.cpp
)
target_link_libraries(mlp_runner PRIVATE mlp_runner_core)

# Standalone smoke harness (no config / json / CLI). Builds and runs forward
# pass over hand-rolled tiny LayerKernelEntry list to keep test hermetic.
add_executable(mlp_smoke
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlp_smoke.cpp
)
target_link_libraries(mlp_smoke PRIVATE mlp_runner_core)

enable_testing()
add_test(NAME mlp_smoke COMMAND mlp_smoke)
