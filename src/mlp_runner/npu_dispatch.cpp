//===- npu_dispatch.cpp - Implementations + factory -------------*- C++ -*-===//

#include "npu_dispatch.h"

#include <stdexcept>
#include <string>

#include "gemm_cpu.h"

namespace mlp_runner {

void CpuDispatcher::dispatch(const LayerKernelEntry& layer,
                             const float* A, const float* B, float* C) {
  gemmCpu(A, B, C, layer.M, layer.K, layer.N);
}

XrtDispatcherStub::XrtDispatcherStub() {
  throw std::runtime_error(
      "XrtDispatcherStub: NPU dispatcher not yet implemented (Step 6-6). "
      "Invoke the runner with --backend cpu for now.");
}

void XrtDispatcherStub::dispatch(const LayerKernelEntry&, const float*,
                                 const float*, float*) {
  // Unreachable — the constructor throws first.
  throw std::runtime_error("XrtDispatcherStub::dispatch called unexpectedly");
}

std::unique_ptr<Dispatcher> makeDispatcher(const std::string& backend) {
  if (backend == "cpu")
    return std::make_unique<CpuDispatcher>();
  if (backend == "npu" || backend == "xrt")
    return std::make_unique<XrtDispatcherStub>();
  throw std::runtime_error("unknown dispatcher backend: " + backend);
}

}  // namespace mlp_runner
