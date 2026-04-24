//===- npu_dispatch.h - GEMM dispatcher abstraction -------------*- C++ -*-===//
//
// Thin interface so the forward pass can be driven against either:
//   * a CPU reference implementation (for verification / --cpu-only), or
//   * the XRT-backed NPU in Step 6-6.
//
// Step 6-3 delivers the CPU dispatcher. NpuDispatcher (XRT-backed) is a stub
// until the host runner integration is complete.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

#include "config_loader.h"

namespace mlp_runner {

/// GEMM dispatcher: one call per (layer, A, B, C) triple. Ownership of A/B/C
/// buffers stays with the caller.
struct Dispatcher {
  virtual ~Dispatcher() = default;
  virtual void dispatch(const LayerKernelEntry& layer,
                        const float* A, const float* B, float* C) = 0;
  virtual const char* name() const = 0;
};

/// Always runs the CPU reference. Use for numerical verification and for
/// --cpu-only smoke runs. Ignores `layer.xclbin_path` entirely.
struct CpuDispatcher : Dispatcher {
  void dispatch(const LayerKernelEntry& layer,
                const float* A, const float* B, float* C) override;
  const char* name() const override { return "cpu"; }
};

/// Placeholder for the XRT-backed dispatcher implemented in Step 6-6.
/// Throws std::runtime_error immediately if instantiated so that misrouted
/// calls fail loud in the current step.
struct XrtDispatcherStub : Dispatcher {
  XrtDispatcherStub();  // throws
  void dispatch(const LayerKernelEntry& layer,
                const float* A, const float* B, float* C) override;
  const char* name() const override { return "xrt-stub"; }
};

/// Factory: "cpu" -> CpuDispatcher; "npu" / "xrt" -> XrtDispatcherStub (throws).
std::unique_ptr<Dispatcher> makeDispatcher(const std::string& backend);

}  // namespace mlp_runner
