//===- xrt_dispatcher.h - XRT-backed NPU dispatcher -------------*- C++ -*-===//
//
// Notion step 21 Step 6-6: replace XrtDispatcherStub with a real per-layer
// dispatcher. Reuses the existing `test/onnx-mlir/src/` tiling + kernel
// dispatch helpers (tile_ops.h / tile_order.h / tiling_param.h) so the
// numerical behaviour of the standalone onnx_matmul host is preserved.
//
// Lifecycle:
//   * One XrtDispatcher per mlp_runner process.
//   * `xrt::device` is opened once on construction.
//   * Per LayerKernelEntry we lazily load the xclbin + instruction stream
//     into a LayerContext and cache it keyed by xclbin path. Subsequent
//     dispatches for the same layer reuse the context.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "npu_dispatch.h"

namespace mlp_runner {

class XrtDispatcher : public Dispatcher {
 public:
  XrtDispatcher();
  ~XrtDispatcher() override;

  void dispatch(const LayerKernelEntry& layer,
                const float* A, const float* B, float* C) override;
  const char* name() const override { return "xrt"; }

 private:
  struct Impl;                       // PIMPL so the header stays XRT-free
  std::unique_ptr<Impl> impl_;
};

}  // namespace mlp_runner
