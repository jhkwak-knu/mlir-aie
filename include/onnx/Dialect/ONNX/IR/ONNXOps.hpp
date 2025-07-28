/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- ONNXOps.hpp - ONNX Operations -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNX_OPS_H
#define ONNX_MLIR_ONNX_OPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "onnx/Dialect/ONNX/IR/ONNXDialect.hpp"

namespace onnx {
// OpSet level supported by onnx-mlir
// To update all occurrence of the current ONNX opset, please grep
// "CURRENT_ONNX_OPSET" and update all locations accordingly.
static constexpr int CURRENT_ONNX_OPSET = 22;
} // end namespace onnx

#define GET_OP_CLASSES
#include "onnx/Dialect/ONNX/IR/ONNX.h.inc"
#endif
