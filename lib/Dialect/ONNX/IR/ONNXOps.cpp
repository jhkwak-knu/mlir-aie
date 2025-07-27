/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXOps.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect operations.
//
//===----------------------------------------------------------------------===//

#include "onnx/Dialect/ONNX/IR/ONNXOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "onnx/Dialect/ONNX/IR/ONNX.cpp.inc"
