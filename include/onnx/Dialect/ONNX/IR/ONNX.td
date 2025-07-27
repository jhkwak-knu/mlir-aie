// SPDX-License-Identifier: Apache-2.0

//===--- ONNXOps.td -- ONNX Dialect Operation Definitions ----*- tablegen -===//
//
// Copyright 2019-2022 The IBM Research Authors
//
// =============================================================================
//
// Defines ONNX Dialect Definitions, Types, and Operations.
//
// When adding non-standard ONNX ops, please add them in AdditionalONNXOps.td.
//
// After changes that impact ONNX, run "make OMONNXOpsIncTranslation".
// After changes that impact the documentation of the ops, run
// "make onnx-mlir-docs".
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_OPS
#define ONNX_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Definition of the ONNX dialect.
//===----------------------------------------------------------------------===//

def ONNX_Dialect : Dialect {
  let name = "onnx";
  let summary = "A high-level dialect for the ONNX specification";
  let cppNamespace = "::mlir";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
}

//===----------------------------------------------------------------------===//
// ONNX Operations
//===----------------------------------------------------------------------===//

// Base class for ONNX dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class ONNX_Op<string mnemonic, list<Trait> traits = []> :
  Op<ONNX_Dialect, mnemonic, traits> ;

// The tablegen code onnxop.in is generated with gen_doc.py
// clone and install onnx
//    git clone --recursive https://github.com/onnx/onnx.git
//  set up env for anaconda3 and for ONNX-MLIR (BOOSTROOT, cmake, gcc ...)
//    cd onnx
// install onnx
//   CC=gcc CXX=g++ pip install -e .
// run the script
//   python onnx/defs/gen_doc.py
// result is in docs/onnx_ops.td.inc
// current limitations:
//  1. Attributes are not processed
//  2. output type inference not implemented except Add
//  3. Type Attribute: 'optional' and 'Variadic hetergeneous' are ignored
//  4. type of string, complex64 and complex128 for input/output are ignored
//  5. unsigned int are treated as signed one

include "mlir/Interfaces/SideEffectInterfaces.td"
include "onnx/Dialect/ONNX/IR/ONNXOps.td.inc"

#endif // ONNX_OPS
