#ifndef ONNX_CONVERSION_PASSES_H
#define ONNX_CONVERSION_PASSES_H

#include "onnx/Conversion/ONNXToAIE/ONNXToAIE.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace onnx {

#define GEN_PASS_DECL
#include "onnx/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "onnx/Conversion/Passes.h.inc"

} // namespace onnx

#endif // ONNX_CONVERSION_PASSES_H
