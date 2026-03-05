#ifndef ONNX_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H
#define ONNX_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace onnx {

std::unique_ptr<mlir::Pass> createConvertONNXToAIEPass();

} // namespace onnx

#endif // ONNX_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H
