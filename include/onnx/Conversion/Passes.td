#ifndef ONNX_CONVERSION_PASSES
#define ONNX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ONNXToAIE
//===----------------------------------------------------------------------===//

def ConvertONNXToAIE : Pass<"convert-onnx-to-aie", operation="func::FuncOp"> {
  let summary = "Convert ONNX dialect (MatMul op) to AIE dialect";
  let description = [{
    This pass converts ONNX dialect (MatMul op) to AIE dialect calls to builtins.
  }];
  let constructor = "onnx::createConvertONNXToAIEPass()";
  let dependentDialects = ["xilinx::AIE::AIEDialect",
                           "onnx::ONNXDialect"];
}

#endif // ONNX_CONVERSION_PASSES
