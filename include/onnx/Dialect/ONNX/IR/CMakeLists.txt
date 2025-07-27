# Add ONNX dialect
add_mlir_dialect(ONNX onnx)
add_mlir_doc(ONNX ONNXDialect ./ -gen-dialect-doc)
set(LLVM_TARGET_DEFINITIONS ONNX.td)
