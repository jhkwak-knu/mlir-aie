add_mlir_dialect_library(ONNX
  ONNXDialect.cpp
  ONNXOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRONNXIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
