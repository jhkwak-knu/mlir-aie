add_mlir_conversion_library(MLIRONNXToAIE
  TileParam.cpp
  AiePlacement.cpp
  AieEmitter.cpp
  ONNXToAIE.cpp

  ADDITIONAL_HEADER_DIRS
  $(CMAKE_CURRENT_SRC_DIR)/../../../../include/onnx/Conversion/ONNXToAIE

  DEPENDS
  MLIRONNXConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIERT
  )
