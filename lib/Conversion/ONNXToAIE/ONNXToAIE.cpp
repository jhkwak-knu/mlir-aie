#include "../PassDetail.h"

#include "onnx/Conversion/ONNXToAIE/ONNXToAIE.h"
#include "onnx/Dialect/ONNX/IR/ONNXOps.hpp"
#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace xilinx::AIE;
using namespace onnx;

namespace {

class ConvertONNXMatMulToAIE
    : public OpConversionPattern<onnx::MatMulOp> {
public:
  using OpConversionPattern<onnx::MatMulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(onnx::MatMulOp op,
                  OpConversionPattern<onnx::MatMulOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << "onnx.MatMul conversion is not implemented\n";
    return failure();
  }
};

void populateONNXRewritePatterns(RewritePatternSet &patterns,
                                 MLIRContext *ctx) {
  // matmul
  patterns.insert<ConvertONNXMatMulToAIE>(ctx);
}

struct ConvertONNXToAIE
    : public ConvertONNXToAIEBase<ConvertONNXToAIE> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateONNXRewritePatterns(patterns, &getContext());
    ConversionTarget target(getContext());

    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addIllegalDialect<onnx::ONNXDialect>();
    target.addLegalDialect<xilinx::AIE::AIEDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> onnx::createConvertONNXToAIEPass() {
  return std::make_unique<ConvertONNXToAIE>();
}
