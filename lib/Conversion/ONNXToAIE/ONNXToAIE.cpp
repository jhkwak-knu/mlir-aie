//===- ONNXToAIE.cpp - ONNX → AIE dialect pass registration ------*- C++ -*-===//
//
// Pass entry point for --convert-onnx-to-aie.  CLI options are defined here
// and forwarded to the library functions in TileParam / AiePlacement / AieEmitter.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "AiePlacement.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "onnx/Dialect/ONNX/IR/ONNXOps.hpp"
#include "onnx/Conversion/ONNXToAIE/ONNXToAIE.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace onnx_to_aie;

namespace {

//===----------------------------------------------------------------------===//
// CLI options (forwarded as function parameters to library code)
//===----------------------------------------------------------------------===//
static llvm::cl::opt<bool>
    DebugSystemInfo("debug-system-info",
                    llvm::cl::desc("Print loaded SystemInfo"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool>
    DebugTileParam("debug-tile-param",
                    llvm::cl::desc("Print loaded TileParam"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool>
    DebugAiePlacement("debug-aie-placement",
                    llvm::cl::desc("Print computed AIE placement"),
                    llvm::cl::init(false));

static llvm::cl::opt<std::string>
    TileParamJson("tile-param-json",
                  llvm::cl::desc("Path to tc.json tiling config file"),
                  llvm::cl::init(""));

static llvm::cl::opt<std::string>
    SystemInfoJson("system-info-json",
                   llvm::cl::desc("Path to xdna2_info.json hardware config file"),
                   llvm::cl::init(""));

static llvm::cl::opt<std::string>
    AieMlirOutput("aie-mlir-output",
                  llvm::cl::desc("Output path for generated aie.mlir"),
                  llvm::cl::init("./aie.mlir"));

//===----------------------------------------------------------------------===//
// Conversion classes for ONNX ops
//===----------------------------------------------------------------------===//
class ConvertONNXMatMulToAIE
    : public OpConversionPattern<onnx::MatMulOp> {

  SystemInfo sysInfo_;

public:
  ConvertONNXMatMulToAIE(MLIRContext *ctx, SystemInfo sysInfo)
      : OpConversionPattern<onnx::MatMulOp>(ctx),
        sysInfo_(std::move(sysInfo)) {}

  LogicalResult
  matchAndRewrite(onnx::MatMulOp op,
                  OpConversionPattern<onnx::MatMulOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    // Find optimal tiling parameters
    MatmulOpInfo opInfo;

    auto ty = op.getResult().getType();
    if (auto shapedTy = llvm::dyn_cast<ShapedType>(ty)) {
      opInfo.elemType = shapedTy.getElementType();
    } else {
      return rewriter.notifyMatchFailure(op, "expected a shaped type");
    }

    auto lhsType = llvm::dyn_cast<ShapedType>(op.getA().getType());
    auto rhsType = llvm::dyn_cast<ShapedType>(op.getB().getType());
    auto resType = llvm::dyn_cast<ShapedType>(op.getResult().getType());

    if (!lhsType || !rhsType || !resType ||
        !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape() ||
        !resType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "MatMul operands/results must have static shapes");
    }

    opInfo.M = static_cast<uint32_t>(resType.getShape()[0]);
    opInfo.N = static_cast<uint32_t>(resType.getShape()[1]);
    opInfo.K = static_cast<uint32_t>(lhsType.getShape()[1]);

    TileParam optimalTileParam = findOptimalTileParam(
        sysInfo_, opInfo, TileParamJson, DebugTileParam);

    // Perform hardware-aware optimization for AIE
    TilingContext tilingCtx = buildTilingContext(optimalTileParam, sysInfo_);
    auto placement = optimizeAiePlacement(tilingCtx, DebugAiePlacement);

    // Generate AIE Ops
    generateAieOps(rewriter, placement, tilingCtx, AieMlirOutput);

    // Dummy to prevent result type errors
    auto resultType = op.getResult().getType();
    auto dummy = rewriter.create<mlir::arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));

    rewriter.replaceOp(op, dummy);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass main (ConvertONNXToAIE)
//===----------------------------------------------------------------------===//
void populateONNXRewritePatterns(RewritePatternSet &patterns,
                                 MLIRContext *ctx, SystemInfo sysInfo) {
  // Matmul
  patterns.insert<ConvertONNXMatMulToAIE>(ctx, std::move(sysInfo));
}

struct ConvertONNXToAIE
    : public ConvertONNXToAIEBase<ConvertONNXToAIE> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<xilinx::AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    // Load system info once at pass level, not per-op in matchAndRewrite.
    if (SystemInfoJson.empty()) {
      llvm::errs() << "Error: --system-info-json is required\n";
      signalPassFailure();
      return;
    }
    auto systemInfo = loadSystemInfo(SystemInfoJson, DebugSystemInfo);
    if (!systemInfo) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    populateONNXRewritePatterns(patterns, &getContext(), std::move(*systemInfo));
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
