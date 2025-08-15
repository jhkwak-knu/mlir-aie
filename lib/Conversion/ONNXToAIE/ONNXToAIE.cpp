#include "../PassDetail.h"

#include "onnx/Conversion/ONNXToAIE/ONNXToAIE.h"

#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/Twine.h"

#include <string>
#include <vector>
#include <optional>
#include <algorithm>

using namespace mlir;
using namespace xilinx::AIE;

namespace {

//===----------------------------------------------------------------------===//
// Debugging options
//===----------------------------------------------------------------------===//
static llvm::cl::opt<bool>
    DebugSystemInfo("debug-system-info",
                    llvm::cl::desc("Print loaded SystemInfo"),
                    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// System Information
//===----------------------------------------------------------------------===//
struct SpmLevel {
  uint32_t level;
  uint32_t numSpms;
  uint64_t spmSizeBytes;
};

struct SystemInfo {
  std::string name;
  uint32_t numSpmLevels;
  std::vector<SpmLevel> spmLevels;
  uint32_t totalCores;
};

std::optional<SystemInfo> loadSystemInfo(const std::string &filePath) {
  // 1) Read the entire file into a MemoryBuffer
  auto bufOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (!bufOrErr) {
    llvm::errs() << "Error: cannot open file '" << filePath << "' (" << bufOrErr.getError().message() << ")\n";
    return std::nullopt;
  }
  StringRef jsonText = (*bufOrErr)->getBuffer();

  // 2) Parse JSON
  auto jsonOrErr = llvm::json::parse(jsonText);
  if (!jsonOrErr) {
    llvm::errs() << "Error: JSON parse failed in '" << filePath << "'\n";
    return std::nullopt;
  }
  auto *rootObj = jsonOrErr->getAsObject();
  if (!rootObj) {
    llvm::errs() << "Error: root JSON is not an object\n";
    return std::nullopt;
  }

  // 3) Extract "system" object
  auto *sysObj = rootObj->getObject("system");
  if (!sysObj) {
    llvm::errs() << "Error: missing 'system' object\n";
    return std::nullopt;
  }

  SystemInfo info;

  // 4) Read scalar fields
  if (auto nameOpt = sysObj->getString("name")) {
    info.name = nameOpt->str();
  } else {
    llvm::errs() << "Error: missing 'name'\n";
    return std::nullopt;
  }

  if (auto lvlOpt = sysObj->getInteger("num_spm_levels")) {
    info.numSpmLevels = static_cast<uint32_t>(*lvlOpt);
  } else {
    llvm::errs() << "Error: missing 'num_spm_levels'\n";
    return std::nullopt;
  }

  if (auto coresOpt = sysObj->getInteger("total_cores")) {
    info.totalCores = static_cast<uint32_t>(*coresOpt);
  } else {
    llvm::errs() << "Error: missing 'total_cores'\n";
    return std::nullopt;
  }

  // 5) Read "spm_levels" array, dynamically sizing vector
  auto *arr = sysObj->getArray("spm_levels");
  if (!arr) {
    llvm::errs() << "Error: missing 'spm_levels' array\n";
    return std::nullopt;
  }
  if (arr->size() != info.numSpmLevels) {
    llvm::errs() << "Error: 'spm_levels' size (" << arr->size()
                 << ") does not match num_spm_levels ("
                 << info.numSpmLevels << ")\n";
    return std::nullopt;
  }
  info.spmLevels.resize(info.numSpmLevels);

  for (uint32_t i = 0; i < info.numSpmLevels; ++i) {
    auto *lvlObj = (*arr)[i].getAsObject();
    if (!lvlObj) {
      llvm::errs() << "Error: spm_levels[" << i << "] is not an object\n";
      return std::nullopt;
    }

    if (auto lvlOpt = lvlObj->getInteger("level")) {
      info.spmLevels[i].level = static_cast<uint32_t>(*lvlOpt);
    } else {
      llvm::errs() << "Error: missing 'level' in spm_levels[" << i << "]\n";
      return std::nullopt;
    }

    if (auto nOpt = lvlObj->getInteger("num_spms")) {
      info.spmLevels[i].numSpms = static_cast<uint32_t>(*nOpt);
    } else {
      llvm::errs() << "Error: missing 'num_spms' in spm_levels[" << i << "]\n";
      return std::nullopt;
    }

    if (auto sOpt = lvlObj->getInteger("spm_size_bytes")) {
      info.spmLevels[i].spmSizeBytes = static_cast<uint64_t>(*sOpt);
    } else {
      llvm::errs() << "Error: missing 'spm_size_bytes' in spm_levels[" << i << "]\n";
      return std::nullopt;
    }
  }

  if (DebugSystemInfo) {
    llvm::dbgs() << "[SystemInfo] Loaded SystemInfo:\n"
                 << "  name:         " << info.name       << "\n"
                 << "  numSpmLevels: " << info.numSpmLevels << "\n"
                 << "  totalCores:   " << info.totalCores   << "\n";
    for (auto &lvl : info.spmLevels)
      llvm::dbgs() << "    level=" << lvl.level
                   << " numSpms=" << lvl.numSpms
                   << " sizeBytes=" << lvl.spmSizeBytes << "\n";  
  }

  return info;
}

//===----------------------------------------------------------------------===//
// Conversion classes for ONNX ops
//===----------------------------------------------------------------------===//
class ConvertONNXMatMulToAIE
    : public OpConversionPattern<onnx::MatMulOp> {

public:
  using OpConversionPattern<onnx::MatMulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(onnx::MatMulOp op,
                  OpConversionPattern<onnx::MatMulOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    Location loc = op.getLoc();

    // Read the system information
    std::string filePath = "/home/ace/ryzen_ai/mlir-aie-dev/mlir-aie/include/onnx/Target/XDNA2/xdna2_info.json";
    auto systemInfo = loadSystemInfo(filePath);
    if (!systemInfo) {
      return rewriter.notifyMatchFailure(op, llvm::Twine("Unable to open JSON file: '") + filePath + "'");
    }

    // Find optimal tiling parameters
//===----------------------------------------------------------------------===//
// Pass main (ConvertONNXToAIE)
//===----------------------------------------------------------------------===//
void populateONNXRewritePatterns(RewritePatternSet &patterns,
                                 MLIRContext *ctx) {
  // Matmul
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
