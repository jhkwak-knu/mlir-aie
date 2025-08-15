#include "../PassDetail.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "onnx/Dialect/ONNX/IR/ONNXOps.hpp"
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

static llvm::cl::opt<bool>
    DebugAiePlacement("debug-aie-placement",
                    llvm::cl::desc("Print computed AIE placement"),
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
// Tiling algorithm
//===----------------------------------------------------------------------===//
struct OpInfo {
  uint32_t M, K, N;
  Type elemType;
};

struct LevelTile {
  uint32_t TM;
  uint32_t TK;
  uint32_t TN;
  Type elemType;
};

struct TileParam {
  uint32_t numLastSpm;
  LevelTile coreTile;
  std::vector<LevelTile> levelTiles;
};

TileParam findOptimalTileParam(const SystemInfo &sysInfo, const OpInfo &opInfo) {
  // TODO: Implement

  TileParam optimalTileParam{
    .numLastSpm = 1,
    .coreTile   = {.TM=32, .TK=32, .TN=32, .elemType=opInfo.elemType},
    .levelTiles = {{.TM=128, .TK=32, .TN=32, .elemType=opInfo.elemType}}
  };

  return optimalTileParam;
}

//===----------------------------------------------------------------------===//
// Hardware-aware optimization (AIE)
//===----------------------------------------------------------------------===//
struct AieBuf {
  std::string name;
  std::string symbol;
  Value bufValue;
  uint32_t bufSize;
  Type elemType;
  Value prodLockValue;
  Value consLockValue;
};

struct AieCommBuf {
  size_t tileCommIdx;
  bool hasOwnBuf;
  size_t bufIdx;
  uint32_t bufOffset;
  uint32_t bufSize;
  Type elemType;

  void setPhysicalBufInfo(size_t idx, bool own, uint32_t offset, uint32_t size, Type type) {
    hasOwnBuf = own;
    bufIdx = idx;
    bufOffset = offset;
    bufSize = size;
    elemType = type;
  }
};

struct AieTile {
  uint32_t col;
  uint32_t row;
  Value value;

  std::vector<AieCommBuf> inCommBufs;
  std::vector<AieCommBuf> outCommBufs;
  std::vector<AieBuf> allocatedBufs;

  bool operator==(const AieTile &o) const {
    return col == o.col && row == o.row;
  }
};

struct TileComm {
  std::string name;
  size_t fromIdx;
  std::vector<size_t> toIdxs;
  uint32_t bufSize;
  Type elemType;
  uint32_t srcCh, dstCh;
  WireBundle srcWire, dstWire;
};

struct AiePlacementResult {
  std::vector<AieTile> aieTiles;
  std::vector<TileComm> tileComms;

  AieTile& getAieTile(size_t idx) { return aieTiles[idx]; }
  TileComm& getTileComm(size_t idx) { return tileComms[idx]; }
};

AiePlacementResult
optimizeAiePlacement(const TileParam &tileParam) {

  AiePlacementResult result;

  // Place on physical AIE tiles
  uint32_t numTilesPerCol = 6; // Shim: 1, Mem: 1, Compute: 4

  for (uint32_t i = 0; i < numTilesPerCol; ++i) {
    for (uint32_t j = 0; j < tileParam.numLastSpm; ++j) {
      AieTile tile{.col=j, .row=i};
      result.aieTiles.push_back(tile);
    }
  }

  // Configure tile communication paths
  // TODO: Optimize the tile communication paths
  auto findAieTileIdx = [&](uint32_t col, uint32_t row) -> size_t {
    for (size_t idx = 0; idx < result.aieTiles.size(); ++idx) {
      const auto &tile = result.aieTiles[idx];
      if (tile.col == col && tile.row == row)
        return idx;
    }
    llvm_unreachable("Tile not found");
  };

  for (uint32_t i = 0; i < tileParam.numLastSpm; ++i) {

    // Shim tile <-> Mem tile
    auto &[TM, TK, TN, elemType] = tileParam.levelTiles[0];
    uint32_t lhsSize = TM * TK;
    uint32_t rhsSize = TK * TN;
    uint32_t resSize = TM * TN;

    size_t shimIdx = findAieTileIdx(i, 0);
    size_t memIdx  = findAieTileIdx(i, 1);

    TileComm lhsComm{.name="lhs", .fromIdx=shimIdx, .toIdxs={memIdx}, .bufSize=lhsSize, .elemType=elemType,
                    .srcCh=0, .dstCh=0, .srcWire=WireBundle::DMA, .dstWire=WireBundle::DMA};
    TileComm rhsComm{.name="rhs", .fromIdx=shimIdx, .toIdxs={memIdx}, .bufSize=rhsSize, .elemType=elemType,
                    .srcCh=1, .dstCh=1, .srcWire=WireBundle::DMA, .dstWire=WireBundle::DMA};
    TileComm resComm{.name="res", .fromIdx=memIdx, .toIdxs={shimIdx}, .bufSize=resSize, .elemType=elemType,
                    .srcCh=0, .dstCh=0, .srcWire=WireBundle::DMA, .dstWire=WireBundle::DMA};

    result.tileComms.push_back(lhsComm);
    result.tileComms.push_back(rhsComm);
    result.tileComms.push_back(resComm);

    // Mem tile <-> Compute tile
    TileComm rhsBroadcastComm = {.name="rhs", .fromIdx=memIdx, .srcCh=5, .dstCh=1, 
                                .srcWire=WireBundle::DMA, .dstWire=WireBundle::DMA};

    for (uint32_t j = 0; j < (numTilesPerCol - 2); ++j) {
      auto &[TM, TK, TN, elemType] = tileParam.coreTile;
      uint32_t lhsSize = TM * TK;
      uint32_t rhsSize = TK * TN;
      uint32_t resSize = TM * TN;

      size_t computeIdx = findAieTileIdx(i, j + 2);

      TileComm lhsComm{.name="lhs", .fromIdx=memIdx, .toIdxs={computeIdx}, .bufSize=lhsSize, .elemType=elemType,
                      .srcCh=j+1, .dstCh=0, .srcWire=WireBundle::DMA, .dstWire=WireBundle::DMA};
      TileComm resComm{.name="res", .fromIdx=computeIdx, .toIdxs={memIdx}, .bufSize=resSize, .elemType=elemType,
                      .srcCh=0, .dstCh=j+2, .srcWire=WireBundle::DMA, .dstWire=WireBundle::DMA};

      result.tileComms.push_back(lhsComm);
      result.tileComms.push_back(resComm);

      rhsBroadcastComm.bufSize = rhsSize;
      rhsBroadcastComm.elemType = elemType;
      rhsBroadcastComm.toIdxs.push_back(computeIdx);
    }

    result.tileComms.push_back(rhsBroadcastComm);
  }

  // Register communication buffers for each tile
  for (size_t commIdx = 0; commIdx < result.tileComms.size(); ++commIdx) {
    const auto &comm = result.tileComms[commIdx];
    auto &prodTile = result.aieTiles[comm.fromIdx];

    prodTile.outCommBufs.push_back({.tileCommIdx = commIdx});

    for (auto toIdx : comm.toIdxs) {
      auto &consTile = result.aieTiles[toIdx];
      consTile.inCommBufs.push_back({.tileCommIdx = commIdx});
    }
  }

  // Allocate physical buffers for each tile
  for (size_t tileIdx = 0; tileIdx < result.aieTiles.size(); ++tileIdx) {
    auto &tile = result.aieTiles[tileIdx];

    auto allocatePhysicalBuf = [&](AieCommBuf &commBuf) {
      const auto &comm = result.tileComms[commBuf.tileCommIdx];
      AieBuf buf{.name=comm.name, .bufSize=comm.bufSize, .elemType=comm.elemType};
      tile.allocatedBufs.push_back(buf);

      size_t bufIdx = tile.allocatedBufs.size() - 1;
      commBuf.setPhysicalBufInfo(bufIdx, true, 0, buf.bufSize, buf.elemType);
    };

    if(tile.row != 1) { // Shim/Compute tile
      // Allocate physical buffers for all communication buffers
      for (auto &commBuf : tile.inCommBufs) {
        allocatePhysicalBuf(commBuf);
      }
      for (auto &commBuf : tile.outCommBufs) {
        allocatePhysicalBuf(commBuf);
      }      
    } else { // Mem tile
      // Allocate physical buffers only for communication with Shim tiles
      auto isShimTile = [&](size_t idx) {
        return result.aieTiles[idx].row == 0;
      };

      for (auto &commBuf : tile.inCommBufs) {
        const auto &comm = result.tileComms[commBuf.tileCommIdx];
        if (isShimTile(comm.fromIdx)) {
          allocatePhysicalBuf(commBuf);
        }
      }
      for (auto &commBuf : tile.outCommBufs) {
        const auto &comm = result.tileComms[commBuf.tileCommIdx];
        if (llvm::any_of(comm.toIdxs, isShimTile)) {
          allocatePhysicalBuf(commBuf);
        }
      }

      // Link communication buffers (with Compute tiles) to physical buffers
      for (auto &commBuf : tile.inCommBufs) { // Link 'res' buffer (Compute -> Mem -> Shim)
        const auto &comm = result.tileComms[commBuf.tileCommIdx];
        if (!isShimTile(comm.fromIdx)) {
          auto linkCommBufIt = llvm::find_if(tile.outCommBufs, [&](AieCommBuf &outBuf) {
            const auto &outComm = result.tileComms[outBuf.tileCommIdx];
            return outBuf.hasOwnBuf && outComm.name == "res";
          });
          assert(linkCommBufIt != tile.outCommBufs.end() && "Result buffer not allocated");

          const auto &linkBuf = tile.allocatedBufs[linkCommBufIt->bufIdx];
          uint32_t offset = (linkBuf.bufSize / 4) * (result.aieTiles[comm.fromIdx].row - 2);
          commBuf.setPhysicalBufInfo(linkCommBufIt->bufIdx, false, offset, comm.bufSize, comm.elemType);
        }
      }
      for (auto &commBuf : tile.outCommBufs) { // Link 'lhs/rhs' buffer (Shim -> Mem -> Compute)
        const auto &comm = result.tileComms[commBuf.tileCommIdx];
        if (!llvm::any_of(comm.toIdxs, isShimTile)) {
          std::string commName = comm.name;

          auto linkCommBufIt = llvm::find_if(tile.inCommBufs, [&](AieCommBuf &inBuf) {
            const auto &inComm = result.tileComms[inBuf.tileCommIdx];
            return inBuf.hasOwnBuf && inComm.name == commName;
          });
          assert(linkCommBufIt != tile.inCommBufs.end() && "Input buffer not found");

          const auto &linkBuf = tile.allocatedBufs[linkCommBufIt->bufIdx];
          uint32_t offset = 0;
          if(commName == "lhs") {
            offset = (linkBuf.bufSize / 4) * (result.aieTiles[comm.toIdxs.front()].row - 2);
          }
          commBuf.setPhysicalBufInfo(linkCommBufIt->bufIdx, false, offset, comm.bufSize, comm.elemType);
        }
      }
    }
  }

  if (DebugAiePlacement) {
    llvm::dbgs() << "[AiePlacement] Computed AIE Placement:\n";

    llvm::dbgs() << "  aieTiles (" << result.aieTiles.size() << "):\n";
    for (size_t tileIdx = 0; tileIdx < result.aieTiles.size(); ++tileIdx) {
      const auto &tile = result.aieTiles[tileIdx];
      llvm::dbgs() << "    Tile (" << tile.col << ", " << tile.row << "):\n";

      // In Buffers
      llvm::dbgs() << "      inCommBufs:\n";
      for (const auto &in : tile.inCommBufs) {
        const auto &comm = result.getTileComm(in.tileCommIdx);
        llvm::dbgs() << "        - name: " << comm.name
                    << ", hasOwnBuf: " << (in.hasOwnBuf ? "true" : "false")
                    << ", offset: " << in.bufOffset
                    << ", size: " << in.bufSize << "\n";
      }

      // Out Buffers
      llvm::dbgs() << "      outCommBufs:\n";
      for (const auto &out : tile.outCommBufs) {
        const auto &comm = result.getTileComm(out.tileCommIdx);
        llvm::dbgs() << "        - name: " << comm.name
                    << ", hasOwnBuf: " << (out.hasOwnBuf ? "true" : "false")
                    << ", offset: " << out.bufOffset
                    << ", size: " << out.bufSize << "\n";
      }

      // Allocated Buffers
      llvm::dbgs() << "      allocatedBufs (" << tile.allocatedBufs.size() << "):\n";
      for (const auto &buf : tile.allocatedBufs) {
        llvm::dbgs() << "        - name: " << buf.name
                    << ", size: " << buf.bufSize
                    << ", elemType: ";
        buf.elemType.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
    }

    llvm::dbgs() << "  tileComms (" << result.tileComms.size() << "):\n";
    for (size_t commIdx = 0; commIdx < result.tileComms.size(); ++commIdx) {
      const auto &comm = result.tileComms[commIdx];
      llvm::dbgs() << "    " << comm.name << ": (" << result.getAieTile(comm.fromIdx).col
                  << ", " << result.getAieTile(comm.fromIdx).row << ") -> ";
      for (size_t toIdx : comm.toIdxs) {
        const auto &toTile = result.getAieTile(toIdx);
        llvm::dbgs() << "(" << toTile.col << ", " << toTile.row << ") ";
      }
      llvm::dbgs() << ", bufSize=" << comm.bufSize << ", elemType=";
      comm.elemType.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  }

  return result;
}

void generateAieOps(ConversionPatternRewriter &rewriter,
                    AiePlacementResult &placement,
                    const TileParam &tileParam) {
  // TODO: Implement
  
  // Create new module for AIE dialect
  MLIRContext *ctx = rewriter.getContext();
  auto loc = mlir::UnknownLoc::get(ctx);
  auto aieModule = ModuleOp::create(loc);
  OpBuilder builder(aieModule.getBodyRegion());
  builder.setInsertionPointToStart(aieModule.getBody());

  // Generate AIE DeviceOp
  std::vector<AIEDevice> devices{AIEDevice::npu2_1col, AIEDevice::npu2_2col,
                                 AIEDevice::npu2_3col, AIEDevice::npu2_4col,
                                 AIEDevice::npu2_5col, AIEDevice::npu2_6col,
                                 AIEDevice::npu2_7col, AIEDevice::npu2};
  auto deviceOp = builder.create<DeviceOp>(loc, devices[tileParam.numLastSpm - 1]);

  // Ensure it has a body block, and point insertion into it
  deviceOp.getRegion().emplaceBlock();
  DeviceOp::ensureTerminator(deviceOp.getBodyRegion(), builder, loc);
  builder.setInsertionPointToStart(deviceOp.getBody());

  // Save the mlir code composed of AIE dialect
  std::error_code ec;
  llvm::raw_fd_ostream out("./aie.mlir", ec);
  aieModule->print(out);
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
    OpInfo opInfo;

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

    TileParam optimalTileParam = findOptimalTileParam(systemInfo.value(), opInfo);

    // Perform hardware-aware optimization for AIE
    auto AiePlacement = optimizeAiePlacement(optimalTileParam);

    // Generate AIE Ops
    generateAieOps(rewriter, AiePlacement, optimalTileParam);
                  
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
                                 MLIRContext *ctx) {
  // Matmul
  patterns.insert<ConvertONNXMatMulToAIE>(ctx);
}

struct ConvertONNXToAIE
    : public ConvertONNXToAIEBase<ConvertONNXToAIE> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<xilinx::AIEX::AIEXDialect>();
  }

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
