#include "../PassDetail.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "onnx/Dialect/ONNX/IR/ONNXOps.hpp"
#include "onnx/Conversion/ONNXToAIE/ONNXToAIE.h"

#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
    .levelTiles = {{.TM=512, .TK=64, .TN=64, .elemType=opInfo.elemType}}
  };

  return optimalTileParam;
}

//===----------------------------------------------------------------------===//
// Hardware-aware optimization (AIE)
//===----------------------------------------------------------------------===//
struct AieBuf {
  std::string name;
  std::string symbol;
  uint32_t bufSize;
  Type elemType;
  Value bufValue;
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
  uint32_t commCount;
  uint32_t commElemSize;
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

  // Set variables
  uint32_t numCompTile = 4;
  uint32_t numTilesPerCol = numCompTile + 2; // Shim: 1, Mem: 1, Compute: 4
  auto numCols = tileParam.numLastSpm;
  auto [TM, TK, TN, elemType] = tileParam.coreTile;
  auto mCountInMemTile = tileParam.levelTiles[0].TM / TM;
  auto kCountInMemTile = tileParam.levelTiles[0].TK / TK;
  auto nCountInMemTile = tileParam.levelTiles[0].TN / TN;
  auto wireBundle = WireBundle::DMA;
  
  // Place on physical AIE tiles
  for (uint32_t i = 0; i < numCols; ++i) {
    for (uint32_t j = 0; j < numTilesPerCol; ++j) {
      AieTile tile{.col=i, .row=j};
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

  for (uint32_t i = 0; i < numCols; ++i) {
    // Shim tile <-> Mem tile
    uint32_t lhsSizeInMemTile = TM * TK * mCountInMemTile * kCountInMemTile;
    uint32_t rhsSizeInMemTile = TK * TN * kCountInMemTile;
    uint32_t resSizeInMemTile = TM * TN * mCountInMemTile * nCountInMemTile;
    uint32_t rhsCommCount = nCountInMemTile;

    size_t shimIdx = findAieTileIdx(i, 0);
    size_t memIdx  = findAieTileIdx(i, 1);

    TileComm lhsCommShimToMem{.name="lhs", .fromIdx=shimIdx, .toIdxs={memIdx}, .commCount=1, .commElemSize=lhsSizeInMemTile,
                              .elemType=elemType, .srcCh=0, .dstCh=0, .srcWire=wireBundle, .dstWire=wireBundle};
    TileComm rhsCommShimToMem{.name="rhs", .fromIdx=shimIdx, .toIdxs={memIdx}, .commCount=rhsCommCount, .commElemSize=rhsSizeInMemTile,
                              .elemType=elemType, .srcCh=1, .dstCh=1, .srcWire=wireBundle, .dstWire=wireBundle};
    TileComm resCommMemToShim{.name="res", .fromIdx=memIdx, .toIdxs={shimIdx}, .commCount=1, .commElemSize=resSizeInMemTile,
                              .elemType=elemType, .srcCh=0, .dstCh=0, .srcWire=wireBundle, .dstWire=wireBundle};

    result.tileComms.push_back(lhsCommShimToMem);
    result.tileComms.push_back(rhsCommShimToMem);
    result.tileComms.push_back(resCommMemToShim);

    // Mem tile <-> Compute tile
    uint32_t lhsSizeInCompTile = TM * TK;
    uint32_t rhsSizeInCompTile = TK * TN;
    uint32_t resSizeInCompTile = TM * TN;

    uint32_t lhsCommMemToCompCount = (mCountInMemTile * kCountInMemTile) / numCompTile;
    uint32_t rhsCommMemToCompCount = kCountInMemTile;
    uint32_t resCommCompToMemCount = (mCountInMemTile * nCountInMemTile) / numCompTile;

    TileComm rhsBcastCommMemToComp = {.name="rhs", .fromIdx=memIdx, .commCount=rhsCommMemToCompCount, .commElemSize=rhsSizeInCompTile,
                                      .elemType=elemType, .srcCh=5, .dstCh=1, .srcWire=wireBundle, .dstWire=wireBundle};

    for (uint32_t j = 0; j < numCompTile; ++j) {
      size_t computeIdx = findAieTileIdx(i, j + 2);

      TileComm lhsCommMemToComp{.name="lhs", .fromIdx=memIdx, .toIdxs={computeIdx}, .commCount=lhsCommMemToCompCount, .commElemSize=lhsSizeInCompTile,
                                .elemType=elemType, .srcCh=j+1, .dstCh=0, .srcWire=wireBundle, .dstWire=wireBundle};
      TileComm resCommCompToMem{.name="res", .fromIdx=computeIdx, .toIdxs={memIdx}, .commCount=resCommCompToMemCount, .commElemSize=resSizeInCompTile,
                                .elemType=elemType, .srcCh=0, .dstCh=j+2, .srcWire=wireBundle, .dstWire=wireBundle};

      result.tileComms.push_back(lhsCommMemToComp);
      result.tileComms.push_back(resCommCompToMem);

      rhsBcastCommMemToComp.toIdxs.push_back(computeIdx);
    }

    result.tileComms.push_back(rhsBcastCommMemToComp);
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
  for (auto &tile : result.aieTiles) {
    auto allocatePhysicalBuf = [&](AieCommBuf &commBuf, bool includeCommCount) {
      const auto &comm = result.tileComms[commBuf.tileCommIdx];
      uint32_t bufSize = includeCommCount ? comm.commElemSize * comm.commCount : comm.commElemSize;
      AieBuf buf{.name=comm.name, .bufSize=bufSize, .elemType=comm.elemType};
      tile.allocatedBufs.push_back(buf);

      size_t bufIdx = tile.allocatedBufs.size() - 1;
      commBuf.setPhysicalBufInfo(bufIdx, true, /*offset*/0, buf.bufSize, buf.elemType);
    };

    if (tile.row != 1) { // Shim/Compute tile
      // Allocate physical buffers for all communication buffers
      bool includeCommCount = tile.row == 0 ? true : false;
      for (auto &commBuf : tile.inCommBufs) {
        allocatePhysicalBuf(commBuf, includeCommCount);
      }
      for (auto &commBuf : tile.outCommBufs) {
        allocatePhysicalBuf(commBuf, includeCommCount);
      }      
    } else { // Mem tile
      // Allocate physical buffers only for communication with Shim tiles
      auto isShimTile = [&](size_t idx) {
        return result.aieTiles[idx].row == 0;
      };

      for (auto &commBuf : tile.inCommBufs) {
        const auto &comm = result.tileComms[commBuf.tileCommIdx];
        if (isShimTile(comm.fromIdx)) {
          allocatePhysicalBuf(commBuf, false);
        }
      }
      for (auto &commBuf : tile.outCommBufs) {
        const auto &comm = result.tileComms[commBuf.tileCommIdx];
        if (llvm::any_of(comm.toIdxs, isShimTile)) {
          allocatePhysicalBuf(commBuf, false);
        }
      }

      // Link communication buffers to physical buffers
      for (auto &commBuf : tile.inCommBufs) { // Link 'res' buffer (Compute -> Mem -> Shim)
        const auto &comm = result.tileComms[commBuf.tileCommIdx];
        if (!isShimTile(comm.fromIdx)) {
          auto linkCommBufIt = llvm::find_if (tile.outCommBufs, [&](AieCommBuf &outBuf) {
            const auto &outComm = result.tileComms[outBuf.tileCommIdx];
            return outBuf.hasOwnBuf && outComm.name == "res";
          });

          if (linkCommBufIt == tile.outCommBufs.end())
            llvm::report_fatal_error("'res' buffer not allocated in Mem Tile");

          const auto &linkBuf = tile.allocatedBufs[linkCommBufIt->bufIdx];
          uint32_t offset = (linkBuf.bufSize / numCompTile) * (result.aieTiles[comm.fromIdx].row - 2);
          uint32_t size = comm.commElemSize * comm.commCount;
          commBuf.setPhysicalBufInfo(linkCommBufIt->bufIdx, false, offset, size, comm.elemType);
        }
      }

      for (auto &commBuf : tile.outCommBufs) { // Link 'lhs/rhs' buffer (Shim -> Mem -> Compute)
        const auto &comm = result.tileComms[commBuf.tileCommIdx];
        if (!llvm::any_of(comm.toIdxs, isShimTile)) {
          std::string commName = comm.name;

          auto linkCommBufIt = llvm::find_if (tile.inCommBufs, [&](AieCommBuf &inBuf) {
            const auto &inComm = result.tileComms[inBuf.tileCommIdx];
            return inBuf.hasOwnBuf && inComm.name == commName;
          });

          if (linkCommBufIt == tile.inCommBufs.end())
            llvm::report_fatal_error("'lhs/rhs' buffer not allocated in Mem Tile");

          const auto &linkBuf = tile.allocatedBufs[linkCommBufIt->bufIdx];
          uint32_t offset = commName == "lhs" ? (linkBuf.bufSize / numCompTile) * (result.aieTiles[comm.toIdxs.front()].row - 2) : 0;
          uint32_t size = comm.commElemSize * comm.commCount;
          commBuf.setPhysicalBufInfo(linkCommBufIt->bufIdx, false, offset, size, comm.elemType);
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
      llvm::dbgs() << ", commCount=" << comm.commCount << ", commElemSize=" << comm.commElemSize << ", elemType=";
      comm.elemType.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  }

  return result;
}

void generateAieOps(ConversionPatternRewriter &rewriter,
                    AiePlacementResult &placement,
                    const TileParam &tileParam) {
  // Set variables
  bool doubleBufferingEnabled = false;
  uint32_t numCompTile = 4;
  auto numCols = tileParam.numLastSpm;
  auto [TM, TK, TN, elemType] = tileParam.coreTile;
  auto mCountInMemTile = tileParam.levelTiles[0].TM / TM;
  auto kCountInMemTile = tileParam.levelTiles[0].TK / TK;
  auto nCountInMemTile = tileParam.levelTiles[0].TN / TN;

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
  auto deviceOp = builder.create<DeviceOp>(loc, devices[numCols - 1]);

  // Ensure it has a body block, and point insertion into it
  deviceOp.getRegion().emplaceBlock();
  DeviceOp::ensureTerminator(deviceOp.getBodyRegion(), builder, loc);
  builder.setInsertionPointToStart(deviceOp.getBody());

  // Generate AIE TileOp
  for (auto &tile : placement.aieTiles) {
    auto tileOp = builder.create<xilinx::AIE::TileOp>(loc, tile.col, tile.row);

    tile.value = tileOp;
  }

  // Generate Ops for each tile
  for (auto &tile : placement.aieTiles) {
    
    if (tile.row == 0) { // Shim tile
      // Generate Memref GlobalOp
      for (auto &buf : tile.allocatedBufs) {
        buf.symbol = std::string("global_") + buf.name + "_" + std::to_string(tile.col) + "_" + std::to_string(tile.row);
        auto globalMemrefNameAttr = builder.getStringAttr(buf.symbol);
        auto globalMemrefType = MemRefType::get({buf.bufSize}, buf.elemType);
        builder.create<memref::GlobalOp>(loc, globalMemrefNameAttr, builder.getStringAttr("public"),
                                                         globalMemrefType, nullptr, false, nullptr);
      }
    } else { // Mem/Compute tile
      // Generate AIE BufferOp
      for (auto &buf : tile.allocatedBufs) {
        buf.symbol = std::string("buf_") + buf.name + "_" + std::to_string(tile.col) + "_" + std::to_string(tile.row);
        auto bufNameAttr = builder.getStringAttr(buf.symbol);
        auto bufMemType = MemRefType::get({buf.bufSize}, buf.elemType);
        auto bufOp = builder.create<xilinx::AIE::BufferOp>(loc, 
                        /*memref*/bufMemType, /*tile*/tile.value, /*sym_name*/bufNameAttr, 
                        /*address*/nullptr, /*initial_value*/nullptr, /*mem_bank*/nullptr);
        
        buf.bufValue = bufOp;
      }

      // Generate AIE LockOp
      uint32_t id = 0;
      for (auto &buf : tile.allocatedBufs) {
        uint32_t numProdToken = doubleBufferingEnabled ? 2 : 1;
        uint32_t numConsToken = 0;

        if (tile.row == 1) { // Mem tile
          if (buf.name == "lhs") {
            numProdToken = nCountInMemTile * numCompTile;
          } else if (buf.name == "rhs") {
            numProdToken = mCountInMemTile / numCompTile;
          } else { // buf.name == "res"
            numProdToken = numCompTile;
          }
        }

        { // Producer lock
          auto idAttr = builder.getI32IntegerAttr(id++);
          auto initAttr = builder.getI32IntegerAttr(numProdToken);
          auto nameAttr = builder.getStringAttr("buf_" + buf.name + "_" + std::to_string(tile.col) + "_" + 
                                                std::to_string(tile.row) + "_prod_lock");
          auto lockOp = builder.create<xilinx::AIE::LockOp>(loc, tile.value, idAttr, initAttr, nameAttr);
          buf.prodLockValue = lockOp;
        }

        { // Consumer lock
          auto idAttr = builder.getI32IntegerAttr(id++);
          auto initAttr = builder.getI32IntegerAttr(numConsToken);
          auto nameAttr = builder.getStringAttr("buf_" + buf.name + "_" + std::to_string(tile.col) + "_" + 
                                                std::to_string(tile.row) + "_cons_lock");
          auto lockOp = builder.create<xilinx::AIE::LockOp>(loc, tile.value, idAttr, initAttr, nameAttr);
          buf.consLockValue = lockOp;
        }
      }
    }
  }

  // Generate AIE FlowOp
  for (auto &comm : placement.tileComms) {
    auto &srcTile = placement.aieTiles[comm.fromIdx];

    for (auto &toIdx : comm.toIdxs) {
      auto &dstTile = placement.aieTiles[toIdx];
      auto srcCh = builder.getI32IntegerAttr(comm.srcCh);
      auto dstCh = builder.getI32IntegerAttr(comm.dstCh);
      auto srcWBAttr = xilinx::AIE::WireBundleAttr::get(builder.getContext(), comm.srcWire);
      auto dstWBAttr = xilinx::AIE::WireBundleAttr::get(builder.getContext(), comm.dstWire);

      builder.create<xilinx::AIE::FlowOp>(loc, srcTile.value, srcWBAttr, srcCh,
                                          dstTile.value, dstWBAttr, dstCh);
    }
  }

  // Generate AIE DMAOp
  for (auto &tile : placement.aieTiles) {

    Operation *dmaOp = nullptr;
    if (tile.row == 0) {
      size_t numBlocks = tile.inCommBufs.size() + tile.outCommBufs.size();

      for (size_t i = 0; i < numBlocks; ++i){
        bool isInput = (i < tile.inCommBufs.size());
        auto &commBuf = isInput ? tile.inCommBufs[i] : tile.outCommBufs[i - tile.inCommBufs.size()];
        auto &comm = placement.tileComms[commBuf.tileCommIdx];
        auto &buf = tile.allocatedBufs[commBuf.bufIdx];

        auto globalSym = SymbolRefAttr::get(builder.getContext(), buf.symbol);
        DMAChannelDir dmaDir = isInput ? DMAChannelDir::S2MM : DMAChannelDir::MM2S;
        DMAChannelDirAttr dmaDirAttr = DMAChannelDirAttr::get(builder.getContext(), dmaDir);
        auto &channelIdx = isInput ? comm.dstCh : comm.srcCh;
  
        builder.create<ShimDMAAllocationOp>(loc, globalSym, dmaDirAttr,
                                            builder.getI64IntegerAttr(channelIdx),
                                            builder.getI64IntegerAttr(tile.col));
      }

      continue;
    } else if (tile.row == 1) {
      dmaOp = builder.create<MemTileDMAOp>(loc, tile.value).getOperation();
    } else {
      dmaOp = builder.create<MemOp>(loc, tile.value).getOperation();
    }

    {
      OpBuilder::InsertionGuard g(builder);
      Region &DMARegion = dmaOp->getRegion(0);

      std::vector<Block*> dmaBlocks;
      std::vector<Block*> bdBlocks;

      size_t numBlocks = tile.inCommBufs.size() + tile.outCommBufs.size();
      for (size_t i = 0; i < numBlocks; ++i){
        Block *dmaBlock = builder.createBlock(&DMARegion);
        dmaBlocks.push_back(dmaBlock);
      }
      for (size_t i = 0; i < numBlocks; ++i){
        Block *bdBlock = builder.createBlock(&DMARegion);
        bdBlocks.push_back(bdBlock);
      }
      Block *endBlock = builder.createBlock(&DMARegion);
      dmaBlocks.push_back(endBlock);

      for (size_t i = 0; i < numBlocks; ++i){
        bool isInput = (i < tile.inCommBufs.size());
        auto &commBuf = isInput ? tile.inCommBufs[i] : tile.outCommBufs[i - tile.inCommBufs.size()];
        auto &comm = placement.tileComms[commBuf.tileCommIdx];

        {
          OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(dmaBlocks[i]);

          DMAChannelDir dmaDir = isInput ? DMAChannelDir::S2MM : DMAChannelDir::MM2S;
          DMAChannelDirAttr dmaDirAttr = DMAChannelDirAttr::get(builder.getContext(), dmaDir);
          auto &channelIdx = isInput ? comm.dstCh : comm.srcCh;
          auto channelIdxAttr = builder.getI32IntegerAttr(channelIdx);
          auto repeatCntAttr = builder.getI32IntegerAttr(0);

          builder.create<DMAStartOp>(loc, dmaDirAttr, channelIdxAttr, repeatCntAttr, bdBlocks[i], dmaBlocks[i+1]);
        }

        {
          OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(bdBlocks[i]);

          auto &buf = tile.allocatedBufs[commBuf.bufIdx];
          auto acquireLockValue = isInput ? buf.prodLockValue : buf.consLockValue;
          auto releaseLockValue = isInput ? buf.consLockValue : buf.prodLockValue;
          uint32_t numToken = 1;
          if (tile.row == 1) {
            if (isInput && buf.name == "lhs") {
              numToken = nCountInMemTile * numCompTile;
            } else if (isInput && buf.name == "rhs") {
              numToken = mCountInMemTile / numCompTile;
            } else if (!isInput && buf.name == "res") {
              numToken = numCompTile;
            }
          } 

          builder.create<UseLockOp>(loc, acquireLockValue, LockAction::AcquireGreaterEqual, numToken);
          builder.create<DMABDOp>(loc, buf.bufValue, commBuf.bufOffset, commBuf.bufSize);
          builder.create<UseLockOp>(loc, releaseLockValue, LockAction::Release, numToken);
          builder.create<NextBDOp>(loc, bdBlocks[i]);
        }
      }

      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(endBlock);
        builder.create<EndOp>(loc);
      }
    }
  }

  // Generate Func FuncOp
  auto funcNameAttr = builder.getStringAttr("extern_kernel");
  auto lhsMemrefType = MemRefType::get({TM * TK}, elemType);
  auto rhsMemrefType = MemRefType::get({TK * TN}, elemType);
  auto resMemrefType = MemRefType::get({TM * TN}, elemType);
  auto i32Type = builder.getI32Type();
  FunctionType funcType = builder.getFunctionType({lhsMemrefType, rhsMemrefType, resMemrefType, i32Type, i32Type, i32Type}, {});
  auto funcOp = builder.create<func::FuncOp>(loc, funcNameAttr, funcType);
  funcOp.setPrivate();

  // Generate AIE CoreOp
  for (auto &tile : placement.aieTiles) {
    if (tile.row < 2) {
      continue;
    }

    // Set buffer pointers (lhs/rhs/res)
    AieBuf *lhsBufPtr = nullptr;
    AieBuf *rhsBufPtr = nullptr;
    AieBuf *resBufPtr = nullptr;

    for (auto &buf : tile.allocatedBufs) {
      const std::string &name = buf.name;
      if (name == "lhs") {
        lhsBufPtr = &buf;
      } else if (name == "rhs") {
        rhsBufPtr = &buf;
      } else if (name == "res") {
        resBufPtr = &buf;
      }
    }

    auto coreOp = builder.create<xilinx::AIE::CoreOp>(loc, tile.value);
    coreOp->setAttr("link_with", builder.getStringAttr("kernel.o"));
    {
      OpBuilder::InsertionGuard g(builder);
      Region &coreRegion = coreOp.getBody();
      Block *coreBlock = builder.createBlock(&coreRegion);
      builder.setInsertionPointToStart(coreBlock);

      // Generate Arith ConstantOp
      auto c0 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
      auto c1 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));
      auto cMax = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0xFFFFFFFFULL));
      auto cLen = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(TM*TN));
      auto cN = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(kCountInMemTile));
      auto c0f = builder.create<mlir::arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0f));
      
      auto cRow = builder.create<mlir::arith::ConstantIntOp>(loc, TM, /*width=*/32);
      auto cCol = builder.create<mlir::arith::ConstantIntOp>(loc, TN, /*width=*/32);
      auto cDep = builder.create<mlir::arith::ConstantIntOp>(loc, TK, /*width=*/32);

      // Generate SCF ForOp (infinite loop)
      auto infiniteLoopOp = builder.create<mlir::scf::ForOp>(loc, c0, cMax, c1);
      {
        OpBuilder::InsertionGuard g(builder);
        Region &infiniteLoopRegion = infiniteLoopOp.getRegion();
        builder.setInsertionPointToStart(&infiniteLoopRegion.back());

        // Generate AIE UseLockOp (res)
        builder.create<UseLockOp>(loc, resBufPtr->prodLockValue, LockAction::AcquireGreaterEqual, 1);

        // Generate SCF ForOp (init loop: res)
        auto initLoopOp = builder.create<mlir::scf::ForOp>(loc, c0, cLen, c1);
        {
          OpBuilder::InsertionGuard g(builder);
          Region &initLoopRegion = initLoopOp.getRegion();
          builder.setInsertionPointToStart(&initLoopRegion.back());

          Value index = initLoopOp.getInductionVar();
          builder.create<mlir::memref::StoreOp>(loc, c0f, resBufPtr->bufValue, ValueRange(index));
        }
  
        // Generate SCF ForOp (calc loop: lhs/rhs)
        auto calcLoopOp = builder.create<mlir::scf::ForOp>(loc, c0, cN, c1);
        {
          OpBuilder::InsertionGuard g(builder);
          Region &calcLoopRegion = calcLoopOp.getRegion();
          builder.setInsertionPointToStart(&calcLoopRegion.back());

          // Generate AIE UseLockOp (lhs/rhs)
          builder.create<UseLockOp>(loc, lhsBufPtr->consLockValue, LockAction::AcquireGreaterEqual, 1);
          builder.create<UseLockOp>(loc, rhsBufPtr->consLockValue, LockAction::AcquireGreaterEqual, 1);

          // Generate Func CallOp
          auto calleeAttr = SymbolRefAttr::get(builder.getContext(), "extern_kernel");
          builder.create<mlir::func::CallOp>(loc, calleeAttr, TypeRange{}, 
                                            ValueRange{lhsBufPtr->bufValue, rhsBufPtr->bufValue, resBufPtr->bufValue,
                                                       cRow, cCol, cDep});
    
          // Generate AIE UseLockOp (lhs/rhs)
          builder.create<UseLockOp>(loc, lhsBufPtr->prodLockValue, LockAction::Release, 1);
          builder.create<UseLockOp>(loc, rhsBufPtr->prodLockValue, LockAction::Release, 1);
        }

        // Generate AIE UseLockOp (res)
        builder.create<UseLockOp>(loc, resBufPtr->consLockValue, LockAction::Release, 1);
      }

      // Generate AIE EndOp
      builder.create<EndOp>(loc);
    }
  }  

  // Generate AIEX RuntimeSequenceOp
  std::string seq_name = "sequence";
  StringAttr seq_sym_name = builder.getStringAttr(seq_name);
  auto seqOp = builder.create<xilinx::AIEX::RuntimeSequenceOp>(loc, seq_sym_name);
  {
    OpBuilder::InsertionGuard g(builder);
    Region &seqRegion = seqOp.getBody();
    Block *seqBlock = builder.createBlock(&seqRegion);
    builder.setInsertionPointToStart(seqBlock);

    auto lhsMemrefType = MemRefType::get({TM * TK * mCountInMemTile * kCountInMemTile}, elemType);
    auto rhsMemrefType = MemRefType::get({TK * TN * kCountInMemTile * nCountInMemTile}, elemType);
    auto resMemrefType = MemRefType::get({TM * TN * mCountInMemTile * nCountInMemTile}, elemType);

    auto arg_lhs = seqBlock->addArgument(lhsMemrefType, loc);
    auto arg_rhs = seqBlock->addArgument(rhsMemrefType, loc);
    auto arg_res = seqBlock->addArgument(resMemrefType, loc);

    uint32_t mCountPerCompTile = mCountInMemTile / numCompTile;
    const std::vector<std::vector<int64_t>> Offsets = { {0, 0, 0, 0},   // lhs
                                                        {0, 0, 0, 0},   // rhs
                                                        {0, 0, 0, 0} }; // res
    const std::vector<std::vector<int64_t>> Sizes = { {mCountInMemTile, kCountInMemTile, TM, TK},                   // lhs
                                                      {nCountInMemTile, kCountInMemTile, TN, TK},                   // rhs
                                                      {numCompTile, nCountInMemTile, TM * mCountPerCompTile, TN} }; // res
    const std::vector<std::vector<int64_t>> Strides = { {TM * TK * kCountInMemTile, TK, TK * kCountInMemTile, 1},                       // lhs
                                                        {TN * TK * kCountInMemTile, TK, TK * kCountInMemTile, 1},                       // rhs
                                                        {TM * TN * mCountPerCompTile * nCountInMemTile, TN, TN * nCountInMemTile, 1} }; // res
                                       
    // Generate AIEX NpuDmaMemcpyNdOp
    uint32_t id = 0;
    for (auto &tile : placement.aieTiles) {
      if (tile.row == 0) { // Shim tile
        for (auto &commBuf : tile.outCommBufs) {
          auto &buf = tile.allocatedBufs[commBuf.bufIdx];
          auto &arg = buf.name == "lhs" ? arg_lhs : arg_rhs;
          size_t idx = buf.name == "lhs" ? 0 : 1;
          StringRef metadata = builder.getStringAttr(buf.symbol);
          builder.create<xilinx::AIEX::NpuDmaMemcpyNdOp>(loc, arg, SmallVector<Value>{}, SmallVector<Value>{}, SmallVector<Value>{},
                                                        ArrayRef(Offsets[idx]), ArrayRef(Sizes[idx]), ArrayRef(Strides[idx]), nullptr, 
                                                        metadata, id++, false, 0, 0, 0, 0, 0, 0);
        }      

        for (auto &commBuf : tile.inCommBufs) {
          auto &buf = tile.allocatedBufs[commBuf.bufIdx];
          auto &arg = arg_res;
          size_t idx = 2;
          StringRef metadata = builder.getStringAttr(buf.symbol);
          builder.create<xilinx::AIEX::NpuDmaMemcpyNdOp>(loc, arg, SmallVector<Value>{}, SmallVector<Value>{}, SmallVector<Value>{},
                                                        ArrayRef(Offsets[idx]), ArrayRef(Sizes[idx]), ArrayRef(Strides[idx]), nullptr, 
                                                        metadata, id++, false, 0, 0, 0, 0, 0, 0);

          builder.create<xilinx::AIEX::NpuDmaWaitOp>(loc, metadata);
        }
      }
    }
  }

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
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
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
