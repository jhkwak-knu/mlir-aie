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
#include <bitset>

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
    DebugTileParam("debug-tile-param",
                    llvm::cl::desc("Print loaded TileParam"),
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
    llvm::dbgs() << "\n";
  }

  return info;
}

//===----------------------------------------------------------------------===//
// Tiling algorithm
//===----------------------------------------------------------------------===//
struct MatmulOpInfo {
  uint32_t M, K, N;
  Type elemType;
};

struct TileSize {
  uint32_t TM, TK, TN;
};

struct LevelParam {
  uint32_t numSpm;
  uint32_t SPm, SPn;
  uint32_t TPm, TPk, TPn;
  TileSize tileSize;
  std::vector<uint32_t> tpOrder;
};

struct TileParam {
  TileSize opSize;
  Type elemType;
  uint32_t numLevel;
  uint32_t numLastSpm;
  bool doubleBufferEnabled;
  std::vector<LevelParam> levels;
};

// ---- Helper getters on a given JSON object ----
static uint32_t getU32Req(const llvm::json::Object &obj, llvm::StringRef key) {
  if (auto v = obj.getInteger(key)) {
    if (*v < 0 || *v > static_cast<int64_t>(UINT32_MAX)) {
      llvm::errs() << "Error: overflow in '" << key << "' (" << *v << ")\n";
      llvm::report_fatal_error("u32 overflow");
    }
    return static_cast<uint32_t>(*v);
  }
  llvm::errs() << "Error: missing integer field '" << key << "'\n";
  llvm::report_fatal_error("missing u32");
}

static bool getBoolReq(const llvm::json::Object &obj, llvm::StringRef key) {
  if (auto v = obj.getBoolean(key)) return *v;
  if (obj.get(key) != nullptr) {
    llvm::errs() << "Error: boolean expected in '" << key << "'\n";
    llvm::report_fatal_error("type mismatch");
  }
  llvm::errs() << "Error: missing boolean field '" << key << "'\n";
  llvm::report_fatal_error("missing bool");
}

static mlir::Type getElemTypeReq(const llvm::json::Object &obj,
                                 llvm::StringRef key,
                                 mlir::MLIRContext &ctx) {
  if (auto s = obj.getString(key)) {
    llvm::StringRef v = *s;
    v = v.trim();

    // floats
    if (v.equals_insensitive("f16"))  return mlir::Float16Type::get(&ctx);
    if (v.equals_insensitive("bf16")) return mlir::BFloat16Type::get(&ctx);
    if (v.equals_insensitive("f32"))  return mlir::Float32Type::get(&ctx);
    if (v.equals_insensitive("f64"))  return mlir::Float64Type::get(&ctx);

    // signed ints
    if (v.equals_insensitive("i8"))   return mlir::IntegerType::get(&ctx, 8,  mlir::IntegerType::Signed);
    if (v.equals_insensitive("i16"))  return mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Signed);
    if (v.equals_insensitive("i32"))  return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
    if (v.equals_insensitive("i64"))  return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);

    // unsigned ints
    if (v.equals_insensitive("ui8"))  return mlir::IntegerType::get(&ctx, 8,  mlir::IntegerType::Unsigned);
    if (v.equals_insensitive("ui16")) return mlir::IntegerType::get(&ctx, 16, mlir::IntegerType::Unsigned);
    if (v.equals_insensitive("ui32")) return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned);
    if (v.equals_insensitive("ui64")) return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Unsigned);

    llvm::errs() << "Error: unsupported elemType string '" << v << "' for key '" << key << "'\n";
    llvm::report_fatal_error("elemType unsupported");
  }

  if (obj.get(key) != nullptr) {
    llvm::errs() << "Error: string expected in '" << key << "'\n";
    llvm::report_fatal_error("type mismatch");
  }
  llvm::errs() << "Error: missing string field '" << key << "'\n";
  llvm::report_fatal_error("missing elemType");
}

static std::vector<uint32_t>
getAxisArrayOpt(const llvm::json::Object &obj, llvm::StringRef key) {
  std::vector<uint32_t> out;
  if (auto *arr = obj.getArray(key)) {
    std::bitset<3> seen;
    for (const auto &it : *arr) {
      if (auto iv = it.getAsInteger()) {
        if (*iv < 0 || *iv > 2) {
          llvm::errs() << "Error: '" << key
                       << "' value out of range [0,2]: " << *iv << "\n";
          llvm::report_fatal_error("tpOrder range error");
        }
        uint32_t v = static_cast<uint32_t>(*iv);
        if (seen.test(v)) {
          llvm::errs() << "Error: duplicate axis in '" << key
                       << "': " << v << "\n";
          llvm::report_fatal_error("tpOrder duplicate");
        }
        seen.set(v);
        out.push_back(v);
      } else {
        llvm::errs() << "Error: non-integer in array '" << key << "'\n";
        llvm::report_fatal_error("tpOrder type mismatch");
      }
    }
  }
  return out;
}

static TileSize getLevelTileSize(const llvm::json::Object &obj) {
  if (!obj.get("TM") || !obj.get("TK") || !obj.get("TN")) {
    llvm::errs() << "Error: level must define TM/TK/TN (unified schema)\n";
    llvm::report_fatal_error("missing tile triple");
  }
  return TileSize{
    .TM = getU32Req(obj, "TM"),
    .TK = getU32Req(obj, "TK"),
    .TN = getU32Req(obj, "TN"),
  };
}

// TODO: Implement the tiling algorithm
TileParam findOptimalTileParam(const SystemInfo &sysInfo, const MatmulOpInfo &opInfo) {

  std::string filePath = "/home/ace/ryzen_ai/mlir-aie-dev/mlir-aie/test/onnx-mlir/out/tc.json";

  auto bufOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (!bufOrErr) {
    llvm::errs() << "Error: cannot open JSON '" << filePath
                 << "' (" << bufOrErr.getError().message() << ")\n";
    llvm::report_fatal_error("tile param json open failed");
  }
  llvm::StringRef jsonText = (*bufOrErr)->getBuffer();

  auto jsonOrErr = llvm::json::parse(jsonText);
  if (!jsonOrErr) {
    llvm::errs() << "Error: JSON parse failed in '" << filePath << "'\n";
    llvm::report_fatal_error("tile param json parse failed");
  }
  auto *rootObj = jsonOrErr->getAsObject();
  if (!rootObj) {
    llvm::errs() << "Error: root JSON is not an object\n";
    llvm::report_fatal_error("tile param json root type error");
  }

  // Top-level fields
  TileParam tp;
  tp.opSize = TileSize{
      .TM = getU32Req(*rootObj, "M"),
      .TK = getU32Req(*rootObj, "K"),
      .TN = getU32Req(*rootObj, "N"),
  };
  tp.elemType            = getElemTypeReq(*rootObj, "elemType", *opInfo.elemType.getContext());
  tp.numLevel            = getU32Req(*rootObj, "numLevel");
  tp.numLastSpm          = getU32Req(*rootObj, "numLastSpm");
  tp.doubleBufferEnabled = getBoolReq(*rootObj, "doubleBuffer");

  auto *levelsArr = rootObj->getArray("levels");
  if (!levelsArr) {
    llvm::errs() << "Error: 'levels' array missing in case\n";
    llvm::report_fatal_error("levels missing");
  }

  tp.levels.reserve(levelsArr->size());
  for (size_t i = 0; i < levelsArr->size(); ++i) {
    auto *Lobj = (*levelsArr)[i].getAsObject();
    if (!Lobj) {
      llvm::errs() << "Error: level[" << i << "] is not an object\n";
      llvm::report_fatal_error("level type error");
    }

    LevelParam lv;
    lv.numSpm = getU32Req(*Lobj, "numSpm");
    lv.SPm    = getU32Req(*Lobj, "SPm");
    lv.SPn    = getU32Req(*Lobj, "SPn");
    lv.TPm    = getU32Req(*Lobj, "TPm");
    lv.TPk    = getU32Req(*Lobj, "TPk");
    lv.TPn    = getU32Req(*Lobj, "TPn");

    lv.tileSize = getLevelTileSize(*Lobj);
    lv.tpOrder  = getAxisArrayOpt(*Lobj, "tpOrder");

    tp.levels.push_back(std::move(lv));
  }

  if (tp.numLevel != tp.levels.size()) {
    llvm::errs() << "Error: numLevel(" << tp.numLevel
                 << ") != levels.size(" << tp.levels.size() << ")\n";
    llvm::report_fatal_error("numLevel mismatch");
  }

  if (DebugTileParam) {
    llvm::dbgs() << "[TileParam] Loaded TileParam:\n"
                << "  opSize:       "
                << "M=" << tp.opSize.TM
                << " K=" << tp.opSize.TK
                << " N=" << tp.opSize.TN << "\n";

    llvm::dbgs() << "  elemType:     ";
    tp.elemType.print(llvm::dbgs());
    llvm::dbgs() << "\n";

    llvm::dbgs() << "  numLevel:     " << tp.numLevel << "\n"
                << "  numLastSpm:   " << tp.numLastSpm << "\n"
                << "  doubleBuffer: " << (tp.doubleBufferEnabled ? "true" : "false") << "\n";

    for (size_t i = 0; i < tp.levels.size(); ++i) {
      const auto &lv = tp.levels[i];
      llvm::dbgs() << "    level=" << i << ":"
                  << " numSpm=" << lv.numSpm
                  << " SPm=" << lv.SPm
                  << " SPn=" << lv.SPn << " |"
                  << " TPm=" << lv.TPm
                  << " TPk=" << lv.TPk
                  << " TPn=" << lv.TPn << " |"
                  << " tile(M/K/N)="
                  << lv.tileSize.TM << "/"
                  << lv.tileSize.TK << "/"
                  << lv.tileSize.TN << " |"
                  << " tpOrder=";

      if (lv.tpOrder.empty()) {
        llvm::dbgs() << "[]\n";
      } else {
        llvm::dbgs() << "[";
        for (size_t j = 0; j < lv.tpOrder.size(); ++j) {
          llvm::dbgs() << lv.tpOrder[j];
          if (j + 1 < lv.tpOrder.size()) llvm::dbgs() << ",";
        }
        llvm::dbgs() << "]\n";
      }
    }
    llvm::dbgs() << "\n";
  }

  return tp;
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
  Value calcLockValue;
};

struct AieBufferDescriptor {
  std::string name;
  bool isPacket;
  uint32_t packetId;
  uint32_t bufIdx;
  uint32_t bufOffset;
  uint32_t bufSize;
  uint32_t nextBdIdx;

  void setBufInfo(uint32_t idx, uint32_t offset, uint32_t size) {
    bufIdx = idx;
    bufOffset = offset;
    bufSize = size;
  }
};

struct AieDma {
  DMAChannelDir dir;
  uint32_t channel;
  uint32_t bdIdx;
};

struct AieTile {
  uint32_t col;
  uint32_t row;
  Value value;

  std::vector<AieBuf> bufs;
  std::vector<AieBufferDescriptor> bds;
  std::vector<AieDma> dmas;

  bool operator==(const AieTile &o) const {
    return col == o.col && row == o.row;
  }

  uint32_t findBufIdx(std::string name) const {
    const size_t pos = name.find_first_of("0123456789");
    const std::string base = name.substr(0, pos);
    
    const char* targets[2];
    size_t tcount = 0;
    if (base == "pres") { targets[0] = "pres"; targets[1] = "res"; tcount = 2; }
    else                { targets[0] = base.c_str();               tcount = 1; }

    for (size_t t = 0; t < tcount; ++t)
      for (uint32_t idx = 0; idx < bufs.size(); ++idx)
        if (bufs[idx].name == targets[t])
          return idx;

    llvm_unreachable("Buffer not found");
  };

  std::pair<bool, uint32_t> findDmaIdx(DMAChannelDir dir, uint32_t channel) {
    for (uint32_t idx = 0; idx < dmas.size(); ++idx) {
      const auto &d = dmas[idx];
      if (d.dir == dir && d.channel == channel)
        return {true, idx};
    }
    return {false, 0};
  }
};

struct AieCircuit {
  std::string name;
  uint32_t size;
  Type elemType;
  std::vector<uint32_t> dstIdxs;
  std::vector<WireBundle> dstBundles;
  std::vector<uint32_t> dstChs;
};

struct AiePacket {
  std::string name;
  uint32_t packetId;
  uint32_t size;
  Type elemType;
  std::vector<uint32_t> dstIdxs;
  std::vector<WireBundle> dstBundles;
  std::vector<uint32_t> dstChs;
};

struct AieComm {
  std::string name;
  uint32_t srcIdx;
  WireBundle srcBundle;
  uint32_t srcCh;
  bool isPacket;
  std::vector<AiePacket> packets;
  std::vector<AieCircuit> circuits;
};

struct AieNpuMemcpyNd {
  std::string name;
  uint32_t id;
  uint32_t bufIdx;
  bool isPacket;
  uint32_t packetType;
  uint32_t packetId;
  bool issueToken;
  bool doWait;
  std::array<int64_t,4> staticOffset;
  std::array<int64_t,4> staticSize;
  std::array<int64_t,4> staticStride;
};

struct AiePlacement {
  std::vector<AieTile> aieTiles;
  std::vector<AieComm> aieComms;
  std::vector<AieNpuMemcpyNd> aieSchedule;

  AieTile& getAieTile(uint32_t idx) { return aieTiles[idx]; }
  AieComm& getAieComm(uint32_t idx) { return aieComms[idx]; }
  uint32_t findAieTileIdx(uint32_t col, uint32_t row) const {
    for (uint32_t idx = 0; idx < aieTiles.size(); ++idx) {
      const auto &tile = aieTiles[idx];
      if (tile.col == col && tile.row == row)
        return idx;
    }
    llvm_unreachable("Tile not found");
  };
};

AiePlacement
optimizeAiePlacement(const TileParam &tileParam) {

  AiePlacement placement;

  // Set variables
  uint32_t numCompTilesPerCol = 4;
  uint32_t numCols = tileParam.numLastSpm / numCompTilesPerCol;

  auto &compTileLevel = tileParam.levels[0];
  auto [compTM, compTK, compTN] = compTileLevel.tileSize;
  uint32_t compTileSPm = compTileLevel.SPm;
  uint32_t compTileSPn = compTileLevel.SPn;
  uint32_t compTileTPm = compTileLevel.TPm;
  uint32_t compTileTPk = compTileLevel.TPk;
  uint32_t compTileTPn = compTileLevel.TPn;
  auto tpOrder = compTileLevel.tpOrder;

  auto elemType = tileParam.elemType;
  // bool doubleBufferEnabled = tileParam.doubleBufferEnabled;
  auto dmaWireBundle = WireBundle::DMA;

  std::array<uint32_t,8> packetIdList = {0, 2, 6, 14, 1, 3, 7, 15};
  
  // Place on physical AIE tiles
  for (uint32_t i = 0; i < numCols; ++i) {
    AieTile shimTile{.col=i, .row=0};
    placement.aieTiles.push_back(shimTile);

    for (uint32_t j = 0; j < numCompTilesPerCol; ++j) {
      AieTile compTile{.col=i, .row=(5-j)};
      placement.aieTiles.push_back(compTile);
    }
  }

  // Allocate physical buffers for each tile
  for (auto &tile : placement.aieTiles) {
    if (tile.row == 0) { // Shim tile
      uint32_t lhsBufSize = compTM * compTK;
      uint32_t rhsBufSize = compTK * compTN;
      uint32_t resBufSize = compTM * compTN + 1;  // packet header (4B)

      AieBuf lhsBuf{.name="lhs", .bufSize=lhsBufSize, .elemType=elemType};
      AieBuf rhsBuf{.name="rhs", .bufSize=rhsBufSize, .elemType=elemType};
      AieBuf resBuf{.name="res", .bufSize=resBufSize, .elemType=elemType};

      tile.bufs.push_back(lhsBuf);
      tile.bufs.push_back(rhsBuf);
      tile.bufs.push_back(resBuf);

      if ((compTileTPk > 1) && (tpOrder[0] != 2)) {
        uint32_t presBufSize = compTM * compTN;
        AieBuf presBuf{.name="pres", .bufSize=presBufSize, .elemType=elemType};
        tile.bufs.push_back(presBuf);
      }
    } else { // Comp tile
      uint32_t lhsBufSize = compTM * compTK;
      uint32_t rhsBufSize = compTK * compTN;
      uint32_t resBufSize = compTM * compTN;

      AieBuf lhsBuf{.name="lhs", .bufSize=lhsBufSize, .elemType=elemType};
      AieBuf rhsBuf{.name="rhs", .bufSize=rhsBufSize, .elemType=elemType};
      AieBuf resBuf{.name="res", .bufSize=resBufSize, .elemType=elemType};

      tile.bufs.push_back(lhsBuf);
      tile.bufs.push_back(rhsBuf);
      tile.bufs.push_back(resBuf);
    }
  }

  // Configure AIE communication paths
  // 1. Shim tile -> Comp tile (input)
  for (uint32_t col = 0; col < numCols; ++col) {
    uint32_t shimIdx = placement.findAieTileIdx(col, 0);
    AieComm inputComm0{.name="input0", .srcIdx=shimIdx, .srcBundle=dmaWireBundle, .srcCh=0, .isPacket=true};
    AieComm inputComm1{.name="input1", .srcIdx=shimIdx, .srcBundle=dmaWireBundle, .srcCh=1, .isPacket=true};
    uint32_t inputPacketIdIdx0 = 0;
    uint32_t inputPacketIdIdx1 = 0;

    // LHS packets (input0)
    uint32_t numLhsPacket = std::max<uint32_t>(compTileSPm / numCols, 1u);
    uint32_t numLhsTarget = numCompTilesPerCol / numLhsPacket;
    for (uint32_t i = 0; i < numLhsPacket; ++i) {
      AiePacket packet;
      packet.name = std::string("lhs") + std::to_string(i*numCols + col);
      packet.packetId = packetIdList[inputPacketIdIdx0++];
      packet.size = compTM * compTK;
      packet.elemType = elemType;

      for (uint32_t j = 0; j < numLhsTarget; ++j) {
        uint32_t row = 5 - ((j * numLhsPacket) + i);
        uint32_t compIdx = placement.findAieTileIdx(col, row);
        packet.dstIdxs.push_back(compIdx);
        packet.dstBundles.push_back(dmaWireBundle);
        packet.dstChs.push_back(0);
      }

      inputComm0.packets.push_back(packet);
    }

    // RHS packets (input1)
    uint32_t numRhsPacket = std::max<uint32_t>(compTileSPn / numCols, 1u);
    uint32_t numRhsTarget = numCompTilesPerCol / numRhsPacket;
    for (uint32_t i = 0; i < numRhsPacket; ++i) {
      uint32_t a = std::max<uint32_t>(numCols / numCompTilesPerCol, 1u);
      uint32_t b = col / numCompTilesPerCol;

      AiePacket packet;
      packet.name = std::string("rhs") + std::to_string((numCompTilesPerCol*((a*col+b)%numCols))/compTileSPm + i);
      packet.packetId = packetIdList[inputPacketIdIdx1++];
      packet.size = compTK * compTN;
      packet.elemType = elemType;

      for (uint32_t j = 0; j < numRhsTarget; ++j) {
        uint32_t colOffset = numCompTilesPerCol * (col / numCompTilesPerCol);
        uint32_t localCol = col % numCompTilesPerCol;
        uint32_t localIdx = (numCompTilesPerCol * localCol) + (i * numRhsTarget) + j;
        uint32_t localnumCols = ((numCols - 1) % numCompTilesPerCol) + 1;
        uint32_t compCol = colOffset + (localIdx % localnumCols);
        uint32_t compRow = 5 - (localIdx / localnumCols);
        uint32_t compIdx = placement.findAieTileIdx(compCol, compRow);
        packet.dstIdxs.push_back(compIdx);
        packet.dstBundles.push_back(dmaWireBundle);
        packet.dstChs.push_back(1);
      }

      inputComm1.packets.push_back(packet);
    }

    // RES packets (input0 or input1 only when a partial sum needs to be transferred)
    if (compTileTPk > 1) {
      AieComm *inputCommPtr = nullptr;
      uint32_t *inputPacketIdPtr = nullptr;
      uint32_t dstCh = 0;

      if (tpOrder[0] == 0) { // M-axis
        inputCommPtr = &inputComm0;
        inputPacketIdPtr = &inputPacketIdIdx0;
        dstCh = 0;
      } else if (tpOrder[0] == 1) { // N-axis
        inputCommPtr = &inputComm1;
        inputPacketIdPtr = &inputPacketIdIdx1;
        dstCh = 1;
      } else { // K-axis
        // No need to transfer the partial sum
      }
      
      if (inputCommPtr) {
        auto &inputComm = *inputCommPtr;
        auto &inputPacketId = *inputPacketIdPtr;
        
        std::vector<AiePacket> newPackets;
        for (auto &packet : inputComm.packets) {
          for (auto dstIdx : packet.dstIdxs) {
            auto &dstTile = placement.aieTiles[dstIdx];
            uint32_t count = (5 - dstTile.row)*numCols + dstTile.col;

            AiePacket newPacket;
            newPacket.name = std::string("pres") + std::to_string(count);
            newPacket.packetId = packetIdList[inputPacketId++];
            newPacket.size = compTM * compTN;
            newPacket.elemType = elemType;
            newPacket.dstIdxs.push_back(dstIdx);
            newPacket.dstBundles.push_back(dmaWireBundle);
            newPacket.dstChs.push_back(dstCh);

            newPackets.push_back(newPacket);
          }
        }

        inputComm.packets.insert(inputComm.packets.begin(), newPackets.begin(), newPackets.end());
      }
    }

    placement.aieComms.push_back(inputComm0);
    placement.aieComms.push_back(inputComm1);
  }

  // 2. Shim tile <- Comp tile (output)
  for (uint32_t col = 0; col < numCols; ++col) {
    uint32_t shimIdx = placement.findAieTileIdx(col, 0);
    uint32_t outputPacketIdIdx = 0;

    std::vector<AieComm> outputComms;
    for (auto &comm : placement.aieComms) {
      if (shimIdx == comm.srcIdx) {
        for (auto &packet : comm.packets) {
          if (packet.name.compare(0, 3, "rhs") == 0) {
            for (auto &dstIdx : packet.dstIdxs) {
              auto &dstTile = placement.aieTiles[dstIdx];
              uint32_t count = (5 - dstTile.row)*numCols + dstTile.col;

              // RES packets (output0)
              AiePacket newPacket;
              newPacket.name = std::string("res") + std::to_string(count);
              newPacket.packetId = packetIdList[outputPacketIdIdx++];
              newPacket.size = compTM * compTN;
              newPacket.elemType = elemType;
              newPacket.dstIdxs.push_back(shimIdx);
              newPacket.dstBundles.push_back(dmaWireBundle);
              newPacket.dstChs.push_back(0);

              AieComm outputComm{.name="output", .srcIdx=dstIdx, .srcBundle=dmaWireBundle, .srcCh=0, .isPacket=true};
              outputComm.packets.push_back(newPacket);

              outputComms.push_back(outputComm);
            }
          }
        }
      }
    }

    placement.aieComms.insert(placement.aieComms.end(), outputComms.begin(), outputComms.end());
  }

  // Configure AIE DMAs, BDs for each tile
  // 1. MM2S
  for (auto &comm : placement.aieComms) {
    auto &srcTile = placement.aieTiles[comm.srcIdx];
    auto [hasSrcDma, srcDmaIdx] = srcTile.findDmaIdx(DMAChannelDir::MM2S, comm.srcCh);
    bool hasLastBd;
    uint32_t lastBdIdx;
    
    if (hasSrcDma) { // DMA configuration already exists
      auto &srcDma = srcTile.dmas[srcDmaIdx];
      uint32_t firstBdIdx, curBdIdx, nextBdIdx;

      firstBdIdx = srcDma.bdIdx;
      curBdIdx = firstBdIdx;
      nextBdIdx = srcTile.bds[curBdIdx].nextBdIdx;

      while (nextBdIdx != firstBdIdx) {
        curBdIdx = nextBdIdx;
        nextBdIdx = srcTile.bds[curBdIdx].nextBdIdx;
      }

      hasLastBd = true;
      lastBdIdx = curBdIdx;
    } else { // DMA configuration does not exist
      AieDma srcDma{.dir=DMAChannelDir::MM2S, .channel=comm.srcCh};
      srcTile.dmas.push_back(srcDma);

      srcDmaIdx = srcTile.dmas.size() - 1;
      hasLastBd = false;
    }

    if (comm.isPacket) { // packet-switched communication
      for (auto &packet : comm.packets) {
        auto &srcDma = srcTile.dmas[srcDmaIdx];

        AieBufferDescriptor bd{.name=packet.name, .isPacket=true, .packetId=packet.packetId};
        bd.bufIdx = srcTile.findBufIdx(packet.name);
        bd.bufSize = packet.size;
        bd.bufOffset = 0;
        bd.nextBdIdx = srcDma.bdIdx;
        srcTile.bds.push_back(bd);

        if (hasLastBd) {
          auto &lastBd = srcTile.bds[lastBdIdx];
          uint32_t next = srcTile.bds.size() - 1;

          lastBd.nextBdIdx = next;
          lastBdIdx = next;
        } else {
          uint32_t next = srcTile.bds.size() - 1;
          
          srcDma.bdIdx = next;
          srcTile.bds[next].nextBdIdx = next;

          hasLastBd = true;
          lastBdIdx = next;
        }
      }
    } else { // circuit-switched communication
      // TODO: implement
    }
  }

  // 2. S2MM
  for (auto &comm : placement.aieComms) {
    if (comm.isPacket) { // packet-switched communication
      for (auto &packet : comm.packets) {
        for (uint32_t i = 0; i < packet.dstIdxs.size(); ++i) {
          uint32_t dstIdx = packet.dstIdxs[i];
          uint32_t dstCh = packet.dstChs[i];

          auto &dstTile = placement.aieTiles[dstIdx];
          auto [hasDstDma, dstDmaIdx] = dstTile.findDmaIdx(DMAChannelDir::S2MM, dstCh);
          bool hasLastBd;
          uint32_t lastBdIdx;
          
          if (hasDstDma) { // DMA configuration already exists
            auto &dstDma = dstTile.dmas[dstDmaIdx];
            uint32_t firstBdIdx, curBdIdx, nextBdIdx;

            firstBdIdx = dstDma.bdIdx;
            curBdIdx = firstBdIdx;
            nextBdIdx = dstTile.bds[curBdIdx].nextBdIdx;

            while (nextBdIdx != firstBdIdx) {
              curBdIdx = nextBdIdx;
              nextBdIdx = dstTile.bds[curBdIdx].nextBdIdx;
            }

            hasLastBd = true;
            lastBdIdx = curBdIdx;
          } else { // DMA configuration does not exist
            AieDma dstDma{.dir=DMAChannelDir::S2MM, .channel=dstCh};
            dstTile.dmas.push_back(dstDma);

            dstDmaIdx = dstTile.dmas.size() - 1;
            hasLastBd = false;
          }

          auto &dstDma = dstTile.dmas[dstDmaIdx];
          AieBufferDescriptor bd{.name=packet.name, .isPacket=true, .packetId=packet.packetId};
          bd.bufIdx = dstTile.findBufIdx(packet.name);
          bd.bufSize = (dstTile.row == 0) ? (packet.size + 1) : packet.size;
          bd.bufOffset = 0;
          bd.nextBdIdx = dstDma.bdIdx;
          dstTile.bds.push_back(bd);

          if (hasLastBd) {
            auto &lastBd = dstTile.bds[lastBdIdx];
            uint32_t next = dstTile.bds.size() - 1;

            lastBd.nextBdIdx = next;
            lastBdIdx = next;
          } else {
            uint32_t next = dstTile.bds.size() - 1;
            
            dstDma.bdIdx = next;
            dstTile.bds[next].nextBdIdx = next;

            hasLastBd = true;
            lastBdIdx = next;
          }
        }
      }
    } else { // circuit-switched communication
      // TODO: implement
    }
  }

  // Configure tile scheduling (runtime sequence)
  std::vector<AieNpuMemcpyNd> lhsTxSchedule;
  std::vector<AieNpuMemcpyNd> rhsTxSchedule;
  std::vector<AieNpuMemcpyNd> presTxSchedule;
  std::vector<AieNpuMemcpyNd> resRxSchedule;

  for (auto &tile : placement.aieTiles) {
    if (tile.row != 0) { // Comp tile
      continue;
    }
    else { // Shim tile
      for (auto &dma : tile.dmas) {
        if (dma.dir == DMAChannelDir::MM2S) {
          uint32_t firstBdIdx = dma.bdIdx;
          uint32_t curBdIdx = firstBdIdx;
          
          do {
            auto &bd = tile.bds[curBdIdx];
            std::array<int64_t,4> defaultSize = {1, 1, 1, static_cast<int64_t>(bd.bufSize)};
            std::array<int64_t,4> defaultStride = {0, 0, 0, 1};
            
            if (bd.name.compare(0, 3, "lhs") == 0) {
              uint32_t idx = static_cast<uint32_t>(std::strtoul(bd.name.c_str() + 3, nullptr, 10));
              uint32_t off = (tpOrder[0] == 0) ? (compTM * compTK * compTileTPm) :
                              ((tpOrder[0] == 2) ? (compTM * compTK * compTileTPk) : (compTM * compTK));

              std::string name = std::string("lhs") + std::to_string(tile.col) + std::to_string(tile.row);
              std::array<int64_t,4> offset = {0, 0, 0, static_cast<int64_t>(off * idx)};
              AieNpuMemcpyNd lhsTx{.name=name, .id=0, .bufIdx=bd.bufIdx, .isPacket=bd.isPacket, .packetType=0, .packetId=bd.packetId,
                                    .issueToken=true, .doWait=true, .staticOffset=offset, .staticSize=defaultSize, .staticStride=defaultStride};

              lhsTxSchedule.push_back(lhsTx);
            } else if (bd.name.compare(0, 3, "rhs") == 0) {
              uint32_t idx = static_cast<uint32_t>(std::strtoul(bd.name.c_str() + 3, nullptr, 10));
              uint32_t off = (tpOrder[0] == 1) ? (compTK * compTN * compTileTPn) :
                              ((tpOrder[0] == 2) ? (compTK * compTN * compTileTPk) : (compTK * compTN));

              std::string name = std::string("rhs") + std::to_string(tile.col) + std::to_string(tile.row);
              std::array<int64_t,4> offset = {0, 0, 0, static_cast<int64_t>(off * idx)};
              AieNpuMemcpyNd rhsTx{.name=name, .id=1, .bufIdx=bd.bufIdx, .isPacket=bd.isPacket, .packetType=0, .packetId=bd.packetId,
                                    .issueToken=true, .doWait=true, .staticOffset=offset, .staticSize=defaultSize, .staticStride=defaultStride};

              rhsTxSchedule.push_back(rhsTx);
            } else { // tile.bufs[bd.bufIdx].name == "pres"
              uint32_t idx = static_cast<uint32_t>(std::strtoul(bd.name.c_str() + 4, nullptr, 10));
              uint32_t off = (tpOrder[0] == 0) ? (compTM * compTN * compTileTPm) :
                              ((tpOrder[0] == 1) ? (compTM * compTN * compTileTPn) : (compTM * compTN));

              std::string name = std::string("pres") + std::to_string(tile.col) + std::to_string(tile.row);
              std::array<int64_t,4> offset = {0, 0, 0, static_cast<int64_t>(off * idx)};
              AieNpuMemcpyNd presTx{.name=name, .id=2, .bufIdx=bd.bufIdx, .isPacket=bd.isPacket, .packetType=0, .packetId=bd.packetId,
                                    .issueToken=true, .doWait=true, .staticOffset=offset, .staticSize=defaultSize, .staticStride=defaultStride};

              presTxSchedule.push_back(presTx);
            }

            curBdIdx = bd.nextBdIdx;
          } while (curBdIdx != firstBdIdx);
        } else { // dma.dir == DMAChannelDir::S2MM
          uint32_t firstBdIdx = dma.bdIdx;
          uint32_t curBdIdx = firstBdIdx;
          
          do {
            auto &bd = tile.bds[curBdIdx];
            uint32_t idx = static_cast<uint32_t>(std::strtoul(bd.name.c_str() + 3, nullptr, 10));
            uint32_t off = (tpOrder[0] == 0) ? ((compTM * compTN) * compTileTPm) :
                            ((tpOrder[0] == 1) ? ((compTM * compTN) * compTileTPn) : (compTM * compTN));
            
            std::string name = std::string("res") + std::to_string(tile.col) + std::to_string(tile.row);
            std::array<int64_t,4> offset = {0, 0, 0, static_cast<int64_t>(off * idx)};
            std::array<int64_t,4> size = {1, 1, 1, static_cast<int64_t>(bd.bufSize)};
            std::array<int64_t,4> stride = {0, 0, 0, 1};
            AieNpuMemcpyNd resRx{.name=name, .id=3, .bufIdx=bd.bufIdx, .isPacket=bd.isPacket, .packetType=0, .packetId=bd.packetId,
                                    .issueToken=true, .doWait=true, .staticOffset=offset, .staticSize=size, .staticStride=stride};

            resRxSchedule.insert(resRxSchedule.begin(), resRx);

            curBdIdx = bd.nextBdIdx;
          } while (curBdIdx != firstBdIdx);
        }
      }
    }
  }

  auto syncChannelParallelSameData = [](std::vector<AieNpuMemcpyNd>& v) {
    auto sameData = [](const AieNpuMemcpyNd& a, const AieNpuMemcpyNd& b) {
      return a.staticOffset[3] == b.staticOffset[3] &&
            a.staticSize[3]   == b.staticSize[3];
    };

    const size_t n = v.size();
    std::vector<char> moved(n, 0);
    std::vector<AieNpuMemcpyNd> out; out.reserve(n);

    for (size_t i = 0; i < n; ++i) {
      if (moved[i]) continue;
      std::vector<size_t> group{ i };
      moved[i] = 1;
      for (size_t j = i + 1; j < n; ++j)
        if (!moved[j] && sameData(v[i], v[j])) { group.push_back(j); moved[j] = 1; }

      for (size_t k = 0; k < group.size(); ++k) {
        auto item = v[group[k]];
        if (k + 1 < group.size()) item.doWait = false;
        out.push_back(std::move(item));
      }
    }
    v.swap(out);
  };

  syncChannelParallelSameData(lhsTxSchedule);
  syncChannelParallelSameData(rhsTxSchedule);
  syncChannelParallelSameData(presTxSchedule);
  syncChannelParallelSameData(resRxSchedule);

  placement.aieSchedule.insert(placement.aieSchedule.end(), presTxSchedule.begin(), presTxSchedule.end());
  placement.aieSchedule.insert(placement.aieSchedule.end(), lhsTxSchedule.begin(), lhsTxSchedule.end());
  placement.aieSchedule.insert(placement.aieSchedule.end(), rhsTxSchedule.begin(), rhsTxSchedule.end());

  if (tpOrder[0] == 0) {
    for (uint32_t i = 0; i < (compTileTPm-1); ++i) {
      placement.aieSchedule.insert(placement.aieSchedule.end(), resRxSchedule.begin(), resRxSchedule.end());

      for (auto &sch : resRxSchedule) {
        sch.staticOffset[3] += (compTM * compTN);
      }

      for (auto &sch : presTxSchedule) {
        sch.staticOffset[3] += (compTM * compTN);
      }
      
      for (auto &sch : lhsTxSchedule) {
        sch.staticOffset[3] += (compTM * compTK);
      }
      
      placement.aieSchedule.insert(placement.aieSchedule.end(), presTxSchedule.begin(), presTxSchedule.end());
      placement.aieSchedule.insert(placement.aieSchedule.end(), lhsTxSchedule.begin(), lhsTxSchedule.end());
    }
  } else if (tpOrder[0] == 1) {    
    for (uint32_t i = 0; i < (compTileTPn-1); ++i) {
      placement.aieSchedule.insert(placement.aieSchedule.end(), resRxSchedule.begin(), resRxSchedule.end());
      
      for (auto &sch : resRxSchedule) {
        sch.staticOffset[3] += (compTM * compTN);
      }

      for (auto &sch : presTxSchedule) {
        sch.staticOffset[3] += (compTM * compTN);
      }

      for (auto &sch : rhsTxSchedule) {
        sch.staticOffset[3] += (compTK * compTN);
      }

      placement.aieSchedule.insert(placement.aieSchedule.end(), presTxSchedule.begin(), presTxSchedule.end());
      placement.aieSchedule.insert(placement.aieSchedule.end(), rhsTxSchedule.begin(), rhsTxSchedule.end());
    }
  } else { // tpOrder[0] == 2
    for (uint32_t i = 0; i < (compTileTPk-1); ++i) {
      for (auto &sch : lhsTxSchedule) {
        sch.staticOffset[3] += (compTM * compTK);
      }
      
      for (auto &sch : rhsTxSchedule) {
        sch.staticOffset[3] += (compTK * compTN);
      }

      placement.aieSchedule.insert(placement.aieSchedule.end(), lhsTxSchedule.begin(), lhsTxSchedule.end());
      placement.aieSchedule.insert(placement.aieSchedule.end(), rhsTxSchedule.begin(), rhsTxSchedule.end());
    }
  }

  placement.aieSchedule.insert(placement.aieSchedule.end(), resRxSchedule.begin(), resRxSchedule.end());

  if (DebugAiePlacement) {
    llvm::dbgs() << "[AiePlacement] Computed AIE Placement:\n";

    // ---- Tiles ----
    llvm::dbgs() << "  aieTiles (" << placement.aieTiles.size() << "):\n";
    for (size_t tileIdx = 0; tileIdx < placement.aieTiles.size(); ++tileIdx) {
      const auto &tile = placement.aieTiles[tileIdx];
      llvm::dbgs() << "    Tile[" << tileIdx << "] (" << tile.col << ", " << tile.row << ")\n";

      // Buffers (no Value prints)
      llvm::dbgs() << "      bufs (" << tile.bufs.size() << "):\n";
      for (size_t i = 0; i < tile.bufs.size(); ++i) {
        const auto &b = tile.bufs[i];
        llvm::dbgs() << "        - [" << i << "] name=" << b.name
                    << " symbol=" << b.symbol
                    << " size=" << b.bufSize
                    << " elemType=";
        if (b.elemType) b.elemType.print(llvm::dbgs());
        else            llvm::dbgs() << "<null>";
        llvm::dbgs() << "\n";
      }

      // Buffer Descriptors
      llvm::dbgs() << "      bds (" << tile.bds.size() << "):\n";
      for (size_t i = 0; i < tile.bds.size(); ++i) {
        const auto &bd = tile.bds[i];
        llvm::dbgs() << "        - [" << i << "]" 
                    << " name=" << bd.name
                    << " isPacket=" << (bd.isPacket ? "true" : "false");
        if (bd.isPacket)
          llvm::dbgs() << " packetId=" << bd.packetId;
        llvm::dbgs() << " bufIdx=" << bd.bufIdx
                    << " bufOffset=" << bd.bufOffset
                    << " bufSize=" << bd.bufSize
                    << " nextBdIdx=" << bd.nextBdIdx << "\n";
      }

      // DMAs
      llvm::dbgs() << "      dmas (" << tile.dmas.size() << "):\n";
      for (size_t i = 0; i < tile.dmas.size(); ++i) {
        const auto &d = tile.dmas[i];
        llvm::dbgs() << "        - [" << i << "] dir=" << static_cast<int>(d.dir)
                    << " channel=" << d.channel
                    << " bdIdx=" << d.bdIdx << "\n";
      }
    }

    // ---- Comms (packet or circuit) ----
    llvm::dbgs() << "  aieComms (" << placement.aieComms.size() << "):\n";
    for (size_t commIdx = 0; commIdx < placement.aieComms.size(); ++commIdx) {
      const auto &comm = placement.aieComms[commIdx];
      const auto &srcTile = placement.getAieTile(comm.srcIdx);

      llvm::dbgs() << "    Comm[" << commIdx << "] name=" << comm.name
                  << " src=(" << srcTile.col << ", " << srcTile.row << ")"
                  << " srcBundle=" << static_cast<int>(comm.srcBundle)
                  << " srcCh=" << comm.srcCh
                  << " isPacket=" << (comm.isPacket ? "true" : "false") << "\n";

      auto printDsts = [&](size_t nd,
                          const std::vector<uint32_t> &dstIdxs,
                          const std::vector<WireBundle> &dstBundles,
                          const std::vector<uint32_t> &dstChs) {
        if (nd != dstIdxs.size() || nd != dstBundles.size() || nd != dstChs.size()) {
          llvm::dbgs() << "            (warn) dst arrays length mismatch: "
                      << "dstIdxs=" << dstIdxs.size()
                      << " dstBundles=" << dstBundles.size()
                      << " dstChs="  << dstChs.size()  << "\n";
        }
        llvm::dbgs() << "            dsts (" << nd << "):\n";
        for (size_t j = 0; j < nd; ++j) {
          uint32_t dstIdx = dstIdxs[j];
          const auto &dstTile = placement.getAieTile(dstIdx);
          llvm::dbgs() << "              • [" << j << "] -> tile("
                      << dstTile.col << ", " << dstTile.row << ")"
                      << " Bundle=" << static_cast<int>(dstBundles[j])
                      << " ch="   << dstChs[j] << "\n";
        }
      };

      if (comm.isPacket) {
        // Packets
        llvm::dbgs() << "      packets (" << comm.packets.size() << "):\n";
        for (size_t pIdx = 0; pIdx < comm.packets.size(); ++pIdx) {
          const auto &p = comm.packets[pIdx];
          llvm::dbgs() << "        - Packet[" << pIdx << "] name=" << p.name
                      << " packetId=" << p.packetId
                      << " size=" << p.size
                      << " elemType=";
          if (p.elemType) p.elemType.print(llvm::dbgs());
          else            llvm::dbgs() << "<null>";
          llvm::dbgs() << "\n";

          size_t nd = std::min({p.dstIdxs.size(), p.dstBundles.size(), p.dstChs.size()});
          printDsts(nd, p.dstIdxs, p.dstBundles, p.dstChs);
        }
        if (!comm.circuits.empty()) {
          llvm::dbgs() << "      (warn) isPacket=true but circuits not empty: "
                      << comm.circuits.size() << "\n";
        }
      } else {
        // Circuits
        llvm::dbgs() << "      circuits (" << comm.circuits.size() << "):\n";
        for (size_t cIdx = 0; cIdx < comm.circuits.size(); ++cIdx) {
          const auto &c = comm.circuits[cIdx];
          llvm::dbgs() << "        - Circuit[" << cIdx << "] name=" << c.name
                      << " size=" << c.size
                      << " elemType=";
          if (c.elemType) c.elemType.print(llvm::dbgs());
          else            llvm::dbgs() << "<null>";
          llvm::dbgs() << "\n";

          size_t nd = std::min({c.dstIdxs.size(), c.dstBundles.size(), c.dstChs.size()});
          printDsts(nd, c.dstIdxs, c.dstBundles, c.dstChs);
        }
        if (!comm.packets.empty()) {
          llvm::dbgs() << "      (warn) isPacket=false but packets not empty: "
                      << comm.packets.size() << "\n";
        }
      }
    }

    // ---- Schedule (NPU DMA memcpy ND) ----
    llvm::dbgs() << "  aieSchedule (" << placement.aieSchedule.size() << "):\n";
    for (size_t sIdx = 0; sIdx < placement.aieSchedule.size(); ++sIdx) {
      const auto &sch = placement.aieSchedule[sIdx];

      auto printI64x4 = [&](const char *label, const int64_t v[4]) {
        llvm::dbgs() << " " << label << "=["
                     << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << "]";
      };

      llvm::dbgs() << "    Sched[" << sIdx << "]"
                   << " name=" << sch.name
                   << " id=" << sch.id
                   << " bufIdx=" << sch.bufIdx
                   << " isPacket=" << (sch.isPacket ? "true" : "false");

      if (sch.isPacket) {
        llvm::dbgs() << " packetType=" << sch.packetType
                     << " packetId="   << sch.packetId;
      }

      llvm::dbgs() << " issueToken=" << (sch.issueToken ? "true" : "false")
                   << " doWait="     << (sch.doWait ? "true" : "false");

      printI64x4("offset", sch.staticOffset.data());
      printI64x4("size",   sch.staticSize.data());
      printI64x4("stride", sch.staticStride.data());

      llvm::dbgs() << "\n";
    }

    llvm::dbgs() << "\n";
  }

  return placement;
}

void generateAieOps(ConversionPatternRewriter &rewriter,
                    AiePlacement &placement,
                    const TileParam &tileParam) {
  // Set variables
  uint32_t numCompTilesPerCol = 4;
  uint32_t numCols = tileParam.numLastSpm / numCompTilesPerCol;

  auto &compTileLevel = tileParam.levels[0];
  auto [compTM, compTK, compTN] = compTileLevel.tileSize;
  uint32_t compTileSPm = compTileLevel.SPm;
  uint32_t compTileSPn = compTileLevel.SPn;
  uint32_t compTileTPm = compTileLevel.TPm;
  uint32_t compTileTPk = compTileLevel.TPk;
  uint32_t compTileTPn = compTileLevel.TPn;
  auto tpOrder = compTileLevel.tpOrder;

  auto elemType = tileParam.elemType;
  bool doubleBufferEnabled = tileParam.doubleBufferEnabled;

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
      for (auto &buf : tile.bufs) {
        buf.symbol = std::string("global_") + buf.name + "_" + std::to_string(tile.col) + "_" + std::to_string(tile.row);
        auto globalMemrefNameAttr = builder.getStringAttr(buf.symbol);
        auto globalMemrefType = MemRefType::get({buf.bufSize}, buf.elemType);
        builder.create<memref::GlobalOp>(loc, globalMemrefNameAttr, builder.getStringAttr("public"),
                                                         globalMemrefType, nullptr, false, nullptr);
      }
    } else { // Comp tile
      // Generate AIE BufferOp
      for (auto &buf : tile.bufs) {
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
      for (auto &buf : tile.bufs) {
        uint32_t numProdToken = 1;
        uint32_t numConsToken = 0;

        { // Producer lock
          auto idAttr = builder.getI32IntegerAttr(id++);
          auto initAttr = builder.getI32IntegerAttr(numProdToken);
          auto nameAttr = builder.getStringAttr(buf.symbol + "_prod_lock");
          auto lockOp = builder.create<xilinx::AIE::LockOp>(loc, tile.value, idAttr, initAttr, nameAttr);
          buf.prodLockValue = lockOp;
        }

        { // Consumer lock
          auto idAttr = builder.getI32IntegerAttr(id++);
          auto initAttr = builder.getI32IntegerAttr(numConsToken);
          auto nameAttr = builder.getStringAttr(buf.symbol + "_cons_lock");
          auto lockOp = builder.create<xilinx::AIE::LockOp>(loc, tile.value, idAttr, initAttr, nameAttr);
          buf.consLockValue = lockOp;
        }

        if (buf.name == "res") {
          if ((compTileTPk > 1) && (tpOrder[0] != 2)) { // Calculator lock
            uint32_t numCalcToken = 0;
            auto idAttr = builder.getI32IntegerAttr(id++);
            auto initAttr = builder.getI32IntegerAttr(numCalcToken);
            auto nameAttr = builder.getStringAttr(buf.symbol + "_calc_lock");
            auto lockOp = builder.create<xilinx::AIE::LockOp>(loc, tile.value, idAttr, initAttr, nameAttr);
            buf.calcLockValue = lockOp;
          }
        }
      }
    }
  }

  // Generate communication paths
  for (auto &comm : placement.aieComms) {
    if (comm.isPacket) {  // packet-switched communication
      auto &srcTile = placement.aieTiles[comm.srcIdx];
      BoolAttr keep_pkt_header = nullptr;

      // Generate AIEX PacketFlowOp
      if (srcTile.row != 0) {
        keep_pkt_header = builder.getBoolAttr(true);
      }
        
      for (auto &packet : comm.packets) {
        int8_t pkt_Id = packet.packetId;

        auto flowOp = builder.create<xilinx::AIE::PacketFlowOp>(loc, pkt_Id, keep_pkt_header, nullptr);
        {
          OpBuilder::InsertionGuard g(builder);
          Region &flowRegion = flowOp.getBodyRegion();
          Block *flowBlock = builder.createBlock(&flowRegion);
          builder.setInsertionPointToStart(flowBlock);

          builder.create<xilinx::AIE::PacketSourceOp>(loc, srcTile.value, comm.srcBundle, static_cast<int32_t>(comm.srcCh));

          for (uint32_t i = 0; i < packet.dstIdxs.size(); ++i) {
            auto dstIdx = packet.dstIdxs[i];
            auto dstBundle = packet.dstBundles[i];
            auto dstCh = packet.dstChs[i];
            auto &dstTile = placement.aieTiles[dstIdx];

            builder.create<xilinx::AIE::PacketDestOp>(loc, dstTile.value, dstBundle, static_cast<int32_t>(dstCh));
          }

          builder.create<EndOp>(loc);
        }
      }
    } else { // circuit-switched communication
      // TODO: implement (FlowOp)
    }
  }

  // Generate AIE DMAOp
  for (auto &tile : placement.aieTiles) {
    Operation *dmaOp = nullptr;

    if (tile.row == 0) { // Shim tile
      for (auto &dma : tile.dmas) {
        std::vector<bool> allocatedBuf(tile.bufs.size(), false);
        uint32_t firstBdIdx = dma.bdIdx;
        uint32_t curBdIdx = firstBdIdx;

        do {
          auto &bd = tile.bds[curBdIdx];
          auto &buf = tile.bufs[bd.bufIdx];

          if (!allocatedBuf[bd.bufIdx]) {
            auto globalSym = SymbolRefAttr::get(builder.getContext(), buf.symbol);
            DMAChannelDirAttr dmaDirAttr = DMAChannelDirAttr::get(builder.getContext(), dma.dir);
            auto &channelIdx = dma.channel;
  
            builder.create<ShimDMAAllocationOp>(loc, globalSym, dmaDirAttr,
                                                builder.getI64IntegerAttr(channelIdx),
                                                builder.getI64IntegerAttr(tile.col));

            allocatedBuf[bd.bufIdx] = true;
          }

          curBdIdx = bd.nextBdIdx;
        } while (curBdIdx != firstBdIdx);
      }

      continue;
    } else { // Comp tile
      dmaOp = builder.create<MemOp>(loc, tile.value).getOperation();
    }

    {
      OpBuilder::InsertionGuard g(builder);
      Region &DMARegion = dmaOp->getRegion(0);

      std::vector<Block*> dmaBlocks;
      std::vector<Block*> bdBlocks;

      uint32_t numDmaBlocks = tile.dmas.size();
      uint32_t numBdBlocks = tile.bds.size();

      for (uint32_t i = 0; i < numDmaBlocks; ++i){
        Block *dmaBlock = builder.createBlock(&DMARegion);
        dmaBlocks.push_back(dmaBlock);
      }
      for (uint32_t i = 0; i < numBdBlocks; ++i){
        Block *bdBlock = builder.createBlock(&DMARegion);
        bdBlocks.push_back(bdBlock);
      }
      Block *endBlock = builder.createBlock(&DMARegion);
      dmaBlocks.push_back(endBlock);

      for (uint32_t i = 0; i < numDmaBlocks; ++i){
        auto &dma = tile.dmas[i];
        uint32_t firstBdIdx = dma.bdIdx;

        {
          OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(dmaBlocks[i]);

          DMAChannelDirAttr dmaDirAttr = DMAChannelDirAttr::get(builder.getContext(), dma.dir);
          auto channelIdxAttr = builder.getI32IntegerAttr(dma.channel);
          auto repeatCntAttr = builder.getI32IntegerAttr(0);

          builder.create<DMAStartOp>(loc, dmaDirAttr, channelIdxAttr, repeatCntAttr, bdBlocks[firstBdIdx], dmaBlocks[i+1]);
        }

        uint32_t curBdIdx = firstBdIdx;
        do {
          auto &bd = tile.bds[curBdIdx];
          auto &buf = tile.bufs[bd.bufIdx];

          {
            OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToStart(bdBlocks[curBdIdx]);

            auto acquireLockValue = (dma.dir == DMAChannelDir::S2MM) ? buf.prodLockValue : buf.consLockValue;
            auto releaseLockValue = (dma.dir == DMAChannelDir::S2MM) ? buf.consLockValue : buf.prodLockValue;
            uint32_t numToken = 1;

            if ((buf.name == "res") && (dma.dir == DMAChannelDir::S2MM)) {
              if ((compTileTPk > 1) && (tpOrder[0] != 2)) {
                releaseLockValue = buf.calcLockValue;
              }
            }

            builder.create<UseLockOp>(loc, acquireLockValue, LockAction::AcquireGreaterEqual, numToken);

            if ((dma.dir == DMAChannelDir::MM2S) && (bd.isPacket == true)) {
              builder.create<DMABDPACKETOp>(loc, 0, bd.packetId);
            } 

            builder.create<DMABDOp>(loc, buf.bufValue, bd.bufOffset, bd.bufSize);
            builder.create<UseLockOp>(loc, releaseLockValue, LockAction::Release, numToken);
            builder.create<NextBDOp>(loc, bdBlocks[bd.nextBdIdx]);
          }

          curBdIdx = bd.nextBdIdx;
        } while (curBdIdx != firstBdIdx);
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
  auto lhsMemrefType = MemRefType::get({compTM * compTK}, elemType);
  auto rhsMemrefType = MemRefType::get({compTK * compTN}, elemType);
  auto resMemrefType = MemRefType::get({compTM * compTN}, elemType);
  auto i32Type = builder.getI32Type();
  auto i1Type = builder.getI1Type();
  FunctionType funcType = builder.getFunctionType({lhsMemrefType, rhsMemrefType, resMemrefType, i32Type, i32Type, i32Type, i1Type}, {});
  auto funcOp = builder.create<func::FuncOp>(loc, funcNameAttr, funcType);
  funcOp.setPrivate();

  // Configure operations of Compute tile
  for (auto &tile : placement.aieTiles) {
    if (tile.row < 2) { // Shim/Mem tile
      continue;
    }

    // Set vector for each buffer (lhs/rhs/res)
    using Args = SmallVector<Value, 3>;
    using InitArgs = SmallVector<Args, 2>;

    InitArgs lhsInitArgs;
    InitArgs rhsInitArgs;
    InitArgs resInitArgs;

    for (auto &buf : tile.bufs) {
      const std::string &name = buf.name;
      
      if (name == "lhs" || name == "lhsdb") {
        Args bufArgs{buf.bufValue, buf.consLockValue, buf.prodLockValue};
        lhsInitArgs.push_back(bufArgs);
      } else if (name == "rhs" || name == "rhsdb") {
        Args bufArgs{buf.bufValue, buf.consLockValue, buf.prodLockValue};
        rhsInitArgs.push_back(bufArgs);
      } else if (name == "res" || name == "resdb") {
        auto acquireValue = ((compTileTPk > 1) && (tpOrder[0] != 2)) ? 
                                      buf.calcLockValue : buf.prodLockValue;
        Args bufArgs{buf.bufValue, acquireValue, buf.consLockValue};
        resInitArgs.push_back(bufArgs);
      }
    }

    InitArgs *reuseInitArgsPtr = nullptr;
    InitArgs *inner1InitArgsPtr = nullptr;
    InitArgs *inner2InitArgsPtr = nullptr;

    if (tpOrder[0] == 0) {
      reuseInitArgsPtr = &rhsInitArgs;
      inner1InitArgsPtr = &resInitArgs;
      inner2InitArgsPtr = &lhsInitArgs;
    } else if (tpOrder[0] == 1) {
      reuseInitArgsPtr = &lhsInitArgs;
      inner1InitArgsPtr = &resInitArgs;
      inner2InitArgsPtr = &rhsInitArgs;
    } else { // tpOrder[0] == 2
      reuseInitArgsPtr = &resInitArgs;
      inner1InitArgsPtr = &lhsInitArgs;
      inner2InitArgsPtr = &rhsInitArgs;
    }

    auto &reuseInitArgs = *reuseInitArgsPtr;
    auto &inner1InitArgs = *inner1InitArgsPtr;
    auto &inner2InitArgs = *inner2InitArgsPtr;

    uint32_t repeatCount = (tpOrder[0] == 0) ? compTileTPm :
                            ((tpOrder[0] == 1) ? compTileTPn : compTileTPk);

    // Generate AIE CoreOp
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
      auto cMax = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0x7FFFFFFFFFFFFFFFULL));
      auto cCnt = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(repeatCount));

      auto cRow = builder.create<mlir::arith::ConstantIntOp>(loc, compTM, /*width=*/32);
      auto cCol = builder.create<mlir::arith::ConstantIntOp>(loc, compTN, /*width=*/32);
      auto cDep = builder.create<mlir::arith::ConstantIntOp>(loc, compTK, /*width=*/32);

      auto trueI1 = builder.create<arith::ConstantIntOp>(loc, /*value=*/1, /*bitWidth=*/1);
      auto falseI1 = builder.create<arith::ConstantIntOp>(loc, /*value=*/0, /*bitWidth=*/1);
      auto accVar = builder.create<memref::AllocOp>(loc, MemRefType::get({}, builder.getI1Type()));

      SmallVector<Value, 11> outerInitArgs;
      outerInitArgs.append(reuseInitArgs[0].begin(), reuseInitArgs[0].end());
      outerInitArgs.append(inner1InitArgs[0].begin(), inner1InitArgs[0].end());
      outerInitArgs.append(inner2InitArgs[0].begin(), inner2InitArgs[0].end());
      outerInitArgs.push_back(trueI1);
      outerInitArgs.push_back(trueI1);

      // Generate SCF ForOp (outer loop: infinite)
      auto outerLoopOp = builder.create<mlir::scf::ForOp>(loc, c0, cMax, c1, outerInitArgs);
      {
        OpBuilder::InsertionGuard g(builder);
        Region &outerLoopRegion = outerLoopOp.getRegion();
        builder.setInsertionPointToStart(&outerLoopRegion.back());

        // Set arguments
        auto outerArgs = outerLoopOp.getRegionIterArgs();
        Args reuseArgs{outerArgs[0], outerArgs[1], outerArgs[2]};
        Args inner1Args{outerArgs[3], outerArgs[4], outerArgs[5]};
        Args inner2Args{outerArgs[6], outerArgs[7], outerArgs[8]};
        Value innerT = outerArgs[9];
        Value outerT = outerArgs[10];

        SmallVector<Value, 7> innerInitArgs;
        innerInitArgs.append(inner1Args.begin(), inner1Args.end());
        innerInitArgs.append(inner2Args.begin(), inner2Args.end());
        innerInitArgs.push_back(innerT);

        // Generate Memref StoreOp
        if ((compTileTPk > 1) && (tpOrder[0] != 2)) {
          builder.create<memref::StoreOp>(loc, trueI1, accVar);
        } else {
          builder.create<memref::StoreOp>(loc, falseI1, accVar);
        }

        // Generate AIE UseLockOp (reuse)
        builder.create<UseLockOp>(loc, reuseArgs[1], LockAction::AcquireGreaterEqual, 1);

        // Generate SCF ForOp (inner loop: calc matmul)
        auto innerLoopOp = builder.create<mlir::scf::ForOp>(loc, c0, cCnt, c1, innerInitArgs);
        {
          OpBuilder::InsertionGuard g(builder);
          Region &innerLoopRegion = innerLoopOp.getRegion();
          builder.setInsertionPointToStart(&innerLoopRegion.back());

          // Set arguments
          auto innerArgs = innerLoopOp.getRegionIterArgs();
          Args arg1{innerArgs[0], innerArgs[1], innerArgs[2]};
          Args arg2{innerArgs[3], innerArgs[4], innerArgs[5]};
          Value innerT = innerArgs[6];

          // Generate AIE UseLockOp (arg1/arg2)
          builder.create<UseLockOp>(loc, arg1[1], LockAction::AcquireGreaterEqual, 1);
          builder.create<UseLockOp>(loc, arg2[1], LockAction::AcquireGreaterEqual, 1);

          // Generate Func CallOp
          Value acc = builder.create<memref::LoadOp>(loc, accVar.getResult(), ValueRange{});

          SmallVector<Value, 4> commonArgs{cRow, cCol, cDep, acc};
          SmallVector<Value, 7> callArgs;
          if (tpOrder[0] == 0) { // arg1: res, arg2: lhs
            callArgs.push_back(arg2[0]);
            callArgs.push_back(reuseArgs[0]);
            callArgs.push_back(arg1[0]);
          } else if (tpOrder[0] == 1) { // arg1: res, arg2: rhs
            callArgs.push_back(reuseArgs[0]);
            callArgs.push_back(arg2[0]);
            callArgs.push_back(arg1[0]);
          } else { // tpOrder[0] == 2, arg1: lhs, arg2: rhs
            callArgs.push_back(arg1[0]);
            callArgs.push_back(arg2[0]);
            callArgs.push_back(reuseArgs[0]);
          }
          callArgs.append(commonArgs.begin(), commonArgs.end());

          auto calleeAttr = SymbolRefAttr::get(builder.getContext(), "extern_kernel");
          builder.create<mlir::func::CallOp>(loc, calleeAttr, TypeRange{}, ValueRange(callArgs));
    
          // Generate AIE UseLockOp (arg1/arg2)
          builder.create<UseLockOp>(loc, arg2[2], LockAction::Release, 1);
          builder.create<UseLockOp>(loc, arg1[2], LockAction::Release, 1);

          // Generate Memref StoreOp (acc)
          if ((compTileTPk > 1) && (tpOrder[0] == 2)) {
            builder.create<memref::StoreOp>(loc, trueI1, accVar);
          }

          // Generate SCF YieldOp
          if (!doubleBufferEnabled) {
            builder.create<mlir::scf::YieldOp>(loc, ValueRange{arg1[0], arg1[1], arg1[2],
                                                                arg2[0], arg2[1], arg2[2], innerT});
          } else {
            Value innerT2 = builder.create<arith::XOrIOp>(loc, /*lhs=*/innerT, /*rhs=*/trueI1);

            llvm::SmallVector<Type, 7> packTys{
              arg1[0].getType(), arg1[1].getType(), arg1[2].getType(),
              arg2[0].getType(), arg2[1].getType(), arg2[2].getType(),
              innerT.getType()
            };

            auto ifPack = builder.create<mlir::scf::IfOp>(loc, TypeRange(packTys),
                                                    /*cond=*/innerT2, /*withElseRegion=*/true);
            // then (db0)
            {
              OpBuilder::InsertionGuard g(builder);
              Block &tb = ifPack.getThenRegion().front();
              builder.setInsertionPointToStart(&tb);
              builder.create<mlir::scf::YieldOp>(loc, ValueRange{
                inner1InitArgs[0][0], inner1InitArgs[0][1], inner1InitArgs[0][2],
                inner2InitArgs[0][0], inner2InitArgs[0][1], inner2InitArgs[0][2],
                innerT2
              });
            }
            // else (db1)
            {
              OpBuilder::InsertionGuard g(builder);
              Block &eb = ifPack.getElseRegion().front();
              builder.setInsertionPointToStart(&eb);
              builder.create<mlir::scf::YieldOp>(loc, ValueRange{
                inner1InitArgs[1][0], inner1InitArgs[1][1], inner1InitArgs[1][2],
                inner2InitArgs[1][0], inner2InitArgs[1][1], inner2InitArgs[1][2],
                innerT2
              });
            }

            builder.create<mlir::scf::YieldOp>(loc, ValueRange{
              ifPack.getResult(0), ifPack.getResult(1), ifPack.getResult(2),
              ifPack.getResult(3), ifPack.getResult(4), ifPack.getResult(5),
              ifPack.getResult(6)
            });
          }
        }

        // Generate AIE UseLockOp (reuse)
        builder.create<UseLockOp>(loc, reuseArgs[2], LockAction::Release, 1);

        // Generate SCF YieldOp
        if (!doubleBufferEnabled) {
          builder.create<mlir::scf::YieldOp>(loc, ValueRange{
            reuseArgs[0], reuseArgs[1], reuseArgs[2], 
            innerLoopOp.getResult(0), innerLoopOp.getResult(1), innerLoopOp.getResult(2),
            innerLoopOp.getResult(3), innerLoopOp.getResult(4), innerLoopOp.getResult(5),
            innerLoopOp.getResult(6), outerT});
        } else {
          Value outerT2 = builder.create<arith::XOrIOp>(loc, /*lhs=*/outerT, /*rhs=*/trueI1);

          llvm::SmallVector<Type, 4> packTys{
            reuseArgs[0].getType(), reuseArgs[1].getType(), reuseArgs[2].getType(),
            outerT.getType()
          };

          auto ifPack = builder.create<mlir::scf::IfOp>(loc, TypeRange(packTys),
                                                  /*cond=*/outerT2, /*withElseRegion=*/true);
          // then (db0)
          {
            OpBuilder::InsertionGuard g(builder);
            Block &tb = ifPack.getThenRegion().front();
            builder.setInsertionPointToStart(&tb);
            builder.create<mlir::scf::YieldOp>(loc, ValueRange{
              reuseInitArgs[0][0], reuseInitArgs[0][1], reuseInitArgs[0][2],
              outerT2
            });
          }
          // else (db1)
          {
            OpBuilder::InsertionGuard g(builder);
            Block &eb = ifPack.getElseRegion().front();
            builder.setInsertionPointToStart(&eb);
            builder.create<mlir::scf::YieldOp>(loc, ValueRange{
              reuseInitArgs[1][0], reuseInitArgs[1][1], reuseInitArgs[1][2],
              outerT2
            });
          }

          builder.create<mlir::scf::YieldOp>(loc, ValueRange{
            ifPack.getResult(0), ifPack.getResult(1), ifPack.getResult(2),
            innerLoopOp.getResult(0), innerLoopOp.getResult(1), innerLoopOp.getResult(2),
            innerLoopOp.getResult(3), innerLoopOp.getResult(4), innerLoopOp.getResult(5),
            innerLoopOp.getResult(6), ifPack.getResult(3)
          });
        }
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

    uint32_t localTPm = (tpOrder[0] == 0) ? compTileTPm : 1;
    uint32_t localTPn = (tpOrder[0] == 1) ? compTileTPn : 1;
    uint32_t localTPk = (tpOrder[0] == 2) ? compTileTPk : 1;

    uint32_t lhsSize = ((compTM * compTileSPm) * compTK) * localTPm * localTPk;
    uint32_t rhsSize = (compTK * (compTN * compTileSPn)) * localTPk * localTPn;
    uint32_t resSize = ((compTM * compTN) * compTileSPm * compTileSPn) * localTPm * localTPn;

    auto lhsMemrefType = MemRefType::get({lhsSize}, elemType);
    auto rhsMemrefType = MemRefType::get({rhsSize}, elemType);
    auto resMemrefType = MemRefType::get({resSize}, elemType);

    auto arg_lhs = seqBlock->addArgument(lhsMemrefType, loc);
    auto arg_rhs = seqBlock->addArgument(rhsMemrefType, loc);
    auto arg_res = seqBlock->addArgument(resMemrefType, loc);

    Value arg_pres;
    if ((compTileTPk > 1) && (tpOrder[0] != 2)) {
      uint32_t presSize = ((compTM * compTN) * compTileSPm * compTileSPn) * localTPm * localTPn;
      auto presMemrefType = MemRefType::get({presSize}, elemType);
      arg_pres = seqBlock->addArgument(presMemrefType, loc);
    }

    // Generate AIEX NpuDmaMemcpyNdOp
    for (auto &sch : placement.aieSchedule) {
      Value arg;
      uint32_t col, row;

      if (sch.name.compare(0, 3, "lhs") == 0) {
        arg = arg_lhs;
        col = static_cast<uint32_t>(sch.name[3] - '0');
        row = static_cast<uint32_t>(sch.name[4] - '0');
      } else if (sch.name.compare(0, 3, "rhs") == 0) {
        arg = arg_rhs;
        col = static_cast<uint32_t>(sch.name[3] - '0');
        row = static_cast<uint32_t>(sch.name[4] - '0');
      } else if (sch.name.compare(0, 3, "res") == 0) {
        arg = arg_res;
        col = static_cast<uint32_t>(sch.name[3] - '0');
        row = static_cast<uint32_t>(sch.name[4] - '0');
      } else { // sch.name.compare(0, 4, "pres") == 0
        arg = arg_pres;
        col = static_cast<uint32_t>(sch.name[4] - '0');
        row = static_cast<uint32_t>(sch.name[5] - '0');
      }
      
      uint32_t shimIdx = placement.findAieTileIdx(col, row);
      AieTile &shimTile = placement.aieTiles[shimIdx];
      
      auto &buf = shimTile.bufs[sch.bufIdx];
      StringRef metadata = builder.getStringAttr(buf.symbol);
      PacketInfoAttr packetAttr = nullptr;
      if (sch.isPacket) {
        packetAttr = PacketInfoAttr::get(builder.getContext(), static_cast<uint16_t>(sch.packetType), static_cast<uint16_t>(sch.packetId));
      }
      
      builder.create<xilinx::AIEX::NpuDmaMemcpyNdOp>(loc, arg, SmallVector<Value>{}, SmallVector<Value>{}, SmallVector<Value>{},
                                                    ArrayRef(sch.staticOffset), ArrayRef(sch.staticSize), ArrayRef(sch.staticStride), 
                                                    packetAttr, metadata, sch.id, sch.issueToken, 0, 0, 0, 0, 0, 0);

      if (sch.doWait) {
        builder.create<xilinx::AIEX::NpuDmaWaitOp>(loc, metadata);
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
