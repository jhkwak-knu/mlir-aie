//===- TileParam.cpp - System info / tile param JSON loaders -----*- C++ -*-===//
//
// Loads hardware system configuration (xdna2_info.json) and tiling parameters
// (tc.json) from JSON files.  Builds TilingContext from TileParam.
//
//===----------------------------------------------------------------------===//

#include "TileParam.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/Twine.h"

#include <bitset>

using namespace mlir;

namespace onnx_to_aie {

//===----------------------------------------------------------------------===//
// JSON helper getters
//===----------------------------------------------------------------------===//
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
  auto *arr = obj.getArray(key);
  if (!arr) {
    llvm::errs() << "Error: missing required array '" << key << "'\n";
    llvm::report_fatal_error("tpOrder missing");
  }
  {
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
    if (out.size() != 3) {
      llvm::errs() << "Error: '" << key << "' must have exactly 3 elements"
                   << " (M=0, N=1, K=2), got " << out.size() << "\n";
      llvm::report_fatal_error("tpOrder size error");
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

//===----------------------------------------------------------------------===//
// loadSystemInfo
//===----------------------------------------------------------------------===//
std::optional<SystemInfo> loadSystemInfo(const std::string &filePath,
                                         bool debug) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (!bufOrErr) {
    llvm::errs() << "Error: cannot open file '" << filePath << "' (" << bufOrErr.getError().message() << ")\n";
    return std::nullopt;
  }
  StringRef jsonText = (*bufOrErr)->getBuffer();

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

  auto *sysObj = rootObj->getObject("system");
  if (!sysObj) {
    llvm::errs() << "Error: missing 'system' object\n";
    return std::nullopt;
  }

  SystemInfo info;

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

  // Parse optional "device" section; absent fields keep DeviceConfig defaults.
  if (auto *devObj = sysObj->getObject("device")) {
    auto &d = info.device;
    if (auto v = devObj->getInteger("max_columns"))
      d.maxColumns = static_cast<uint32_t>(*v);
    if (auto v = devObj->getInteger("comp_tiles_per_col"))
      d.compTilesPerCol = static_cast<uint32_t>(*v);
    if (auto v = devObj->getInteger("shim_row"))
      d.shimRow = static_cast<uint32_t>(*v);
    if (auto v = devObj->getInteger("comp_tile_first_row"))
      d.compTileFirstRow = static_cast<uint32_t>(*v);
    if (auto v = devObj->getInteger("mem_tile_mem_bytes"))
      d.memTileMemBytes = static_cast<uint32_t>(*v);
    if (auto v = devObj->getInteger("pkt_hdr_bytes"))
      d.pktHdrBytes = static_cast<uint32_t>(*v);
  }

  if (debug) {
    llvm::dbgs() << "[SystemInfo] Loaded SystemInfo:\n"
                 << "  name:         " << info.name       << "\n"
                 << "  numSpmLevels: " << info.numSpmLevels << "\n"
                 << "  totalCores:   " << info.totalCores   << "\n";
    for (auto &lvl : info.spmLevels)
      llvm::dbgs() << "    level=" << lvl.level
                   << " numSpms=" << lvl.numSpms
                   << " sizeBytes=" << lvl.spmSizeBytes << "\n";
    const auto &d = info.device;
    llvm::dbgs() << "  device:\n"
                 << "    maxColumns=" << d.maxColumns
                 << " compTilesPerCol=" << d.compTilesPerCol
                 << " shimRow=" << d.shimRow
                 << " compTileFirstRow=" << d.compTileFirstRow
                 << " memTileMemBytes=" << d.memTileMemBytes
                 << " pktHdrBytes=" << d.pktHdrBytes << "\n";
    llvm::dbgs() << "\n";
  }

  return info;
}

//===----------------------------------------------------------------------===//
// findOptimalTileParam
//===----------------------------------------------------------------------===//
// TODO(algorithm): Implement the full spatio-temporal tiling optimization here.
// During development and validation, tiling parameters are loaded from an
// external JSON file (--tile-param-json) produced by cost_model.py, which
// enumerates all valid (SPm,SPn,TPm,TPk,TPn) candidates, applies the cost
// model, and selects the optimal configuration. Once the cost model is
// validated against real hardware, this function will perform the full
// search and selection internally, using sysInfo (hardware constraints) and
// opInfo (matrix dimensions / element type).
TileParam findOptimalTileParam(const SystemInfo &sysInfo,
                               const MatmulOpInfo &opInfo,
                               const std::string &jsonPath,
                               bool debug) {
  // sysInfo is unused until the optimization algorithm is implemented.
  (void)sysInfo;

  if (jsonPath.empty()) {
    llvm::errs() << "Error: tile-param-json path is required\n";
    llvm::report_fatal_error("tile param json path not specified");
  }

  auto bufOrErr = llvm::MemoryBuffer::getFile(jsonPath);
  if (!bufOrErr) {
    llvm::errs() << "Error: cannot open JSON '" << jsonPath
                 << "' (" << bufOrErr.getError().message() << ")\n";
    llvm::report_fatal_error("tile param json open failed");
  }
  llvm::StringRef jsonText = (*bufOrErr)->getBuffer();

  auto jsonOrErr = llvm::json::parse(jsonText);
  if (!jsonOrErr) {
    llvm::errs() << "Error: JSON parse failed in '" << jsonPath << "'\n";
    llvm::report_fatal_error("tile param json parse failed");
  }
  auto *rootObj = jsonOrErr->getAsObject();
  if (!rootObj) {
    llvm::errs() << "Error: root JSON is not an object\n";
    llvm::report_fatal_error("tile param json root type error");
  }

  // Flat schema: all tiling fields at root level (no levels[] nesting).
  TileParam tp;
  tp.opSize = TileSize{
      .TM = getU32Req(*rootObj, "M"),
      .TK = getU32Req(*rootObj, "K"),
      .TN = getU32Req(*rootObj, "N"),
  };
  tp.elemType            = getElemTypeReq(*rootObj, "elemType", *opInfo.elemType.getContext());
  tp.numCores            = getU32Req(*rootObj, "numCores");
  tp.doubleBufferEnabled = getBoolReq(*rootObj, "doubleBuffer");

  tp.SPm  = getU32Req(*rootObj, "SPm");
  tp.SPn  = getU32Req(*rootObj, "SPn");
  tp.TPm  = getU32Req(*rootObj, "TPm");
  tp.TPk  = getU32Req(*rootObj, "TPk");
  tp.TPn  = getU32Req(*rootObj, "TPn");

  tp.tileSize = getLevelTileSize(*rootObj);
  tp.tpOrder  = getAxisArrayOpt(*rootObj, "tpOrder");

  if (debug) {
    llvm::dbgs() << "[TileParam] Loaded TileParam:\n"
                << "  opSize:       "
                << "M=" << tp.opSize.TM
                << " K=" << tp.opSize.TK
                << " N=" << tp.opSize.TN << "\n";

    llvm::dbgs() << "  elemType:     ";
    tp.elemType.print(llvm::dbgs());
    llvm::dbgs() << "\n";

    llvm::dbgs() << "  numCores:     " << tp.numCores << "\n"
                << "  doubleBuffer: " << (tp.doubleBufferEnabled ? "true" : "false") << "\n"
                << "  SPm=" << tp.SPm
                << " SPn=" << tp.SPn << " |"
                << " TPm=" << tp.TPm
                << " TPk=" << tp.TPk
                << " TPn=" << tp.TPn << " |"
                << " tile(M/K/N)="
                << tp.tileSize.TM << "/"
                << tp.tileSize.TK << "/"
                << tp.tileSize.TN << " |"
                << " tpOrder=";

    if (tp.tpOrder.empty()) {
      llvm::dbgs() << "[]\n";
    } else {
      llvm::dbgs() << "[";
      for (size_t j = 0; j < tp.tpOrder.size(); ++j) {
        llvm::dbgs() << tp.tpOrder[j];
        if (j + 1 < tp.tpOrder.size()) llvm::dbgs() << ",";
      }
      llvm::dbgs() << "]\n";
    }
    llvm::dbgs() << "\n";
  }

  return tp;
}

//===----------------------------------------------------------------------===//
// buildTilingContext / needsPres / getElemBytes
//===----------------------------------------------------------------------===//
TilingContext buildTilingContext(const TileParam &tp, const SystemInfo &sysInfo) {
  TilingContext tc;
  tc.device = sysInfo.device;
  tc.numCompTilesPerCol = tc.device.compTilesPerCol;

  // numCores must be a positive multiple of compTilesPerCol so that
  // the tile grid fills complete columns without remainder.
  if (tp.numCores == 0 || tp.numCores % tc.device.compTilesPerCol != 0)
    llvm::report_fatal_error(
        llvm::Twine("numCores=") + llvm::Twine(tp.numCores) +
        " must be a positive multiple of " +
        llvm::Twine(tc.device.compTilesPerCol));

  tc.numCols = tp.numCores / tc.numCompTilesPerCol;

  if (tc.numCols > tc.device.maxColumns)
    llvm::report_fatal_error(
        llvm::Twine("numCols=") + llvm::Twine(tc.numCols) +
        " exceeds maximum supported columns (" +
        llvm::Twine(tc.device.maxColumns) + ")");
  tc.compTM              = tp.tileSize.TM;
  tc.compTK              = tp.tileSize.TK;
  tc.compTN              = tp.tileSize.TN;
  tc.compTileSPm         = tp.SPm;
  tc.compTileSPn         = tp.SPn;
  tc.compTileTPm         = tp.TPm;
  tc.compTileTPk         = tp.TPk;
  tc.compTileTPn         = tp.TPn;
  tc.tpOrder             = tp.tpOrder;
  tc.elemType            = tp.elemType;
  tc.doubleBufferEnabled = tp.doubleBufferEnabled;
  return tc;
}

bool needsPres(const TilingContext &ctx) {
  return ctx.compTileTPk > 1 && ctx.tpOrder[0] != AXIS_K;
}

uint32_t getElemBytes(mlir::Type t) {
  if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(t))
    t = shaped.getElementType();
  unsigned bits = t.getIntOrFloatBitWidth();
  return bits / 8;
}

} // namespace onnx_to_aie
