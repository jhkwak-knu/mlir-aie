//===- TileParam.h - Tiling parameter types and loaders ---------*- C++ -*-===//
//
// Data structures for hardware system info, tiling parameters, and tiling
// context.  JSON loading functions for SystemInfo and TileParam.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_CONVERSION_ONNXTOAIE_TILEPARAM_H
#define ONNX_CONVERSION_ONNXTOAIE_TILEPARAM_H

#include "mlir/IR/Types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace onnx_to_aie {

//===----------------------------------------------------------------------===//
// Device configuration (loaded from JSON, with XDNA2 defaults)
//===----------------------------------------------------------------------===//
struct DeviceConfig {
  uint32_t maxColumns       = 8; // max AIE columns (npu2 device family)
  uint32_t compTilesPerCol  = 4; // compute tiles per column (rows 2-5 on XDNA2)
  uint32_t shimRow          = 0; // shim tile row
  uint32_t compTileFirstRow = 2; // first compute tile row
  uint32_t memTileMemBytes  = 524288; // 512KB mem tile capacity (future use)
  uint32_t pktHdrBytes      = 4; // DMA switch packet header size in bytes

  uint32_t compTileLastRow() const {
    return compTileFirstRow + compTilesPerCol - 1;
  }
};

//===----------------------------------------------------------------------===//
// Layout constants (device-independent)
//===----------------------------------------------------------------------===//
// Packet IDs use power-of-2 assignment: (1u << tileIndex) per type.
// Each ID has exactly one bit set, so the pathfinder can distinguish
// individual flows with a single-bit mask — minimising arbiter and
// msel usage in the AIE2 switchbox (6 arbiters, 4 msels each).
// LHS/RHS/RES share the same ID space {1,2,4,8} on disjoint DMA
// channels / directions, so they never collide in the switch fabric.
// PRES adds offset 16 (bit4) to separate from LHS on a shared channel.
static constexpr uint32_t PRES_PKT_ID_OFFSET     = 16;
// Upper bound that makes an SCF ForOp behave as an infinite loop in the core
static constexpr int64_t  CORE_LOOP_INFINITE     = 0x7FFFFFFFFFFFFFFFLL;

// Axis indices used in tpOrder to select the innermost temporal loop axis.
// tpOrder[0] determines which axis is reused across iterations:
//   AXIS_M (0) -> RHS reuse, AXIS_N (1) -> LHS reuse, AXIS_K (2) -> local accumulation.
enum AxisId : uint32_t { AXIS_M = 0, AXIS_N = 1, AXIS_K = 2 };

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
  DeviceConfig device; // defaults used when JSON lacks "device" section
};

std::optional<SystemInfo> loadSystemInfo(const std::string &filePath,
                                         bool debug = false);

//===----------------------------------------------------------------------===//
// Tiling parameters
//===----------------------------------------------------------------------===//
struct MatmulOpInfo {
  uint32_t M, K, N;
  mlir::Type elemType;
};

struct TileSize {
  uint32_t TM, TK, TN;
};

/// Per-level tiling parameters.  Currently only level 0 (DRAM <-> compute
/// tile) is used; the array structure mirrors xdna2_info.json's spm_levels[]
/// so that future mem-tile (L2) support can be added without schema changes.
struct LevelParam {
  uint32_t SPm, SPn;
  uint32_t TPm, TPk, TPn;
  TileSize tileSize;
  std::vector<uint32_t> tpOrder;
};

struct TileParam {
  TileSize opSize;
  mlir::Type elemType;
  uint32_t numCores;
  bool doubleBufferEnabled;
  bool traceEnabled = false;
  /// One entry per memory-hierarchy level.  Only single-level (levels[0])
  /// is supported; the parser rejects inputs with levels.size() != 1.
  std::vector<LevelParam> levels;
};

TileParam findOptimalTileParam(const SystemInfo &sysInfo,
                               const MatmulOpInfo &opInfo,
                               const std::string &jsonPath,
                               bool debug = false);

//===----------------------------------------------------------------------===//
// Tiling context (derived values used throughout placement and emission)
//===----------------------------------------------------------------------===//
struct TilingContext {
  DeviceConfig device;
  uint32_t numCols;
  uint32_t numCompTilesPerCol;
  uint32_t compTM, compTK, compTN;
  uint32_t compTileSPm, compTileSPn;
  uint32_t compTileTPm, compTileTPk, compTileTPn;
  std::vector<uint32_t> tpOrder;
  mlir::Type elemType;
  bool doubleBufferEnabled;
  bool traceEnabled = false;
};

TilingContext buildTilingContext(const TileParam &tp, const SystemInfo &sysInfo);

/// Returns true when partial-sum (pres) input buffers are needed.
/// This happens when K is split temporally (TPk > 1) AND K is not
/// the innermost loop axis (which would accumulate locally).
bool needsPres(const TilingContext &ctx);

/// Element byte size for a given MLIR type (e.g. f32 -> 4).
uint32_t getElemBytes(mlir::Type t);

} // namespace onnx_to_aie

#endif // ONNX_CONVERSION_ONNXTOAIE_TILEPARAM_H
