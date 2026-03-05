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
// Layout constants
//===----------------------------------------------------------------------===//
// Number of compute tiles stacked per column (rows 2-5 on XDNA2)
static constexpr uint32_t NUM_COMP_TILES_PER_COL = 4;
// Maximum number of AIE columns supported by the npu2 device family.
// Corresponds to the size of the devices[] array in emitDeviceOp.
static constexpr uint32_t NUM_MAX_COLS           = 8;
// 4-byte header inserted by the DMA switch before each outgoing packet payload.
static constexpr uint32_t PKT_HDR_BYTES          = 4;
// pres packet IDs start above lhs/rhs range to avoid packet-filter collisions
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

struct LevelParam {
  uint32_t numSpm;
  uint32_t SPm, SPn;
  uint32_t TPm, TPk, TPn;
  TileSize tileSize;
  std::vector<uint32_t> tpOrder;
};

struct TileParam {
  TileSize opSize;
  mlir::Type elemType;
  uint32_t numLevel;
  uint32_t numLastSpm;
  bool doubleBufferEnabled;
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
  uint32_t numCols;
  uint32_t numCompTilesPerCol;
  uint32_t compTM, compTK, compTN;
  uint32_t compTileSPm, compTileSPn;
  uint32_t compTileTPm, compTileTPk, compTileTPn;
  std::vector<uint32_t> tpOrder;
  mlir::Type elemType;
  bool doubleBufferEnabled;
};

TilingContext buildTilingContext(const TileParam &tp);

/// Returns true when partial-sum (pres) input buffers are needed.
/// This happens when K is split temporally (TPk > 1) AND K is not
/// the innermost loop axis (which would accumulate locally).
bool needsPres(const TilingContext &ctx);

/// Element byte size for a given MLIR type (e.g. f32 -> 4).
uint32_t getElemBytes(mlir::Type t);

} // namespace onnx_to_aie

#endif // ONNX_CONVERSION_ONNXTOAIE_TILEPARAM_H
