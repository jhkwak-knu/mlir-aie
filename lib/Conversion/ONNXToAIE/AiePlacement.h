//===- AiePlacement.h - AIE tile/buffer/DMA structs + placement --*- C++ -*-===//
//
// Hardware abstraction data structures for AIE tiles, buffers, DMAs, and
// communication paths.  Placement and schedule optimisation entry points.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_CONVERSION_ONNXTOAIE_AIEPLACEMENT_H
#define ONNX_CONVERSION_ONNXTOAIE_AIEPLACEMENT_H

#include "TileParam.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onnx_to_aie {

using xilinx::AIE::WireBundle;
using xilinx::AIE::DMAChannelDir;

//===----------------------------------------------------------------------===//
// Hardware abstraction structs
//===----------------------------------------------------------------------===//
struct AieBuf {
  std::string name;
  std::string symbol;
  uint32_t bufSize;
  mlir::Type elemType;
  mlir::Value bufValue;
  mlir::Value prodLockValue;
  mlir::Value consLockValue;
  mlir::Value calcLockValue;  // extra lock for res buffer when pres forwarding is active
  int32_t prodLockInit = -1;
  int32_t consLockInit = -1;
};

struct AieBufferDescriptor {
  std::string name;
  bool isPacket = false;
  uint32_t packetId = 0;
  uint32_t bufIdx = 0;
  uint32_t bufOffset = 0;
  uint32_t bufSize = 0;
  uint32_t nextBdIdx = 0;
  // Set the buffer reference to point at the same AieBuf identified by bufIdx.
  void setBufInfo(const std::vector<AieBuf> &bufs) {
    if (bufIdx < bufs.size()) {
      bufSize       = bufs[bufIdx].bufSize;
      bufOffset     = 0;
    }
  }
};

struct AieDma {
  DMAChannelDir dir;
  uint32_t channel;
  uint32_t bdIdx = 0;
};

struct AieTile {
  uint32_t col, row;
  std::string type;
  std::vector<AieBuf> bufs;
  std::vector<AieBufferDescriptor> bds;
  std::vector<AieDma> dmas;
  mlir::Value value;

  AieBuf& getAieBuf(uint32_t idx) { return bufs[idx]; }
  AieBufferDescriptor& getAieBd(uint32_t idx) { return bds[idx]; }
  AieDma& getAieDma(uint32_t idx) { return dmas[idx]; }

  const AieBuf& getAieBuf(uint32_t idx) const { return bufs[idx]; }
  const AieBufferDescriptor& getAieBd(uint32_t idx) const { return bds[idx]; }
  const AieDma& getAieDma(uint32_t idx) const { return dmas[idx]; }

  int findBufIdx(const std::string &n) const {
    for (size_t i = 0; i < bufs.size(); ++i)
      if (bufs[i].name == n) return static_cast<int>(i);
    return -1;
  }

  std::pair<bool, uint32_t> findDmaIdx(DMAChannelDir d, uint32_t ch) const {
    for (size_t i = 0; i < dmas.size(); ++i)
      if (dmas[i].dir == d && dmas[i].channel == ch)
        return {true, static_cast<uint32_t>(i)};
    return {false, 0};
  }
};

struct AieCircuit {
  std::string name;
  uint32_t size;
  mlir::Type elemType;
  std::vector<uint32_t> dstIdxs;
  std::vector<WireBundle> dstBundles;
  std::vector<uint32_t> dstChs;
};

struct AiePacket {
  std::string name;
  uint32_t packetId;
  uint32_t size;
  mlir::Type elemType;
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

struct AieNpuWait {
  uint32_t col, row;
  uint32_t bufIdx;
};

struct AieNpuMemcpyNd {
  std::string name;
  uint32_t id;
  uint32_t shimCol, shimRow;
  uint32_t bufIdx;
  bool isPacket;
  uint32_t packetType;
  uint32_t packetId;
  bool issueToken;
  std::vector<AieNpuWait> waitBufs;
  std::array<int64_t,4> staticOffset;
  std::array<int64_t,4> staticSize;
  std::array<int64_t,4> staticStride;
};

struct ShimBdSchedules {
  std::vector<AieNpuMemcpyNd> lhsTx, rhsTx, presTx, resRx;
};

//===----------------------------------------------------------------------===//
// AiePlacement
//===----------------------------------------------------------------------===//
struct AiePlacement {
  std::vector<AieTile> aieTiles;
  std::vector<AieComm> aieComms;
  std::vector<AieNpuMemcpyNd> aieSchedule;
  // Maps (col, row) -> index into aieTiles for O(1) lookup.
  std::map<std::pair<uint32_t,uint32_t>, uint32_t> tileIdxMap;

  AieTile& getAieTile(uint32_t idx) { return aieTiles[idx]; }
  AieComm& getAieComm(uint32_t idx) { return aieComms[idx]; }

  const AieTile& getAieTile(uint32_t idx) const { return aieTiles[idx]; }
  const AieComm& getAieComm(uint32_t idx) const { return aieComms[idx]; }

  uint32_t findAieTileIdx(uint32_t col, uint32_t row) const {
    auto it = tileIdxMap.find({col, row});
    if (it == tileIdxMap.end())
      llvm::report_fatal_error(llvm::Twine("tile not found at col=") +
                               llvm::Twine(col) + " row=" + llvm::Twine(row));
    return it->second;
  }
};

//===----------------------------------------------------------------------===//
// Placement + emission entry points
//===----------------------------------------------------------------------===//
AiePlacement optimizeAiePlacement(const TilingContext &tilingCtx,
                                  bool debug = false);

void generateAieOps(mlir::ConversionPatternRewriter &rewriter,
                    AiePlacement &placement,
                    const TilingContext &tilingCtx,
                    const TileParam &tileParam,
                    const std::string &outputPath);

} // namespace onnx_to_aie

#endif // ONNX_CONVERSION_ONNXTOAIE_AIEPLACEMENT_H
