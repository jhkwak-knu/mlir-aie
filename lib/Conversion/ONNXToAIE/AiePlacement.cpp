//===- AiePlacement.cpp - AIE tile placement and schedule --------*- C++ -*-===//
//
// Places tiles, allocates buffers, configures DMA communications, and builds
// the NPU runtime schedule for a given TilingContext.
//
//===----------------------------------------------------------------------===//

#include "AiePlacement.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

using namespace mlir;

namespace onnx_to_aie {

//===----------------------------------------------------------------------===//
// placeTiles
//===----------------------------------------------------------------------===//
// Place shim and compute tiles in a column-major grid.
// Each column has 1 shim tile and compTilesPerCol compute tiles (highest row first).
static void placeTiles(AiePlacement &placement, const TilingContext &tilingCtx) {
  const auto &numCols            = tilingCtx.numCols;
  const auto &numCompTilesPerCol = tilingCtx.numCompTilesPerCol;
  const auto &device             = tilingCtx.device;

  for (uint32_t i = 0; i < numCols; ++i) {
    AieTile shimTile{.col=i, .row=device.shimRow};
    placement.aieTiles.push_back(shimTile);
    placement.tileIdxMap[{shimTile.col, shimTile.row}] = placement.aieTiles.size() - 1;

    for (uint32_t j = 0; j < numCompTilesPerCol; ++j) {
      AieTile compTile{.col=i, .row=(device.compTileLastRow() - j)};
      placement.aieTiles.push_back(compTile);
      placement.tileIdxMap[{compTile.col, compTile.row}] = placement.aieTiles.size() - 1;
    }
  }
}

//===----------------------------------------------------------------------===//
// allocateBuffers
//===----------------------------------------------------------------------===//
// Allocate lhs/rhs/res (and optionally pres) buffers for each tile.
// Shim tiles include a packet-header overhead in the res buffer.
// pres buffer is needed when partial sums must be forwarded between tiles:
//   TPk>1 and K is not the innermost (reuse) axis (tpOrder[0] != AXIS_K).
static void allocateBuffers(AiePlacement &placement, const TilingContext &tilingCtx) {
  const auto &compTM   = tilingCtx.compTM;
  const auto &compTK   = tilingCtx.compTK;
  const auto &compTN   = tilingCtx.compTN;
  const auto &elemType = tilingCtx.elemType;
  const auto &device   = tilingCtx.device;

  for (auto &tile : placement.aieTiles) {
    if (tile.row == device.shimRow) { // Shim tile
      uint32_t lhsBufSize = compTM * compTK;
      uint32_t rhsBufSize = compTK * compTN;
      uint32_t resBufSize = compTM * compTN + (device.pktHdrBytes / getElemBytes(elemType)); // header overhead in elements

      AieBuf lhsBuf{.name="lhs", .bufSize=lhsBufSize, .elemType=elemType};
      AieBuf rhsBuf{.name="rhs", .bufSize=rhsBufSize, .elemType=elemType};
      AieBuf resBuf{.name="res", .bufSize=resBufSize, .elemType=elemType};

      tile.bufs.push_back(lhsBuf);
      tile.bufs.push_back(rhsBuf);
      tile.bufs.push_back(resBuf);

      if (needsPres(tilingCtx)) {
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
}

//===----------------------------------------------------------------------===//
// configureInputComms
//===----------------------------------------------------------------------===//
// Configure shim->comp packet communication paths for lhs, rhs, and pres data.
// pres packets share the same DMA channel as lhs (M-axis) or rhs (N-axis),
// depending on which axis is innermost (tpOrder[0]).
static void configureInputComms(AiePlacement &placement, const TilingContext &tilingCtx) {
  const auto &numCols            = tilingCtx.numCols;
  const auto &numCompTilesPerCol = tilingCtx.numCompTilesPerCol;
  const auto &compTM             = tilingCtx.compTM;
  const auto &compTK             = tilingCtx.compTK;
  const auto &compTN             = tilingCtx.compTN;
  const auto &compTileSPm        = tilingCtx.compTileSPm;
  const auto &tpOrder            = tilingCtx.tpOrder;
  const auto &elemType           = tilingCtx.elemType;
  const auto &device             = tilingCtx.device;
  auto dmaWireBundle = WireBundle::DMA;

  // 1. Shim tile -> Comp tile (input)
  for (uint32_t col = 0; col < numCols; ++col) {
    // Generate input (LHS/RHS) communications for each Shim tile
    uint32_t shimIdx = placement.findAieTileIdx(col, device.shimRow);
    AieComm lhsComm{.name="lhs", .srcIdx=shimIdx, .srcBundle=dmaWireBundle, .srcCh=0, .isPacket=true};
    AieComm rhsComm{.name="rhs", .srcIdx=shimIdx, .srcBundle=dmaWireBundle, .srcCh=1, .isPacket=true};

    // Find the input (LHS/RHS) packets required by the compute tiles in this column.
    // Packet IDs use destination-based bitmask: each tile at position i in the
    // column sets bit i.  Multicast packets OR the bits of all destination tiles.
    // This guarantees that at every intermediate switchbox the pathfinder's
    // 5-bit mask/value routing can distinguish pass-through flows from delivery
    // flows, because delivery tiles have their position bit set while
    // pass-through flows for other tiles do not.
    llvm::MapVector<uint32_t, AiePacket> lhsPackets;
    llvm::MapVector<uint32_t, AiePacket> rhsPackets;

    for (uint32_t i = 0; i < numCompTilesPerCol; ++i) {
      uint32_t l_idx = numCompTilesPerCol * col + i;
      uint32_t m_idx = l_idx % compTileSPm;
      uint32_t n_idx = l_idx / compTileSPm;

      uint32_t compIdx = placement.findAieTileIdx(col, device.compTileLastRow() - i);

      { // LHS packets
        auto it = lhsPackets.find(m_idx);
        if (it == lhsPackets.end()) {
          AiePacket pkt;
          pkt.name     = std::string("lhs") + std::to_string(m_idx);
          pkt.packetId = (1u << i);
          pkt.size     = compTM * compTK;
          pkt.elemType = elemType;

          pkt.dstIdxs.push_back(compIdx);
          pkt.dstBundles.push_back(dmaWireBundle);
          pkt.dstChs.push_back(0);

          lhsPackets.insert({m_idx, std::move(pkt)});
        } else {
          it->second.packetId |= (1u << i);
          it->second.dstIdxs.push_back(compIdx);
          it->second.dstBundles.push_back(dmaWireBundle);
          it->second.dstChs.push_back(0);
        }
      }

      { // RHS packets
        auto it = rhsPackets.find(n_idx);
        if (it == rhsPackets.end()) {
          AiePacket pkt;
          pkt.name     = std::string("rhs") + std::to_string(n_idx);
          pkt.packetId = (1u << i);
          pkt.size     = compTK * compTN;
          pkt.elemType = elemType;

          pkt.dstIdxs.push_back(compIdx);
          pkt.dstBundles.push_back(dmaWireBundle);
          pkt.dstChs.push_back(1);

          rhsPackets.insert({n_idx, std::move(pkt)});
        } else {
          it->second.packetId |= (1u << i);
          it->second.dstIdxs.push_back(compIdx);
          it->second.dstBundles.push_back(dmaWireBundle);
          it->second.dstChs.push_back(1);
        }
      }
    }

    // Register LHS packets to the LHS communication
    for (auto &kv : lhsPackets) {
      lhsComm.packets.push_back(std::move(kv.second));
    }

    // Register RHS packets to the RHS communication
    for (auto &kv : rhsPackets) {
      rhsComm.packets.push_back(std::move(kv.second));
    }

    // Register PRES packets when partial sums must be forwarded between K iterations.
    // needsPres() guarantees tpOrder[0] is 0 (M) or 1 (N), never 2 (K).
    if (needsPres(tilingCtx)) {
      AieComm *presComm;
      uint32_t dstCh;

      if (tpOrder[0] == AXIS_M) { // M-axis innermost: share LHS DMA channel
        presComm = &lhsComm;
        dstCh = 0;
      } else { // N-axis innermost (tpOrder[0] == AXIS_N): share RHS DMA channel
        presComm = &rhsComm;
        dstCh = 1;
      }

      uint32_t presPacketCnt = 0;
      for (uint32_t i = 0; i < numCompTilesPerCol; ++i) {
        uint32_t l_idx = numCompTilesPerCol * col + i;
        uint32_t compIdx = placement.findAieTileIdx(col, device.compTileLastRow() - i);

        AiePacket pkt;
        pkt.name = std::string("pres") + std::to_string(l_idx);
        pkt.packetId = (1u << presPacketCnt++) + PRES_PKT_ID_OFFSET;
        pkt.size = compTM * compTN;
        pkt.elemType = elemType;

        pkt.dstIdxs.push_back(compIdx);
        pkt.dstBundles.push_back(dmaWireBundle);
        pkt.dstChs.push_back(dstCh);

        presComm->packets.push_back(pkt);
      }
    }

    placement.aieComms.push_back(lhsComm);
    placement.aieComms.push_back(rhsComm);
  }
}

//===----------------------------------------------------------------------===//
// configureOutputComms
//===----------------------------------------------------------------------===//
// Configure comp->shim packet communication paths for res data.
// Each compute tile sends its result to the shim tile in the same column.
static void configureOutputComms(AiePlacement &placement, const TilingContext &tilingCtx) {
  const auto &numCols            = tilingCtx.numCols;
  const auto &numCompTilesPerCol = tilingCtx.numCompTilesPerCol;
  const auto &compTM             = tilingCtx.compTM;
  const auto &compTN             = tilingCtx.compTN;
  const auto &elemType           = tilingCtx.elemType;
  const auto &device             = tilingCtx.device;
  auto dmaWireBundle = WireBundle::DMA;

  // 2. Shim tile <- Comp tile (output)
  for (uint32_t col = 0; col < numCols; ++col) {
    uint32_t shimIdx = placement.findAieTileIdx(col, device.shimRow);
    uint32_t resPacketCnt = 0;

    // Generate and register RES communications for each Shim tile
    for (uint32_t i = 0; i < numCompTilesPerCol; ++i) {
      uint32_t l_idx = numCompTilesPerCol * col + i;
      uint32_t compIdx = placement.findAieTileIdx(col, device.compTileLastRow() - i);

      AiePacket pkt;
      pkt.name = std::string("res") + std::to_string(l_idx);
      pkt.packetId = (1u << resPacketCnt++);
      pkt.size = compTM * compTN;
      pkt.elemType = elemType;

      pkt.dstIdxs.push_back(shimIdx);
      pkt.dstBundles.push_back(dmaWireBundle);
      pkt.dstChs.push_back(0);

      AieComm resComm{.name="res", .srcIdx=compIdx, .srcBundle=dmaWireBundle, .srcCh=0, .isPacket=true};
      resComm.packets.push_back(pkt);

      placement.aieComms.push_back(resComm);
    }
  }
}

//===----------------------------------------------------------------------===//
// configureDmas
//===----------------------------------------------------------------------===//
// Configure MM2S (send) and S2MM (receive) DMAs and BD chains for all comms.
// S2MM BDs on shim tiles add PKT_HDR_BYTES overhead to account for the packet header.
static void configureDmas(AiePlacement &placement, const TilingContext &tilingCtx) {
  const auto &elemType = tilingCtx.elemType;
  const auto &device   = tilingCtx.device;

  // 1. MM2S (Send)
  for (auto &comm : placement.aieComms) {
    auto &srcTile = placement.getAieTile(comm.srcIdx);
    auto [hasSrcDma, srcDmaIdx] = srcTile.findDmaIdx(DMAChannelDir::MM2S, comm.srcCh);
    bool hasLastBd;
    uint32_t lastBdIdx;

    // Configure DMAs
    if (hasSrcDma) { // DMA configuration already exists
      auto &srcDma = srcTile.getAieDma(srcDmaIdx);
      uint32_t firstBdIdx, curBdIdx, nextBdIdx;

      firstBdIdx = srcDma.bdIdx;
      curBdIdx = firstBdIdx;
      nextBdIdx = srcTile.getAieBd(curBdIdx).nextBdIdx;

      while (nextBdIdx != firstBdIdx) {
        curBdIdx = nextBdIdx;
        nextBdIdx = srcTile.getAieBd(curBdIdx).nextBdIdx;
      }

      hasLastBd = true;
      lastBdIdx = curBdIdx;
    } else { // DMA configuration does not exist
      AieDma srcDma{.dir=DMAChannelDir::MM2S, .channel=comm.srcCh};
      srcTile.dmas.push_back(srcDma);

      srcDmaIdx = srcTile.dmas.size() - 1;
      hasLastBd = false;
    }

    // Configure BDs
    if (comm.isPacket) { // packet-switched communication
      for (auto &packet : comm.packets) {
        auto &srcDma = srcTile.getAieDma(srcDmaIdx);

        AieBufferDescriptor bd{.name=packet.name, .isPacket=true, .packetId=packet.packetId};
        // Resolve buffer index: packet names like "lhs0" map to buf "lhs".
        // pres packets are appended to lhs/rhs comms but need the "pres" buffer.
        std::string bufName = (packet.name.compare(0, 4, "pres") == 0)
                                  ? "pres" : comm.name;
        bd.bufIdx = srcTile.findBufIdx(bufName);
        bd.bufSize = packet.size;
        bd.bufOffset = 0;
        bd.nextBdIdx = srcDma.bdIdx;
        srcTile.bds.push_back(bd);

        if (hasLastBd) {
          auto &lastBd = srcTile.getAieBd(lastBdIdx);
          uint32_t next = srcTile.bds.size() - 1;

          lastBd.nextBdIdx = next;
          lastBdIdx = next;
        } else {
          uint32_t next = srcTile.bds.size() - 1;
          auto &newBd = srcTile.getAieBd(next);

          srcDma.bdIdx = next;
          newBd.nextBdIdx = next;

          hasLastBd = true;
          lastBdIdx = next;
        }
      }
    } else { // circuit-switched communication
      // TODO: implement
    }
  }

  // 2. S2MM (Receive)
  for (auto &comm : placement.aieComms) {
    if (comm.isPacket) { // packet-switched communication
      for (auto &packet : comm.packets) {
        for (uint32_t i = 0; i < packet.dstIdxs.size(); ++i) {
          uint32_t dstIdx = packet.dstIdxs[i];
          uint32_t dstCh = packet.dstChs[i];

          auto &dstTile = placement.getAieTile(dstIdx);
          auto [hasDstDma, dstDmaIdx] = dstTile.findDmaIdx(DMAChannelDir::S2MM, dstCh);
          bool hasLastBd;
          uint32_t lastBdIdx;

          if (hasDstDma) { // DMA configuration already exists
            auto &dstDma = dstTile.getAieDma(dstDmaIdx);
            uint32_t firstBdIdx, curBdIdx, nextBdIdx;

            firstBdIdx = dstDma.bdIdx;
            curBdIdx = firstBdIdx;
            nextBdIdx = dstTile.getAieBd(curBdIdx).nextBdIdx;

            while (nextBdIdx != firstBdIdx) {
              curBdIdx = nextBdIdx;
              nextBdIdx = dstTile.getAieBd(curBdIdx).nextBdIdx;
            }

            hasLastBd = true;
            lastBdIdx = curBdIdx;
          } else { // DMA configuration does not exist
            AieDma dstDma{.dir=DMAChannelDir::S2MM, .channel=dstCh};
            dstTile.dmas.push_back(dstDma);

            dstDmaIdx = dstTile.dmas.size() - 1;
            hasLastBd = false;
          }

          auto &dstDma = dstTile.getAieDma(dstDmaIdx);
          AieBufferDescriptor bd{.name=packet.name, .isPacket=true, .packetId=packet.packetId};
          // Resolve buffer index: packet names like "lhs0" map to buf "lhs".
          // pres packets need special handling: shim tiles have a dedicated
          // "pres" buffer, but comp tiles receive pres (partial-sum) data
          // into the "res" buffer which serves as the accumulator input.
          std::string bufName;
          if (packet.name.compare(0, 4, "pres") == 0) {
            bufName = (dstTile.row == device.shimRow) ? "pres" : "res";
          } else {
            bufName = comm.name;
          }
          bd.bufIdx = dstTile.findBufIdx(bufName);
          // shim tile receives the DMA switch header alongside the payload; add header overhead in elements
          bd.bufSize = (dstTile.row == device.shimRow) ? (packet.size + (device.pktHdrBytes / getElemBytes(elemType))) : packet.size;
          bd.bufOffset = 0;
          bd.nextBdIdx = dstDma.bdIdx;
          dstTile.bds.push_back(bd);

          if (hasLastBd) {
            auto &lastBd = dstTile.getAieBd(lastBdIdx);
            uint32_t next = dstTile.bds.size() - 1;

            lastBd.nextBdIdx = next;
            lastBdIdx = next;
          } else {
            uint32_t next = dstTile.bds.size() - 1;
            auto &newBd = dstTile.getAieBd(next);

            dstDma.bdIdx = next;
            newBd.nextBdIdx = next;

            hasLastBd = true;
            lastBdIdx = next;
          }
        }
      }
    } else { // circuit-switched communication
      // TODO: implement
    }
  }
}

//===----------------------------------------------------------------------===//
// Schedule helpers
//===----------------------------------------------------------------------===//
// Group DMA entries that transfer the same data (same offset and size) so that
// only the last entry in each group carries a merged wait-list.  This lets the
// runtime issue parallel transfers on separate channels and wait once.
static void syncChannelParallelSameData(std::vector<AieNpuMemcpyNd> &v) {
  auto sameData = [](const AieNpuMemcpyNd &a, const AieNpuMemcpyNd &b) {
    return a.staticOffset[3] == b.staticOffset[3] &&
           a.staticSize[3]   == b.staticSize[3];
  };

  const size_t n = v.size();
  std::vector<char> moved(n, 0);
  std::vector<AieNpuMemcpyNd> out;
  out.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    if (moved[i]) continue;

    std::vector<size_t> group{i};
    moved[i] = 1;
    for (size_t j = i + 1; j < n; ++j)
      if (!moved[j] && sameData(v[i], v[j])) { group.push_back(j); moved[j] = 1; }

    std::vector<AieNpuWait> mergedWait;
    for (size_t k = 0; k < group.size(); ++k) {
      auto item = v[group[k]];
      mergedWait.insert(mergedWait.end(), item.waitBufs.begin(), item.waitBufs.end());
      item.waitBufs.clear();

      if (k == (group.size() - 1)) {
        item.waitBufs = mergedWait;
      }

      out.push_back(std::move(item));
    }
  }

  v.swap(out);
}

// Collect per-buffer DMA schedules from shim tiles by walking each DMA's
// BD chain and categorising entries as lhs/rhs/pres TX or res RX.
static ShimBdSchedules collectShimBdSchedules(
    const AiePlacement &placement, const TilingContext &tilingCtx) {
  const auto &compTM   = tilingCtx.compTM;
  const auto &compTK   = tilingCtx.compTK;
  const auto &compTN   = tilingCtx.compTN;
  const auto &elemType = tilingCtx.elemType;
  const auto &device   = tilingCtx.device;

  ShimBdSchedules sched;

  for (auto &tile : placement.aieTiles) {
    if (tile.row != device.shimRow) continue; // only shim tiles

    for (auto &dma : tile.dmas) {
      if (dma.dir == DMAChannelDir::MM2S) {
        uint32_t firstBdIdx = dma.bdIdx;
        uint32_t curBdIdx = firstBdIdx;

        do {
          auto &bd = tile.getAieBd(curBdIdx);
          std::array<int64_t,4> defaultSize = {1, 1, 1, static_cast<int64_t>(bd.bufSize)};
          std::array<int64_t,4> defaultStride = {0, 0, 0, 1};

          if (bd.name.compare(0, 3, "lhs") == 0) {
            uint32_t idx = static_cast<uint32_t>(std::strtoul(bd.name.c_str() + 3, nullptr, 10));
            uint32_t off = compTM * compTK;

            std::string name = "lhs";
            std::array<int64_t,4> offset = {0, 0, 0, static_cast<int64_t>(off * idx)};
            AieNpuMemcpyNd lhsTx{.name=name, .id=0, .shimCol=tile.col, .shimRow=tile.row, .bufIdx=bd.bufIdx, .isPacket=bd.isPacket, .packetType=0, .packetId=bd.packetId,
                                  .issueToken=true, .waitBufs={AieNpuWait{tile.col, tile.row, bd.bufIdx}},
                                  .staticOffset=offset, .staticSize=defaultSize, .staticStride=defaultStride};

            sched.lhsTx.push_back(lhsTx);
          } else if (bd.name.compare(0, 3, "rhs") == 0) {
            uint32_t idx = static_cast<uint32_t>(std::strtoul(bd.name.c_str() + 3, nullptr, 10));
            uint32_t off = compTK * compTN;

            std::string name = "rhs";
            std::array<int64_t,4> offset = {0, 0, 0, static_cast<int64_t>(off * idx)};
            AieNpuMemcpyNd rhsTx{.name=name, .id=1, .shimCol=tile.col, .shimRow=tile.row, .bufIdx=bd.bufIdx, .isPacket=bd.isPacket, .packetType=0, .packetId=bd.packetId,
                                  .issueToken=true, .waitBufs={AieNpuWait{tile.col, tile.row, bd.bufIdx}},
                                  .staticOffset=offset, .staticSize=defaultSize, .staticStride=defaultStride};

            sched.rhsTx.push_back(rhsTx);
          } else { // tile.bufs[bd.bufIdx].name == "pres"
            uint32_t idx = static_cast<uint32_t>(std::strtoul(bd.name.c_str() + 4, nullptr, 10));
            uint32_t off = compTM * compTN;

            std::string name = "pres";
            std::array<int64_t,4> offset = {0, 0, 0, static_cast<int64_t>(off * idx)};
            AieNpuMemcpyNd presTx{.name=name, .id=2, .shimCol=tile.col, .shimRow=tile.row, .bufIdx=bd.bufIdx, .isPacket=bd.isPacket, .packetType=0, .packetId=bd.packetId,
                                  .issueToken=true, .waitBufs={AieNpuWait{tile.col, tile.row, bd.bufIdx}},
                                  .staticOffset=offset, .staticSize=defaultSize, .staticStride=defaultStride};

            sched.presTx.push_back(presTx);
          }

          curBdIdx = bd.nextBdIdx;
        } while (curBdIdx != firstBdIdx);
      } else { // dma.dir == DMAChannelDir::S2MM
        uint32_t firstBdIdx = dma.bdIdx;
        uint32_t curBdIdx = firstBdIdx;

        do {
          auto &bd = tile.getAieBd(curBdIdx);
          uint32_t idx = static_cast<uint32_t>(std::strtoul(bd.name.c_str() + 3, nullptr, 10));
          uint32_t off = compTM * compTN + (device.pktHdrBytes / getElemBytes(elemType)); // header overhead in elements

          std::string name = "res";
          std::array<int64_t,4> offset = {0, 0, 0, static_cast<int64_t>(off * idx)};
          std::array<int64_t,4> size = {1, 1, 1, static_cast<int64_t>(bd.bufSize)};
          std::array<int64_t,4> stride = {0, 0, 0, 1};
          AieNpuMemcpyNd resRx{.name=name, .id=3, .shimCol=tile.col, .shimRow=tile.row, .bufIdx=bd.bufIdx, .isPacket=bd.isPacket, .packetType=0, .packetId=bd.packetId,
                                  .issueToken=true, .waitBufs={AieNpuWait{tile.col, tile.row, bd.bufIdx}},
                                  .staticOffset=offset, .staticSize=size, .staticStride=stride};

          // resRxSchedule.insert(resRxSchedule.begin(), resRx);
          sched.resRx.push_back(resRx);

          curBdIdx = bd.nextBdIdx;
        } while (curBdIdx != firstBdIdx);
      }
    }
  }

  return sched;
}

// Assemble the final NPU schedule by interleaving lhs/rhs/pres/res transfers
// according to the innermost loop axis, advancing offsets each iteration.
static void assembleScheduleByAxis(
    AiePlacement &placement, const TilingContext &tilingCtx,
    std::vector<AieNpuMemcpyNd> &lhsTxSchedule,
    std::vector<AieNpuMemcpyNd> &rhsTxSchedule,
    std::vector<AieNpuMemcpyNd> &presTxSchedule,
    std::vector<AieNpuMemcpyNd> &resRxSchedule) {
  const auto &compTM      = tilingCtx.compTM;
  const auto &compTK      = tilingCtx.compTK;
  const auto &compTN      = tilingCtx.compTN;
  const auto &compTileTPm = tilingCtx.compTileTPm;
  const auto &compTileTPk = tilingCtx.compTileTPk;
  const auto &compTileTPn = tilingCtx.compTileTPn;
  const auto &tpOrder     = tilingCtx.tpOrder;
  const auto &elemType    = tilingCtx.elemType;
  const auto &device      = tilingCtx.device;

  const uint32_t lhsTxWaitCnt  = static_cast<uint32_t>(std::count_if(
      lhsTxSchedule.begin(), lhsTxSchedule.end(),
      [](const AieNpuMemcpyNd &s) { return !s.waitBufs.empty(); }));

  const uint32_t rhsTxWaitCnt  = static_cast<uint32_t>(std::count_if(
      rhsTxSchedule.begin(), rhsTxSchedule.end(),
      [](const AieNpuMemcpyNd &s) { return !s.waitBufs.empty(); }));

  const uint32_t presTxWaitCnt = static_cast<uint32_t>(std::count_if(
      presTxSchedule.begin(), presTxSchedule.end(),
      [](const AieNpuMemcpyNd &s) { return !s.waitBufs.empty(); }));

  const uint32_t resRxWaitCnt  = static_cast<uint32_t>(std::count_if(
      resRxSchedule.begin(), resRxSchedule.end(),
      [](const AieNpuMemcpyNd &s) { return !s.waitBufs.empty(); }));

  if (tpOrder[0] == AXIS_M) {
    placement.aieSchedule.insert(placement.aieSchedule.end(), rhsTxSchedule.begin(), rhsTxSchedule.end());

    for (uint32_t i = 0; i < compTileTPm; ++i) {
      placement.aieSchedule.insert(placement.aieSchedule.end(), lhsTxSchedule.begin(), lhsTxSchedule.end());
      placement.aieSchedule.insert(placement.aieSchedule.end(), presTxSchedule.begin(), presTxSchedule.end());
      placement.aieSchedule.insert(placement.aieSchedule.end(), resRxSchedule.begin(), resRxSchedule.end());

      for (auto &sch : lhsTxSchedule) {
        sch.staticOffset[3] += (compTM * compTK) * lhsTxWaitCnt;
      }

      for (auto &sch : presTxSchedule) {
        sch.staticOffset[3] += (compTM * compTN) * presTxWaitCnt;
      }

      for (auto &sch : resRxSchedule) {
        sch.staticOffset[3] += (compTM * compTN + (device.pktHdrBytes / getElemBytes(elemType))) * resRxWaitCnt; // header overhead in elements
      }
    }
  } else if (tpOrder[0] == AXIS_N) {
    placement.aieSchedule.insert(placement.aieSchedule.end(), lhsTxSchedule.begin(), lhsTxSchedule.end());

    for (uint32_t i = 0; i < compTileTPn; ++i) {
      placement.aieSchedule.insert(placement.aieSchedule.end(), rhsTxSchedule.begin(), rhsTxSchedule.end());
      placement.aieSchedule.insert(placement.aieSchedule.end(), presTxSchedule.begin(), presTxSchedule.end());
      placement.aieSchedule.insert(placement.aieSchedule.end(), resRxSchedule.begin(), resRxSchedule.end());

      for (auto &sch : rhsTxSchedule) {
        sch.staticOffset[3] += (compTK * compTN) * rhsTxWaitCnt;
      }

      for (auto &sch : presTxSchedule) {
        sch.staticOffset[3] += (compTM * compTN) * presTxWaitCnt;
      }

      for (auto &sch : resRxSchedule) {
        sch.staticOffset[3] += (compTM * compTN + (device.pktHdrBytes / getElemBytes(elemType))) * resRxWaitCnt; // header overhead in elements
      }
    }
  } else { // tpOrder[0] == AXIS_K
    for (uint32_t i = 0; i < compTileTPk; ++i) {
      placement.aieSchedule.insert(placement.aieSchedule.end(), lhsTxSchedule.begin(), lhsTxSchedule.end());
      placement.aieSchedule.insert(placement.aieSchedule.end(), rhsTxSchedule.begin(), rhsTxSchedule.end());

      for (auto &sch : lhsTxSchedule) {
        sch.staticOffset[3] += (compTM * compTK) * lhsTxWaitCnt;
      }

      for (auto &sch : rhsTxSchedule) {
        sch.staticOffset[3] += (compTK * compTN) * rhsTxWaitCnt;
      }
    }

    placement.aieSchedule.insert(placement.aieSchedule.end(), resRxSchedule.begin(), resRxSchedule.end());
  }
}

//===----------------------------------------------------------------------===//
// buildSchedule
//===----------------------------------------------------------------------===//
// Build the NPU DMA runtime schedule (NpuMemcpyNd sequence).
// Determines the order and offsets of lhs/rhs/pres TX and res RX transfers
// based on tpOrder[0] (innermost loop axis):
//   AXIS_M: rhs fixed, lhs cycles over TPm iterations
//   AXIS_N: lhs fixed, rhs cycles over TPn iterations
//   AXIS_K: lhs and rhs both cycle over TPk iterations, no pres
static void buildSchedule(AiePlacement &placement, const TilingContext &tilingCtx) {
  auto sched = collectShimBdSchedules(placement, tilingCtx);

  syncChannelParallelSameData(sched.lhsTx);
  syncChannelParallelSameData(sched.rhsTx);
  syncChannelParallelSameData(sched.presTx);
  syncChannelParallelSameData(sched.resRx);

  assembleScheduleByAxis(placement, tilingCtx,
                         sched.lhsTx, sched.rhsTx, sched.presTx, sched.resRx);
}

//===----------------------------------------------------------------------===//
// debugDumpPlacement
//===----------------------------------------------------------------------===//
// Print the computed AiePlacement to dbgs() when debug is enabled.
static void debugDumpPlacement(const AiePlacement &placement, bool debug) {
  if (!debug)
    return;

  llvm::dbgs() << "[AiePlacement] Computed AIE Placement:\n";

  // ---- Tiles ----
  llvm::dbgs() << "  aieTiles (" << placement.aieTiles.size() << "):\n";
  for (size_t tileIdx = 0; tileIdx < placement.aieTiles.size(); ++tileIdx) {
    const auto &tile = placement.getAieTile(tileIdx);
    llvm::dbgs() << "    Tile[" << tileIdx << "] (" << tile.col << ", " << tile.row << ")\n";

    // Buffers (no Value prints)
    llvm::dbgs() << "      bufs (" << tile.bufs.size() << "):\n";
    for (size_t i = 0; i < tile.bufs.size(); ++i) {
      const auto &b = tile.getAieBuf(i);
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
      const auto &bd = tile.getAieBd(i);
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
      const auto &d = tile.getAieDma(i);
      llvm::dbgs() << "        - [" << i << "] dir=" << static_cast<int>(d.dir)
                  << " channel=" << d.channel
                  << " bdIdx=" << d.bdIdx << "\n";
    }
  }

  // ---- Comms (packet or circuit) ----
  llvm::dbgs() << "  aieComms (" << placement.aieComms.size() << "):\n";
  for (size_t commIdx = 0; commIdx < placement.aieComms.size(); ++commIdx) {
    const auto &comm = placement.getAieComm(commIdx);
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

    auto printU32VecWait = [&](const char *label, const std::vector<AieNpuWait> &vec) {
      llvm::dbgs() << " " << label << "=[";
      for (size_t i = 0; i < vec.size(); ++i) {
        if (i) llvm::dbgs() << ", ";
        llvm::dbgs() << "{c=" << vec[i].col
                    << ", r=" << vec[i].row
                    << ", b=" << vec[i].bufIdx << "}";
      }
      llvm::dbgs() << "]";
    };

    llvm::dbgs() << "    Sched[" << sIdx << "]"
                 << " name=" << sch.name
                 << " shimCol=" << sch.shimCol
                 << " shimRow=" << sch.shimRow
                 << " id=" << sch.id
                 << " bufIdx=" << sch.bufIdx
                 << " isPacket=" << (sch.isPacket ? "true" : "false");

    if (sch.isPacket) {
      llvm::dbgs() << " packetType=" << sch.packetType
                   << " packetId="   << sch.packetId;
    }

    llvm::dbgs() << " issueToken=" << (sch.issueToken ? "true" : "false")
                << " hasWait="    << (!sch.waitBufs.empty() ? "true" : "false");
    printU32VecWait("waitBufs", sch.waitBufs);

    printI64x4("offset", sch.staticOffset.data());
    printI64x4("size",   sch.staticSize.data());
    printI64x4("stride", sch.staticStride.data());

    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "\n";
}

//===----------------------------------------------------------------------===//
// optimizeAiePlacement
//===----------------------------------------------------------------------===//
AiePlacement
optimizeAiePlacement(const TilingContext &tilingCtx, bool debug) {
  AiePlacement placement;
  placeTiles(placement, tilingCtx);
  allocateBuffers(placement, tilingCtx);
  configureInputComms(placement, tilingCtx);
  configureOutputComms(placement, tilingCtx);
  configureDmas(placement, tilingCtx);
  buildSchedule(placement, tilingCtx);
  debugDumpPlacement(placement, debug);
  return placement;
}

} // namespace onnx_to_aie
