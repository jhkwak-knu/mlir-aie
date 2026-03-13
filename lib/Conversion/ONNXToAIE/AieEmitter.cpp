//===- AieEmitter.cpp - Generate AIE/AIEX dialect ops -----------*- C++ -*-===//
//
// Emits DeviceOp, TileOp, BufferOp, LockOp, PacketFlowOp, MemOp/DMAOp,
// CoreOp (with SCF loops), and RuntimeSequenceOp from an AiePlacement.
//
//===----------------------------------------------------------------------===//

#include "AiePlacement.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace xilinx::AIE;

namespace onnx_to_aie {

//===----------------------------------------------------------------------===//
// Trace configuration constants (AIE2 core tile)
//===----------------------------------------------------------------------===//
// Register addresses
static constexpr uint32_t TRACE_CTRL0_ADDR  = 0x340D0;  // start/stop/mode
static constexpr uint32_t TRACE_CTRL1_ADDR  = 0x340D4;  // packet config
static constexpr uint32_t TRACE_EVENT0_ADDR = 0x340E0;  // events 0-3
static constexpr uint32_t TRACE_EVENT1_ADDR = 0x340E4;  // events 4-7
static constexpr uint32_t TRACE_TIMER_ADDR  = 0x34000;  // timer control
static constexpr uint32_t TRACE_BCAST_ADDR  = 0x34010;  // broadcast base (+N*4)
static constexpr uint32_t TRACE_EVTGEN_ADDR = 0x34008;  // event generation

// Core tile event codes
static constexpr uint8_t EVT_NONE         = 0x00;
static constexpr uint8_t EVT_INSTR_EVT_0  = 0x21; // INSTR_EVENT_0 (CoreEvent=33) — kernel start
static constexpr uint8_t EVT_INSTR_EVT_1  = 0x22; // INSTR_EVENT_1 (CoreEvent=34) — kernel end
static constexpr uint8_t EVT_BCAST_14   = 0x79;  // BROADCAST_14 (107+14)
static constexpr uint8_t EVT_BCAST_15   = 0x7A;  // BROADCAST_15 (107+15)
static constexpr uint8_t EVT_USER_EVT_0 = 0x7E;  // USER_EVENT_0 (stop signal)
static constexpr uint8_t EVT_USER_EVT_1 = 0x7F;  // USER_EVENT_1 (start signal)

// Memory module trace register addresses (AIE2 compute tile)
static constexpr uint32_t MEM_TRACE_CTRL0_ADDR  = 0x140D0;  // start/stop/mode
static constexpr uint32_t MEM_TRACE_CTRL1_ADDR  = 0x140D4;  // packet config
static constexpr uint32_t MEM_TRACE_EVENT0_ADDR = 0x140E0;  // events 0-3
static constexpr uint32_t MEM_TRACE_EVENT1_ADDR = 0x140E4;  // events 4-7
static constexpr uint32_t MEM_TRACE_TIMER_ADDR  = 0x14000;  // timer control

// Memory module DMA event codes (MemEvent values)
// S2MM ch0 = LHS input (+PRES when M-inner), ch1 = RHS input (+PRES when N-inner)
// MM2S ch0 = RES output
// STALLED_LOCK: level event, active while DMA waits for lock acquisition.
// Transfer duration = FINISHED_BD timestamp - STALLED_LOCK fall timestamp.
static constexpr uint8_t MEM_EVT_DMA_S2MM_0_STARV = 35;  // LHS stream starvation
static constexpr uint8_t MEM_EVT_DMA_S2MM_1_STARV = 36;  // RHS stream starvation
static constexpr uint8_t MEM_EVT_DMA_MM2S_0_STALL = 33;  // RES lock stall
static constexpr uint8_t MEM_EVT_DMA_S2MM_0_DONE  = 23;  // LHS done
static constexpr uint8_t MEM_EVT_DMA_S2MM_1_DONE  = 24;  // RHS done
static constexpr uint8_t MEM_EVT_DMA_MM2S_0_DONE  = 25;  // RES done
// MemEvent BROADCAST_14/15 have same code (121/122) as CoreEvent — reuse EVT_BCAST_*

// Shim DMA trace config
static constexpr uint32_t TRACE_BD_ID      = 15;
static constexpr uint32_t TRACE_PER_STREAM = 65536;  // bytes per trace stream (64KB)
static constexpr uint32_t XDNA2_COL_SHIFT  = 25;
static constexpr uint32_t TRACE_BCAST_NUM      = 15;  // broadcast channel for start
static constexpr uint32_t TRACE_BCAST_STOP_NUM = 14;  // broadcast channel for stop

//===----------------------------------------------------------------------===//
// Type aliases used by emitCoreOps
//===----------------------------------------------------------------------===//
using Args = SmallVector<Value, 3>;
using InitArgs = SmallVector<Args, 2>;

struct AxisBufferSelection {
  InitArgs *reuse;   // buffer reused across inner loop iterations
  InitArgs *inner1;  // first buffer cycled each iteration
  InitArgs *inner2;  // second buffer cycled each iteration
  uint32_t repeatCount;
};

//===----------------------------------------------------------------------===//
// selectAxisBuffers
//===----------------------------------------------------------------------===//
// Maps tpOrder[0] axis to the corresponding reuse/inner1/inner2 buffer roles
// and the repeat count for the inner SCF loop.
static AxisBufferSelection selectAxisBuffers(
    const std::vector<uint32_t> &tpOrder,
    InitArgs &lhsInitArgs, InitArgs &rhsInitArgs, InitArgs &resInitArgs,
    uint32_t compTileTPm, uint32_t compTileTPn, uint32_t compTileTPk) {
  AxisBufferSelection sel;
  if (tpOrder[0] == AXIS_M) {
    sel = {&rhsInitArgs, &lhsInitArgs, &resInitArgs, compTileTPm};
  } else if (tpOrder[0] == AXIS_N) {
    sel = {&lhsInitArgs, &rhsInitArgs, &resInitArgs, compTileTPn};
  } else { // AXIS_K
    sel = {&resInitArgs, &lhsInitArgs, &rhsInitArgs, compTileTPk};
  }
  return sel;
}

//===----------------------------------------------------------------------===//
// buildKernelCallArgs
//===----------------------------------------------------------------------===//
// Build the kernel call argument list with lhs/rhs/res ordered according to
// tpOrder[0].  The reuse buffer occupies a fixed slot while inner1/inner2
// are placed in their canonical positions (A=lhs, B=rhs, C=res).
static SmallVector<Value, 7> buildKernelCallArgs(
    const std::vector<uint32_t> &tpOrder,
    const Args &arg1, const Args &arg2, const Args &reuseArgs,
    Value cRow, Value cCol, Value cDep, Value acc) {
  SmallVector<Value, 7> callArgs;
  if (tpOrder[0] == AXIS_M) { // reuse=rhs, arg1=lhs, arg2=res
    callArgs.push_back(arg1[0]);
    callArgs.push_back(reuseArgs[0]);
    callArgs.push_back(arg2[0]);
  } else if (tpOrder[0] == AXIS_N) { // reuse=lhs, arg1=rhs, arg2=res
    callArgs.push_back(reuseArgs[0]);
    callArgs.push_back(arg1[0]);
    callArgs.push_back(arg2[0]);
  } else { // AXIS_K: reuse=res, arg1=lhs, arg2=rhs
    callArgs.push_back(arg1[0]);
    callArgs.push_back(arg2[0]);
    callArgs.push_back(reuseArgs[0]);
  }
  callArgs.push_back(cRow);
  callArgs.push_back(cCol);
  callArgs.push_back(cDep);
  callArgs.push_back(acc);
  return callArgs;
}

//===----------------------------------------------------------------------===//
// emitDeviceOp
//===----------------------------------------------------------------------===//
// Creates DeviceOp sized to numCols columns.
// Sets builder insertion point to DeviceOp body for subsequent emit calls.
static DeviceOp emitDeviceOp(OpBuilder &builder, Location loc,
                              const TilingContext &tilingCtx) {
  const auto &numCols = tilingCtx.numCols;
  std::vector<AIEDevice> devices{AIEDevice::npu2_1col, AIEDevice::npu2_2col,
                                 AIEDevice::npu2_3col, AIEDevice::npu2_4col,
                                 AIEDevice::npu2_5col, AIEDevice::npu2_6col,
                                 AIEDevice::npu2_7col, AIEDevice::npu2};
  auto deviceOp = builder.create<DeviceOp>(loc, devices[numCols - 1]);
  deviceOp.getRegion().emplaceBlock();
  DeviceOp::ensureTerminator(deviceOp.getBodyRegion(), builder, loc);
  builder.setInsertionPointToStart(deviceOp.getBody());
  return deviceOp;
}

//===----------------------------------------------------------------------===//
// emitTileOps
//===----------------------------------------------------------------------===//
// Creates TileOp for every tile in placement and stores the result in tile.value.
static void emitTileOps(OpBuilder &builder, Location loc,
                        AiePlacement &placement) {
  for (auto &tile : placement.aieTiles) {
    auto tileOp = builder.create<xilinx::AIE::TileOp>(loc, tile.col, tile.row);
    tile.value = tileOp;
  }
}

//===----------------------------------------------------------------------===//
// emitBufferAndLockOps
//===----------------------------------------------------------------------===//
// Creates GlobalOp (shim tiles) and BufferOp+LockOp (comp tiles).
// Comp tile res buffer gets an extra calc lock when partial sums must be forwarded
// (TPk>1 and K is not the innermost reuse axis).
static void emitBufferAndLockOps(OpBuilder &builder, Location loc,
                                 AiePlacement &placement,
                                 const TilingContext &tilingCtx) {
  const auto &device = tilingCtx.device;

  // Generate Ops for each tile
  for (auto &tile : placement.aieTiles) {
    if (tile.row == device.shimRow) { // Shim tile
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
          if (needsPres(tilingCtx)) { // Calculator lock
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
}

//===----------------------------------------------------------------------===//
// emitTraceFlowOps
//===----------------------------------------------------------------------===//
// Creates PacketFlowOps routing each compute tile's Trace port to the shim
// tile's DMA ch1 (S2MM) in the same column.  Each tile gets a unique packet_id
// (1-based within the column) so parse_trace.py can de-interleave per tile.
static void emitTraceFlowOps(OpBuilder &builder, Location loc,
                              AiePlacement &placement,
                              const TilingContext &tilingCtx) {
  if (!tilingCtx.traceEnabled) return;
  const auto &device = tilingCtx.device;

  for (uint32_t col = 0; col < tilingCtx.numCols; ++col) {
    uint32_t shimIdx = placement.findAieTileIdx(col, device.shimRow);
    auto &shimTile = placement.getAieTile(shimIdx);

    for (uint32_t i = 0; i < tilingCtx.numCompTilesPerCol; ++i) {
      uint32_t row = device.compTileLastRow() - i;
      uint32_t compIdx = placement.findAieTileIdx(col, row);
      auto &compTile = placement.getAieTile(compIdx);

      // Core trace: Trace port 0, pktId 1..N
      uint32_t corePktId = i + 1;
      auto coreFlowOp = builder.create<PacketFlowOp>(
          loc, static_cast<int8_t>(corePktId),
          /*keep_pkt_header=*/builder.getBoolAttr(true), /*bp_id=*/nullptr);
      {
        OpBuilder::InsertionGuard g(builder);
        Region &flowRegion = coreFlowOp.getBodyRegion();
        Block *flowBlock = builder.createBlock(&flowRegion);
        builder.setInsertionPointToStart(flowBlock);
        builder.create<PacketSourceOp>(
            loc, compTile.value, WireBundle::Trace, 0);
        builder.create<PacketDestOp>(
            loc, shimTile.value, WireBundle::DMA, 1);
        builder.create<EndOp>(loc);
      }

      // Memory module trace: Trace port 1, pktId (N+1)..2N
      uint32_t memPktId = i + 1 + tilingCtx.numCompTilesPerCol;
      auto memFlowOp = builder.create<PacketFlowOp>(
          loc, static_cast<int8_t>(memPktId),
          /*keep_pkt_header=*/builder.getBoolAttr(true), /*bp_id=*/nullptr);
      {
        OpBuilder::InsertionGuard g(builder);
        Region &flowRegion = memFlowOp.getBodyRegion();
        Block *flowBlock = builder.createBlock(&flowRegion);
        builder.setInsertionPointToStart(flowBlock);
        builder.create<PacketSourceOp>(
            loc, compTile.value, WireBundle::Trace, 1);
        builder.create<PacketDestOp>(
            loc, shimTile.value, WireBundle::DMA, 1);
        builder.create<EndOp>(loc);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// emitPacketFlowOps
//===----------------------------------------------------------------------===//
// Creates PacketFlowOp/PacketSourceOp/PacketDestOp for each packet comm.
// comp->shim flows set keep_pkt_header=true to preserve the packet header.
static void emitPacketFlowOps(OpBuilder &builder, Location loc,
                               const AiePlacement &placement,
                               const TilingContext &tilingCtx) {
  const auto &device = tilingCtx.device;
  for (auto &comm : placement.aieComms) {
    if (comm.isPacket) {
      const auto &srcTile = placement.getAieTile(comm.srcIdx);
      BoolAttr keep_pkt_header = nullptr;
      if (srcTile.row != device.shimRow)
        keep_pkt_header = builder.getBoolAttr(true);

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
            auto &dstTile = placement.getAieTile(packet.dstIdxs[i]);
            builder.create<xilinx::AIE::PacketDestOp>(loc, dstTile.value, packet.dstBundles[i], static_cast<int32_t>(packet.dstChs[i]));
          }
          builder.create<EndOp>(loc);
        }
      }
    } else { // circuit-switched communication
      // TODO: implement (FlowOp)
    }
  }
}

//===----------------------------------------------------------------------===//
// emitMemDmaOps
//===----------------------------------------------------------------------===//
// Creates ShimDMAAllocationOp (shim tiles) and MemOp/DMAStartOp/DMABDOp (comp tiles).
// Comp tile res S2MM BDs release calc lock instead of cons lock when pres is needed
// (TPk>1 and K is not the innermost reuse axis).
static void emitMemDmaOps(OpBuilder &builder, Location loc,
                          AiePlacement &placement,
                          const TilingContext &tilingCtx) {
  const auto &device = tilingCtx.device;

  // Generate AIE DMAOp
  for (auto &tile : placement.aieTiles) {
    // Shim tile
    if (tile.row == device.shimRow) {
      for (auto &dma : tile.dmas) {
        std::vector<bool> allocatedBuf(tile.bufs.size(), false);
        uint32_t firstBdIdx = dma.bdIdx;
        uint32_t curBdIdx = firstBdIdx;

        do {
          auto &bd = tile.getAieBd(curBdIdx);
          auto &buf = tile.getAieBuf(bd.bufIdx);

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
    }

    // Comp tile
    auto dmaOp = builder.create<MemOp>(loc, tile.value).getOperation();
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
        auto &dma = tile.getAieDma(i);
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
          auto &bd = tile.getAieBd(curBdIdx);
          auto &buf = tile.getAieBuf(bd.bufIdx);

          {
            OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToStart(bdBlocks[curBdIdx]);

            auto acquireLockValue = (dma.dir == DMAChannelDir::S2MM) ? buf.prodLockValue : buf.consLockValue;
            auto releaseLockValue = (dma.dir == DMAChannelDir::S2MM) ? buf.consLockValue : buf.prodLockValue;
            uint32_t numToken = 1;

            if ((buf.name == "res") && (dma.dir == DMAChannelDir::S2MM)) {
              if (needsPres(tilingCtx)) {
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

}

//===----------------------------------------------------------------------===//
// emitCoreOps
//===----------------------------------------------------------------------===//
// Declares the extern_kernel FuncOp and creates CoreOp+SCF ForOp for each compute tile.
// Inner loop iterates over the innermost axis (tpOrder[0]); outer loop runs infinitely.
// Uses pres partial-sum accumulation when K is not the innermost reuse axis (TPk>1, tpOrder[0]!=AXIS_K).
static void emitCoreOps(OpBuilder &builder, Location loc,
                        AiePlacement &placement,
                        const TilingContext &tilingCtx) {
  const auto &compTM              = tilingCtx.compTM;
  const auto &compTK              = tilingCtx.compTK;
  const auto &compTN              = tilingCtx.compTN;
  const auto &compTileTPm         = tilingCtx.compTileTPm;
  const auto &compTileTPk         = tilingCtx.compTileTPk;
  const auto &compTileTPn         = tilingCtx.compTileTPn;
  const auto &tpOrder             = tilingCtx.tpOrder;
  const auto &elemType            = tilingCtx.elemType;
  const auto &doubleBufferEnabled = tilingCtx.doubleBufferEnabled;

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

  const auto &device = tilingCtx.device;

  // Configure operations of Compute tile
  for (auto &tile : placement.aieTiles) {
    // Shim/Mem tile: skip non-compute tiles
    if (tile.row < device.compTileFirstRow) {
      continue;
    }

    // Collect per-buffer init args (lhs/rhs/res) from tile buffers
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
        auto acquireValue = needsPres(tilingCtx) ?
                                      buf.calcLockValue : buf.prodLockValue;
        Args bufArgs{buf.bufValue, acquireValue, buf.consLockValue};
        resInitArgs.push_back(bufArgs);
      }
    }

    auto axisSel = selectAxisBuffers(tpOrder,
        lhsInitArgs, rhsInitArgs, resInitArgs,
        compTileTPm, compTileTPn, compTileTPk);
    auto &reuseInitArgs = *axisSel.reuse;
    auto &inner1InitArgs = *axisSel.inner1;
    auto &inner2InitArgs = *axisSel.inner2;
    uint32_t repeatCount = axisSel.repeatCount;

    // Generate Memref GlobalOp
    std::string flagName = std::string("flag_") + std::to_string(tile.col) + "_" + std::to_string(tile.row);
    auto flagMemrefNameAttr = builder.getStringAttr(flagName);
    auto i1Ty = builder.getI1Type();
    auto flagMemrefType = MemRefType::get({}, i1Ty);
    builder.create<memref::GlobalOp>(loc, flagMemrefNameAttr, builder.getStringAttr("private"),
                                                      flagMemrefType, nullptr, false, nullptr);

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
      auto cMax = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(CORE_LOOP_INFINITE));
      auto cCnt = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(repeatCount));

      // Dimension args (N_ROW, N_COL, N_DEP) are always uint32_t in the kernel
      // signature, regardless of the element data type.
      auto cRow = builder.create<mlir::arith::ConstantIntOp>(loc, compTM, /*width=*/32);
      auto cCol = builder.create<mlir::arith::ConstantIntOp>(loc, compTN, /*width=*/32);
      auto cDep = builder.create<mlir::arith::ConstantIntOp>(loc, compTK, /*width=*/32);

      auto trueI1 = builder.create<arith::ConstantIntOp>(loc, /*value=*/1, /*bitWidth=*/1);
      auto falseI1 = builder.create<arith::ConstantIntOp>(loc, /*value=*/0, /*bitWidth=*/1);

      auto accVar = builder.create<memref::GetGlobalOp>(loc, flagMemrefType, flagName);

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
        if (needsPres(tilingCtx)) {
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

          auto callArgs = buildKernelCallArgs(tpOrder, arg1, arg2, reuseArgs,
                                              cRow, cCol, cDep, acc);

          // Mark kernel start/end for trace timing (INSTR_EVENT_0/1)
          if (tilingCtx.traceEnabled)
            builder.create<EventOp>(loc, /*val=*/0);

          auto calleeAttr = SymbolRefAttr::get(builder.getContext(), "extern_kernel");
          builder.create<mlir::func::CallOp>(loc, calleeAttr, TypeRange{}, ValueRange(callArgs));

          if (tilingCtx.traceEnabled)
            builder.create<EventOp>(loc, /*val=*/1);

          // Generate AIE UseLockOp (arg1/arg2)
          builder.create<UseLockOp>(loc, arg2[2], LockAction::Release, 1);
          builder.create<UseLockOp>(loc, arg1[2], LockAction::Release, 1);

          // Generate Memref StoreOp (acc)
          if ((compTileTPk > 1) && (tpOrder[0] == AXIS_K)) {
            builder.create<memref::StoreOp>(loc, trueI1, accVar);
          }

          // Generate SCF YieldOp
          if (!doubleBufferEnabled) {
            builder.create<mlir::scf::YieldOp>(loc, ValueRange{arg1[0], arg1[1], arg1[2],
                                                                arg2[0], arg2[1], arg2[2], innerT});
          } else {
            // TODO: complete double-buffer implementation
            // DB requires lhsdb/rhsdb/resdb buffers so that inner1InitArgs
            // and inner2InitArgs each have 2 entries (db0 and db1).
            if (inner1InitArgs.size() < 2 || inner2InitArgs.size() < 2)
              llvm::report_fatal_error(
                  "double buffer args not initialized: "
                  "lhsdb/rhsdb/resdb buffers must be allocated");

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
          // TODO: complete double-buffer implementation
          // DB requires lhsdb/rhsdb/resdb buffers so that reuseInitArgs
          // has 2 entries (db0 and db1).
          if (reuseInitArgs.size() < 2)
            llvm::report_fatal_error(
                "double buffer args not initialized: "
                "lhsdb/rhsdb/resdb buffers must be allocated");

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
}

//===----------------------------------------------------------------------===//
// emitRuntimeSequenceOp
//===----------------------------------------------------------------------===//
// Creates RuntimeSequenceOp and NpuDmaMemcpyNdOp/NpuDmaWaitOp for the NPU DMA schedule.
// Buffer sizes are derived from tilingCtx; pres argument is added only when needed
// (TPk>1 and K is not the innermost reuse axis).
static void emitRuntimeSequenceOp(OpBuilder &builder, Location loc,
                                   AiePlacement &placement,
                                   const TilingContext &tilingCtx) {
  const auto &compTM      = tilingCtx.compTM;
  const auto &compTK      = tilingCtx.compTK;
  const auto &compTN      = tilingCtx.compTN;
  const auto &compTileSPm = tilingCtx.compTileSPm;
  const auto &compTileSPn = tilingCtx.compTileSPn;
  const auto &compTileTPm = tilingCtx.compTileTPm;
  const auto &compTileTPk = tilingCtx.compTileTPk;
  const auto &compTileTPn = tilingCtx.compTileTPn;
  const auto &tpOrder     = tilingCtx.tpOrder;
  const auto &elemType    = tilingCtx.elemType;

  // Generate AIEX RuntimeSequenceOp
  std::string seq_name = "sequence";
  StringAttr seq_sym_name = builder.getStringAttr(seq_name);
  auto seqOp = builder.create<xilinx::AIEX::RuntimeSequenceOp>(loc, seq_sym_name);
  {
    OpBuilder::InsertionGuard g(builder);
    Region &seqRegion = seqOp.getBody();
    Block *seqBlock = builder.createBlock(&seqRegion);
    builder.setInsertionPointToStart(seqBlock);

    uint32_t localTPm = (tpOrder[0] == AXIS_M) ? compTileTPm : 1;
    uint32_t localTPn = (tpOrder[0] == AXIS_N) ? compTileTPn : 1;
    uint32_t localTPk = (tpOrder[0] == AXIS_K) ? compTileTPk : 1;

    const auto &device = tilingCtx.device;

    uint32_t lhsSize = ((compTM * compTileSPm) * compTK) * localTPm * localTPk;
    uint32_t rhsSize = (compTK * (compTN * compTileSPn)) * localTPk * localTPn;
    uint32_t resSize = ((compTM * compTN + (device.pktHdrBytes / getElemBytes(elemType))) * compTileSPm * compTileSPn) * localTPm * localTPn; // header overhead in elements

    auto lhsMemrefType = MemRefType::get({lhsSize}, elemType);
    auto rhsMemrefType = MemRefType::get({rhsSize}, elemType);
    auto resMemrefType = MemRefType::get({resSize}, elemType);

    auto arg_lhs = seqBlock->addArgument(lhsMemrefType, loc);
    auto arg_rhs = seqBlock->addArgument(rhsMemrefType, loc);
    auto arg_res = seqBlock->addArgument(resMemrefType, loc);

    Value arg_pres;
    if (needsPres(tilingCtx)) {
      uint32_t presSize = ((compTM * compTN) * compTileSPm * compTileSPn) * localTPm * localTPn;
      auto presMemrefType = MemRefType::get({presSize}, elemType);
      arg_pres = seqBlock->addArgument(presMemrefType, loc);
    }

    // Add trace buffer argument (last arg) when trace is enabled
    // 2 streams per tile: core trace + memory module trace
    Value arg_trace;
    if (tilingCtx.traceEnabled) {
      uint32_t numStreams = tilingCtx.numCols * tilingCtx.numCompTilesPerCol * 2;
      uint32_t totalTraceBytes = TRACE_PER_STREAM * numStreams;
      auto traceType = MemRefType::get(
          {static_cast<int64_t>(totalTraceBytes / 4)}, builder.getI32Type());
      arg_trace = seqBlock->addArgument(traceType, loc);
    }

    // Emit trace setup ops before DMA schedule so tracing captures all activity
    if (tilingCtx.traceEnabled) {
      // Per-tile: configure core trace and memory module trace
      for (uint32_t col = 0; col < tilingCtx.numCols; ++col) {
        for (uint32_t i = 0; i < tilingCtx.numCompTilesPerCol; ++i) {
          uint32_t row = device.compTileLastRow() - i;
          auto colAttr = builder.getI32IntegerAttr(col);
          auto rowAttr = builder.getI32IntegerAttr(row);
          uint32_t corePktId = i + 1;
          uint32_t memPktId  = i + 1 + tilingCtx.numCompTilesPerCol;

          //--- Core trace (0x340xx) ---

          // Trace Control 0: start=BROADCAST_15, stop=BROADCAST_14, mode=event-time
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              TRACE_CTRL0_ADDR,
              static_cast<uint32_t>((EVT_BCAST_14 << 24) | (EVT_BCAST_15 << 16)),
              nullptr, colAttr, rowAttr);

          // Trace Control 1: packet mode, type=CORE(0), packet_id
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              TRACE_CTRL1_ADDR,
              static_cast<uint32_t>((0u << 12) | (corePktId & 0x1F)),
              nullptr, colAttr, rowAttr);

          // Event Group 0: [3]=NONE [2]=NONE [1]=INSTR_EVENT_1 [0]=INSTR_EVENT_0
          uint32_t coreEg0 = (EVT_NONE << 24) | (EVT_NONE << 16) |
                             (EVT_INSTR_EVT_1 << 8) | EVT_INSTR_EVT_0;
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              TRACE_EVENT0_ADDR, coreEg0, nullptr, colAttr, rowAttr);

          // Event Group 1: [7..4] = all NONE (disabled)
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              TRACE_EVENT1_ADDR, static_cast<uint32_t>(0), nullptr, colAttr, rowAttr);

          // Timer control: reset on BROADCAST_15 for cross-tile synchronization
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              TRACE_TIMER_ADDR,
              static_cast<uint32_t>((EVT_BCAST_15 & 0x7F) << 8),
              nullptr, colAttr, rowAttr);

          //--- Memory module trace (0x140xx) ---

          // Trace Control 0: same start/stop broadcast events
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              MEM_TRACE_CTRL0_ADDR,
              static_cast<uint32_t>((EVT_BCAST_14 << 24) | (EVT_BCAST_15 << 16)),
              nullptr, colAttr, rowAttr);

          // Trace Control 1: packet mode, type=MEM(1), packet_id
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              MEM_TRACE_CTRL1_ADDR,
              static_cast<uint32_t>((1u << 12) | (memPktId & 0x1F)),
              nullptr, colAttr, rowAttr);

          // Event Group 0: [3]=S2MM_1_DONE [2]=S2MM_1_STARV [1]=S2MM_0_DONE [0]=S2MM_0_STARV
          uint32_t memEg0 = (MEM_EVT_DMA_S2MM_1_DONE  << 24) |
                            (MEM_EVT_DMA_S2MM_1_STARV << 16) |
                            (MEM_EVT_DMA_S2MM_0_DONE  << 8)  |
                             MEM_EVT_DMA_S2MM_0_STARV;
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              MEM_TRACE_EVENT0_ADDR, memEg0, nullptr, colAttr, rowAttr);

          // Event Group 1: [7..6]=NONE [5]=MM2S_0_DONE [4]=MM2S_0_STALL
          uint32_t memEg1 = (EVT_NONE << 24) | (EVT_NONE << 16) |
                            (MEM_EVT_DMA_MM2S_0_DONE  << 8) |
                             MEM_EVT_DMA_MM2S_0_STALL;
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              MEM_TRACE_EVENT1_ADDR, memEg1, nullptr, colAttr, rowAttr);

          // Timer control: reset on BROADCAST_15 for cross-tile synchronization
          builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
              MEM_TRACE_TIMER_ADDR,
              static_cast<uint32_t>((EVT_BCAST_15 & 0x7F) << 8),
              nullptr, colAttr, rowAttr);
        }
      }

      // Per-column: configure shim DMA BD for trace collection
      // 2 streams per tile (core + mem) share the same shim DMA BD
      for (uint32_t col = 0; col < tilingCtx.numCols; ++col) {
        uint32_t traceSizeBytes = TRACE_PER_STREAM * tilingCtx.numCompTilesPerCol * 2;
        uint32_t traceOffsetBytes = col * traceSizeBytes;
        // NpuWriteBdOp buffer_length/offset are in 32-bit words (AIE2 addr gen granularity)
        uint32_t traceLenWords = traceSizeBytes / 4;
        uint32_t traceOffWords = traceOffsetBytes / 4;

        // Shim DMA BD with enable_packet=1 to receive interleaved packets
        builder.create<xilinx::AIEX::NpuWriteBdOp>(loc,
            /*column=*/col, /*bd_id=*/TRACE_BD_ID,
            /*buffer_length=*/traceLenWords, /*buffer_offset=*/traceOffWords,
            /*enable_packet=*/1, /*out_of_order_id=*/0,
            /*packet_id=*/0, /*packet_type=*/0,
            /*d0_size=*/0, /*d0_stride=*/0,
            /*d1_size=*/0, /*d1_stride=*/0,
            /*d2_size=*/0, /*d2_stride=*/0,
            /*iteration_current=*/0, /*iteration_size=*/0, /*iteration_stride=*/0,
            /*next_bd=*/0, /*row=*/0, /*use_next_bd=*/0, /*valid_bd=*/1,
            /*lock_rel_val=*/0, /*lock_rel_id=*/0,
            /*lock_acq_enable=*/0, /*lock_acq_val=*/0, /*lock_acq_id=*/0,
            /*d0_zero_before=*/0, /*d1_zero_before=*/0, /*d2_zero_before=*/0,
            /*d0_zero_after=*/0, /*d1_zero_after=*/0, /*d2_zero_after=*/0);

        // Patch DDR address for trace buffer arg
        uint32_t traceArgIdx = needsPres(tilingCtx) ? 4 : 3;
        uint32_t patchAddr = (col << XDNA2_COL_SHIFT) |
                             (0x1D004 + TRACE_BD_ID * 0x20);
        builder.create<xilinx::AIEX::NpuAddressPatchOp>(loc,
            patchAddr, traceArgIdx, traceOffsetBytes);

        // Start shim S2MM ch1
        builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
            static_cast<uint32_t>(0x1D20C),
            static_cast<uint32_t>(TRACE_BD_ID),
            nullptr,
            builder.getI32IntegerAttr(col),
            builder.getI32IntegerAttr(0));
      }

      // Broadcast timer sync: shim tile col=0 generates USER_EVENT_1 → broadcast 15
      auto shimCol0 = builder.getI32IntegerAttr(0);
      auto shimRow0 = builder.getI32IntegerAttr(0);

      // Timer control on shim tile: reset on USER_EVENT_1
      builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
          TRACE_TIMER_ADDR,
          static_cast<uint32_t>((EVT_USER_EVT_1 & 0x7F) << 8),
          nullptr, shimCol0, shimRow0);

      // Broadcast config: broadcast 15 = USER_EVENT_1
      builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
          static_cast<uint32_t>(TRACE_BCAST_ADDR + TRACE_BCAST_NUM * 4),
          static_cast<uint32_t>(EVT_USER_EVT_1),
          nullptr, shimCol0, shimRow0);

      // Generate USER_EVENT_1 → triggers broadcast → all tiles start tracing
      builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
          TRACE_EVTGEN_ADDR,
          static_cast<uint32_t>(EVT_USER_EVT_1 & 0x7F),
          nullptr, shimCol0, shimRow0);
    }

    // Generate AIEX NpuDmaMemcpyNdOp
    for (auto &sch : placement.aieSchedule) {
      Value arg;

      if (sch.name.compare(0, 3, "lhs") == 0) {
        arg = arg_lhs;
      } else if (sch.name.compare(0, 3, "rhs") == 0) {
        arg = arg_rhs;
      } else if (sch.name.compare(0, 3, "res") == 0) {
        arg = arg_res;
      } else { // name starts with "pres"
        // Invariant: a "pres" schedule entry is only generated when pres buffers
        // are allocated, which requires needsPres(tilingCtx) -- the same
        // condition that initializes arg_pres above.
        assert(arg_pres && "pres schedule entry present but needsPres() returned false");
        arg = arg_pres;
      }
      uint32_t col = sch.shimCol;
      uint32_t row = sch.shimRow;

      uint32_t shimIdx = placement.findAieTileIdx(col, row);
      AieTile &shimTile = placement.getAieTile(shimIdx);

      auto &buf = shimTile.getAieBuf(sch.bufIdx);
      StringRef metadata = builder.getStringAttr(buf.symbol);
      PacketInfoAttr packetAttr = nullptr;
      if (sch.isPacket) {
        packetAttr = PacketInfoAttr::get(builder.getContext(), static_cast<uint16_t>(sch.packetType), static_cast<uint16_t>(sch.packetId));
      }

      builder.create<xilinx::AIEX::NpuDmaMemcpyNdOp>(loc, arg, SmallVector<Value>{}, SmallVector<Value>{}, SmallVector<Value>{},
                                                    ArrayRef(sch.staticOffset), ArrayRef(sch.staticSize), ArrayRef(sch.staticStride),
                                                    packetAttr, metadata, sch.id, sch.issueToken, 0, 0, 0, 0, 0, 0);

      for (auto &waitBuf : sch.waitBufs) {
        uint32_t tileIdx = placement.findAieTileIdx(waitBuf.col, waitBuf.row);
        AieTile &tile = placement.getAieTile(tileIdx);
        auto &buf = tile.getAieBuf(waitBuf.bufIdx);
        StringRef metadata = builder.getStringAttr(buf.symbol);
        builder.create<xilinx::AIEX::NpuDmaWaitOp>(loc, metadata);
      }
    }

    // Stop trace: USER_EVENT_0 → broadcast 14 → all tiles stop tracing
    // Must come after all DMA waits so trace captures the full execution.
    if (tilingCtx.traceEnabled) {
      auto shimCol0 = builder.getI32IntegerAttr(0);
      auto shimRow0 = builder.getI32IntegerAttr(0);

      // Broadcast config: broadcast 14 = USER_EVENT_0
      builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
          static_cast<uint32_t>(TRACE_BCAST_ADDR + TRACE_BCAST_STOP_NUM * 4),
          static_cast<uint32_t>(EVT_USER_EVT_0),
          nullptr, shimCol0, shimRow0);

      // Generate USER_EVENT_0 → triggers broadcast 14 → all tiles stop tracing
      builder.create<xilinx::AIEX::NpuWrite32Op>(loc,
          TRACE_EVTGEN_ADDR,
          static_cast<uint32_t>(EVT_USER_EVT_0 & 0x7F),
          nullptr, shimCol0, shimRow0);
    }
  }
}

//===----------------------------------------------------------------------===//
// generateAieOps (entry point)
//===----------------------------------------------------------------------===//
void generateAieOps(ConversionPatternRewriter &rewriter,
                    AiePlacement &placement,
                    const TilingContext &tilingCtx,
                    const std::string &outputPath) {
  MLIRContext *ctx = rewriter.getContext();
  auto loc = mlir::UnknownLoc::get(ctx);
  auto aieModule = ModuleOp::create(loc);
  OpBuilder builder(aieModule.getBodyRegion());
  builder.setInsertionPointToStart(aieModule.getBody());

  emitDeviceOp(builder, loc, tilingCtx);
  emitTileOps(builder, loc, placement);
  emitBufferAndLockOps(builder, loc, placement, tilingCtx);
  emitTraceFlowOps(builder, loc, placement, tilingCtx);
  emitPacketFlowOps(builder, loc, placement, tilingCtx);
  emitMemDmaOps(builder, loc, placement, tilingCtx);
  emitCoreOps(builder, loc, placement, tilingCtx);
  emitRuntimeSequenceOp(builder, loc, placement, tilingCtx);

  // Save the mlir code composed of AIE dialect
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec);
  if (ec)
    llvm::report_fatal_error(llvm::Twine("cannot open aie.mlir output '") +
                             outputPath + "': " + ec.message());
  aieModule->print(out);
}

} // namespace onnx_to_aie
