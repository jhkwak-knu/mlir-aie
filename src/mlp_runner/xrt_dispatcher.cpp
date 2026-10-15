//===- xrt_dispatcher.cpp - Implementation ----------------------*- C++ -*-===//

#include "xrt_dispatcher.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

// Reuse the tiling helpers from test/onnx-mlir/src/. We include those
// headers without modification.
#include "tile_ops.h"
#include "tile_order.h"
#include "tiling_param.h"

#include "gemm_cpu.h"  // for softmax / relu definitions (not used here but keeps link graph tight)

namespace mlp_runner {

// ---------------------------------------------------------------------------
// PIMPL state
// ---------------------------------------------------------------------------

struct XrtDispatcher::Impl {
  xrt::device device;
  bool device_open = false;

  struct LayerContext {
    tilingParam tp;
    TileOrderConfig toc;
    xrt::kernel kernel;
    xrt::bo bo_instr;
    xrt::bo bo_inA;
    xrt::bo bo_inB;
    xrt::bo bo_outC;
    xrt::bo bo_inC;      // valid only when useInC
    std::vector<uint32_t> instr_v;
    int chunkASize = 0;
    int chunkBSize = 0;
    int chunkCSize = 0;
    int chunkOutCSize = 0;
    bool useInC = false;
    int totalSteps = 0;
  };

  std::map<std::string, std::unique_ptr<LayerContext>> cache;

  LayerContext& getOrLoad(const LayerKernelEntry& layer);

  void runLayer(LayerContext& ctx, const float* A, const float* B, float* C);
};

// ---------------------------------------------------------------------------
// tc.json -> tilingParam loader (same schema generate_configs.py emits)
// ---------------------------------------------------------------------------

static tilingParam _loadTpFromFile(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("cannot open tc.json: " + path);
  json doc;
  ifs >> doc;
  tilingParam tp;
  tp.M = doc.at("M").get<uint32_t>();
  tp.K = doc.at("K").get<uint32_t>();
  tp.N = doc.at("N").get<uint32_t>();
  const auto& lvl = doc.at("levels").at(0);
  tp.SPm = lvl.at("SPm").get<uint32_t>();
  tp.SPn = lvl.at("SPn").get<uint32_t>();
  tp.TPm = lvl.at("TPm").get<uint32_t>();
  tp.TPk = lvl.at("TPk").get<uint32_t>();
  tp.TPn = lvl.at("TPn").get<uint32_t>();
  tp.TM  = lvl.at("TM").get<uint32_t>();
  tp.TK  = lvl.at("TK").get<uint32_t>();
  tp.TN  = lvl.at("TN").get<uint32_t>();
  for (const auto& v : lvl.at("tpOrder"))
    tp.tpOrder.push_back(v.get<uint32_t>());
  if (tp.tpOrder.size() != 3)
    throw std::runtime_error("tc.json levels[0].tpOrder must have 3 entries");
  return tp;
}

// ---------------------------------------------------------------------------
// getOrLoad: XRT device/kernel setup per layer (cached by xclbin path)
// ---------------------------------------------------------------------------

XrtDispatcher::Impl::LayerContext&
XrtDispatcher::Impl::getOrLoad(const LayerKernelEntry& layer) {
  auto it = cache.find(layer.xclbin_path);
  if (it != cache.end()) return *it->second;

  auto ctx = std::make_unique<LayerContext>();
  // Read tc.json sitting next to the xclbin (compile_kernels.py drops it there).
  const std::string tc_path =
      layer.xclbin_path.substr(0, layer.xclbin_path.find_last_of('/')) + "/tc.json";
  ctx->tp = _loadTpFromFile(tc_path);
  ctx->toc = buildTileOrders(ctx->tp);

  // Compute chunk sizes (same formulas as host.cpp line 131-139).
  int chunkTPm = (ctx->tp.tpOrder[0] == AXIS_M) ? ctx->tp.TPm : 1;
  int chunkTPn = (ctx->tp.tpOrder[0] == AXIS_N) ? ctx->tp.TPn : 1;
  int chunkTPk = (ctx->tp.tpOrder[0] == AXIS_K) ? ctx->tp.TPk : 1;
  ctx->chunkASize = (ctx->tp.TM * ctx->tp.TK) * ctx->tp.SPm * chunkTPm * chunkTPk;
  ctx->chunkBSize = (ctx->tp.TN * ctx->tp.TK) * ctx->tp.SPn * chunkTPn * chunkTPk;
  ctx->chunkCSize = (ctx->tp.TM * ctx->tp.TN) * ctx->tp.SPm * ctx->tp.SPn
                    * chunkTPm * chunkTPn;
  ctx->chunkOutCSize =
      (ctx->tp.TM * ctx->tp.TN + PKT_HDR_ELEMS) * ctx->tp.SPm * ctx->tp.SPn
      * chunkTPm * chunkTPn;
  ctx->useInC = (ctx->tp.TPk > 1) && (ctx->tp.tpOrder[0] != AXIS_K);
  ctx->totalSteps = ctx->toc.outerTP * ctx->toc.innerTP;

  // Load instruction stream.
  ctx->instr_v = test_utils::load_instr_binary(layer.insts_path);

  // Open device once.
  if (!device_open) {
    test_utils::init_xrt_load_kernel(device, ctx->kernel, /*verbosity=*/0,
                                      layer.xclbin_path, "MLIR_AIE");
    device_open = true;
  } else {
    xrt::kernel k;
    test_utils::init_xrt_load_kernel(device, k, /*verbosity=*/0,
                                      layer.xclbin_path, "MLIR_AIE");
    ctx->kernel = std::move(k);
  }

  ctx->bo_instr = xrt::bo(device,
                           ctx->instr_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, ctx->kernel.group_id(1));
  ctx->bo_inA = xrt::bo(device,
                         ctx->chunkASize * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, ctx->kernel.group_id(3));
  ctx->bo_inB = xrt::bo(device,
                         ctx->chunkBSize * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, ctx->kernel.group_id(4));
  ctx->bo_outC = xrt::bo(device,
                          ctx->chunkOutCSize * sizeof(DATATYPE),
                          XRT_BO_FLAGS_HOST_ONLY, ctx->kernel.group_id(5));
  if (ctx->useInC) {
    ctx->bo_inC = xrt::bo(device,
                           ctx->chunkCSize * sizeof(DATATYPE),
                           XRT_BO_FLAGS_HOST_ONLY, ctx->kernel.group_id(6));
  }

  // Copy instruction stream once; it never changes.
  std::memcpy(ctx->bo_instr.map<void*>(),
              ctx->instr_v.data(),
              ctx->instr_v.size() * sizeof(int));
  ctx->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  cache.emplace(layer.xclbin_path, std::move(ctx));
  return *cache[layer.xclbin_path];
}

// ---------------------------------------------------------------------------
// runLayer: stage -> dispatch -> decode for a single forward pass
// ---------------------------------------------------------------------------

void XrtDispatcher::Impl::runLayer(LayerContext& ctx, const float* A,
                                   const float* B, float* C) {
  const auto& tp = ctx.tp;
  const auto& toc = ctx.toc;

  // 1. Convert inputs to bfloat16 with truncation, transpose B ([K,N] -> [N,K]).
  //    host.cpp operates on std::vector<DATATYPE> matA [M,K] + matB [N,K]; we
  //    synthesize those from the float32 buffers the caller provided.
  const int matASize = tp.M * tp.K;
  const int matBSize = tp.N * tp.K;
  const int matCSize = tp.M * tp.N;
  std::vector<DATATYPE> matA(matASize), matB(matBSize), matC(matCSize, DATATYPE(0));

  for (int i = 0; i < matASize; ++i)
    matA[i] = static_cast<DATATYPE>(bf16_trunc(A[i]));
  // B caller: [K,N] row-major -> bf16 [N,K] (transpose).
  for (uint32_t k = 0; k < tp.K; ++k) {
    for (uint32_t n = 0; n < tp.N; ++n) {
      float v = B[k * tp.N + n];
      matB[n * tp.K + k] = static_cast<DATATYPE>(bf16_trunc(v));
    }
  }

  // 2. Stage each temporal step's A/B chunk in mmul layout.
  struct StagedStep {
    std::vector<DATATYPE> a;
    std::vector<DATATYPE> b;
    std::vector<std::pair<int, int>> cTileOrder;
  };
  std::vector<StagedStep> staged(ctx.totalSteps);
  for (int oi = 0; oi < toc.outerTP; ++oi) {
    for (int ij = 0; ij < toc.innerTP; ++ij) {
      int step = oi * toc.innerTP + ij;
      auto& ss = staged[step];

      std::vector<std::pair<int, int>> aTileOrder;
      for (const auto& base : toc.chunkATileOrderBase) {
        aTileOrder.emplace_back(
            base.first  + toc.matAInnerOffset.first  * ij + toc.matAOuterOffset.first  * oi,
            base.second + toc.matAInnerOffset.second * ij + toc.matAOuterOffset.second * oi);
      }
      ss.a.resize(ctx.chunkASize, static_cast<DATATYPE>(0));
      int aCount = 0;
      for (const auto& [tr, tc] : aTileOrder) {
        auto tile = extract_tile_1d_strict<DATATYPE>(
            matA, tp.M, tp.K, tp.TM, tp.TK, tr, tc);
        auto tiled = tile_to_mmul_layout<DATATYPE, MMUL_R, MMUL_S>(
            tile.data(), tp.TM, tp.TK);
        std::copy(tiled.begin(), tiled.end(), ss.a.data() + aCount);
        aCount += tiled.size();
      }

      std::vector<std::pair<int, int>> bTileOrder;
      for (const auto& base : toc.chunkBTileOrderBase) {
        bTileOrder.emplace_back(
            base.first  + toc.matBInnerOffset.first  * ij + toc.matBOuterOffset.first  * oi,
            base.second + toc.matBInnerOffset.second * ij + toc.matBOuterOffset.second * oi);
      }
      ss.b.resize(ctx.chunkBSize, static_cast<DATATYPE>(0));
      int bCount = 0;
      for (const auto& [tr, tc] : bTileOrder) {
        auto tile = extract_tile_1d_strict<DATATYPE>(
            matB, tp.N, tp.K, tp.TN, tp.TK, tr, tc);
        auto tiled = tile_to_mmul_layout<DATATYPE, MMUL_T, MMUL_S>(
            tile.data(), tp.TN, tp.TK);
        std::copy(tiled.begin(), tiled.end(), ss.b.data() + bCount);
        bCount += tiled.size();
      }

      for (const auto& base : toc.chunkCTileOrderBase) {
        ss.cTileOrder.emplace_back(
            base.first  + toc.matCInnerOffset.first  * ij + toc.matCOuterOffset.first  * oi,
            base.second + toc.matCInnerOffset.second * ij + toc.matCOuterOffset.second * oi);
      }
    }
  }

  // 3. Zero pres / output buffers.
  auto* bufInA  = ctx.bo_inA.map<DATATYPE*>();
  auto* bufInB  = ctx.bo_inB.map<DATATYPE*>();
  auto* bufOut  = ctx.bo_outC.map<DATATYPE*>();
  DATATYPE* bufInC = nullptr;
  if (ctx.useInC) {
    bufInC = ctx.bo_inC.map<DATATYPE*>();
    std::memset(bufInC, 0, ctx.chunkCSize * sizeof(DATATYPE));
    ctx.bo_inC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // 4. Dispatch one iteration across totalSteps (matches host.cpp verify block).
  const unsigned opcode = 3;
  auto runKernel = [&]() {
    xrt::run run;
    if (ctx.useInC)
      run = ctx.kernel(opcode, ctx.bo_instr, ctx.instr_v.size(),
                        ctx.bo_inA, ctx.bo_inB, ctx.bo_outC, ctx.bo_inC);
    else
      run = ctx.kernel(opcode, ctx.bo_instr, ctx.instr_v.size(),
                        ctx.bo_inA, ctx.bo_inB, ctx.bo_outC);
    run.wait();
  };

  for (int step = 0; step < ctx.totalSteps; ++step) {
    const auto& ss = staged[step];
    std::memcpy(bufInA, ss.a.data(), ctx.chunkASize * sizeof(DATATYPE));
    std::memcpy(bufInB, ss.b.data(), ctx.chunkBSize * sizeof(DATATYPE));
    std::memset(bufOut, 0, ctx.chunkOutCSize * sizeof(DATATYPE));

    if (ctx.useInC) {
      // Extract partial sums from current matC (accumulated from prior steps).
      int cCount = 0;
      for (const auto& [tr, tc] : ss.cTileOrder) {
        auto tile = extract_tile_1d_strict<DATATYPE>(
            matC, tp.M, tp.N, tp.TM, tp.TN, tr, tc);
        auto tiled = tile_to_mmul_layout<DATATYPE, MMUL_R, MMUL_T>(
            tile.data(), tp.TM, tp.TN);
        std::copy(tiled.begin(), tiled.end(), bufInC + cCount);
        cCount += tiled.size();
      }
    }

    ctx.bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx.bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx.bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (ctx.useInC) ctx.bo_inC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    runKernel();
    ctx.bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Decode packet headers + write tiles back to matC (same as host.cpp
    // verify block).
    int repeatCount = (toc.reuseTPAxis != AXIS_K) ? toc.reuseTP : 1;
    for (int ri = 0; ri < repeatCount; ++ri) {
      int outerOff = tp.SPm * tp.SPn * ri;
      for (int rj = 0; rj < static_cast<int>((tp.SPm * tp.SPn) / 4); ++rj) {
        int innerOff = 4 * rj;
        for (int rk = 0; rk < 4; ++rk) {
          int idx = outerOff + innerOff + rk;
          uint32_t hdr;
          std::memcpy(&hdr,
                      &bufOut[(tp.TM * tp.TN + PKT_HDR_ELEMS) * idx],
                      sizeof(hdr));
          uint32_t rawId = hdr & 0x1F, pid = 0;
          while (rawId > 1) { rawId >>= 1; ++pid; }
          auto tilePos = ss.cTileOrder[outerOff + innerOff + pid];
          const DATATYPE* tiledOut =
              &bufOut[(tp.TM * tp.TN + PKT_HDR_ELEMS) * idx + PKT_HDR_ELEMS];
          auto val = untile_from_mmul_layout<DATATYPE, MMUL_R, MMUL_T>(
              tiledOut, tp.TM, tp.TN);
          write_tile_1d_strict<DATATYPE>(matC, tp.M, tp.N, tp.TM, tp.TN,
                                          tilePos.first, tilePos.second, val);
        }
      }
    }
  }

  // 5. Convert matC bf16 -> float32 into caller buffer [M, N].
  for (int i = 0; i < matCSize; ++i) C[i] = static_cast<float>(matC[i]);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

XrtDispatcher::XrtDispatcher() : impl_(std::make_unique<Impl>()) {}
XrtDispatcher::~XrtDispatcher() = default;

void XrtDispatcher::dispatch(const LayerKernelEntry& layer,
                             const float* A, const float* B, float* C) {
  auto& ctx = impl_->getOrLoad(layer);
  impl_->runLayer(ctx, A, B, C);
}

}  // namespace mlp_runner
