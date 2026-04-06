//===- tiling_param.h - Tiling parameter types and I/O ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef TILING_PARAM_H
#define TILING_PARAM_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stdfloat>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE = std::bfloat16_t;
#endif

// Packet header is 4 bytes; number of DATATYPE elements it occupies.
static constexpr size_t PKT_HDR_ELEMS = 4 / sizeof(DATATYPE);

// AIE2 to_vector<bfloat16>() uses truncation (round-toward-zero), not
// round-to-nearest-even.  This helper replicates that behaviour so the
// host reference matches the NPU's intermediate bf16 precision.
static inline float bf16_trunc(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  bits &= 0xFFFF0000u;          // zero the lower 16 bits (truncate)
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

#include "nlohmann/json.hpp"
using json = nlohmann::json;

// Axis indices used in tpOrder to select the innermost temporal loop axis.
// tpOrder[0] determines which axis is reused across iterations:
//   AXIS_M (0) -> RHS reuse, AXIS_N (1) -> LHS reuse, AXIS_K (2) -> local accumulation.
enum AxisId : uint32_t { AXIS_M = 0, AXIS_N = 1, AXIS_K = 2 };

struct tilingParam {
  uint32_t M, K, N;
  uint32_t SPm, SPn, TPm, TPk, TPn;
  uint32_t TM, TK, TN;
  std::vector<uint32_t> tpOrder; // 0:M, 1:N, 2:K
};

static inline tilingParam loadTilingParam(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("cannot open: " + path);
  json j; ifs >> j;

  tilingParam tp{};
  tp.M  = j.at("M").get<uint32_t>();
  tp.K  = j.at("K").get<uint32_t>();
  tp.N  = j.at("N").get<uint32_t>();

  const auto& L0 = j.at("levels").at(0);
  tp.SPm = L0.at("SPm").get<uint32_t>();
  tp.SPn = L0.at("SPn").get<uint32_t>();
  tp.TPm = L0.at("TPm").get<uint32_t>();
  tp.TPk = L0.at("TPk").get<uint32_t>();
  tp.TPn = L0.at("TPn").get<uint32_t>();
  tp.TM  = L0.at("TM").get<uint32_t>();
  tp.TK  = L0.at("TK").get<uint32_t>();
  tp.TN  = L0.at("TN").get<uint32_t>();

  if (L0.contains("tpOrder")) {
    for (auto& v : L0["tpOrder"]) tp.tpOrder.push_back(v.get<uint32_t>());
  }
  return tp;
}

// Write structured JSON result to a file for machine-readable log parsing.
// Energy fields are set to -1 when RAPL measurement is unavailable.
// Extended diagnostics struct to avoid long parameter lists.
struct ExtendedDiag {
  double idle_post_mw = -1.0;
  double bracket_idle_mean_mw = -1.0;
  double batch_energy_cv_pct = -1.0;
  double batch_step_cv_pct = -1.0;
  double batch_best_wall_s = -1.0;
  double batch_best_active_uj = -1.0;
  double core_energy_per_iter_uj = -1.0;
};

static inline void writeJsonResult(const std::string &path, const std::string &status,
                                   int errors, int iterations, int warmup,
                                   double avgUs, double minUs, double maxUs,
                                   double stepAvgUs, double stepMinUs, double stepMaxUs,
                                   double idlePkgMw, double activePkgMw,
                                   double npuPowerMw, double npuEnergyUj,
                                   double npuEnergyPerIterUj, double wallElapsedS,
                                   int nBatches = 0, int nInner = 0,
                                   double batchMinAvgUs = -1.0,
                                   double batchMinStepAvgUs = -1.0,
                                   double batchMinEnergyPerIterUj = -1.0,
                                   const ExtendedDiag &diag = {}) {
  json j;
  j["status"] = status;
  j["errors"] = errors;
  j["iterations"] = iterations;
  j["warmup"] = warmup;
  j["avg_us"] = avgUs;
  j["min_us"] = minUs;
  j["max_us"] = maxUs;
  j["step_avg_us"] = stepAvgUs;
  j["step_min_us"] = stepMinUs;
  j["step_max_us"] = stepMaxUs;
  j["idle_pkg_mw"] = idlePkgMw;
  j["active_pkg_mw"] = activePkgMw;
  j["npu_power_mw"] = npuPowerMw;
  j["npu_energy_uj"] = npuEnergyUj;
  j["npu_energy_per_iter_uj"] = npuEnergyPerIterUj;
  j["wall_elapsed_s"] = wallElapsedS;
  j["n_batches"] = nBatches;
  j["n_inner"] = nInner;
  j["batch_min_avg_us"] = batchMinAvgUs;
  j["batch_min_step_avg_us"] = batchMinStepAvgUs;
  j["batch_min_energy_per_iter_uj"] = batchMinEnergyPerIterUj;
  // Extended diagnostics
  j["idle_post_mw"] = diag.idle_post_mw;
  j["bracket_idle_mean_mw"] = diag.bracket_idle_mean_mw;
  j["batch_energy_cv_pct"] = diag.batch_energy_cv_pct;
  j["batch_step_cv_pct"] = diag.batch_step_cv_pct;
  j["batch_best_wall_s"] = diag.batch_best_wall_s;
  j["batch_best_active_uj"] = diag.batch_best_active_uj;
  j["core_energy_per_iter_uj"] = diag.core_energy_per_iter_uj;

  std::ofstream ofs(path);
  if (!ofs) {
    std::cerr << "Warning: cannot write JSON result to " << path << "\n";
    return;
  }
  ofs << j.dump(2) << "\n";
}

#endif // TILING_PARAM_H
