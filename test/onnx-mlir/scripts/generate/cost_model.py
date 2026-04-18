#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cost_model.py — Exhaustive search + constraint filtering for spatio-temporal
parallelization on XDNA2 AIE arrays.

Stage 1: Enumerate all (numCores, SPm, SPn, TPm, TPk, TPn) candidates.
Stage 2: Apply hardware/software constraints to filter valid candidates.
Stage 3: Cost model evaluation (performance + energy).
Stage 4: EDP-based optimal selection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiling_common import (                       # noqa: E402
    DEFAULT_OP_PATH, DEFAULT_SYS_PATH, DEFAULT_CALIB_PATH,
    CTILE_RESERVED_BYTES, ELEM_SIZE_MAP, MMUL_R, MMUL_S, MMUL_T,
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
    OpCase, SystemInfo, CalibCoeffs, DEFAULT_COEFFS,
    divisors, factor_pairs, ws_bytes,
    load_op_list, load_system_info, load_calibration, write_tc_list,
    build_tp_order, build_metadata,
)
import models as _models                          # noqa: E402

# Hardware coefficients for XDNA2 (Ryzen AI 9 HX 370, Strix Point, TSMC N4P).
# Sources: AMD XDNA2 spec (256 MACs/cycle BF16 per tile, 32 tiles, ~1.5 GHz),
#          LPDDR5X-7500 measured bandwidth ~80 GB/s (Chips and Cheese),
#          Horowitz 2014 scaled to N4P for energy estimates.
PEAK_MACS       = 256    # MACs/Cycle/Tile (BF16)
BANDWIDTH_BPC   = 4      # Bytes/Cycle (NoC stream bandwidth per channel)
ALPHA_CYCLES    = 20     # Cycles per temporal iteration (pipeline drain/fill)
E_MAC_PJ        = 0.2    # pJ per MAC (TSMC N4P estimate)
E_DRAM_PJ       = 40     # pJ per Byte DRAM access (LPDDR5X estimate)
P_STATIC_PJ     = 27     # pJ/Cycle per Tile (default mode, 0.04W @ 1.5 GHz)


@dataclass
class Candidate:
    """A single spatio-temporal parallelization configuration."""
    # Spatial
    num_cores: int
    num_columns: int
    SPm: int
    SPn: int
    # Temporal
    TPm: int
    TPk: int
    TPn: int
    # Tile sizes (derived)
    TM: int
    TK: int
    TN: int
    # Working set
    ws_bytes: int = 0

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn


@dataclass
class CostResult:
    """Cost model evaluation result for a candidate + tpOrder pair."""
    candidate: Candidate
    tp_order: int           # 0=M, 1=N, 2=K (innermost temporal axis)
    # Performance (Cycles)
    t_comp: float
    t_comm: float
    t_overhead: float
    t_total: float
    # Energy (pJ)
    e_dynamic_comp: float
    e_dynamic_comm: float
    e_static: float
    e_total: float
    # EDP (pJ * Cycles)
    edp: float


# ============================================================
# Stage 1: Exhaustive enumeration
# ============================================================
def enumerate_candidates(
    op: OpCase, sys_info: SystemInfo,
    exclude_cores: Optional[set] = None,
) -> List[Candidate]:
    """
    Enumerate ALL (num_cores, SPm, SPn, TPm, TPk, TPn) combinations.

    Loop hierarchy (resource usage as the primary independent variable):
      1. num_cores  ∈ [1, C]         — resource budget (PE count)
      2. SPm        ∈ [1, num_cores]  — spatial M-axis split
         SPn        = num_cores / SPm  (determined, not a free variable)
      3. TPm ∈ [1, M/SPm], TPk ∈ [1, K], TPn ∈ [1, N/SPn] — temporal splits
         (exact ranges depend on spatial split; M/SPm ≤ M, N/SPn ≤ N)

    Theoretical search space (upper bound):
      Total spatial iterations = ∑_{c=1}^{C} c = C(C+1)/2 = O(C²).
      Temporal ranges are bounded by M, K, N (since M/SPm ≤ M, N/SPn ≤ N).
      Upper bound: O(C² × M × K × N).

    Implementation uses divisor ranges as a lossless compression:
      - SPm × SPn via factor_pairs(num_cores)  — only integer SPn values
      - TPm/TPk/TPn via divisors(M/K/N)        — non-divisors always fail C2/C3
      Using divisors of M (not M/SPm) is safe: values that do not divide
      M/SPm are eliminated by constraint C3, so the result set is identical.

    Actual complexity: O(C · d(C) · d(M) · d(K) · d(N))
      d(n) = number of divisors of n  (average O(ln n))

    All hardware constraints are deferred to Stage 2 filters.
    """
    candidates: List[Candidate] = []
    eb = op.elem_bytes

    divs_M = divisors(op.M)
    divs_K = divisors(op.K)
    divs_N = divisors(op.N)

    for num_cores in range(1, sys_info.total_cores + 1):
        if exclude_cores and num_cores in exclude_cores:
            continue
        for SPm, SPn in factor_pairs(num_cores):
            for TPm in divs_M:
                for TPk in divs_K:
                    for TPn in divs_N:
                        # Tile sizes — 0 when spatial×temporal does not
                        # evenly divide the problem dimension.
                        sp_tp_m = SPm * TPm
                        sp_tp_n = SPn * TPn
                        TM = op.M // sp_tp_m if op.M % sp_tp_m == 0 else 0
                        TK = op.K // TPk
                        TN = op.N // sp_tp_n if op.N % sp_tp_n == 0 else 0
                        ws = ws_bytes(TM, TK, TN, eb) if (TM > 0 and TN > 0) else 0
                        num_cols = math.ceil(num_cores / sys_info.comp_tiles_per_col)

                        candidates.append(Candidate(
                            num_cores=num_cores,
                            num_columns=num_cols,
                            SPm=SPm, SPn=SPn,
                            TPm=TPm, TPk=TPk, TPn=TPn,
                            TM=TM, TK=TK, TN=TN,
                            ws_bytes=ws,
                        ))

    return candidates


# ============================================================
# Stage 2: Constraint filters
# ============================================================
def c1_memory(c: Candidate, ct_limit: int) -> bool:
    """C1: Working set must fit in compute tile memory."""
    return c.ws_bytes <= ct_limit


def c2_spatial_divisibility(c: Candidate, op: OpCase) -> bool:
    """C2: Spatial split must evenly divide M and N."""
    return (op.M % c.SPm == 0) and (op.N % c.SPn == 0)


def c3_temporal_divisibility(c: Candidate, op: OpCase) -> bool:
    """C3: Temporal split must evenly divide per-core block sizes."""
    M0 = op.M // c.SPm
    K0 = op.K
    N0 = op.N // c.SPn
    return (M0 % c.TPm == 0) and (K0 % c.TPk == 0) and (N0 % c.TPn == 0)


def c4_column_alignment(c: Candidate, sys_info: SystemInfo) -> bool:
    """C4: Core count must be a multiple of comp_tiles_per_col (column power gating)."""
    return c.num_cores % sys_info.comp_tiles_per_col == 0


def c5_mmul_shape(c: Candidate) -> bool:
    """C5: Tile dimensions must satisfy bf16 mmul<4,8,8> 2x2 expansion alignment."""
    return (c.TM % (2 * MMUL_R) == 0) and (c.TK % MMUL_S == 0) and (c.TN % (2 * MMUL_T) == 0)


@dataclass
class FilterResult:
    """Tracks how many candidates each constraint removes."""
    name: str
    before: int
    after: int

    @property
    def removed(self) -> int:
        return self.before - self.after


def filter_candidates(
    candidates: List[Candidate],
    op: OpCase,
    sys_info: SystemInfo,
) -> Tuple[List[Candidate], List[FilterResult]]:
    """
    Apply all constraints sequentially and track the filtering effect of each.
    """
    ct_limit = sys_info.ct_usable_bytes
    results: List[FilterResult] = []

    filters = [
        ("C1: memory",               lambda c: c1_memory(c, ct_limit)),
        ("C2: spatial divisibility",  lambda c: c2_spatial_divisibility(c, op)),
        ("C3: temporal divisibility", lambda c: c3_temporal_divisibility(c, op)),
        ("C4: column alignment",      lambda c: c4_column_alignment(c, sys_info)),
        ("C5: mmul shape",            lambda c: c5_mmul_shape(c)),
    ]

    current = candidates
    for name, fn in filters:
        before = len(current)
        current = [c for c in current if fn(c)]
        results.append(FilterResult(name=name, before=before, after=len(current)))

    return current, results


# ============================================================
# Stage 3: Cost model evaluation
# ============================================================
def total_data_bytes(op: OpCase, c: Candidate, tp_order: int) -> int:
    """Total data transfer volume (bytes). Delegates to models.py."""
    return _models.total_data_bytes(
        op.M, op.K, op.N, op.elem_bytes,
        c.SPm, c.SPn, c.TPm, c.TPk, c.TPn, tp_order)


# --- Performance functions (unit: Cycles) ---

def perf_compute(
    op: OpCase, c: Candidate, coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """T_comp: compute time assuming all cores run in parallel."""
    eff = coeffs.eff_macs if coeffs else PEAK_MACS
    return (op.M * op.N * op.K) / (c.SPm * c.SPn * eff)


def perf_comm(
    op: OpCase, c: Candidate, tp_order: int,
    coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """T_comm: pure data transfer time at stream bandwidth."""
    bw = coeffs.bw_eff_bpc if coeffs else BANDWIDTH_BPC
    return total_data_bytes(op, c, tp_order) / bw


def _d_total(c: Candidate, tp_order: int) -> float:
    """Refined total DMA descriptor setups (D_total) for Candidate c."""
    return _models.d_total(
        c.SPm, c.SPn, c.num_cores,
        c.TPm, c.TPk, c.TPn, c.tp_total, tp_order)


def _dma_ops_per_step(c: Candidate, tp_order: int) -> int:
    """Delegates to models.dma_ops_per_step()."""
    return _models.dma_ops_per_step(c.SPm, c.SPn, c.num_cores, tp_order)


def perf_overhead(
    c: Candidate, coeffs: Optional[CalibCoeffs] = None,
    tp_order: int = TP_AXIS_K,
) -> float:
    """T_overhead: delegates to the appropriate PerfModel variant.

    DMA-Bottleneck (v16): D * max(L_SETUP, avg_tile/BW) + sync overhead
    DMA-Refined (v13+):   L_DMA * D_total + sync overhead
    Core-Sync (v9):       L_DMA * N_dma * TP_total + sync overhead
    """
    if coeffs and coeffs.calibrated:
        if coeffs.perf_model == "DMA-Bottleneck":
            d_tot = _d_total(c, tp_order)
            op_dummy = OpCase(M=1, K=1, N=1, elem_type="bf16")
            # Need actual data_bytes for avg_tile calculation
            # Caller passes data_bytes=0; compute here for the overhead
            l_setup = coeffs.l_setup_cy if hasattr(coeffs, 'l_setup_cy') else coeffs.l_dma_cy
            _, t_dma, t_sync = _models.PerfModel.components_dma_bottleneck(
                macs=0, data_bytes=0,
                n_cores=c.num_cores, tp_total=c.tp_total, d_total_val=d_tot,
                eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
                l_sync=coeffs.l_sync_cy, l_core=coeffs.l_core_cy,
                l_setup=l_setup, l_startup=coeffs.l_startup_cy)
            return t_dma + t_sync
        elif coeffs.perf_model == "DMA-Refined":
            d_tot = _d_total(c, tp_order)
            _, _, t_ovh = _models.PerfModel.components_dma_refined(
                macs=0, data_bytes=0,
                n_cores=c.num_cores, tp_total=c.tp_total, d_total_val=d_tot,
                eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
                l_sync=coeffs.l_sync_cy, l_core=coeffs.l_core_cy,
                l_dma=coeffs.l_dma_cy, l_startup=coeffs.l_startup_cy)
            return t_ovh
        else:
            n_dma = _dma_ops_per_step(c, tp_order)
            _, _, t_ovh = _models.PerfModel.components_v9(
                macs=0, data_bytes=0,
                n_cores=c.num_cores, tp_total=c.tp_total, n_dma=n_dma,
                eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
                l_sync=coeffs.l_sync_cy, l_core=coeffs.l_core_cy,
                l_dma=coeffs.l_dma_cy, l_startup=coeffs.l_startup_cy)
            return t_ovh
    return ALPHA_CYCLES * (c.TPm * c.TPn * c.TPk)


# --- Energy functions (unit: pJ) ---
# When coeffs.energy_calibrated is True, energy_total_calibrated() is used
# instead of the three component functions below.

def energy_dynamic_comp(
    op: OpCase, coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """E_dynamic_comp: total MAC energy (constant across candidates)."""
    if coeffs and coeffs.energy_calibrated:
        e_mac = coeffs.energy_params.get("e_mac_pj", E_MAC_PJ)
    else:
        e_mac = E_MAC_PJ
    return op.M * op.N * op.K * e_mac


def energy_dynamic_comm(
    op: OpCase, c: Candidate, tp_order: int,
    coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """E_dynamic_comm: DRAM access energy proportional to transfer volume."""
    if coeffs and coeffs.energy_calibrated:
        e_dram = coeffs.energy_params.get("e_byte_pj", E_DRAM_PJ)
    else:
        e_dram = E_DRAM_PJ
    return total_data_bytes(op, c, tp_order) * e_dram


def energy_static(
    c: Candidate, t_total: float,
    coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """E_static: leakage energy for active tiles over total execution time."""
    if coeffs and coeffs.energy_calibrated:
        p_static = coeffs.energy_params.get("p_static_pj", P_STATIC_PJ)
    else:
        p_static = P_STATIC_PJ
    return (c.SPm * c.SPn) * p_static * t_total


def energy_total_calibrated(
    op: OpCase, c: Candidate, tp_order: int,
    t_comp: float, t_comm: float, t_overhead: float, t_total: float,
    coeffs: CalibCoeffs,
) -> Tuple[float, float, float, float]:
    """Compute energy using calibrated model. Delegates to models.EnergyModel.

    Returns (e_comp_pj, e_comm_pj, e_static_pj, e_total_pj).
    Component breakdown is approximate for models that don't decompose.
    """
    features = {
        "macs": op.M * op.N * op.K,
        "data_bytes": total_data_bytes(op, c, tp_order),
        "n_cores": c.SPm * c.SPn,
        "tp_total": c.tp_total,
        "n_dma": _dma_ops_per_step(c, tp_order),
        "d_total": _d_total(c, tp_order),
        "t_total_cy": t_total,
        "t_comp_cy": t_comp,
        "t_comm_cy": t_comm,
        "t_overhead_cy": t_overhead,
    }
    e_total = _models.EnergyModel.predict(
        coeffs.energy_model, coeffs.energy_params, features)

    # Approximate component breakdown for reporting
    e_mac_pj = coeffs.energy_params.get("e_mac_pj", E_MAC_PJ)
    e_dram_pj = coeffs.energy_params.get("e_dram_pj",
                coeffs.energy_params.get("e_byte_pj", E_DRAM_PJ))
    e_comp = features["macs"] * e_mac_pj
    e_comm = features["data_bytes"] * e_dram_pj
    e_static = max(0.0, e_total - e_comp - e_comm)
    return e_comp, e_comm, e_static, e_total


# --- Combined evaluation ---

def evaluate_candidate(
    op: OpCase, c: Candidate, tp_order: int,
    coeffs: Optional[CalibCoeffs] = None,
) -> CostResult:
    """Evaluate a single candidate with a specific tpOrder."""
    if coeffs and coeffs.calibrated and coeffs.perf_model == "DMA-Bottleneck":
        # v16: T = T_comp + D*max(L_SETUP, avg_tile/BW) + sync + startup
        # T_comm is integrated into the DMA bottleneck term (no separate T_comm)
        l_setup = coeffs.l_setup_cy if hasattr(coeffs, 'l_setup_cy') else coeffs.l_dma_cy
        data_bytes = total_data_bytes(op, c, tp_order)
        d_tot = _d_total(c, tp_order)
        tc_cy, td_cy, to_cy = _models.PerfModel.components_dma_bottleneck(
            macs=op.M * op.N * op.K,
            data_bytes=data_bytes,
            n_cores=c.num_cores, tp_total=c.tp_total, d_total_val=d_tot,
            eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
            l_sync=coeffs.l_sync_cy, l_core=coeffs.l_core_cy,
            l_setup=l_setup, l_startup=coeffs.l_startup_cy)
        tc = tc_cy  # T_comp in cycles
        tm = td_cy  # T_dma in cycles (replaces T_comm)
        to = to_cy  # T_sync + T_startup in cycles
        tt = tc + tm + to
    else:
        tc = perf_compute(op, c, coeffs)
        tm = perf_comm(op, c, tp_order, coeffs)
        to = perf_overhead(c, coeffs, tp_order)
        tt = tc + tm + to

    if coeffs and coeffs.energy_calibrated:
        edc, edm, es, et = energy_total_calibrated(
            op, c, tp_order, tc, tm, to, tt, coeffs)
    else:
        edc = energy_dynamic_comp(op, coeffs)
        edm = energy_dynamic_comm(op, c, tp_order, coeffs)
        es = energy_static(c, tt, coeffs)
        et = edc + edm + es

    return CostResult(
        candidate=c, tp_order=tp_order,
        t_comp=tc, t_comm=tm, t_overhead=to, t_total=tt,
        e_dynamic_comp=edc, e_dynamic_comm=edm, e_static=es, e_total=et,
        edp=tt * et,
    )


# ============================================================
# Stage 4: EDP-based optimal selection
# ============================================================
def select_optimal(
    valid: List[Candidate], op: OpCase,
    coeffs: Optional[CalibCoeffs] = None,
    keep_all_tporders: bool = False,
) -> List[CostResult]:
    """
    For each candidate, evaluate all 3 tpOrders, keep the one with lowest EDP.
    Return the full list sorted by EDP ascending.

    If keep_all_tporders=True, also returns a dict mapping candidate index
    to all 3 CostResults (for tpOrder verification).
    """
    best_per_candidate: List[CostResult] = []
    all_tporder_results: Dict[int, List[CostResult]] = {}
    for i, c in enumerate(valid):
        results = [evaluate_candidate(op, c, tpo, coeffs) for tpo in (0, 1, 2)]
        best = min(results, key=lambda r: r.edp)
        best_per_candidate.append(best)
        if keep_all_tporders:
            all_tporder_results[i] = results

    best_per_candidate.sort(key=lambda r: r.edp)
    if keep_all_tporders:
        return best_per_candidate, all_tporder_results
    return best_per_candidate


def select_tporder_verify_cases(
    all_ranked: Dict[int, List[CostResult]],
    all_tporder_data: Dict[int, Dict[int, List[CostResult]]],
    ops: list,
    n_verify: int = 19,
) -> List[Dict[str, Any]]:
    """Select tpOrder verification cases: 1 per workload, pick the candidate
    with the largest EDP ratio across 3 tpOrders. Return extra tc entries
    for the non-best tpOrders (2 per selected candidate).
    """
    verify_tcs: List[Dict[str, Any]] = []
    n_verify = min(n_verify, len(all_ranked))

    # For each workload, find the candidate with max tpOrder EDP spread
    workload_picks = []
    for op_idx, ranked in all_ranked.items():
        tpo_data = all_tporder_data.get(op_idx, {})
        if not tpo_data:
            continue
        # Build a map from candidate identity to tporder results
        # ranked contains best-per-candidate; we need to find which
        # valid-index each ranked entry maps to
        best_ratio = 0.0
        best_cand_results = None
        best_cr = None
        for valid_idx, results in tpo_data.items():
            edps = [r.edp for r in results]
            if min(edps) <= 0:
                continue
            ratio = max(edps) / min(edps)
            if ratio > best_ratio:
                best_ratio = ratio
                best_cand_results = results
                best_cr = min(results, key=lambda r: r.edp)

        if best_cand_results and best_cr:
            workload_picks.append((op_idx, best_ratio, best_cr, best_cand_results))

    # Sort by ratio descending, take top n_verify
    workload_picks.sort(key=lambda x: x[1], reverse=True)
    selected = workload_picks[:n_verify]

    for op_idx, ratio, best_cr, all_results in selected:
        op = ops[op_idx]
        best_tpo = best_cr.tp_order
        for r in all_results:
            if r.tp_order != best_tpo:
                tc = cost_result_to_tc(op, r)
                tc["tporder_verify"] = True
                verify_tcs.append(tc)
        print(f"  tpOrder verify: M{op.M}_K{op.K}_N{op.N} "
              f"SP=({best_cr.candidate.SPm},{best_cr.candidate.SPn}) "
              f"ratio={ratio:.2f}")

    return verify_tcs


# ============================================================
# Reporting
# ============================================================
def print_search_summary(
    op: OpCase,
    total: int,
    valid: List[Candidate],
    filter_results: List[FilterResult],
) -> None:
    print(f"\n{'='*60}")
    print(f"Op: M={op.M}, K={op.K}, N={op.N}, type={op.elem_type}")
    print(f"{'='*60}")
    print(f"Total enumerated:  {total}")

    for fr in filter_results:
        pct = fr.removed / fr.before * 100 if fr.before > 0 else 0
        print(f"  {fr.name:<30s}  {fr.before:>6d} -> {fr.after:>6d}  "
              f"(removed {fr.removed:>5d}, {pct:5.1f}%)")

    print(f"Valid candidates:  {len(valid)}")

    if valid:
        # Summary by num_cores
        core_counts: Dict[int, int] = {}
        for c in valid:
            core_counts[c.num_cores] = core_counts.get(c.num_cores, 0) + 1
        print(f"\nCandidates by core count:")
        for nc in sorted(core_counts):
            print(f"  {nc:>3d} cores: {core_counts[nc]:>5d} candidates")

        # Show a few examples
        print(f"\nFirst 5 candidates:")
        for c in valid[:5]:
            print(f"  cores={c.num_cores} SP=({c.SPm},{c.SPn}) "
                  f"TP=({c.TPm},{c.TPk},{c.TPn}) "
                  f"Tile=({c.TM},{c.TK},{c.TN}) ws={c.ws_bytes}B")


TP_AXIS_NAMES = {0: "M", 1: "N", 2: "K"}


def print_cost_summary(op: OpCase, ranked: List[CostResult]) -> None:
    """Print Stage 3+4 results: best candidates by core count and overall."""
    if not ranked:
        print("\n  No valid candidates to evaluate.")
        return

    print(f"\n--- Stage 3+4: Cost Model & EDP Ranking ---")

    # Best per core count
    best_by_cores: Dict[int, CostResult] = {}
    for r in ranked:
        nc = r.candidate.num_cores
        if nc not in best_by_cores:
            best_by_cores[nc] = r

    print(f"\n  Best EDP per core count:")
    print(f"  {'cores':>5s}  {'SP':>7s}  {'TP':>11s}  {'tpOrd':>5s}  "
          f"{'T_total':>12s}  {'E_total':>12s}  {'EDP':>14s}")
    for nc in sorted(best_by_cores):
        r = best_by_cores[nc]
        c = r.candidate
        print(f"  {nc:>5d}  ({c.SPm:>2d},{c.SPn:>2d})  "
              f"({c.TPm:>2d},{c.TPk:>2d},{c.TPn:>2d})  "
              f"    {TP_AXIS_NAMES[r.tp_order]}  "
              f"{r.t_total:>12.1f}  {r.e_total:>12.1f}  {r.edp:>14.1f}")

    # Overall top 5
    print(f"\n  Top 5 by EDP:")
    for i, r in enumerate(ranked[:5]):
        c = r.candidate
        print(f"  #{i+1}  cores={c.num_cores} SP=({c.SPm},{c.SPn}) "
              f"TP=({c.TPm},{c.TPk},{c.TPn}) "
              f"tpOrder={TP_AXIS_NAMES[r.tp_order]}  "
              f"T={r.t_total:.1f} E={r.e_total:.1f} EDP={r.edp:.1f}")


# ============================================================
# tc_list.json output (flat schema for the build/run pipeline)
# ============================================================
def cost_result_to_tc(
    op: OpCase, cr: CostResult, edp_rank: int = 0,
) -> Dict[str, Any]:
    """Convert a CostResult to a flat tc.json entry for the pipeline."""
    c = cr.candidate
    tp_order_full = build_tp_order(cr.tp_order)
    entry: Dict[str, Any] = {
        "M": op.M, "K": op.K, "N": op.N,
        "elemType": op.elem_type,
        "numCores": c.num_cores,
        "doubleBuffer": False,
        "t_total_pred": round(cr.t_total, 2),
        "levels": [
            {
                "SPm": c.SPm, "SPn": c.SPn,
                "TPm": c.TPm, "TPk": c.TPk, "TPn": c.TPn,
                "TM": c.TM, "TK": c.TK, "TN": c.TN,
                "tpOrder": tp_order_full,
            }
        ],
    }
    if edp_rank > 0:
        entry["edp_rank"] = edp_rank
        entry["edp_pred"] = round(cr.edp, 2)
        entry["e_total_pred"] = round(cr.e_total, 2)
    return entry


# ============================================================
# CLI
# ============================================================
def select_validation_candidates(
    ranked: List[CostResult],
    top_n: int = 0,
    per_core_top: int = 0,
    random_sample: int = 0,
) -> List[CostResult]:
    """Select validation candidates from EDP-ranked list.

    If all sampling params are 0, returns all candidates (sorted by T_total).
    Otherwise, builds a deduplicated set from:
      1. EDP Top-N overall
      2. EDP Top-K per core count
      3. Random M from remaining (seed=42 for reproducibility)
    Returns selected candidates sorted by EDP ascending.
    """
    if top_n <= 0 and per_core_top <= 0 and random_sample <= 0:
        return sorted(ranked, key=lambda r: r.t_total)

    selected_indices: set = set()

    # 1. EDP Top-N overall (ranked is already sorted by EDP ascending)
    if top_n > 0:
        for i in range(min(top_n, len(ranked))):
            selected_indices.add(i)

    # 2. Per-core-count Top-K
    if per_core_top > 0:
        core_counts: Dict[int, List[int]] = {}
        for i, r in enumerate(ranked):
            nc = r.candidate.num_cores
            if nc not in core_counts:
                core_counts[nc] = []
            core_counts[nc].append(i)
        for nc in sorted(core_counts):
            for i in core_counts[nc][:per_core_top]:
                selected_indices.add(i)

    # 3. Random sample from remaining
    if random_sample > 0:
        remaining = [i for i in range(len(ranked)) if i not in selected_indices]
        rng = random.Random(42)
        k = min(random_sample, len(remaining))
        for i in rng.sample(remaining, k):
            selected_indices.add(i)

    result = [ranked[i] for i in sorted(selected_indices)]
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exhaustive search + constraint filtering for parallelization configs")
    p.add_argument("--op", default=str(DEFAULT_OP_PATH),
                   help="Path to op_list.json")
    p.add_argument("--sys", default=str(DEFAULT_SYS_PATH),
                   help="Path to xdna2_info.json")
    p.add_argument("--out", default="",
                   help="Path to write cost results JSON (optional)")
    p.add_argument("--validate", default="",
                   help="Path to write validation tc_list.json (optional)")
    p.add_argument("--op-index", type=int, default=-1,
                   help="Run only this 0-based op index (-1 = all)")
    p.add_argument("--calib", default=str(DEFAULT_CALIB_PATH),
                   help="Path to calibration.json (empty string to skip)")
    p.add_argument("--top-n", type=int, default=0,
                   help="Validation: select EDP Top-N per size (0=all)")
    p.add_argument("--per-core-top", type=int, default=0,
                   help="Validation: select Top-K per core count per size")
    p.add_argument("--random-sample", type=int, default=0,
                   help="Validation: add N random candidates from remaining")
    p.add_argument("--tporder-verify", type=int, default=0,
                   help="Add tpOrder verification cases: N workloads x 2 extra tpOrders")
    p.add_argument("--exclude-cores", default="",
                   help="Comma-separated numCores values to exclude from enumeration "
                        "(e.g., '24' to skip all 24-core configs)")
    return p.parse_args(argv)


def process_op(
    op: OpCase, sys_info: SystemInfo,
    coeffs: Optional[CalibCoeffs] = None,
    keep_all_tporders: bool = False,
    exclude_cores: Optional[set] = None,
):
    """Run Stage 1 through Stage 4 for a single op case."""
    all_candidates = enumerate_candidates(op, sys_info, exclude_cores=exclude_cores)
    valid, filter_results = filter_candidates(all_candidates, op, sys_info)
    print_search_summary(op, len(all_candidates), valid, filter_results)
    if keep_all_tporders:
        ranked, tpo_data = select_optimal(valid, op, coeffs, keep_all_tporders=True)
        print_cost_summary(op, ranked)
        return ranked, tpo_data
    ranked = select_optimal(valid, op, coeffs)
    print_cost_summary(op, ranked)
    return ranked


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    op_path = Path(args.op).resolve()
    sys_path = Path(args.sys).resolve()

    try:
        ops = load_op_list(op_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    try:
        sys_info = load_system_info(sys_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # Load calibration coefficients
    if args.calib:
        coeffs = load_calibration(Path(args.calib).resolve())
    else:
        coeffs = DEFAULT_COEFFS

    print(f"[INFO] {len(ops)} ops from {op_path}")
    print(f"[INFO] HW: {sys_info.total_cores} cores, "
          f"{sys_info.comp_tiles_per_col} tiles/col, "
          f"{sys_info.max_columns} cols, "
          f"{sys_info.spm_size_bytes}B/tile "
          f"(usable {sys_info.ct_usable_bytes}B)")
    if coeffs.calibrated:
        print(f"[INFO] Calibration: eff_macs={coeffs.eff_macs}, "
              f"l_sync={coeffs.l_sync_cy:.0f}, "
              f"l_core={coeffs.l_core_cy:.0f}, "
              f"l_startup={coeffs.l_startup_cy:.0f}")
    else:
        print(f"[INFO] Calibration: not loaded (using defaults)")

    # Select ops to process
    if args.op_index >= 0:
        if args.op_index >= len(ops):
            print(f"[ERROR] op-index {args.op_index} out of range", file=sys.stderr)
            return 1
        targets = [(args.op_index, ops[args.op_index])]
    else:
        targets = list(enumerate(ops))

    need_tporder = args.tporder_verify > 0
    exclude_cores_set = set()
    if args.exclude_cores:
        exclude_cores_set = {int(x) for x in args.exclude_cores.split(",") if x.strip()}
        print(f"[info] excluding numCores: {sorted(exclude_cores_set)}")
    all_ranked: Dict[int, List[CostResult]] = {}
    all_tporder_data: Dict[int, Dict[int, List[CostResult]]] = {}
    for idx, op in targets:
        if need_tporder:
            ranked, tpo_data = process_op(op, sys_info, coeffs, keep_all_tporders=True,
                                          exclude_cores=exclude_cores_set)
            all_tporder_data[idx] = tpo_data
        else:
            ranked = process_op(op, sys_info, coeffs, exclude_cores=exclude_cores_set)
        all_ranked[idx] = ranked

    # Write output if requested
    if args.out:
        out_path = Path(args.out).resolve()
        output = {}
        for idx, ranked in all_ranked.items():
            op = ops[idx]
            key = f"M{op.M}_K{op.K}_N{op.N}"
            output[key] = {
                "op": {"M": op.M, "K": op.K, "N": op.N, "elemType": op.elem_type},
                "num_ranked": len(ranked),
                "ranked": [
                    {
                        **asdict(r.candidate),
                        "tp_order": r.tp_order,
                        "t_comp": r.t_comp,
                        "t_comm": r.t_comm,
                        "t_overhead": r.t_overhead,
                        "t_total": r.t_total,
                        "e_dynamic_comp": r.e_dynamic_comp,
                        "e_dynamic_comm": r.e_dynamic_comm,
                        "e_static": r.e_static,
                        "e_total": r.e_total,
                        "edp": r.edp,
                    }
                    for r in ranked
                ],
            }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\n[INFO] Wrote {out_path}")

    # Write validation tc_list.json if requested
    if args.validate:
        val_path = Path(args.validate).resolve()
        tc_cases: List[Dict[str, Any]] = []
        for idx, ranked in all_ranked.items():
            op = ops[idx]
            selected = select_validation_candidates(
                ranked,
                top_n=args.top_n,
                per_core_top=args.per_core_top,
                random_sample=args.random_sample,
            )
            for cr in selected:
                edp_rank = ranked.index(cr) + 1
                tc_cases.append(cost_result_to_tc(op, cr, edp_rank=edp_rank))
            sampling = ""
            if args.top_n > 0 or args.per_core_top > 0 or args.random_sample > 0:
                sampling = (f" (top-{args.top_n} + core-top-{args.per_core_top}"
                            f" + rand-{args.random_sample}"
                            f" from {len(ranked)} total)")
            print(f"\n[INFO] Op M{op.M}_K{op.K}_N{op.N}: "
                  f"{len(selected)} validation candidates selected{sampling}")
        # Add tpOrder verification cases
        if args.tporder_verify > 0 and all_tporder_data:
            print(f"\n[INFO] Selecting tpOrder verification cases...")
            verify_tcs = select_tporder_verify_cases(
                all_ranked, all_tporder_data, ops,
                n_verify=args.tporder_verify,
            )
            tc_cases.extend(verify_tcs)
            print(f"[INFO] Added {len(verify_tcs)} tpOrder verification cases")

        meta = build_metadata(calib_path=Path(args.calib), coeffs=coeffs)
        write_tc_list(tc_cases, val_path, metadata=meta)
        print(f"[INFO] Wrote {val_path} ({len(tc_cases)} cases total)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
