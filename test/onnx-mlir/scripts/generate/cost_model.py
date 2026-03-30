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
    build_tp_order,
)

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
def enumerate_candidates(op: OpCase, sys_info: SystemInfo) -> List[Candidate]:
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
    """
    Total data transfer volume (bytes) across all cores, with reuse applied.

    The innermost temporal axis determines which operands stay in tile memory
    (temporal reuse) and which are shared across spatial cores (spatial reuse).
    Final transfer = raw transfer / (TR × SR) per operand.
    """
    M, K, N = op.M, op.K, op.N
    eb = op.elem_bytes

    if tp_order == TP_AXIS_M:
        # M innermost: RHS reused across TPm iterations and SPm cores.
        lhs = M * K * c.TPn
        rhs = K * N
        out = 2 * M * N * c.TPk
    elif tp_order == TP_AXIS_N:
        # N innermost: LHS reused across TPn iterations and SPn cores.
        lhs = M * K
        rhs = K * N * c.TPm
        out = 2 * M * N * c.TPk
    else:
        # K innermost: OUT reused across TPk iterations (local accumulation).
        lhs = M * K * c.TPn
        rhs = K * N * c.TPm
        out = 2 * M * N

    return (lhs + rhs + out) * eb


# --- Performance functions (unit: Cycles) ---

def perf_compute(
    op: OpCase, c: Candidate, coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """T_comp: compute time assuming all cores run in parallel.

    Uses eff_macs (trace-calibrated effective MACs/cycle/tile).
    """
    macs = coeffs.eff_macs if coeffs else PEAK_MACS
    return (op.M * op.N * op.K) / (c.SPm * c.SPn * macs)


def perf_comm(
    op: OpCase, c: Candidate, tp_order: int,
    coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """T_comm: pure data transfer time at stream bandwidth.

    bw_eff_bpc = 4.0 B/cy (single DMA stream bandwidth, fixed).
    Additional DMA overhead (lock, starvation, descriptor setup) is
    captured by T_overhead terms (L_SYNC, L_SYNC2, L_DMA).
    """
    bw = coeffs.bw_eff_bpc if coeffs else BANDWIDTH_BPC
    return total_data_bytes(op, c, tp_order) / bw


def _dma_ops_per_step(c: Candidate, tp_order: int) -> int:
    """Number of unique DMA descriptor setups per temporal iteration.

    Each unique data source/sink requires one DMA descriptor program.
    Balanced SP (SPm ~= SPn) minimizes this count via AM-GM inequality:
      SPm + SPn >= 2*sqrt(N_cores), equality at SPm = SPn.
    """
    if tp_order == TP_AXIS_M:
        return c.SPm + 2 * c.num_cores
    elif tp_order == TP_AXIS_N:
        return c.SPn + 2 * c.num_cores
    else:  # TP_AXIS_K
        return c.SPm + c.SPn


def perf_overhead(
    c: Candidate, coeffs: Optional[CalibCoeffs] = None,
    tp_order: int = TP_AXIS_K,
) -> float:
    """T_overhead: sync + per-core sync + DMA setup + startup cost.

    v9: T_overhead = L_SYNC*TP + L_CORE*P*TP + L_DMA*N_dma*TP + L_STARTUP
      L_SYNC: base per-iteration synchronization cost
      L_CORE: per-core per-iteration barrier cost (O(P) contention)
      L_DMA:  per-DMA-descriptor setup cost per iteration
      L_STARTUP: one-time NPU dispatch overhead
    v8 compat: l_core_cy applied as L_CORE*P (no TP scaling) when
               l_sync2_cy == 0.
    """
    if coeffs and coeffs.calibrated:
        dma_cost = coeffs.l_dma_cy * _dma_ops_per_step(c, tp_order) * c.tp_total
        if coeffs.l_sync2_cy != 0:
            # v9: L_CORE means per-core per-iteration (stored in l_sync2_cy)
            return (coeffs.l_sync_cy * c.tp_total
                    + coeffs.l_sync2_cy * c.num_cores * c.tp_total
                    + dma_cost
                    + coeffs.l_startup_cy)
        else:
            # v8 compat: L_CORE means per-core fixed (no TP scaling)
            return (coeffs.l_sync_cy * c.tp_total
                    + dma_cost
                    + coeffs.l_core_cy * c.num_cores
                    + coeffs.l_startup_cy)
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
    """Compute energy using fully calibrated model (E-A/E-B/E-C/E-D).

    Returns (e_comp_pj, e_comm_pj, e_static_pj, e_total_pj).
    For models that don't decompose into components, e_comp/e_comm/e_static
    are set to 0 and e_total contains the full prediction.
    """
    CLOCK_MHZ = 1500
    params = coeffs.energy_params
    model = coeffs.energy_model

    if model == "E-A":
        p_active_uw = params.get("p_active_uw", 0)
        e_startup_uj = params.get("e_startup_uj", 0)
        t_total_us = t_total / CLOCK_MHZ
        e_total_pj = (p_active_uw * t_total_us / 1e6 + e_startup_uj) * 1e6
        return 0.0, 0.0, 0.0, e_total_pj

    elif model == "E-B":
        p_comp = params.get("p_comp_uw", 0)
        p_dma = params.get("p_dma_uw", 0)
        p_idle = params.get("p_idle_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        e_comp = p_comp * (t_comp / CLOCK_MHZ) / 1e6 * 1e6   # uW*us/1e6=uJ, *1e6=pJ
        e_comm = p_dma * (t_comm / CLOCK_MHZ) / 1e6 * 1e6
        e_static = p_idle * (t_overhead / CLOCK_MHZ) / 1e6 * 1e6
        e_total_pj = e_comp + e_comm + e_static + e_startup * 1e6
        return e_comp, e_comm, e_static, e_total_pj

    elif model == "E-C":
        # Component-based: same structure as theoretical, just different constants
        e_mac = params.get("e_mac_pj", E_MAC_PJ)
        e_byte = params.get("e_byte_pj", E_DRAM_PJ)
        p_static = params.get("p_static_pj", P_STATIC_PJ)
        e_startup = params.get("e_startup_uj", 0)
        e_comp = op.M * op.N * op.K * e_mac
        e_comm = total_data_bytes(op, c, tp_order) * e_byte
        e_st = (c.SPm * c.SPn) * p_static * t_total
        e_total_pj = e_comp + e_comm + e_st + e_startup * 1e6
        return e_comp, e_comm, e_st, e_total_pj

    elif model == "E-D":
        p_core_uw = params.get("p_core_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        t_total_us = t_total / CLOCK_MHZ
        n_cores = c.SPm * c.SPn
        e_total_pj = (p_core_uw * n_cores * t_total_us / 1e6 + e_startup) * 1e6
        return 0.0, 0.0, 0.0, e_total_pj

    elif model == "E-F":
        p_base_uw = params.get("p_base_uw", 0)
        p_core_uw = params.get("p_core_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        t_total_us = t_total / CLOCK_MHZ
        n_cores = c.SPm * c.SPn
        e_total_pj = ((p_base_uw + p_core_uw * n_cores) * t_total_us / 1e6
                      + e_startup) * 1e6
        return 0.0, 0.0, 0.0, e_total_pj

    elif model == "T-A":
        # Calibrated theoretical: E = E_mac*MACs + E_dram*bytes + P_static*N*T
        e_mac = params.get("e_mac_pj", E_MAC_PJ)
        e_dram = params.get("e_dram_pj", E_DRAM_PJ)
        p_static = params.get("p_static_pj", P_STATIC_PJ)
        n_cores = c.SPm * c.SPn
        e_comp = op.M * op.N * op.K * e_mac
        e_comm = total_data_bytes(op, c, tp_order) * e_dram
        e_st = n_cores * p_static * t_total
        return e_comp, e_comm, e_st, e_comp + e_comm + e_st

    elif model == "T-B":
        # Base + per-core power: E = E_mac*MACs + E_dram*bytes + (P_base + P_core*N)*T
        e_mac = params.get("e_mac_pj", E_MAC_PJ)
        e_dram = params.get("e_dram_pj", E_DRAM_PJ)
        p_base_uw = params.get("p_base_uw", 0)
        p_core_uw = params.get("p_core_uw", 0)
        n_cores = c.SPm * c.SPn
        t_total_us = t_total / CLOCK_MHZ
        e_comp = op.M * op.N * op.K * e_mac
        e_comm = total_data_bytes(op, c, tp_order) * e_dram
        # P(uW) * t(us) = uW*us = pJ
        e_st = (p_base_uw + p_core_uw * n_cores) * t_total_us
        return e_comp, e_comm, e_st, e_comp + e_comm + e_st

    elif model == "T-C":
        # Base + per-core + startup: same as T-B + E_startup
        e_mac = params.get("e_mac_pj", E_MAC_PJ)
        e_dram = params.get("e_dram_pj", E_DRAM_PJ)
        p_base_uw = params.get("p_base_uw", 0)
        p_core_uw = params.get("p_core_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        n_cores = c.SPm * c.SPn
        t_total_us = t_total / CLOCK_MHZ
        e_comp = op.M * op.N * op.K * e_mac
        e_comm = total_data_bytes(op, c, tp_order) * e_dram
        e_st = (p_base_uw + p_core_uw * n_cores) * t_total_us
        e_total_pj = e_comp + e_comm + e_st + e_startup * 1e6
        return e_comp, e_comm, e_st, e_total_pj

    else:
        # Unknown model — fall back to theoretical constants
        edc = op.M * op.N * op.K * E_MAC_PJ
        edm = total_data_bytes(op, c, tp_order) * E_DRAM_PJ
        es = (c.SPm * c.SPn) * P_STATIC_PJ * t_total
        return edc, edm, es, edc + edm + es


# --- Combined evaluation ---

def evaluate_candidate(
    op: OpCase, c: Candidate, tp_order: int,
    coeffs: Optional[CalibCoeffs] = None,
) -> CostResult:
    """Evaluate a single candidate with a specific tpOrder."""
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
) -> List[CostResult]:
    """
    For each candidate, evaluate all 3 tpOrders, keep the one with lowest EDP.
    Return the full list sorted by EDP ascending.
    """
    best_per_candidate: List[CostResult] = []
    for c in valid:
        results = [evaluate_candidate(op, c, tpo, coeffs) for tpo in (0, 1, 2)]
        best = min(results, key=lambda r: r.edp)
        best_per_candidate.append(best)

    best_per_candidate.sort(key=lambda r: r.edp)
    return best_per_candidate


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
    return p.parse_args(argv)


def process_op(
    op: OpCase, sys_info: SystemInfo,
    coeffs: Optional[CalibCoeffs] = None,
) -> List[CostResult]:
    """Run Stage 1 through Stage 4 for a single op case."""
    all_candidates = enumerate_candidates(op, sys_info)
    valid, filter_results = filter_candidates(all_candidates, op, sys_info)
    print_search_summary(op, len(all_candidates), valid, filter_results)
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

    all_ranked: Dict[int, List[CostResult]] = {}
    for idx, op in targets:
        ranked = process_op(op, sys_info, coeffs)
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
        write_tc_list(tc_cases, val_path)
        print(f"[INFO] Wrote {val_path} ({len(tc_cases)} cases)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
