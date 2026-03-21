#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_energy_model.py — Validate energy cost model against NPU measurements.

Compares theoretical, calibrated, and T-A/T-B/T-C energy models against
measured data.  Includes EDP core-count optimal verification.

Usage:
    # Validate with wall_elapsed_s filter
    python3 scripts/analyze/validate_energy_model.py \\
        --energy out/calibration/result_energy.csv \\
        --tc out/calibration/tc_list.json \\
        --calib data/calibration.json \\
        --min-wall-s 0.005

    # With cross-validation
    python3 scripts/analyze/validate_energy_model.py \\
        --energy out/calibration/result_energy.csv \\
        --tc out/calibration/tc_list.json \\
        --calib data/calibration.json \\
        --min-wall-s 0.005 \\
        --cross-validate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import (  # noqa: E402
    OpCase, CalibCoeffs,
    load_calibration, DEFAULT_CALIB_PATH,
    load_system_info, DEFAULT_SYS_PATH,
)
from cost_model import (  # noqa: E402
    Candidate, evaluate_candidate, total_data_bytes,
    E_MAC_PJ, E_DRAM_PJ, P_STATIC_PJ,
    PEAK_MACS, BANDWIDTH_BPC, ALPHA_CYCLES,
)

CLOCK_MHZ = 1500
DEFAULT_MIN_WALL_S = 0.005
_COMP_TILES_PER_COL = 4


# ---------------------------------------------------------------------------
# Column mapping helper
# ---------------------------------------------------------------------------
def _col(row: Dict[str, str], *keys: str) -> str:
    """Return first matching column value from a CSV row dict."""
    for k in keys:
        if k in row:
            return row[k]
    raise KeyError(f"None of {keys} found in CSV columns: {list(row.keys())}")


def _load_tp_order_map(tc_path: Path) -> Dict[int, int]:
    """Load tc_list.json and build case_index -> tpOrder[0] map (1-based)."""
    if not tc_path or not tc_path.is_file():
        return {}
    with tc_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    result: Dict[int, int] = {}
    for i, case in enumerate(doc.get("cases", []), start=1):
        levels = case.get("levels", [{}])
        tp_order = levels[0].get("tpOrder", [2, 0, 1])
        result[i] = tp_order[0]
    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EnergyRow:
    """Merged energy measurement + tc_list entry."""
    case_index: int
    M: int
    K: int
    N: int
    SPm: int
    SPn: int
    TPm: int
    TPk: int
    TPn: int
    TM: int
    TK: int
    TN: int
    num_cores: int
    tp_order_inner: int
    avg_iter_us: float
    npu_energy_uj: float
    npu_per_iter_uj: float
    npu_avg_power_mw: float
    power_cv_pct: float
    wall_elapsed_s: float

    @property
    def n_cores(self) -> int:
        return self.SPm * self.SPn

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.K}x{self.N}"

    @property
    def n_columns(self) -> int:
        return math.ceil(self.n_cores / _COMP_TILES_PER_COL)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_energy_csv(
    csv_path: Path,
    tc_path: Optional[Path] = None,
    min_wall_s: float = DEFAULT_MIN_WALL_S,
) -> Tuple[List[EnergyRow], int, int]:
    """Load energy CSV with validity filter.

    Returns (valid_rows, total_count, skipped_count).
    """
    tp_order_map = _load_tp_order_map(tc_path) if tc_path else {}

    rows: List[EnergyRow] = []
    total = 0
    skipped = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            case_idx = int(row["case_index"])
            per_iter = float(_col(row, "npu_per_iter_uj",
                                  "npu_energy_per_iter_uj"))
            wall_s = float(row.get("wall_elapsed_s", "0"))

            # Filter: negative energy or short measurement window
            if per_iter <= 0 or wall_s < min_wall_s:
                skipped += 1
                continue

            # tp_order_inner: CSV column or tc_list.json lookup
            if "tp_order_inner" in row:
                tp_inner = int(row["tp_order_inner"])
            elif case_idx in tp_order_map:
                tp_inner = tp_order_map[case_idx]
            else:
                tp_inner = 2  # fallback: K-inner

            rows.append(EnergyRow(
                case_index=case_idx,
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
                num_cores=int(_col(row, "num_cores", "numSpm")),
                tp_order_inner=tp_inner,
                avg_iter_us=float(_col(row, "avg_iter_us", "avg_us")),
                npu_energy_uj=float(_col(row, "npu_energy_uj")),
                npu_per_iter_uj=per_iter,
                npu_avg_power_mw=float(_col(row, "npu_avg_power_mw",
                                            "npu_power_mw")),
                power_cv_pct=float(row.get("power_cv_pct", "0.0")),
                wall_elapsed_s=wall_s,
            ))
    return rows, total, skipped


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _make_op_cand(row: EnergyRow) -> Tuple[OpCase, Candidate]:
    op = OpCase(M=row.M, K=row.K, N=row.N, elem_type="bf16")
    cand = Candidate(
        num_cores=row.n_cores, num_columns=row.n_columns,
        SPm=row.SPm, SPn=row.SPn,
        TPm=row.TPm, TPk=row.TPk, TPn=row.TPn,
        TM=row.TM, TK=row.TK, TN=row.TN,
    )
    return op, cand


def _perf_components(row: EnergyRow, coeffs: Optional[CalibCoeffs]):
    """Compute performance components (cycles)."""
    op, cand = _make_op_cand(row)
    if coeffs and coeffs.calibrated:
        t_comp = (op.M * op.N * op.K) / (row.n_cores * coeffs.eff_macs)
        t_dma = total_data_bytes(op, cand, row.tp_order_inner) / coeffs.bw_eff_bpc
        t_ovh = (coeffs.l_sync_cy * cand.tp_total
                 + coeffs.l_core_cy * row.n_cores
                 + coeffs.l_startup_cy)
    else:
        t_comp = (op.M * op.N * op.K) / (row.n_cores * PEAK_MACS)
        t_dma = total_data_bytes(op, cand, row.tp_order_inner) / BANDWIDTH_BPC
        t_ovh = ALPHA_CYCLES * cand.tp_total
    return t_comp, t_dma, t_ovh


def predict_energy_theoretical(row: EnergyRow) -> Dict[str, float]:
    """Predict energy using the existing Horowitz-based constants."""
    op, cand = _make_op_cand(row)
    t_comp = (op.M * op.N * op.K) / (row.n_cores * PEAK_MACS)
    t_dma = total_data_bytes(op, cand, row.tp_order_inner) / BANDWIDTH_BPC
    t_ovh = ALPHA_CYCLES * cand.tp_total
    t_total = t_comp + t_dma + t_ovh

    e_comp = op.M * op.N * op.K * E_MAC_PJ
    e_comm = total_data_bytes(op, cand, row.tp_order_inner) * E_DRAM_PJ
    e_static = row.n_cores * P_STATIC_PJ * t_total
    e_total = e_comp + e_comm + e_static

    return {
        "t_total_cy": t_total,
        "e_comp_pj": e_comp, "e_comm_pj": e_comm, "e_static_pj": e_static,
        "e_total_pj": e_total, "e_total_uj": e_total / 1e6,
    }


def predict_energy_calibrated(
    row: EnergyRow, coeffs: CalibCoeffs,
) -> Dict[str, float]:
    """Predict energy using calibrated perf + theoretical energy constants."""
    op, cand = _make_op_cand(row)
    t_comp, t_dma, t_ovh = _perf_components(row, coeffs)
    t_total = t_comp + t_dma + t_ovh

    e_comp = op.M * op.N * op.K * E_MAC_PJ
    e_comm = total_data_bytes(op, cand, row.tp_order_inner) * E_DRAM_PJ
    e_static = row.n_cores * P_STATIC_PJ * t_total
    e_total = e_comp + e_comm + e_static

    return {
        "t_total_cy": t_total,
        "e_comp_pj": e_comp, "e_comm_pj": e_comm, "e_static_pj": e_static,
        "e_total_pj": e_total, "e_total_uj": e_total / 1e6,
    }


def predict_energy_with_energy_calib(
    row: EnergyRow, coeffs: CalibCoeffs,
) -> Dict[str, float]:
    """Predict energy using fully calibrated model (both perf and energy)."""
    if not coeffs.energy_calibrated:
        return predict_energy_calibrated(row, coeffs)

    op, cand = _make_op_cand(row)
    params = coeffs.energy_params
    model = coeffs.energy_model
    t_comp, t_dma, t_ovh = _perf_components(row, coeffs)
    t_total_cy = t_comp + t_dma + t_ovh
    t_total_us = t_total_cy / CLOCK_MHZ

    if model == "E-A":
        p_active_uw = params.get("p_active_uw", 0)
        e_startup_uj = params.get("e_startup_uj", 0)
        e_total_uj = p_active_uw * t_total_us / 1e6 + e_startup_uj

    elif model == "E-B":
        p_comp = params.get("p_comp_uw", 0)
        p_dma = params.get("p_dma_uw", 0)
        p_idle = params.get("p_idle_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        e_total_uj = (p_comp * t_comp / CLOCK_MHZ / 1e6
                      + p_dma * t_dma / CLOCK_MHZ / 1e6
                      + p_idle * t_ovh / CLOCK_MHZ / 1e6
                      + e_startup)

    elif model == "E-C":
        e_mac = params.get("e_mac_pj", E_MAC_PJ)
        e_byte = params.get("e_byte_pj", E_DRAM_PJ)
        p_static = params.get("p_static_pj", P_STATIC_PJ)
        e_startup = params.get("e_startup_uj", 0)
        macs = row.M * row.K * row.N
        data_bytes = total_data_bytes(op, cand, row.tp_order_inner)
        e_total_pj = (e_mac * macs + e_byte * data_bytes
                      + p_static * row.n_cores * t_total_cy)
        e_total_uj = e_total_pj / 1e6 + e_startup

    elif model == "E-D":
        p_core_uw = params.get("p_core_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        e_total_uj = p_core_uw * row.n_cores * t_total_us / 1e6 + e_startup

    elif model == "E-F":
        p_base_uw = params.get("p_base_uw", 0)
        p_core_uw = params.get("p_core_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        e_total_uj = (p_base_uw + p_core_uw * row.n_cores) * t_total_us / 1e6 + e_startup

    elif model == "T-A":
        e_mac_pj = params.get("e_mac_pj", E_MAC_PJ)
        e_dram_pj = params.get("e_dram_pj", E_DRAM_PJ)
        p_static_pj = params.get("p_static_pj", P_STATIC_PJ)
        macs = row.M * row.K * row.N
        data_bytes = total_data_bytes(op, cand, row.tp_order_inner)
        e_total_pj = (e_mac_pj * macs + e_dram_pj * data_bytes
                      + p_static_pj * row.n_cores * t_total_cy)
        e_total_uj = e_total_pj / 1e6

    elif model == "T-B":
        e_mac_pj = params.get("e_mac_pj", E_MAC_PJ)
        e_dram_pj = params.get("e_dram_pj", E_DRAM_PJ)
        p_base_uw = params.get("p_base_uw", 0)
        p_core_uw = params.get("p_core_uw", 0)
        macs = row.M * row.K * row.N
        data_bytes = total_data_bytes(op, cand, row.tp_order_inner)
        # p_base_uw and p_core_uw are in uW; t_total_us in us
        # P * t = uW * us = 1e-6 W * 1e-6 s = 1e-12 J = pJ -> /1e6 = uJ
        e_total_pj = e_mac_pj * macs + e_dram_pj * data_bytes
        e_total_uj = e_total_pj / 1e6 + (p_base_uw + p_core_uw * row.n_cores) * t_total_us / 1e6

    elif model == "T-C":
        e_mac_pj = params.get("e_mac_pj", E_MAC_PJ)
        e_dram_pj = params.get("e_dram_pj", E_DRAM_PJ)
        p_base_uw = params.get("p_base_uw", 0)
        p_core_uw = params.get("p_core_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        macs = row.M * row.K * row.N
        data_bytes = total_data_bytes(op, cand, row.tp_order_inner)
        e_total_pj = e_mac_pj * macs + e_dram_pj * data_bytes
        e_total_uj = (e_total_pj / 1e6
                      + (p_base_uw + p_core_uw * row.n_cores) * t_total_us / 1e6
                      + e_startup)

    elif model.startswith("L-"):
        a_time = params.get("a_time", 1.0)
        C = params.get("C", 1.0)
        log_e = math.log(C) + a_time * math.log(max(t_total_us, 1e-6))
        if "c_cores" in params:
            log_e += params["c_cores"] * math.log(row.n_cores)
        if "c_cols" in params and "c_cores" not in params:
            log_e += params["c_cols"] * math.log(row.n_columns)
        e_total_uj = math.exp(log_e)

    else:
        pred = predict_energy_theoretical(row)
        return pred

    return {
        "t_total_cy": t_total_cy,
        "e_comp_pj": 0, "e_comm_pj": 0, "e_static_pj": 0,
        "e_total_pj": e_total_uj * 1e6,
        "e_total_uj": e_total_uj,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_mape(preds: List[float], actuals: List[float]) -> float:
    if not actuals:
        return float("nan")
    total = sum(abs(p - a) / a for p, a in zip(preds, actuals) if a > 0)
    n = sum(1 for a in actuals if a > 0)
    return (total / n * 100) if n > 0 else float("nan")


def compute_rmse(preds: List[float], actuals: List[float]) -> float:
    if not actuals:
        return float("nan")
    mse = sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(actuals)
    return mse ** 0.5


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def validate_model(
    rows: List[EnergyRow],
    coeffs: Optional[CalibCoeffs] = None,
) -> None:
    """Full validation report."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  Energy Model Validation Report")
    print(sep)
    print(f"  Cases: {len(rows)}")

    # Predict with theoretical model
    theo_preds = [predict_energy_theoretical(r) for r in rows]
    theo_uj = [p["e_total_uj"] for p in theo_preds]
    actual_uj = [r.npu_per_iter_uj for r in rows]

    rho_theo = spearman_rank_correlation(theo_uj, actual_uj)
    mape_theo = compute_mape(theo_uj, actual_uj)
    rmse_theo = compute_rmse(theo_uj, actual_uj)

    print(f"\n  --- Theoretical Model (Horowitz 2014 constants) ---")
    print(f"    E_MAC={E_MAC_PJ} pJ, E_DRAM={E_DRAM_PJ} pJ, P_STATIC={P_STATIC_PJ} pJ/cy")
    print(f"    Spearman rho: {rho_theo:.4f}")
    print(f"    MAPE:         {mape_theo:.1f}%")
    print(f"    RMSE:         {rmse_theo:.2f} uJ")

    # Calibrated model (if available)
    if coeffs and coeffs.calibrated:
        calib_preds = [predict_energy_calibrated(r, coeffs) for r in rows]
        calib_uj = [p["e_total_uj"] for p in calib_preds]
        rho_calib = spearman_rank_correlation(calib_uj, actual_uj)
        mape_calib = compute_mape(calib_uj, actual_uj)

        print(f"\n  --- Calibrated Perf + Theoretical Energy ---")
        print(f"    Spearman rho: {rho_calib:.4f}")
        print(f"    MAPE:         {mape_calib:.1f}%")

    if coeffs and coeffs.energy_calibrated:
        ecalib_preds = [predict_energy_with_energy_calib(r, coeffs) for r in rows]
        ecalib_uj = [p["e_total_uj"] for p in ecalib_preds]
        rho_ecalib = spearman_rank_correlation(ecalib_uj, actual_uj)
        mape_ecalib = compute_mape(ecalib_uj, actual_uj)

        print(f"\n  --- Fully Calibrated Energy Model ({coeffs.energy_model}) ---")
        print(f"    Spearman rho: {rho_ecalib:.4f}")
        print(f"    MAPE:         {mape_ecalib:.1f}%")

    # Energy component breakdown (theoretical model)
    print(f"\n  --- Component Breakdown (Theoretical, average) ---")
    avg_e_comp = sum(p["e_comp_pj"] for p in theo_preds) / len(theo_preds)
    avg_e_comm = sum(p["e_comm_pj"] for p in theo_preds) / len(theo_preds)
    avg_e_static = sum(p["e_static_pj"] for p in theo_preds) / len(theo_preds)
    avg_e_total = avg_e_comp + avg_e_comm + avg_e_static
    if avg_e_total > 0:
        print(f"    E_comp:   {avg_e_comp/1e6:>10.2f} uJ  ({avg_e_comp/avg_e_total*100:>5.1f}%)")
        print(f"    E_comm:   {avg_e_comm/1e6:>10.2f} uJ  ({avg_e_comm/avg_e_total*100:>5.1f}%)")
        print(f"    E_static: {avg_e_static/1e6:>10.2f} uJ  ({avg_e_static/avg_e_total*100:>5.1f}%)")

    # Per-size analysis
    print(f"\n  --- Per-Size Analysis ---")
    size_groups: Dict[str, List[Tuple[EnergyRow, Dict]]] = {}
    for r, p in zip(rows, theo_preds):
        key = r.size_key
        if key not in size_groups:
            size_groups[key] = []
        size_groups[key].append((r, p))

    print(f"    {'Size':<12s}  {'N':>4s}  {'rho':>8s}  {'MAPE':>8s}  "
          f"{'Pred_avg':>10s}  {'Actual_avg':>10s}")
    for key in sorted(size_groups.keys()):
        group = size_groups[key]
        g_preds = [p["e_total_uj"] for _, p in group]
        g_actual = [r.npu_per_iter_uj for r, _ in group]
        g_rho = spearman_rank_correlation(g_preds, g_actual) if len(group) >= 3 else float("nan")
        g_mape = compute_mape(g_preds, g_actual)
        print(f"    {key:<12s}  {len(group):>4d}  {g_rho:>8.4f}  {g_mape:>7.1f}%  "
              f"{sum(g_preds)/len(g_preds):>10.2f}  {sum(g_actual)/len(g_actual):>10.2f}")

    # Per-core analysis
    print(f"\n  --- Per-Core Count Analysis ---")
    core_groups: Dict[int, List[Tuple[EnergyRow, Dict]]] = {}
    for r, p in zip(rows, theo_preds):
        nc = r.n_cores
        if nc not in core_groups:
            core_groups[nc] = []
        core_groups[nc].append((r, p))

    print(f"    {'Cores':>5s}  {'N':>4s}  {'rho':>8s}  {'MAPE':>8s}")
    for nc in sorted(core_groups.keys()):
        group = core_groups[nc]
        g_preds = [p["e_total_uj"] for _, p in group]
        g_actual = [r.npu_per_iter_uj for r, _ in group]
        g_rho = spearman_rank_correlation(g_preds, g_actual) if len(group) >= 3 else float("nan")
        g_mape = compute_mape(g_preds, g_actual)
        print(f"    {nc:>5d}  {len(group):>4d}  {g_rho:>8.4f}  {g_mape:>7.1f}%")

    # Energy vs time linearity check
    print(f"\n  --- Energy vs Time Linearity ---")
    times_us = [r.avg_iter_us for r in rows]
    energies_uj = [r.npu_per_iter_uj for r in rows]
    rho_et = spearman_rank_correlation(times_us, energies_uj)
    print(f"    Spearman rho(time, energy): {rho_et:.4f}")
    if rho_et > 0.95:
        print(f"    -> Strong linear relationship — E ~ P_active * t model may suffice")
    elif rho_et > 0.80:
        print(f"    -> Moderate relationship — multi-component model recommended")
    else:
        print(f"    -> Weak relationship — energy structure differs from time")

    # Top 10 worst errors
    print(f"\n  --- Top 10 Worst Errors (Theoretical) ---")
    errors = []
    for r, p in zip(rows, theo_preds):
        actual = r.npu_per_iter_uj
        pred = p["e_total_uj"]
        err_pct = abs(pred - actual) / actual * 100 if actual > 0 else 0
        errors.append((r, pred, actual, err_pct))
    errors.sort(key=lambda x: -x[3])

    print(f"    {'#':>4s}  {'Size':>12s}  {'cores':>5s}  "
          f"{'Pred_uJ':>10s}  {'Actual_uJ':>10s}  {'error%':>8s}")
    for r, pred, actual, err in errors[:10]:
        print(f"    {r.case_index:>4d}  {r.size_key:>12s}  {r.n_cores:>5d}  "
              f"{pred:>10.2f}  {actual:>10.2f}  {err:>7.1f}%")

    print(f"\n{sep}")


# ---------------------------------------------------------------------------
# EDP core-count optimal verification
# ---------------------------------------------------------------------------
def edp_core_optimal_verification(
    rows: List[EnergyRow],
    coeffs: Optional[CalibCoeffs],
) -> None:
    """Compare measured vs model-predicted EDP-optimal core count per size."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  EDP Core-Count Optimal Verification")
    print(sep)

    if not coeffs or not coeffs.energy_calibrated:
        print("  No energy calibration available — showing measured EDP only")

    # Group by size, find best energy per (size, cores) combination
    size_core: Dict[str, Dict[int, List[EnergyRow]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        size_core[r.size_key][r.n_cores].append(r)

    valid_sizes = {sk: d for sk, d in size_core.items() if len(d) >= 2}
    if not valid_sizes:
        print("  No sizes with multiple core counts available")
        print(sep)
        return

    print(f"\n  {'Size':<16s}  {'Cores':>5s}  {'T(us)':>10s}  {'E(uJ)':>10s}  "
          f"{'EDP':>12s}  {'P(mW)':>8s}  {'Model E':>10s}  {'M-EDP':>12s}")

    match_count = 0
    total_sizes = 0

    for sk in sorted(valid_sizes.keys()):
        core_data = valid_sizes[sk]
        total_sizes += 1

        # For each core count, pick the config with lowest measured energy
        best_per_core: Dict[int, EnergyRow] = {}
        for nc, rlist in core_data.items():
            best_per_core[nc] = min(rlist, key=lambda r: r.npu_per_iter_uj)

        # Measured EDP
        meas_edp = {nc: r.avg_iter_us * r.npu_per_iter_uj
                    for nc, r in best_per_core.items()}
        meas_opt = min(meas_edp, key=meas_edp.get)

        # Model predictions
        model_edp = {}
        for nc, r in best_per_core.items():
            if coeffs and coeffs.energy_calibrated:
                pred = predict_energy_with_energy_calib(r, coeffs)
                model_e = pred["e_total_uj"]
                model_t = pred["t_total_cy"] / CLOCK_MHZ
                model_edp[nc] = model_e * model_t
            elif coeffs and coeffs.calibrated:
                pred = predict_energy_calibrated(r, coeffs)
                model_e = pred["e_total_uj"]
                model_t = pred["t_total_cy"] / CLOCK_MHZ
                model_edp[nc] = model_e * model_t
            else:
                pred = predict_energy_theoretical(r)
                model_e = pred["e_total_uj"]
                model_t = pred["t_total_cy"] / CLOCK_MHZ
                model_edp[nc] = model_e * model_t

        model_opt = min(model_edp, key=model_edp.get) if model_edp else -1

        for nc in sorted(best_per_core.keys()):
            r = best_per_core[nc]
            edp = meas_edp[nc]
            p_mw = r.npu_per_iter_uj / r.avg_iter_us * 1e3 if r.avg_iter_us > 0 else 0
            m_e = model_edp.get(nc, 0)

            meas_mark = " <-- meas" if nc == meas_opt else ""
            model_mark = " <-- model" if nc == model_opt else ""
            markers = meas_mark + model_mark

            print(f"  {sk:<16s}  {nc:>5d}  {r.avg_iter_us:>10.1f}  "
                  f"{r.npu_per_iter_uj:>10.1f}  {edp:>12.0f}  "
                  f"{p_mw:>8.0f}  {m_e / r.avg_iter_us * 1e3 if r.avg_iter_us > 0 else 0:>10.1f}  "
                  f"{m_e:>12.0f}{markers}")

        if meas_opt == model_opt:
            match_count += 1
        print()

    if total_sizes > 0:
        rate = match_count / total_sizes * 100
        target = 80
        status = "PASS" if rate >= target else "FAIL"
        print(f"  EDP-optimal core match: {match_count}/{total_sizes} "
              f"({rate:.0f}%) [{status}, target >= {target}%]")

    print(sep)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate(
    rows: List[EnergyRow],
    coeffs: Optional[CalibCoeffs],
    train_frac: float = 0.8,
    seed: int = 42,
) -> None:
    """80/20 train/test split validation."""
    print(f"\n{'=' * 70}")
    print(f"  Cross-Validation ({int(train_frac*100)}/{int((1-train_frac)*100)} split)")
    print(f"{'=' * 70}")

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    split = int(len(indices) * train_frac)
    train_idx = sorted(indices[:split])
    test_idx = sorted(indices[split:])

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    print(f"  Train: {len(train_rows)}, Test: {len(test_rows)}")

    # Evaluate on test set
    if coeffs and coeffs.energy_calibrated:
        preds = [predict_energy_with_energy_calib(r, coeffs)["e_total_uj"]
                 for r in test_rows]
        model_name = f"Calibrated ({coeffs.energy_model})"
    elif coeffs and coeffs.calibrated:
        preds = [predict_energy_calibrated(r, coeffs)["e_total_uj"]
                 for r in test_rows]
        model_name = "Calibrated Perf + Theoretical Energy"
    else:
        preds = [predict_energy_theoretical(r)["e_total_uj"]
                 for r in test_rows]
        model_name = "Theoretical"

    actuals = [r.npu_per_iter_uj for r in test_rows]
    rho = spearman_rank_correlation(preds, actuals)
    mape = compute_mape(preds, actuals)

    print(f"\n  Model: {model_name}")
    print(f"  Test set Spearman rho: {rho:.4f}  {'PASS' if rho > 0.90 else 'FAIL'} (target > 0.90)")
    print(f"  Test set MAPE:         {mape:.1f}%  {'PASS' if mape < 50 else 'FAIL'} (target < 50%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate energy cost model against measurements")
    p.add_argument("--energy", required=True,
                    help="Path to energy_results.csv")
    p.add_argument("--tc", default="",
                    help="Path to tc_list.json (optional, for metadata)")
    p.add_argument("--calib", default="",
                    help="Path to calibration.json")
    p.add_argument("--min-wall-s", type=float, default=DEFAULT_MIN_WALL_S,
                    help=f"Minimum wall_elapsed_s for valid measurement (default: {DEFAULT_MIN_WALL_S})")
    p.add_argument("--cross-validate", action="store_true",
                    help="Run 80/20 cross-validation")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    global _COMP_TILES_PER_COL

    args = parse_args(argv)

    energy_path = Path(args.energy).resolve()
    tc_path = Path(args.tc).resolve() if args.tc else None
    min_wall_s = args.min_wall_s

    # Load HW info for tiles-per-column
    try:
        sys_info = load_system_info(DEFAULT_SYS_PATH)
        _COMP_TILES_PER_COL = sys_info.comp_tiles_per_col
    except Exception:
        pass

    rows, total, skipped = load_energy_csv(energy_path, tc_path, min_wall_s)
    if not rows:
        print("[ERROR] No valid data in energy CSV", file=sys.stderr)
        return 1
    print(f"[INFO] {len(rows)} valid of {total} rows "
          f"({skipped} filtered, min_wall={min_wall_s*1000:.0f}ms)")

    # Load calibration if specified
    coeffs = None
    if args.calib:
        calib_path = Path(args.calib).resolve()
        coeffs = load_calibration(calib_path)
        if coeffs.calibrated:
            print(f"[INFO] Calibration loaded: eff_macs={coeffs.eff_macs}")
        if coeffs.energy_calibrated:
            print(f"[INFO] Energy model: {coeffs.energy_model}")

    # Validate
    validate_model(rows, coeffs)

    # EDP core-count analysis
    edp_core_optimal_verification(rows, coeffs)

    # Cross-validate
    if args.cross_validate:
        cross_validate(rows, coeffs)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
