#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_energy_basis.py — Energy model redesign analysis (Task 8).

Step 1-A: Compare pkg vs pkg-idle energy measurement bases.
Step 1-B: Analyze T-B model error structure.
Step 2:   Calibrate candidate models (T-B, T-D, T-E) with global optimization.
Step 3:   Validate and compare all models (EDP Core Accuracy, Regret, MAPE).

Usage:
    python3 scripts/analyze/analyze_energy_basis.py \
        --csv out/reports/result_phase1.csv \
        --tc out/tc_list_dedup.json \
        --calib data/calibration.json
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import (  # noqa: E402
    OpCase, CalibCoeffs,
    load_calibration, DEFAULT_CALIB_PATH,
    spearman_rank_correlation,
)
from cost_model import (  # noqa: E402
    Candidate, total_data_bytes, _dma_ops_per_step,
    E_MAC_PJ, E_DRAM_PJ, P_STATIC_PJ,
)
# NOTE: Canonical model formulas live in models.py (EnergyModel).
# The _predict_* functions below use optimization-internal units (uJ/cy)
# which differ from models.py's deployment units (pJ/uW/us).
# Results are unit-converted when stored in calibration.json.
import models as _models  # noqa: E402

from calibrate_energy import (  # noqa: E402
    ECalibRow, load_energy_data, compute_features, compute_mape,
    _log_space_mse, _predict_tb, EnergyFitResult,
    CLOCK_MHZ, DEFAULT_MIN_WALL_S,
)


# ============================================================================
# Helpers
# ============================================================================

def _median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _stdev(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _spearman(xs: List[float], ys: List[float]) -> float:
    return spearman_rank_correlation(xs, ys)


def _edp_core_accuracy_and_regret(
    rows: List[ECalibRow],
    energy_pred_fn,
    time_pred_fn,
) -> Tuple[int, int, float, float]:
    """Compute EDP Core Accuracy and Regret across workload sizes.

    Returns: (n_correct, n_sizes, mean_regret_pct, max_regret_pct)
    """
    # Group by size
    by_size: Dict[str, Dict[int, List[Tuple[ECalibRow, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        e_pred = energy_pred_fn(r)
        t_pred = time_pred_fn(r)
        if e_pred > 0 and t_pred > 0:
            by_size[r.size_key][r.n_cores].append((r, e_pred, t_pred))

    n_correct = 0
    n_sizes = 0
    regrets = []

    for size_key, cores_dict in sorted(by_size.items()):
        if len(cores_dict) < 2:
            continue
        n_sizes += 1

        # Measured EDP: use median within each core count
        meas_edp: Dict[int, float] = {}
        model_edp: Dict[int, float] = {}

        for nc, entries in cores_dict.items():
            # Measured: median energy * median time
            energies = [r.npu_per_iter_uj for r, _, _ in entries]
            times = [r.avg_iter_us for r, _, _ in entries]
            meas_edp[nc] = _median(energies) * _median(times)

            # Model: median predicted E * median predicted T
            pred_es = [ep for _, ep, _ in entries]
            pred_ts = [tp for _, _, tp in entries]
            model_edp[nc] = _median(pred_es) * _median(pred_ts)

        meas_opt = min(meas_edp, key=meas_edp.get)
        model_opt = min(model_edp, key=model_edp.get)

        if meas_opt == model_opt:
            n_correct += 1

        # Regret: measured EDP of model's choice vs measured optimum
        if meas_edp[meas_opt] > 0:
            regret = (meas_edp[model_opt] - meas_edp[meas_opt]) / meas_edp[meas_opt] * 100
            regrets.append(max(0, regret))

    mean_regret = sum(regrets) / len(regrets) if regrets else 0.0
    max_regret = max(regrets) if regrets else 0.0
    return n_correct, n_sizes, mean_regret, max_regret


# ============================================================================
# Step 1-A: Energy Measurement Basis Analysis
# ============================================================================

def step1a_energy_basis(csv_path: Path, tc_path: Path, min_wall_s: float):
    """Compare pkg vs pkg-idle energy measurement bases."""
    print("=" * 72)
    print("STEP 1-A: Energy Measurement Basis Analysis")
    print("=" * 72)

    # Load raw CSV for all three bases
    rows_npu, skip_npu = load_energy_data(csv_path, tc_path, min_wall_s, gt_mode="npu")
    rows_pkg, skip_pkg = load_energy_data(csv_path, tc_path, min_wall_s, gt_mode="active_pkg")
    rows_cor, skip_cor = load_energy_data(csv_path, tc_path, min_wall_s, gt_mode="corrected")

    print(f"\nData availability:")
    print(f"  npu (idle-subtracted): {len(rows_npu)} valid, {len(skip_npu)} skipped")
    print(f"  active_pkg (raw):      {len(rows_pkg)} valid, {len(skip_pkg)} skipped")
    print(f"  corrected (pkg-idle):  {len(rows_cor)} valid, {len(skip_cor)} skipped")

    # idle_pkg_mw distribution
    idle_vals = [r.idle_pkg_mw for r in rows_npu if r.idle_pkg_mw > 0]
    if idle_vals:
        mean_idle = sum(idle_vals) / len(idle_vals)
        std_idle = _stdev(idle_vals)
        cv = std_idle / mean_idle * 100 if mean_idle > 0 else 0
        print(f"\nidle_pkg_mw distribution (n={len(idle_vals)}):")
        print(f"  mean={mean_idle:.1f} mW, std={std_idle:.1f} mW, CV={cv:.1f}%")
        print(f"  min={min(idle_vals):.1f}, max={max(idle_vals):.1f}, "
              f"median={_median(idle_vals):.1f}")
    else:
        print("\n[WARN] No valid idle_pkg_mw values found")

    # Rank correlation between bases
    # Build case_index -> energy maps for intersection
    npu_map = {r.case_index: r.npu_per_iter_uj for r in rows_npu}
    pkg_map = {r.case_index: r.npu_per_iter_uj for r in rows_pkg}
    cor_map = {r.case_index: r.npu_per_iter_uj for r in rows_cor}

    common = set(npu_map) & set(pkg_map) & set(cor_map)
    if common:
        npu_vals = [npu_map[c] for c in sorted(common)]
        pkg_vals = [pkg_map[c] for c in sorted(common)]
        cor_vals = [cor_map[c] for c in sorted(common)]

        rho_npu_pkg = _spearman(npu_vals, pkg_vals)
        rho_npu_cor = _spearman(npu_vals, cor_vals)
        rho_pkg_cor = _spearman(pkg_vals, cor_vals)

        print(f"\nRank correlation between bases (n={len(common)}):")
        print(f"  npu vs active_pkg: rho={rho_npu_pkg:.4f}")
        print(f"  npu vs corrected:  rho={rho_npu_cor:.4f}")
        print(f"  active_pkg vs corrected: rho={rho_pkg_cor:.4f}")

    # Check for negative/extreme values after idle subtraction
    negative_count = sum(1 for r in rows_cor if r.npu_per_iter_uj < 0)
    if rows_cor:
        min_energy = min(r.npu_per_iter_uj for r in rows_cor)
        max_energy = max(r.npu_per_iter_uj for r in rows_cor)
        print(f"\ncorrected mode energy range: [{min_energy:.1f}, {max_energy:.1f}] uJ")
        print(f"  negative values: {negative_count}")

    # CPU idle baseline check
    cpu_path = csv_path.parent.parent.parent / "paper" / "analysis" / "cpu_energy.csv"
    if cpu_path.is_file():
        print(f"\nCPU energy data: {cpu_path} (exists)")
        with cpu_path.open("r") as f:
            reader = csv.DictReader(f)
            cpu_rows = list(reader)
        if cpu_rows:
            cpu_idle = [float(r.get("idle_pkg_mw", 0)) for r in cpu_rows
                        if float(r.get("idle_pkg_mw", 0)) > 0]
            if cpu_idle:
                print(f"  CPU idle_pkg_mw: mean={sum(cpu_idle)/len(cpu_idle):.1f}, "
                      f"min={min(cpu_idle):.1f}, max={max(cpu_idle):.1f}")
    else:
        print(f"\nCPU energy data: not found at {cpu_path}")

    print(f"\n--- Basis Decision ---")
    print(f"Recommendation: Use 'npu' (idle-subtracted) basis.")
    print(f"  - Idle is near-constant (low CV), so rank ordering is preserved")
    print(f"  - NPU-specific measurement isolates NPU energy from CPU background")
    print(f"  - Consistent with all prior calibration work")

    return rows_npu


# ============================================================================
# Step 1-B: Error Structure Analysis
# ============================================================================

def step1b_error_structure(
    rows: List[ECalibRow],
    features: List[Dict],
    coeffs: CalibCoeffs,
):
    """Analyze T-B model residuals by key variables."""
    print("\n" + "=" * 72)
    print("STEP 1-B: Error Structure Analysis (T-B model)")
    print("=" * 72)

    # First re-fit T-B with current data to get predictions
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        print("[ERROR] scipy required for T-B fitting")
        return

    y = [r.npu_per_iter_uj for r in rows]
    macs_arr = [float(f["macs"]) for f in features]
    bytes_arr = [float(f["data_bytes"]) for f in features]
    t_arr = [float(f["t_total_cy"]) for f in features]
    nt_arr = [float(r.n_cores * f["t_total_cy"]) for r, f in zip(rows, features)]

    def objective(params):
        preds = _predict_tb(params, macs_arr, bytes_arr, t_arr, nt_arr)
        return _log_space_mse(preds, y)

    # Bounds based on prior T-B results: e_mac~16pJ, e_dram~1370pJ,
    # p_base~2.9MW (1.9e-3 uJ/cy), p_core~107kW (7e-5 uJ/cy)
    bounds = [
        (1e-18, 1e-3),   # e_mac (uJ/MAC), up to ~1000 pJ
        (1e-18, 1e-1),   # e_dram (uJ/byte), up to ~100,000 pJ
        (1e-18, 1e0),    # p_base (uJ/cy)
        (1e-18, 1e0),    # p_core (uJ/cy)
    ]
    result = differential_evolution(objective, bounds, seed=42, polish=True,
                                    maxiter=3000, tol=1e-14, popsize=30)
    preds = _predict_tb(result.x, macs_arr, bytes_arr, t_arr, nt_arr)
    rho = _spearman(preds, y)
    mape = compute_mape(preds, y)

    e_mac, e_dram, p_base, p_core = result.x
    print(f"\nT-B (re-fitted with DE, v9 T_total):")
    print(f"  e_mac  = {e_mac*1e6:.2f} pJ/MAC")
    print(f"  e_dram = {e_dram*1e6:.2f} pJ/byte")
    print(f"  p_base = {p_base * CLOCK_MHZ * 1e6:.0f} uW ({p_base * CLOCK_MHZ:.3f} W)")
    print(f"  p_core = {p_core * CLOCK_MHZ * 1e6:.0f} uW ({p_core * CLOCK_MHZ * 1e3:.1f} mW)")
    print(f"  rho={rho:.4f}, MAPE={mape:.1f}%")

    # Residuals
    residuals = [(p - a) / a * 100 for p, a in zip(preds, y)]

    # Variables to correlate with residuals
    variables = {
        "P (cores)": [float(r.n_cores) for r in rows],
        "T_total_us": [f["t_total_us"] for f in features],
        "Total_Data": [float(f["data_bytes"]) for f in features],
        "MACs": [float(f["macs"]) for f in features],
        "SP_ratio": [max(r.SPm, r.SPn) / min(r.SPm, r.SPn) for r in rows],
        "N_dma": [float(f["n_dma"]) for f in features],
        "tpOrder": [float(r.tp_order_inner) for r in rows],
        "TP_total": [float(r.tp_total) for r in rows],
        "N_dma*TP": [float(f["n_dma"] * r.tp_total) for r, f in zip(rows, features)],
    }

    print(f"\nResidual correlation analysis (n={len(rows)}):")
    print(f"  {'Variable':<15} {'rho':>8} {'abs(rho)':>8}  Interpretation")
    print(f"  {'-'*15} {'-'*8} {'-'*8}  {'-'*30}")

    correlations = []
    for name, vals in variables.items():
        rho_resid = _spearman(vals, residuals)
        correlations.append((name, rho_resid))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, rho_resid in correlations:
        strength = "STRONG" if abs(rho_resid) > 0.3 else ("moderate" if abs(rho_resid) > 0.15 else "weak")
        print(f"  {name:<15} {rho_resid:>+8.4f} {abs(rho_resid):>8.4f}  {strength}")

    # SP shape analysis: balanced vs unbalanced
    balanced = [(p, r_pct) for (r_row, p, r_pct) in zip(rows, preds, residuals)
                if r_row.SPm == r_row.SPn]
    unbalanced = [(p, r_pct) for (r_row, p, r_pct) in zip(rows, preds, residuals)
                  if r_row.SPm != r_row.SPn]

    if balanced and unbalanced:
        bal_mape = sum(abs(r) for _, r in balanced) / len(balanced)
        unbal_mape = sum(abs(r) for _, r in unbalanced) / len(unbalanced)
        print(f"\nSP shape error breakdown:")
        print(f"  Balanced (SPm==SPn):   n={len(balanced)}, mean|residual|={bal_mape:.1f}%")
        print(f"  Unbalanced (SPm!=SPn): n={len(unbalanced)}, mean|residual|={unbal_mape:.1f}%")

    return result.x  # Return T-B params for comparison


# ============================================================================
# Step 2: Candidate Model Calibration (differential_evolution)
# ============================================================================

def _predict_td(params, macs_arr, bytes_arr, ndma_tp_arr, t_arr, nt_arr):
    """T-D via models.EnergyModel.predict_td_optim()."""
    return _models.EnergyModel.predict_td_optim(params, {
        "macs": macs_arr, "data_bytes": bytes_arr,
        "ndma_tp": ndma_tp_arr, "t_total_cy": t_arr, "nt": nt_arr,
    })


def _predict_te(params, macs_arr, bytes_arr, ndma_tp_arr, tp_arr, t_arr, nt_arr):
    """T-E via models.EnergyModel.predict_te_optim()."""
    return _models.EnergyModel.predict_te_optim(params, {
        "macs": macs_arr, "data_bytes": bytes_arr,
        "ndma_tp": ndma_tp_arr, "tp_total": tp_arr,
        "t_total_cy": t_arr, "nt": nt_arr,
    })


def _predict_tf(params, macs_arr, bytes_arr, ndma_tp_arr, tp_arr,
                t_comp_arr, t_comm_arr, nt_arr):
    """T-F via models.EnergyModel.predict_tf_optim()."""
    return _models.EnergyModel.predict_tf_optim(params, {
        "macs": macs_arr, "data_bytes": bytes_arr,
        "ndma_tp": ndma_tp_arr, "tp_total": tp_arr,
        "t_comp_cy": t_comp_arr, "t_comm_cy": t_comm_arr, "nt": nt_arr,
    })


def step2_calibrate(
    rows: List[ECalibRow],
    features: List[Dict],
    coeffs: CalibCoeffs,
) -> Dict[str, Tuple[Any, EnergyFitResult]]:
    """Calibrate T-B, T-D, T-E, L-B models with global optimization."""
    print("\n" + "=" * 72)
    print("STEP 2: Model Calibration (differential_evolution)")
    print("=" * 72)

    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        print("[ERROR] scipy required")
        return {}

    y = [r.npu_per_iter_uj for r in rows]
    macs_arr = [float(f["macs"]) for f in features]
    bytes_arr = [float(f["data_bytes"]) for f in features]
    t_arr = [float(f["t_total_cy"]) for f in features]
    nt_arr = [float(r.n_cores * f["t_total_cy"]) for r, f in zip(rows, features)]
    ndma_tp_arr = [float(f["n_dma"] * r.tp_total) for r, f in zip(rows, features)]
    tp_arr = [float(r.tp_total) for r in rows]
    # Per-component times in cycles (for T-F model)
    t_comp_arr = [f["t_comp_us"] * CLOCK_MHZ for f in features]
    t_comm_arr = [f["t_dma_us"] * CLOCK_MHZ for f in features]

    results: Dict[str, Tuple[Any, EnergyFitResult]] = {}

    # --- T-B: 4 params ---
    print("\n[T-B] Fitting (4 params)...")
    def obj_tb(params):
        preds = _predict_tb(params, macs_arr, bytes_arr, t_arr, nt_arr)
        return _log_space_mse(preds, y)

    bounds_tb = [
        (1e-18, 1e-3),   # e_mac (uJ/MAC)
        (1e-18, 1e-1),   # e_dram (uJ/byte)
        (1e-18, 1e0),    # p_base (uJ/cy)
        (1e-18, 1e0),    # p_core (uJ/cy)
    ]
    res_tb = differential_evolution(obj_tb, bounds_tb, seed=42, polish=True,
                                    maxiter=3000, tol=1e-14, popsize=30)
    preds_tb = _predict_tb(res_tb.x, macs_arr, bytes_arr, t_arr, nt_arr)
    rho_tb = _spearman(preds_tb, y)
    mape_tb = compute_mape(preds_tb, y)
    e_mac, e_dram, p_base, p_core = res_tb.x
    fit_tb = EnergyFitResult(
        "T-B",
        {"e_mac_pj": round(e_mac * 1e6, 4), "e_dram_pj": round(e_dram * 1e6, 4),
         "p_base_uw": round(p_base * CLOCK_MHZ * 1e6, 1),
         "p_core_uw": round(p_core * CLOCK_MHZ * 1e6, 1)},
        4, rho_tb, mape_tb, preds_tb)
    results["T-B"] = (res_tb.x, fit_tb)
    print(f"  e_mac={e_mac*1e6:.2f} pJ, e_dram={e_dram*1e6:.2f} pJ, "
          f"p_base={p_base*CLOCK_MHZ*1e6:.0f} uW, p_core={p_core*CLOCK_MHZ*1e6:.0f} uW")
    print(f"  rho={rho_tb:.4f}, MAPE={mape_tb:.1f}%")

    # --- T-D: 5 params (T-B + DMA descriptor) ---
    print("\n[T-D] Fitting (5 params)...")
    def obj_td(params):
        preds = _predict_td(params, macs_arr, bytes_arr, ndma_tp_arr, t_arr, nt_arr)
        return _log_space_mse(preds, y)

    bounds_td = [
        (1e-18, 1e-3),   # e_mac
        (1e-18, 1e-1),   # e_dram
        (1e-18, 1e3),    # e_dma (uJ per DMA*TP)
        (1e-18, 1e0),    # p_base
        (1e-18, 1e0),    # p_core
    ]
    res_td = differential_evolution(obj_td, bounds_td, seed=42, polish=True,
                                    maxiter=3000, tol=1e-14, popsize=30)
    preds_td = _predict_td(res_td.x, macs_arr, bytes_arr, ndma_tp_arr, t_arr, nt_arr)
    rho_td = _spearman(preds_td, y)
    mape_td = compute_mape(preds_td, y)
    e_mac, e_dram, e_dma, p_base, p_core = res_td.x
    fit_td = EnergyFitResult(
        "T-D",
        {"e_mac_pj": round(e_mac * 1e6, 4), "e_dram_pj": round(e_dram * 1e6, 4),
         "e_dma_uj": round(e_dma, 6),
         "p_base_uw": round(p_base * CLOCK_MHZ * 1e6, 1),
         "p_core_uw": round(p_core * CLOCK_MHZ * 1e6, 1)},
        5, rho_td, mape_td, preds_td)
    results["T-D"] = (res_td.x, fit_td)
    print(f"  e_mac={e_mac*1e6:.2f} pJ, e_dram={e_dram*1e6:.2f} pJ, "
          f"e_dma={e_dma:.4f} uJ/DMA*TP")
    print(f"  p_base={p_base*CLOCK_MHZ*1e6:.0f} uW, p_core={p_core*CLOCK_MHZ*1e6:.0f} uW")
    print(f"  rho={rho_td:.4f}, MAPE={mape_td:.1f}%")

    # --- T-E: 6 params (T-D + sync) ---
    print("\n[T-E] Fitting (6 params)...")
    def obj_te(params):
        preds = _predict_te(params, macs_arr, bytes_arr, ndma_tp_arr, tp_arr, t_arr, nt_arr)
        return _log_space_mse(preds, y)

    bounds_te = [
        (1e-18, 1e-3),   # e_mac
        (1e-18, 1e-1),   # e_dram
        (1e-18, 1e3),    # e_dma
        (1e-18, 1e4),    # e_sync (uJ per TP iteration)
        (1e-18, 1e0),    # p_base
        (1e-18, 1e0),    # p_core
    ]
    res_te = differential_evolution(obj_te, bounds_te, seed=42, polish=True,
                                    maxiter=3000, tol=1e-14, popsize=30)
    preds_te = _predict_te(res_te.x, macs_arr, bytes_arr, ndma_tp_arr, tp_arr, t_arr, nt_arr)
    rho_te = _spearman(preds_te, y)
    mape_te = compute_mape(preds_te, y)
    e_mac, e_dram, e_dma, e_sync, p_base, p_core = res_te.x
    fit_te = EnergyFitResult(
        "T-E",
        {"e_mac_pj": round(e_mac * 1e6, 4), "e_dram_pj": round(e_dram * 1e6, 4),
         "e_dma_uj": round(e_dma, 6), "e_sync_uj": round(e_sync, 4),
         "p_base_uw": round(p_base * CLOCK_MHZ * 1e6, 1),
         "p_core_uw": round(p_core * CLOCK_MHZ * 1e6, 1)},
        6, rho_te, mape_te, preds_te)
    results["T-E"] = (res_te.x, fit_te)
    print(f"  e_mac={e_mac*1e6:.2f} pJ, e_dram={e_dram*1e6:.2f} pJ, "
          f"e_dma={e_dma:.4f} uJ/DMA*TP, e_sync={e_sync:.2f} uJ/TP")
    print(f"  p_base={p_base*CLOCK_MHZ*1e6:.0f} uW, p_core={p_core*CLOCK_MHZ*1e6:.0f} uW")
    print(f"  rho={rho_te:.4f}, MAPE={mape_te:.1f}%")

    # --- L-B: 3 params (power-law baseline) ---
    print("\n[L-B] Fitting (3 params)...")
    log_t = [math.log(f["t_total_us"]) if f["t_total_us"] > 0 else 0 for f in features]
    log_n = [math.log(r.n_cores) for r in rows]
    log_y = [math.log(e) if e > 0 else 0 for e in y]

    def obj_lb(params):
        a_time, c_cores, log_C = params
        preds = [math.exp(log_C + a_time * log_t[i] + c_cores * log_n[i])
                 for i in range(len(y))]
        return _log_space_mse(preds, y)

    bounds_lb = [
        (0.5, 2.0),    # a_time
        (0.0, 2.0),    # c_cores
        (-5.0, 5.0),   # log_C
    ]
    res_lb = differential_evolution(obj_lb, bounds_lb, seed=42, polish=True,
                                    maxiter=1000, tol=1e-12)
    a_time, c_cores, log_C = res_lb.x
    C = math.exp(log_C)
    preds_lb = [C * (features[i]["t_total_us"] ** a_time) * (rows[i].n_cores ** c_cores)
                for i in range(len(rows))]
    rho_lb = _spearman(preds_lb, y)
    mape_lb = compute_mape(preds_lb, y)
    fit_lb = EnergyFitResult(
        "L-B",
        {"a_time": round(a_time, 6), "c_cores": round(c_cores, 6), "C": round(C, 4)},
        3, rho_lb, mape_lb, preds_lb)
    results["L-B"] = (res_lb.x, fit_lb)
    print(f"  a_time={a_time:.4f}, c_cores={c_cores:.4f}, C={C:.4f}")
    print(f"  rho={rho_lb:.4f}, MAPE={mape_lb:.1f}%")

    # --- T-F: 5 params (separate compute/comm power) ---
    print("\n[T-F] Fitting (5 params: p_comp, p_comm, e_dma, e_sync, p_idle)...")
    def obj_tf(params):
        preds = _predict_tf(params, macs_arr, bytes_arr, ndma_tp_arr, tp_arr,
                            t_comp_arr, t_comm_arr, nt_arr)
        return _log_space_mse(preds, y)

    bounds_tf = [
        (1e-18, 1e0),    # p_comp (uJ/cy, power during compute phase)
        (1e-18, 1e0),    # p_comm (uJ/cy, power during DMA phase)
        (1e-18, 1e3),    # e_dma (uJ per DMA*TP)
        (1e-18, 1e4),    # e_sync (uJ per TP)
        (1e-18, 1e0),    # p_idle (uJ/cy per core, leakage)
    ]
    res_tf = differential_evolution(obj_tf, bounds_tf, seed=42, polish=True,
                                    maxiter=3000, tol=1e-14, popsize=30)
    preds_tf = _predict_tf(res_tf.x, macs_arr, bytes_arr, ndma_tp_arr, tp_arr,
                           t_comp_arr, t_comm_arr, nt_arr)
    rho_tf = _spearman(preds_tf, y)
    mape_tf = compute_mape(preds_tf, y)
    p_comp, p_comm, e_dma_f, e_sync_f, p_idle = res_tf.x
    fit_tf = EnergyFitResult(
        "T-F",
        {"p_comp_uw": round(p_comp * CLOCK_MHZ * 1e6, 1),
         "p_comm_uw": round(p_comm * CLOCK_MHZ * 1e6, 1),
         "e_dma_uj": round(e_dma_f, 6),
         "e_sync_uj": round(e_sync_f, 4),
         "p_idle_uw": round(p_idle * CLOCK_MHZ * 1e6, 1)},
        5, rho_tf, mape_tf, preds_tf)
    results["T-F"] = (res_tf.x, fit_tf)
    print(f"  p_comp={p_comp*CLOCK_MHZ*1e6:.0f} uW, p_comm={p_comm*CLOCK_MHZ*1e6:.0f} uW")
    print(f"  e_dma={e_dma_f:.4f} uJ/DMA*TP, e_sync={e_sync_f:.2f} uJ/TP")
    print(f"  p_idle={p_idle*CLOCK_MHZ*1e6:.0f} uW/core")
    print(f"  rho={rho_tf:.4f}, MAPE={mape_tf:.1f}%")

    return results


# ============================================================================
# Step 3: Validation & Comparison
# ============================================================================

def step3_compare(
    rows: List[ECalibRow],
    features: List[Dict],
    model_results: Dict[str, Tuple[Any, EnergyFitResult]],
):
    """Compare all models on EDP Core Accuracy, Regret, MAPE, rho."""
    print("\n" + "=" * 72)
    print("STEP 3: Validation & Comparison")
    print("=" * 72)

    y = [r.npu_per_iter_uj for r in rows]
    macs_arr = [float(f["macs"]) for f in features]
    bytes_arr = [float(f["data_bytes"]) for f in features]
    t_arr = [float(f["t_total_cy"]) for f in features]
    nt_arr = [float(r.n_cores * f["t_total_cy"]) for r, f in zip(rows, features)]
    ndma_tp_arr = [float(f["n_dma"] * r.tp_total) for r, f in zip(rows, features)]
    tp_arr = [float(r.tp_total) for r in rows]
    t_comp_arr = [f["t_comp_us"] * CLOCK_MHZ for f in features]
    t_comm_arr = [f["t_dma_us"] * CLOCK_MHZ for f in features]

    # Build row index for O(1) lookup
    row_idx_map = {id(r): i for i, r in enumerate(rows)}

    ALL_MODELS = ["T-B", "T-D", "T-E", "T-F", "L-B"]

    def _make_fast_energy_fn(model_name, params):
        if model_name == "T-B":
            def fn(r):
                i = row_idx_map[id(r)]
                return _predict_tb(params, [macs_arr[i]], [bytes_arr[i]],
                                   [t_arr[i]], [nt_arr[i]])[0]
            return fn
        elif model_name == "T-D":
            def fn(r):
                i = row_idx_map[id(r)]
                return _predict_td(params, [macs_arr[i]], [bytes_arr[i]],
                                   [ndma_tp_arr[i]], [t_arr[i]], [nt_arr[i]])[0]
            return fn
        elif model_name == "T-E":
            def fn(r):
                i = row_idx_map[id(r)]
                return _predict_te(params, [macs_arr[i]], [bytes_arr[i]],
                                   [ndma_tp_arr[i]], [tp_arr[i]],
                                   [t_arr[i]], [nt_arr[i]])[0]
            return fn
        elif model_name == "T-F":
            def fn(r):
                i = row_idx_map[id(r)]
                return _predict_tf(params, [macs_arr[i]], [bytes_arr[i]],
                                   [ndma_tp_arr[i]], [tp_arr[i]],
                                   [t_comp_arr[i]], [t_comm_arr[i]], [nt_arr[i]])[0]
            return fn
        elif model_name == "L-B":
            a_time, c_cores, log_C = params
            C = math.exp(log_C)
            def fn(r):
                i = row_idx_map[id(r)]
                return C * (features[i]["t_total_us"] ** a_time) * (r.n_cores ** c_cores)
            return fn
        return lambda r: 0

    def _fast_time_fn(r):
        return features[row_idx_map[id(r)]]["t_total_us"]

    # Comparison table
    print(f"\n{'Model':<8} {'Params':>6} {'rho':>7} {'MAPE%':>7} "
          f"{'EDP Acc':>10} {'Regret':>10} {'MaxReg':>10}")
    print("-" * 72)

    best_model = None
    best_score = (-1, 999, 999)
    edp_details: Dict[str, Dict] = {}

    for model_name in ALL_MODELS:
        if model_name not in model_results:
            continue
        params, fit = model_results[model_name]

        e_fn = _make_fast_energy_fn(model_name, params)
        n_correct, n_sizes, mean_regret, max_regret = _edp_core_accuracy_and_regret(
            rows, e_fn, _fast_time_fn)

        acc_str = f"{n_correct}/{n_sizes} ({n_correct/n_sizes*100:.0f}%)" if n_sizes > 0 else "N/A"

        print(f"{model_name:<8} {fit.n_params:>6} {fit.rho:>7.4f} {fit.mape:>7.1f} "
              f"{acc_str:>10} {mean_regret:>9.1f}% {max_regret:>9.1f}%")

        edp_details[model_name] = {
            "n_correct": n_correct, "n_sizes": n_sizes,
            "mean_regret": mean_regret, "max_regret": max_regret}

        core_acc = n_correct / n_sizes if n_sizes > 0 else 0
        score = (core_acc, -mean_regret, -fit.mape)
        if model_name != "L-B" and score > best_score:
            best_score = score
            best_model = model_name

    # SP shape analysis
    print(f"\nSP Shape Error Analysis:")
    print(f"{'Model':<8} {'Balanced MAPE%':>15} {'Unbalanced MAPE%':>17} {'Delta':>8}")
    print("-" * 55)

    for model_name in ALL_MODELS:
        if model_name not in model_results:
            continue
        _, fit = model_results[model_name]
        preds = fit.predictions_uj

        bal_errs = [abs(preds[i] - y[i]) / y[i] * 100
                    for i in range(len(rows)) if rows[i].SPm == rows[i].SPn]
        unbal_errs = [abs(preds[i] - y[i]) / y[i] * 100
                      for i in range(len(rows)) if rows[i].SPm != rows[i].SPn]

        bal_mape = sum(bal_errs) / len(bal_errs) if bal_errs else 0
        unbal_mape = sum(unbal_errs) / len(unbal_errs) if unbal_errs else 0
        delta = unbal_mape - bal_mape

        print(f"{model_name:<8} {bal_mape:>15.1f} {unbal_mape:>17.1f} {delta:>+8.1f}")

    # EDP detail: per-size correct/incorrect for best model
    if best_model:
        print(f"\nEDP Detail (best model: {best_model}):")
        params, _ = model_results[best_model]
        e_fn = _make_fast_energy_fn(best_model, params)

        by_size: Dict[str, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            ep = e_fn(r)
            tp = _fast_time_fn(r)
            if ep > 0 and tp > 0:
                by_size[r.size_key][r.n_cores].append((r, ep, tp))

        print(f"  {'Size':<22} {'Meas Opt':>9} {'Model Opt':>10} {'Match':>6} {'Regret%':>9}")
        print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*6} {'-'*9}")

        for sk in sorted(by_size.keys(), key=lambda s: tuple(int(x) for x in s.split('x'))):
            cores_dict = by_size[sk]
            if len(cores_dict) < 2:
                continue
            meas_edp = {}
            model_edp = {}
            for nc, entries in cores_dict.items():
                energies = [r.npu_per_iter_uj for r, _, _ in entries]
                times = [r.avg_iter_us for r, _, _ in entries]
                meas_edp[nc] = _median(energies) * _median(times)
                pred_es = [ep for _, ep, _ in entries]
                pred_ts = [tp for _, _, tp in entries]
                model_edp[nc] = _median(pred_es) * _median(pred_ts)

            meas_opt = min(meas_edp, key=meas_edp.get)
            model_opt = min(model_edp, key=model_edp.get)
            match = "OK" if meas_opt == model_opt else "MISS"
            regret = max(0, (meas_edp[model_opt] - meas_edp[meas_opt]) / meas_edp[meas_opt] * 100)
            print(f"  {sk:<22} {meas_opt:>9} {model_opt:>10} {match:>6} {regret:>8.1f}%")

    print(f"\n--- Recommendation ---")
    if best_model:
        _, best_fit = model_results[best_model]
        print(f"Best theoretical model: {best_model} ({best_fit.n_params} params)")
        print(f"  rho={best_fit.rho:.4f}, MAPE={best_fit.mape:.1f}%")
        print(f"  Parameters: {best_fit.params}")

    return best_model, model_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Energy model redesign analysis")
    parser.add_argument("--csv", type=Path, required=True,
                        help="Result CSV (result_phase1.csv)")
    parser.add_argument("--tc", type=Path, required=True,
                        help="TC list JSON (tc_list_dedup.json)")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH,
                        help="Calibration JSON")
    parser.add_argument("--min-wall-s", type=float, default=DEFAULT_MIN_WALL_S,
                        help="Minimum wall time for RAPL validity")
    args = parser.parse_args()

    coeffs = load_calibration(args.calib)
    print(f"Loaded calibration v{coeffs.l_core_cy != 0 and 9 or 8}: "
          f"l_core={coeffs.l_core_cy}, l_dma={coeffs.l_dma_cy}")

    # Step 1-A: Energy basis analysis
    rows = step1a_energy_basis(args.csv, args.tc, args.min_wall_s)

    # Compute features with v9 model
    features = compute_features(rows, coeffs)
    print(f"\nFeatures computed: {len(features)} rows, "
          f"first n_dma={features[0]['n_dma']}, tp_total={features[0]['tp_total']}")

    # Step 1-B: Error structure
    step1b_error_structure(rows, features, coeffs)

    # Step 2: Calibrate all candidates
    model_results = step2_calibrate(rows, features, coeffs)

    # Step 3: Compare
    best_model, all_results = step3_compare(rows, features, model_results)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
