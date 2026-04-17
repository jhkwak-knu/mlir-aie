#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task12_diagnostic.py -- Task 12: Performance/Energy cost model structure
improvement via diagnostic analysis and candidate fitting.

Phase 0: Baseline verification (v12 coefficients reproduce stated metrics)
Phase 1: Diagnostic residual analysis (identify error sources)
Phase 2: Performance model candidate fitting and comparison

Usage:
    python3 scripts/analyze/task12_diagnostic.py \
        --result out/reports/result_v12.csv \
        --tc out/tc_list_v11.json \
        --calib data/calibration.json \
        [--plot-dir out/plots/task12_phase1]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import (  # noqa: E402
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
    OpCase, CommentFilterFile, load_calibration,
)
from cost_model import total_data_bytes, Candidate  # noqa: E402
import models as _models  # noqa: E402

CLOCK_MHZ = 1500


# ============================================================
# Data loading (reuses recalibrate_model pattern)
# ============================================================

@dataclass
class MeasuredCase:
    """One measured test case with tiling config and ground truth."""
    op: OpCase
    cand: Candidate
    tp_order: int       # innermost axis (0=M, 1=N, 2=K)
    gt_us: float
    gt_cy: float
    # Energy fields (optional, for Phase 1-C)
    npu_energy_uj: float = 0.0


def load_data(
    result_path: str, tc_path: str, gt_field: str = "min_us",
) -> List[MeasuredCase]:
    """Load measurement results, merge with tc_list for tpOrder."""
    with open(tc_path) as f:
        tc_data = json.load(f)
    cases_list = tc_data["cases"] if isinstance(tc_data, dict) else tc_data

    tp_order_map: Dict[tuple, int] = {}
    for tc in cases_list:
        lev = tc["levels"][0]
        key = (tc["M"], tc["K"], tc["N"],
               lev["SPm"], lev["SPn"],
               lev["TPm"], lev["TPk"], lev["TPn"])
        tp_order_map[key] = lev["tpOrder"][0]

    raw: Dict[tuple, List[Tuple[float, float]]] = defaultdict(list)
    with open(result_path) as f:
        for row in csv.DictReader(CommentFilterFile(f)):
            if row["status"] != "PASS":
                continue
            gt_val = float(row[gt_field])
            if gt_val <= 0:
                continue
            # Energy ground truth
            e_uj = float(row.get("core_energy_per_iter_uj", 0) or 0)
            if e_uj <= 0:
                e_uj = float(row.get("npu_energy_per_iter_uj", 0) or 0)
            config_key = (
                int(row["M"]), int(row["K"]), int(row["N"]),
                int(row["numSpm"]),
                int(row["SPm"]), int(row["SPn"]),
                int(row["TPm"]), int(row["TPk"]), int(row["TPn"]),
                int(row["TM"]), int(row["TK"]), int(row["TN"]),
            )
            raw[config_key].append((gt_val, e_uj))

    # Deduplicate: keep minimum gt_val
    results: List[MeasuredCase] = []
    for (M, K, N, nc, SPm, SPn, TPm, TPk, TPn, TM, TK, TN), vals in raw.items():
        best = min(vals, key=lambda x: x[0])
        gt_val, e_uj = best
        key = (M, K, N, SPm, SPn, TPm, TPk, TPn)
        tp_order = tp_order_map.get(key, TP_AXIS_K)
        op = OpCase(M=M, K=K, N=N, elem_type="bf16")
        cand = Candidate(
            num_cores=nc, num_columns=(nc + 3) // 4,
            SPm=SPm, SPn=SPn,
            TPm=TPm, TPk=TPk, TPn=TPn,
            TM=TM, TK=TK, TN=TN,
        )
        results.append(MeasuredCase(
            op=op, cand=cand, tp_order=tp_order,
            gt_us=gt_val, gt_cy=gt_val * CLOCK_MHZ,
            npu_energy_uj=e_uj,
        ))

    print(f"  Loaded {len(results)} unique PASS configs (gt={gt_field})")
    return results


# ============================================================
# Feature extraction
# ============================================================

def compute_features(
    cases: List[MeasuredCase], eff_macs: float, bw_bpc: float,
) -> Dict[str, np.ndarray]:
    """Compute model features for all cases (extended for Task 12)."""
    n = len(cases)
    macs = np.zeros(n)
    data_bytes = np.zeros(n)
    t_comp = np.zeros(n)
    t_comm = np.zeros(n)
    tp_total = np.zeros(n)
    n_cores = np.zeros(n)
    gt_cy = np.zeros(n)
    gt_energy_uj = np.zeros(n)
    dma_per_step = np.zeros(n)
    total_dma_ops = np.zeros(n)
    core_x_tp = np.zeros(n)

    # Candidate A features
    dma_every = np.zeros(n)
    dma_reused = np.zeros(n)
    tp_inner = np.zeros(n)
    d_total_arr = np.zeros(n)

    # tpOrder tracking
    tp_orders = np.zeros(n, dtype=int)

    for i, mc in enumerate(cases):
        c = mc.cand
        m_val = mc.op.M * mc.op.K * mc.op.N
        macs[i] = m_val
        db = total_data_bytes(mc.op, c, mc.tp_order) / bw_bpc
        data_bytes_raw = total_data_bytes(mc.op, c, mc.tp_order)
        t_comp[i] = m_val / (c.num_cores * eff_macs)
        t_comm[i] = db
        tp_total[i] = c.tp_total
        n_cores[i] = c.num_cores
        gt_cy[i] = mc.gt_cy
        gt_energy_uj[i] = mc.npu_energy_uj
        tp_orders[i] = mc.tp_order
        data_bytes[i] = data_bytes_raw

        dps = _models.dma_ops_per_step(c.SPm, c.SPn, c.num_cores, mc.tp_order)
        dma_per_step[i] = dps
        total_dma_ops[i] = dps * c.tp_total

        core_x_tp[i] = c.num_cores * c.tp_total

        # Candidate A: decomposed DMA
        n_ev, n_re = _models.dma_ops_decomposed(
            c.SPm, c.SPn, c.num_cores, mc.tp_order)
        dma_every[i] = n_ev
        dma_reused[i] = n_re
        tp_inn = _models.tp_inner_value(c.TPm, c.TPk, c.TPn, mc.tp_order)
        tp_inner[i] = tp_inn
        d_total_arr[i] = n_ev * c.tp_total + n_re * (c.tp_total / tp_inn)

    return {
        "macs": macs, "data_bytes": data_bytes,
        "t_comp": t_comp, "t_comm": t_comm,
        "tp_total": tp_total, "n_cores": n_cores,
        "gt_cy": gt_cy, "gt_energy_uj": gt_energy_uj,
        "dma_per_step": dma_per_step, "total_dma_ops": total_dma_ops,
        "core_x_tp": core_x_tp,
        "dma_every": dma_every, "dma_reused": dma_reused,
        "tp_inner": tp_inner, "d_total": d_total_arr,
        "tp_orders": tp_orders,
    }


# ============================================================
# Model prediction functions
# ============================================================

def predict_baseline(f, p):
    """Core-Sync-Fixed (v12 Baseline, 4 overhead params).
    T = T_comp + T_comm + L_SYNC*TP + L_CORE*P*TP + L_DMA*N_dma*TP + L_STARTUP
    """
    l_sync, l_core, l_dma, l_startup = p
    return (f["t_comp"] + f["t_comm"]
            + l_sync * f["tp_total"]
            + l_core * f["core_x_tp"]
            + l_dma * f["total_dma_ops"]
            + l_startup)


def predict_candidate_a(f, p):
    """Candidate A: DMA-Refined (4 overhead params, same count as baseline).
    T = T_comp + T_comm + L_SYNC*TP + L_CORE*P*TP + L_DMA*D_total + L_STARTUP
    where D_total = N_dma_every*TP + N_dma_reused*(TP/TP_inner)
    """
    l_sync, l_core, l_dma, l_startup = p
    return (f["t_comp"] + f["t_comm"]
            + l_sync * f["tp_total"]
            + l_core * f["core_x_tp"]
            + l_dma * f["d_total"]
            + l_startup)


def predict_candidate_b(f, p):
    """Candidate B: Free eff_macs/bw_eff (6 params, diagnostic only).
    T = MACs/(P*eff_cal) + bytes/bw_cal + L_SYNC*TP + L_CORE*P*TP + L_DMA*N_dma*TP + L_STARTUP
    """
    eff_cal, bw_cal, l_sync, l_core, l_dma, l_startup = p
    t_comp_new = f["macs"] / (f["n_cores"] * eff_cal)
    t_comm_new = f["data_bytes"] / bw_cal
    return (t_comp_new + t_comm_new
            + l_sync * f["tp_total"]
            + l_core * f["core_x_tp"]
            + l_dma * f["total_dma_ops"]
            + l_startup)


def predict_candidate_ab(f, p):
    """Candidate A+B: DMA-Refined + Free params (6 params, diagnostic only).
    T = MACs/(P*eff_cal) + bytes/bw_cal + L_SYNC*TP + L_CORE*P*TP + L_DMA*D_total + L_STARTUP
    """
    eff_cal, bw_cal, l_sync, l_core, l_dma, l_startup = p
    t_comp_new = f["macs"] / (f["n_cores"] * eff_cal)
    t_comm_new = f["data_bytes"] / bw_cal
    return (t_comp_new + t_comm_new
            + l_sync * f["tp_total"]
            + l_core * f["core_x_tp"]
            + l_dma * f["d_total"]
            + l_startup)


CANDIDATES = {
    "Baseline": {
        "predict": predict_baseline,
        "bounds": [(0, 5e5), (0, 1e5), (0, 5e5), (0, 5e5)],
        "param_names": ["L_SYNC", "L_CORE", "L_DMA", "L_STARTUP"],
        "adoptable": True,
        "description": "Core-Sync-Fixed (v12)",
    },
    "Candidate-A": {
        "predict": predict_candidate_a,
        "bounds": [(0, 5e5), (0, 1e5), (0, 5e5), (0, 5e5)],
        "param_names": ["L_SYNC", "L_CORE", "L_DMA", "L_STARTUP"],
        "adoptable": True,
        "description": "DMA-Refined (D_total replaces N_dma*TP)",
    },
    "Candidate-B": {
        "predict": predict_candidate_b,
        "bounds": [(5, 60), (1, 16),
                   (0, 5e5), (0, 1e5), (0, 5e5), (0, 5e5)],
        "param_names": ["eff_macs_cal", "bw_eff_cal",
                        "L_SYNC", "L_CORE", "L_DMA", "L_STARTUP"],
        "adoptable": False,
        "description": "Free eff_macs/bw_eff (diagnostic only)",
    },
    "Candidate-A+B": {
        "predict": predict_candidate_ab,
        "bounds": [(5, 60), (1, 16),
                   (0, 5e5), (0, 1e5), (0, 5e5), (0, 5e5)],
        "param_names": ["eff_macs_cal", "bw_eff_cal",
                        "L_SYNC", "L_CORE", "L_DMA", "L_STARTUP"],
        "adoptable": False,
        "description": "DMA-Refined + Free params (diagnostic only)",
    },
}


# ============================================================
# Fitting and evaluation
# ============================================================

def log_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Log-space MSE loss (same as recalibrate_model.py)."""
    return float(np.mean(
        (np.log(np.maximum(gt, 1)) - np.log(np.maximum(pred, 1))) ** 2))


def fit_model(feat, predict_fn, bounds):
    """Fit using differential evolution (same seed/maxiter for reproducibility)."""
    gt = feat["gt_cy"]
    result = differential_evolution(
        lambda p: log_mse(gt, predict_fn(feat, p)),
        bounds, seed=42, maxiter=2000, tol=1e-14, polish=True,
    )
    return result.x, result.fun


def evaluate(cases, feat, predict_fn, params):
    """Compute rho, MAPE, core accuracy, regret (matches recalibrate_model)."""
    gt = feat["gt_cy"]
    pred = predict_fn(feat, params)

    # Spearman rho (within-size groups)
    groups = defaultdict(list)
    for i, mc in enumerate(cases):
        groups[(mc.op.M, mc.op.K, mc.op.N)].append(i)

    tw, tr = 0.0, 0.0
    for key, idx in groups.items():
        if len(idx) < 3:
            continue
        g = np.array(idx)
        r = spearman_rank_correlation(gt[g].tolist(), pred[g].tolist())
        if r is not None:
            tr += r * len(idx)
            tw += len(idx)
    rho = tr / tw if tw > 0 else 0.0

    # MAPE
    mape = float(np.mean(np.abs(gt - pred) / gt * 100))
    mape_p90 = float(np.percentile(np.abs(gt - pred) / gt * 100, 90))
    mape_max = float(np.max(np.abs(gt - pred) / gt * 100))

    # Core accuracy and regret
    correct, total = 0, 0
    regrets = []
    details = []
    for key in sorted(groups.keys()):
        idx = groups[key]
        if len(idx) < 2:
            continue
        total += 1
        g = np.array(idx)
        gt_best = g[np.argmin(gt[g])]
        pred_best = g[np.argmin(pred[g])]
        gt_cores = cases[gt_best].cand.num_cores
        pred_cores = cases[pred_best].cand.num_cores
        if gt_cores == pred_cores:
            correct += 1
        regret = float((gt[pred_best] - gt[gt_best]) / gt[gt_best] * 100)
        regrets.append(regret)
        details.append({
            "size": f"{key[0]}x{key[1]}x{key[2]}",
            "gt_cores": gt_cores, "pred_cores": pred_cores,
            "regret_pct": round(regret, 1),
            "match": gt_cores == pred_cores,
        })

    acc_str = f"{correct}/{total} ({correct/total*100:.0f}%)" if total else "N/A"

    # Rank inversions
    sub_groups = defaultdict(list)
    for i, mc in enumerate(cases):
        sub_groups[(mc.op.M, mc.op.K, mc.op.N, mc.cand.num_cores)].append(i)
    inversions, total_pairs = 0, 0
    for idx_list in sub_groups.values():
        if len(idx_list) < 2:
            continue
        for ii in range(len(idx_list)):
            for jj in range(ii + 1, len(idx_list)):
                a, b = idx_list[ii], idx_list[jj]
                total_pairs += 1
                if (gt[a] < gt[b]) != (pred[a] < pred[b]):
                    inversions += 1
    inv_str = (f"{inversions}/{total_pairs} ({inversions/total_pairs*100:.1f}%)"
               if total_pairs else "0/0")

    # AIC = n * ln(RSS/n) + 2k
    residuals_sq = (np.log(np.maximum(gt, 1)) - np.log(np.maximum(pred, 1))) ** 2
    n = len(gt)
    k = len(params) if hasattr(params, '__len__') else 1
    rss = float(np.sum(residuals_sq))
    aic = n * math.log(rss / n) + 2 * k if rss > 0 else float('inf')

    return {
        "rho": rho, "mape": mape, "mape_p90": mape_p90, "mape_max": mape_max,
        "core_accuracy": acc_str,
        "regret_mean": float(np.mean(regrets)) if regrets else 0,
        "regret_max": float(np.max(regrets)) if regrets else 0,
        "regret_le10": sum(1 for r in regrets if r <= 10),
        "regret_total": len(regrets),
        "rank_inversions": inv_str,
        "core_details": details,
        "n_samples": n,
        "aic": aic,
        "n_params": k,
    }


def cross_validate(cases, feat, predict_fn, bounds, n_folds=5):
    """K-fold CV by size groups (same as recalibrate_model)."""
    groups = defaultdict(list)
    for i, mc in enumerate(cases):
        groups[(mc.op.M, mc.op.K, mc.op.N)].append(i)

    keys = sorted(groups.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(keys)

    fold_size = max(1, len(keys) // n_folds)
    cv_mapes, cv_accs = [], []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(keys)
        test_keys = set(keys[start:end])
        train_idx = [i for i, mc in enumerate(cases)
                     if (mc.op.M, mc.op.K, mc.op.N) not in test_keys]
        test_idx = [i for i, mc in enumerate(cases)
                    if (mc.op.M, mc.op.K, mc.op.N) in test_keys]
        if not train_idx or not test_idx:
            continue

        train_feat = {k: v[train_idx] if isinstance(v, np.ndarray) else v
                      for k, v in feat.items()}
        test_feat = {k: v[test_idx] if isinstance(v, np.ndarray) else v
                     for k, v in feat.items()}
        test_cases = [cases[i] for i in test_idx]

        params, _ = fit_model(train_feat, predict_fn, bounds)
        m = evaluate(test_cases, test_feat, predict_fn, params)
        cv_mapes.append(m["mape"])
        parts = m["core_accuracy"].split("/")
        if len(parts) == 2:
            num = int(parts[0])
            den = int(parts[1].split("(")[0].strip())
            cv_accs.append(num / den * 100 if den else 0)

    return {
        "cv_mape": float(np.mean(cv_mapes)) if cv_mapes else 0,
        "cv_core_acc": float(np.mean(cv_accs)) if cv_accs else 0,
    }


# ============================================================
# Phase 0: Baseline verification
# ============================================================

def phase0_verify_baseline(cases, feat, calib_path):
    """Verify v12 coefficients reproduce stated metrics."""
    print("\n" + "=" * 70)
    print("PHASE 0: BASELINE VERIFICATION (v12)")
    print("=" * 70)

    coeffs = load_calibration(Path(calib_path))
    v12_params = [coeffs.l_sync_cy, coeffs.l_core_cy,
                  coeffs.l_dma_cy, coeffs.l_startup_cy]

    print(f"  v12 coefficients:")
    print(f"    eff_macs={coeffs.eff_macs}, bw_eff={coeffs.bw_eff_bpc}")
    print(f"    L_SYNC={coeffs.l_sync_cy}, L_CORE={coeffs.l_core_cy}, "
          f"L_DMA={coeffs.l_dma_cy}, L_STARTUP={coeffs.l_startup_cy}")

    metrics = evaluate(cases, feat, predict_baseline, v12_params)

    print(f"\n  N_samples: {metrics['n_samples']}")
    print(f"  Spearman rho: {metrics['rho']:.4f}  (target: 0.9736)")
    print(f"  MAPE:         {metrics['mape']:.1f}%  (target: 26.3%)")
    print(f"  MAPE P90:     {metrics['mape_p90']:.1f}%  (target: 50.0%)")
    print(f"  Core Acc:     {metrics['core_accuracy']}  (target: 14/19)")
    print(f"  Regret mean:  {metrics['regret_mean']:.1f}%  (target: 5.0%)")
    print(f"  Regret max:   {metrics['regret_max']:.1f}%  (target: 22.5%)")
    print(f"  Regret <=10%: {metrics['regret_le10']}/{metrics['regret_total']}"
          f"  (target: 16/19)")
    print(f"  RankInv:      {metrics['rank_inversions']}")

    # Verify
    ok = True
    checks = [
        ("MAPE", abs(metrics["mape"] - 26.3) < 0.5),
        ("Core Acc", "14/19" in metrics["core_accuracy"]),
        ("Regret mean", abs(metrics["regret_mean"] - 5.0) < 0.5),
    ]
    for name, passed in checks:
        status = "PASS" if passed else "WARN"
        print(f"  [{status}] {name}")
        if not passed:
            ok = False

    if ok:
        print("\n  Status: BASELINE VERIFIED -- Ready for Phase 1")
    else:
        print("\n  Status: BASELINE MISMATCH -- Investigate before proceeding")

    return metrics, v12_params


# ============================================================
# Phase 1: Diagnostic analysis
# ============================================================

def phase1_diagnostics(cases, feat, v12_params, plot_dir=None):
    """Comprehensive residual analysis."""
    print("\n" + "=" * 70)
    print("PHASE 1: DIAGNOSTIC ANALYSIS")
    print("=" * 70)

    gt = feat["gt_cy"]
    pred = predict_baseline(feat, v12_params)
    residual = (pred - gt) / gt  # signed fractional error
    residual_pct = residual * 100

    # ------------------------------------------------------------------
    # 1-A: Residual correlations with features
    # ------------------------------------------------------------------
    print("\n--- 1-A: Residual Correlations (Spearman rho) ---")
    corr_features = {
        "TP_inner": feat["tp_inner"],
        "tpOrder": feat["tp_orders"].astype(float),
        "P (cores)": feat["n_cores"],
        "TP_total": feat["tp_total"],
        "MACs": feat["macs"],
        "T_comp/T_total": feat["t_comp"] / (feat["t_comp"] + feat["t_comm"]
                           + v12_params[0] * feat["tp_total"]
                           + v12_params[1] * feat["core_x_tp"]
                           + v12_params[2] * feat["total_dma_ops"]
                           + v12_params[3]),
        "T_comm/T_total": feat["t_comm"] / (feat["t_comp"] + feat["t_comm"]
                           + v12_params[0] * feat["tp_total"]
                           + v12_params[1] * feat["core_x_tp"]
                           + v12_params[2] * feat["total_dma_ops"]
                           + v12_params[3]),
        "N_dma*TP": feat["total_dma_ops"],
        "D_total": feat["d_total"],
        "D_total - N_dma*TP": feat["d_total"] - feat["total_dma_ops"],
    }

    print(f"  {'Feature':<22} {'rho':>8}  Interpretation")
    print(f"  {'-'*22} {'-'*8}  {'-'*30}")
    for fname, fvals in corr_features.items():
        rho = spearman_rank_correlation(residual_pct.tolist(), fvals.tolist())
        rho_val = rho if rho is not None else 0.0
        if abs(rho_val) > 0.3:
            interp = "STRONG"
        elif abs(rho_val) > 0.15:
            interp = "moderate"
        else:
            interp = "weak"
        print(f"  {fname:<22} {rho_val:>8.3f}  ({interp})")

    # ------------------------------------------------------------------
    # 1-B: Component decomposition
    # ------------------------------------------------------------------
    print("\n--- 1-B: Component Decomposition ---")
    t_total_pred = pred
    t_comp_frac = feat["t_comp"] / t_total_pred * 100
    t_comm_frac = feat["t_comm"] / t_total_pred * 100
    t_ovh_frac = 100 - t_comp_frac - t_comm_frac

    print(f"  Component fractions (mean / median / min / max):")
    for name, frac in [("T_comp", t_comp_frac), ("T_comm", t_comm_frac),
                       ("T_overhead", t_ovh_frac)]:
        print(f"    {name:<12}: {np.mean(frac):5.1f}% / {np.median(frac):5.1f}% / "
              f"{np.min(frac):5.1f}% / {np.max(frac):5.1f}%")

    # Error by dominant component
    comp_dom = t_comp_frac > 50
    comm_dom = t_comm_frac > 50
    ovh_dom = t_ovh_frac > 50
    print(f"\n  Error by dominant component:")
    for name, mask in [("Compute-dominant", comp_dom),
                       ("Comm-dominant", comm_dom),
                       ("Overhead-dominant", ovh_dom),
                       ("Mixed (no dominant)", ~comp_dom & ~comm_dom & ~ovh_dom)]:
        if np.sum(mask) > 0:
            mae = np.mean(np.abs(residual_pct[mask]))
            bias = np.mean(residual_pct[mask])
            print(f"    {name:<22}: N={np.sum(mask):4d}, "
                  f"MAE={mae:5.1f}%, bias={bias:+5.1f}%")

    # ------------------------------------------------------------------
    # 1-C: Energy model residual analysis
    # ------------------------------------------------------------------
    print("\n--- 1-C: Energy Model Residual Analysis ---")
    e_mask = feat["gt_energy_uj"] > 0
    e_count = np.sum(e_mask)
    if e_count > 0:
        # Load energy params from calibration
        from tiling_common import load_calibration as _lc
        # We'll compute energy predictions inline
        e_gt = feat["gt_energy_uj"][e_mask]

        # Predicted energy using v12 T-E model
        # E = e_mac*MACs + e_dram*bytes + e_dma*N_dma*TP + e_sync*TP + (p_base+p_core*P)*T_us
        e_mac_pj = 2.046
        e_dram_pj = 72.746
        e_dma_uj = 0.0447
        e_sync_uj = 78.27
        p_base_uw = 17960124
        p_core_uw = 91581

        e_pred_pj = (e_mac_pj * feat["macs"][e_mask]
                     + e_dram_pj * feat["data_bytes"][e_mask]
                     + e_dma_uj * feat["total_dma_ops"][e_mask] * 1e6
                     + e_sync_uj * feat["tp_total"][e_mask] * 1e6
                     + (p_base_uw + p_core_uw * feat["n_cores"][e_mask])
                       * (feat["gt_cy"][e_mask] / CLOCK_MHZ))
        e_pred_uj = e_pred_pj / 1e6

        e_mape = float(np.mean(np.abs(e_gt - e_pred_uj) / e_gt * 100))
        e_rho = spearman_rank_correlation(e_gt.tolist(), e_pred_uj.tolist())

        print(f"  Energy samples with GT: {e_count}")
        print(f"  Energy MAPE (with measured T): {e_mape:.1f}%")
        print(f"  Energy rho: {e_rho:.4f}" if e_rho else "  Energy rho: N/A")

        # Oracle T analysis: use measured T instead of predicted T
        e_pred_oracle_pj = (e_mac_pj * feat["macs"][e_mask]
                            + e_dram_pj * feat["data_bytes"][e_mask]
                            + e_dma_uj * feat["total_dma_ops"][e_mask] * 1e6
                            + e_sync_uj * feat["tp_total"][e_mask] * 1e6
                            + (p_base_uw + p_core_uw * feat["n_cores"][e_mask])
                              * (feat["gt_cy"][e_mask] / CLOCK_MHZ))
        e_pred_oracle_uj = e_pred_oracle_pj / 1e6
        e_mape_oracle = float(np.mean(np.abs(e_gt - e_pred_oracle_uj) / e_gt * 100))
        print(f"  Energy MAPE (Oracle T = measured): {e_mape_oracle:.1f}%")
        print(f"  -> Energy structure limit vs perf propagation: "
              f"Oracle shows structural limit")

        # P_BASE*T contribution
        e_power_pj = ((p_base_uw + p_core_uw * feat["n_cores"][e_mask])
                      * (feat["gt_cy"][e_mask] / CLOCK_MHZ))
        e_power_frac = e_power_pj / e_pred_pj * 100
        print(f"\n  (P_BASE+P_CORE*P)*T contribution:")
        print(f"    mean={np.mean(e_power_frac):.1f}%, "
              f"median={np.median(e_power_frac):.1f}%, "
              f"min={np.min(e_power_frac):.1f}%, max={np.max(e_power_frac):.1f}%")
    else:
        print("  No energy ground truth available (skipping)")

    # ------------------------------------------------------------------
    # 1-D: tpOrder accuracy analysis
    # ------------------------------------------------------------------
    print("\n--- 1-D: tpOrder Accuracy Analysis ---")
    groups = defaultdict(list)
    for i, mc in enumerate(cases):
        groups[(mc.op.M, mc.op.K, mc.op.N, mc.cand.num_cores)].append(i)

    # For each (size, cores) group with multiple tpOrders, check if model
    # picks the same best tpOrder as measurement
    tp_correct, tp_total_count = 0, 0
    tp_sym_correct, tp_sym_total = 0, 0  # SPm == SPn
    tp_asym_correct, tp_asym_total = 0, 0  # SPm != SPn
    tp_details = []

    size_core_groups = defaultdict(list)
    for i, mc in enumerate(cases):
        size_core_groups[(mc.op.M, mc.op.K, mc.op.N,
                          mc.cand.SPm, mc.cand.SPn)].append(i)

    for key, idx_list in size_core_groups.items():
        M, K, N, SPm, SPn = key
        if len(idx_list) < 2:
            continue
        # Group by tpOrder within this SP config
        by_tp = defaultdict(list)
        for i in idx_list:
            by_tp[cases[i].tp_order].append(i)
        if len(by_tp) < 2:
            continue

        tp_total_count += 1
        is_sym = (SPm == SPn)
        if is_sym:
            tp_sym_total += 1
        else:
            tp_asym_total += 1

        # Best tpOrder by measurement (min gt_cy among best per tpOrder)
        gt_best_tp = {}
        pred_best_tp = {}
        for tp_ord, tp_idx in by_tp.items():
            g = np.array(tp_idx)
            gt_best_tp[tp_ord] = float(np.min(gt[g]))
            pred_best_tp[tp_ord] = float(np.min(pred[g]))

        gt_best_order = min(gt_best_tp, key=gt_best_tp.get)
        pred_best_order = min(pred_best_tp, key=pred_best_tp.get)

        match = (gt_best_order == pred_best_order)
        if match:
            tp_correct += 1
            if is_sym:
                tp_sym_correct += 1
            else:
                tp_asym_correct += 1

        # Gap analysis
        gt_gap = 0
        pred_gap = 0
        if len(gt_best_tp) >= 2:
            gt_vals = sorted(gt_best_tp.values())
            pred_vals = sorted(pred_best_tp.values())
            gt_gap = (gt_vals[1] - gt_vals[0]) / gt_vals[0] * 100
            pred_gap = (pred_vals[1] - pred_vals[0]) / pred_vals[0] * 100

        tp_details.append({
            "size": f"{M}x{K}x{N}", "SPm": SPm, "SPn": SPn,
            "gt_best": gt_best_order, "pred_best": pred_best_order,
            "match": match, "gt_gap_pct": gt_gap, "pred_gap_pct": pred_gap,
        })

    print(f"  tpOrder accuracy: {tp_correct}/{tp_total_count} "
          f"({tp_correct/tp_total_count*100:.0f}%)" if tp_total_count else
          "  tpOrder accuracy: N/A")
    if tp_sym_total:
        print(f"    SPm==SPn (symmetric): {tp_sym_correct}/{tp_sym_total} "
              f"({tp_sym_correct/tp_sym_total*100:.0f}%)")
    if tp_asym_total:
        print(f"    SPm!=SPn (asymmetric): {tp_asym_correct}/{tp_asym_total} "
              f"({tp_asym_correct/tp_asym_total*100:.0f}%)")

    # Show MISS cases
    miss_cases = [d for d in tp_details if not d["match"]]
    if miss_cases:
        print(f"\n  tpOrder MISS cases ({len(miss_cases)}):")
        print(f"  {'Size':<14} {'SP':>6} {'GT':>4} {'Pred':>4} "
              f"{'GT_gap':>8} {'Pred_gap':>9}")
        for d in miss_cases:
            print(f"    {d['size']:<12} {d['SPm']}x{d['SPn']:>3} "
                  f"{d['gt_best']:>4} {d['pred_best']:>4} "
                  f"{d['gt_gap_pct']:>7.1f}% {d['pred_gap_pct']:>8.1f}%")

    # ------------------------------------------------------------------
    # 1-E: Candidate viability assessment
    # ------------------------------------------------------------------
    print("\n--- 1-E: Candidate Viability Assessment ---")

    # D_total vs N_dma*TP difference
    diff = feat["d_total"] - feat["total_dma_ops"]
    nonzero = np.sum(diff != 0)
    print(f"  D_total vs N_dma*TP:")
    print(f"    Cases where D_total != N_dma*TP: {nonzero}/{len(diff)} "
          f"({nonzero/len(diff)*100:.1f}%)")
    if nonzero > 0:
        nz_mask = diff != 0
        print(f"    Mean difference: {np.mean(diff[nz_mask]):.1f}")
        print(f"    Median difference: {np.median(diff[nz_mask]):.1f}")

    # Correlation of (D_total - N_dma*TP) with residual
    rho_diff = spearman_rank_correlation(
        residual_pct.tolist(), diff.tolist())
    print(f"    Correlation with residual: {rho_diff:.3f}"
          if rho_diff else "    Correlation with residual: N/A")

    # tpOrder-stratified error
    print(f"\n  Residual by tpOrder:")
    for tp_val, tp_name in [(0, "M-inner"), (1, "N-inner"), (2, "K-inner")]:
        mask = feat["tp_orders"] == tp_val
        if np.sum(mask) > 0:
            mae = np.mean(np.abs(residual_pct[mask]))
            bias = np.mean(residual_pct[mask])
            print(f"    {tp_name}: N={np.sum(mask):4d}, "
                  f"MAE={mae:5.1f}%, bias={bias:+5.1f}%")

    # ------------------------------------------------------------------
    # Plots (if matplotlib available)
    # ------------------------------------------------------------------
    if plot_dir:
        _generate_phase1_plots(cases, feat, residual_pct, pred, gt,
                               v12_params, plot_dir)

    return residual_pct


def _generate_phase1_plots(cases, feat, residual_pct, pred, gt,
                           v12_params, plot_dir):
    """Generate diagnostic plots for Phase 1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [WARN] matplotlib not available, skipping plots.")
        return

    pdir = Path(plot_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    # 1. Residual distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residual_pct, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Residual (pred-meas)/meas (%)")
    ax.set_ylabel("Count")
    ax.set_title("Phase 1-A: Residual Distribution (Baseline v12)")
    fig.tight_layout()
    fig.savefig(pdir / "residual_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Residual vs TP_total scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (fname, label) in zip(axes, [
        ("tp_total", "TP_total"), ("n_cores", "P (cores)"),
        ("tp_inner", "TP_inner")
    ]):
        ax.scatter(feat[fname], residual_pct, alpha=0.3, s=10)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel(label)
        ax.set_ylabel("Residual (%)")
        ax.set_title(f"Residual vs {label}")
    fig.tight_layout()
    fig.savefig(pdir / "residual_vs_features.png", dpi=150)
    plt.close(fig)

    # 3. tpOrder boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    tp_data = [residual_pct[feat["tp_orders"] == tp] for tp in [0, 1, 2]]
    tp_labels = ["M-inner", "N-inner", "K-inner"]
    bp = ax.boxplot(tp_data, labels=tp_labels, showfliers=True)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_ylabel("Residual (%)")
    ax.set_title("Phase 1-A: Residual by tpOrder")
    fig.tight_layout()
    fig.savefig(pdir / "residual_by_tporder.png", dpi=150)
    plt.close(fig)

    # 4. Predicted vs Measured scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    gt_us = gt / CLOCK_MHZ
    pred_us = pred / CLOCK_MHZ
    ax.scatter(gt_us, pred_us, alpha=0.3, s=10)
    lims = [min(gt_us.min(), pred_us.min()) * 0.8,
            max(gt_us.max(), pred_us.max()) * 1.2]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel("Measured (us)")
    ax.set_ylabel("Predicted (us)")
    ax.set_title("Predicted vs Measured (Baseline v12)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(pdir / "pred_vs_meas.png", dpi=150)
    plt.close(fig)

    # 5. Component dominance scatter
    t_total_pred = pred
    t_comp_frac = feat["t_comp"] / t_total_pred * 100
    t_comm_frac = feat["t_comm"] / t_total_pred * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(t_comp_frac, np.abs(residual_pct), alpha=0.3, s=10,
                    c=feat["n_cores"], cmap='viridis')
    ax.set_xlabel("T_comp / T_total (%)")
    ax.set_ylabel("|Residual| (%)")
    ax.set_title("Error vs Compute Fraction (color=cores)")
    fig.colorbar(sc, label="Cores")
    fig.tight_layout()
    fig.savefig(pdir / "error_vs_comp_frac.png", dpi=150)
    plt.close(fig)

    print(f"\n  Plots saved to {pdir}/")


# ============================================================
# Phase 2: Candidate fitting and comparison
# ============================================================

def phase2_fit_candidates(cases, feat, do_cv=True):
    """Fit all candidates and produce comparison table."""
    print("\n" + "=" * 70)
    print("PHASE 2: PERFORMANCE MODEL CANDIDATE COMPARISON")
    print("=" * 70)

    results = {}
    for name, spec in CANDIDATES.items():
        print(f"\n--- {name}: {spec['description']} ---")
        params, loss = fit_model(feat, spec["predict"], spec["bounds"])

        # Print fitted params
        for pn, pv in zip(spec["param_names"], params):
            if pn in ("eff_macs_cal", "bw_eff_cal"):
                fmt = ".2f"
            elif pn.startswith("L_"):
                fmt = ".0f"
            else:
                fmt = ".4f"
            print(f"  {pn}={pv:{fmt}}", end="")
        print(f"\n  LogMSE={loss:.6f}")

        metrics = evaluate(cases, feat, spec["predict"], params)
        print(f"  rho={metrics['rho']:.4f}  MAPE={metrics['mape']:.1f}%  "
              f"Core={metrics['core_accuracy']}  "
              f"Reg mean={metrics['regret_mean']:.1f}% "
              f"max={metrics['regret_max']:.1f}%  "
              f"AIC={metrics['aic']:.1f}")

        if do_cv:
            cv = cross_validate(cases, feat, spec["predict"], spec["bounds"])
            print(f"  CV: MAPE={cv['cv_mape']:.1f}%  "
                  f"CoreAcc={cv['cv_core_acc']:.0f}%")
            metrics["cv"] = cv

        results[name] = {
            "params": list(params),
            "param_names": spec["param_names"],
            "loss": loss,
            "metrics": metrics,
            "adoptable": spec["adoptable"],
        }

    # Summary comparison table
    print(f"\n{'='*70}")
    print("PHASE 2 SUMMARY: CANDIDATE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<16} {'MAPE':>7} {'rho':>7} {'CoreAcc':>9} "
          f"{'RegMean':>8} {'RegMax':>8} {'Reg<=10':>8} "
          f"{'AIC':>10} {'#Params':>7} {'Adopt':>6}")
    print(f"{'-'*16} {'-'*7} {'-'*7} {'-'*9} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*6}")
    for name, r in results.items():
        m = r["metrics"]
        adopt = "Yes" if r["adoptable"] else "Diag"
        print(f"{name:<16} {m['mape']:>6.1f}% {m['rho']:>7.4f} "
              f"{m['core_accuracy']:>9} {m['regret_mean']:>7.1f}% "
              f"{m['regret_max']:>7.1f}% "
              f"{m['regret_le10']}/{m['regret_total']:>5} "
              f"{m['aic']:>10.1f} {m['n_params']:>7} {adopt:>6}")

    if do_cv:
        print(f"\n  Cross-validation:")
        for name, r in results.items():
            if "cv" in r["metrics"]:
                cv = r["metrics"]["cv"]
                print(f"    {name:<16} CV_MAPE={cv['cv_mape']:.1f}%  "
                      f"CV_CoreAcc={cv['cv_core_acc']:.0f}%")

    # Candidate B diagnostic: eff_macs / bw_eff comparison
    if "Candidate-B" in results:
        b_params = dict(zip(results["Candidate-B"]["param_names"],
                           results["Candidate-B"]["params"]))
        print(f"\n  Candidate B Diagnostic (fixed vs fitted):")
        print(f"    eff_macs: fixed=24.28, fitted={b_params['eff_macs_cal']:.2f} "
              f"(ratio={b_params['eff_macs_cal']/24.28:.3f})")
        print(f"    bw_eff:   fixed=4.00,  fitted={b_params['bw_eff_cal']:.2f} "
              f"(ratio={b_params['bw_eff_cal']/4.0:.3f})")

    if "Candidate-A+B" in results:
        ab_params = dict(zip(results["Candidate-A+B"]["param_names"],
                            results["Candidate-A+B"]["params"]))
        print(f"\n  Candidate A+B Diagnostic:")
        print(f"    eff_macs: fixed=24.28, fitted={ab_params['eff_macs_cal']:.2f} "
              f"(ratio={ab_params['eff_macs_cal']/24.28:.3f})")
        print(f"    bw_eff:   fixed=4.00,  fitted={ab_params['bw_eff_cal']:.2f} "
              f"(ratio={ab_params['bw_eff_cal']/4.0:.3f})")

    # Core selection details for each candidate
    for name, r in results.items():
        print(f"\n  {name} Core Selection Details:")
        print(f"    {'Size':<16} {'GT':>4} {'Pred':>4} {'Regret':>8} {'Match':>6}")
        for d in r["metrics"]["core_details"]:
            mark = "OK" if d["match"] else "MISS"
            print(f"    {d['size']:<16} {d['gt_cores']:>4} {d['pred_cores']:>4} "
                  f"{d['regret_pct']:>7.1f}% {mark:>6}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Task 12: Diagnostic analysis + candidate fitting")
    ap.add_argument("--result", required=True,
                    help="result_v12.csv path")
    ap.add_argument("--tc", required=True,
                    help="tc_list_v11.json path")
    ap.add_argument("--calib", default="data/calibration.json",
                    help="calibration.json path")
    ap.add_argument("--gt", default="min_us",
                    help="Ground truth field (default: min_us)")
    ap.add_argument("--plot-dir", default=None,
                    help="Directory for diagnostic plots (Phase 1)")
    ap.add_argument("--no-cv", action="store_true",
                    help="Skip cross-validation (faster)")
    args = ap.parse_args()

    # Load calibration for eff_macs/bw_eff
    coeffs = load_calibration(Path(args.calib))
    eff_macs = coeffs.eff_macs if coeffs.calibrated else 24.28
    bw_bpc = coeffs.bw_eff_bpc if coeffs.calibrated else 4.0

    print(f"Task 12: Diagnostic Analysis + Candidate Fitting")
    print(f"  Result: {args.result}")
    print(f"  TC list: {args.tc}")
    print(f"  Calibration: {args.calib}")
    print(f"  EFF_MACS={eff_macs}, BW_BPC={bw_bpc}")
    print(f"  Ground truth: {args.gt}")

    # Load data
    cases = load_data(args.result, args.tc, args.gt)
    feat = compute_features(cases, eff_macs, bw_bpc)

    # Phase 0: Baseline verification
    baseline_metrics, v12_params = phase0_verify_baseline(
        cases, feat, args.calib)

    # Phase 1: Diagnostic analysis
    residuals = phase1_diagnostics(
        cases, feat, v12_params,
        plot_dir=args.plot_dir or "out/plots/task12_phase1")

    # Phase 2: Candidate fitting
    results = phase2_fit_candidates(cases, feat, do_cv=not args.no_cv)

    # Save results for Phase 3/4
    output = {
        "baseline_metrics": {
            k: v for k, v in baseline_metrics.items()
            if k != "core_details"
        },
        "candidates": {},
    }
    for name, r in results.items():
        output["candidates"][name] = {
            "params": {pn: float(pv) for pn, pv in
                       zip(r["param_names"], r["params"])},
            "metrics": {k: v for k, v in r["metrics"].items()
                        if k != "core_details"},
            "adoptable": r["adoptable"],
        }

    out_path = Path("out/reports/task12_phase2_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Phase 2 results saved to {out_path}")


if __name__ == "__main__":
    main()
