#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task14_energy_alternatives.py -- Task 14: Energy Model Alternative Structures.

Explores P-standalone energy model variants to bypass P*T vs T collinearity:

  E1 (3-param): E = P_sys*T + E_core*P + E_DMA*D_total
  E2 (4-param): E = P_sys*T + P_core*P*T + E_core*P + E_DMA*D_total

Phases:
  0:  Collinearity diagnostic (correlations + VIF)
  1A: E1 unconstrained fit
  1B: E1 constrained fit (E_core >= lower bound)
  1C: E2 unconstrained fit
  1D: Oracle T diagnostics (E1 + E2)
  2:  Comprehensive comparison (B1, B2, E1, E2 variants)
  3:  EDP evaluation

Usage:
    python3 scripts/analyze/task14_energy_alternatives.py \
        --result out/reports/result_v12.csv \
        --tc out/tc_list_v11.json \
        --calib data/calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402
from task13_compact_energy import (  # noqa: E402
    EnergyRow,
    load_energy_rows,
    compute_features,
    log_mse,
    fit_model,
    evaluate_model,
    evaluate_edp,
    compute_aic,
    predict_compact,
    predict_compact_oracle,
    predict_te_baseline,
    params_to_calib_t3,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import load_calibration  # noqa: E402

CLOCK_MHZ = 1500


# ============================================================
# E1/E2 prediction functions (optimization units: uJ)
# ============================================================

def predict_e1(params, feat):
    """E1: E = P_sys*T + E_core*P + E_DMA*D_total (uJ)."""
    p_sys, e_core, e_dma = params
    return (p_sys * feat["t_total_cy"]
            + e_core * feat["n_cores"]
            + e_dma * feat["d_total"])


def predict_e1_oracle(params, feat):
    """E1 with measured T (Oracle diagnostic)."""
    p_sys, e_core, e_dma = params
    return (p_sys * feat["t_meas_cy"]
            + e_core * feat["n_cores"]
            + e_dma * feat["d_total"])


def predict_e2(params, feat):
    """E2: E = P_sys*T + P_core*P*T + E_core*P + E_DMA*D_total (uJ)."""
    p_sys, p_core, e_core, e_dma = params
    return (p_sys * feat["t_total_cy"]
            + p_core * feat["nt"]
            + e_core * feat["n_cores"]
            + e_dma * feat["d_total"])


def predict_e2_oracle(params, feat):
    """E2 with measured T (Oracle diagnostic)."""
    p_sys, p_core, e_core, e_dma = params
    return (p_sys * feat["t_meas_cy"]
            + p_core * feat["nt_meas"]
            + e_core * feat["n_cores"]
            + e_dma * feat["d_total"])


# ============================================================
# Parameter conversion
# ============================================================

def params_to_calib_e1(params):
    """Convert E1 optimization params to calibration units."""
    p_sys, e_core, e_dma = params
    return {
        "p_sys_uw": round(p_sys * CLOCK_MHZ * 1e6, 1),
        "e_core_uj": round(e_core, 4),
        "e_dma_uj": round(e_dma, 6),
    }


def params_to_calib_e2(params):
    """Convert E2 optimization params to calibration units."""
    p_sys, p_core, e_core, e_dma = params
    return {
        "p_sys_uw": round(p_sys * CLOCK_MHZ * 1e6, 1),
        "p_core_uw": round(p_core * CLOCK_MHZ * 1e6, 1),
        "e_core_uj": round(e_core, 4),
        "e_dma_uj": round(e_dma, 6),
    }


# ============================================================
# Component analysis
# ============================================================

def component_analysis_e1(feat, params, use_oracle=False):
    """Analyze per-component energy contribution for E1."""
    p_sys, e_core, e_dma = params
    t_key = "t_meas_cy" if use_oracle else "t_total_cy"
    e_sys = p_sys * feat[t_key]
    e_core_term = e_core * feat["n_cores"]
    e_dma_term = e_dma * feat["d_total"]
    total = e_sys + e_core_term + e_dma_term
    return {
        "P_sys*T": float(np.mean(e_sys / total * 100)),
        "E_core*P": float(np.mean(e_core_term / total * 100)),
        "E_DMA*D_total": float(np.mean(e_dma_term / total * 100)),
    }


def component_analysis_e2(feat, params, use_oracle=False):
    """Analyze per-component energy contribution for E2."""
    p_sys, p_core, e_core, e_dma = params
    t_key = "t_meas_cy" if use_oracle else "t_total_cy"
    nt_key = "nt_meas" if use_oracle else "nt"
    e_sys = p_sys * feat[t_key]
    e_pcore = p_core * feat[nt_key]
    e_core_term = e_core * feat["n_cores"]
    e_dma_term = e_dma * feat["d_total"]
    total = e_sys + e_pcore + e_core_term + e_dma_term
    return {
        "P_sys*T": float(np.mean(e_sys / total * 100)),
        "P_core*P*T": float(np.mean(e_pcore / total * 100)),
        "E_core*P": float(np.mean(e_core_term / total * 100)),
        "E_DMA*D_total": float(np.mean(e_dma_term / total * 100)),
    }


# ============================================================
# Phase 0: Collinearity diagnostic
# ============================================================

def compute_vif(X):
    """Compute VIF for each column via OLS (no statsmodels needed).

    VIF_j = 1 / (1 - R^2_j) where R^2_j is from regressing X_j on
    all other columns.
    """
    n_features = X.shape[1]
    vif = np.zeros(n_features)
    for j in range(n_features):
        X_other = np.delete(X, j, axis=1)
        A = np.column_stack([X_other, np.ones(X.shape[0])])
        coeffs, _, _, _ = np.linalg.lstsq(A, X[:, j], rcond=None)
        pred = A @ coeffs
        ss_res = np.sum((X[:, j] - pred) ** 2)
        ss_tot = np.sum((X[:, j] - np.mean(X[:, j])) ** 2)
        r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        vif[j] = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else float("inf")
    return vif


def run_phase0(feat):
    """Phase 0: Collinearity diagnostic."""
    t = feat["t_total_cy"]
    pt = feat["nt"]        # P*T
    p = feat["n_cores"]    # P
    d = feat["d_total"]    # D_total

    # Feature pairs for correlation analysis
    pairs = [
        ("P*T vs T", pt, t),
        ("P vs T", p, t),
        ("P vs D_total", p, d),
        ("P vs P*T", p, pt),
        ("T vs D_total", t, d),
        ("P*T vs D_total", pt, d),
    ]

    print(f"\n  Feature Correlations:")
    print(f"  {'Pair':<20} {'Pearson':>9} {'Spearman':>9} {'R^2':>8}")
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*8}")

    corr_results = {}
    for label, x, y in pairs:
        r_pearson, _ = pearsonr(x, y)
        r_spearman, _ = spearmanr(x, y)
        corr_results[label] = {
            "pearson": round(float(r_pearson), 4),
            "spearman": round(float(r_spearman), 4),
            "r_squared": round(float(r_pearson ** 2), 4),
        }
        print(f"  {label:<20} {r_pearson:>9.4f} {r_spearman:>9.4f} "
              f"{r_pearson**2:>8.4f}")

    # VIF analysis
    # T-3 features: {T, P*T, D_total}
    X_t3 = np.column_stack([t, pt, d])
    vif_t3 = compute_vif(X_t3)
    t3_names = ["T", "P*T", "D_total"]

    # E1 features: {T, P, D_total}
    X_e1 = np.column_stack([t, p, d])
    vif_e1 = compute_vif(X_e1)
    e1_names = ["T", "P", "D_total"]

    print(f"\n  Variance Inflation Factors (VIF):")
    print(f"  {'Feature':<12} {'T-3 (P*T)':>10} {'E1 (P)':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for i in range(3):
        print(f"  {t3_names[i]:<12} {vif_t3[i]:>10.2f} {vif_e1[i]:>10.2f}")

    vif_results = {
        "T-3": {n: round(float(v), 2) for n, v in zip(t3_names, vif_t3)},
        "E1": {n: round(float(v), 2) for n, v in zip(e1_names, vif_e1)},
    }

    # Collinearity summary
    pt_t_r2 = corr_results["P*T vs T"]["r_squared"]
    p_t_r2 = corr_results["P vs T"]["r_squared"]
    print(f"\n  Summary:")
    print(f"    P*T vs T: R^2={pt_t_r2:.4f} (baseline collinearity)")
    print(f"    P  vs T:  R^2={p_t_r2:.4f} "
          f"({'lower' if p_t_r2 < pt_t_r2 else 'higher'} -- "
          f"{'collinearity reduced' if p_t_r2 < pt_t_r2 else 'no improvement'})")
    print(f"    VIF(P*T)={vif_t3[1]:.2f} vs VIF(P)={vif_e1[1]:.2f} "
          f"({'improved' if vif_e1[1] < vif_t3[1] else 'no improvement'})")

    return {"correlations": corr_results, "vif": vif_results}


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Task 14: Energy Model Alternatives (P Standalone)")
    ap.add_argument("--result", required=True)
    ap.add_argument("--tc", required=True)
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--output", default="out/reports/task14_results.json")
    args = ap.parse_args()

    print("=" * 70)
    print("TASK 14: ENERGY MODEL ALTERNATIVE STRUCTURES (P STANDALONE)")
    print("=" * 70)

    # Load data
    coeffs = load_calibration(Path(args.calib))
    rows = load_energy_rows(args.result, args.tc)
    n_samples = len(rows)
    print(f"  Loaded {n_samples} valid energy samples")

    feat = compute_features(rows, coeffs)
    gt_uj = feat["gt_uj"]

    results = {}

    # Shared optimization bounds
    p_sys_lo = 1e6 / (CLOCK_MHZ * 1e6)      # 1 W in uJ/cy
    p_sys_hi = 30e6 / (CLOCK_MHZ * 1e6)     # 30 W in uJ/cy
    p_core_lo = 0.0
    p_core_hi = 500e3 / (CLOCK_MHZ * 1e6)   # 500 mW in uJ/cy
    e_core_lo = 0.0
    e_core_hi = 10000.0                       # 10000 uJ/core
    e_dma_lo = 0.1
    e_dma_hi = 100.0                          # 100 uJ/DMA

    # ================================================================
    # Phase 0: Collinearity diagnostic
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 0: COLLINEARITY DIAGNOSTIC")
    print(f"{'='*70}")

    results["Phase0"] = run_phase0(feat)

    # ================================================================
    # Phase 1A: E1 unconstrained fit
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 1A: E1 UNCONSTRAINED (P_sys*T + E_core*P + E_DMA*D_total)")
    print(f"{'='*70}")

    bounds_e1 = [
        (p_sys_lo, p_sys_hi),
        (e_core_lo, e_core_hi),
        (e_dma_lo, e_dma_hi),
    ]

    params_1a, loss_1a = fit_model(feat, predict_e1, bounds_e1, gt_uj)
    calib_1a = params_to_calib_e1(params_1a)
    print(f"  Params: P_sys={calib_1a['p_sys_uw']/1e6:.2f}W, "
          f"E_core={calib_1a['e_core_uj']:.4f}uJ, "
          f"E_DMA={calib_1a['e_dma_uj']:.4f}uJ")
    print(f"  LogMSE: {loss_1a:.6f}")

    metrics_1a = evaluate_model(
        feat, predict_e1, params_1a, gt_uj, rows, "E1 Unconstrained")
    pred_1a = metrics_1a.pop("pred")

    comp_1a = component_analysis_e1(feat, params_1a)
    print(f"  Components: {comp_1a}")

    aic_1a = compute_aic(gt_uj, pred_1a, 3)
    print(f"  AIC: {aic_1a:.1f}")

    e_core_identified = calib_1a["e_core_uj"] > 0.01
    print(f"  E_core identifiability: {calib_1a['e_core_uj']:.4f} uJ "
          f"({'identified' if e_core_identified else 'converged to ~0'})")

    results["Phase1A"] = {
        "model": "E1", "variant": "unconstrained",
        "params": calib_1a,
        "metrics": metrics_1a, "loss": loss_1a,
        "aic": aic_1a, "components": comp_1a,
    }

    # ================================================================
    # Phase 1B: E1 with E_core lower bound (if 1A yields ~0)
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 1B: E1 CONSTRAINED (E_core >= 100 uJ)")
    print(f"{'='*70}")

    e_core_lb = 100.0  # 100 uJ lower bound
    bounds_e1_constrained = [
        (p_sys_lo, p_sys_hi),
        (e_core_lb, e_core_hi),
        (e_dma_lo, e_dma_hi),
    ]

    params_1b, loss_1b = fit_model(
        feat, predict_e1, bounds_e1_constrained, gt_uj)
    calib_1b = params_to_calib_e1(params_1b)
    print(f"  Params: P_sys={calib_1b['p_sys_uw']/1e6:.2f}W, "
          f"E_core={calib_1b['e_core_uj']:.4f}uJ, "
          f"E_DMA={calib_1b['e_dma_uj']:.4f}uJ")
    print(f"  LogMSE: {loss_1b:.6f}")

    metrics_1b = evaluate_model(
        feat, predict_e1, params_1b, gt_uj, rows, "E1 Constrained")
    pred_1b = metrics_1b.pop("pred")

    comp_1b = component_analysis_e1(feat, params_1b)
    print(f"  Components: {comp_1b}")

    aic_1b = compute_aic(gt_uj, pred_1b, 3)
    print(f"  AIC: {aic_1b:.1f}")
    print(f"  MAPE delta vs 1A: {metrics_1b['mape'] - metrics_1a['mape']:+.2f}%p")

    results["Phase1B"] = {
        "model": "E1", "variant": "constrained_e_core_100uj",
        "params": calib_1b,
        "metrics": metrics_1b, "loss": loss_1b,
        "aic": aic_1b, "components": comp_1b,
    }

    # ================================================================
    # Phase 1C: E2 unconstrained fit
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 1C: E2 UNCONSTRAINED (P_sys*T + P_core*P*T + E_core*P + E_DMA*D)")
    print(f"{'='*70}")

    bounds_e2 = [
        (p_sys_lo, p_sys_hi),
        (p_core_lo, p_core_hi),
        (e_core_lo, e_core_hi),
        (e_dma_lo, e_dma_hi),
    ]

    params_1c, loss_1c = fit_model(feat, predict_e2, bounds_e2, gt_uj)
    calib_1c = params_to_calib_e2(params_1c)
    print(f"  Params: P_sys={calib_1c['p_sys_uw']/1e6:.2f}W, "
          f"P_core={calib_1c['p_core_uw']/1e3:.2f}mW, "
          f"E_core={calib_1c['e_core_uj']:.4f}uJ, "
          f"E_DMA={calib_1c['e_dma_uj']:.4f}uJ")
    print(f"  LogMSE: {loss_1c:.6f}")

    metrics_1c = evaluate_model(
        feat, predict_e2, params_1c, gt_uj, rows, "E2 Unconstrained")
    pred_1c = metrics_1c.pop("pred")

    comp_1c = component_analysis_e2(feat, params_1c)
    print(f"  Components: {comp_1c}")

    aic_1c = compute_aic(gt_uj, pred_1c, 4)
    print(f"  AIC: {aic_1c:.1f}")

    p_core_1c_mw = calib_1c["p_core_uw"] / 1000
    e_core_1c = calib_1c["e_core_uj"]
    print(f"  Identifiability:")
    print(f"    P_core = {p_core_1c_mw:.2f} mW "
          f"({'identified' if p_core_1c_mw > 1 else 'converged to ~0'})")
    print(f"    E_core = {e_core_1c:.4f} uJ "
          f"({'identified' if e_core_1c > 0.01 else 'converged to ~0'})")

    results["Phase1C"] = {
        "model": "E2", "variant": "unconstrained",
        "params": calib_1c,
        "metrics": metrics_1c, "loss": loss_1c,
        "aic": aic_1c, "components": comp_1c,
    }

    # ================================================================
    # Phase 1D: Oracle T diagnostics
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 1D: ORACLE T DIAGNOSTICS")
    print(f"{'='*70}")

    # E1 Oracle T
    print("\n  --- E1 Oracle T ---")
    params_1d_e1, loss_1d_e1 = fit_model(
        feat, predict_e1_oracle, bounds_e1, gt_uj)
    calib_1d_e1 = params_to_calib_e1(params_1d_e1)
    print(f"  Params: P_sys={calib_1d_e1['p_sys_uw']/1e6:.2f}W, "
          f"E_core={calib_1d_e1['e_core_uj']:.4f}uJ, "
          f"E_DMA={calib_1d_e1['e_dma_uj']:.4f}uJ")

    metrics_1d_e1 = evaluate_model(
        feat, predict_e1_oracle, params_1d_e1, gt_uj, rows, "E1 Oracle T")
    pred_1d_e1 = metrics_1d_e1.pop("pred")

    comp_1d_e1 = component_analysis_e1(feat, params_1d_e1, use_oracle=True)
    print(f"  Components: {comp_1d_e1}")
    print(f"  E_core (Oracle) = {calib_1d_e1['e_core_uj']:.4f} uJ "
          f"({'identified' if calib_1d_e1['e_core_uj'] > 0.01 else '~0'})")
    print(f"  Perf error propagation: "
          f"{metrics_1a['mape'] - metrics_1d_e1['mape']:.1f}%p")

    # E2 Oracle T
    print("\n  --- E2 Oracle T ---")
    params_1d_e2, loss_1d_e2 = fit_model(
        feat, predict_e2_oracle, bounds_e2, gt_uj)
    calib_1d_e2 = params_to_calib_e2(params_1d_e2)
    print(f"  Params: P_sys={calib_1d_e2['p_sys_uw']/1e6:.2f}W, "
          f"P_core={calib_1d_e2['p_core_uw']/1e3:.2f}mW, "
          f"E_core={calib_1d_e2['e_core_uj']:.4f}uJ, "
          f"E_DMA={calib_1d_e2['e_dma_uj']:.4f}uJ")

    metrics_1d_e2 = evaluate_model(
        feat, predict_e2_oracle, params_1d_e2, gt_uj, rows, "E2 Oracle T")
    pred_1d_e2 = metrics_1d_e2.pop("pred")

    comp_1d_e2 = component_analysis_e2(feat, params_1d_e2, use_oracle=True)
    print(f"  Components: {comp_1d_e2}")
    print(f"  P_core (Oracle) = {calib_1d_e2['p_core_uw']/1e3:.2f} mW, "
          f"E_core (Oracle) = {calib_1d_e2['e_core_uj']:.4f} uJ")
    print(f"  Perf error propagation: "
          f"{metrics_1c['mape'] - metrics_1d_e2['mape']:.1f}%p")

    results["Phase1D"] = {
        "E1_oracle": {
            "params": calib_1d_e1,
            "metrics": metrics_1d_e1,
            "loss": loss_1d_e1,
            "components": comp_1d_e1,
        },
        "E2_oracle": {
            "params": calib_1d_e2,
            "metrics": metrics_1d_e2,
            "loss": loss_1d_e2,
            "components": comp_1d_e2,
        },
    }

    # ================================================================
    # Phase 2: Comprehensive comparison
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: COMPREHENSIVE COMPARISON")
    print(f"{'='*70}")

    # Load 7-param baseline (B1)
    calib_json = json.loads(Path(args.calib).read_text())
    e_section = calib_json.get("energy", {})
    bl_params_calib = {
        "e_mac_pj": e_section.get("e_mac_pj", 2.0463),
        "e_dram_pj": e_section.get("e_dram_pj", 72.7462),
        "e_dma_uj": e_section.get("e_dma_uj", 23.115563),
        "e_sync_uj": e_section.get("e_sync_uj", 8.9414),
        "p_base_uw": e_section.get("p_base_uw", 14295069),
        "p_core_uw": e_section.get("p_core_uw", 50000),
    }
    bl_optim = (
        bl_params_calib["e_mac_pj"] / 1e6,
        bl_params_calib["e_dram_pj"] / 1e6,
        bl_params_calib["e_dma_uj"],
        bl_params_calib["e_sync_uj"],
        bl_params_calib["p_base_uw"] / (CLOCK_MHZ * 1e6),
        bl_params_calib["p_core_uw"] / (CLOCK_MHZ * 1e6),
    )

    metrics_bl = evaluate_model(
        feat, predict_te_baseline, bl_optim, gt_uj, rows,
        "B1: 7-param Baseline")
    pred_bl = metrics_bl.pop("pred")
    aic_bl = compute_aic(gt_uj, pred_bl, 7)

    # T-3 unconstrained (B2) -- recompute from task13 params
    bounds_t3 = [
        (p_sys_lo, p_sys_hi),
        (p_core_lo, p_core_hi),
        (e_dma_lo, e_dma_hi),
    ]
    params_b2, loss_b2 = fit_model(feat, predict_compact, bounds_t3, gt_uj)
    calib_b2 = params_to_calib_t3(params_b2)
    metrics_b2 = evaluate_model(
        feat, predict_compact, params_b2, gt_uj, rows,
        "B2: T-3 Unconstrained")
    pred_b2 = metrics_b2.pop("pred")
    aic_b2 = compute_aic(gt_uj, pred_b2, 3)

    # Summary table
    print(f"\n  {'Model':<32} {'#P':>3} {'MAPE':>7} {'rho':>7} "
          f"{'ws_rho':>7} {'AIC':>8}")
    print(f"  {'-'*32} {'-'*3} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    table_rows = [
        ("B1: 7-param (E-Improved-A)", 7, metrics_bl, aic_bl),
        ("B2: T-3 Unconstrained", 3, metrics_b2, aic_b2),
        ("E1: Unconstrained", 3, metrics_1a, aic_1a),
        ("E1: Constrained (>=100uJ)", 3, metrics_1b, aic_1b),
        ("E2: Unconstrained", 4, metrics_1c, aic_1c),
        ("E1: Oracle T (diag)", 3, metrics_1d_e1, float("nan")),
        ("E2: Oracle T (diag)", 4, metrics_1d_e2, float("nan")),
    ]
    for name, nparams, m, aic in table_rows:
        print(f"  {name:<32} {nparams:>3} {m['mape']:>6.1f}% {m['rho']:>7.4f} "
              f"{m['ws_rho']:>7.4f} {aic:>8.1f}")

    # Component contribution comparison
    print(f"\n  Component Contribution Comparison:")
    print(f"  {'Model':<32} {'P_sys*T':>8} {'P_core*P*T':>11} "
          f"{'E_core*P':>9} {'E_DMA*D':>8}")
    print(f"  {'-'*32} {'-'*8} {'-'*11} {'-'*9} {'-'*8}")

    comp_rows = [
        ("E1: Unconstrained", comp_1a),
        ("E1: Constrained", comp_1b),
        ("E2: Unconstrained", comp_1c),
        ("E1: Oracle T", comp_1d_e1),
        ("E2: Oracle T", comp_1d_e2),
    ]
    for name, comp in comp_rows:
        ps = comp.get("P_sys*T", 0)
        ppt = comp.get("P_core*P*T", 0)
        ep = comp.get("E_core*P", 0)
        ed = comp.get("E_DMA*D_total", 0)
        print(f"  {name:<32} {ps:>7.1f}% {ppt:>10.1f}% {ep:>8.1f}% {ed:>7.1f}%")

    # E_core / P_core identifiability summary
    print(f"\n  Identifiability Summary:")
    print(f"    E1 unconstrained: E_core = {calib_1a['e_core_uj']:.4f} uJ "
          f"({'IDENTIFIED' if e_core_identified else 'ZERO'})")
    print(f"    E2 unconstrained: P_core = {p_core_1c_mw:.2f} mW, "
          f"E_core = {e_core_1c:.4f} uJ")
    print(f"    E1 Oracle T:      E_core = {calib_1d_e1['e_core_uj']:.4f} uJ")
    print(f"    E2 Oracle T:      P_core = {calib_1d_e2['p_core_uw']/1e3:.2f} mW, "
          f"E_core = {calib_1d_e2['e_core_uj']:.4f} uJ")

    results["Phase2"] = {
        "baseline_7param": {
            "params": bl_params_calib,
            "metrics": metrics_bl,
            "aic": aic_bl,
        },
        "t3_unconstrained": {
            "params": calib_b2,
            "metrics": metrics_b2,
            "aic": aic_b2,
        },
    }

    # ================================================================
    # Phase 3: EDP comprehensive evaluation
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: EDP COMPREHENSIVE EVALUATION")
    print(f"{'='*70}")

    # Determine best candidate for EDP
    # If E1/E2 unconstrained has E_core/P_core > 0, use it;
    # otherwise fall back to B2 (T-3)
    if e_core_identified:
        best_name = "E1 Unconstrained"
        best_params = params_1a
        best_predict = predict_e1
    elif p_core_1c_mw > 1 or e_core_1c > 0.01:
        best_name = "E2 Unconstrained"
        best_params = params_1c
        best_predict = predict_e2
    else:
        best_name = "B2 (T-3)"
        best_params = params_b2
        best_predict = predict_compact

    print(f"  Best candidate for EDP: {best_name}")

    # EDP for best candidate
    edp_best = evaluate_edp(
        rows, feat, best_params, best_predict, coeffs, gt_uj)

    # EDP for baselines
    edp_bl = evaluate_edp(
        rows, feat, bl_optim, predict_te_baseline, coeffs, gt_uj)
    edp_b2 = evaluate_edp(
        rows, feat, params_b2, predict_compact, coeffs, gt_uj)

    # If best is not E1 or E2, also compute E1/E2 EDP for comparison
    edp_e1 = evaluate_edp(
        rows, feat, params_1a, predict_e1, coeffs, gt_uj)
    edp_e2 = evaluate_edp(
        rows, feat, params_1c, predict_e2, coeffs, gt_uj)

    print(f"\n  {'Metric':<22} {'B1(7p)':>10} {'B2(T-3)':>10} "
          f"{'E1':>10} {'E2':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for metric in ["core_accuracy", "regret_mean", "regret_max",
                    "regret_le10", "global_rho"]:
        vals = [edp_bl[metric], edp_b2[metric],
                edp_e1[metric], edp_e2[metric]]
        if isinstance(vals[0], float):
            print(f"  {metric:<22} "
                  + "".join(f"{v:>9.1f}%" for v in vals))
        else:
            print(f"  {metric:<22} "
                  + "".join(f"{str(v):>10}" for v in vals))

    # Per-size EDP details (E1 vs B2 comparison)
    print(f"\n  Per-size EDP Details (E1 vs B2):")
    print(f"  {'Size':<20} {'GT':>4} {'B2':>4} {'B2_Reg':>7} "
          f"{'E1':>4} {'E1_Reg':>7} {'E2':>4} {'E2_Reg':>7}")
    print(f"  {'-'*20} {'-'*4} {'-'*4} {'-'*7} "
          f"{'-'*4} {'-'*7} {'-'*4} {'-'*7}")

    for b2_d, e1_d, e2_d in zip(edp_b2["details"],
                                  edp_e1["details"],
                                  edp_e2["details"]):
        print(f"  {b2_d['size']:<20} {b2_d['gt_cores']:>4} "
              f"{b2_d['pred_cores']:>4} {b2_d['regret_pct']:>6.1f}% "
              f"{e1_d['pred_cores']:>4} {e1_d['regret_pct']:>6.1f}% "
              f"{e2_d['pred_cores']:>4} {e2_d['regret_pct']:>6.1f}%")

    # tpOrder accuracy (check if E1/E2 changes tpOrder selection)
    tp_match_b2, tp_match_e1, tp_match_e2 = 0, 0, 0
    tp_total_count = 0
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.M, r.K, r.N)].append(i)

    pred_t_us = feat["t_total_cy"] / CLOCK_MHZ
    gt_t_us = np.array([r.min_us for r in rows])

    pred_edp_b2 = pred_t_us * predict_compact(params_b2, feat)
    pred_edp_e1 = pred_t_us * predict_e1(params_1a, feat)
    pred_edp_e2 = pred_t_us * predict_e2(params_1c, feat)
    gt_edp = gt_t_us * gt_uj

    for key in sorted(groups.keys()):
        idx = groups[key]
        if len(idx) < 2:
            continue
        g = np.array(idx)

        # Check tpOrder of GT best vs predicted best
        gt_best = g[np.argmin(gt_edp[g])]
        b2_best = g[np.argmin(pred_edp_b2[g])]
        e1_best = g[np.argmin(pred_edp_e1[g])]
        e2_best = g[np.argmin(pred_edp_e2[g])]

        gt_tp = rows[gt_best].tp_order_inner
        if rows[b2_best].tp_order_inner == gt_tp:
            tp_match_b2 += 1
        if rows[e1_best].tp_order_inner == gt_tp:
            tp_match_e1 += 1
        if rows[e2_best].tp_order_inner == gt_tp:
            tp_match_e2 += 1
        tp_total_count += 1

    print(f"\n  tpOrder Selection Accuracy:")
    print(f"    B2 (T-3):  {tp_match_b2}/{tp_total_count}")
    print(f"    E1:        {tp_match_e1}/{tp_total_count}")
    print(f"    E2:        {tp_match_e2}/{tp_total_count}")

    results["Phase3"] = {
        "edp_b1": {k: v for k, v in edp_bl.items() if k != "details"},
        "edp_b2": {k: v for k, v in edp_b2.items() if k != "details"},
        "edp_e1": {k: v for k, v in edp_e1.items() if k != "details"},
        "edp_e2": {k: v for k, v in edp_e2.items() if k != "details"},
        "per_size_e1": edp_e1["details"],
        "per_size_e2": edp_e2["details"],
        "tporder_accuracy": {
            "b2": f"{tp_match_b2}/{tp_total_count}",
            "e1": f"{tp_match_e1}/{tp_total_count}",
            "e2": f"{tp_match_e2}/{tp_total_count}",
        },
    }

    # ================================================================
    # Final summary
    # ================================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY & RECOMMENDATION")
    print(f"{'='*70}")

    print(f"\n  Models Compared:")
    print(f"  {'Model':<32} {'#P':>3} {'MAPE':>7} {'AIC':>8} "
          f"{'EDP Acc':>8} {'Reg_mean':>9}")
    print(f"  {'-'*32} {'-'*3} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")

    summary_rows = [
        ("B1: 7-param (E-Improved-A)", 7, metrics_bl, aic_bl, edp_bl),
        ("B2: T-3 Unconstrained", 3, metrics_b2, aic_b2, edp_b2),
        ("E1: Unconstrained", 3, metrics_1a, aic_1a, edp_e1),
        ("E1: Constrained (>=100uJ)", 3, metrics_1b, aic_1b, None),
        ("E2: Unconstrained", 4, metrics_1c, aic_1c, edp_e2),
    ]
    for name, nparams, m, aic, edp in summary_rows:
        edp_acc = edp["core_accuracy"] if edp else "n/a"
        reg_mean = f"{edp['regret_mean']:.1f}%" if edp else "n/a"
        print(f"  {name:<32} {nparams:>3} {m['mape']:>6.1f}% {aic:>8.1f} "
              f"{str(edp_acc):>8} {reg_mean:>9}")

    # Key findings
    print(f"\n  Key Findings:")
    print(f"    1. E_core identifiability (E1 unconstrained): "
          f"{'YES' if e_core_identified else 'NO (converged to ~0)'}")
    e2_both = (p_core_1c_mw > 1) and (e_core_1c > 0.01)
    e2_e_only = (p_core_1c_mw <= 1) and (e_core_1c > 0.01)
    e2_p_only = (p_core_1c_mw > 1) and (e_core_1c <= 0.01)
    e2_neither = (p_core_1c_mw <= 1) and (e_core_1c <= 0.01)
    if e2_both:
        e2_verdict = "Both P_core and E_core survived"
    elif e2_e_only:
        e2_verdict = "Only E_core survived (fixed energy model preferred)"
    elif e2_p_only:
        e2_verdict = "Only P_core survived (time-proportional model preferred)"
    else:
        e2_verdict = "Neither survived (no per-core signal)"
    print(f"    2. E2 identifiability: {e2_verdict}")
    print(f"    3. Collinearity bypass: "
          f"VIF reduction from {results['Phase0']['vif']['T-3'].get('P*T', 0):.2f} "
          f"to {results['Phase0']['vif']['E1'].get('P', 0):.2f}")

    # ================================================================
    # Save results
    # ================================================================
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_results = json.loads(json.dumps(results, default=lambda o:
        round(float(o), 6) if isinstance(o, (np.floating, np.integer)) else
        o.tolist() if isinstance(o, np.ndarray) else str(o)))

    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
