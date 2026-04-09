#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task13_compact_energy.py -- Task 13: 3-Parameter Compact Energy Model Validation.

Validates a simplified 3-parameter energy model that consolidates the
7-parameter T-E model's structurally unidentifiable terms:

  E = P_sys * T + P_core * P * T + E_DMA * D_total

Phases:
  1A: Unconstrained fit (P_core free)
  1B: Constrained fit (P_core >= 50mW) if 1A yields P_core ~ 0
  1C: Oracle T diagnostic (measured T instead of predicted)
  2:  Comparison table vs 7-param baseline (E-Improved-A)
  3:  EDP comprehensive evaluation
  4A: 4-param variant (E_SYNC separated)
  4B: P_core fixed-value sweep

Usage:
    python3 scripts/analyze/task13_compact_energy.py \
        --result out/reports/result_v12.csv \
        --tc out/tc_list_v11.json \
        --calib data/calibration.json
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
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import (  # noqa: E402
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
    OpCase, load_calibration,
    CommentFilterFile,
)
from cost_model import Candidate, total_data_bytes  # noqa: E402
import models as _models  # noqa: E402

CLOCK_MHZ = 1500


# ============================================================
# Data loading (reuses task12_energy.py pattern)
# ============================================================

@dataclass
class EnergyRow:
    """Energy measurement row."""
    case_index: int
    M: int; K: int; N: int
    SPm: int; SPn: int
    TPm: int; TPk: int; TPn: int
    TM: int; TK: int; TN: int
    num_cores: int
    tp_order_inner: int
    min_us: float
    avg_iter_us: float
    npu_per_iter_uj: float
    wall_s: float

    @property
    def n_cores(self): return self.SPm * self.SPn
    @property
    def tp_total(self): return self.TPm * self.TPk * self.TPn
    @property
    def n_columns(self): return math.ceil(self.n_cores / 4)

    def make_op(self):
        return OpCase(M=self.M, K=self.K, N=self.N, elem_type="bf16")

    def make_cand(self):
        return Candidate(
            num_cores=self.n_cores, num_columns=self.n_columns,
            SPm=self.SPm, SPn=self.SPn,
            TPm=self.TPm, TPk=self.TPk, TPn=self.TPn,
            TM=self.TM, TK=self.TK, TN=self.TN,
        )


def load_energy_rows(csv_path, tc_path, gt_field="npu_energy_per_iter_uj",
                     min_wall_s=0.005):
    """Load v12 energy data."""
    with open(tc_path) as f:
        tc_data = json.load(f)
    cases_list = tc_data["cases"] if isinstance(tc_data, dict) else tc_data
    tp_order_map = {}
    for i, tc in enumerate(cases_list, start=1):
        tp_order_map[i] = tc["levels"][0].get("tpOrder", [2, 0, 1])[0]

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(CommentFilterFile(f)):
            if row["status"] != "PASS":
                continue
            gt_val = float(row.get(gt_field, 0) or 0)
            if gt_val <= 0:
                continue
            wall_s = float(row.get("wall_elapsed_s", -1))
            if wall_s <= 0:
                wall_s = float(row.get("batch_best_wall_s", -1))
            if wall_s < min_wall_s:
                continue

            case_idx = int(row["case_index"])
            rows.append(EnergyRow(
                case_index=case_idx,
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
                num_cores=int(row["numSpm"]),
                tp_order_inner=tp_order_map.get(case_idx, 2),
                min_us=float(row.get("min_us", 0)),
                avg_iter_us=float(row.get("avg_us", 0)),
                npu_per_iter_uj=gt_val,
                wall_s=wall_s,
            ))
    return rows


# ============================================================
# Feature computation
# ============================================================

def compute_features(rows, coeffs):
    """Compute features for all models (T-3, T-3S, T-E baseline)."""
    n = len(rows)
    macs = np.zeros(n)
    data_bytes = np.zeros(n)
    t_total_cy = np.zeros(n)
    t_meas_cy = np.zeros(n)
    n_cores_arr = np.zeros(n)
    tp_total_arr = np.zeros(n)
    ndma_tp = np.zeros(n)
    d_total_arr = np.zeros(n)
    gt_uj = np.zeros(n)

    for i, r in enumerate(rows):
        op = r.make_op()
        cand = r.make_cand()
        nc = r.n_cores
        m = r.M * r.K * r.N
        macs[i] = m
        db = total_data_bytes(op, cand, r.tp_order_inner)
        data_bytes[i] = db
        n_cores_arr[i] = nc
        tp_total_arr[i] = r.tp_total
        gt_uj[i] = r.npu_per_iter_uj
        t_meas_cy[i] = r.min_us * CLOCK_MHZ

        # DMA decomposition
        n_dma = _models.dma_ops_per_step(r.SPm, r.SPn, nc, r.tp_order_inner)
        ndma_tp[i] = n_dma * r.tp_total
        n_ev, n_re = _models.dma_ops_decomposed(
            r.SPm, r.SPn, nc, r.tp_order_inner)
        tp_inn = _models.tp_inner_value(r.TPm, r.TPk, r.TPn, r.tp_order_inner)
        d_total_arr[i] = n_ev * r.tp_total + n_re * (r.tp_total / tp_inn)

        # Predicted T (Candidate A, DMA-Refined from calibration.json v13)
        t_comp = m / (nc * coeffs.eff_macs)
        t_comm = db / coeffs.bw_eff_bpc
        t_ovh = (coeffs.l_sync_cy * r.tp_total
                 + coeffs.l_sync2_cy * nc * r.tp_total
                 + coeffs.l_dma_cy * d_total_arr[i]
                 + coeffs.l_startup_cy)
        t_total_cy[i] = t_comp + t_comm + t_ovh

    nt = n_cores_arr * t_total_cy
    nt_meas = n_cores_arr * t_meas_cy

    return {
        "macs": macs, "data_bytes": data_bytes,
        "t_total_cy": t_total_cy, "n_cores": n_cores_arr,
        "tp_total": tp_total_arr, "ndma_tp": ndma_tp,
        "nt": nt, "d_total": d_total_arr,
        "gt_uj": gt_uj,
        "t_meas_cy": t_meas_cy, "nt_meas": nt_meas,
    }


# ============================================================
# Prediction functions (optimization units: uJ)
# ============================================================

def predict_compact(params, feat):
    """T-3: E = P_sys*T + P_core*P*T + E_DMA*D_total (uJ)."""
    p_sys, p_core, e_dma = params
    return (p_sys * feat["t_total_cy"]
            + p_core * feat["nt"]
            + e_dma * feat["d_total"])


def predict_compact_oracle(params, feat):
    """T-3 with measured T (Oracle)."""
    p_sys, p_core, e_dma = params
    return (p_sys * feat["t_meas_cy"]
            + p_core * feat["nt_meas"]
            + e_dma * feat["d_total"])


def predict_compact_sync(params, feat):
    """T-3S: T-3 + E_SYNC*TP (4-param variant)."""
    p_sys, p_core, e_dma, e_sync = params
    return (p_sys * feat["t_total_cy"]
            + p_core * feat["nt"]
            + e_dma * feat["d_total"]
            + e_sync * feat["tp_total"])


def predict_compact_sync_oracle(params, feat):
    """T-3S with measured T (Oracle)."""
    p_sys, p_core, e_dma, e_sync = params
    return (p_sys * feat["t_meas_cy"]
            + p_core * feat["nt_meas"]
            + e_dma * feat["d_total"]
            + e_sync * feat["tp_total"])


def predict_compact_fixed_pcore(p_core_fixed, feat):
    """Return a prediction function with P_core fixed."""
    def predict(params, feat_inner):
        p_sys, e_dma = params
        return (p_sys * feat_inner["t_total_cy"]
                + p_core_fixed * feat_inner["nt"]
                + e_dma * feat_inner["d_total"])
    return predict


def predict_te_baseline(params, feat):
    """7-param T-E baseline (for comparison)."""
    e_mac, e_dram, e_dma, e_sync, p_base, p_core = params
    return (e_mac * feat["macs"]
            + e_dram * feat["data_bytes"]
            + e_dma * feat["d_total"]
            + e_sync * feat["tp_total"]
            + p_base * feat["t_total_cy"]
            + p_core * feat["nt"])


# ============================================================
# Fitting and evaluation
# ============================================================

def log_mse(gt, pred):
    """Log-space MSE objective."""
    return float(np.mean(
        (np.log(np.maximum(gt, 1e-10)) - np.log(np.maximum(pred, 1e-10))) ** 2))


def fit_model(feat, predict_fn, bounds, gt_uj):
    """Fit energy model with differential_evolution."""
    result = differential_evolution(
        lambda p: log_mse(gt_uj, predict_fn(p, feat)),
        bounds, seed=42, maxiter=3000, tol=1e-14, popsize=30, polish=True,
    )
    return result.x, result.fun


def evaluate_model(feat, predict_fn, params, gt_uj, rows, label=""):
    """Evaluate: MAPE, rho, ws_rho."""
    pred = predict_fn(params, feat)
    mape = float(np.mean(np.abs(gt_uj - pred) / gt_uj * 100))
    rho = spearman_rank_correlation(gt_uj.tolist(), pred.tolist())

    # Within-size Spearman rho
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.M, r.K, r.N)].append(i)

    tw, tr = 0.0, 0.0
    for key, idx in groups.items():
        if len(idx) < 3:
            continue
        g = np.array(idx)
        r_val = spearman_rank_correlation(gt_uj[g].tolist(), pred[g].tolist())
        if r_val is not None:
            tr += r_val * len(idx)
            tw += len(idx)
    ws_rho = tr / tw if tw > 0 else 0.0

    if label:
        print(f"  [{label}] MAPE={mape:.1f}%, rho={rho:.4f}, ws_rho={ws_rho:.4f}")

    return {"mape": mape, "rho": rho, "ws_rho": ws_rho, "pred": pred}


def component_analysis(feat, params, gt_uj, model_type="T-3"):
    """Analyze per-component energy contribution."""
    if model_type == "T-3":
        p_sys, p_core, e_dma = params
        e_sys = p_sys * feat["t_total_cy"]
        e_core = p_core * feat["nt"]
        e_dma_term = e_dma * feat["d_total"]
        total = e_sys + e_core + e_dma_term
        pct_sys = float(np.mean(e_sys / total * 100))
        pct_core = float(np.mean(e_core / total * 100))
        pct_dma = float(np.mean(e_dma_term / total * 100))
        return {"P_sys*T": pct_sys, "P_core*P*T": pct_core,
                "E_DMA*D_total": pct_dma}
    elif model_type == "T-3S":
        p_sys, p_core, e_dma, e_sync = params
        e_sys = p_sys * feat["t_total_cy"]
        e_core = p_core * feat["nt"]
        e_dma_term = e_dma * feat["d_total"]
        e_sync_term = e_sync * feat["tp_total"]
        total = e_sys + e_core + e_dma_term + e_sync_term
        pct_sys = float(np.mean(e_sys / total * 100))
        pct_core = float(np.mean(e_core / total * 100))
        pct_dma = float(np.mean(e_dma_term / total * 100))
        pct_sync = float(np.mean(e_sync_term / total * 100))
        return {"P_sys*T": pct_sys, "P_core*P*T": pct_core,
                "E_DMA*D_total": pct_dma, "E_SYNC*TP": pct_sync}
    return {}


def compute_aic(gt_uj, pred_uj, k):
    """AIC = n * ln(RSS/n) + 2*k."""
    n = len(gt_uj)
    rss = float(np.sum((np.log(gt_uj) - np.log(np.maximum(pred_uj, 1e-10))) ** 2))
    return n * math.log(rss / n) + 2 * k


def params_to_calib_t3(params):
    """Convert T-3 optimization params to calibration.json units."""
    p_sys, p_core, e_dma = params
    return {
        "p_sys_uw": round(p_sys * CLOCK_MHZ * 1e6, 1),
        "p_core_uw": round(p_core * CLOCK_MHZ * 1e6, 1),
        "e_dma_uj": round(e_dma, 6),
    }


def params_to_calib_t3s(params):
    """Convert T-3S optimization params to calibration.json units."""
    p_sys, p_core, e_dma, e_sync = params
    return {
        "p_sys_uw": round(p_sys * CLOCK_MHZ * 1e6, 1),
        "p_core_uw": round(p_core * CLOCK_MHZ * 1e6, 1),
        "e_dma_uj": round(e_dma, 6),
        "e_sync_uj": round(e_sync, 4),
    }


# ============================================================
# EDP evaluation
# ============================================================

def evaluate_edp(rows, feat, energy_params, energy_predict_fn,
                 coeffs, gt_uj):
    """Compute EDP core accuracy and regret."""
    pred_e_uj = energy_predict_fn(energy_params, feat)
    pred_t_us = feat["t_total_cy"] / CLOCK_MHZ
    pred_edp = pred_t_us * pred_e_uj

    gt_t_us = np.array([r.min_us for r in rows])
    gt_edp = gt_t_us * gt_uj

    # Global rho
    global_rho = spearman_rank_correlation(gt_edp.tolist(), pred_edp.tolist())

    # Group by problem size
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.M, r.K, r.N)].append(i)

    correct, total = 0, 0
    regrets = []
    details = []

    for key in sorted(groups.keys()):
        idx = groups[key]
        if len(idx) < 2:
            continue
        total += 1
        g = np.array(idx)

        gt_best_idx = g[np.argmin(gt_edp[g])]
        pred_best_idx = g[np.argmin(pred_edp[g])]

        gt_cores = rows[gt_best_idx].n_cores
        pred_cores = rows[pred_best_idx].n_cores

        if gt_cores == pred_cores:
            correct += 1

        regret = float((gt_edp[pred_best_idx] - gt_edp[gt_best_idx])
                       / gt_edp[gt_best_idx] * 100)
        regrets.append(max(0, regret))

        details.append({
            "size": f"{key[0]}x{key[1]}x{key[2]}",
            "gt_cores": gt_cores, "pred_cores": pred_cores,
            "regret_pct": round(regret, 1),
            "match": gt_cores == pred_cores,
        })

    le10 = sum(1 for r in regrets if r <= 10)
    return {
        "core_accuracy": f"{correct}/{total}",
        "regret_mean": float(np.mean(regrets)) if regrets else 0,
        "regret_max": float(np.max(regrets)) if regrets else 0,
        "regret_le10": f"{le10}/{total}",
        "global_rho": global_rho,
        "details": details,
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Task 13: 3-Parameter Compact Energy Model Validation")
    ap.add_argument("--result", required=True)
    ap.add_argument("--tc", required=True)
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--output", default="out/reports/task13_results.json")
    args = ap.parse_args()

    print("=" * 70)
    print("TASK 13: 3-PARAMETER COMPACT ENERGY MODEL VALIDATION")
    print("=" * 70)

    # Load data
    coeffs = load_calibration(Path(args.calib))
    rows = load_energy_rows(args.result, args.tc)
    n_samples = len(rows)
    print(f"  Loaded {n_samples} valid energy samples")

    feat = compute_features(rows, coeffs)
    gt_uj = feat["gt_uj"]

    results = {}

    # ----------------------------------------------------------------
    # Phase 1A: Unconstrained 3-param fit
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 1A: UNCONSTRAINED 3-PARAM FIT")
    print(f"{'='*70}")

    # Bounds: p_sys in uJ/cy, p_core in uJ/cy, e_dma in uJ
    # p_sys: 1W ~ 30W -> uJ/cy units: W / (CLOCK_MHZ * 1e6) = uJ/cy
    p_sys_lo = 1e6 / (CLOCK_MHZ * 1e6)        # 1 W
    p_sys_hi = 30e6 / (CLOCK_MHZ * 1e6)        # 30 W
    p_core_lo = 0.0                              # 0 mW (free)
    p_core_hi = 500e3 / (CLOCK_MHZ * 1e6)       # 500 mW
    e_dma_lo = 0.1                               # 0.1 uJ
    e_dma_hi = 100.0                             # 100 uJ

    bounds_t3 = [
        (p_sys_lo, p_sys_hi),
        (p_core_lo, p_core_hi),
        (e_dma_lo, e_dma_hi),
    ]

    params_1a, loss_1a = fit_model(feat, predict_compact, bounds_t3, gt_uj)
    calib_1a = params_to_calib_t3(params_1a)
    print(f"  Params (calib units): {calib_1a}")
    print(f"  LogMSE: {loss_1a:.6f}")

    metrics_1a = evaluate_model(
        feat, predict_compact, params_1a, gt_uj, rows, "T-3 Unconstrained")
    pred_1a = metrics_1a.pop("pred")

    comp_1a = component_analysis(feat, params_1a, gt_uj, "T-3")
    print(f"  Component contribution: {comp_1a}")

    aic_1a = compute_aic(gt_uj, pred_1a, 3)
    print(f"  AIC: {aic_1a:.1f}")

    results["Phase1A"] = {
        "model": "T-3",
        "variant": "unconstrained",
        "params": calib_1a,
        "metrics": metrics_1a,
        "loss": loss_1a,
        "aic": aic_1a,
        "components": comp_1a,
    }

    # ----------------------------------------------------------------
    # Phase 1B: P_core constrained (>= 50mW) if 1A yields ~0
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 1B: CONSTRAINED 3-PARAM FIT (P_core >= 50mW)")
    print(f"{'='*70}")

    p_core_1a_mw = calib_1a["p_core_uw"] / 1000
    print(f"  Phase 1A P_core = {p_core_1a_mw:.2f} mW")

    p_core_50mw = 50e3 / (CLOCK_MHZ * 1e6)  # 50 mW in uJ/cy
    bounds_t3_constrained = [
        (p_sys_lo, p_sys_hi),
        (p_core_50mw, p_core_hi),
        (e_dma_lo, e_dma_hi),
    ]

    params_1b, loss_1b = fit_model(
        feat, predict_compact, bounds_t3_constrained, gt_uj)
    calib_1b = params_to_calib_t3(params_1b)
    print(f"  Params (calib units): {calib_1b}")
    print(f"  LogMSE: {loss_1b:.6f}")

    metrics_1b = evaluate_model(
        feat, predict_compact, params_1b, gt_uj, rows, "T-3 Constrained")
    pred_1b = metrics_1b.pop("pred")

    comp_1b = component_analysis(feat, params_1b, gt_uj, "T-3")
    print(f"  Component contribution: {comp_1b}")

    aic_1b = compute_aic(gt_uj, pred_1b, 3)
    print(f"  AIC: {aic_1b:.1f}")
    print(f"  MAPE delta vs 1A: {metrics_1b['mape'] - metrics_1a['mape']:+.2f}%p")

    results["Phase1B"] = {
        "model": "T-3",
        "variant": "constrained_p_core_50mw",
        "params": calib_1b,
        "metrics": metrics_1b,
        "loss": loss_1b,
        "aic": aic_1b,
        "components": comp_1b,
    }

    # ----------------------------------------------------------------
    # Phase 1C: Oracle T diagnostic
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 1C: ORACLE T DIAGNOSTIC")
    print(f"{'='*70}")

    params_1c, loss_1c = fit_model(
        feat, predict_compact_oracle, bounds_t3, gt_uj)
    calib_1c = params_to_calib_t3(params_1c)
    print(f"  Params (calib units): {calib_1c}")
    print(f"  LogMSE: {loss_1c:.6f}")

    metrics_1c = evaluate_model(
        feat, predict_compact_oracle, params_1c, gt_uj, rows, "T-3 Oracle T")
    pred_1c = metrics_1c.pop("pred")

    comp_1c = component_analysis(feat, params_1c, gt_uj, "T-3")
    print(f"  Component contribution: {comp_1c}")
    print(f"  MAPE (Oracle T): {metrics_1c['mape']:.1f}% "
          f"-- energy structure limit")
    print(f"  Perf error propagation: "
          f"{metrics_1a['mape'] - metrics_1c['mape']:.1f}%p")

    results["Phase1C"] = {
        "model": "T-3",
        "variant": "oracle_t",
        "params": calib_1c,
        "metrics": metrics_1c,
        "loss": loss_1c,
        "components": comp_1c,
        "diagnostic": True,
    }

    # ----------------------------------------------------------------
    # Phase 2: Comparison with 7-param baseline
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 2: COMPARISON WITH 7-PARAM BASELINE")
    print(f"{'='*70}")

    # Load 7-param baseline from calibration.json
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

    # Convert to optim units for prediction
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
        "7-param Baseline (E-Improved-A)")
    pred_bl = metrics_bl.pop("pred")
    aic_bl = compute_aic(gt_uj, pred_bl, 7)

    # Summary table
    print(f"\n  {'Model':<28} {'#P':>3} {'MAPE':>7} {'rho':>7} "
          f"{'ws_rho':>7} {'AIC':>8} {'LogMSE':>8}")
    print(f"  {'-'*28} {'-'*3} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
    table = [
        ("Baseline (7-param E-Imp-A)", 7, metrics_bl, aic_bl, float("nan")),
        ("T-3 Unconstrained", 3, metrics_1a, aic_1a, loss_1a),
        ("T-3 Constrained (50mW)", 3, metrics_1b, aic_1b, loss_1b),
        ("T-3 Oracle T (diagnostic)", 3, metrics_1c, float("nan"), loss_1c),
    ]
    for name, nparams, m, aic, loss in table:
        print(f"  {name:<28} {nparams:>3} {m['mape']:>6.1f}% {m['rho']:>7.4f} "
              f"{m['ws_rho']:>7.4f} {aic:>8.1f} {loss:>8.6f}")

    results["Phase2"] = {
        "baseline_7param": {
            "params": bl_params_calib,
            "metrics": metrics_bl,
            "aic": aic_bl,
        },
        "comparison_note": (
            f"T-3 has {7-3}=4 fewer params. "
            f"MAPE delta: {metrics_1a['mape'] - metrics_bl['mape']:+.1f}%p (unconstrained), "
            f"{metrics_1b['mape'] - metrics_bl['mape']:+.1f}%p (constrained)"
        ),
    }

    # ----------------------------------------------------------------
    # Phase 3: EDP comprehensive evaluation
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 3: EDP COMPREHENSIVE EVALUATION")
    print(f"{'='*70}")

    # Use constrained model (Phase 1B) as primary, compare with baseline
    edp_compact = evaluate_edp(
        rows, feat, params_1b, predict_compact, coeffs, gt_uj)
    edp_baseline = evaluate_edp(
        rows, feat, bl_optim, predict_te_baseline, coeffs, gt_uj)

    print(f"\n  {'Metric':<22} {'Baseline (7p)':>14} {'Compact (3p)':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14}")
    for metric in ["core_accuracy", "regret_mean", "regret_max",
                    "regret_le10", "global_rho"]:
        bl_val = edp_baseline[metric]
        cp_val = edp_compact[metric]
        if isinstance(bl_val, float):
            print(f"  {metric:<22} {bl_val:>13.1f}% {cp_val:>13.1f}%")
        else:
            print(f"  {metric:<22} {str(bl_val):>14} {str(cp_val):>14}")

    # Per-size details
    print(f"\n  Per-size EDP details:")
    print(f"  {'Size':<20} {'BL_GT':>5} {'BL_P':>5} {'BL_Reg':>7} "
          f"{'CP_GT':>5} {'CP_P':>5} {'CP_Reg':>7}")
    for bl_d, cp_d in zip(edp_baseline["details"], edp_compact["details"]):
        bl_mark = "OK" if bl_d["match"] else "MISS"
        cp_mark = "OK" if cp_d["match"] else "MISS"
        print(f"  {bl_d['size']:<20} {bl_d['gt_cores']:>5} "
              f"{bl_d['pred_cores']:>4}{bl_mark:>3} {bl_d['regret_pct']:>6.1f}% "
              f"{cp_d['gt_cores']:>5} "
              f"{cp_d['pred_cores']:>4}{cp_mark:>3} {cp_d['regret_pct']:>6.1f}%")

    results["Phase3"] = {
        "compact_edp": {k: v for k, v in edp_compact.items() if k != "details"},
        "baseline_edp": {k: v for k, v in edp_baseline.items() if k != "details"},
        "per_size": edp_compact["details"],
    }

    # ----------------------------------------------------------------
    # Phase 4A: 4-param variant (E_SYNC separated)
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 4A: 4-PARAM VARIANT (E_SYNC SEPARATED)")
    print(f"{'='*70}")

    bounds_t3s = [
        (p_sys_lo, p_sys_hi),
        (p_core_lo, p_core_hi),
        (e_dma_lo, e_dma_hi),
        (0.1, 10000.0),           # e_sync uJ
    ]

    params_4a, loss_4a = fit_model(
        feat, predict_compact_sync, bounds_t3s, gt_uj)
    calib_4a = params_to_calib_t3s(params_4a)
    print(f"  Params (calib units): {calib_4a}")
    print(f"  LogMSE: {loss_4a:.6f}")

    metrics_4a = evaluate_model(
        feat, predict_compact_sync, params_4a, gt_uj, rows, "T-3S")
    pred_4a = metrics_4a.pop("pred")

    comp_4a = component_analysis(feat, params_4a, gt_uj, "T-3S")
    print(f"  Component contribution: {comp_4a}")

    aic_4a = compute_aic(gt_uj, pred_4a, 4)
    print(f"  AIC: {aic_4a:.1f}")
    print(f"  P_core = {calib_4a['p_core_uw']/1000:.2f} mW "
          f"({'identified' if calib_4a['p_core_uw'] > 1000 else 'still ~0'})")

    # 4A Oracle T
    params_4a_oracle, loss_4a_oracle = fit_model(
        feat, predict_compact_sync_oracle, bounds_t3s, gt_uj)
    calib_4a_oracle = params_to_calib_t3s(params_4a_oracle)
    metrics_4a_oracle = evaluate_model(
        feat, predict_compact_sync_oracle, params_4a_oracle, gt_uj, rows,
        "T-3S Oracle T")
    _ = metrics_4a_oracle.pop("pred")

    results["Phase4A"] = {
        "model": "T-3S",
        "params": calib_4a,
        "metrics": metrics_4a,
        "loss": loss_4a,
        "aic": aic_4a,
        "components": comp_4a,
        "oracle_t": {
            "params": calib_4a_oracle,
            "metrics": metrics_4a_oracle,
        },
    }

    # ----------------------------------------------------------------
    # Phase 4B: P_core fixed-value sweep
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 4B: P_core FIXED-VALUE SWEEP")
    print(f"{'='*70}")

    sweep_mw = [0, 25, 50, 75, 100, 150, 200]
    print(f"\n  {'P_core (mW)':>12} {'MAPE':>7} {'rho':>7} {'ws_rho':>7}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7}")

    sweep_results = []
    bounds_2param = [(p_sys_lo, p_sys_hi), (e_dma_lo, e_dma_hi)]

    for p_mw in sweep_mw:
        p_fixed = p_mw * 1e3 / (CLOCK_MHZ * 1e6)
        pred_fn = predict_compact_fixed_pcore(p_fixed, feat)
        params_sw, _ = fit_model(feat, pred_fn, bounds_2param, gt_uj)
        pred_sw = pred_fn(params_sw, feat)
        mape_sw = float(np.mean(np.abs(gt_uj - pred_sw) / gt_uj * 100))
        rho_sw = spearman_rank_correlation(gt_uj.tolist(), pred_sw.tolist())

        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[(r.M, r.K, r.N)].append(i)
        tw, tr = 0.0, 0.0
        for key, idx in groups.items():
            if len(idx) < 3:
                continue
            g = np.array(idx)
            r_val = spearman_rank_correlation(
                gt_uj[g].tolist(), pred_sw[g].tolist())
            if r_val is not None:
                tr += r_val * len(idx)
                tw += len(idx)
        ws_rho_sw = tr / tw if tw > 0 else 0.0

        p_sys_calib = round(params_sw[0] * CLOCK_MHZ * 1e6, 1)
        e_dma_calib = round(params_sw[1], 6)

        print(f"  {p_mw:>12} {mape_sw:>6.1f}% {rho_sw:>7.4f} {ws_rho_sw:>7.4f}"
              f"  [P_sys={p_sys_calib/1e6:.2f}W, E_DMA={e_dma_calib:.2f}uJ]")

        sweep_results.append({
            "p_core_mw": p_mw,
            "mape": round(mape_sw, 2),
            "rho": round(rho_sw, 4),
            "ws_rho": round(ws_rho_sw, 4),
            "p_sys_uw": p_sys_calib,
            "e_dma_uj": e_dma_calib,
        })

    results["Phase4B"] = {"sweep": sweep_results}

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Model':<28} {'#P':>3} {'MAPE':>7} {'rho':>7} "
          f"{'ws_rho':>7} {'AIC':>8}")
    print(f"  {'-'*28} {'-'*3} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    final_table = [
        ("Baseline (7p E-Imp-A)", 7, metrics_bl, aic_bl),
        ("T-3 Unconstrained", 3, metrics_1a, aic_1a),
        ("T-3 Constrained (50mW)", 3, metrics_1b, aic_1b),
        ("T-3S (4p, E_SYNC sep)", 4, metrics_4a, aic_4a),
        ("T-3 Oracle T", 3, metrics_1c, float("nan")),
        ("T-3S Oracle T", 3, metrics_4a_oracle, float("nan")),
    ]
    for name, nparams, m, aic in final_table:
        print(f"  {name:<28} {nparams:>3} {m['mape']:>6.1f}% {m['rho']:>7.4f} "
              f"{m['ws_rho']:>7.4f} {aic:>8.1f}")

    print(f"\n  EDP Comparison:")
    print(f"    Baseline:  Core Acc={edp_baseline['core_accuracy']}, "
          f"Regret mean={edp_baseline['regret_mean']:.1f}%, "
          f"max={edp_baseline['regret_max']:.1f}%")
    print(f"    Compact:   Core Acc={edp_compact['core_accuracy']}, "
          f"Regret mean={edp_compact['regret_mean']:.1f}%, "
          f"max={edp_compact['regret_max']:.1f}%")

    # Key finding: P_core identifiability
    p_core_1a_mw = calib_1a["p_core_uw"] / 1000
    p_core_4a_mw = calib_4a["p_core_uw"] / 1000
    print(f"\n  P_core Identifiability:")
    print(f"    T-3 unconstrained: {p_core_1a_mw:.2f} mW "
          f"({'converged to 0' if p_core_1a_mw < 1 else 'identified'})")
    print(f"    T-3S (E_SYNC sep): {p_core_4a_mw:.2f} mW "
          f"({'converged to 0' if p_core_4a_mw < 1 else 'identified'})")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove non-serializable numpy arrays
    clean_results = json.loads(json.dumps(results, default=lambda o:
        round(float(o), 6) if isinstance(o, (np.floating, np.integer)) else
        o.tolist() if isinstance(o, np.ndarray) else str(o)))

    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
