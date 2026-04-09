#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task12_energy.py -- Task 12 Phase 3: Energy model synchronization.

Applies Candidate A's DMA-Refined structure to the energy model:
  Baseline: E_DMA * N_dma * TP_total
  Improved: E_DMA * D_total  (where D_total uses tpOrder-dependent decomposition)

Also evaluates Oracle T variant (measured T instead of predicted) to separate
energy model structural limit from performance error propagation.

Usage:
    python3 scripts/analyze/task12_energy.py \
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
from cost_model import (  # noqa: E402
    Candidate, total_data_bytes,
)
import models as _models  # noqa: E402

import csv
import math
from dataclasses import dataclass

CLOCK_MHZ = 1500


# ============================================================
# Data loading (v12-compatible, bypasses calibrate_energy.load_energy_data
# which requires wall_elapsed_s > 0; v12 uses batch_best_wall_s)
# ============================================================

@dataclass
class EnergyRow:
    """Energy measurement row for Phase 3 fitting."""
    case_index: int
    M: int; K: int; N: int
    SPm: int; SPn: int
    TPm: int; TPk: int; TPn: int
    TM: int; TK: int; TN: int
    num_cores: int
    tp_order_inner: int
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


def load_energy_rows(
    csv_path: str, tc_path: str, gt_field: str = "npu_energy_per_iter_uj",
    min_wall_s: float = 0.005,
) -> List[EnergyRow]:
    """Load v12 energy data directly (compatible with batch_best_wall_s)."""
    # Load tpOrder from tc_list
    with open(tc_path) as f:
        tc_data = json.load(f)
    cases_list = tc_data["cases"] if isinstance(tc_data, dict) else tc_data
    tp_order_map: Dict[int, int] = {}
    for i, tc in enumerate(cases_list, start=1):
        tp_order = tc["levels"][0].get("tpOrder", [2, 0, 1])
        tp_order_map[i] = tp_order[0]

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(CommentFilterFile(f)):
            if row["status"] != "PASS":
                continue
            gt_val = float(row.get(gt_field, 0) or 0)
            if gt_val <= 0:
                continue

            # Use batch_best_wall_s if wall_elapsed_s is invalid
            wall_s = float(row.get("wall_elapsed_s", -1))
            if wall_s <= 0:
                wall_s = float(row.get("batch_best_wall_s", -1))
            if wall_s < min_wall_s:
                continue

            case_idx = int(row["case_index"])
            tp_inner = tp_order_map.get(case_idx, 2)

            rows.append(EnergyRow(
                case_index=case_idx,
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
                num_cores=int(row["numSpm"]),
                tp_order_inner=tp_inner,
                avg_iter_us=float(row.get("avg_us", 0)),
                npu_per_iter_uj=gt_val,
                wall_s=wall_s,
            ))

    return rows


# ============================================================
# Feature computation (includes D_total)
# ============================================================

def compute_energy_features(
    rows: List[EnergyRow], coeffs,
) -> Dict[str, np.ndarray]:
    """Compute all features needed for energy model fitting."""
    n = len(rows)
    macs = np.zeros(n)
    data_bytes = np.zeros(n)
    t_total_cy = np.zeros(n)
    n_cores_arr = np.zeros(n)
    tp_total_arr = np.zeros(n)
    ndma_tp = np.zeros(n)
    d_total = np.zeros(n)
    gt_uj = np.zeros(n)
    t_meas_cy = np.zeros(n)

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
        t_meas_cy[i] = r.avg_iter_us * CLOCK_MHZ

        # N_dma and DMA decomposition
        n_dma = _models.dma_ops_per_step(r.SPm, r.SPn, nc, r.tp_order_inner)
        ndma_tp[i] = n_dma * r.tp_total

        n_ev, n_re = _models.dma_ops_decomposed(
            r.SPm, r.SPn, nc, r.tp_order_inner)
        tp_inn = _models.tp_inner_value(r.TPm, r.TPk, r.TPn, r.tp_order_inner)
        d_total[i] = n_ev * r.tp_total + n_re * (r.tp_total / tp_inn)

        # Predicted T from calibrated perf model
        if coeffs and coeffs.calibrated:
            t_comp, t_comm, t_ovh = _models.PerfModel.components_v9(
                macs=m, data_bytes=db,
                n_cores=nc, tp_total=r.tp_total, n_dma=n_dma,
                eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
                l_sync=coeffs.l_sync_cy, l_sync2=coeffs.l_sync2_cy,
                l_dma=coeffs.l_dma_cy, l_startup=coeffs.l_startup_cy)
            t_total_cy[i] = t_comp + t_comm + t_ovh
        else:
            t_total_cy[i] = t_meas_cy[i]

    nt = n_cores_arr * t_total_cy
    nt_meas = n_cores_arr * t_meas_cy

    return {
        "macs": macs, "data_bytes": data_bytes,
        "t_total_cy": t_total_cy, "n_cores": n_cores_arr,
        "tp_total": tp_total_arr, "ndma_tp": ndma_tp,
        "nt": nt, "d_total": d_total,
        "gt_uj": gt_uj,
        "t_meas_cy": t_meas_cy, "nt_meas": nt_meas,
    }


# ============================================================
# Energy model prediction functions (optimization units: uJ)
# ============================================================

def predict_te_baseline(params, feat):
    """T-E Baseline: E_DMA * N_dma * TP (same as v12)."""
    e_mac, e_dram, e_dma, e_sync, p_base, p_core = params
    return (e_mac * feat["macs"]
            + e_dram * feat["data_bytes"]
            + e_dma * feat["ndma_tp"]
            + e_sync * feat["tp_total"]
            + p_base * feat["t_total_cy"]
            + p_core * feat["nt"])


def predict_te_dma_refined(params, feat):
    """T-E with D_total: E_DMA * D_total (Candidate A structure)."""
    e_mac, e_dram, e_dma, e_sync, p_base, p_core = params
    return (e_mac * feat["macs"]
            + e_dram * feat["data_bytes"]
            + e_dma * feat["d_total"]
            + e_sync * feat["tp_total"]
            + p_base * feat["t_total_cy"]
            + p_core * feat["nt"])


def predict_te_oracle(params, feat):
    """T-E with measured T (Oracle): diagnoses energy structure limit."""
    e_mac, e_dram, e_dma, e_sync, p_base, p_core = params
    return (e_mac * feat["macs"]
            + e_dram * feat["data_bytes"]
            + e_dma * feat["ndma_tp"]
            + e_sync * feat["tp_total"]
            + p_base * feat["t_meas_cy"]
            + p_core * feat["nt_meas"])


def predict_te_dma_refined_oracle(params, feat):
    """T-E DMA-Refined + Oracle T."""
    e_mac, e_dram, e_dma, e_sync, p_base, p_core = params
    return (e_mac * feat["macs"]
            + e_dram * feat["data_bytes"]
            + e_dma * feat["d_total"]
            + e_sync * feat["tp_total"]
            + p_base * feat["t_meas_cy"]
            + p_core * feat["nt_meas"])


# ============================================================
# Fitting
# ============================================================

def log_mse(gt, pred):
    """Log-space MSE."""
    return float(np.mean(
        (np.log(np.maximum(gt, 1e-10)) - np.log(np.maximum(pred, 1e-10))) ** 2))


def fit_energy_model(feat, predict_fn, bounds, gt_uj):
    """Fit energy model with differential_evolution."""
    result = differential_evolution(
        lambda p: log_mse(gt_uj, predict_fn(p, feat)),
        bounds, seed=42, maxiter=3000, tol=1e-14, popsize=30, polish=True,
    )
    return result.x, result.fun


def evaluate_energy(feat, predict_fn, params, gt_uj, rows, label=""):
    """Evaluate energy model: MAPE, rho, ws_rho."""
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
        r_val = spearman_rank_correlation(
            gt_uj[g].tolist(), pred[g].tolist())
        if r_val is not None:
            tr += r_val * len(idx)
            tw += len(idx)
    ws_rho = tr / tw if tw > 0 else 0.0

    if label:
        print(f"  [{label}] MAPE={mape:.1f}%, rho={rho:.4f}, ws_rho={ws_rho:.4f}")

    return {"mape": mape, "rho": rho, "ws_rho": ws_rho, "pred": pred}


def params_to_calib(params):
    """Convert optimization-unit params to calibration.json units."""
    e_mac, e_dram, e_dma, e_sync, p_base, p_core = params
    return {
        "e_mac_pj": round(e_mac * 1e6, 4),
        "e_dram_pj": round(e_dram * 1e6, 4),
        "e_dma_uj": round(e_dma, 6),
        "e_sync_uj": round(e_sync, 4),
        "p_base_uw": round(p_base * CLOCK_MHZ * 1e6, 1),
        "p_core_uw": round(p_core * CLOCK_MHZ * 1e6, 1),
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Task 12 Phase 3: Energy model synchronization")
    ap.add_argument("--result", required=True)
    ap.add_argument("--tc", required=True)
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--gt-mode", default="npu",
                    help="Energy GT mode: npu, active_pkg, corrected")
    ap.add_argument("--min-wall-s", type=float, default=0.005)
    args = ap.parse_args()

    print("=" * 70)
    print("PHASE 3: ENERGY MODEL SYNCHRONIZATION")
    print("=" * 70)

    # Load data
    coeffs = load_calibration(Path(args.calib))
    rows = load_energy_rows(
        args.result, args.tc,
        gt_field="npu_energy_per_iter_uj",
        min_wall_s=args.min_wall_s)
    print(f"  Loaded {len(rows)} valid energy samples")

    ext_feat = compute_energy_features(rows, coeffs)
    gt_uj = ext_feat["gt_uj"]

    # v12 T-E constrained bounds (10% of v9 values as lower bound)
    # v9: e_mac=20.46e-6, e_dram=727.5e-6, e_dma=0.447, e_sync=89.4
    # p_base=1.28e6/(1500*1e6), p_core=112.6e3/(1500*1e6)
    v9_e_mac = 20.4634e-6   # uJ per MAC
    v9_e_dram = 727.4622e-6  # uJ per byte
    v9_e_dma = 0.446966      # uJ per DMA*TP
    v9_e_sync = 89.4143      # uJ per TP
    bounds_te = [
        (v9_e_mac * 0.1, 1e-3),    # e_mac
        (v9_e_dram * 0.1, 1e-1),   # e_dram
        (v9_e_dma * 0.1, 1e3),     # e_dma
        (v9_e_sync * 0.1, 1e4),    # e_sync
        (1e-18, 1e0),              # p_base
        (1e-18, 1e0),              # p_core
    ]

    results = {}

    # --- E-Baseline: T-E with N_dma*TP (v12 structure) ---
    print("\n--- E-Baseline: T-E with N_dma*TP (v12 structure) ---")
    params_bl, loss_bl = fit_energy_model(
        ext_feat, predict_te_baseline, bounds_te, gt_uj)
    calib_bl = params_to_calib(params_bl)
    print(f"  Params: {calib_bl}")
    print(f"  LogMSE: {loss_bl:.6f}")
    metrics_bl = evaluate_energy(
        ext_feat, predict_te_baseline, params_bl, gt_uj, rows, "E-Baseline")
    results["E-Baseline"] = {
        "params": calib_bl, "metrics": metrics_bl, "loss": loss_bl,
    }

    # --- E-Improved: T-E with D_total (Candidate A structure) ---
    print("\n--- E-Improved: T-E with D_total (Candidate A structure) ---")
    params_imp, loss_imp = fit_energy_model(
        ext_feat, predict_te_dma_refined, bounds_te, gt_uj)
    calib_imp = params_to_calib(params_imp)
    print(f"  Params: {calib_imp}")
    print(f"  LogMSE: {loss_imp:.6f}")
    metrics_imp = evaluate_energy(
        ext_feat, predict_te_dma_refined, params_imp, gt_uj, rows, "E-Improved")
    results["E-Improved"] = {
        "params": calib_imp, "metrics": metrics_imp, "loss": loss_imp,
    }

    # --- E-Baseline + Oracle T ---
    print("\n--- E-Baseline + Oracle T (diagnostic) ---")
    params_oracle, loss_oracle = fit_energy_model(
        ext_feat, predict_te_oracle, bounds_te, gt_uj)
    calib_oracle = params_to_calib(params_oracle)
    print(f"  Params: {calib_oracle}")
    metrics_oracle = evaluate_energy(
        ext_feat, predict_te_oracle, params_oracle, gt_uj, rows,
        "E-Baseline+OracleT")
    results["E-Baseline+OracleT"] = {
        "params": calib_oracle, "metrics": metrics_oracle,
        "diagnostic": True,
    }

    # --- E-Improved + Oracle T ---
    print("\n--- E-Improved + Oracle T (diagnostic) ---")
    params_imp_oracle, loss_imp_oracle = fit_energy_model(
        ext_feat, predict_te_dma_refined_oracle, bounds_te, gt_uj)
    calib_imp_oracle = params_to_calib(params_imp_oracle)
    print(f"  Params: {calib_imp_oracle}")
    metrics_imp_oracle = evaluate_energy(
        ext_feat, predict_te_dma_refined_oracle, params_imp_oracle, gt_uj,
        rows, "E-Improved+OracleT")
    results["E-Improved+OracleT"] = {
        "params": calib_imp_oracle, "metrics": metrics_imp_oracle,
        "diagnostic": True,
    }

    # Summary table
    print(f"\n{'='*70}")
    print("PHASE 3 SUMMARY: ENERGY MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'MAPE':>7} {'rho':>7} {'ws_rho':>7} {'LogMSE':>10}")
    print(f"{'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")
    for name, r in results.items():
        m = r["metrics"]
        loss = r.get("loss", float("nan"))
        print(f"{name:<25} {m['mape']:>6.1f}% {m['rho']:>7.4f} "
              f"{m['ws_rho']:>7.4f} {loss:>10.6f}")

    # Coefficient comparison
    print(f"\n  Coefficient Comparison (E-Baseline vs E-Improved):")
    for key in ["e_mac_pj", "e_dram_pj", "e_dma_uj", "e_sync_uj",
                "p_base_uw", "p_core_uw"]:
        bl_val = results["E-Baseline"]["params"][key]
        imp_val = results["E-Improved"]["params"][key]
        ratio = imp_val / bl_val if bl_val != 0 else float("inf")
        print(f"    {key:<12}: Baseline={bl_val:>12.4f}  "
              f"Improved={imp_val:>12.4f}  ratio={ratio:.3f}")

    # Save results
    output = {}
    for name, r in results.items():
        output[name] = {
            "params": r["params"],
            "mape": r["metrics"]["mape"],
            "rho": r["metrics"]["rho"],
            "ws_rho": r["metrics"]["ws_rho"],
        }

    out_path = Path("out/reports/task12_phase3_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Phase 3 results saved to {out_path}")


if __name__ == "__main__":
    main()
