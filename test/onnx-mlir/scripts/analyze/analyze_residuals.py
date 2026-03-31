#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_residuals.py — Residual analysis for v8 DMA-add performance model.

Computes prediction errors and correlates them with candidate features
to identify the best nonlinear correction for L_SYNC.

Outputs:
  1. Spearman rho of residuals vs each feature
  2. (TP_total bin) x (n_cores bin) heatmap of mean error %
  3. Top 10 over/under-predicted cases
  4. Assessment of P*TP_total cross-term significance for Model A
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import OpCase, TP_AXIS_K  # noqa: E402
from cost_model import Candidate, total_data_bytes  # noqa: E402
import models as _models  # noqa: E402

EFF_MACS = 24.28
BW_BPC = 4.0
CLOCK_MHZ = 1500


# ============================================================
# Data loading (same as analyze_alpha.py)
# ============================================================

def load_data(result_path: str, tc_path: str) -> list:
    """Load measurement data and merge with tpOrder."""
    import csv

    with open(tc_path) as f:
        tc_data = json.load(f)
    cases_list = tc_data["cases"] if isinstance(tc_data, dict) else tc_data

    tp_order_map = {}
    for tc in cases_list:
        lev = tc["levels"][0]
        key = (tc["M"], tc["K"], tc["N"],
               lev["SPm"], lev["SPn"],
               lev["TPm"], lev["TPk"], lev["TPn"])
        tp_order_map[key] = lev["tpOrder"][0]

    rows_raw = {}
    with open(result_path) as f:
        for row in csv.DictReader(f):
            if row["status"] != "PASS":
                continue
            gt_val = float(row["min_us"])
            if gt_val <= 0:
                continue
            config_key = (
                int(row["M"]), int(row["K"]), int(row["N"]),
                int(row["numSpm"]),
                int(row["SPm"]), int(row["SPn"]),
                int(row["TPm"]), int(row["TPk"]), int(row["TPn"]),
                int(row["TM"]), int(row["TK"]), int(row["TN"]),
            )
            if config_key not in rows_raw or gt_val < rows_raw[config_key][1]:
                rows_raw[config_key] = (row, gt_val)

    results = []
    for config_key, (row, gt_val) in rows_raw.items():
        M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
        SPm, SPn = int(row["SPm"]), int(row["SPn"])
        TPm, TPk, TPn = int(row["TPm"]), int(row["TPk"]), int(row["TPn"])
        TM, TK, TN = int(row["TM"]), int(row["TK"]), int(row["TN"])
        nc = int(row["numSpm"])
        key = (M, K, N, SPm, SPn, TPm, TPk, TPn)
        tp_order = tp_order_map.get(key, TP_AXIS_K)

        results.append({
            "M": M, "K": K, "N": N,
            "SPm": SPm, "SPn": SPn, "nc": nc,
            "TPm": TPm, "TPk": TPk, "TPn": TPn,
            "TM": TM, "TK": TK, "TN": TN,
            "tp_order": tp_order,
            "gt_us": gt_val,
            "gt_cy": gt_val * CLOCK_MHZ,
            "tp_total": TPm * TPk * TPn,
        })
    return results


def _dma_ops_per_step(d: dict) -> int:
    """Delegates to models.dma_ops_per_step()."""
    return _models.dma_ops_per_step(d["SPm"], d["SPn"], d["nc"], d["tp_order"])


def _total_data_bytes(d: dict) -> float:
    op = OpCase(M=d["M"], K=d["K"], N=d["N"], elem_type="bf16")
    cand = Candidate(
        num_cores=d["nc"], num_columns=(d["nc"] + 3) // 4,
        SPm=d["SPm"], SPn=d["SPn"],
        TPm=d["TPm"], TPk=d["TPk"], TPn=d["TPn"],
        TM=d["TM"], TK=d["TK"], TN=d["TN"],
    )
    return total_data_bytes(op, cand, d["tp_order"])


# ============================================================
# Prediction with v8 model
# ============================================================

def predict_v8(data: list, calib: dict) -> np.ndarray:
    """Compute performance predictions (in cycles) using models.py."""
    preds = np.zeros(len(data))
    for i, d in enumerate(data):
        macs = d["M"] * d["K"] * d["N"]
        data_bytes = _total_data_bytes(d)
        n_dma = _dma_ops_per_step(d)
        preds[i] = _models.PerfModel.predict_v9(
            macs=macs, data_bytes=data_bytes,
            n_cores=d["nc"], tp_total=d["tp_total"], n_dma=n_dma,
            eff_macs=EFF_MACS, bw_bpc=BW_BPC,
            l_sync=calib["l_sync_cy"],
            l_sync2=calib.get("l_sync2_cy", 0),
            l_dma=calib["l_dma_cy"],
            l_startup=calib["l_startup_cy"])
    return preds


# ============================================================
# Residual analysis
# ============================================================

def analyze_residuals(data: list, calib: dict):
    """Main residual analysis."""
    n = len(data)
    pred_cy = predict_v8(data, calib)
    gt_cy = np.array([d["gt_cy"] for d in data])

    # Signed residual (positive = overprediction)
    residual = pred_cy - gt_cy
    # Relative error (positive = overprediction)
    rel_error = residual / gt_cy * 100

    # Feature vectors
    tp_total = np.array([d["tp_total"] for d in data])
    n_cores = np.array([d["nc"] for d in data])
    p_x_tp = n_cores * tp_total
    macs = np.array([d["M"] * d["K"] * d["N"] for d in data], dtype=float)
    sp_ratio = np.array([max(d["SPm"], d["SPn"]) / max(min(d["SPm"], d["SPn"]), 1)
                         for d in data])
    n_dma = np.array([_dma_ops_per_step(d) for d in data], dtype=float)

    features = {
        "TP_total": tp_total,
        "P (cores)": n_cores,
        "P*TP_total": p_x_tp,
        "MACs": macs,
        "SP_ratio": sp_ratio,
        "N_dma": n_dma,
    }

    # 1. Spearman correlations
    print("\n--- 1. Residual Correlation with Features ---")
    print(f"  {'Feature':>15s}  {'rho(residual)':>14s}  {'rho(|residual|)':>16s}  Interpretation")
    correlations = {}
    for fname, fvec in features.items():
        rho_signed = spearman_rank_correlation(fvec.tolist(), residual.tolist())
        rho_abs = spearman_rank_correlation(fvec.tolist(),
                                            np.abs(residual).tolist())
        rho_signed = rho_signed if rho_signed is not None else 0.0
        rho_abs = rho_abs if rho_abs is not None else 0.0
        correlations[fname] = {"rho_signed": rho_signed, "rho_abs": rho_abs}

        # Interpret: negative rho_signed means feature increases -> overprediction decreases
        # (i.e., underprediction increases)
        if abs(rho_signed) < 0.1:
            interp = "no correlation"
        elif rho_signed < -0.3:
            interp = "STRONG: underpredicts as feature increases"
        elif rho_signed < -0.1:
            interp = "moderate: underpredicts as feature increases"
        elif rho_signed > 0.3:
            interp = "STRONG: overpredicts as feature increases"
        else:
            interp = "moderate: overpredicts as feature increases"

        print(f"  {fname:>15s}  {rho_signed:>14.4f}  {rho_abs:>16.4f}  {interp}")

    # 2. Heatmap: (TP_total bin) x (n_cores bin)
    print("\n--- 2. Mean Relative Error (%) Heatmap ---")
    print("  (Positive = overprediction, Negative = underprediction)")
    tp_bins = [(1, 4, "[1-4]"), (5, 16, "[5-16]"), (17, 64, "[17-64]"),
               (65, 256, "[65-256]"), (257, 100000, "[256+]")]
    core_bins = [(4, 4, "4c"), (8, 8, "8c"), (12, 12, "12c"),
                 (16, 16, "16c"), (20, 20, "20c"), (24, 32, "24-32c")]

    print(f"\n  {'TP\\\\Cores':>10s}", end="")
    for _, _, cb_name in core_bins:
        print(f"  {cb_name:>8s}", end="")
    print(f"  {'ALL':>8s}")

    heatmap = {}
    for tp_lo, tp_hi, tp_name in tp_bins:
        print(f"  {tp_name:>10s}", end="")
        row_errors = []
        for c_lo, c_hi, cb_name in core_bins:
            mask = ((tp_total >= tp_lo) & (tp_total <= tp_hi) &
                    (n_cores >= c_lo) & (n_cores <= c_hi))
            cell_errors = rel_error[mask]
            if len(cell_errors) > 0:
                mean_err = float(np.mean(cell_errors))
                heatmap[(tp_name, cb_name)] = {
                    "mean_err_pct": mean_err, "n": int(mask.sum())
                }
                print(f"  {mean_err:>7.1f}%", end="")
                row_errors.extend(cell_errors.tolist())
            else:
                print(f"  {'---':>8s}", end="")
        if row_errors:
            print(f"  {np.mean(row_errors):>7.1f}%")
        else:
            print(f"  {'---':>8s}")

    # Row: ALL cores
    print(f"  {'ALL':>10s}", end="")
    for c_lo, c_hi, cb_name in core_bins:
        mask = (n_cores >= c_lo) & (n_cores <= c_hi)
        if mask.sum() > 0:
            print(f"  {float(np.mean(rel_error[mask])):>7.1f}%", end="")
        else:
            print(f"  {'---':>8s}", end="")
    print(f"  {float(np.mean(rel_error)):>7.1f}%")

    # 3. Sample counts per cell
    print(f"\n  Sample counts:")
    print(f"  {'TP\\\\Cores':>10s}", end="")
    for _, _, cb_name in core_bins:
        print(f"  {cb_name:>8s}", end="")
    print()
    for tp_lo, tp_hi, tp_name in tp_bins:
        print(f"  {tp_name:>10s}", end="")
        for c_lo, c_hi, cb_name in core_bins:
            mask = ((tp_total >= tp_lo) & (tp_total <= tp_hi) &
                    (n_cores >= c_lo) & (n_cores <= c_hi))
            print(f"  {int(mask.sum()):>8d}", end="")
        print()

    # 4. TP_total bin MAPE (for v8 baseline reference)
    print("\n--- 3. v8 MAPE by TP_total Bin ---")
    print(f"  {'Bin':>10s}  {'N':>4s}  {'MAPE%':>7s}  {'MeanErr%':>9s}  {'MedianErr%':>11s}")
    for tp_lo, tp_hi, tp_name in tp_bins:
        mask = (tp_total >= tp_lo) & (tp_total <= tp_hi)
        if mask.sum() == 0:
            continue
        bin_mape = float(np.mean(np.abs(rel_error[mask])))
        bin_mean = float(np.mean(rel_error[mask]))
        bin_median = float(np.median(rel_error[mask]))
        print(f"  {tp_name:>10s}  {int(mask.sum()):>4d}  {bin_mape:>7.1f}  "
              f"{bin_mean:>9.1f}  {bin_median:>11.1f}")

    # 5. Top underpredicted and overpredicted
    print("\n--- 4. Top 10 Underpredicted Cases ---")
    print(f"  (Most negative relative error = largest underprediction)")
    sorted_idx = np.argsort(rel_error)
    print(f"  {'#':>3s}  {'Size':>20s}  {'P':>3s}  {'TP':>5s}  "
          f"{'Pred_us':>10s}  {'Actual_us':>10s}  {'Err%':>7s}")
    for rank, idx in enumerate(sorted_idx[:10]):
        d = data[idx]
        print(f"  {rank+1:>3d}  {d['M']}x{d['K']}x{d['N']:>5d}  "
              f"{d['nc']:>3d}  {d['tp_total']:>5d}  "
              f"{pred_cy[idx]/CLOCK_MHZ:>10.1f}  {d['gt_us']:>10.1f}  "
              f"{rel_error[idx]:>7.1f}")

    print("\n--- 5. Top 10 Overpredicted Cases ---")
    print(f"  {'#':>3s}  {'Size':>20s}  {'P':>3s}  {'TP':>5s}  "
          f"{'Pred_us':>10s}  {'Actual_us':>10s}  {'Err%':>7s}")
    for rank, idx in enumerate(sorted_idx[-10:][::-1]):
        d = data[idx]
        print(f"  {rank+1:>3d}  {d['M']}x{d['K']}x{d['N']:>5d}  "
              f"{d['nc']:>3d}  {d['tp_total']:>5d}  "
              f"{pred_cy[idx]/CLOCK_MHZ:>10.1f}  {d['gt_us']:>10.1f}  "
              f"{rel_error[idx]:>7.1f}")

    # 6. P*TP_total significance assessment
    print("\n--- 6. Model A (Core-Sync) Preliminary Assessment ---")
    rho_pxtp = correlations["P*TP_total"]["rho_signed"]
    rho_tp = correlations["TP_total"]["rho_signed"]
    rho_p = correlations["P (cores)"]["rho_signed"]
    print(f"  P*TP_total rho with residual: {rho_pxtp:.4f}")
    print(f"  TP_total rho:                 {rho_tp:.4f}")
    print(f"  P rho:                        {rho_p:.4f}")
    if abs(rho_pxtp) > abs(rho_tp) and abs(rho_pxtp) > 0.15:
        print(f"  -> P*TP_total has STRONGER correlation than TP_total alone.")
        print(f"  -> Model A (Core-Sync interaction) is well-motivated.")
    elif abs(rho_pxtp) > 0.1:
        print(f"  -> P*TP_total shows moderate correlation.")
        print(f"  -> Model A may provide marginal improvement.")
    else:
        print(f"  -> P*TP_total shows weak correlation.")
        print(f"  -> Model A may not significantly improve over v8.")


def main():
    parser = argparse.ArgumentParser(description="Residual analysis for v8 model")
    parser.add_argument("--result", required=True)
    parser.add_argument("--tc", required=True)
    parser.add_argument("--calib", required=True)
    args = parser.parse_args()

    with open(args.calib) as f:
        calib = json.load(f)

    print("Loading data...")
    data = load_data(args.result, args.tc)
    print(f"  {len(data)} unique configs loaded")

    print("\n" + "=" * 60)
    print("  Step 1-B: Residual Analysis")
    print("=" * 60)

    analyze_residuals(data, calib)


if __name__ == "__main__":
    main()
