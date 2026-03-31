#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_alpha.py — Alpha interpretability and contribution ratio analysis.

Analyzes the meaning of alpha=0.01 in the v8 DMA-add model:
  T_total = alpha*T_comp + beta*T_comm + T_overhead

Outputs:
  1. Contribution ratios (alpha*T_comp / T_total, etc.) by TP_total bin
  2. alpha=1.0 vs alpha=0.01 accuracy comparison
  3. Interpretation for paper presentation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import OpCase, TP_AXIS_K  # noqa: E402
from cost_model import Candidate, total_data_bytes  # noqa: E402
import models as _models  # noqa: E402

# Fixed constants (from v8 calibration)
EFF_MACS = 24.28
BW_BPC = 4.0
CLOCK_MHZ = 1500


# ============================================================
# Data loading (reuse recalibrate_model pattern)
# ============================================================

def load_data(result_path: str, tc_path: str) -> list:
    """Load measurement data and merge with tpOrder from tc_list."""
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

    # Deduplicate: keep min per unique config
    raw: Dict[tuple, List[float]] = defaultdict(list)
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
            raw[config_key].append(gt_val)
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
    """Compute total data bytes for a case."""
    op = OpCase(M=d["M"], K=d["K"], N=d["N"], elem_type="bf16")
    cand = Candidate(
        num_cores=d["nc"], num_columns=(d["nc"] + 3) // 4,
        SPm=d["SPm"], SPn=d["SPn"],
        TPm=d["TPm"], TPk=d["TPk"], TPn=d["TPn"],
        TM=d["TM"], TK=d["TK"], TN=d["TN"],
    )
    return total_data_bytes(op, cand, d["tp_order"])


# ============================================================
# Contribution analysis
# ============================================================

def analyze_contributions(data: list, calib: dict) -> dict:
    """Compute contribution ratios for each component."""
    # TP_total bins
    bins = {
        "[1-4]": (1, 4),
        "[5-16]": (5, 16),
        "[17-64]": (17, 64),
        "[65-256]": (65, 256),
        "[256+]": (257, 1e9),
    }

    bin_contribs = {b: {"comp": [], "comm": [], "overhead": [], "n": 0}
                    for b in bins}
    all_contribs = {"comp": [], "comm": [], "overhead": []}

    for d in data:
        macs = d["M"] * d["K"] * d["N"]
        data_bytes = _total_data_bytes(d)
        n_dma = _dma_ops_per_step(d)
        t_comp, t_comm, t_overhead = _models.PerfModel.components_v9(
            macs=macs, data_bytes=data_bytes,
            n_cores=d["nc"], tp_total=d["tp_total"], n_dma=n_dma,
            eff_macs=EFF_MACS, bw_bpc=BW_BPC,
            l_sync=calib["l_sync_cy"],
            l_sync2=calib.get("l_sync2_cy", 0),
            l_dma=calib["l_dma_cy"],
            l_startup=calib["l_startup_cy"])

        t_total = t_comp + t_comm + t_overhead
        if t_total <= 0:
            continue

        comp_pct = t_comp / t_total * 100
        comm_pct = t_comm / t_total * 100
        ovhd_pct = t_overhead / t_total * 100

        all_contribs["comp"].append(comp_pct)
        all_contribs["comm"].append(comm_pct)
        all_contribs["overhead"].append(ovhd_pct)

        tp = d["tp_total"]
        for bname, (lo, hi) in bins.items():
            if lo <= tp <= hi:
                bin_contribs[bname]["comp"].append(comp_pct)
                bin_contribs[bname]["comm"].append(comm_pct)
                bin_contribs[bname]["overhead"].append(ovhd_pct)
                bin_contribs[bname]["n"] += 1
                break

    # Summarize
    summary = {"overall": {}, "by_tp_bin": {}}
    for comp in ["comp", "comm", "overhead"]:
        arr = all_contribs[comp]
        summary["overall"][comp] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    for bname in bins:
        bc = bin_contribs[bname]
        if bc["n"] == 0:
            continue
        summary["by_tp_bin"][bname] = {
            "n": bc["n"],
            "comp_mean": float(np.mean(bc["comp"])),
            "comm_mean": float(np.mean(bc["comm"])),
            "overhead_mean": float(np.mean(bc["overhead"])),
        }

    return summary


# ============================================================
# Alpha=1.0 comparison
# ============================================================

def compare_alpha_fixed(data: list, calib: dict) -> dict:
    """Compare v8 (alpha=0.01) vs alpha=1.0 fixed (refit other params)."""
    n = len(data)
    t_comp = np.zeros(n)
    t_comm = np.zeros(n)
    tp_total = np.zeros(n)
    n_cores = np.zeros(n)
    total_dma_ops = np.zeros(n)
    gt_cy = np.zeros(n)

    for i, d in enumerate(data):
        t_comp[i] = (d["M"] * d["K"] * d["N"]) / (d["nc"] * EFF_MACS)
        t_comm[i] = _total_data_bytes(d) / BW_BPC
        tp_total[i] = d["tp_total"]
        n_cores[i] = d["nc"]
        total_dma_ops[i] = _dma_ops_per_step(d) * d["tp_total"]
        gt_cy[i] = d["gt_cy"]

    feat = {
        "t_comp": t_comp, "t_comm": t_comm,
        "tp_total": tp_total, "n_cores": n_cores,
        "total_dma_ops": total_dma_ops, "gt_cy": gt_cy,
    }

    def predict_free(f, p):
        """DMA-add with free alpha."""
        a, b, ls, ld, lc, lst = p
        return (a * f["t_comp"] + b * f["t_comm"]
                + ls * f["tp_total"] + ld * f["total_dma_ops"]
                + lc * f["n_cores"] + lst)

    def predict_fixed_alpha1(f, p):
        """DMA-add with alpha=1.0 fixed."""
        b, ls, ld, lc, lst = p
        return (1.0 * f["t_comp"] + b * f["t_comm"]
                + ls * f["tp_total"] + ld * f["total_dma_ops"]
                + lc * f["n_cores"] + lst)

    def log_mse(gt, pred):
        return float(np.mean((np.log(np.maximum(gt, 1)) -
                              np.log(np.maximum(pred, 1))) ** 2))

    def mape(gt, pred):
        return float(np.mean(np.abs(gt - pred) / gt * 100))

    # Fit free alpha (should reproduce v8)
    bounds_free = [(0.001, 10), (0.01, 10), (0, 5e5), (0, 5e5), (0, 5e5), (0, 5e5)]
    res_free = differential_evolution(
        lambda p: log_mse(gt_cy, predict_free(feat, p)),
        bounds_free, seed=42, maxiter=2000, tol=1e-14, polish=True,
    )
    pred_free = predict_free(feat, res_free.x)

    # Fit alpha=1.0 fixed
    bounds_fixed = [(0.01, 10), (0, 5e5), (0, 5e5), (0, 5e5), (0, 5e5)]
    res_fixed = differential_evolution(
        lambda p: log_mse(gt_cy, predict_fixed_alpha1(feat, p)),
        bounds_fixed, seed=42, maxiter=2000, tol=1e-14, polish=True,
    )
    pred_fixed = predict_fixed_alpha1(feat, res_fixed.x)

    gt_list = gt_cy.tolist()
    return {
        "alpha_free": {
            "alpha": round(float(res_free.x[0]), 4),
            "beta": round(float(res_free.x[1]), 4),
            "mape": round(mape(gt_cy, pred_free), 1),
            "rho": round(spearman_rank_correlation(
                pred_free.tolist(), gt_list), 4),
            "params": [round(float(v), 1) for v in res_free.x],
        },
        "alpha_1.0": {
            "alpha": 1.0,
            "beta": round(float(res_fixed.x[0]), 4),
            "mape": round(mape(gt_cy, pred_fixed), 1),
            "rho": round(spearman_rank_correlation(
                pred_fixed.tolist(), gt_list), 4),
            "params": [round(float(v), 1) for v in res_fixed.x],
        },
    }


# ============================================================
# T_comp vs T_comm magnitude analysis
# ============================================================

def analyze_magnitude(data: list) -> dict:
    """Analyze raw T_comp vs T_comm magnitudes (before alpha/beta scaling)."""
    ratios = []
    for d in data:
        t_comp = (d["M"] * d["K"] * d["N"]) / (d["nc"] * EFF_MACS)
        t_comm = _total_data_bytes(d) / BW_BPC
        if t_comm > 0:
            ratios.append(t_comp / t_comm)

    arr = np.array(ratios)
    return {
        "t_comp_over_t_comm": {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pct_below_0.1": round(float(np.mean(arr < 0.1) * 100), 1),
            "pct_below_1.0": round(float(np.mean(arr < 1.0) * 100), 1),
        },
        "interpretation": (
            "T_comp/T_comm < 1.0 means communication dominates compute. "
            "With eff_macs=24.28 (9.5% of peak 256), compute is inherently "
            "small relative to DMA transfer time."
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Alpha interpretability analysis")
    parser.add_argument("--result", required=True, help="result CSV path")
    parser.add_argument("--tc", required=True, help="tc_list JSON path")
    parser.add_argument("--calib", required=True, help="calibration JSON path")
    args = parser.parse_args()

    with open(args.calib) as f:
        calib = json.load(f)

    print("Loading data...")
    data = load_data(args.result, args.tc)
    print(f"  {len(data)} unique configs loaded")

    print("\n" + "=" * 60)
    print("  Step 1-A: Alpha Interpretability Analysis")
    print("=" * 60)

    # 1. Contribution ratios
    print("\n--- 1. Contribution Ratios ---")
    contribs = analyze_contributions(data, calib)

    print(f"\nOverall (N={len(data)}):")
    for comp in ["comp", "comm", "overhead"]:
        s = contribs["overall"][comp]
        print(f"  {comp:>10s}: mean={s['mean']:5.1f}%  median={s['median']:5.1f}%  "
              f"range=[{s['min']:.1f}%, {s['max']:.1f}%]")

    print(f"\nBy TP_total bin:")
    print(f"  {'Bin':>10s}  {'N':>4s}  {'Comp%':>7s}  {'Comm%':>7s}  {'Ovhd%':>7s}")
    for bname, bs in contribs["by_tp_bin"].items():
        print(f"  {bname:>10s}  {bs['n']:>4d}  {bs['comp_mean']:>7.1f}  "
              f"{bs['comm_mean']:>7.1f}  {bs['overhead_mean']:>7.1f}")

    # 2. T_comp vs T_comm magnitude
    print("\n--- 2. Raw T_comp / T_comm Ratio (before alpha/beta) ---")
    mag = analyze_magnitude(data)
    m = mag["t_comp_over_t_comm"]
    print(f"  Mean:   {m['mean']:.3f}")
    print(f"  Median: {m['median']:.3f}")
    print(f"  Range:  [{m['min']:.3f}, {m['max']:.3f}]")
    print(f"  Cases with T_comp/T_comm < 0.1: {m['pct_below_0.1']}%")
    print(f"  Cases with T_comp/T_comm < 1.0: {m['pct_below_1.0']}%")

    # 3. alpha=1.0 vs alpha=0.01 comparison
    print("\n--- 3. Alpha=1.0 vs Alpha=free Comparison ---")
    print("  (Fitting with differential_evolution, may take ~30s...)")
    alpha_cmp = compare_alpha_fixed(data, calib)

    print(f"\n  Alpha free:  alpha={alpha_cmp['alpha_free']['alpha']}, "
          f"MAPE={alpha_cmp['alpha_free']['mape']}%, "
          f"rho={alpha_cmp['alpha_free']['rho']}")
    print(f"  Alpha=1.0:   alpha=1.0, "
          f"MAPE={alpha_cmp['alpha_1.0']['mape']}%, "
          f"rho={alpha_cmp['alpha_1.0']['rho']}")

    delta_mape = alpha_cmp["alpha_1.0"]["mape"] - alpha_cmp["alpha_free"]["mape"]
    print(f"\n  MAPE difference: {delta_mape:+.1f} pp "
          f"({'worse' if delta_mape > 0 else 'better'} with alpha=1.0)")

    # 4. Interpretation
    print("\n--- 4. Paper Interpretation ---")
    comp_mean = contribs["overall"]["comp"]["mean"]
    comm_mean = contribs["overall"]["comm"]["mean"]
    ovhd_mean = contribs["overall"]["overhead"]["mean"]
    print(f"  alpha=0.01 means T_comp contributes only {comp_mean:.1f}% "
          f"of total predicted time.")
    print(f"  T_comm contributes {comm_mean:.1f}%, overhead {ovhd_mean:.1f}%.")
    print(f"  Root cause: eff_macs=24.28 is 9.5% of peak (256 MACs/cy).")
    print(f"  At this utilization, compute completes much faster than DMA transfer,")
    print(f"  making T_comp negligible relative to T_comm + T_overhead.")
    if abs(delta_mape) < 1.0:
        print(f"  Since alpha=1.0 vs alpha=free gives only {abs(delta_mape):.1f} pp "
              f"MAPE difference,")
        print(f"  alpha's exact value is statistically insignificant.")
        print(f"  Paper suggestion: report alpha as fitted but note T_comp's minimal role.")
    else:
        print(f"  Alpha=1.0 degrades MAPE by {delta_mape:.1f} pp, so alpha "
              f"has non-trivial effect despite small contribution.")


if __name__ == "__main__":
    main()
