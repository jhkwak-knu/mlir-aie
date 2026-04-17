#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task12_edp.py -- Task 12 Phase 4: EDP comprehensive evaluation.

Computes EDP = T * E for Baseline vs Improved models,
evaluates core-count selection accuracy, regret, and tpOrder accuracy.

Usage:
    python3 scripts/analyze/task12_edp.py \
        --result out/reports/result_v12.csv \
        --tc out/tc_list_v11.json \
        --calib data/calibration.json \
        [--plot-dir out/plots/task12_phase4]
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

CLOCK_MHZ = 1500


# ============================================================
# Data loading
# ============================================================

@dataclass
class EdpCase:
    """One case with measured T, E, and configuration."""
    op: OpCase
    cand: Candidate
    tp_order: int
    gt_us: float        # measured T (min_us)
    gt_energy_uj: float  # measured E (npu_energy_per_iter_uj)


def load_edp_data(result_path, tc_path):
    """Load data for EDP evaluation."""
    with open(tc_path) as f:
        tc_data = json.load(f)
    cases_list = tc_data["cases"] if isinstance(tc_data, dict) else tc_data
    tp_order_map = {}
    for i, tc in enumerate(cases_list, start=1):
        lev = tc["levels"][0]
        key = (tc["M"], tc["K"], tc["N"],
               lev["SPm"], lev["SPn"],
               lev["TPm"], lev["TPk"], lev["TPn"])
        tp_order_map[key] = lev["tpOrder"][0]

    raw = defaultdict(list)
    with open(result_path) as f:
        for row in csv.DictReader(CommentFilterFile(f)):
            if row["status"] != "PASS":
                continue
            t_us = float(row.get("min_us", 0))
            e_uj = float(row.get("npu_energy_per_iter_uj", 0) or 0)
            if t_us <= 0 or e_uj <= 0:
                continue
            config_key = (
                int(row["M"]), int(row["K"]), int(row["N"]),
                int(row["numSpm"]),
                int(row["SPm"]), int(row["SPn"]),
                int(row["TPm"]), int(row["TPk"]), int(row["TPn"]),
                int(row["TM"]), int(row["TK"]), int(row["TN"]),
            )
            raw[config_key].append((t_us, e_uj))

    results = []
    for (M, K, N, nc, SPm, SPn, TPm, TPk, TPn, TM, TK, TN), vals in raw.items():
        # Keep min time entry (consistent with perf model GT)
        best = min(vals, key=lambda x: x[0])
        t_us, e_uj = best
        key = (M, K, N, SPm, SPn, TPm, TPk, TPn)
        tp_order = tp_order_map.get(key, TP_AXIS_K)
        op = OpCase(M=M, K=K, N=N, elem_type="bf16")
        cand = Candidate(
            num_cores=nc, num_columns=(nc + 3) // 4,
            SPm=SPm, SPn=SPn, TPm=TPm, TPk=TPk, TPn=TPn,
            TM=TM, TK=TK, TN=TN)
        results.append(EdpCase(op=op, cand=cand, tp_order=tp_order,
                               gt_us=t_us, gt_energy_uj=e_uj))

    print(f"  Loaded {len(results)} cases with both T and E")
    return results


# ============================================================
# Feature + Prediction
# ============================================================

def compute_predictions(cases, coeffs, phase2_path, phase3_path):
    """Compute Baseline and Improved T/E predictions for each case."""
    # Load Phase 2 results (Candidate A perf params)
    with open(phase2_path) as f:
        p2 = json.load(f)
    cand_a_params = p2["candidates"]["Candidate-A"]["params"]
    l_sync_a = cand_a_params["L_SYNC"]
    l_core_a = cand_a_params["L_CORE"]
    l_dma_a = cand_a_params["L_DMA"]
    l_startup_a = cand_a_params["L_STARTUP"]

    # Load Phase 3 results (energy params)
    with open(phase3_path) as f:
        p3 = json.load(f)
    e_bl_params = p3["E-Baseline"]["params"]
    e_imp_params = p3["E-Improved"]["params"]

    n = len(cases)
    # Arrays for predictions
    t_baseline_cy = np.zeros(n)
    t_improved_cy = np.zeros(n)
    e_baseline_uj = np.zeros(n)
    e_improved_uj = np.zeros(n)
    gt_t_cy = np.zeros(n)
    gt_e_uj = np.zeros(n)

    for i, mc in enumerate(cases):
        c = mc.cand
        op = mc.op
        macs = op.M * op.K * op.N
        db = total_data_bytes(op, c, mc.tp_order)
        nc = c.num_cores
        tp_tot = c.tp_total
        n_dma = _models.dma_ops_per_step(c.SPm, c.SPn, nc, mc.tp_order)
        n_ev, n_re = _models.dma_ops_decomposed(c.SPm, c.SPn, nc, mc.tp_order)
        tp_inn = _models.tp_inner_value(c.TPm, c.TPk, c.TPn, mc.tp_order)
        d_tot = n_ev * tp_tot + n_re * (tp_tot / tp_inn)

        gt_t_cy[i] = mc.gt_us * CLOCK_MHZ
        gt_e_uj[i] = mc.gt_energy_uj

        # Baseline T (v12)
        t_comp = macs / (nc * coeffs.eff_macs)
        t_comm = db / coeffs.bw_eff_bpc
        t_bl = (t_comp + t_comm
                + coeffs.l_sync_cy * tp_tot
                + coeffs.l_core_cy * nc * tp_tot
                + coeffs.l_dma_cy * n_dma * tp_tot
                + coeffs.l_startup_cy)
        t_baseline_cy[i] = t_bl

        # Improved T (Candidate A)
        t_imp = (t_comp + t_comm
                 + l_sync_a * tp_tot
                 + l_core_a * nc * tp_tot
                 + l_dma_a * d_tot
                 + l_startup_a)
        t_improved_cy[i] = t_imp

        # Baseline E
        t_bl_us = t_bl / CLOCK_MHZ
        e_bl = (e_bl_params["e_mac_pj"] * macs
                + e_bl_params["e_dram_pj"] * db
                + e_bl_params["e_dma_uj"] * n_dma * tp_tot * 1e6
                + e_bl_params["e_sync_uj"] * tp_tot * 1e6
                + (e_bl_params["p_base_uw"] + e_bl_params["p_core_uw"] * nc)
                  * t_bl_us)
        e_baseline_uj[i] = e_bl / 1e6  # pJ -> uJ

        # Improved E
        t_imp_us = t_imp / CLOCK_MHZ
        e_imp = (e_imp_params["e_mac_pj"] * macs
                 + e_imp_params["e_dram_pj"] * db
                 + e_imp_params["e_dma_uj"] * d_tot * 1e6
                 + e_imp_params["e_sync_uj"] * tp_tot * 1e6
                 + (e_imp_params["p_base_uw"] + e_imp_params["p_core_uw"] * nc)
                   * t_imp_us)
        e_improved_uj[i] = e_imp / 1e6  # pJ -> uJ

    return {
        "gt_t_cy": gt_t_cy, "gt_e_uj": gt_e_uj,
        "t_baseline_cy": t_baseline_cy, "t_improved_cy": t_improved_cy,
        "e_baseline_uj": e_baseline_uj, "e_improved_uj": e_improved_uj,
    }


# ============================================================
# EDP evaluation
# ============================================================

def evaluate_edp(cases, preds, label=""):
    """Compute EDP metrics: core accuracy, regret, rho."""
    gt_edp = preds["gt_t_cy"] / CLOCK_MHZ * preds["gt_e_uj"]  # us * uJ
    pred_edp_bl = preds["t_baseline_cy"] / CLOCK_MHZ * preds["e_baseline_uj"]
    pred_edp_imp = preds["t_improved_cy"] / CLOCK_MHZ * preds["e_improved_uj"]

    # Global rho
    rho_bl = spearman_rank_correlation(gt_edp.tolist(), pred_edp_bl.tolist())
    rho_imp = spearman_rank_correlation(gt_edp.tolist(), pred_edp_imp.tolist())

    # Group by problem size
    groups = defaultdict(list)
    for i, mc in enumerate(cases):
        groups[(mc.op.M, mc.op.K, mc.op.N)].append(i)

    results = {}
    for model_name, pred_edp in [("Baseline", pred_edp_bl),
                                  ("Improved", pred_edp_imp)]:
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

            gt_cores = cases[gt_best_idx].cand.num_cores
            pred_cores = cases[pred_best_idx].cand.num_cores

            if gt_cores == pred_cores:
                correct += 1

            # Regret: measured EDP at predicted best vs measured optimal
            regret = float((gt_edp[pred_best_idx] - gt_edp[gt_best_idx])
                          / gt_edp[gt_best_idx] * 100)
            regrets.append(max(0, regret))

            details.append({
                "size": f"{key[0]}x{key[1]}x{key[2]}",
                "gt_cores": gt_cores, "pred_cores": pred_cores,
                "regret_pct": round(regret, 1),
                "match": gt_cores == pred_cores,
            })

        acc_str = f"{correct}/{total}" if total else "N/A"
        le10 = sum(1 for r in regrets if r <= 10)
        results[model_name] = {
            "core_accuracy": acc_str,
            "regret_mean": float(np.mean(regrets)) if regrets else 0,
            "regret_max": float(np.max(regrets)) if regrets else 0,
            "regret_le10": f"{le10}/{total}",
            "details": details,
        }

    return results, rho_bl, rho_imp, gt_edp, pred_edp_bl, pred_edp_imp


def evaluate_tporder(cases, preds):
    """Evaluate tpOrder selection accuracy under Improved model."""
    gt_edp = preds["gt_t_cy"] / CLOCK_MHZ * preds["gt_e_uj"]
    pred_edp_imp = preds["t_improved_cy"] / CLOCK_MHZ * preds["e_improved_uj"]
    pred_edp_bl = preds["t_baseline_cy"] / CLOCK_MHZ * preds["e_baseline_uj"]

    # Group by (size, SP config) to compare tpOrders
    groups = defaultdict(list)
    for i, mc in enumerate(cases):
        groups[(mc.op.M, mc.op.K, mc.op.N,
                mc.cand.SPm, mc.cand.SPn)].append(i)

    results = {"Baseline": {"correct": 0, "total": 0, "details": []},
               "Improved": {"correct": 0, "total": 0, "details": []}}

    for key, idx_list in groups.items():
        M, K, N, SPm, SPn = key
        # Group by tpOrder
        by_tp = defaultdict(list)
        for i in idx_list:
            by_tp[cases[i].tp_order].append(i)
        if len(by_tp) < 2:
            continue

        for model_name, pred_edp in [("Baseline", pred_edp_bl),
                                      ("Improved", pred_edp_imp)]:
            results[model_name]["total"] += 1

            gt_best_tp = {}
            pred_best_tp = {}
            for tp_ord, tp_idx in by_tp.items():
                g = np.array(tp_idx)
                gt_best_tp[tp_ord] = float(np.min(gt_edp[g]))
                pred_best_tp[tp_ord] = float(np.min(pred_edp[g]))

            gt_best = min(gt_best_tp, key=gt_best_tp.get)
            pred_best = min(pred_best_tp, key=pred_best_tp.get)

            if gt_best == pred_best:
                results[model_name]["correct"] += 1

            results[model_name]["details"].append({
                "size": f"{M}x{K}x{N}", "SP": f"{SPm}x{SPn}",
                "gt_best": gt_best, "pred_best": pred_best,
                "match": gt_best == pred_best,
            })

    return results


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Task 12 Phase 4: EDP comprehensive evaluation")
    ap.add_argument("--result", required=True)
    ap.add_argument("--tc", required=True)
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--phase2", default="out/reports/task12_phase2_results.json")
    ap.add_argument("--phase3", default="out/reports/task12_phase3_results.json")
    ap.add_argument("--plot-dir", default="out/plots/task12_phase4")
    args = ap.parse_args()

    print("=" * 70)
    print("PHASE 4: EDP COMPREHENSIVE EVALUATION")
    print("=" * 70)

    coeffs = load_calibration(Path(args.calib))
    cases = load_edp_data(args.result, args.tc)
    preds = compute_predictions(cases, coeffs, args.phase2, args.phase3)

    # EDP evaluation
    edp_results, rho_bl, rho_imp, gt_edp, pred_edp_bl, pred_edp_imp = \
        evaluate_edp(cases, preds)

    print(f"\n--- EDP Core Selection ---")
    print(f"  {'Metric':<22} {'Baseline':>12} {'Improved':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12}")
    for metric in ["core_accuracy", "regret_mean", "regret_max", "regret_le10"]:
        bl_val = edp_results["Baseline"][metric]
        imp_val = edp_results["Improved"][metric]
        if isinstance(bl_val, float):
            print(f"  {metric:<22} {bl_val:>11.1f}% {imp_val:>11.1f}%")
        else:
            print(f"  {metric:<22} {str(bl_val):>12} {str(imp_val):>12}")
    print(f"  {'Global rho':<22} {rho_bl:>12.4f} {rho_imp:>12.4f}")

    # Detailed per-size comparison
    print(f"\n--- EDP Per-Size Details ---")
    print(f"  {'Size':<20} {'BL_GT':>5} {'BL_Pred':>7} {'BL_Reg':>7} "
          f"{'IMP_GT':>6} {'IMP_Pred':>8} {'IMP_Reg':>8}")
    bl_details = edp_results["Baseline"]["details"]
    imp_details = edp_results["Improved"]["details"]
    for bl_d, imp_d in zip(bl_details, imp_details):
        bl_mark = "OK" if bl_d["match"] else "MISS"
        imp_mark = "OK" if imp_d["match"] else "MISS"
        delta = imp_d["regret_pct"] - bl_d["regret_pct"]
        delta_str = f"({delta:+.1f})"
        print(f"  {bl_d['size']:<20} {bl_d['gt_cores']:>5} {bl_d['pred_cores']:>5} "
              f"{bl_mark:>5} "
              f"{imp_d['gt_cores']:>6} {imp_d['pred_cores']:>6} "
              f"{imp_mark:>6} {delta_str:>8}")

    # Worst-case top 3
    print(f"\n--- Worst-case EDP Regret (Top 3) ---")
    for model_name in ["Baseline", "Improved"]:
        details = sorted(edp_results[model_name]["details"],
                        key=lambda d: d["regret_pct"], reverse=True)
        print(f"  {model_name}:")
        for d in details[:3]:
            print(f"    {d['size']}: GT={d['gt_cores']}c, Pred={d['pred_cores']}c, "
                  f"Regret={d['regret_pct']:.1f}%")

    # tpOrder evaluation
    print(f"\n--- tpOrder Selection Accuracy ---")
    tp_results = evaluate_tporder(cases, preds)
    for model_name in ["Baseline", "Improved"]:
        r = tp_results[model_name]
        c, t = r["correct"], r["total"]
        pct = c / t * 100 if t else 0
        print(f"  {model_name}: {c}/{t} ({pct:.0f}%)")

        miss = [d for d in r["details"] if not d["match"]]
        if miss:
            print(f"    MISS cases:")
            for d in miss:
                print(f"      {d['size']} SP={d['SP']}: GT={d['gt_best']}, "
                      f"Pred={d['pred_best']}")

    # Generate plots
    _generate_edp_plots(cases, gt_edp, pred_edp_bl, pred_edp_imp,
                        edp_results, args.plot_dir)

    # Save results
    output = {
        "baseline": {
            "core_accuracy": edp_results["Baseline"]["core_accuracy"],
            "regret_mean": edp_results["Baseline"]["regret_mean"],
            "regret_max": edp_results["Baseline"]["regret_max"],
            "regret_le10": edp_results["Baseline"]["regret_le10"],
            "global_rho": rho_bl,
        },
        "improved": {
            "core_accuracy": edp_results["Improved"]["core_accuracy"],
            "regret_mean": edp_results["Improved"]["regret_mean"],
            "regret_max": edp_results["Improved"]["regret_max"],
            "regret_le10": edp_results["Improved"]["regret_le10"],
            "global_rho": rho_imp,
        },
        "tporder": {
            "baseline": f"{tp_results['Baseline']['correct']}/{tp_results['Baseline']['total']}",
            "improved": f"{tp_results['Improved']['correct']}/{tp_results['Improved']['total']}",
        },
    }
    out_path = Path("out/reports/task12_phase4_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Phase 4 results saved to {out_path}")


def _generate_edp_plots(cases, gt_edp, pred_edp_bl, pred_edp_imp,
                        edp_results, plot_dir):
    """Generate EDP evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [WARN] matplotlib not available, skipping plots.")
        return

    pdir = Path(plot_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    # 1. EDP Predicted vs Measured (Baseline vs Improved)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pred_edp, title in zip(axes,
        [pred_edp_bl, pred_edp_imp],
        ["Baseline (v12)", "Improved (Candidate A)"]):
        ax.scatter(gt_edp, pred_edp, alpha=0.3, s=10)
        lims = [min(gt_edp.min(), pred_edp.min()) * 0.5,
                max(gt_edp.max(), pred_edp.max()) * 2]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel("Measured EDP (us*uJ)")
        ax.set_ylabel("Predicted EDP (us*uJ)")
        ax.set_title(f"EDP: {title}")
        ax.set_xscale('log')
        ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(pdir / "edp_pred_vs_meas.png", dpi=150)
    plt.close(fig)

    # 2. Per-size EDP regret comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bl_details = edp_results["Baseline"]["details"]
    imp_details = edp_results["Improved"]["details"]
    sizes = [d["size"] for d in bl_details]
    bl_regrets = [d["regret_pct"] for d in bl_details]
    imp_regrets = [d["regret_pct"] for d in imp_details]

    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x - width/2, bl_regrets, width, label='Baseline', alpha=0.7)
    ax.bar(x + width/2, imp_regrets, width, label='Improved', alpha=0.7)
    ax.axhline(10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("EDP Regret (%)")
    ax.set_title("EDP Regret by Problem Size")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=45, ha='right', fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "edp_regret_comparison.png", dpi=150)
    plt.close(fig)

    # 3. EDP landscape for 2-3 representative sizes
    groups = defaultdict(list)
    for i, mc in enumerate(cases):
        groups[(mc.op.M, mc.op.K, mc.op.N)].append(i)

    # Pick sizes with high regret change
    sizes_to_plot = []
    for bl_d, imp_d in zip(bl_details, imp_details):
        delta = abs(bl_d["regret_pct"] - imp_d["regret_pct"])
        sizes_to_plot.append((delta, bl_d["size"]))
    sizes_to_plot.sort(reverse=True)
    plot_sizes = [s for _, s in sizes_to_plot[:3]]

    fig, axes = plt.subplots(1, min(3, len(plot_sizes)),
                             figsize=(5 * min(3, len(plot_sizes)), 5))
    if len(plot_sizes) == 1:
        axes = [axes]

    for ax, size_str in zip(axes, plot_sizes):
        parts = size_str.split("x")
        M, K, N = int(parts[0]), int(parts[1]), int(parts[2])
        idx = groups.get((M, K, N), [])
        if not idx:
            continue

        # Group by core count
        by_cores = defaultdict(list)
        for i in idx:
            by_cores[cases[i].cand.num_cores].append(i)

        cores_list = sorted(by_cores.keys())
        gt_edp_by_core = [np.median(gt_edp[np.array(by_cores[nc])])
                          for nc in cores_list]
        bl_edp_by_core = [np.median(pred_edp_bl[np.array(by_cores[nc])])
                          for nc in cores_list]
        imp_edp_by_core = [np.median(pred_edp_imp[np.array(by_cores[nc])])
                           for nc in cores_list]

        ax.plot(cores_list, gt_edp_by_core, 'ko-', label='Measured', linewidth=2)
        ax.plot(cores_list, bl_edp_by_core, 'b^--', label='Baseline', alpha=0.7)
        ax.plot(cores_list, imp_edp_by_core, 'rs--', label='Improved', alpha=0.7)
        ax.set_xlabel("Cores")
        ax.set_ylabel("EDP (us*uJ)")
        ax.set_title(f"{size_str}")
        ax.legend(fontsize=8)
        ax.set_yscale('log')

    fig.suptitle("EDP Landscape (Baseline vs Improved vs Measured)")
    fig.tight_layout()
    fig.savefig(pdir / "edp_landscape.png", dpi=150)
    plt.close(fig)

    print(f"\n  EDP plots saved to {pdir}/")


if __name__ == "__main__":
    main()
