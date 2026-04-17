#!/usr/bin/env python3
"""Compute EDP accuracy with OLD and NEW calibrations.

EDP = T * E. For each matmul size, pick the core count with minimum
predicted EDP and compare to measured optimal.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analyze"))
sys.path.insert(0, str(ROOT / "scripts" / "generate"))

import recalibrate_model as RM  # noqa
from models import total_data_bytes  # noqa

CLOCK_MHZ = 1500.0
ELEM_BYTES = 2


def predict_perf_v9(L_SYNC, L_CORE, L_DMA, L_STARTUP,
                    t_comp, t_comm, tp_total, core_x_tp, dma_ops):
    """Returns predicted cycles."""
    return (t_comp + t_comm
            + L_SYNC * tp_total
            + L_CORE * core_x_tp
            + L_DMA * dma_ops
            + L_STARTUP)


def predict_energy_te(p, macs, bytes_arr, n_dma, tp_total, t_us, nc):
    """Returns predicted energy in uJ."""
    e_mac_pj, e_dram_pj, e_dma_uj, e_sync_uj, p_base_uw, p_core_uw = p
    e_pj = (e_mac_pj * macs
            + e_dram_pj * bytes_arr
            + e_dma_uj * n_dma * tp_total * 1e6
            + e_sync_uj * tp_total * 1e6
            + (p_base_uw + p_core_uw * nc) * t_us)
    return e_pj / 1e6


def compute_edp_accuracy(cases, gt_e, t_pred_cy, e_pred_uj, label):
    """Compare predicted EDP-optimal core count vs measured-optimal."""
    # Ground truth: T_meas * E_meas
    gt_t_us = np.array([c.gt_us for c in cases])
    gt_edp = gt_t_us * gt_e

    # Predicted EDP
    t_pred_us = t_pred_cy / CLOCK_MHZ
    pred_edp = t_pred_us * e_pred_uj

    # Group by size
    by_size = defaultdict(list)
    for i, c in enumerate(cases):
        by_size[(c.op.M, c.op.K, c.op.N)].append(i)

    # For each size, find min-EDP core count
    results = []
    match_core = 0
    total = 0
    regrets = []
    for size, idx in sorted(by_size.items()):
        if len(idx) < 2:
            continue
        total += 1
        idx_arr = np.array(idx)
        # Group by core count within this size
        core_groups = defaultdict(list)
        for i in idx:
            core_groups[cases[i].cand.num_cores].append(i)
        # For each core count, best EDP (min over tilings)
        core_best_gt = {}
        core_best_pred = {}
        for c, lst in core_groups.items():
            core_best_gt[c] = min(gt_edp[i] for i in lst)
            core_best_pred[c] = min(pred_edp[i] for i in lst)
        gt_best_core = min(core_best_gt, key=lambda k: core_best_gt[k])
        pred_best_core = min(core_best_pred, key=lambda k: core_best_pred[k])
        gt_opt_edp = core_best_gt[gt_best_core]
        pred_choice_edp = core_best_gt[pred_best_core]  # actual EDP when using predicted choice
        regret = (pred_choice_edp - gt_opt_edp) / gt_opt_edp * 100
        regrets.append(regret)
        match = gt_best_core == pred_best_core
        if match:
            match_core += 1
        results.append({
            "size": f"{size[0]}x{size[1]}x{size[2]}",
            "gt_cores": gt_best_core,
            "pred_cores": pred_best_core,
            "regret": regret,
            "match": match,
        })

    print(f"\n=== {label} ===")
    print(f"Core Accuracy: {match_core}/{total} "
          f"({match_core*100/total:.0f}%)")
    print(f"Regret: mean={np.mean(regrets):.1f}%  "
          f"max={max(regrets):.1f}%  med={np.median(regrets):.1f}%")
    print(f"\n{'Size':<18s} {'GT':>4s} {'Pred':>5s} {'Regret%':>8s} {'':>6s}")
    print("-" * 50)
    for d in results:
        mark = "OK" if d["match"] else "MISS"
        print(f"  {d['size']:<16s} {d['gt_cores']:>4d} "
              f"{d['pred_cores']:>5d} {d['regret']:>7.1f}% {mark:>6s}")
    return {
        "label": label, "match": match_core, "total": total,
        "regret_mean": float(np.mean(regrets)),
        "regret_max": float(max(regrets)),
        "regret_med": float(np.median(regrets)),
        "details": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tc", required=True)
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--new-perf", nargs=4, type=float, required=True,
                    metavar=("L_SYNC", "L_CORE", "L_DMA", "L_STARTUP"),
                    help="NEW perf coefficients")
    ap.add_argument("--new-energy", nargs=6, type=float, required=True,
                    metavar=("e_mac_pj", "e_dram_pj", "e_dma_uj", "e_sync_uj",
                             "p_base_uw", "p_core_uw"),
                    help="NEW energy coefficients")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    RM._load_hw_constants(args.calib)
    cases = RM.load_data(args.csv, args.tc, "batch_min_avg_us")
    feat = RM.compute_features(cases)

    # Load energy GT
    e_map = defaultdict(list)
    with open(args.csv) as f:
        lines = [l for l in f if not l.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        if row.get("status") != "PASS":
            continue
        try:
            e = float(row["batch_min_energy_per_iter_uj"])
            if e <= 0:
                continue
        except (ValueError, KeyError):
            continue
        k = (int(row["M"]), int(row["K"]), int(row["N"]),
             int(row["numSpm"]), int(row["SPm"]), int(row["SPn"]),
             int(row["TPm"]), int(row["TPk"]), int(row["TPn"]),
             int(row["TM"]), int(row["TK"]), int(row["TN"]))
        e_map[k].append(e)

    gt_e = np.zeros(len(cases))
    for i, c in enumerate(cases):
        k = (c.op.M, c.op.K, c.op.N, c.cand.num_cores,
             c.cand.SPm, c.cand.SPn, c.cand.TPm, c.cand.TPk, c.cand.TPn,
             c.cand.TM, c.cand.TK, c.cand.TN)
        if e_map[k]:
            gt_e[i] = min(e_map[k])

    # Feature arrays for T-E
    macs = np.array([c.op.M*c.op.K*c.op.N for c in cases], dtype=float)
    bytes_arr = np.array([total_data_bytes(c.op.M, c.op.K, c.op.N, ELEM_BYTES,
                                            c.cand.SPm, c.cand.SPn,
                                            c.cand.TPm, c.cand.TPk, c.cand.TPn,
                                            c.tp_order)
                          for c in cases], dtype=float)
    tp_total = np.array([c.cand.tp_total for c in cases], dtype=float)
    n_dma = np.array([RM._dma_ops_per_step(c) for c in cases], dtype=float)
    t_us = np.array([c.gt_us for c in cases], dtype=float)
    nc = np.array([c.cand.num_cores for c in cases], dtype=float)

    with open(args.calib) as f:
        calib = json.load(f)

    old_perf = [float(calib["l_sync_cy"]), float(calib["l_core_cy"]),
                float(calib["l_dma_cy"]), float(calib["l_startup_cy"])]
    old_energy_p = calib.get("energy", {}).get("params", {})
    old_energy = [
        old_energy_p.get("e_mac_pj", 0),
        old_energy_p.get("e_dram_pj", 0),
        old_energy_p.get("e_dma_uj", 0),
        old_energy_p.get("e_sync_uj", 0),
        old_energy_p.get("p_base_uw", 0),
        old_energy_p.get("p_core_uw", 0),
    ]

    # --- Predictions ---
    t_old = predict_perf_v9(*old_perf,
                            t_comp=feat["t_comp"], t_comm=feat["t_comm"],
                            tp_total=feat["tp_total"],
                            core_x_tp=feat["core_x_tp"],
                            dma_ops=feat["total_dma_ops"])
    t_new = predict_perf_v9(*args.new_perf,
                            t_comp=feat["t_comp"], t_comm=feat["t_comm"],
                            tp_total=feat["tp_total"],
                            core_x_tp=feat["core_x_tp"],
                            dma_ops=feat["total_dma_ops"])
    e_old = predict_energy_te(old_energy, macs, bytes_arr,
                               n_dma, tp_total, t_us, nc)
    e_new = predict_energy_te(args.new_energy, macs, bytes_arr,
                               n_dma, tp_total, t_us, nc)

    # Also measured T + measured E as oracle
    t_meas = np.array([c.gt_us * CLOCK_MHZ for c in cases], dtype=float)

    print("\n" + "=" * 70)
    print("EDP Core-Selection Accuracy")
    print("=" * 70)
    results = []
    results.append(compute_edp_accuracy(
        cases, gt_e, t_meas, gt_e,
        "Oracle (measured T * measured E)"))
    results.append(compute_edp_accuracy(
        cases, gt_e, t_old, e_old,
        "OLD (fit from 452 samples)"))
    results.append(compute_edp_accuracy(
        cases, gt_e, t_new, e_new,
        "NEW (refit on 1005 clean samples)"))
    # Hybrid: meas T + NEW E, NEW T + meas E
    results.append(compute_edp_accuracy(
        cases, gt_e, t_meas, e_new,
        "Hybrid: measured T + NEW E"))
    results.append(compute_edp_accuracy(
        cases, gt_e, t_new, gt_e,
        "Hybrid: NEW T + measured E"))

    print("\n" + "=" * 70)
    print("SUMMARY — EDP Core Selection")
    print("=" * 70)
    print(f"{'Scheme':<40s} {'Core Acc':>10s} {'RegMean%':>10s} "
          f"{'RegMax%':>10s}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<40s} "
              f"{r['match']:>3d}/{r['total']:<4d}     "
              f"{r['regret_mean']:>9.1f} {r['regret_max']:>10.1f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
