#!/usr/bin/env python3
"""Compare OLD vs NEW calibration on a common clean subset.

Uses existing load_data/compute_features/fit_model from recalibrate_model.py
for performance, and models.EnergyModel for energy to stay consistent with
the project's fitting pipeline.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]  # test/onnx-mlir
sys.path.insert(0, str(ROOT / "scripts" / "analyze"))
sys.path.insert(0, str(ROOT / "scripts" / "generate"))

import recalibrate_model as RM  # noqa: E402
import models as MD  # noqa: E402
from models import total_data_bytes  # noqa: E402


def eval_metrics(gt, pred, group_keys=None):
    """Return (within-group rho, global rho, MAPE)."""
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    pred = np.maximum(pred, 1e-6)
    # Global rho
    rho_global = spearmanr(gt, pred).statistic
    # Within-group rho (weighted by group size), matches calibration.json
    if group_keys is not None:
        from collections import defaultdict
        groups = defaultdict(list)
        for i, k in enumerate(group_keys):
            groups[k].append(i)
        tw, tr = 0.0, 0.0
        for _, idx in groups.items():
            if len(idx) < 3:
                continue
            g = np.array(idx)
            r = spearmanr(gt[g], pred[g]).statistic
            if r is not None and not np.isnan(r):
                tr += r * len(idx)
                tw += len(idx)
        rho_wg = tr / tw if tw > 0 else float("nan")
    else:
        rho_wg = float("nan")
    mape = float(np.mean(np.abs(gt - pred) / gt) * 100)
    return rho_wg, rho_global, mape


# Module-level wrapper so multiprocessing can pickle
_fit_ctx = {}


def _te_loss(p):
    return _fit_ctx["loss_fn"](p)


def fit_te(feat, bounds):
    macs = feat["macs"]; bytes_arr = feat["bytes"]
    n_dma = feat["n_dma"]; tp = feat["tp_total"]
    t_us = feat["t_us"]; nc = feat["n_cores"]
    gt = feat["gt_e_uj"]

    def loss(p):
        pred = predict_te_np(p, macs, bytes_arr, n_dma, tp, t_us, nc)
        pred = np.maximum(pred, 1e-6)
        return float(np.mean((np.log(pred) - np.log(gt)) ** 2))

    _fit_ctx["loss_fn"] = loss
    res = differential_evolution(
        _te_loss, bounds, seed=42, tol=1e-8, maxiter=1000,
        popsize=15, polish=True)
    return res.x, res.fun


def predict_te_np(p, macs, bytes_arr, n_dma, tp_total, t_us, nc):
    """T-E in uJ. Matches EnergyModel.predict_te (pJ internally / 1e6).

    E_pJ = e_mac(pJ/MAC) * macs + e_dram(pJ/byte) * bytes
         + e_dma(uJ) * n_dma * TP * 1e6 + e_sync(uJ) * TP * 1e6
         + (p_base + p_core * P)(uW) * t_total_us
    return E_uJ = E_pJ / 1e6.

    Where macs = M*K*N, bytes = total_data_bytes (tpOrder-dependent).
    """
    e_mac_pj, e_dram_pj, e_dma_uj, e_sync_uj, p_base_uw, p_core_uw = p
    e_pj = (e_mac_pj * macs
            + e_dram_pj * bytes_arr
            + e_dma_uj * n_dma * tp_total * 1e6
            + e_sync_uj * tp_total * 1e6
            + (p_base_uw + p_core_uw * nc) * t_us)
    return e_pj / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tc", required=True)
    ap.add_argument("--calib", default="data/calibration.json")
    args = ap.parse_args()

    # Load HW constants
    RM._load_hw_constants(args.calib)
    print(f"[INFO] EFF_MACS={RM.EFF_MACS}, BW_BPC={RM.BW_BPC}")

    # Load cases using recalibrate_model's infrastructure (skips invalid rows)
    cases = RM.load_data(args.csv, args.tc, "batch_min_avg_us")
    print(f"[INFO] Cases loaded: {len(cases)}")

    # Compute perf features
    feat_p = RM.compute_features(cases)

    # --- Load OLD calibration ---
    with open(args.calib) as f:
        calib = json.load(f)

    # v9 Core-Sync: alpha=beta=1 fixed, L_CORE=0 fixed (4 params fitted)
    #   T = t_comp + t_comm + L_SYNC*TP + L_CORE*P*TP
    #       + L_DMA*N_dma + L_STARTUP
    old_perf = np.array([
        float(calib["l_sync_cy"]),
        float(calib["l_core_cy"]),
        float(calib["l_dma_cy"]),
        float(calib["l_startup_cy"]),
    ])

    def predict_core_sync(f, p):
        ls, ls2, ld, lst = p
        return (f["t_comp"] + f["t_comm"]
                + ls * f["tp_total"] + ls2 * f["core_x_tp"]
                + ld * f["total_dma_ops"]
                + lst)

    group_keys_p = [(c.op.M, c.op.K, c.op.N) for c in cases]
    # Evaluate OLD on current data
    pred_old = predict_core_sync(feat_p, old_perf)
    rho_old_wg, rho_old, mape_old = eval_metrics(
        feat_p["gt_cy"], pred_old, group_keys_p)

    # --- NEW fit on current data (v9 Core-Sync: 4 params) ---
    bounds = [(0, 5e5), (0, 1e5), (0, 5e5), (0, 5e5)]
    print("\n[NEW] Fitting Core-Sync...")
    new_perf, _ = RM.fit_model(feat_p, predict_core_sync, bounds)
    pred_new = predict_core_sync(feat_p, new_perf)
    rho_new_wg, rho_new, mape_new = eval_metrics(
        feat_p["gt_cy"], pred_new, group_keys_p)

    # ==========================================================
    print("\n" + "=" * 72)
    print("PERFORMANCE MODEL (Core-Sync, 7 parameters)")
    print("=" * 72)
    names = ["L_SYNC", "L_CORE", "L_DMA", "L_STARTUP"]
    print(f"\n{'Coeff':<11} {'OLD (400-case)':>16} {'NEW (clean refit)':>20}")
    print("-" * 52)
    for nm, o, n in zip(names, old_perf, new_perf):
        print(f"{nm:<11} {o:>16.4f} {n:>20.4f}")

    old_fit = calib.get("fitted_from", {})
    print(f"\n{'Metric':<22} {'OLD':>12} {'NEW':>12}")
    print("-" * 48)
    print(f"{'rho (fit-time)':<22} {old_fit.get('spearman_rho', float('nan')):>12.4f} "
          f"{'-':>12}")
    print(f"{'MAPE (fit-time) %':<22} {old_fit.get('mape_pct', float('nan')):>12.1f} "
          f"{'-':>12}")
    print(f"{'n_samples (fit)':<22} {old_fit.get('n_samples', '?'):>12} "
          f"{len(cases):>12}")
    print(f"{'rho_wg (within-size)':<22} {rho_old_wg:>12.4f} {rho_new_wg:>12.4f}")
    print(f"{'rho_global':<22} {rho_old:>12.4f} {rho_new:>12.4f}")
    print(f"{'MAPE (eval on clean)':<22} {mape_old:>12.1f} {mape_new:>12.1f}")

    # ==========================================================
    # ENERGY MODEL (T-E)
    # ==========================================================
    # Build energy features matching EnergyModel.predict_te convention:
    #   macs = M*K*N (NOT 2*M*K*N); bytes = total_data_bytes(tpOrder-dep).
    ELEM_BYTES = 2  # bf16
    macs = np.array([c.op.M * c.op.K * c.op.N for c in cases], dtype=float)
    bytes_arr = np.array([
        total_data_bytes(c.op.M, c.op.K, c.op.N, ELEM_BYTES,
                         c.cand.SPm, c.cand.SPn,
                         c.cand.TPm, c.cand.TPk, c.cand.TPn,
                         c.tp_order)
        for c in cases], dtype=float)
    tp_arr = np.array([c.cand.tp_total for c in cases], dtype=float)
    n_dma_per_step = np.array([RM._dma_ops_per_step(c) for c in cases],
                              dtype=float)
    t_us = np.array([c.gt_us for c in cases], dtype=float)
    nc = np.array([c.cand.num_cores for c in cases], dtype=float)

    # Pull ground-truth energy per iter from csv (batch_min_energy_per_iter_uj)
    import csv as _csv
    from collections import defaultdict
    e_map = defaultdict(list)
    with open(args.csv) as f:
        lines = [l for l in f if not l.lstrip().startswith("#")]
    for row in _csv.DictReader(lines):
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
    valid = np.ones(len(cases), dtype=bool)
    for i, c in enumerate(cases):
        k = (c.op.M, c.op.K, c.op.N, c.cand.num_cores,
             c.cand.SPm, c.cand.SPn, c.cand.TPm, c.cand.TPk, c.cand.TPn,
             c.cand.TM, c.cand.TK, c.cand.TN)
        if k in e_map and e_map[k]:
            gt_e[i] = min(e_map[k])
        else:
            valid[i] = False

    # Restrict to rows with valid energy
    idx = np.where(valid)[0]
    print(f"\n[INFO] Energy-valid cases: {len(idx)} / {len(cases)}")
    feat_e = {
        "macs": macs[idx], "bytes": bytes_arr[idx],
        "n_dma": n_dma_per_step[idx], "tp_total": tp_arr[idx],
        "t_us": t_us[idx], "n_cores": nc[idx],
        "gt_e_uj": gt_e[idx],
    }

    # OLD energy coefficients
    energy_params = calib.get("energy", {}).get("params", {})
    old_energy = np.array([
        energy_params.get("e_mac_pj", 0),
        energy_params.get("e_dram_pj", 0),
        energy_params.get("e_dma_uj", 0),
        energy_params.get("e_sync_uj", 0),
        energy_params.get("p_base_uw", 0),
        energy_params.get("p_core_uw", 0),
    ])
    group_keys_e = [(cases[i].op.M, cases[i].op.K, cases[i].op.N)
                    for i in idx]
    pred_old_e = predict_te_np(old_energy, feat_e["macs"], feat_e["bytes"],
                                feat_e["n_dma"], feat_e["tp_total"],
                                feat_e["t_us"], feat_e["n_cores"])
    rho_old_e_wg, rho_old_e, mape_old_e = eval_metrics(
        feat_e["gt_e_uj"], pred_old_e, group_keys_e)

    bounds_e = [(0, 100), (0, 5000), (0, 100), (0, 1000),
                (1e4, 1e7), (1e3, 5e5)]
    print("\n[NEW] Fitting T-E...")
    new_energy, _ = fit_te(feat_e, bounds_e)
    pred_new_e = predict_te_np(new_energy, feat_e["macs"], feat_e["bytes"],
                                feat_e["n_dma"], feat_e["tp_total"],
                                feat_e["t_us"], feat_e["n_cores"])
    rho_new_e_wg, rho_new_e, mape_new_e = eval_metrics(
        feat_e["gt_e_uj"], pred_new_e, group_keys_e)

    print("\n" + "=" * 72)
    print("ENERGY MODEL (T-E, 6 parameters)")
    print("=" * 72)
    e_names = ["e_mac_pj", "e_dram_pj", "e_dma_uj", "e_sync_uj",
               "p_base_uw", "p_core_uw"]
    print(f"\n{'Coeff':<12} {'OLD (400-case)':>18} {'NEW (clean refit)':>20}")
    print("-" * 54)
    for nm, o, n in zip(e_names, old_energy, new_energy):
        print(f"{nm:<12} {o:>18.4f} {n:>20.4f}")

    old_e_fit = calib.get("energy", {}).get("fitted_from", {})
    print(f"\n{'Metric':<22} {'OLD':>12} {'NEW':>12}")
    print("-" * 48)
    print(f"{'rho (fit-time)':<22} {old_e_fit.get('spearman_rho', float('nan')):>12.4f} "
          f"{'-':>12}")
    print(f"{'MAPE (fit-time) %':<22} {old_e_fit.get('mape_pct', float('nan')):>12.1f} "
          f"{'-':>12}")
    print(f"{'n_samples (fit)':<22} {old_e_fit.get('n_samples', '?'):>12} "
          f"{len(idx):>12}")
    print(f"{'rho_wg (within-size)':<22} {rho_old_e_wg:>12.4f} {rho_new_e_wg:>12.4f}")
    print(f"{'rho_global':<22} {rho_old_e:>12.4f} {rho_new_e:>12.4f}")
    print(f"{'MAPE (eval on clean)':<22} {mape_old_e:>12.1f} {mape_new_e:>12.1f}")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'':<22} {'OLD (400-case)':>18} {'NEW (clean refit)':>20}")
    print("-" * 62)
    print(f"{'Perf rho_wg':<22} {rho_old_wg:>18.4f} {rho_new_wg:>20.4f}")
    print(f"{'Perf MAPE %':<22} {mape_old:>18.1f} {mape_new:>20.1f}")
    print(f"{'Energy rho_wg':<22} {rho_old_e_wg:>18.4f} {rho_new_e_wg:>20.4f}")
    print(f"{'Energy MAPE %':<22} {mape_old_e:>18.1f} {mape_new_e:>20.1f}")


if __name__ == "__main__":
    main()
