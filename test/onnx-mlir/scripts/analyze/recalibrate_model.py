#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
recalibrate_model.py — Refit performance cost model on expanded measurement data.

Compares four model variants:
  D+B (v5 refit): T = T_comp + T_comm + L_SYNC*TP + L_CORE*N + L_STARTUP
  Scaled:         T = a*T_comp + b*T_comm + L_SYNC*TP + L_CORE*N + L_STARTUP
  Overlap:        T = max(T_comp,T_comm) + g*min(T_comp,T_comm) + L_SYNC*TP + L_CORE*N + L_STARTUP
  Nonlinear:      T = T_comp + T_comm + L_SYNC*TP^alpha + L_CORE*N + L_STARTUP

Where:
  T_comp = M*K*N / (N_cores * EFF_MACS)
  T_comm = total_data_bytes / BW

Deduplicates repeated measurements (keeps min_us per unique config).
Uses differential evolution (global optimizer) for robust fitting.

Usage:
    python3 scripts/analyze/recalibrate_model.py \
        --result out/reports/result_all.csv \
        --tc out/tc_list_all.json \
        [--gt min_us] [--cv] [--output data/calibration.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import OpCase, CommentFilterFile  # noqa: E402
from cost_model import (  # noqa: E402
    Candidate, total_data_bytes, TP_AXIS_K,
)
import models as _models  # noqa: E402

# Fixed trace-verified constants
EFF_MACS = 24.28   # MACs/cycle (trace ss_kernel_cy median)
BW_BPC = 4.0       # Bytes/cycle
CLOCK_MHZ = 1500


# ============================================================
# Data loading
# ============================================================

@dataclass
class MeasuredCase:
    """One measured test case with tiling config and ground truth."""
    op: OpCase
    cand: Candidate
    tp_order: int
    gt_us: float
    gt_cy: float


def load_data(
    result_path: str, tc_path: str, gt_field: str,
) -> List[MeasuredCase]:
    """Load and deduplicate measurement results (keep min per config)."""
    # Load tc_list for tpOrder lookup
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

    # Collect all PASS rows, group by unique config
    raw: Dict[tuple, List[float]] = defaultdict(list)
    with open(result_path) as f:
        for row in csv.DictReader(CommentFilterFile(f)):
            if row["status"] != "PASS":
                continue
            gt_val = float(row[gt_field])
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

    n_raw = sum(len(v) for v in raw.values())
    n_dup = sum(1 for v in raw.values() if len(v) > 1)

    # Deduplicate: keep minimum
    results: List[MeasuredCase] = []
    for (M, K, N, nc, SPm, SPn, TPm, TPk, TPn, TM, TK, TN), vals in raw.items():
        gt_val = min(vals)
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
        ))

    print(f"  {n_raw} PASS rows -> {len(results)} unique configs "
          f"({n_dup} duplicated, min kept)")
    return results


# ============================================================
# Feature extraction
# ============================================================

def _dma_ops_per_step(mc: MeasuredCase) -> int:
    """Delegates to models.dma_ops_per_step()."""
    c = mc.cand
    return _models.dma_ops_per_step(c.SPm, c.SPn, c.num_cores, mc.tp_order)


def compute_features(cases: List[MeasuredCase]) -> Dict[str, np.ndarray]:
    """Compute model features for all cases."""
    n = len(cases)
    t_comp = np.zeros(n)
    t_comm = np.zeros(n)
    t_comm_raw = np.zeros(n)
    tp_total = np.zeros(n)
    n_cores = np.zeros(n)
    gt_cy = np.zeros(n)
    total_dma_ops = np.zeros(n)   # DMA descriptor setups across all iterations
    dma_per_step = np.zeros(n)    # DMA ops per temporal iteration

    for i, mc in enumerate(cases):
        t_comp[i] = (mc.op.M * mc.op.K * mc.op.N) / (mc.cand.num_cores * EFF_MACS)
        raw = total_data_bytes(mc.op, mc.cand, mc.tp_order) / BW_BPC
        t_comm_raw[i] = raw
        t_comm[i] = raw
        tp_total[i] = mc.cand.tp_total
        n_cores[i] = mc.cand.num_cores
        gt_cy[i] = mc.gt_cy
        dps = _dma_ops_per_step(mc)
        dma_per_step[i] = dps
        total_dma_ops[i] = dps * mc.cand.tp_total

    # Core-Sync: P * TP_total interaction (barrier cost scales with cores)
    core_x_tp = n_cores * tp_total

    return {
        "t_comp": t_comp, "t_comm": t_comm, "t_comm_raw": t_comm_raw,
        "tp_total": tp_total, "n_cores": n_cores,
        "gt_cy": gt_cy,
        "total_dma_ops": total_dma_ops, "dma_per_step": dma_per_step,
        "core_x_tp": core_x_tp,
    }


# ============================================================
# Model definitions
# ============================================================

def predict_baseline(f, p):
    """D+B v5: T_comp + T_comm + L_SYNC*TP + L_CORE*N + L_STARTUP."""
    l_sync, l_core, l_startup = p
    return f["t_comp"] + f["t_comm"] + l_sync*f["tp_total"] + l_core*f["n_cores"] + l_startup


def predict_scaled(f, p):
    """Scaled: a*T_comp + b*T_comm + L_SYNC*TP + L_CORE*N + L_STARTUP."""
    a, b, l_sync, l_core, l_startup = p
    return a*f["t_comp"] + b*f["t_comm"] + l_sync*f["tp_total"] + l_core*f["n_cores"] + l_startup


def predict_overlap(f, p):
    """Overlap: max(comp,comm) + g*min(comp,comm) + overhead."""
    gamma, l_sync, l_core, l_startup = p
    t_max = np.maximum(f["t_comp"], f["t_comm"])
    t_min = np.minimum(f["t_comp"], f["t_comm"])
    return t_max + gamma*t_min + l_sync*f["tp_total"] + l_core*f["n_cores"] + l_startup


def predict_nonlinear(f, p):
    """Nonlinear: T_comp + T_comm + L_SYNC*TP^alpha + L_CORE*N + L_STARTUP."""
    l_sync, alpha, l_core, l_startup = p
    return (f["t_comp"] + f["t_comm"]
            + l_sync * np.power(f["tp_total"], alpha)
            + l_core*f["n_cores"] + l_startup)


# --- v7 candidates: general-purpose terms only ---

def predict_superlinear(f, p):
    """Superlinear core overhead: coordination cost grows as N^gamma.

    Physical basis: barrier sync and shared-resource arbitration cost
    scales superlinearly with participant count (Amdahl extension).
    """
    a, b, l_sync, l_core, gamma, l_startup = p
    return (a*f["t_comp"] + b*f["t_comm"]
            + l_sync*f["tp_total"]
            + l_core * np.power(f["n_cores"], gamma)
            + l_startup)


def predict_contention(f, p):
    """BW contention: effective bandwidth degrades with concurrent DMA streams.

    Physical basis: shared memory bandwidth suffers queuing delay
    when many cores issue DMA simultaneously (general to any DMA arch).
    T_comm_eff = (beta + beta_core * N_cores) * raw_bytes / BW
    """
    a, b, b_core, l_sync, l_core, l_startup = p
    return (a*f["t_comp"]
            + (b + b_core * f["n_cores"]) * f["t_comm_raw"]
            + l_sync*f["tp_total"]
            + l_core*f["n_cores"]
            + l_startup)


def predict_combined(f, p):
    """Superlinear + contention combined."""
    a, b, b_core, l_sync, l_core, gamma, l_startup = p
    return (a*f["t_comp"]
            + (b + b_core * f["n_cores"]) * f["t_comm_raw"]
            + l_sync*f["tp_total"]
            + l_core * np.power(f["n_cores"], gamma)
            + l_startup)


MODELS = {
    "D+B (v5 refit)": {
        "predict": predict_baseline,
        "bounds": [(0, 5e5), (0, 5e5), (0, 5e5)],
        "param_names": ["L_SYNC", "L_CORE", "L_STARTUP"],
    },
    "Scaled": {
        "predict": predict_scaled,
        "bounds": [(0.001, 10), (0.01, 10), (0, 5e5), (0, 5e5), (0, 5e5)],
        "param_names": ["alpha", "beta", "L_SYNC", "L_CORE", "L_STARTUP"],
    },
    "Overlap": {
        "predict": predict_overlap,
        "bounds": [(0, 1.5), (0, 5e5), (0, 5e5), (0, 5e5)],
        "param_names": ["gamma", "L_SYNC", "L_CORE", "L_STARTUP"],
    },
    "Nonlinear": {
        "predict": predict_nonlinear,
        "bounds": [(0, 5e5), (0.01, 3.0), (0, 5e5), (0, 5e5)],
        "param_names": ["L_SYNC", "alpha", "L_CORE", "L_STARTUP"],
    },
    "DMA-replace": {
        "predict": lambda f, p: (
            p[0]*f["t_comp"] + p[1]*f["t_comm"]
            + p[2]*f["total_dma_ops"]
            + p[3]*f["n_cores"] + p[4]
        ),
        "bounds": [(0.001, 10), (0.01, 10), (0, 5e5), (0, 5e5), (0, 5e5)],
        "param_names": ["alpha", "beta", "L_DMA", "L_CORE", "L_STARTUP"],
    },
    "DMA-add": {
        "predict": lambda f, p: (
            p[0]*f["t_comp"] + p[1]*f["t_comm"]
            + p[2]*f["tp_total"] + p[3]*f["total_dma_ops"]
            + p[4]*f["n_cores"] + p[5]
        ),
        "bounds": [(0.001, 10), (0.01, 10), (0, 5e5), (0, 5e5), (0, 5e5), (0, 5e5)],
        "param_names": ["alpha", "beta", "L_SYNC", "L_DMA", "L_CORE", "L_STARTUP"],
    },
    "DMA-only": {
        "predict": lambda f, p: (
            p[0]*f["t_comp"] + p[1]*f["t_comm"]
            + p[2]*f["total_dma_ops"]
            + p[3]
        ),
        "bounds": [(0.001, 10), (0.01, 10), (0, 5e5), (0, 5e5)],
        "param_names": ["alpha", "beta", "L_DMA", "L_STARTUP"],
    },
    "Superlinear": {
        "predict": predict_superlinear,
        "bounds": [(0.001, 10), (0.01, 10), (0, 5e5), (0, 5e5), (1.0, 2.0), (0, 5e5)],
        "param_names": ["alpha", "beta", "L_SYNC", "L_CORE", "gamma", "L_STARTUP"],
    },
    "Contention": {
        "predict": predict_contention,
        "bounds": [(0.001, 10), (0.01, 10), (0, 1e3), (0, 5e5), (0, 5e5), (0, 5e5)],
        "param_names": ["alpha", "beta", "beta_core", "L_SYNC", "L_CORE", "L_STARTUP"],
    },
    "Combined": {
        "predict": predict_combined,
        "bounds": [(0.001, 10), (0.01, 10), (0, 1e3), (0, 5e5), (0, 5e5), (1.0, 2.0), (0, 5e5)],
        "param_names": ["alpha", "beta", "beta_core", "L_SYNC", "L_CORE", "gamma", "L_STARTUP"],
    },
    "Core-Sync": {
        "predict": lambda f, p: (
            p[0]*f["t_comp"] + p[1]*f["t_comm"]
            + p[2]*f["tp_total"] + p[3]*f["core_x_tp"]
            + p[4]*f["total_dma_ops"]
            + p[5]*f["n_cores"] + p[6]
        ),
        "bounds": [(0.001, 10), (0.01, 10), (0, 5e5), (0, 1e5),
                   (0, 5e5), (0, 5e5), (0, 5e5)],
        "param_names": ["alpha", "beta", "L_SYNC", "L_SYNC2",
                        "L_DMA", "L_CORE", "L_STARTUP"],
    },
}


# ============================================================
# Fitting (differential evolution for global optimum)
# ============================================================

def log_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Log-space MSE loss."""
    return float(np.mean((np.log(np.maximum(gt, 1)) - np.log(np.maximum(pred, 1))) ** 2))


def fit_model(feat, predict_fn, bounds):
    """Fit using differential evolution (global optimizer)."""
    gt = feat["gt_cy"]
    result = differential_evolution(
        lambda p: log_mse(gt, predict_fn(feat, p)),
        bounds, seed=42, maxiter=2000, tol=1e-14, polish=True,
    )
    return result.x, result.fun


# ============================================================
# Evaluation
# ============================================================

def evaluate(cases, feat, predict_fn, params):
    """Compute rho, MAPE, core accuracy, regret."""
    gt = feat["gt_cy"]
    pred = predict_fn(feat, params)
    n = len(cases)

    # Within-group Spearman rho
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

    # Core accuracy & regret
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
        regret = (gt[pred_best] - gt[gt_best]) / gt[gt_best] * 100
        regrets.append(regret)
        details.append({
            "size": f"{key[0]}x{key[1]}x{key[2]}",
            "gt_cores": gt_cores, "pred_cores": pred_cores,
            "regret_pct": round(regret, 1),
            "match": gt_cores == pred_cores,
        })

    acc_str = f"{correct}/{total} ({correct/total*100:.0f}%)" if total else "N/A"

    # Within-group rank inversions (same size + cores)
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
    inv_pct = inversions / total_pairs * 100 if total_pairs else 0

    return {
        "rho": rho, "mape": mape,
        "core_accuracy": acc_str,
        "regret_mean": float(np.mean(regrets)) if regrets else 0,
        "regret_max": float(np.max(regrets)) if regrets else 0,
        "rank_inversions": f"{inversions}/{total_pairs} ({inv_pct:.1f}%)",
        "core_details": details,
        "n_samples": n,
    }


# ============================================================
# Cross-validation
# ============================================================

def cross_validate(cases, feat, predict_fn, bounds, n_folds=5):
    """K-fold CV by size groups."""
    groups = defaultdict(list)
    for i, mc in enumerate(cases):
        groups[(mc.op.M, mc.op.K, mc.op.N)].append(i)

    keys = sorted(groups.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(keys)

    fold_size = max(1, len(keys) // n_folds)
    cv_rhos, cv_mapes, cv_accs = [], [], []

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

        train_feat = {k: v[train_idx] for k, v in feat.items()}
        test_feat = {k: v[test_idx] for k, v in feat.items()}
        test_cases = [cases[i] for i in test_idx]

        params, _ = fit_model(train_feat, predict_fn, bounds)
        m = evaluate(test_cases, test_feat, predict_fn, params)
        cv_rhos.append(m["rho"])
        cv_mapes.append(m["mape"])
        parts = m["core_accuracy"].split("/")
        if len(parts) == 2:
            num = int(parts[0])
            den = int(parts[1].split("(")[0].strip())
            cv_accs.append(num / den * 100 if den else 0)

    return {
        "cv_rho": float(np.mean(cv_rhos)) if cv_rhos else 0,
        "cv_mape": float(np.mean(cv_mapes)) if cv_mapes else 0,
        "cv_core_acc": float(np.mean(cv_accs)) if cv_accs else 0,
    }


# ============================================================
# Output
# ============================================================

def write_calibration(calib_path, output_path, model_name, params, param_names, metrics):
    """Update calibration.json with new perf model."""
    try:
        with open(calib_path) as f:
            calib = json.load(f)
    except FileNotFoundError:
        calib = {}

    pdict = dict(zip(param_names, [round(float(v), 4) for v in params]))

    if model_name == "Scaled":
        # Effective EFF_MACS = EFF_MACS / alpha (alpha scales T_comp down)
        calib["model"] = "D+B-S"
        calib["perf_alpha"] = pdict["alpha"]
        calib["perf_beta"] = pdict["beta"]
        calib["l_sync_cy"] = pdict["L_SYNC"]
        calib["l_core_cy"] = pdict["L_CORE"]
        calib["l_startup_cy"] = pdict["L_STARTUP"]
    elif model_name == "Overlap":
        calib["model"] = "D+B-OV"
        calib["perf_gamma"] = pdict["gamma"]
        calib["l_sync_cy"] = pdict["L_SYNC"]
        calib["l_core_cy"] = pdict["L_CORE"]
        calib["l_startup_cy"] = pdict["L_STARTUP"]
    elif model_name == "Nonlinear":
        calib["model"] = "D+B-NL"
        calib["l_sync_cy"] = pdict["L_SYNC"]
        calib["l_sync_alpha"] = pdict["alpha"]
        calib["l_core_cy"] = pdict["L_CORE"]
        calib["l_startup_cy"] = pdict["L_STARTUP"]
    elif model_name == "Core-Sync":
        calib["model"] = "Core-Sync"
        calib["perf_alpha"] = pdict["alpha"]
        calib["perf_beta"] = pdict["beta"]
        calib["l_sync_cy"] = pdict["L_SYNC"]
        calib["l_sync2_cy"] = pdict["L_SYNC2"]
        calib["l_dma_cy"] = pdict["L_DMA"]
        calib["l_core_cy"] = pdict["L_CORE"]
        calib["l_startup_cy"] = pdict["L_STARTUP"]
    else:  # D+B baseline
        calib["model"] = "D+B"
        calib["l_sync_cy"] = pdict["L_SYNC"]
        calib["l_core_cy"] = pdict["L_CORE"]
        calib["l_startup_cy"] = pdict["L_STARTUP"]

    calib["version"] = calib.get("version", 5) + 1
    calib["eff_macs"] = EFF_MACS
    calib["bw_eff_bpc"] = BW_BPC
    calib["clock_mhz"] = CLOCK_MHZ

    calib["fitted_from"] = {
        "n_samples": metrics["n_samples"],
        "spearman_rho": round(metrics["rho"], 4),
        "mape_pct": round(metrics["mape"], 1),
        "core_accuracy": metrics["core_accuracy"],
        "regret_mean_pct": round(metrics["regret_mean"], 1),
        "regret_max_pct": round(metrics["regret_max"], 1),
        "ground_truth": "min_us",
        "eff_macs_source": "trace ss_kernel_cy median (fixed)",
        "eff_macs_fixed": True,
        "model_type": model_name,
    }

    with open(output_path, "w") as f:
        json.dump(calib, f, indent=2)
        f.write("\n")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Recalibrate perf cost model")
    ap.add_argument("--result", required=True)
    ap.add_argument("--tc", required=True)
    ap.add_argument("--gt", default="min_us")
    ap.add_argument("--output", default=None)
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--cv", action="store_true")
    args = ap.parse_args()

    print(f"Loading data: {args.result}")
    cases = load_data(args.result, args.tc, args.gt)

    feat = compute_features(cases)

    results = {}
    for name, spec in MODELS.items():
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")

        params, loss = fit_model(feat, spec["predict"], spec["bounds"])

        # Print params
        for pn, pv in zip(spec["param_names"], params):
            fmt = ".4f" if pn in ("alpha", "beta", "gamma", "beta_core") else ".0f"
            print(f"  {pn}={pv:{fmt}}", end="")
        print(f"\n  LogMSE={loss:.6f}")

        metrics = evaluate(cases, feat, spec["predict"], params)
        print(f"  rho={metrics['rho']:.4f}  MAPE={metrics['mape']:.1f}%  "
              f"Core={metrics['core_accuracy']}  "
              f"Reg mean={metrics['regret_mean']:.1f}% max={metrics['regret_max']:.1f}%  "
              f"RankInv={metrics['rank_inversions']}")

        if args.cv:
            cv = cross_validate(cases, feat, spec["predict"], spec["bounds"])
            print(f"  CV: rho={cv['cv_rho']:.4f}  MAPE={cv['cv_mape']:.1f}%  "
                  f"CoreAcc={cv['cv_core_acc']:.0f}%")
            metrics["cv"] = cv

        results[name] = {"params": params, "loss": loss, "metrics": metrics}

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'rho':>6} {'MAPE':>7} {'CoreAcc':>10} "
          f"{'RegMean':>8} {'RegMax':>8} {'RankInv':>14} {'LogMSE':>10}")
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<18} {m['rho']:>6.4f} {m['mape']:>6.1f}% "
              f"{m['core_accuracy']:>10} {m['regret_mean']:>7.1f}% "
              f"{m['regret_max']:>7.1f}% {m['rank_inversions']:>14} "
              f"{r['loss']:>10.6f}")

    # Select best: core_accuracy first, then regret_mean
    def score(r):
        m = r["metrics"]
        parts = m["core_accuracy"].split("/")
        acc = int(parts[0]) / int(parts[1].split("(")[0].strip()) if len(parts) == 2 else 0
        return (acc, -m["regret_mean"])

    best_name = max(results, key=lambda n: score(results[n]))
    best = results[best_name]
    print(f"\nBest: {best_name}")

    # Core selection details
    print(f"\n{'Size':<20} {'GT':>4} {'Pred':>4} {'Regret':>8} {'Match':>6}")
    for d in best["metrics"]["core_details"]:
        mark = "OK" if d["match"] else "MISS"
        print(f"  {d['size']:<18} {d['gt_cores']:>4} {d['pred_cores']:>4} "
              f"{d['regret_pct']:>7.1f}% {mark:>6}")

    if args.output:
        spec = MODELS[best_name]
        write_calibration(args.calib, args.output, best_name,
                          best["params"], spec["param_names"],
                          best["metrics"])
        print(f"\nCalibration written to {args.output}")


if __name__ == "__main__":
    main()
