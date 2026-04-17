#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task15_remeasure_eval.py -- Task 15: High-Regret Workload Re-measurement & EDP Re-evaluation.

Phases:
  0:  Model version verification (Notion-specified vs codebase)
  1-prep: Filter tc_list for 7 target workloads
  1-post: Merge re-measured results with existing v12 data
  2:  EDP re-evaluation (no re-calibration, v12 vs v13 comparison)

Usage:
    # Phase 0: Verify model coefficients
    python3 scripts/analyze/task15_remeasure_eval.py \
        --result-v12 out/reports/result_v12.csv \
        --tc out/tc_list_v11.json \
        --calib data/calibration.json \
        --phase 0

    # Phase 1-prep: Create filtered tc_list for 7 target workloads
    python3 scripts/analyze/task15_remeasure_eval.py \
        --tc out/tc_list_v11.json \
        --phase 1-prep

    # Phase 1-post: Merge re-measured results into v13
    python3 scripts/analyze/task15_remeasure_eval.py \
        --result-v12 out/reports/result_v12.csv \
        --result-remeasured out/reports/result_v13_remeasured.csv \
        --phase 1-post

    # Phase 2: EDP re-evaluation
    python3 scripts/analyze/task15_remeasure_eval.py \
        --result-v12 out/reports/result_v12.csv \
        --result-v13 out/reports/result_v13.csv \
        --tc out/tc_list_v11.json \
        --calib data/calibration.json \
        --phase 2

    # All phases (after measurement is done):
    python3 scripts/analyze/task15_remeasure_eval.py \
        --result-v12 out/reports/result_v12.csv \
        --result-v13 out/reports/result_v13.csv \
        --tc out/tc_list_v11.json \
        --calib data/calibration.json \
        --phase all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402
from task13_compact_energy import EnergyRow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import load_calibration, CommentFilterFile, OpCase  # noqa: E402
from cost_model import Candidate, total_data_bytes  # noqa: E402
import models as _models  # noqa: E402

CLOCK_MHZ = 1500

# Ground-truth measurement columns used by Task 15 EDP evaluation.
# Matches the Draft v3 analysis done by Claude.ai (Notion task).
# batch_min_avg_us: minimum across outer-batches of per-batch iteration avg
# batch_min_energy_per_iter_uj: matching energy per iteration
GT_TIME_COL = "batch_min_avg_us"
GT_ENERGY_COL = "batch_min_energy_per_iter_uj"


def load_task15_rows(csv_path, tc_path, min_wall_s=0.005):
    """Load rows using Task 15 / Draft v3 GT conventions.

    GT time  = batch_min_avg_us
    GT energy = batch_min_energy_per_iter_uj
    """
    with open(tc_path) as f:
        tc_data = json.load(f)
    cases_list = tc_data["cases"] if isinstance(tc_data, dict) else tc_data
    tp_order_map = {}
    for i, tc in enumerate(cases_list, start=1):
        tp_order_map[i] = tc["levels"][0].get("tpOrder", [2, 0, 1])[0]

    rows = []
    import csv as _csv
    with open(csv_path) as f:
        for row in _csv.DictReader(CommentFilterFile(f)):
            if row["status"] != "PASS":
                continue
            gt_e = float(row.get(GT_ENERGY_COL, 0) or 0)
            gt_t = float(row.get(GT_TIME_COL, 0) or 0)
            if gt_e <= 0 or gt_t <= 0:
                continue
            wall_s = float(row.get("wall_elapsed_s", -1))
            if wall_s <= 0:
                wall_s = float(row.get("batch_best_wall_s", -1))
            if wall_s < min_wall_s:
                continue
            case_idx = int(row["case_index"])
            r = EnergyRow(
                case_index=case_idx,
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
                num_cores=int(row["numSpm"]),
                tp_order_inner=tp_order_map.get(case_idx, 2),
                min_us=gt_t,   # Task 15: GT time = batch_min_avg_us (stored here)
                avg_iter_us=float(row.get("avg_us", 0)),
                npu_per_iter_uj=gt_e,  # Task 15: GT energy = batch_min_energy_per_iter_uj
                wall_s=wall_s,
            )
            rows.append(r)
    return rows


def compute_task15_features(rows, coeffs):
    """Compute features for Task 15 EDP evaluation.

    Predicts T with Candidate A DMA-Refined and returns feature dict
    containing both predicted time and GT energy (batch_min_*).
    """
    n = len(rows)
    macs = np.zeros(n)
    data_bytes = np.zeros(n)
    t_total_cy = np.zeros(n)
    n_cores_arr = np.zeros(n)
    tp_total_arr = np.zeros(n)
    d_total_arr = np.zeros(n)
    gt_uj = np.zeros(n)

    for i, r in enumerate(rows):
        op = r.make_op()
        cand = r.make_cand()
        nc = r.n_cores
        macs[i] = r.M * r.K * r.N
        db = total_data_bytes(op, cand, r.tp_order_inner)
        data_bytes[i] = db
        n_cores_arr[i] = nc
        tp_total_arr[i] = r.tp_total
        gt_uj[i] = r.npu_per_iter_uj

        n_ev, n_re = _models.dma_ops_decomposed(
            r.SPm, r.SPn, nc, r.tp_order_inner)
        tp_inn = _models.tp_inner_value(r.TPm, r.TPk, r.TPn, r.tp_order_inner)
        d_total_arr[i] = n_ev * r.tp_total + n_re * (r.tp_total / tp_inn)

        # Predicted T (Candidate A DMA-Refined)
        t_comp = macs[i] / (nc * coeffs.eff_macs)
        t_comm = db / coeffs.bw_eff_bpc
        t_ovh = (coeffs.l_sync_cy * r.tp_total
                 + coeffs.l_core_cy * nc * r.tp_total
                 + coeffs.l_dma_cy * d_total_arr[i]
                 + coeffs.l_startup_cy)
        t_total_cy[i] = t_comp + t_comm + t_ovh

    return {
        "macs": macs, "data_bytes": data_bytes,
        "t_total_cy": t_total_cy, "n_cores": n_cores_arr,
        "tp_total": tp_total_arr, "d_total": d_total_arr,
        "gt_uj": gt_uj,
    }

# ============================================================
# Model coefficients (calibration.json v15, fitted on result_v14_clean.csv)
# ============================================================

# Performance model: DMA-Refined (matches calibration.json v15)
NOTION_PERF = {
    "eff_macs": 24.28,
    "eff_bw": 4.0,
    "L_SYNC": 6622,
    "L_CORE": 1272,
    "L_DMA": 1788,
    "L_STARTUP": 53599,
    "clock_mhz": 1500,
}

# Energy model: 1-G, 3-parameter (P_CORE=40mW fixed)
# E = (P_BASE + P_CORE*P) * T_pred + E_DMA * D_total
NOTION_ENERGY = {
    "P_CORE_mW": 40.0,        # fixed (architecture specification)
    "E_DMA_uJ": 18.07,        # calibrated on v14
    "P_BASE_W": 8.64,         # calibrated on v14
}

# Target workloads for re-measurement (Regret > 10%)
TARGET_WORKLOADS = [
    (768, 768, 768),
    (1024, 1024, 1024),
    (256, 1024, 1024),
    (256, 64, 256),
    (128, 768, 3072),
    (32, 768, 768),
    (32, 32, 32),
]


# ============================================================
# 1-G Energy model prediction (using Notion coefficients)
# ============================================================

def predict_1g_energy(feat):
    """Predict energy using 1-G model with Notion-specified coefficients.

    E(uJ) = (P_BASE + P_CORE*P) * T_total(us) + E_DMA * D_total
    """
    p_base_uw = NOTION_ENERGY["P_BASE_W"] * 1e6
    p_core_uw = NOTION_ENERGY["P_CORE_mW"] * 1e3
    e_dma_uj = NOTION_ENERGY["E_DMA_uJ"]

    t_us = feat["t_total_cy"] / CLOCK_MHZ
    n_cores = feat["n_cores"]
    d_total = feat["d_total"]

    # E(pJ) = (p_base_uw + p_core_uw * P) * t_us + e_dma_uj * D_total * 1e6
    e_pj = (p_base_uw + p_core_uw * n_cores) * t_us + e_dma_uj * d_total * 1e6
    return e_pj / 1e6  # return uJ


# ============================================================
# Phase 0: Model version verification
# ============================================================

def run_phase0(calib_path):
    """Verify Notion-specified coefficients against calibration.json."""
    print("=" * 70)
    print("PHASE 0: MODEL VERSION VERIFICATION")
    print("=" * 70)

    coeffs = load_calibration(Path(calib_path))

    # --- Performance model ---
    print("\n  [Performance Model: Candidate A, DMA-Refined]")
    perf_checks = [
        ("eff_macs", NOTION_PERF["eff_macs"], coeffs.eff_macs),
        ("eff_bw (bw_eff_bpc)", NOTION_PERF["eff_bw"], coeffs.bw_eff_bpc),
        ("L_SYNC (l_sync_cy)", NOTION_PERF["L_SYNC"], coeffs.l_sync_cy),
        ("L_CORE (l_core_cy)", NOTION_PERF["L_CORE"], coeffs.l_core_cy),
        ("L_DMA (l_dma_cy)", NOTION_PERF["L_DMA"], coeffs.l_dma_cy),
        ("L_STARTUP (l_startup_cy)", NOTION_PERF["L_STARTUP"], coeffs.l_startup_cy),
    ]
    # clock_mhz is a constant (1500), not stored in CalibCoeffs

    perf_ok = True
    print(f"  {'Parameter':<30} {'Notion':>12} {'Code':>12} {'Match':>7}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*7}")
    for name, notion_val, code_val in perf_checks:
        match = abs(notion_val - code_val) < 0.01
        if not match:
            perf_ok = False
        print(f"  {name:<30} {notion_val:>12.2f} {code_val:>12.2f} "
              f"{'OK' if match else 'MISMATCH':>7}")

    print(f"\n  Performance model: {'ALL MATCH' if perf_ok else 'MISMATCH DETECTED'}")

    # --- Energy model ---
    print(f"\n  [Energy Model: 1-G, 3-parameter]")
    print(f"  Notion formula: E = (P_BASE + P_CORE*P) * T + E_DMA * D_total")

    # Check what's in calibration.json
    with open(calib_path) as f:
        calib_data = json.load(f)

    energy_section = calib_data.get("energy", {})
    energy_model_name = energy_section.get("model", "unknown")
    energy_params = energy_section.get("params", {})

    print(f"\n  calibration.json energy model: {energy_model_name}")
    print(f"  Notion energy model: 1-G (3-parameter, T-3 with P_CORE=40mW fixed)")

    if energy_model_name != "1-G":
        print(f"\n  ** STRUCTURE MISMATCH: calibration.json uses '{energy_model_name}' "
              f"(not '1-G')")

    energy_checks = [
        ("P_CORE (mW)", NOTION_ENERGY["P_CORE_mW"],
         energy_params.get("p_core_uw", 0) / 1000),
        ("E_DMA (uJ)", NOTION_ENERGY["E_DMA_uJ"],
         energy_params.get("e_dma_uj", 0)),
        ("P_BASE (W)", NOTION_ENERGY["P_BASE_W"],
         energy_params.get("p_base_uw", energy_params.get("p_sys_uw", 0)) / 1e6),
    ]

    energy_ok = True
    print(f"\n  {'Parameter':<20} {'Notion':>12} {'Code':>12} {'Match':>7}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*7}")
    for name, notion_val, code_val in energy_checks:
        match = abs(notion_val - code_val) < 0.01
        if not match:
            energy_ok = False
        print(f"  {name:<20} {notion_val:>12.4f} {code_val:>12.4f} "
              f"{'OK' if match else 'MISMATCH':>7}")

    print(f"\n  Energy model: {'ALL MATCH' if energy_ok else 'MISMATCH DETECTED'}")

    if not energy_ok:
        print(f"\n  ** ACTION: Using Notion-specified 1-G coefficients directly")
        print(f"    P_BASE = {NOTION_ENERGY['P_BASE_W']} W")
        print(f"    P_CORE = {NOTION_ENERGY['P_CORE_mW']} mW (fixed)")
        print(f"    E_DMA  = {NOTION_ENERGY['E_DMA_uJ']} uJ")
        print(f"    (calibration.json retains existing model; "
              f"1-G coefficients used in evaluation only)")

    # --- N_dma structure check ---
    print(f"\n  [DMA Count Structure]")
    perf_model_type = calib_data.get("model", "unknown")
    uses_d_total = "DMA-Refined" in perf_model_type
    print(f"  calibration.json perf model: {perf_model_type}")
    print(f"  Uses D_total (DMA-Refined): {'YES' if uses_d_total else 'NO'}")
    print(f"  Notion N_dma = D_total (N_every*TP + N_reused*(TP/TP_inner)): YES")
    dma_ok = uses_d_total
    print(f"  DMA structure: {'MATCH' if dma_ok else 'MISMATCH'}")

    all_ok = perf_ok and energy_ok and dma_ok
    return {
        "perf_match": perf_ok,
        "energy_match": energy_ok,
        "dma_match": dma_ok,
        "all_match": all_ok,
        "energy_model_in_code": energy_model_name,
        "action": "none" if all_ok else "using Notion 1-G coefficients directly",
    }


# ============================================================
# Phase 1-prep: Filter tc_list for target workloads
# ============================================================

def run_phase1_prep(tc_path, output_path=None):
    """Filter tc_list_v11.json to include only target workload cases."""
    print("\n" + "=" * 70)
    print("PHASE 1-PREP: FILTER TC_LIST FOR TARGET WORKLOADS")
    print("=" * 70)

    with open(tc_path) as f:
        tc_data = json.load(f)

    metadata = tc_data.get("metadata", {})
    cases = tc_data["cases"]
    total = len(cases)

    target_set = set(TARGET_WORKLOADS)
    filtered = []
    for case in cases:
        key = (case["M"], case["K"], case["N"])
        if key in target_set:
            filtered.append(case)

    # Count per workload
    counts = defaultdict(int)
    for case in filtered:
        counts[(case["M"], case["K"], case["N"])] += 1

    print(f"\n  Source: {tc_path} ({total} total cases)")
    print(f"  Target workloads: {len(TARGET_WORKLOADS)}")
    print(f"\n  {'Workload':<20} {'Cases':>6}")
    print(f"  {'-'*20} {'-'*6}")
    for wl in TARGET_WORKLOADS:
        print(f"  {wl[0]}x{wl[1]}x{wl[2]:<10} {counts.get(wl, 0):>6}")
    print(f"  {'TOTAL':<20} {len(filtered):>6}")

    if output_path is None:
        output_path = str(Path(tc_path).parent / "tc_list_v13_target.json")

    output_data = {
        "metadata": {
            **metadata,
            "filtered_for": "task15_remeasure",
            "target_workloads": [f"{m}x{k}x{n}" for m, k, n in TARGET_WORKLOADS],
            "source_tc_list": str(Path(tc_path).name),
        },
        "cases": filtered,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Output: {output_path} ({len(filtered)} cases)")
    return output_path


# ============================================================
# Phase 1-post: Merge re-measured results
# ============================================================

def run_phase1_post(v12_path, remeasured_path, output_path=None):
    """Merge re-measured 7-workload data into v12 to create v13."""
    print("\n" + "=" * 70)
    print("PHASE 1-POST: MERGE RE-MEASURED RESULTS INTO V13")
    print("=" * 70)

    target_set = set(TARGET_WORKLOADS)

    if output_path is None:
        output_path = str(Path(v12_path).parent / "result_v13.csv")

    # Read v12 rows (keep non-target workloads)
    v12_rows = []
    v12_header_comments = []
    v12_header = None
    with open(v12_path) as f:
        for line in f:
            if line.startswith("#"):
                v12_header_comments.append(line.rstrip())
                continue
            if v12_header is None:
                v12_header = line.rstrip()
                continue
            parts = line.strip().split(",")
            if len(parts) >= 13:
                try:
                    m, k, n = int(parts[10]), int(parts[11]), int(parts[12])
                    if (m, k, n) not in target_set:
                        v12_rows.append(line.rstrip())
                except ValueError:
                    v12_rows.append(line.rstrip())

    # Read re-measured rows (all are target workloads)
    remeasured_rows = []
    remeasured_header = None
    with open(remeasured_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            if remeasured_header is None:
                remeasured_header = line.rstrip()
                continue
            remeasured_rows.append(line.rstrip())

    print(f"  v12 non-target rows: {len(v12_rows)}")
    print(f"  Re-measured rows: {len(remeasured_rows)}")
    print(f"  Total v13 rows: {len(v12_rows) + len(remeasured_rows)}")

    # Check header compatibility
    if v12_header != remeasured_header:
        print(f"\n  WARNING: Header mismatch between v12 and re-measured CSV")
        print(f"    v12 columns: {len(v12_header.split(','))}")
        print(f"    remeasured columns: {len(remeasured_header.split(','))}")

    # Write merged result
    with open(output_path, "w") as f:
        # Metadata comments
        f.write(f"# tc_list: tc_list_v11.json\n")
        f.write(f"# source: result_v12.csv (12 workloads) + "
                f"result_v13_remeasured.csv (7 workloads)\n")
        f.write(f"# note: Task 15 selective re-measurement merge\n")
        # Use the header with most columns
        header = v12_header if v12_header else remeasured_header
        f.write(header + "\n")
        for row in v12_rows:
            f.write(row + "\n")
        for row in remeasured_rows:
            f.write(row + "\n")

    print(f"  Output: {output_path}")
    return output_path


# ============================================================
# Phase 2: EDP re-evaluation (v12 vs v13 comparison)
# ============================================================

def compute_edp_per_workload(rows, feat, gt_uj):
    """Compute EDP metrics per workload.

    Returns dict: {(M,K,N): {gt_cores, pred_cores, regret, gt_edp_best, ...}}
    """
    pred_e_uj = predict_1g_energy(feat)
    pred_t_us = feat["t_total_cy"] / CLOCK_MHZ
    pred_edp = pred_t_us * pred_e_uj

    gt_t_us = np.array([r.min_us for r in rows])
    gt_edp = gt_t_us * gt_uj

    # Global metrics
    global_rho = spearman_rank_correlation(gt_edp.tolist(), pred_edp.tolist())

    # Group by problem size
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.M, r.K, r.N)].append(i)

    per_size = {}
    correct, total = 0, 0
    regrets = []

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
        regret = max(0, regret)
        regrets.append(regret)

        per_size[key] = {
            "gt_cores": gt_cores,
            "pred_cores": pred_cores,
            "regret_pct": round(regret, 1),
            "match": gt_cores == pred_cores,
            "n_cases": len(idx),
        }

    le10 = sum(1 for r in regrets if r <= 10)
    summary = {
        "core_accuracy": f"{correct}/{total}",
        "regret_mean": round(float(np.mean(regrets)), 1) if regrets else 0,
        "regret_max": round(float(np.max(regrets)), 1) if regrets else 0,
        "regret_le10": f"{le10}/{total}",
        "global_rho": round(global_rho, 4) if global_rho else 0,
    }

    return per_size, summary


def run_phase2(v12_path, v13_path, tc_path, calib_path):
    """Phase 2: EDP comparison between v12 and v13.

    Uses batch_min_avg_us + batch_min_energy_per_iter_uj as GT
    (matches Draft v3 Claude.ai analysis convention for Task 15).
    """
    print("\n" + "=" * 70)
    print("PHASE 2: EDP RE-EVALUATION (V12 vs V13)")
    print(f"  GT convention: T={GT_TIME_COL}, E={GT_ENERGY_COL}")
    print("=" * 70)

    coeffs = load_calibration(Path(calib_path))

    # Load both datasets
    print(f"\n  Loading v12: {v12_path}")
    rows_v12 = load_task15_rows(v12_path, tc_path)
    feat_v12 = compute_task15_features(rows_v12, coeffs)
    gt_uj_v12 = feat_v12["gt_uj"]
    print(f"    {len(rows_v12)} valid samples")

    print(f"  Loading v13: {v13_path}")
    rows_v13 = load_task15_rows(v13_path, tc_path)
    feat_v13 = compute_task15_features(rows_v13, coeffs)
    gt_uj_v13 = feat_v13["gt_uj"]
    print(f"    {len(rows_v13)} valid samples")

    # Compute EDP per workload
    per_v12, summary_v12 = compute_edp_per_workload(rows_v12, feat_v12, gt_uj_v12)
    per_v13, summary_v13 = compute_edp_per_workload(rows_v13, feat_v13, gt_uj_v13)

    target_set = set(TARGET_WORKLOADS)

    # --- Per-workload comparison table ---
    print(f"\n  {'='*90}")
    print(f"  PER-WORKLOAD EDP COMPARISON (v12 vs v13)")
    print(f"  {'='*90}")
    print(f"  {'Workload':<20} {'v12 GT P*':>9} {'v12 Reg%':>9} "
          f"{'v13 GT P*':>9} {'v13 Reg%':>9} {'GT Chg':>7} {'Reg Chg':>9} "
          f"{'Target':>7}")
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*9} {'-'*7}")

    all_sizes = sorted(set(list(per_v12.keys()) + list(per_v13.keys())))
    comparison_rows = []

    for key in all_sizes:
        size_str = f"{key[0]}x{key[1]}x{key[2]}"
        is_target = key in target_set

        d12 = per_v12.get(key, {})
        d13 = per_v13.get(key, {})

        gt12 = d12.get("gt_cores", "-")
        reg12 = d12.get("regret_pct", "-")
        gt13 = d13.get("gt_cores", "-")
        reg13 = d13.get("regret_pct", "-")

        gt_changed = ""
        reg_delta = ""
        if isinstance(gt12, int) and isinstance(gt13, int):
            gt_changed = "YES" if gt12 != gt13 else "no"
        if isinstance(reg12, (int, float)) and isinstance(reg13, (int, float)):
            delta = reg13 - reg12
            reg_delta = f"{delta:+.1f}%p"

        gt12_s = str(gt12) if gt12 != "-" else "-"
        reg12_s = f"{reg12:.1f}%" if isinstance(reg12, (int, float)) else "-"
        gt13_s = str(gt13) if gt13 != "-" else "-"
        reg13_s = f"{reg13:.1f}%" if isinstance(reg13, (int, float)) else "-"
        tgt_s = "RE-MEAS" if is_target else ""

        print(f"  {size_str:<20} {gt12_s:>9} {reg12_s:>9} "
              f"{gt13_s:>9} {reg13_s:>9} {gt_changed:>7} {reg_delta:>9} "
              f"{tgt_s:>7}")

        comparison_rows.append({
            "workload": size_str,
            "v12_gt_cores": gt12,
            "v12_regret": reg12,
            "v13_gt_cores": gt13,
            "v13_regret": reg13,
            "gt_changed": gt_changed,
            "regret_delta": round(delta, 1) if isinstance(reg12, (int, float)) and isinstance(reg13, (int, float)) else None,
            "is_target": is_target,
        })

    # --- Non-target sanity check ---
    print(f"\n  [Sanity Check: Non-target workloads]")
    non_target_ok = True
    for row in comparison_rows:
        if not row["is_target"] and row["regret_delta"] is not None:
            if abs(row["regret_delta"]) > 0.05:
                print(f"    WARNING: {row['workload']} regret changed by "
                      f"{row['regret_delta']:+.1f}%p (expected no change)")
                non_target_ok = False
    if non_target_ok:
        print(f"    All 12 non-target workloads: regret unchanged (OK)")

    # --- Summary comparison ---
    print(f"\n  {'='*60}")
    print(f"  EDP SUMMARY COMPARISON")
    print(f"  {'='*60}")
    print(f"  {'Metric':<25} {'v12':>12} {'v13':>12} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    metrics_pairs = [
        ("Core Accuracy", summary_v12["core_accuracy"], summary_v13["core_accuracy"]),
        ("Regret mean", f"{summary_v12['regret_mean']:.1f}%",
         f"{summary_v13['regret_mean']:.1f}%"),
        ("Regret max", f"{summary_v12['regret_max']:.1f}%",
         f"{summary_v13['regret_max']:.1f}%"),
        ("Regret <=10%", summary_v12["regret_le10"], summary_v13["regret_le10"]),
        ("Global rho", f"{summary_v12['global_rho']:.4f}",
         f"{summary_v13['global_rho']:.4f}"),
    ]

    for name, v12_val, v13_val in metrics_pairs:
        delta = ""
        # Compute numeric delta for regret mean/max
        if "mean" in name.lower() or "max" in name.lower():
            v12_num = float(str(v12_val).replace("%", ""))
            v13_num = float(str(v13_val).replace("%", ""))
            delta = f"{v13_num - v12_num:+.1f}%p"
        print(f"  {name:<25} {str(v12_val):>12} {str(v13_val):>12} {delta:>10}")

    # --- GT change analysis ---
    print(f"\n  [GT P* Change Analysis (re-measured workloads)]")
    gt_changed_count = 0
    for row in comparison_rows:
        if row["is_target"]:
            if row["gt_changed"] == "YES":
                gt_changed_count += 1
                print(f"    {row['workload']}: GT P* {row['v12_gt_cores']} -> "
                      f"{row['v13_gt_cores']} (measurement noise likely)")
            else:
                print(f"    {row['workload']}: GT P* unchanged at "
                      f"{row['v12_gt_cores']}")

    if gt_changed_count > 0:
        print(f"\n    {gt_changed_count}/7 workloads had GT P* change "
              f"-> measurement noise confirmed")
    else:
        print(f"\n    0/7 workloads had GT P* change -> noise is NOT the cause")

    return {
        "per_workload": comparison_rows,
        "v12_summary": summary_v12,
        "v13_summary": summary_v13,
        "gt_changed_count": gt_changed_count,
        "non_target_sanity": non_target_ok,
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Task 15: High-Regret Re-measurement & EDP Re-evaluation")
    ap.add_argument("--result-v12", default="out/reports/result_v12.csv")
    ap.add_argument("--result-v13", default="out/reports/result_v13.csv")
    ap.add_argument("--result-remeasured", default="out/reports/result_v13_remeasured.csv")
    ap.add_argument("--tc", default="out/tc_list_v11.json")
    ap.add_argument("--calib", default="data/calibration.json")
    ap.add_argument("--phase", required=True,
                    choices=["0", "1-prep", "1-post", "2", "all"],
                    help="Phase to run")
    ap.add_argument("--output", default="out/reports/task15_results.json")
    args = ap.parse_args()

    results = {}

    if args.phase in ("0", "all"):
        results["phase0"] = run_phase0(args.calib)

    if args.phase in ("1-prep", "all"):
        tc_out = run_phase1_prep(args.tc)
        results["phase1_prep"] = {"output": tc_out}

    if args.phase == "1-post":
        merged = run_phase1_post(args.result_v12, args.result_remeasured)
        results["phase1_post"] = {"output": merged}

    if args.phase in ("2", "all"):
        results["phase2"] = run_phase2(
            args.result_v12, args.result_v13, args.tc, args.calib)

    # Save results
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
