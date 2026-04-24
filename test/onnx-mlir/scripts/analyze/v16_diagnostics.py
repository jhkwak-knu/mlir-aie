#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v16_diagnostics.py -- Phase 0: Error pattern analysis for v15 cost model.

Analyzes where and why the current performance/energy models fail,
to guide candidate formula design for v16.

Usage:
    python3 scripts/analyze/v16_diagnostics.py \
        --result out/reports/result_v14_clean.csv \
        --tc out/tc_list_v14.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

# Resolve project paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "generate"))
from models import (  # noqa: E402
    CLOCK_MHZ, PerfModel, EnergyModel,
    d_total, total_data_bytes, dma_ops_per_step,
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
)
from tiling_common import load_calibration  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EFF_MACS = 24.28       # Fixed from trace measurement
BW_BPC = 4.0           # Current baseline fixed BW
ELEM_BYTES = 2         # bf16


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class DiagRow:
    """One measurement case with all fields needed for diagnostics."""
    case_index: int
    SPm: int
    SPn: int
    TPm: int
    TPk: int
    TPn: int
    TM: int
    TK: int
    TN: int
    M: int
    K: int
    N: int
    tp_order: List[int]
    # Ground truth
    gt_time_us: float
    gt_energy_uj: float

    @property
    def n_cores(self) -> int:
        return self.SPm * self.SPn

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn

    @property
    def macs(self) -> int:
        # M*K*N ops (not *2); eff_macs calibrated with same convention
        return self.M * self.K * self.N

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.K}x{self.N}"

    @property
    def num_cols(self) -> int:
        tiles_per_col = 8
        return math.ceil(self.n_cores / tiles_per_col)


def load_data(csv_path: Path, tc_path: Path) -> List[DiagRow]:
    """Load CSV + tc_list, merge on case_index."""
    with tc_path.open("r", encoding="utf-8") as f:
        tc_doc = json.load(f)
    tc_cases = tc_doc["cases"]

    rows: List[DiagRow] = []
    with csv_path.open("r", encoding="utf-8") as f:
        # Skip comment lines (start with #)
        lines = [line for line in f if not line.startswith("#")]
    import io
    with io.StringIO("".join(lines)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "PASS":
                continue
            idx = int(row["case_index"])
            tc = tc_cases[idx - 1]
            lvl = tc["levels"][0]

            gt_time = float(row.get("batch_min_avg_us", "0"))
            gt_energy = float(row.get("batch_min_energy_per_iter_uj", "0"))
            if gt_time <= 0 or gt_energy <= 0:
                continue

            rows.append(DiagRow(
                case_index=idx,
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                tp_order=lvl["tpOrder"],
                gt_time_us=gt_time,
                gt_energy_uj=gt_energy,
            ))
    return rows


# ---------------------------------------------------------------------------
# Prediction with current v15 model
# ---------------------------------------------------------------------------
def predict_perf(r: DiagRow, calib: dict) -> Tuple[float, float, float]:
    """Predict T using DMA-Refined v15. Returns (T_comp_cy, T_comm_cy, T_overhead_cy)."""
    macs = r.macs
    data_bytes = total_data_bytes(
        r.M, r.K, r.N, ELEM_BYTES,
        r.SPm, r.SPn, r.TPm, r.TPk, r.TPn, r.tp_order[0])
    d_tot = d_total(
        r.SPm, r.SPn, r.n_cores,
        r.TPm, r.TPk, r.TPn, r.tp_total, r.tp_order[0])

    t_comp, t_comm, t_overhead = PerfModel.components_dma_refined(
        macs=macs, data_bytes=data_bytes,
        n_cores=r.n_cores, tp_total=r.tp_total, d_total_val=d_tot,
        eff_macs=calib["eff_macs"], bw_bpc=calib["bw_eff_bpc"],
        l_sync=calib["l_sync_cy"], l_pe=calib["l_pe_cy"],
        l_dma=calib["l_dma_cy"], l_startup=calib["l_startup_cy"],
    )
    return t_comp, t_comm, t_overhead


def predict_energy(r: DiagRow, t_pred_us: float, calib: dict) -> float:
    """Predict energy using 1-G model."""
    d_tot = d_total(
        r.SPm, r.SPn, r.n_cores,
        r.TPm, r.TPk, r.TPn, r.tp_total, r.tp_order[0])
    e_params = calib["energy"]["params"]
    p_base = e_params["p_base_uw"]   # uW
    p_pe = e_params["p_pe_uw"]       # uW
    e_dma = e_params["e_dma_uj"]     # uJ

    # E = (P_BASE + P_PE*P) * T(us) + E_DMA * D_total
    # uW * us = pJ -> convert to uJ
    e_pj = (p_base + p_pe * r.n_cores) * t_pred_us
    e_uj = e_pj / 1e6 + e_dma * d_tot
    return e_uj


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def mape_p90(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.percentile(np.abs((actual - pred) / actual) * 100, 90))


def signed_error_pct(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean signed error: positive = over-prediction."""
    return float(np.mean((pred - actual) / actual) * 100)


# ---------------------------------------------------------------------------
# Grouping and analysis
# ---------------------------------------------------------------------------
def analyze_by_group(
    rows: List[DiagRow],
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    key_fn,
    label: str,
):
    """Print MAPE/bias breakdown by group key."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[key_fn(r)].append(i)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'Group':<25} {'Count':>6} {'MAPE':>8} {'P90':>8} {'Bias':>8}")
    print(f"{'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    sorted_keys = sorted(groups.keys(), key=lambda k: str(k))
    for key in sorted_keys:
        idxs = groups[key]
        a = gt_arr[idxs]
        p = pred_arr[idxs]
        m = mape(a, p)
        p90 = mape_p90(a, p)
        bias = signed_error_pct(a, p)
        print(f"{str(key):<25} {len(idxs):>6} {m:>7.1f}% {p90:>7.1f}% {bias:>+7.1f}%")


def residual_correlations(
    rows: List[DiagRow],
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
):
    """Compute correlation of residual with various features."""
    residual = (gt_arr - pred_arr)  # in same units

    features = {
        "TP_total": np.array([r.tp_total for r in rows], dtype=float),
        "P (cores)": np.array([r.n_cores for r in rows], dtype=float),
        "P*TP": np.array([r.n_cores * r.tp_total for r in rows], dtype=float),
        "P^2": np.array([r.n_cores ** 2 for r in rows], dtype=float),
        "P^2*TP": np.array([r.n_cores ** 2 * r.tp_total for r in rows], dtype=float),
        "num_cols": np.array([r.num_cols for r in rows], dtype=float),
        "num_cols*TP": np.array([r.num_cols * r.tp_total for r in rows], dtype=float),
        "D_total": np.array([
            d_total(r.SPm, r.SPn, r.n_cores, r.TPm, r.TPk, r.TPn,
                    r.tp_total, r.tp_order[0])
            for r in rows], dtype=float),
        "data_bytes": np.array([
            total_data_bytes(r.M, r.K, r.N, ELEM_BYTES,
                             r.SPm, r.SPn, r.TPm, r.TPk, r.TPn, r.tp_order[0])
            for r in rows], dtype=float),
        "MACs": np.array([r.macs for r in rows], dtype=float),
    }

    print(f"\n{'='*70}")
    print(f"  Residual Correlations (residual = GT - pred)")
    print(f"{'='*70}")
    print(f"{'Feature':<20} {'Pearson r':>10} {'Spearman rho':>14}")
    print(f"{'-'*20} {'-'*10} {'-'*14}")

    for name, feat in features.items():
        pearson_r = float(np.corrcoef(residual, feat)[0, 1])
        sp_rho, _ = spearmanr(residual, feat)
        print(f"{name:<20} {pearson_r:>+10.4f} {sp_rho:>+14.4f}")


def component_analysis(
    rows: List[DiagRow],
    calib: dict,
):
    """Analyze T_comp/T_comm/T_overhead contribution ratios."""
    ratios_comp = []
    ratios_comm = []
    ratios_overhead = []

    for r in rows:
        t_comp, t_comm, t_overhead = predict_perf(r, calib)
        t_total = t_comp + t_comm + t_overhead
        if t_total > 0:
            ratios_comp.append(t_comp / t_total)
            ratios_comm.append(t_comm / t_total)
            ratios_overhead.append(t_overhead / t_total)

    rc = np.array(ratios_comp) * 100
    rm = np.array(ratios_comm) * 100
    ro = np.array(ratios_overhead) * 100

    print(f"\n{'='*70}")
    print(f"  Component Contribution (% of T_total)")
    print(f"{'='*70}")
    print(f"{'Component':<15} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, arr in [("T_comp", rc), ("T_comm", rm), ("T_overhead", ro)]:
        print(f"{name:<15} {np.mean(arr):>7.1f}% {np.median(arr):>7.1f}% "
              f"{np.min(arr):>7.1f}% {np.max(arr):>7.1f}%")


def energy_oracle_test(
    rows: List[DiagRow],
    calib: dict,
):
    """Test energy model accuracy with T_meas vs T_pred."""
    gt_e = []
    pred_e_tpred = []
    pred_e_tmeas = []

    for r in rows:
        t_comp, t_comm, t_overhead = predict_perf(r, calib)
        t_pred_us = (t_comp + t_comm + t_overhead) / CLOCK_MHZ

        e_pred_tpred = predict_energy(r, t_pred_us, calib)
        e_pred_tmeas = predict_energy(r, r.gt_time_us, calib)

        gt_e.append(r.gt_energy_uj)
        pred_e_tpred.append(e_pred_tpred)
        pred_e_tmeas.append(e_pred_tmeas)

    gt_e = np.array(gt_e)
    pred_tpred = np.array(pred_e_tpred)
    pred_tmeas = np.array(pred_e_tmeas)

    print(f"\n{'='*70}")
    print(f"  Energy Oracle Test (T_pred vs T_meas)")
    print(f"{'='*70}")
    print(f"{'Input':<15} {'MAPE':>8} {'P90':>8} {'rho':>8} {'Bias':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, pred in [("T_pred (v15)", pred_tpred), ("T_meas (oracle)", pred_tmeas)]:
        m = mape(gt_e, pred)
        p90 = mape_p90(gt_e, pred)
        rho, _ = spearmanr(gt_e, pred)
        bias = signed_error_pct(gt_e, pred)
        print(f"{name:<15} {m:>7.1f}% {p90:>7.1f}% {rho:>7.4f} {bias:>+7.1f}%")

    print(f"\n  -> If oracle MAPE << T_pred MAPE, energy error is dominated by T_pred error.")
    print(f"  -> If oracle MAPE is similar, energy formula itself needs improvement.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="v16 Phase 0: Error diagnostics")
    parser.add_argument("--result", required=True, help="Path to result_v14_clean.csv")
    parser.add_argument("--tc", required=True, help="Path to tc_list_v14.json")
    parser.add_argument("--calib", default=None, help="Path to calibration.json (default: data/calibration.json)")
    args = parser.parse_args()

    csv_path = Path(args.result)
    tc_path = Path(args.tc)
    calib_path = Path(args.calib) if args.calib else SCRIPT_DIR.parent.parent / "data" / "calibration.json"

    # Load calibration
    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)

    # Load data
    rows = load_data(csv_path, tc_path)
    print(f"Loaded {len(rows)} PASS cases from {csv_path.name}")

    # Compute predictions
    gt_time = np.array([r.gt_time_us for r in rows])
    pred_time_us = []
    for r in rows:
        t_comp, t_comm, t_overhead = predict_perf(r, calib)
        pred_time_us.append((t_comp + t_comm + t_overhead) / CLOCK_MHZ)
    pred_time = np.array(pred_time_us)

    # Overall metrics
    overall_mape = mape(gt_time, pred_time)
    overall_p90 = mape_p90(gt_time, pred_time)
    overall_rho, _ = spearmanr(gt_time, pred_time)
    overall_bias = signed_error_pct(gt_time, pred_time)

    print(f"\n{'='*70}")
    print(f"  Overall Performance Model Metrics (v15 Baseline)")
    print(f"{'='*70}")
    print(f"  MAPE:     {overall_mape:.1f}%")
    print(f"  P90:      {overall_p90:.1f}%")
    print(f"  Rho:      {overall_rho:.4f}")
    print(f"  Bias:     {overall_bias:+.1f}%")
    print(f"  Samples:  {len(rows)}")

    # Per-size analysis
    analyze_by_group(rows, gt_time, pred_time,
                     key_fn=lambda r: r.size_key,
                     label="Per-Size MAPE (Performance)")

    # Per-core analysis
    analyze_by_group(rows, gt_time, pred_time,
                     key_fn=lambda r: r.n_cores,
                     label="Per-Core-Count MAPE (Performance)")

    # Per-tpOrder analysis
    tp_label = {0: "M-inner", 1: "N-inner", 2: "K-inner"}
    analyze_by_group(rows, gt_time, pred_time,
                     key_fn=lambda r: tp_label.get(r.tp_order[0], f"?{r.tp_order[0]}"),
                     label="Per-tpOrder MAPE (Performance)")

    # Residual correlations (in cycles for scale consistency)
    gt_cy = gt_time * CLOCK_MHZ
    pred_cy = pred_time * CLOCK_MHZ
    residual_correlations(rows, gt_cy, pred_cy)

    # Component analysis
    component_analysis(rows, calib)

    # Energy oracle test
    energy_oracle_test(rows, calib)

    # Save structured results
    out_dir = csv_path.parent
    out_path = out_dir / "v16_diagnostics.json"
    results = {
        "overall": {
            "mape": round(overall_mape, 2),
            "p90": round(overall_p90, 2),
            "rho": round(overall_rho, 4),
            "bias": round(overall_bias, 2),
            "n_samples": len(rows),
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved diagnostics to {out_path}")


if __name__ == "__main__":
    main()
