#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_perf_model.py -- Systematic validation of the D+B performance cost model.

Produces a 9-section console report and optional matplotlib visualizations.
Validates prediction accuracy, per-size/per-core breakdowns, component dominance,
optimal config selection, error distribution, residual analysis, and worst cases.

Usage:
    python3 scripts/analyze/validate_perf_model.py \
        --csv out/calibration/result.csv \
        --tc out/calibration/tc_list.json \
        --calib data/calibration.json \
        [--ground-truth min_us] \
        [--plot out/calibration/plots]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import cost model functions
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate"))
from cost_model import (  # noqa: E402
    Candidate, perf_compute, perf_comm, perf_overhead, total_data_bytes,
)
from tiling_common import (  # noqa: E402
    CalibCoeffs, OpCase, load_calibration,
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
    CommentFilterFile,
)

# Import spearman from analyze_ranking (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

CLOCK_MHZ = 1500
TP_AXIS_NAMES = {0: "M", 1: "N", 2: "K"}


# ============================================================
# Data structures
# ============================================================
@dataclass
class ValidationRow:
    """Single validation case with predicted and actual values."""
    case_index: int
    # Tiling parameters
    num_cores: int
    SPm: int
    SPn: int
    TPm: int
    TPk: int
    TPn: int
    TM: int
    TK: int
    TN: int
    # Problem size
    M: int
    K: int
    N: int
    # tpOrder (innermost axis)
    tp_order: int
    # Predictions (cycles)
    t_comp: float
    t_comm: float
    t_overhead: float
    t_total_pred_cy: float
    # Predictions (us)
    t_total_pred_us: float
    # CSV recorded prediction (us)
    t_total_csv_us: float
    # Ground truth (us)
    actual_us: float

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.K}x{self.N}"

    @property
    def error_pct(self) -> float:
        """Signed percentage error: (pred - actual) / actual * 100."""
        if self.actual_us == 0:
            return float("inf")
        return (self.t_total_pred_us - self.actual_us) / self.actual_us * 100

    @property
    def ape(self) -> float:
        """Absolute percentage error."""
        return abs(self.error_pct)


# ============================================================
# Data loading
# ============================================================
def load_tp_order_map(tc_path: Path = None,
                      tc_archive_path: Path = None,
                      case_indices: list = None) -> Dict[int, int]:
    """Load case_index -> tpOrder[0] mapping.

    Sources (mutually exclusive):
      - tc_path: tc_list.json (1-based position in cases array)
      - tc_archive_path: archive directory with per-case tc.json
    """
    mapping: Dict[int, int] = {}
    if tc_archive_path and case_indices:
        for ci in case_indices:
            tc_file = tc_archive_path / f"case_{ci:03d}" / "tc.json"
            if tc_file.is_file():
                with tc_file.open("r", encoding="utf-8") as f:
                    tc = json.load(f)
                levels = tc.get("levels", [])
                if levels:
                    mapping[ci] = levels[0].get("tpOrder", [2, 0, 1])[0]
    elif tc_path:
        with tc_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        for i, case in enumerate(doc["cases"], start=1):
            levels = case.get("levels", [])
            if levels:
                tp_order = levels[0].get("tpOrder", [2, 0, 1])
                mapping[i] = tp_order[0]
    return mapping


def load_validation_data(
    csv_path: Path,
    tc_path: Path,
    coeffs: CalibCoeffs,
    ground_truth_col: str,
    min_quality: float = 0.0,
    tc_archive_path: Path = None,
) -> List[ValidationRow]:
    """Parse result.csv + tiling config and compute predictions."""
    # First pass: collect case indices for archive mode
    case_indices = None
    if tc_archive_path:
        with csv_path.open("r", encoding="utf-8") as f:
            case_indices = [int(r["case_index"]) for r in csv.DictReader(CommentFilterFile(f))
                           if r.get("status") == "PASS"]
    tp_map = load_tp_order_map(tc_path, tc_archive_path, case_indices)

    rows: List[ValidationRow] = []
    skipped = 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(CommentFilterFile(f))
        for row in reader:
            if row.get("status") != "PASS":
                continue
            dq = float(row.get("data_quality", "1.0"))
            if dq < min_quality:
                skipped += 1
                continue

            case_idx = int(row["case_index"])
            gt_val = row.get(ground_truth_col, "")
            t_pred_csv = row.get("t_total_pred", "")
            if not gt_val or gt_val == "-" or not t_pred_csv or t_pred_csv == "-":
                continue

            actual_us = float(gt_val)
            t_csv_cy = float(t_pred_csv)
            if actual_us <= 0:
                continue

            M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
            SPm, SPn = int(row["SPm"]), int(row["SPn"])
            TPm, TPk, TPn = int(row["TPm"]), int(row["TPk"]), int(row["TPn"])
            TM, TK, TN = int(row["TM"]), int(row["TK"]), int(row["TN"])
            num_cores = int(row["numSpm"])

            tp_order = tp_map.get(case_idx, TP_AXIS_K)

            # Reconstruct Candidate and OpCase for prediction
            op = OpCase(M=M, K=K, N=N, elem_type="bf16")
            cand = Candidate(
                num_cores=num_cores, num_columns=0,
                SPm=SPm, SPn=SPn,
                TPm=TPm, TPk=TPk, TPn=TPn,
                TM=TM, TK=TK, TN=TN,
            )

            t_comp = perf_compute(op, cand, coeffs)
            t_comm = perf_comm(op, cand, tp_order, coeffs)
            t_over = perf_overhead(cand, coeffs, tp_order)
            t_total_cy = t_comp + t_comm + t_over
            t_total_us = t_total_cy / CLOCK_MHZ

            rows.append(ValidationRow(
                case_index=case_idx,
                num_cores=num_cores,
                SPm=SPm, SPn=SPn,
                TPm=TPm, TPk=TPk, TPn=TPn,
                TM=TM, TK=TK, TN=TN,
                M=M, K=K, N=N,
                tp_order=tp_order,
                t_comp=t_comp, t_comm=t_comm, t_overhead=t_over,
                t_total_pred_cy=t_total_cy,
                t_total_pred_us=t_total_us,
                t_total_csv_us=t_csv_cy / CLOCK_MHZ,
                actual_us=actual_us,
            ))

    if skipped > 0:
        print(f"[INFO] Skipped {skipped} cases below min_quality={min_quality}")
    return rows


# ============================================================
# Metric computation
# ============================================================
def compute_mape(rows: List[ValidationRow]) -> float:
    if not rows:
        return float("nan")
    return sum(r.ape for r in rows) / len(rows)


def compute_mdape(rows: List[ValidationRow]) -> float:
    if not rows:
        return float("nan")
    apes = sorted(r.ape for r in rows)
    n = len(apes)
    if n % 2 == 1:
        return apes[n // 2]
    return (apes[n // 2 - 1] + apes[n // 2]) / 2


def compute_bias(rows: List[ValidationRow]) -> float:
    if not rows:
        return float("nan")
    return sum(r.error_pct for r in rows) / len(rows)


def compute_rho(rows: List[ValidationRow]) -> float:
    if len(rows) < 2:
        return float("nan")
    preds = [r.t_total_pred_us for r in rows]
    actuals = [r.actual_us for r in rows]
    return spearman_rank_correlation(preds, actuals)


# ============================================================
# Section printers
# ============================================================
def print_section1(coeffs: CalibCoeffs, csv_path: Path, tc_path: Path,
                   gt_col: str, n_valid: int) -> None:
    """Section 1: Model parameters."""
    print(f"\n{'='*72}")
    print(f"  D+B Performance Model Validation Report")
    print(f"{'='*72}")
    print(f"\n--- Section 1: Model Parameters ---\n")
    print(f"  Data source:     {csv_path}")
    print(f"  tpOrder source:  {tc_path}")
    print(f"  Ground truth:    {gt_col}")
    print(f"  Valid cases:     {n_valid}")
    print(f"\n  D+B Coefficients:")
    print(f"    eff_macs     = {coeffs.eff_macs:.2f} MACs/cy")
    print(f"    bw_eff_bpc   = {coeffs.bw_eff_bpc:.1f} B/cy")
    print(f"    l_sync_cy    = {coeffs.l_sync_cy:.0f} cy ({coeffs.l_sync_cy/CLOCK_MHZ:.1f} us)")
    print(f"    l_core_cy    = {coeffs.l_core_cy:.0f} cy ({coeffs.l_core_cy/CLOCK_MHZ:.1f} us)")
    print(f"    l_startup_cy = {coeffs.l_startup_cy:.0f} cy ({coeffs.l_startup_cy/CLOCK_MHZ:.1f} us)")


def print_section2(rows: List[ValidationRow]) -> None:
    """Section 2: Overall metrics."""
    print(f"\n--- Section 2: Overall Metrics ---\n")
    rho = compute_rho(rows)
    mape = compute_mape(rows)
    mdape = compute_mdape(rows)
    bias = compute_bias(rows)
    print(f"  Spearman rho:  {rho:.4f}")
    print(f"  MAPE:          {mape:.1f}%")
    print(f"  MdAPE:         {mdape:.1f}%")
    print(f"  Bias%:         {bias:+.1f}%")


def print_section3(rows: List[ValidationRow]) -> None:
    """Section 3: Per matrix-size analysis."""
    print(f"\n--- Section 3: Per Matrix Size ---\n")
    sizes = sorted(set(r.size_key for r in rows),
                   key=lambda s: int(s.split("x")[0]))

    header = (f"  {'Size':<16s} {'N':>4s}  {'rho':>7s}  {'MAPE':>6s}  "
              f"{'MdAPE':>6s}  {'Pred_avg':>9s}  {'Act_avg':>9s}  {'Bias%':>7s}")
    print(header)
    print(f"  {'-'*76}")

    for sk in sizes:
        sub = [r for r in rows if r.size_key == sk]
        rho = compute_rho(sub)
        mape = compute_mape(sub)
        mdape = compute_mdape(sub)
        bias = compute_bias(sub)
        pred_avg = sum(r.t_total_pred_us for r in sub) / len(sub)
        act_avg = sum(r.actual_us for r in sub) / len(sub)
        rho_s = f"{rho:.4f}" if not math.isnan(rho) else "n/a"
        print(f"  {sk:<16s} {len(sub):>4d}  {rho_s:>7s}  {mape:>5.1f}%  "
              f"{mdape:>5.1f}%  {pred_avg:>9.1f}  {act_avg:>9.1f}  {bias:>+6.1f}%")


def print_section4(rows: List[ValidationRow]) -> None:
    """Section 4: Per core-count analysis."""
    print(f"\n--- Section 4: Per Core Count ---\n")
    core_counts = sorted(set(r.num_cores for r in rows))

    print(f"  {'Cores':>5s}  {'N':>4s}  {'rho':>7s}  {'MAPE':>7s}  {'Bias%':>7s}")
    print(f"  {'-'*38}")

    for nc in core_counts:
        sub = [r for r in rows if r.num_cores == nc]
        rho = compute_rho(sub)
        mape = compute_mape(sub)
        bias = compute_bias(sub)
        rho_s = f"{rho:.4f}" if not math.isnan(rho) else "n/a"
        print(f"  {nc:>5d}  {len(sub):>4d}  {rho_s:>7s}  {mape:>6.1f}%  {bias:>+6.1f}%")


def print_section5(rows: List[ValidationRow]) -> None:
    """Section 5: Component dominance analysis."""
    print(f"\n--- Section 5: Component Dominance (avg % of T_total) ---\n")
    sizes = sorted(set(r.size_key for r in rows),
                   key=lambda s: int(s.split("x")[0]))

    print(f"  {'Size':<16s} {'T_comp%':>8s}  {'T_dma%':>7s}  {'T_over%':>8s}")
    print(f"  {'-'*44}")

    for sk in sizes:
        sub = [r for r in rows if r.size_key == sk]
        comp_pcts = [r.t_comp / r.t_total_pred_cy * 100 for r in sub if r.t_total_pred_cy > 0]
        comm_pcts = [r.t_comm / r.t_total_pred_cy * 100 for r in sub if r.t_total_pred_cy > 0]
        over_pcts = [r.t_overhead / r.t_total_pred_cy * 100 for r in sub if r.t_total_pred_cy > 0]
        n = len(comp_pcts)
        if n == 0:
            continue
        print(f"  {sk:<16s} {sum(comp_pcts)/n:>7.1f}%  {sum(comm_pcts)/n:>6.1f}%  "
              f"{sum(over_pcts)/n:>7.1f}%")


def print_section6(rows: List[ValidationRow]) -> None:
    """Section 6: Optimal config selection accuracy (per size)."""
    print(f"\n--- Section 6: Optimal Config Selection ---\n")
    sizes = sorted(set(r.size_key for r in rows),
                   key=lambda s: int(s.split("x")[0]))

    print(f"  {'Size':<16s} {'N':>4s}  {'Top1-Hit':>8s}  {'Top3-Hit':>8s}  "
          f"{'Regret%':>8s}  {'Best_pred':>10s}  {'Best_act':>10s}")
    print(f"  {'-'*72}")

    total_top1 = 0
    total_top3 = 0
    total_sizes = 0

    for sk in sizes:
        sub = [r for r in rows if r.size_key == sk]
        if len(sub) < 2:
            continue
        total_sizes += 1

        # Sort by prediction (ascending = fastest first)
        by_pred = sorted(sub, key=lambda r: r.t_total_pred_us)
        # Sort by actual (ascending = fastest first)
        by_actual = sorted(sub, key=lambda r: r.actual_us)

        actual_best = by_actual[0]
        actual_best_idx = by_actual[0].case_index
        actual_top3_idx = {r.case_index for r in by_actual[:3]}

        pred_best = by_pred[0]
        pred_best_idx = pred_best.case_index

        top1_hit = pred_best_idx == actual_best_idx
        top3_hit = pred_best_idx in actual_top3_idx

        if top1_hit:
            total_top1 += 1
        if top3_hit:
            total_top3 += 1

        # Regret: how much worse is predicted-best vs actual-best (in actual time)
        pred_best_actual_us = pred_best.actual_us
        regret_pct = ((pred_best_actual_us - actual_best.actual_us)
                      / actual_best.actual_us * 100)

        print(f"  {sk:<16s} {len(sub):>4d}  {'YES' if top1_hit else 'no':>8s}  "
              f"{'YES' if top3_hit else 'no':>8s}  {regret_pct:>+7.1f}%  "
              f"{pred_best.t_total_pred_us:>10.1f}  {actual_best.actual_us:>10.1f}")

    if total_sizes > 0:
        print(f"\n  Summary: Top-1 {total_top1}/{total_sizes} "
              f"({total_top1/total_sizes*100:.0f}%), "
              f"Top-3 {total_top3}/{total_sizes} "
              f"({total_top3/total_sizes*100:.0f}%)")


def print_section7(rows: List[ValidationRow]) -> None:
    """Section 7: Error distribution by bands."""
    print(f"\n--- Section 7: Error Distribution ---\n")
    bands = [
        ("< 10%", 0, 10),
        ("10%-20%", 10, 20),
        ("20%-50%", 20, 50),
        ("50%-100%", 50, 100),
        ("> 100%", 100, float("inf")),
    ]

    print(f"  {'Error band':<12s}  {'Count':>5s}  {'Pct':>6s}")
    print(f"  {'-'*28}")

    n = len(rows)
    for label, lo, hi in bands:
        count = sum(1 for r in rows if lo <= r.ape < hi)
        pct = count / n * 100 if n > 0 else 0
        print(f"  {label:<12s}  {count:>5d}  {pct:>5.1f}%")


def print_section8(rows: List[ValidationRow]) -> None:
    """Section 8: Residual analysis (over/under-prediction)."""
    print(f"\n--- Section 8: Residual Analysis ---\n")
    n = len(rows)
    over = [r for r in rows if r.error_pct > 0]
    under = [r for r in rows if r.error_pct < 0]
    exact = [r for r in rows if r.error_pct == 0]

    print(f"  Over-prediction  (+): {len(over):>4d} ({len(over)/n*100:.1f}%)")
    print(f"  Under-prediction (-): {len(under):>4d} ({len(under)/n*100:.1f}%)")
    if exact:
        print(f"  Exact match:          {len(exact):>4d}")

    # Per-size bias direction
    sizes = sorted(set(r.size_key for r in rows),
                   key=lambda s: int(s.split("x")[0]))
    print(f"\n  Per-size bias direction:")
    print(f"  {'Size':<16s}  {'Bias%':>7s}  {'Direction':<12s}")
    print(f"  {'-'*40}")
    for sk in sizes:
        sub = [r for r in rows if r.size_key == sk]
        bias = compute_bias(sub)
        direction = "OVER" if bias > 5 else ("UNDER" if bias < -5 else "balanced")
        print(f"  {sk:<16s}  {bias:>+6.1f}%  {direction:<12s}")


def print_section9(rows: List[ValidationRow]) -> None:
    """Section 9: Top 10 worst-error cases."""
    print(f"\n--- Section 9: Top 10 Maximum Error Cases ---\n")
    worst = sorted(rows, key=lambda r: r.ape, reverse=True)[:10]

    print(f"  {'#':>2s}  {'case':>5s}  {'Size':<16s} {'cores':>5s}  "
          f"{'SP':>7s}  {'TP':>11s}  {'tpOrd':>5s}  "
          f"{'Pred_us':>8s}  {'Act_us':>8s}  {'Err%':>7s}")
    print(f"  {'-'*92}")

    for i, r in enumerate(worst):
        print(f"  {i+1:>2d}  {r.case_index:>5d}  {r.size_key:<16s} {r.num_cores:>5d}  "
              f"({r.SPm:>2d},{r.SPn:>2d})  ({r.TPm:>2d},{r.TPk:>2d},{r.TPn:>2d})  "
              f"    {TP_AXIS_NAMES.get(r.tp_order, '?')}  "
              f"{r.t_total_pred_us:>8.1f}  {r.actual_us:>8.1f}  {r.error_pct:>+6.1f}%")


def print_integrity_check(rows: List[ValidationRow]) -> None:
    """Compare CSV t_total_pred (pre-calibration) vs recomputed (calibrated).

    The CSV was generated before calibration, so deviations are expected.
    This section quantifies the calibration impact.
    """
    diffs = []
    for r in rows:
        if r.t_total_csv_us > 0:
            diff_pct = (r.t_total_pred_us - r.t_total_csv_us) / r.t_total_csv_us * 100
            diffs.append((r.case_index, diff_pct))

    if not diffs:
        return

    abs_diffs = [abs(d) for _, d in diffs]
    max_diff = max(abs_diffs)
    mean_diff = sum(d for _, d in diffs) / len(diffs)
    matched = sum(1 for d in abs_diffs if d < 0.01)

    if matched == len(diffs):
        print(f"\n  Integrity: CSV predictions match recomputed values (all within 0.01%)")
    else:
        print(f"\n  Note: CSV t_total_pred uses pre-calibration coefficients.")
        print(f"  Calibrated predictions differ: mean={mean_diff:+.1f}%, max={max_diff:.1f}%")
        print(f"  (This is expected -- validation uses calibrated D+B coefficients.)")


# ============================================================
# Visualization
# ============================================================
def generate_plots(rows: List[ValidationRow], plot_dir: Path) -> None:
    """Generate 6 matplotlib charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("\n  [WARN] matplotlib not available, skipping plots.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    preds = [r.t_total_pred_us for r in rows]
    actuals = [r.actual_us for r in rows]
    errors = [r.error_pct for r in rows]
    apes = [r.ape for r in rows]
    sizes = sorted(set(r.size_key for r in rows),
                   key=lambda s: int(s.split("x")[0]))

    # 1. Scatter: predicted vs actual (log-log)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Color by size
    size_colors = {}
    cmap = plt.cm.tab10
    for i, sk in enumerate(sizes):
        size_colors[sk] = cmap(i % 10)

    for sk in sizes:
        sub = [r for r in rows if r.size_key == sk]
        ax.scatter([r.actual_us for r in sub], [r.t_total_pred_us for r in sub],
                   c=[size_colors[sk]], label=sk, alpha=0.7, s=30)

    # Perfect line and error bands
    lo = min(min(actuals), min(preds)) * 0.8
    hi = max(max(actuals), max(preds)) * 1.2
    line = [lo, hi]
    ax.plot(line, line, "k-", linewidth=1, label="perfect")
    ax.fill_between(line, [v * 0.8 for v in line], [v * 1.2 for v in line],
                    alpha=0.1, color="green", label="+/-20%")
    ax.fill_between(line, [v * 0.5 for v in line], [v * 1.5 for v in line],
                    alpha=0.05, color="orange", label="+/-50%")

    ax.set_xlabel("Actual (us)")
    ax.set_ylabel("Predicted (us)")
    ax.set_title("D+B Model: Predicted vs Actual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # 2. Histogram: signed error distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    mean_err = sum(errors) / len(errors)
    ax.axvline(mean_err, color="blue", linestyle="--", linewidth=1, label=f"mean={mean_err:.1f}%")
    ax.set_xlabel("Signed Error %")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "hist_error_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Boxplot: APE by size
    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = [[r.ape for r in rows if r.size_key == sk] for sk in sizes]
    ax.boxplot(box_data, tick_labels=sizes, showfliers=True)
    ax.set_ylabel("Absolute Percentage Error (%)")
    ax.set_title("Error by Matrix Size")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "boxplot_error_by_size.png", dpi=150)
    plt.close(fig)

    # 4. Stacked bar: component dominance
    fig, ax = plt.subplots(figsize=(8, 5))
    comp_avgs, comm_avgs, over_avgs = [], [], []
    for sk in sizes:
        sub = [r for r in rows if r.size_key == sk and r.t_total_pred_cy > 0]
        if not sub:
            comp_avgs.append(0)
            comm_avgs.append(0)
            over_avgs.append(0)
            continue
        comp_avgs.append(sum(r.t_comp / r.t_total_pred_cy for r in sub) / len(sub) * 100)
        comm_avgs.append(sum(r.t_comm / r.t_total_pred_cy for r in sub) / len(sub) * 100)
        over_avgs.append(sum(r.t_overhead / r.t_total_pred_cy for r in sub) / len(sub) * 100)

    x = range(len(sizes))
    ax.bar(x, comp_avgs, label="T_comp")
    ax.bar(x, comm_avgs, bottom=comp_avgs, label="T_dma")
    ax.bar(x, over_avgs,
           bottom=[a + b for a, b in zip(comp_avgs, comm_avgs)], label="T_overhead")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sizes, rotation=45)
    ax.set_ylabel("% of T_total")
    ax.set_title("Component Dominance by Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "stacked_component_dominance.png", dpi=150)
    plt.close(fig)

    # 5. Bar: regret by size
    fig, ax = plt.subplots(figsize=(8, 5))
    regrets = []
    for sk in sizes:
        sub = [r for r in rows if r.size_key == sk]
        if len(sub) < 2:
            regrets.append(0)
            continue
        by_pred = sorted(sub, key=lambda r: r.t_total_pred_us)
        by_actual = sorted(sub, key=lambda r: r.actual_us)
        regret = ((by_pred[0].actual_us - by_actual[0].actual_us)
                  / by_actual[0].actual_us * 100)
        regrets.append(regret)

    colors = ["green" if r < 10 else ("orange" if r < 20 else "red") for r in regrets]
    ax.bar(range(len(sizes)), regrets, color=colors)
    ax.set_xticks(list(range(len(sizes))))
    ax.set_xticklabels(sizes, rotation=45)
    ax.set_ylabel("Regret %")
    ax.set_title("Selection Regret by Size")
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, label="20% threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "bar_regret_by_size.png", dpi=150)
    plt.close(fig)

    # 6. Residual vs predicted
    fig, ax = plt.subplots(figsize=(8, 5))
    for sk in sizes:
        sub = [r for r in rows if r.size_key == sk]
        ax.scatter([r.t_total_pred_us for r in sub],
                   [r.error_pct for r in sub],
                   c=[size_colors[sk]], label=sk, alpha=0.6, s=25)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Predicted (us)")
    ax.set_ylabel("Signed Error %")
    ax.set_title("Residuals vs Predicted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_vs_predicted.png", dpi=150)
    plt.close(fig)

    print(f"\n  [INFO] 6 plots saved to {plot_dir}/")


# ============================================================
# CLI
# ============================================================
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate D+B performance cost model predictions")
    p.add_argument("--csv", required=True,
                   help="Path to result.csv with predictions and actuals")
    tc_group = p.add_mutually_exclusive_group(required=True)
    tc_group.add_argument("--tc",
                          help="Path to tc_list.json (for tpOrder mapping)")
    tc_group.add_argument("--tc-archive",
                          help="Path to archive directory (per-case tc.json)")
    p.add_argument("--calib", required=True,
                   help="Path to calibration.json")
    p.add_argument("--ground-truth", default="min_us",
                   help="CSV column for ground truth (default: min_us)")
    p.add_argument("--min-quality", type=float, default=0.0,
                   help="Minimum data_quality threshold (default: 0.0)")
    p.add_argument("--plot", default="",
                   help="Directory for matplotlib plots (omit to skip)")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv).resolve()
    tc_path = Path(args.tc).resolve() if args.tc else None
    tc_archive_path = Path(args.tc_archive).resolve() if args.tc_archive else None
    calib_path = Path(args.calib).resolve()

    if not csv_path.is_file():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        return 1
    if tc_path and not tc_path.is_file():
        print(f"[ERROR] tc_list.json not found: {tc_path}", file=sys.stderr)
        return 1

    coeffs = load_calibration(calib_path)
    if not coeffs.calibrated:
        print(f"[WARN] Calibration not loaded, using defaults", file=sys.stderr)

    rows = load_validation_data(csv_path, tc_path, coeffs, args.ground_truth,
                                args.min_quality, tc_archive_path=tc_archive_path)
    if len(rows) < 2:
        print(f"[ERROR] Need at least 2 valid cases, got {len(rows)}", file=sys.stderr)
        return 1

    # Console report
    tc_display = tc_archive_path or tc_path
    print_section1(coeffs, csv_path, tc_display, args.ground_truth, len(rows))
    print_integrity_check(rows)
    print_section2(rows)
    print_section3(rows)
    print_section4(rows)
    print_section5(rows)
    print_section6(rows)
    print_section7(rows)
    print_section8(rows)
    print_section9(rows)

    print(f"\n{'='*72}")

    # Plots
    if args.plot:
        generate_plots(rows, Path(args.plot).resolve())

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
