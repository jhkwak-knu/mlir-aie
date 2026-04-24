#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibrate_energy.py — Fit energy model coefficients from NPU measurements.

Models:
  --- Linear family (OLS) ---
  E-A: E = P_active * t + E_startup                              (2p)
  E-B: E = P_comp * t_comp + P_dma * t_dma + P_idle * t_ovh + E_startup  (4p)
  E-C: E = E_mac * MACs + E_byte * bytes + P_static * N * t + E_startup  (4p)
  E-D: E = P_core * N_cores * t + E_startup                      (2p)
  E-F: E = (P_base + P_core * N_cores) * t + E_startup           (3p)
  E-G: E = P_col * N_columns * t + E_startup                     (2p)
  E-H: E = (P_base + P_col * N_columns) * t + E_startup          (3p)
  E-I: E = (P_base + P_col * N_col + P_core * N) * t + E_startup (4p)

  --- Log-space family (OLS on log(E)) ---
  L-A: E = C * t^a                                               (2p)
  L-B: E = C * t^a * N^c                                         (3p)
  L-C: E = C * t^a * N_col^c                                     (3p)
  L-D: E = C * t^a * N^c1 * N_col^c2                             (4p)

  --- Weighted log-space family ---
  WL-A/WL-B/WL-C: Active-fraction weighted versions of L-A/L-B/L-C

  --- Theory-calibrated family (scipy nonlinear opt, log-space MSE) ---
  T-A: E = E_MAC_cal * MACs + E_DRAM_cal * bytes + P_STATIC_cal * N * T    (3p)
  T-B: E = E_MAC_cal * MACs + E_DRAM_cal * bytes + (P_BASE + P_PE * N) * T  (4p)
  T-C: E = E_MAC_cal * MACs + E_DRAM_cal * bytes + (P_BASE + P_PE * N) * T + E_STARTUP  (5p)

Writes calibration.json v3 with energy section.

Usage:
    python3 scripts/analyze/calibrate_energy.py \\
        --energy out/calibration/result_energy.csv \\
        --tc out/calibration/tc_list.json \\
        --calib data/calibration.json \\
        --min-wall-s 0.005 \\
        --output data/calibration.json
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
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import (  # noqa: E402
    OpCase, CalibCoeffs,
    load_calibration, atomic_write_json, DEFAULT_CALIB_PATH,
    load_system_info, DEFAULT_SYS_PATH,
    CommentFilterFile,
    spearman_rank_correlation,
)
from cost_model import (  # noqa: E402
    Candidate, total_data_bytes, _dma_ops_per_step,
    E_MAC_PJ, E_DRAM_PJ, P_STATIC_PJ,
)
import models as _models  # noqa: E402

CLOCK_MHZ = 1500
# Loaded at startup from xdna2_info.json; fallback = 4
_COMP_TILES_PER_COL = 4

# Minimum RAPL measurement window for reliable data.
# RAPL counters update ~1ms; 5ms guarantees >= 5 updates.
DEFAULT_MIN_WALL_S = 0.005


# ---------------------------------------------------------------------------
# Column mapping helper
# ---------------------------------------------------------------------------
def _col(row: Dict[str, str], *keys: str) -> str:
    """Return first matching column value from a CSV row dict."""
    for k in keys:
        if k in row:
            return row[k]
    raise KeyError(f"None of {keys} found in CSV columns: {list(row.keys())}")


def _load_tp_order_map(tc_path: Path) -> Dict[int, int]:
    """Load tc_list.json and build case_index -> tpOrder[0] map (1-based)."""
    if not tc_path or not tc_path.is_file():
        return {}
    with tc_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    result: Dict[int, int] = {}
    for i, case in enumerate(doc.get("cases", []), start=1):
        levels = case.get("levels", [{}])
        tp_order = levels[0].get("tpOrder", [2, 0, 1])
        result[i] = tp_order[0]
    return result


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class ECalibRow:
    """Merged energy measurement for calibration."""
    case_index: int
    M: int
    K: int
    N: int
    SPm: int
    SPn: int
    TPm: int
    TPk: int
    TPn: int
    TM: int
    TK: int
    TN: int
    num_cores: int
    tp_order_inner: int
    avg_iter_us: float
    npu_per_iter_uj: float
    wall_elapsed_s: float
    idle_pkg_mw: float = -1.0
    active_pkg_mw: float = -1.0
    n_iterations: int = 10

    @property
    def n_cores(self) -> int:
        return self.SPm * self.SPn

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.K}x{self.N}"

    def make_op(self) -> OpCase:
        return OpCase(M=self.M, K=self.K, N=self.N, elem_type="bf16")

    @property
    def n_columns(self) -> int:
        return math.ceil(self.n_cores / _COMP_TILES_PER_COL)

    def make_cand(self) -> Candidate:
        return Candidate(
            num_cores=self.n_cores,
            num_columns=self.n_columns,
            SPm=self.SPm, SPn=self.SPn,
            TPm=self.TPm, TPk=self.TPk, TPn=self.TPn,
            TM=self.TM, TK=self.TK, TN=self.TN,
        )


@dataclass
class SkippedRow:
    """Record of a filtered-out measurement with reason."""
    case_index: int
    size_key: str
    n_cores: int
    reason: str


@dataclass
class EnergyFitResult:
    """Result of fitting one energy model."""
    model: str
    params: Dict[str, float]
    n_params: int
    rho: float
    mape: float
    predictions_uj: List[float]


# ---------------------------------------------------------------------------
# Loading with validity filter
# ---------------------------------------------------------------------------
def load_energy_data(
    csv_path: Path,
    tc_path: Optional[Path] = None,
    min_wall_s: float = DEFAULT_MIN_WALL_S,
    max_wall_s: float = 0.0,
    gt_mode: str = "npu",
) -> Tuple[List[ECalibRow], List[SkippedRow]]:
    """Load energy CSV into calibration rows, filtering unreliable measurements.

    gt_mode controls which energy value is used as ground truth:
      "npu"        -- npu_per_iter_uj (idle-subtracted, original)
      "active_pkg" -- active_pkg total energy per iter (no idle subtraction)
      "corrected"  -- active_pkg minus global median idle, per iter

    Filters:
      1. GT energy <= 0 (negative energy from RAPL noise)
      2. wall_elapsed_s < min_wall_s (too short for reliable RAPL reading)

    Returns (valid_rows, skipped_rows).
    """
    tp_order_map = _load_tp_order_map(tc_path) if tc_path else {}

    # First pass: collect all rows to compute global idle median for 'corrected'
    raw_rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(CommentFilterFile(f))
        for row in reader:
            raw_rows.append(dict(row))

    # Compute global idle median for 'corrected' mode
    idle_values = []
    for row in raw_rows:
        idle_mw = float(row.get("idle_pkg_mw", row.get("idle_uncore_mw", "-1")))
        if idle_mw > 0:
            idle_values.append(idle_mw)
    idle_median_mw = sorted(idle_values)[len(idle_values) // 2] if idle_values else 0.0

    rows: List[ECalibRow] = []
    skipped: List[SkippedRow] = []

    for row in raw_rows:
        # Skip rows without wall time (e.g. RUN_FAIL with fewer columns).
        # Prefer wall_elapsed_s; fall back to batch_best_wall_s for v14+ CSVs.
        wall_raw = row.get("wall_elapsed_s")
        if wall_raw is None or wall_raw == "":
            continue
        wall_s = float(wall_raw)
        if wall_s <= 0:
            wall_s = float(row.get("batch_best_wall_s", -1))
        case_idx = int(row["case_index"])
        M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
        size_key = f"{M}x{K}x{N}"
        n_cores = int(_col(row, "num_cores", "numSpm"))
        n_iters = int(row.get("iters", "10"))

        # Read raw RAPL columns
        idle_mw = float(row.get("idle_pkg_mw",
                                row.get("idle_uncore_mw", "-1")))
        active_mw = float(row.get("active_pkg_mw",
                                  row.get("active_uncore_mw", "-1")))
        npu_per_iter = float(_col(row, "batch_min_energy_per_iter_uj",
                                  "npu_energy_per_iter_uj"))

        # Compute GT energy per iteration based on mode
        if gt_mode == "active_pkg":
            if active_mw <= 0 or wall_s <= 0:
                per_iter = -1.0
            else:
                # active_mw * wall_s * 1000 = total energy in uJ
                per_iter = active_mw * wall_s * 1000.0 / n_iters
        elif gt_mode == "corrected":
            if active_mw <= 0 or wall_s <= 0:
                per_iter = -1.0
            else:
                per_iter = (active_mw - idle_median_mw) * wall_s * 1000.0 / n_iters
        else:  # "npu"
            per_iter = npu_per_iter

        # Filter 1: negative or zero energy
        if per_iter <= 0:
            skipped.append(SkippedRow(
                case_idx, size_key, n_cores,
                f"non-positive energy ({per_iter:.1f} uJ, gt={gt_mode})"))
            continue

        # Filter 2: short measurement window
        if wall_s < min_wall_s:
            skipped.append(SkippedRow(
                case_idx, size_key, n_cores,
                f"short window ({wall_s*1000:.1f}ms < {min_wall_s*1000:.0f}ms)"))
            continue

        # Filter 3: long measurement window (idle error amplification)
        if max_wall_s > 0 and wall_s > max_wall_s:
            skipped.append(SkippedRow(
                case_idx, size_key, n_cores,
                f"long window ({wall_s:.2f}s > {max_wall_s:.2f}s)"))
            continue

        # tp_order_inner: CSV column or tc_list.json lookup
        if "tp_order_inner" in row:
            tp_inner = int(row["tp_order_inner"])
        elif case_idx in tp_order_map:
            tp_inner = tp_order_map[case_idx]
        else:
            tp_inner = 2  # fallback: K-inner

        rows.append(ECalibRow(
            case_index=case_idx,
            M=M, K=K, N=N,
            SPm=int(row["SPm"]), SPn=int(row["SPn"]),
            TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
            TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
            num_cores=n_cores,
            tp_order_inner=tp_inner,
            avg_iter_us=float(_col(row, "avg_iter_us", "avg_us")),
            npu_per_iter_uj=per_iter,
            wall_elapsed_s=wall_s,
            idle_pkg_mw=idle_mw,
            active_pkg_mw=active_mw,
            n_iterations=n_iters,
        ))

    if gt_mode != "npu":
        print(f"[INFO] GT mode: {gt_mode}")
        if gt_mode == "corrected":
            print(f"[INFO] Global idle median: {idle_median_mw:.1f} mW "
                  f"(from {len(idle_values)} samples)")

    return rows, skipped


def print_idle_power_analysis(
    rows: List[ECalibRow],
    skipped: List[SkippedRow],
) -> None:
    """Print idle power stability analysis across all loaded data."""
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  Idle Power Stability Analysis")
    print(sep)

    # Collect idle and active from valid + skipped (all raw data)
    all_idle = [r.idle_pkg_mw for r in rows if r.idle_pkg_mw > 0]
    all_active = [r.active_pkg_mw for r in rows if r.active_pkg_mw > 0]

    if not all_idle:
        print("  No idle_pkg_mw data available")
        print(sep)
        return

    all_idle_sorted = sorted(all_idle)
    n = len(all_idle_sorted)
    idle_mean = sum(all_idle) / n
    idle_std = (sum((x - idle_mean) ** 2 for x in all_idle) / n) ** 0.5
    idle_cv = idle_std / idle_mean * 100 if idle_mean > 0 else 0
    idle_median = all_idle_sorted[n // 2]
    idle_q1 = all_idle_sorted[n // 4]
    idle_q3 = all_idle_sorted[3 * n // 4]
    idle_iqr = idle_q3 - idle_q1

    print(f"  idle_pkg_mw (n={n}):")
    print(f"    min={all_idle_sorted[0]:.0f}  q1={idle_q1:.0f}  "
          f"median={idle_median:.0f}  q3={idle_q3:.0f}  "
          f"max={all_idle_sorted[-1]:.0f}")
    print(f"    mean={idle_mean:.0f}  std={idle_std:.0f}  CV={idle_cv:.1f}%")
    print(f"    IQR={idle_iqr:.0f}  outlier_fence=[{idle_q1 - 1.5*idle_iqr:.0f}, "
          f"{idle_q3 + 1.5*idle_iqr:.0f}]")

    n_outliers = sum(1 for x in all_idle
                     if x < idle_q1 - 1.5 * idle_iqr or x > idle_q3 + 1.5 * idle_iqr)
    print(f"    outliers: {n_outliers}/{n} ({n_outliers/n*100:.0f}%)")

    if all_active:
        act_sorted = sorted(all_active)
        na = len(act_sorted)
        act_mean = sum(all_active) / na
        act_std = (sum((x - act_mean) ** 2 for x in all_active) / na) ** 0.5
        act_cv = act_std / act_mean * 100 if act_mean > 0 else 0
        print(f"\n  active_pkg_mw (n={na}):")
        print(f"    min={act_sorted[0]:.0f}  median={act_sorted[na//2]:.0f}  "
              f"max={act_sorted[-1]:.0f}")
        print(f"    mean={act_mean:.0f}  std={act_std:.0f}  CV={act_cv:.1f}%")

    # Per-size breakdown
    size_idle: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r.idle_pkg_mw > 0:
            size_idle[r.size_key].append(r.idle_pkg_mw)

    if size_idle:
        print(f"\n  Per-size idle_pkg_mw:")
        print(f"  {'Size':<16s}  {'N':>4s}  {'Mean':>8s}  {'Std':>8s}  {'CV%':>6s}  "
              f"{'Median':>8s}")
        for sk in sorted(size_idle.keys()):
            vals = size_idle[sk]
            ns = len(vals)
            m = sum(vals) / ns
            s = (sum((x - m) ** 2 for x in vals) / ns) ** 0.5 if ns > 1 else 0
            cv = s / m * 100 if m > 0 else 0
            med = sorted(vals)[ns // 2]
            print(f"  {sk:<16s}  {ns:>4d}  {m:>8.0f}  {s:>8.0f}  {cv:>5.1f}%  "
                  f"{med:>8.0f}")

    # idle > active cases (causes negative energy)
    neg_count = sum(1 for r in rows
                    if r.idle_pkg_mw > 0 and r.active_pkg_mw > 0
                    and r.idle_pkg_mw > r.active_pkg_mw)
    total_with_both = sum(1 for r in rows
                          if r.idle_pkg_mw > 0 and r.active_pkg_mw > 0)
    if total_with_both > 0:
        print(f"\n  idle > active (negative energy): {neg_count}/{total_with_both} "
              f"({neg_count/total_with_both*100:.0f}%)")

    print(sep)


def print_validity_report(
    rows: List[ECalibRow],
    skipped: List[SkippedRow],
    total_csv: int,
    min_wall_s: float,
    gt_mode: str = "npu",
) -> None:
    """Print filter results: size x cores matrix and skipped case list."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Measurement Validity Report (gt_mode={gt_mode})")
    print(sep)
    print(f"  Total CSV rows:    {total_csv}")
    print(f"  Filtered out:      {len(skipped)}")
    print(f"  Valid:             {len(rows)}")
    print(f"  Min wall_elapsed:  {min_wall_s*1000:.0f} ms")

    # Size x cores coverage matrix
    size_core: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        size_core[r.size_key][r.n_cores] += 1

    all_cores = sorted(set(r.n_cores for r in rows))
    all_sizes = sorted(size_core.keys())

    print(f"\n  Coverage matrix (valid cases):")
    hdr = f"  {'Size':<16s}" + "".join(f"  {nc:>3d}c" for nc in all_cores) + "  Total"
    print(hdr)
    for sk in all_sizes:
        counts = [size_core[sk].get(nc, 0) for nc in all_cores]
        total = sum(counts)
        row_str = f"  {sk:<16s}" + "".join(f"  {c:>4d}" for c in counts) + f"  {total:>5d}"
        print(row_str)

    # Warn about empty cells
    empty_cells = []
    for sk in all_sizes:
        for nc in all_cores:
            if size_core[sk].get(nc, 0) == 0:
                empty_cells.append((sk, nc))
    if empty_cells:
        print(f"\n  WARNING: {len(empty_cells)} empty (size, cores) cells:")
        for sk, nc in empty_cells[:10]:
            print(f"    {sk} / {nc} cores: 0 valid cases")
        if len(empty_cells) > 10:
            print(f"    ... and {len(empty_cells) - 10} more")

    # Skipped case summary by reason
    if skipped:
        reason_counts: Dict[str, int] = defaultdict(int)
        for s in skipped:
            if "negative" in s.reason or "non-positive" in s.reason:
                reason_counts["non-positive energy"] += 1
            elif "long" in s.reason:
                reason_counts["long window"] += 1
            else:
                reason_counts["short window"] += 1
        print(f"\n  Skipped reasons:")
        for reason, count in sorted(reason_counts.items()):
            print(f"    {reason}: {count}")

    print(sep)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_features(
    rows: List[ECalibRow], coeffs: Optional[CalibCoeffs],
) -> List[Dict[str, float]]:
    """Compute model features for each row using calibrated perf model.

    Delegates performance computation to models.PerfModel.components_v9().
    """
    features = []
    for r in rows:
        op = r.make_op()
        cand = r.make_cand()

        n_dma = _dma_ops_per_step(cand, r.tp_order_inner)
        data_bytes = total_data_bytes(op, cand, r.tp_order_inner)
        macs = r.M * r.K * r.N

        if coeffs and coeffs.calibrated:
            t_comp_cy, t_dma_cy, t_ovh_cy = _models.PerfModel.components_v9(
                macs=macs, data_bytes=data_bytes,
                n_cores=r.n_cores, tp_total=r.tp_total, n_dma=n_dma,
                eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
                l_sync=coeffs.l_sync_cy, l_pe=coeffs.l_pe_cy,
                l_dma=coeffs.l_dma_cy, l_startup=coeffs.l_startup_cy)
        else:
            t_comp_cy = macs / (r.n_cores * 256.0)
            t_dma_cy = data_bytes / 4.0
            t_ovh_cy = 20.0 * r.tp_total

        t_total_cy = t_comp_cy + t_dma_cy + t_ovh_cy
        t_total_us = t_total_cy / CLOCK_MHZ

        features.append({
            "t_comp_us": t_comp_cy / CLOCK_MHZ,
            "t_dma_us": t_dma_cy / CLOCK_MHZ,
            "t_ovh_us": t_ovh_cy / CLOCK_MHZ,
            "t_total_us": t_total_us,
            "t_measured_us": r.avg_iter_us,
            "n_cores": r.n_cores,
            "n_columns": r.n_columns,
            "macs": macs,
            "data_bytes": data_bytes,
            "t_total_cy": t_total_cy,
            "n_dma": n_dma,
            "tp_total": r.tp_total,
        })
    return features


# ---------------------------------------------------------------------------
# Model fitting (OLS)
# ---------------------------------------------------------------------------
def _ols_1var(x: List[float], y: List[float]) -> Tuple[float, float]:
    """OLS: y = a*x + b. Returns (a, b)."""
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-20:
        return 0.0, sum(y) / n if n > 0 else 0.0
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


def _ols_multi(
    xs: List[List[float]], y: List[float], *, intercept: bool = True,
) -> List[float]:
    """General OLS: y = b0*x0 + b1*x1 + ... [+ c]. Returns coefficients.

    If intercept=True, appends an all-ones column and the last returned
    coefficient is the intercept term.
    """
    cols = list(xs)
    if intercept:
        cols.append([1.0] * len(y))
    dim = len(cols)
    n = len(y)

    # Normal equations: A * beta = B
    A = [[sum(cols[i][k] * cols[j][k] for k in range(n))
          for j in range(dim)] for i in range(dim)]
    B = [sum(cols[i][k] * y[k] for k in range(n)) for i in range(dim)]

    # Gaussian elimination with partial pivoting
    for col in range(dim):
        max_row = max(range(col, dim), key=lambda r: abs(A[r][col]))
        A[col], A[max_row] = A[max_row], A[col]
        B[col], B[max_row] = B[max_row], B[col]
        if abs(A[col][col]) < 1e-20:
            continue
        for row in range(col + 1, dim):
            factor = A[row][col] / A[col][col]
            for j in range(col, dim):
                A[row][j] -= factor * A[col][j]
            B[row] -= factor * B[col]

    # Back substitution
    beta = [0.0] * dim
    for i in range(dim - 1, -1, -1):
        if abs(A[i][i]) < 1e-20:
            continue
        beta[i] = (B[i] - sum(A[i][j] * beta[j]
                               for j in range(i + 1, dim))) / A[i][i]
    return beta


def compute_mape(preds: List[float], actuals: List[float]) -> float:
    if not actuals:
        return float("nan")
    total = sum(abs(p - a) / a for p, a in zip(preds, actuals) if a > 0)
    n = sum(1 for a in actuals if a > 0)
    return (total / n * 100) if n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Helper: build predictions and metrics from coefficients + feature columns
# ---------------------------------------------------------------------------
def _eval_linear(
    coefs: List[float], xs: List[List[float]], y: List[float],
    *, intercept_idx: int = -1,
) -> Tuple[List[float], float, float]:
    """Compute predictions = sum(coef_i * x_i) and return (preds, rho, mape).

    intercept_idx: which coef is the intercept (default: last).
    """
    n = len(y)
    preds = []
    for k in range(n):
        preds.append(sum(c * xs[j][k] for j, c in enumerate(coefs)
                         if j < len(xs))
                     + (coefs[intercept_idx] if intercept_idx < len(coefs) else 0))
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)
    return preds, rho, mape


# ---------------------------------------------------------------------------
# Model E-A: E = P_active * t + E_startup  (2p)
# ---------------------------------------------------------------------------
def fit_model_ea(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    x = [[f[time_key] for f in features]]
    y = [r.npu_per_iter_uj for r in rows]
    a, b = _ols_multi(x, y, intercept=True)
    preds, rho, mape = _eval_linear([a, b], x + [[1.0]*len(y)], y,
                                     intercept_idx=len(x))
    return EnergyFitResult(
        model="E-A",
        params={"p_active_uw": round(a * 1e6, 1),
                "e_startup_uj": round(b, 2)},
        n_params=2, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Model E-B: E = P_comp*t_comp + P_dma*t_dma + P_idle*t_ovh + E_startup (4p)
# ---------------------------------------------------------------------------
def fit_model_eb(
    rows: List[ECalibRow], features: List[Dict],
) -> EnergyFitResult:
    xs = [[f["t_comp_us"] for f in features],
          [f["t_dma_us"] for f in features],
          [f["t_ovh_us"] for f in features]]
    y = [r.npu_per_iter_uj for r in rows]
    coefs = _ols_multi(xs, y, intercept=True)
    preds = [sum(coefs[j] * xs[j][k] for j in range(3)) + coefs[3]
             for k in range(len(y))]
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)
    return EnergyFitResult(
        model="E-B",
        params={"p_comp_uw": round(coefs[0]*1e6, 1),
                "p_dma_uw": round(coefs[1]*1e6, 1),
                "p_idle_uw": round(coefs[2]*1e6, 1),
                "e_startup_uj": round(coefs[3], 2)},
        n_params=4, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Model E-C: E = E_mac*MACs + E_byte*bytes + P_static*N*t + E_startup (4p)
# ---------------------------------------------------------------------------
def fit_model_ec(
    rows: List[ECalibRow], features: List[Dict],
) -> EnergyFitResult:
    xs = [[float(f["macs"]) for f in features],
          [float(f["data_bytes"]) for f in features],
          [float(r.n_cores * f["t_total_cy"])
           for r, f in zip(rows, features)]]
    y = [r.npu_per_iter_uj for r in rows]
    coefs = _ols_multi(xs, y, intercept=True)
    preds = [sum(coefs[j] * xs[j][k] for j in range(3)) + coefs[3]
             for k in range(len(y))]
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)
    return EnergyFitResult(
        model="E-C",
        params={"e_mac_pj": round(coefs[0]*1e6, 4),
                "e_byte_pj": round(coefs[1]*1e6, 4),
                "p_static_pj": round(coefs[2]*1e6, 4),
                "e_startup_uj": round(coefs[3], 2)},
        n_params=4, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Model E-D: E = P_core * N_cores * t + E_startup  (2p)
# ---------------------------------------------------------------------------
def fit_model_ed(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    xs = [[float(r.n_cores) * f[time_key]
           for r, f in zip(rows, features)]]
    y = [r.npu_per_iter_uj for r in rows]
    a, b = _ols_multi(xs, y, intercept=True)
    preds = [a * xs[0][k] + b for k in range(len(y))]
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)
    return EnergyFitResult(
        model="E-D",
        params={"p_pe_uw": round(a*1e6, 1),
                "e_startup_uj": round(b, 2)},
        n_params=2, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Model E-F: E = (P_base + P_core * N) * t + E_startup  (3p)
# ---------------------------------------------------------------------------
def fit_model_ef(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    xs = [[f[time_key] for f in features],
          [float(r.n_cores) * f[time_key]
           for r, f in zip(rows, features)]]
    y = [r.npu_per_iter_uj for r in rows]
    coefs = _ols_multi(xs, y, intercept=True)
    preds = [coefs[0]*xs[0][k] + coefs[1]*xs[1][k] + coefs[2]
             for k in range(len(y))]
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)
    return EnergyFitResult(
        model="E-F",
        params={"p_base_uw": round(coefs[0]*1e6, 1),
                "p_pe_uw": round(coefs[1]*1e6, 1),
                "e_startup_uj": round(coefs[2], 2)},
        n_params=3, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Model E-G: E = P_col * N_col * t + E_startup  (2p)
# ---------------------------------------------------------------------------
def fit_model_eg(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    xs = [[float(f["n_columns"]) * f[time_key] for f in features]]
    y = [r.npu_per_iter_uj for r in rows]
    a, b = _ols_multi(xs, y, intercept=True)
    preds = [a * xs[0][k] + b for k in range(len(y))]
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)
    return EnergyFitResult(
        model="E-G",
        params={"p_col_uw": round(a*1e6, 1),
                "e_startup_uj": round(b, 2)},
        n_params=2, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Model E-H: E = (P_base + P_col * N_col) * t + E_startup  (3p)
# ---------------------------------------------------------------------------
def fit_model_eh(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    xs = [[f[time_key] for f in features],
          [float(f["n_columns"]) * f[time_key] for f in features]]
    y = [r.npu_per_iter_uj for r in rows]
    coefs = _ols_multi(xs, y, intercept=True)
    preds = [coefs[0]*xs[0][k] + coefs[1]*xs[1][k] + coefs[2]
             for k in range(len(y))]
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)
    return EnergyFitResult(
        model="E-H",
        params={"p_base_uw": round(coefs[0]*1e6, 1),
                "p_col_uw": round(coefs[1]*1e6, 1),
                "e_startup_uj": round(coefs[2], 2)},
        n_params=3, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Model E-I: E = (P_base + P_col*N_col + P_core*N) * t + E_startup  (4p)
# ---------------------------------------------------------------------------
def fit_model_ei(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    xs = [[f[time_key] for f in features],
          [float(f["n_columns"]) * f[time_key] for f in features],
          [float(r.n_cores) * f[time_key]
           for r, f in zip(rows, features)]]
    y = [r.npu_per_iter_uj for r in rows]
    coefs = _ols_multi(xs, y, intercept=True)
    preds = [coefs[0]*xs[0][k] + coefs[1]*xs[1][k]
             + coefs[2]*xs[2][k] + coefs[3]
             for k in range(len(y))]
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)
    return EnergyFitResult(
        model="E-I",
        params={"p_base_uw": round(coefs[0]*1e6, 1),
                "p_col_uw": round(coefs[1]*1e6, 1),
                "p_pe_uw": round(coefs[2]*1e6, 1),
                "e_startup_uj": round(coefs[3], 2)},
        n_params=4, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Log-space models — fit log(E) for proportional error minimization
# ---------------------------------------------------------------------------
def _fit_log_model(
    rows: List[ECalibRow],
    features: List[Dict],
    model_name: str,
    param_names: List[str],
    build_xs_fn,
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    """Generic log-space OLS fitter.

    build_xs_fn(rows, features, time_key) returns list of feature columns
    (each already log-transformed as needed).  The last coefficient is the
    intercept log(C).
    """
    y_raw = [r.npu_per_iter_uj for r in rows]
    log_y = [math.log(v) for v in y_raw]
    xs = build_xs_fn(rows, features, time_key)
    coefs = _ols_multi(xs, log_y, intercept=True)

    # Predict in log space, then exponentiate
    preds_log = [sum(coefs[j] * xs[j][k] for j in range(len(xs)))
                 + coefs[-1] for k in range(len(log_y))]
    preds = [math.exp(v) for v in preds_log]
    rho = spearman_rank_correlation(preds, y_raw)
    mape = compute_mape(preds, y_raw)

    params = {}
    for i, name in enumerate(param_names):
        params[name] = round(coefs[i], 6)
    # Last coef is log(C); store as C for interpretability
    params["C"] = round(math.exp(coefs[-1]), 4)

    return EnergyFitResult(
        model=model_name, params=params,
        n_params=len(param_names) + 1,  # +1 for intercept C
        rho=rho, mape=mape, predictions_uj=preds,
    )


def fit_model_log_t(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    """L-A: log(E) = a*log(t) + log(C)  ->  E = C * t^a  (2p)"""
    def build(rows, features, tk):
        return [[math.log(f[tk]) for f in features]]
    return _fit_log_model(rows, features, "L-A", ["a_time"], build,
                          time_key=time_key)


def fit_model_log_tn(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    """L-B: log(E) = a*log(t) + c*log(N) + log(C)  ->  E = C * t^a * N^c  (3p)"""
    def build(rows, features, tk):
        return [[math.log(f[tk]) for f in features],
                [math.log(r.n_cores) for r in rows]]
    return _fit_log_model(rows, features, "L-B",
                          ["a_time", "c_cores"], build, time_key=time_key)


def fit_model_log_tcol(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    """L-C: log(E) = a*log(t) + c*log(N_col) + log(C)  (3p)"""
    def build(rows, features, tk):
        return [[math.log(f[tk]) for f in features],
                [math.log(f["n_columns"]) for f in features]]
    return _fit_log_model(rows, features, "L-C",
                          ["a_time", "c_cols"], build, time_key=time_key)


def fit_model_log_tncol(
    rows: List[ECalibRow], features: List[Dict],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    """L-D: log(E) = a*log(t) + c1*log(N) + c2*log(N_col) + log(C)  (4p)"""
    def build(rows, features, tk):
        return [[math.log(f[tk]) for f in features],
                [math.log(r.n_cores) for r in rows],
                [math.log(f["n_columns"]) for f in features]]
    return _fit_log_model(rows, features, "L-D",
                          ["a_time", "c_cores", "c_cols"], build,
                          time_key=time_key)


# ---------------------------------------------------------------------------
# Weighted log-space models
# ---------------------------------------------------------------------------
def _active_fraction(r: ECalibRow, coeffs: Optional[CalibCoeffs]) -> float:
    """Fraction of t_total spent on compute + DMA (vs overhead)."""
    op = r.make_op()
    cand = r.make_cand()
    if coeffs and coeffs.calibrated:
        n_dma = _dma_ops_per_step(cand, r.tp_order_inner)
        t_comp, t_dma, t_ovh = _models.PerfModel.components_v9(
            macs=r.M * r.K * r.N,
            data_bytes=total_data_bytes(op, cand, r.tp_order_inner),
            n_cores=r.n_cores, tp_total=r.tp_total, n_dma=n_dma,
            eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
            l_sync=coeffs.l_sync_cy, l_pe=coeffs.l_pe_cy,
            l_dma=coeffs.l_dma_cy, l_startup=coeffs.l_startup_cy)
    else:
        t_comp = (r.M * r.K * r.N) / (r.n_cores * 256.0)
        t_dma = total_data_bytes(op, cand, r.tp_order_inner) / 4.0
        t_ovh = 20.0 * r.tp_total
    t_active = t_comp + t_dma
    return t_active / (t_active + t_ovh)


def _ols_weighted(
    xs: List[List[float]], y: List[float], w: List[float],
    *, intercept: bool = True,
) -> List[float]:
    """Weighted OLS: minimize sum(w_i * (y_i - X_i*beta)^2)."""
    cols = list(xs)
    if intercept:
        cols.append([1.0] * len(y))
    dim = len(cols)
    n = len(y)

    A = [[sum(w[k] * cols[i][k] * cols[j][k] for k in range(n))
          for j in range(dim)] for i in range(dim)]
    B = [sum(w[k] * cols[i][k] * y[k] for k in range(n)) for i in range(dim)]

    for col in range(dim):
        max_row = max(range(col, dim), key=lambda r: abs(A[r][col]))
        A[col], A[max_row] = A[max_row], A[col]
        B[col], B[max_row] = B[max_row], B[col]
        if abs(A[col][col]) < 1e-20:
            continue
        for row in range(col + 1, dim):
            factor = A[row][col] / A[col][col]
            for j in range(col, dim):
                A[row][j] -= factor * A[col][j]
            B[row] -= factor * B[col]

    beta = [0.0] * dim
    for i in range(dim - 1, -1, -1):
        if abs(A[i][i]) < 1e-20:
            continue
        beta[i] = (B[i] - sum(A[i][j] * beta[j]
                               for j in range(i + 1, dim))) / A[i][i]
    return beta


def fit_model_wlog_tn(
    rows: List[ECalibRow], features: List[Dict],
    coeffs: Optional[CalibCoeffs],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    """WL-B: Weighted log(E) = a*log(t) + c*log(N) + log(C)"""
    y_raw = [r.npu_per_iter_uj for r in rows]
    log_y = [math.log(v) for v in y_raw]
    xs = [[math.log(f[time_key]) for f in features],
          [math.log(r.n_cores) for r in rows]]
    weights = [_active_fraction(r, coeffs) for r in rows]

    coefs = _ols_weighted(xs, log_y, weights, intercept=True)
    preds_log = [coefs[0]*xs[0][k] + coefs[1]*xs[1][k] + coefs[2]
                 for k in range(len(log_y))]
    preds = [math.exp(v) for v in preds_log]
    rho = spearman_rank_correlation(preds, y_raw)
    mape = compute_mape(preds, y_raw)

    return EnergyFitResult(
        model="WL-B",
        params={"a_time": round(coefs[0], 6),
                "c_cores": round(coefs[1], 6),
                "C": round(math.exp(coefs[2]), 4)},
        n_params=3, rho=rho, mape=mape, predictions_uj=preds,
    )


def fit_model_wlog_tcol(
    rows: List[ECalibRow], features: List[Dict],
    coeffs: Optional[CalibCoeffs],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    """WL-C: Weighted log(E) = a*log(t) + c*log(N_col) + log(C)"""
    y_raw = [r.npu_per_iter_uj for r in rows]
    log_y = [math.log(v) for v in y_raw]
    xs = [[math.log(f[time_key]) for f in features],
          [math.log(f["n_columns"]) for f in features]]
    weights = [_active_fraction(r, coeffs) for r in rows]

    coefs = _ols_weighted(xs, log_y, weights, intercept=True)
    preds_log = [coefs[0]*xs[0][k] + coefs[1]*xs[1][k] + coefs[2]
                 for k in range(len(log_y))]
    preds = [math.exp(v) for v in preds_log]
    rho = spearman_rank_correlation(preds, y_raw)
    mape = compute_mape(preds, y_raw)

    return EnergyFitResult(
        model="WL-C",
        params={"a_time": round(coefs[0], 6),
                "c_cols": round(coefs[1], 6),
                "C": round(math.exp(coefs[2]), 4)},
        n_params=3, rho=rho, mape=mape, predictions_uj=preds,
    )


def fit_model_wlog_t(
    rows: List[ECalibRow], features: List[Dict],
    coeffs: Optional[CalibCoeffs],
    *, time_key: str = "t_total_us",
) -> EnergyFitResult:
    """WL-A: Weighted log(E) = a*log(t) + log(C) (no core term, baseline)"""
    y_raw = [r.npu_per_iter_uj for r in rows]
    log_y = [math.log(v) for v in y_raw]
    xs = [[math.log(f[time_key]) for f in features]]
    weights = [_active_fraction(r, coeffs) for r in rows]

    coefs = _ols_weighted(xs, log_y, weights, intercept=True)
    preds_log = [coefs[0]*xs[0][k] + coefs[1] for k in range(len(log_y))]
    preds = [math.exp(v) for v in preds_log]
    rho = spearman_rank_correlation(preds, y_raw)
    mape = compute_mape(preds, y_raw)

    return EnergyFitResult(
        model="WL-A",
        params={"a_time": round(coefs[0], 6),
                "C": round(math.exp(coefs[1]), 4)},
        n_params=2, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Theory-calibrated models (T-A, T-B, T-C) — scipy nonlinear optimization
# ---------------------------------------------------------------------------
def _predict_ta(params, macs_arr, bytes_arr, nt_arr):
    """T-A via models.EnergyModel.predict_ta_optim()."""
    return _models.EnergyModel.predict_ta_optim(params, {
        "macs": macs_arr, "data_bytes": bytes_arr, "nt": nt_arr,
    })


def _predict_tb(params, macs_arr, bytes_arr, t_arr, nt_arr):
    """T-B via models.EnergyModel.predict_tb_optim()."""
    return _models.EnergyModel.predict_tb_optim(params, {
        "macs": macs_arr, "data_bytes": bytes_arr,
        "t_total_cy": t_arr, "nt": nt_arr,
    })


def _predict_tc(params, macs_arr, bytes_arr, t_arr, nt_arr):
    """T-C via models.EnergyModel.predict_tc_optim()."""
    return _models.EnergyModel.predict_tc_optim(params, {
        "macs": macs_arr, "data_bytes": bytes_arr,
        "t_total_cy": t_arr, "nt": nt_arr,
    })


def _log_space_mse(preds, actuals):
    """Log-space MSE for proportional error minimization."""
    total = 0.0
    for p, a in zip(preds, actuals):
        if p > 0 and a > 0:
            diff = math.log(p) - math.log(a)
            total += diff * diff
    return total / len(actuals)


def fit_model_ta(
    rows: List[ECalibRow], features: List[Dict],
) -> EnergyFitResult:
    """T-A: Calibrated theoretical model (structure preserved, 3 constants fitted).

    E(uJ) = e_mac * MACs + e_dram * bytes + p_static * N_cores * T_total(cy)
    All terms in uJ.  Fitted via log-space MSE with scipy L-BFGS-B.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("[WARN] scipy not available, skipping T-A model", file=sys.stderr)
        return EnergyFitResult("T-A", {}, 3, float("nan"), float("nan"), [])

    y = [r.npu_per_iter_uj for r in rows]

    # Feature arrays (raw physical quantities, energy in uJ)
    # e_mac is in uJ/MAC, so MACs * e_mac = uJ
    macs_arr = [float(f["macs"]) for f in features]
    bytes_arr = [float(f["data_bytes"]) for f in features]
    nt_arr = [float(r.n_cores * f["t_total_cy"]) for r, f in zip(rows, features)]

    def objective(params):
        preds = _predict_ta(params, macs_arr, bytes_arr, nt_arr)
        return _log_space_mse(preds, y)

    # Initial values: theoretical × 500 (mid-range of 325-1390x underestimate)
    # Convert pJ constants to uJ: E_MAC_PJ * 1e-6 = uJ/MAC
    x0 = [E_MAC_PJ * 1e-6 * 500, E_DRAM_PJ * 1e-6 * 500, P_STATIC_PJ * 1e-6 * 500]
    bounds = [(1e-12, None), (1e-12, None), (1e-12, None)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    e_mac, e_dram, p_static = result.x

    preds = _predict_ta(result.x, macs_arr, bytes_arr, nt_arr)
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)

    return EnergyFitResult(
        model="T-A",
        params={
            "e_mac_uj": round(e_mac, 10),
            "e_dram_uj": round(e_dram, 10),
            "p_static_uj_cy": round(p_static, 10),
            # Also store in pJ for comparison with theoretical
            "e_mac_pj": round(e_mac * 1e6, 4),
            "e_dram_pj": round(e_dram * 1e6, 4),
            "p_static_pj": round(p_static * 1e6, 4),
        },
        n_params=3, rho=rho, mape=mape, predictions_uj=preds,
    )


def fit_model_tb(
    rows: List[ECalibRow], features: List[Dict],
) -> EnergyFitResult:
    """T-B: Base power + per-core power theoretical model (4 constants).

    E(uJ) = e_mac * MACs + e_dram * bytes + (p_base + p_pe * N) * T
           = e_mac * MACs + e_dram * bytes + p_base * T + p_pe * N * T
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("[WARN] scipy not available, skipping T-B model", file=sys.stderr)
        return EnergyFitResult("T-B", {}, 4, float("nan"), float("nan"), [])

    y = [r.npu_per_iter_uj for r in rows]
    macs_arr = [float(f["macs"]) for f in features]
    bytes_arr = [float(f["data_bytes"]) for f in features]
    t_arr = [float(f["t_total_cy"]) for f in features]
    nt_arr = [float(r.n_cores * f["t_total_cy"]) for r, f in zip(rows, features)]

    def objective(params):
        preds = _predict_tb(params, macs_arr, bytes_arr, t_arr, nt_arr)
        return _log_space_mse(preds, y)

    x0 = [E_MAC_PJ * 1e-6 * 500, E_DRAM_PJ * 1e-6 * 500,
           1e-3, P_STATIC_PJ * 1e-6 * 500]
    bounds = [(1e-12, None), (1e-12, None), (1e-12, None), (1e-12, None)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    e_mac, e_dram, p_base, p_pe = result.x

    preds = _predict_tb(result.x, macs_arr, bytes_arr, t_arr, nt_arr)
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)

    # Unit conversion: p_base (uJ/cy) → uW
    # uJ/cy * cy/us = uJ/us = W; * 1e6 = uW
    p_base_uw = p_base * CLOCK_MHZ * 1e6
    p_pe_uw = p_pe * CLOCK_MHZ * 1e6

    return EnergyFitResult(
        model="T-B",
        params={
            "e_mac_pj": round(e_mac * 1e6, 4),
            "e_dram_pj": round(e_dram * 1e6, 4),
            "p_base_uw": round(p_base_uw, 1),
            "p_pe_uw": round(p_pe_uw, 1),
        },
        n_params=4, rho=rho, mape=mape, predictions_uj=preds,
    )


def fit_model_tc(
    rows: List[ECalibRow], features: List[Dict],
) -> EnergyFitResult:
    """T-C: Base power + per-core power + startup energy (5 constants).

    E(uJ) = e_mac * MACs + e_dram * bytes + (p_base + p_pe * N) * T + e_startup
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("[WARN] scipy not available, skipping T-C model", file=sys.stderr)
        return EnergyFitResult("T-C", {}, 5, float("nan"), float("nan"), [])

    y = [r.npu_per_iter_uj for r in rows]
    macs_arr = [float(f["macs"]) for f in features]
    bytes_arr = [float(f["data_bytes"]) for f in features]
    t_arr = [float(f["t_total_cy"]) for f in features]
    nt_arr = [float(r.n_cores * f["t_total_cy"]) for r, f in zip(rows, features)]

    def objective(params):
        preds = _predict_tc(params, macs_arr, bytes_arr, t_arr, nt_arr)
        return _log_space_mse(preds, y)

    x0 = [E_MAC_PJ * 1e-6 * 500, E_DRAM_PJ * 1e-6 * 500,
           1e-3, P_STATIC_PJ * 1e-6 * 500, 100.0]
    bounds = [(1e-12, None), (1e-12, None), (1e-12, None), (1e-12, None), (0.0, None)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    e_mac, e_dram, p_base, p_pe, e_startup = result.x

    preds = _predict_tc(result.x, macs_arr, bytes_arr, t_arr, nt_arr)
    rho = spearman_rank_correlation(preds, y)
    mape = compute_mape(preds, y)

    p_base_uw = p_base * CLOCK_MHZ * 1e6
    p_pe_uw = p_pe * CLOCK_MHZ * 1e6

    return EnergyFitResult(
        model="T-C",
        params={
            "e_mac_pj": round(e_mac * 1e6, 4),
            "e_dram_pj": round(e_dram * 1e6, 4),
            "p_base_uw": round(p_base_uw, 1),
            "p_pe_uw": round(p_pe_uw, 1),
            "e_startup_uj": round(e_startup, 2),
        },
        n_params=5, rho=rho, mape=mape, predictions_uj=preds,
    )


# ---------------------------------------------------------------------------
# Theoretical model diagnosis (uncalibrated)
# ---------------------------------------------------------------------------
def diagnose_theoretical_model(
    rows: List[ECalibRow], features: List[Dict],
) -> None:
    """Print diagnostic report for the uncalibrated theoretical energy model."""
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  Theoretical Model Diagnosis (E_MAC={E_MAC_PJ}pJ, E_DRAM={E_DRAM_PJ}pJ, P_STATIC={P_STATIC_PJ}pJ)")
    print(sep)

    y = [r.npu_per_iter_uj for r in rows]

    # Predict with raw theoretical constants
    preds_uj = []
    ratios = []
    for r, f in zip(rows, features):
        op = r.make_op()
        cand = r.make_cand()
        macs = r.M * r.K * r.N
        data_bytes = total_data_bytes(op, cand, r.tp_order_inner)
        e_comp_pj = macs * E_MAC_PJ
        e_comm_pj = data_bytes * E_DRAM_PJ
        e_static_pj = r.n_cores * P_STATIC_PJ * f["t_total_cy"]
        e_total_uj = (e_comp_pj + e_comm_pj + e_static_pj) / 1e6
        preds_uj.append(e_total_uj)
        if e_total_uj > 0:
            ratios.append(r.npu_per_iter_uj / e_total_uj)

    rho = spearman_rank_correlation(preds_uj, y)
    mape = compute_mape(preds_uj, y)
    print(f"  Overall rho={rho:.4f}, MAPE={mape:.1f}%")
    if ratios:
        print(f"  Actual/Theoretical ratio: min={min(ratios):.0f}x, "
              f"max={max(ratios):.0f}x, median={sorted(ratios)[len(ratios)//2]:.0f}x")

    # Per-size breakdown
    size_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r, pred in zip(rows, preds_uj):
        size_data[r.size_key].append((pred, r.npu_per_iter_uj))

    print(f"\n  {'Size':<16s}  {'N':>4s}  {'rho':>8s}  {'Ratio(med)':>10s}")
    for sk in sorted(size_data.keys()):
        pairs = size_data[sk]
        sp = [p for p, _ in pairs]
        sa = [a for _, a in pairs]
        sr = spearman_rank_correlation(sp, sa) if len(pairs) >= 3 else float("nan")
        rs = sorted(a / p for p, a in pairs if p > 0)
        med = rs[len(rs) // 2] if rs else 0
        print(f"  {sk:<16s}  {len(pairs):>4d}  {sr:>8.4f}  {med:>9.0f}x")
    print(sep)


# ---------------------------------------------------------------------------
# EDP core-count optimal analysis
# ---------------------------------------------------------------------------
def edp_pe_optimal_analysis(
    rows: List[ECalibRow],
    all_fits: List[EnergyFitResult],
    features: List[Dict],
) -> None:
    """Compare EDP-optimal core count: model predictions vs measurement."""
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  EDP Core-Count Optimal Analysis")
    print(sep)

    # Group by size_key, picking TP=1 cases for clean comparison
    # (TP=1 means SPm*SPn = cores, TPm=TPk=TPn=1)
    size_core_data: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for i, r in enumerate(rows):
        # Collect measured time and energy per (size, cores)
        key = r.size_key
        nc = r.n_cores
        # Pick best (lowest energy) if multiple configs per (size, cores)
        if nc not in size_core_data[key] or r.npu_per_iter_uj < size_core_data[key][nc][1]:
            size_core_data[key][nc] = (r.avg_iter_us, r.npu_per_iter_uj)

    # Only analyze sizes with >= 2 core counts
    valid_sizes = {sk: d for sk, d in size_core_data.items() if len(d) >= 2}
    if not valid_sizes:
        print("  No sizes with multiple core counts — skipping")
        print(sep)
        return

    # For each model, find predicted EDP-optimal core count per size
    print(f"\n  {'Size':<16s}  {'Meas':>8s}", end="")
    model_names = [ft.model for ft in all_fits if ft.predictions_uj]
    for mn in model_names[:6]:  # limit columns
        print(f"  {mn:>8s}", end="")
    print()

    # Build per-row lookup: (size_key, case_index) -> model prediction
    row_preds: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict))
    for ft in all_fits:
        if not ft.predictions_uj:
            continue
        for i, (r, pred) in enumerate(zip(rows, ft.predictions_uj)):
            # Store by (size, n_cores): keep best (lowest energy) config
            key = r.size_key
            nc = r.n_cores
            if nc not in row_preds[ft.model][key] or pred < row_preds[ft.model][key][nc]:
                row_preds[ft.model][key][nc] = pred

    match_counts = defaultdict(int)
    total_sizes = 0

    for sk in sorted(valid_sizes.keys()):
        core_data = valid_sizes[sk]
        # Measured: EDP = E * T
        meas_edp = {nc: e * t for nc, (t, e) in core_data.items()}
        meas_opt = min(meas_edp, key=meas_edp.get)

        print(f"  {sk:<16s}  {meas_opt:>5d}c  ", end="")
        total_sizes += 1

        for ft in all_fits[:6]:
            if not ft.predictions_uj:
                print(f"  {'---':>8s}", end="")
                continue
            model_data = row_preds.get(ft.model, {}).get(sk, {})
            if not model_data:
                print(f"  {'---':>8s}", end="")
                continue

            # Model EDP: pred_energy * modeled_time
            # For fair comparison, use model's energy with actual measured time
            model_edp = {}
            for nc in core_data:
                if nc in model_data:
                    t_meas = core_data[nc][0]
                    model_edp[nc] = model_data[nc] * t_meas

            if model_edp:
                model_opt = min(model_edp, key=model_edp.get)
                marker = "*" if model_opt == meas_opt else " "
                print(f"  {model_opt:>5d}c{marker} ", end="")
                if model_opt == meas_opt:
                    match_counts[ft.model] += 1
            else:
                print(f"  {'---':>8s}", end="")
        print()

    # Summary
    print(f"\n  EDP-optimal core count match rate:")
    for ft in all_fits[:6]:
        if ft.predictions_uj:
            rate = match_counts[ft.model] / total_sizes * 100 if total_sizes > 0 else 0
            print(f"    {ft.model:>6s}: {match_counts[ft.model]}/{total_sizes} "
                  f"({rate:.0f}%)")
    print(sep)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def select_best_model(fits: List[EnergyFitResult]) -> EnergyFitResult:
    """Select best model. Prefer simpler model unless complex one is better.

    A more complex model is selected if:
      - rho improves by > 0.02  (strong ranking improvement), OR
      - rho improves AND mape improves by >= 3%  (both metrics better)
    """
    valid = [f for f in fits if f.predictions_uj and not math.isnan(f.rho)]
    if not valid:
        return fits[0]
    fits_sorted = sorted(valid, key=lambda f: (f.n_params, f.model))
    best = fits_sorted[0]
    for ft in fits_sorted[1:]:
        if ft.rho > best.rho + 0.02:
            best = ft
        elif ft.rho > best.rho and ft.mape < best.mape - 3.0:
            best = ft
    return best


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(
    fits: List[EnergyFitResult],
    best: EnergyFitResult,
    rows: List[ECalibRow],
    title: str = "Model Comparison",
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Cases: {len(rows)}")

    print(f"\n  {'Model':>5s}  {'Params':>6s}  {'rho':>8s}  {'MAPE':>8s}  {'Parameters':>40s}")
    for ft in fits:
        if not ft.predictions_uj:
            continue
        params_str = ", ".join(f"{k}={v}" for k, v in ft.params.items()
                               if not k.endswith("_uj") or k == "e_startup_uj")
        if len(params_str) > 55:
            params_str = params_str[:52] + "..."
        print(f"  {ft.model:>5s}  {ft.n_params:>6d}  {ft.rho:>8.4f}  {ft.mape:>7.1f}%  {params_str}")

    print(f"\n  Selected: {best.model}")

    # Per-size validation
    print(f"\n  --- Per-Size Validation ({best.model}) ---")
    size_groups: Dict[str, List[Tuple[ECalibRow, float]]] = {}
    for r, pred in zip(rows, best.predictions_uj):
        key = r.size_key
        if key not in size_groups:
            size_groups[key] = []
        size_groups[key].append((r, pred))

    print(f"  {'Size':<12s}  {'N':>4s}  {'rho':>8s}  {'MAPE':>8s}")
    for key in sorted(size_groups.keys()):
        group = size_groups[key]
        g_preds = [p for _, p in group]
        g_actual = [r.npu_per_iter_uj for r, _ in group]
        g_rho = spearman_rank_correlation(g_preds, g_actual) if len(group) >= 3 else float("nan")
        g_mape = compute_mape(g_preds, g_actual)
        print(f"  {key:<12s}  {len(group):>4d}  {g_rho:>8.4f}  {g_mape:>7.1f}%")

    # Per-core-count validation
    print(f"\n  --- Per-Core-Count Validation ({best.model}) ---")
    core_groups: Dict[int, List[Tuple[ECalibRow, float]]] = {}
    for r, pred in zip(rows, best.predictions_uj):
        nc = r.n_cores
        if nc not in core_groups:
            core_groups[nc] = []
        core_groups[nc].append((r, pred))

    print(f"  {'Cores':>5s}  {'Cols':>4s}  {'N':>4s}  {'rho':>8s}  {'MAPE':>8s}")
    for nc in sorted(core_groups.keys()):
        group = core_groups[nc]
        n_col = math.ceil(nc / _COMP_TILES_PER_COL)
        g_preds = [p for _, p in group]
        g_actual = [r.npu_per_iter_uj for r, _ in group]
        g_rho = (spearman_rank_correlation(g_preds, g_actual)
                 if len(group) >= 3 else float("nan"))
        g_mape = compute_mape(g_preds, g_actual)
        print(f"  {nc:>5d}  {n_col:>4d}  {len(group):>4d}  "
              f"{g_rho:>8.4f}  {g_mape:>7.1f}%")

    # Top 10 worst
    print(f"\n  --- Top 10 Worst Errors ({best.model}) ---")
    errors = []
    for r, pred in zip(rows, best.predictions_uj):
        actual = r.npu_per_iter_uj
        err = abs(pred - actual) / actual * 100 if actual > 0 else 0
        errors.append((r, pred, actual, err))
    errors.sort(key=lambda x: -x[3])

    print(f"  {'#':>4s}  {'Size':>12s}  {'cores':>5s}  {'cols':>4s}  "
          f"{'Pred':>10s}  {'Actual':>10s}  {'err%':>8s}")
    for r, pred, actual, err in errors[:10]:
        print(f"  {r.case_index:>4d}  {r.size_key:>12s}  {r.n_cores:>5d}  "
              f"{r.n_columns:>4d}  "
              f"{pred:>10.2f}  {actual:>10.2f}  {err:>7.1f}%")

    print(sep)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def update_calibration_json(
    existing_path: Path,
    out_path: Path,
    best: EnergyFitResult,
    n_samples: int,
    min_wall_s: float,
    gt_mode: str = "npu",
) -> None:
    """Update calibration.json with energy section."""
    if existing_path.is_file():
        with existing_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    else:
        doc = {}

    # Store only the model-essential parameters (not diagnostic ones)
    stored_params = {}
    for k, v in best.params.items():
        # Skip raw _uj / _uj_cy variants when pJ/uW versions exist
        if k.endswith("_uj") and k != "e_startup_uj":
            continue
        if k.endswith("_uj_cy"):
            continue
        stored_params[k] = v

    gt_labels = {
        "npu": "npu_per_iter_uj",
        "active_pkg": "active_pkg_per_iter_uj",
        "corrected": "corrected_npu_per_iter_uj",
    }

    doc["energy"] = {
        "model": best.model,
        "params": stored_params,
        "gt_mode": gt_mode,
        "fitted_from": {
            "n_samples": n_samples,
            "min_wall_s": min_wall_s,
            "spearman_rho": round(best.rho, 4),
            "mape_pct": round(best.mape, 1),
            "ground_truth": gt_labels.get(gt_mode, gt_mode),
        },
    }

    atomic_write_json(doc, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit energy model coefficients from NPU measurements")
    p.add_argument("--energy", required=True,
                    help="Path to energy_results.csv")
    p.add_argument("--tc", default="",
                    help="Path to tc_list.json (for tpOrder lookup)")
    p.add_argument("--calib", default=str(DEFAULT_CALIB_PATH),
                    help="Path to existing calibration.json (to merge into)")
    p.add_argument("--output", default="",
                    help="Path to write updated calibration.json (default: same as --calib)")
    p.add_argument("--min-wall-s", type=float, default=DEFAULT_MIN_WALL_S,
                    help=f"Minimum wall_elapsed_s for valid measurement (default: {DEFAULT_MIN_WALL_S})")
    p.add_argument("--max-wall-s", type=float, default=0.0,
                    help="Maximum wall_elapsed_s (0=no limit). Long measurements "
                         "amplify idle subtraction error.")
    p.add_argument("--gt-mode", choices=["npu", "active_pkg", "corrected"],
                    default="npu",
                    help="Ground truth mode: npu (idle-subtracted), "
                         "active_pkg (total package), corrected (global idle median)")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    global _COMP_TILES_PER_COL

    args = parse_args(argv)

    energy_path = Path(args.energy).resolve()
    calib_path = Path(args.calib).resolve()
    out_path = Path(args.output).resolve() if args.output else calib_path
    tc_path = Path(args.tc).resolve() if args.tc else None
    min_wall_s = args.min_wall_s
    max_wall_s = args.max_wall_s
    gt_mode = args.gt_mode

    # Load HW info for tiles-per-column
    try:
        sys_info = load_system_info(DEFAULT_SYS_PATH)
        _COMP_TILES_PER_COL = sys_info.comp_tiles_per_col
    except Exception:
        pass  # keep default = 4

    # Load data with validity filter
    rows, skipped = load_energy_data(energy_path, tc_path, min_wall_s,
                                     max_wall_s=max_wall_s, gt_mode=gt_mode)
    total_csv = len(rows) + len(skipped)

    # Step 0: Idle power stability analysis
    print_idle_power_analysis(rows, skipped)

    # Step 1: Validity report
    print_validity_report(rows, skipped, total_csv, min_wall_s, gt_mode=gt_mode)

    if not rows:
        print("[ERROR] No valid energy data", file=sys.stderr)
        return 1
    print(f"\n[INFO] {len(rows)} valid energy measurements (of {total_csv} total)")
    print(f"[INFO] tiles_per_col={_COMP_TILES_PER_COL}")

    # Load existing calibration for perf model features
    coeffs = load_calibration(calib_path)
    if coeffs.calibrated:
        print(f"[INFO] Perf calibration loaded: eff_macs={coeffs.eff_macs}")
    else:
        print("[WARN] No perf calibration — using defaults for feature computation")

    # Compute features
    features = compute_features(rows, coeffs)

    # Step 2: Diagnose theoretical model (uncalibrated)
    diagnose_theoretical_model(rows, features)

    # --- Fit all models ---

    # Linear models with MODELED time
    fits_linear = [
        fit_model_ea(rows, features),
        fit_model_eb(rows, features),
        fit_model_ec(rows, features),
        fit_model_ed(rows, features),
        fit_model_ef(rows, features),
        fit_model_eg(rows, features),
        fit_model_eh(rows, features),
        fit_model_ei(rows, features),
    ]
    best_linear = select_best_model(fits_linear)
    print_report(fits_linear, best_linear, rows, "Linear Models (Modeled Time)")

    # Log-space models with MODELED time
    fits_log = [
        fit_model_log_t(rows, features),
        fit_model_log_tn(rows, features),
        fit_model_log_tcol(rows, features),
        fit_model_log_tncol(rows, features),
    ]
    best_log = select_best_model(fits_log)
    print_report(fits_log, best_log, rows, "Log-Space Models (Modeled Time)")

    # Log-space models with MEASURED time
    fits_log_m = [
        fit_model_log_t(rows, features, time_key="t_measured_us"),
        fit_model_log_tn(rows, features, time_key="t_measured_us"),
        fit_model_log_tcol(rows, features, time_key="t_measured_us"),
        fit_model_log_tncol(rows, features, time_key="t_measured_us"),
    ]
    for ft in fits_log_m:
        ft.model = ft.model + "(m)"
    best_log_m = select_best_model(fits_log_m)
    print_report(fits_log_m, best_log_m, rows, "Log-Space Models (Measured Time)")

    # Weighted log-space models
    fits_wlog = [
        fit_model_wlog_t(rows, features, coeffs),
        fit_model_wlog_tn(rows, features, coeffs),
        fit_model_wlog_tcol(rows, features, coeffs),
    ]
    best_wlog = select_best_model(fits_wlog)
    print_report(fits_wlog, best_wlog, rows, "Weighted Log-Space Models (Modeled Time)")

    # Weighted log-space with MEASURED time
    fits_wlog_m = [
        fit_model_wlog_t(rows, features, coeffs, time_key="t_measured_us"),
        fit_model_wlog_tn(rows, features, coeffs, time_key="t_measured_us"),
        fit_model_wlog_tcol(rows, features, coeffs, time_key="t_measured_us"),
    ]
    for ft in fits_wlog_m:
        ft.model = ft.model + "(m)"
    best_wlog_m = select_best_model(fits_wlog_m)
    print_report(fits_wlog_m, best_wlog_m, rows, "Weighted Log-Space Models (Measured Time)")

    # Step 2-3: Theory-calibrated models (T-A, T-B, T-C)
    fits_theory = [
        fit_model_ta(rows, features),
        fit_model_tb(rows, features),
        fit_model_tc(rows, features),
    ]
    fits_theory_valid = [f for f in fits_theory if f.predictions_uj]
    if fits_theory_valid:
        best_theory = select_best_model(fits_theory_valid)
        print_report(fits_theory_valid, best_theory, rows,
                     "Theory-Calibrated Models (scipy, log-space MSE)")

    # --- Overall selection among all modeled-time models ---
    all_modeled = fits_linear + fits_log + fits_wlog + fits_theory_valid
    overall_best = select_best_model(all_modeled)
    print(f"\n{'='*70}")
    print(f"  OVERALL BEST (modeled time): {overall_best.model} "
          f"(rho={overall_best.rho:.4f}, MAPE={overall_best.mape:.1f}%)")
    print(f"{'='*70}")

    # Step 3-3: EDP core-count optimal analysis
    # Collect top models for comparison
    edp_models = [overall_best]
    for ft in [best_linear, best_log, best_wlog]:
        if ft.model != overall_best.model:
            edp_models.append(ft)
    if fits_theory_valid:
        for ft in fits_theory_valid:
            if ft.model != overall_best.model:
                edp_models.append(ft)
    edp_pe_optimal_analysis(rows, edp_models, features)

    # Write output
    update_calibration_json(calib_path, out_path, overall_best, len(rows),
                            min_wall_s, gt_mode=gt_mode)
    print(f"\n[INFO] Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
