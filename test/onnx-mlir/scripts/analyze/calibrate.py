#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibrate.py — Fit cost model coefficients from NPU measurement data.

Reads result.csv (batch runner output) + tc_list.json (tpOrder info),
fits EFF_MACS, BW_EFF, and overhead coefficients, then writes
data/calibration.json.

Seven overhead models are compared:
  A:  residual = L_SYNC * tp_total                            (BW fixed)
  B:  residual = L_SYNC * tp_total + L_STARTUP                (BW fixed)
  C:  residual = L_SYNC * tp_total + L_CONFIG * n_host_calls   (BW fixed)
  D:  residual = L_SYNC * tp_total + L_CORE * N_cores          (BW fixed)
  D+B: residual = L_SYNC * tp + L_CORE * N + L_STARTUP        (BW fixed)
  E:  y = (1/BW_FIT) * data + L_SYNC * tp_total                (BW fitted)
  F:  y = (1/BW_FIT) * data + L_SYNC * tp_total + L_CORE * N   (BW fitted)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import (  # noqa: E402
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
    OpCase,
    atomic_write_json,
)
from cost_model import total_data_bytes, Candidate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLOCK_MHZ = 1500        # XDNA2 tile clock (derived from TOPS spec)
BW_EFF_BPC = 4.0        # bytes/cycle (fixed, not fitted)
DEFAULT_EFF_MACS = 51.4  # fallback if no kernel trace data


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class CalibRow:
    """Merged CSV + tc_list entry for one calibration case."""
    case_index: int
    # Spatial / temporal config
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
    # Measurements
    actual_us: float     # ground truth (selected via --ground-truth)
    min_us: float
    avg_us: float
    t_total_pred: float
    ss_kernel_cy: int
    data_quality: float  # trace data quality (0.0-1.0+)
    # tpOrder from tc_list.json
    tp_order: List[int]  # [innermost, middle, outermost]

    @property
    def n_cores(self) -> int:
        return self.SPm * self.SPn

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn

    @property
    def n_host_calls(self) -> int:
        """Host outer loop iterations (model C only)."""
        tp_by_axis = {0: self.TPm, 1: self.TPn, 2: self.TPk}
        return tp_by_axis[self.tp_order[1]] * tp_by_axis[self.tp_order[2]]

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.K}x{self.N}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_and_merge(
    csv_path: Path,
    tc_path: Optional[Path] = None,
    ground_truth_col: str = "min_us",
    min_quality: float = 0.0,
    tc_archive_path: Optional[Path] = None,
) -> List[CalibRow]:
    """Load CSV + tiling config, merge on case_index, keep PASS rows only.

    Tiling config (tpOrder) can come from either:
      - tc_path: tc_list.json (legacy, 0-based indexing)
      - tc_archive_path: archive directory with per-case tc.json

    Args:
        ground_truth_col: CSV column to use as ground truth (default: min_us).
        min_quality: Minimum data_quality threshold; rows below are skipped.
        tc_archive_path: Path to archive directory (e.g. archive_v3/).
    """
    tc_cases = None
    if tc_path and not tc_archive_path:
        with tc_path.open("r", encoding="utf-8") as f:
            tc_doc = json.load(f)
        tc_cases = tc_doc["cases"]

    rows: List[CalibRow] = []
    skipped_quality = 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "PASS":
                continue

            # Quality filter (trace-based ground truth requires quality gate)
            dq = float(row.get("data_quality", "1.0"))
            if dq < min_quality:
                skipped_quality += 1
                continue

            idx = int(row["case_index"])

            # Load tiling config from archive or tc_list
            if tc_archive_path:
                tc_file = tc_archive_path / f"case_{idx:03d}" / "tc.json"
                if not tc_file.is_file():
                    continue
                with tc_file.open("r", encoding="utf-8") as f2:
                    tc = json.load(f2)
            elif tc_cases:
                tc = tc_cases[idx - 1]
            else:
                continue
            lvl = tc["levels"][0]

            gt_val = row.get(ground_truth_col, "")
            if not gt_val or gt_val in ("-", ""):
                continue
            actual_us = float(gt_val)
            if actual_us <= 0:
                continue

            rows.append(CalibRow(
                case_index=idx,
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                actual_us=actual_us,
                min_us=float(row["min_us"]),
                avg_us=float(row["avg_us"]),
                t_total_pred=float(row["t_total_pred"]),
                ss_kernel_cy=int(row.get("ss_kernel_cy", "0")),
                data_quality=dq,
                tp_order=lvl["tpOrder"],
            ))
    if skipped_quality > 0:
        print(f"[INFO] Skipped {skipped_quality} cases below min_quality={min_quality}")
    return rows


# ---------------------------------------------------------------------------
# Phase A: EFF_MACS
# ---------------------------------------------------------------------------
def fit_eff_macs(rows: List[CalibRow]) -> Tuple[float, List[float]]:
    """Compute per-case MACs/cycle from kernel trace, return median."""
    samples = []
    for r in rows:
        if r.ss_kernel_cy > 0:
            eff = (r.TM * r.TK * r.TN) / r.ss_kernel_cy
            samples.append(eff)
    if not samples:
        return DEFAULT_EFF_MACS, []
    samples.sort()
    n = len(samples)
    median = (samples[n // 2] + samples[(n - 1) // 2]) / 2.0
    return median, samples


# ---------------------------------------------------------------------------
# Phase C: Overhead fitting (3 models)
# ---------------------------------------------------------------------------
def compute_residuals(
    rows: List[CalibRow], eff_macs: float
) -> List[Tuple[CalibRow, float]]:
    """For each row, compute residual = T_actual - T_comp - T_dma (in cycles)."""
    results = []
    for r in rows:
        t_actual_cy = r.actual_us * CLOCK_MHZ
        t_comp = (r.M * r.K * r.N) / (r.n_cores * eff_macs)
        op = OpCase(M=r.M, K=r.K, N=r.N, elem_type="bf16")
        cand = Candidate(
            num_cores=r.n_cores, num_columns=0,
            SPm=r.SPm, SPn=r.SPn,
            TPm=r.TPm, TPk=r.TPk, TPn=r.TPn,
            TM=r.TM, TK=r.TK, TN=r.TN,
        )
        t_dma = total_data_bytes(op, cand, r.tp_order[0]) / BW_EFF_BPC
        residual = t_actual_cy - t_comp - t_dma
        results.append((r, residual))
    return results


@dataclass
class FitResult:
    """Result of fitting one overhead model."""
    model: str
    l_sync: float
    l_startup: float  # model B only
    l_config: float   # model C only
    l_core: float     # model D/F: per-core setup cost
    bw_eff: float     # model E/F: fitted DMA bandwidth (B/cycle)
    rho: float
    mape: float
    predictions: List[float]  # T_pred in us for each row


def fit_model_a(residuals: List[Tuple[CalibRow, float]]) -> Tuple[float, List[float]]:
    """Model A: residual = L_SYNC * tp_total (1-var, no intercept)."""
    sum_xy = sum(r.tp_total * res for r, res in residuals)
    sum_xx = sum(r.tp_total ** 2 for r, _ in residuals)
    l_sync = sum_xy / sum_xx if sum_xx > 0 else 0.0
    preds = [l_sync * r.tp_total for r, _ in residuals]
    return l_sync, preds


def fit_model_b(
    residuals: List[Tuple[CalibRow, float]]
) -> Tuple[float, float, List[float]]:
    """Model B: residual = L_SYNC * tp_total + L_STARTUP (2-var with intercept)."""
    n = len(residuals)
    sx = sum(r.tp_total for r, _ in residuals)
    sy = sum(res for _, res in residuals)
    sxx = sum(r.tp_total ** 2 for r, _ in residuals)
    sxy = sum(r.tp_total * res for r, res in residuals)
    # Standard OLS: y = a*x + b
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, 0.0, [0.0] * n
    l_sync = (n * sxy - sx * sy) / denom
    l_startup = (sy - l_sync * sx) / n
    preds = [l_sync * r.tp_total + l_startup for r, _ in residuals]
    return l_sync, l_startup, preds


def fit_model_c(
    residuals: List[Tuple[CalibRow, float]]
) -> Tuple[float, float, List[float]]:
    """Model C: residual = L_SYNC * tp_total + L_CONFIG * n_host_calls (2-var, no intercept)."""
    # x1 = tp_total, x2 = n_host_calls, y = residual
    s11 = sum(r.tp_total ** 2 for r, _ in residuals)
    s22 = sum(r.n_host_calls ** 2 for r, _ in residuals)
    s12 = sum(r.tp_total * r.n_host_calls for r, _ in residuals)
    s1y = sum(r.tp_total * res for r, res in residuals)
    s2y = sum(r.n_host_calls * res for r, res in residuals)
    # Cramer's rule
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return 0.0, 0.0, [0.0] * len(residuals)
    l_sync = (s1y * s22 - s2y * s12) / det
    l_config = (s2y * s11 - s1y * s12) / det
    preds = [l_sync * r.tp_total + l_config * r.n_host_calls
             for r, _ in residuals]
    return l_sync, l_config, preds


def compute_residuals_no_dma(
    rows: List[CalibRow], eff_macs: float
) -> List[Tuple[CalibRow, float, int]]:
    """Residual = T_actual - T_comp (no DMA subtracted), plus data_bytes per row."""
    results = []
    for r in rows:
        t_actual_cy = r.actual_us * CLOCK_MHZ
        t_comp = (r.M * r.K * r.N) / (r.n_cores * eff_macs)
        op = OpCase(M=r.M, K=r.K, N=r.N, elem_type="bf16")
        cand = Candidate(
            num_cores=r.n_cores, num_columns=0,
            SPm=r.SPm, SPn=r.SPn,
            TPm=r.TPm, TPk=r.TPk, TPn=r.TPn,
            TM=r.TM, TK=r.TK, TN=r.TN,
        )
        data_bytes = total_data_bytes(op, cand, r.tp_order[0])
        residual = t_actual_cy - t_comp
        results.append((r, residual, data_bytes))
    return results


def fit_model_d(
    residuals: List[Tuple[CalibRow, float]]
) -> Tuple[float, float, List[float]]:
    """Model D: residual = L_SYNC * tp_total + L_CORE * N_cores (2-var, no intercept)."""
    # x1 = tp_total, x2 = n_cores, y = residual
    s11 = sum(r.tp_total ** 2 for r, _ in residuals)
    s22 = sum(r.n_cores ** 2 for r, _ in residuals)
    s12 = sum(r.tp_total * r.n_cores for r, _ in residuals)
    s1y = sum(r.tp_total * res for r, res in residuals)
    s2y = sum(r.n_cores * res for r, res in residuals)
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return 0.0, 0.0, [0.0] * len(residuals)
    l_sync = (s1y * s22 - s2y * s12) / det
    l_core = (s2y * s11 - s1y * s12) / det
    preds = [l_sync * r.tp_total + l_core * r.n_cores
             for r, _ in residuals]
    return l_sync, l_core, preds


def fit_model_db(
    residuals: List[Tuple[CalibRow, float]],
    rows: List[CalibRow],
    eff_macs: float,
) -> Tuple[float, float, float, List[float]]:
    """Model D+B: residual = L_SYNC*tp + L_CORE*N_cores + L_STARTUP.

    2D grid search over (L_CORE, L_STARTUP) minimizing MAPE,
    with L_SYNC fitted via OLS at each grid point.
    Returns (l_sync, l_core, l_startup, overhead_preds).
    """
    # Precompute T_actual and (T_comp + T_dma) per row for fast MAPE eval
    t_actuals_cy = [r.actual_us * CLOCK_MHZ for r in rows]
    t_base_cy = [t_act - res for t_act, (_, res) in zip(t_actuals_cy, residuals)]

    def _mape_for(l_sync: float, l_core: int, l_startup: int) -> float:
        total = 0.0
        for i, (r, _) in enumerate(residuals):
            t_pred = t_base_cy[i] + l_sync * r.tp_total + l_core * r.n_cores + l_startup
            total += abs(t_pred - t_actuals_cy[i]) / t_actuals_cy[i]
        return total / len(residuals) * 100

    def _fit_sync(l_core: int, l_startup: int) -> float:
        sum_xy = sum(r.tp_total * (res - l_startup - l_core * r.n_cores)
                     for r, res in residuals)
        sum_xx = sum(r.tp_total ** 2 for r, _ in residuals)
        return sum_xy / sum_xx if sum_xx > 0 else 0.0

    # Coarse grid: L_CORE [0,30K] x L_STARTUP [0,120K]
    CORE_COARSE = range(0, 30001, 2000)
    STARTUP_COARSE = range(0, 120001, 10000)
    best_mape = float("inf")
    best_lc, best_ls = 0, 0
    for lc in CORE_COARSE:
        for ls in STARTUP_COARSE:
            lsync = _fit_sync(lc, ls)
            mape = _mape_for(lsync, lc, ls)
            if mape < best_mape:
                best_mape, best_lc, best_ls = mape, lc, ls

    # Fine grid: +/- 3K around best L_CORE, +/- 10K around best L_STARTUP
    CORE_FINE = range(max(0, best_lc - 3000), best_lc + 3001, 500)
    STARTUP_FINE = range(max(0, best_ls - 10000), best_ls + 10001, 2000)
    for lc in CORE_FINE:
        for ls in STARTUP_FINE:
            lsync = _fit_sync(lc, ls)
            mape = _mape_for(lsync, lc, ls)
            if mape < best_mape:
                best_mape, best_lc, best_ls = mape, lc, ls

    # Ultra-fine grid: +/- 1K around best L_CORE, +/- 2K around best L_STARTUP
    CORE_ULTRA = range(max(0, best_lc - 1000), best_lc + 1001, 100)
    STARTUP_ULTRA = range(max(0, best_ls - 2000), best_ls + 2001, 500)
    for lc in CORE_ULTRA:
        for ls in STARTUP_ULTRA:
            lsync = _fit_sync(lc, ls)
            mape = _mape_for(lsync, lc, ls)
            if mape < best_mape:
                best_mape, best_lc, best_ls = mape, lc, ls

    l_sync = _fit_sync(best_lc, best_ls)
    preds = [l_sync * r.tp_total + best_lc * r.n_cores + best_ls
             for r, _ in residuals]
    return l_sync, float(best_lc), float(best_ls), preds


def fit_model_e(
    residuals_no_dma: List[Tuple[CalibRow, float, int]]
) -> Tuple[float, float, float, List[float]]:
    """Model E: y = (1/BW_FIT) * data_bytes + L_SYNC * tp_total (2-var, no intercept).

    Returns (l_sync, bw_fit, inv_bw, overhead_preds).
    """
    # x1 = data_bytes, x2 = tp_total, y = residual (T_actual - T_comp)
    s11 = sum(d ** 2 for _, _, d in residuals_no_dma)
    s22 = sum(r.tp_total ** 2 for r, _, _ in residuals_no_dma)
    s12 = sum(d * r.tp_total for r, _, d in residuals_no_dma)
    s1y = sum(d * res for _, res, d in residuals_no_dma)
    s2y = sum(r.tp_total * res for r, res, _ in residuals_no_dma)
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return 0.0, BW_EFF_BPC, 0.0, [0.0] * len(residuals_no_dma)
    inv_bw = (s1y * s22 - s2y * s12) / det
    l_sync = (s2y * s11 - s1y * s12) / det
    # Guard against non-positive fitted bandwidth
    bw_fit = (1.0 / inv_bw) if inv_bw > 0 else BW_EFF_BPC
    preds = [inv_bw * d + l_sync * r.tp_total
             for r, _, d in residuals_no_dma]
    return l_sync, bw_fit, inv_bw, preds


def fit_model_f(
    residuals_no_dma: List[Tuple[CalibRow, float, int]]
) -> Tuple[float, float, float, float, List[float]]:
    """Model F: y = (1/BW_FIT)*data + L_SYNC*tp + L_CORE*N_cores (3-var, no intercept).

    Returns (l_sync, l_core, bw_fit, inv_bw, overhead_preds).
    """
    # x1 = data_bytes, x2 = tp_total, x3 = n_cores
    s11 = sum(d ** 2 for _, _, d in residuals_no_dma)
    s22 = sum(r.tp_total ** 2 for r, _, _ in residuals_no_dma)
    s33 = sum(r.n_cores ** 2 for r, _, _ in residuals_no_dma)
    s12 = sum(d * r.tp_total for r, _, d in residuals_no_dma)
    s13 = sum(d * r.n_cores for r, _, d in residuals_no_dma)
    s23 = sum(r.tp_total * r.n_cores for r, _, _ in residuals_no_dma)
    s1y = sum(d * res for _, res, d in residuals_no_dma)
    s2y = sum(r.tp_total * res for r, res, _ in residuals_no_dma)
    s3y = sum(r.n_cores * res for r, res, _ in residuals_no_dma)

    # 3x3 system via Cramer's rule: A*x = b
    # A = [[s11,s12,s13],[s12,s22,s23],[s13,s23,s33]]
    # b = [s1y, s2y, s3y]
    def det3(a):
        """Determinant of 3x3 matrix stored as list of 3 rows."""
        return (a[0][0] * (a[1][1]*a[2][2] - a[1][2]*a[2][1])
              - a[0][1] * (a[1][0]*a[2][2] - a[1][2]*a[2][0])
              + a[0][2] * (a[1][0]*a[2][1] - a[1][1]*a[2][0]))

    A = [[s11, s12, s13],
         [s12, s22, s23],
         [s13, s23, s33]]
    det_A = det3(A)
    if abs(det_A) < 1e-12:
        return 0.0, 0.0, BW_EFF_BPC, 0.0, [0.0] * len(residuals_no_dma)

    # Replace column 0 with b for inv_bw
    A1 = [[s1y, s12, s13],
          [s2y, s22, s23],
          [s3y, s23, s33]]
    inv_bw = det3(A1) / det_A

    # Replace column 1 with b for l_sync
    A2 = [[s11, s1y, s13],
          [s12, s2y, s23],
          [s13, s3y, s33]]
    l_sync = det3(A2) / det_A

    # Replace column 2 with b for l_core
    A3 = [[s11, s12, s1y],
          [s12, s22, s2y],
          [s13, s23, s3y]]
    l_core = det3(A3) / det_A

    bw_fit = (1.0 / inv_bw) if inv_bw > 0 else BW_EFF_BPC
    preds = [inv_bw * d + l_sync * r.tp_total + l_core * r.n_cores
             for r, _, d in residuals_no_dma]
    return l_sync, l_core, bw_fit, inv_bw, preds


def evaluate_model(
    rows: List[CalibRow],
    eff_macs: float,
    overhead_preds_cy: List[float],
    model_name: str,
    l_sync: float,
    l_startup: float = 0.0,
    l_config: float = 0.0,
    l_core: float = 0.0,
    bw_eff: float = BW_EFF_BPC,
) -> FitResult:
    """Compute T_pred (us) for each row and evaluate rho/MAPE.

    For models E/F, overhead_preds_cy already includes fitted DMA term,
    so bw_eff is stored but DMA is NOT added separately here.
    Models A/B/C/D use fixed BW and add DMA from total_data_bytes.
    """
    # Models with fitted BW embed DMA cost in overhead_preds_cy
    bw_fitted = bw_eff != BW_EFF_BPC
    t_preds_us = []
    t_actuals_us = []
    for i, r in enumerate(rows):
        t_comp = (r.M * r.K * r.N) / (r.n_cores * eff_macs)
        if bw_fitted:
            # DMA already included in overhead_preds_cy
            t_pred_cy = t_comp + overhead_preds_cy[i]
        else:
            op = OpCase(M=r.M, K=r.K, N=r.N, elem_type="bf16")
            cand = Candidate(
                num_cores=r.n_cores, num_columns=0,
                SPm=r.SPm, SPn=r.SPn,
                TPm=r.TPm, TPk=r.TPk, TPn=r.TPn,
                TM=r.TM, TK=r.TK, TN=r.TN,
            )
            t_dma = total_data_bytes(op, cand, r.tp_order[0]) / bw_eff
            t_pred_cy = t_comp + t_dma + overhead_preds_cy[i]
        t_pred_us = t_pred_cy / CLOCK_MHZ
        t_preds_us.append(t_pred_us)
        t_actuals_us.append(r.actual_us)

    rho = spearman_rank_correlation(t_preds_us, t_actuals_us)
    mape = compute_mape(t_preds_us, t_actuals_us)
    return FitResult(
        model=model_name,
        l_sync=l_sync, l_startup=l_startup, l_config=l_config,
        l_core=l_core, bw_eff=bw_eff,
        rho=rho, mape=mape, predictions=t_preds_us,
    )


def compute_mape(preds: List[float], actuals: List[float]) -> float:
    """Mean Absolute Percentage Error."""
    if not actuals:
        return float("nan")
    total = sum(abs(p - a) / a for p, a in zip(preds, actuals) if a > 0)
    n = sum(1 for a in actuals if a > 0)
    return (total / n * 100) if n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class SizeMetrics:
    """Per-size validation metrics."""
    n: int
    old_rho: float
    new_rho: float
    old_mape: float
    new_mape: float


def validate_by_size(
    rows: List[CalibRow],
    new_preds: List[float],
) -> Dict[str, SizeMetrics]:
    """Compute Spearman rho and MAPE grouped by matrix size."""
    # Group by size
    groups: Dict[str, List[Tuple[CalibRow, float]]] = {}
    for r, pred in zip(rows, new_preds):
        key = r.size_key
        if key not in groups:
            groups[key] = []
        groups[key].append((r, pred))

    metrics: Dict[str, SizeMetrics] = {}
    for key, group in sorted(groups.items()):
        old_preds = [r.t_total_pred for r, _ in group]
        new_preds_g = [p for _, p in group]
        actuals = [r.actual_us for r, _ in group]

        old_rho = spearman_rank_correlation(old_preds, actuals)
        new_rho = spearman_rank_correlation(new_preds_g, actuals)
        old_mape = compute_mape(old_preds, actuals)
        # Convert old_preds from cycles to us for fair MAPE comparison
        old_preds_us = [p / CLOCK_MHZ for p in old_preds]
        old_mape = compute_mape(old_preds_us, actuals)
        new_mape = compute_mape(new_preds_g, actuals)

        metrics[key] = SizeMetrics(
            n=len(group),
            old_rho=old_rho, new_rho=new_rho,
            old_mape=old_mape, new_mape=new_mape,
        )
    return metrics


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(
    eff_macs: float,
    eff_macs_samples: List[float],
    fits: List[FitResult],
    best: FitResult,
    rows: List[CalibRow],
    size_metrics: Dict[str, SizeMetrics],
    old_rho: float,
    old_mape: float,
) -> None:
    """Print calibration report to console."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Cost Model Calibration Report")
    print(sep)

    # Phase A
    n_eff = len(eff_macs_samples)
    if n_eff > 0:
        print(f"Phase A: EFF_MACS = {eff_macs:.1f} MACs/cycle  "
              f"(N={n_eff}, range [{min(eff_macs_samples):.1f}, "
              f"{max(eff_macs_samples):.1f}])")
    else:
        print(f"Phase A: EFF_MACS = {eff_macs:.1f} MACs/cycle  (default, no trace data)")

    # Phase B
    print(f"Phase B: BW_EFF = {BW_EFF_BPC:.1f} B/cycle (fixed)")

    # Phase C - all models
    print(f"\nPhase C: Overhead model comparison (N={len(rows)})")
    print(f"  {'Model':>5s}  {'L_SYNC':>12s}  {'L_STARTUP':>12s}  "
          f"{'L_CONFIG':>12s}  {'L_CORE':>12s}  {'BW_EFF':>8s}  "
          f"{'rho':>8s}  {'MAPE':>8s}")
    for ft in fits:
        bw_str = f"{ft.bw_eff:.2f}" if ft.bw_eff != BW_EFF_BPC else "fixed"
        print(f"  {ft.model:>5s}  {ft.l_sync:>12.0f}  {ft.l_startup:>12.0f}  "
              f"{ft.l_config:>12.0f}  {ft.l_core:>12.0f}  {bw_str:>8s}  "
              f"{ft.rho:>8.4f}  {ft.mape:>7.1f}%")

    print(f"\n  Selected: Model {best.model}")

    # Validation
    print(f"\n--- Validation ---")
    print(f"Overall:  old_rho={old_rho:.4f}  new_rho={best.rho:.4f}  "
          f"old_MAPE={old_mape:.1f}%  new_MAPE={best.mape:.1f}%")

    print(f"\n  {'Size':<12s}  {'N':>4s}  {'Old_rho':>8s}  {'New_rho':>8s}  "
          f"{'Old_MAPE':>9s}  {'New_MAPE':>9s}")
    for key, m in sorted(size_metrics.items()):
        print(f"  {key:<12s}  {m.n:>4d}  {m.old_rho:>8.4f}  {m.new_rho:>8.4f}  "
              f"{m.old_mape:>8.1f}%  {m.new_mape:>8.1f}%")

    # Per-case detail: top 10 worst absolute errors
    print(f"\n--- Per-case detail (top 10 worst error) ---")
    errors = []
    for r, pred in zip(rows, best.predictions):
        err_pct = abs(pred - r.actual_us) / r.actual_us * 100 if r.actual_us > 0 else 0
        errors.append((r, pred, err_pct))
    errors.sort(key=lambda x: -x[2])

    print(f"  {'#':>4s}  {'cores':>5s}  {'SP':>7s}  {'TP':>11s}  "
          f"{'T_pred':>10s}  {'T_actual':>10s}  {'error%':>8s}")
    for r, pred, err in errors[:10]:
        print(f"  {r.case_index:>4d}  {r.n_cores:>5d}  "
              f"({r.SPm:>2d},{r.SPn:>2d})  "
              f"({r.TPm:>2d},{r.TPk:>2d},{r.TPn:>2d})  "
              f"{pred:>10.1f}  {r.min_us:>10.1f}  {err:>7.1f}%")

    print(sep)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_calibration_json(
    out_path: Path,
    best: FitResult,
    eff_macs: float,
    n_samples: int,
    ground_truth_col: str = "min_us",
    min_quality: float = 0.0,
) -> None:
    """Write calibration.json with fitted coefficients."""
    fitted_from: Dict[str, Any] = {
        "n_samples": n_samples,
        "spearman_rho": round(best.rho, 4),
        "mape_pct": round(best.mape, 1),
        "ground_truth": ground_truth_col,
    }
    if min_quality > 0:
        fitted_from["min_quality"] = min_quality
    if ground_truth_col != "min_us":
        fitted_from["ground_truth_description"] = (
            "hw_timer-based NPU-only execution time"
            if ground_truth_col == "matmul_npu_us"
            else ground_truth_col
        )

    doc: Dict[str, Any] = {
        "version": 4,
        "target": "xdna2",
        "model": best.model,
        "eff_macs": round(eff_macs, 2),
        "bw_eff_bpc": round(best.bw_eff, 4),
        "l_sync_cy": round(best.l_sync),
        "l_startup_cy": round(best.l_startup),
        "l_config_cy": round(best.l_config),
        "l_core_cy": round(best.l_core),
        "clock_mhz": CLOCK_MHZ,
        "fitted_from": fitted_from,
    }
    atomic_write_json(doc, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def select_best_model(fits: List[FitResult]) -> FitResult:
    """Select best model. Prefer simpler model unless complex one is clearly better."""
    # Sort by model name (A < B < C) to prefer simpler
    fits_sorted = sorted(fits, key=lambda f: f.model)
    best = fits_sorted[0]  # default to A
    for ft in fits_sorted[1:]:
        # Require >0.02 rho improvement to justify extra complexity
        if ft.rho > best.rho + 0.02:
            best = ft
        # If equal rho, require >5% MAPE improvement
        elif abs(ft.rho - best.rho) <= 0.02 and ft.mape < best.mape - 5.0:
            best = ft
    return best


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Fit cost model coefficients from NPU measurements")
    parser.add_argument("--csv", required=True,
                        help="Path to result.csv")
    tc_group = parser.add_mutually_exclusive_group(required=True)
    tc_group.add_argument("--tc",
                          help="Path to tc_list.json")
    tc_group.add_argument("--tc-archive",
                          help="Path to archive directory (per-case tc.json)")
    parser.add_argument("--output", default="data/calibration.json",
                        help="Path to write calibration.json")
    parser.add_argument("--ground-truth", default="batch_min_avg_us",
                        help="CSV column for ground truth (default: batch_min_avg_us)")
    parser.add_argument("--min-quality", type=float, default=0.0,
                        help="Minimum data_quality threshold (default: 0.0)")
    parser.add_argument("--model", default="",
                        help="Force specific model (e.g. 'D+B') instead of auto-select")
    parser.add_argument("--fix-eff-macs", type=float, default=0.0,
                        help="Fix EFF_MACS to this value (skip Phase A fitting)")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv).resolve()
    tc_path = Path(args.tc).resolve() if args.tc else None
    tc_archive_path = Path(args.tc_archive).resolve() if args.tc_archive else None
    out_path = Path(args.output).resolve()
    gt_col = args.ground_truth
    min_q = args.min_quality

    # Load data
    rows = load_and_merge(csv_path, tc_path, gt_col, min_q,
                          tc_archive_path=tc_archive_path)
    if not rows:
        print("[ERROR] No PASS rows found", file=sys.stderr)
        return 1
    print(f"[INFO] Loaded {len(rows)} PASS cases")

    # Phase A: EFF_MACS
    if args.fix_eff_macs > 0:
        eff_macs = args.fix_eff_macs
        eff_samples = []
        print(f"[INFO] EFF_MACS fixed at {eff_macs:.2f} (--fix-eff-macs)")
    else:
        eff_macs, eff_samples = fit_eff_macs(rows)

    # Phase C: Overhead fitting
    residuals = compute_residuals(rows, eff_macs)

    # Model A
    l_sync_a, ovh_a = fit_model_a(residuals)
    fit_a = evaluate_model(rows, eff_macs, ovh_a, "A", l_sync_a)

    # Model B
    l_sync_b, l_startup_b, ovh_b = fit_model_b(residuals)
    fit_b = evaluate_model(rows, eff_macs, ovh_b, "B",
                           l_sync_b, l_startup=l_startup_b)

    # Model C
    l_sync_c, l_config_c, ovh_c = fit_model_c(residuals)
    fit_c = evaluate_model(rows, eff_macs, ovh_c, "C",
                           l_sync_c, l_config=l_config_c)

    # Model D: per-core setup cost (BW fixed)
    l_sync_d, l_core_d, ovh_d = fit_model_d(residuals)
    fit_d = evaluate_model(rows, eff_macs, ovh_d, "D",
                           l_sync_d, l_core=l_core_d)

    # Model D+B: per-core cost + startup (grid search)
    l_sync_db, l_core_db, l_startup_db, ovh_db = fit_model_db(
        residuals, rows, eff_macs)
    fit_db = evaluate_model(rows, eff_macs, ovh_db, "D+B",
                            l_sync_db, l_startup=l_startup_db,
                            l_core=l_core_db)

    # Model E: fitted DMA bandwidth (no per-core cost)
    residuals_nd = compute_residuals_no_dma(rows, eff_macs)
    l_sync_e, bw_e, _, ovh_e = fit_model_e(residuals_nd)
    fit_e = evaluate_model(rows, eff_macs, ovh_e, "E",
                           l_sync_e, bw_eff=bw_e)

    # Model F: fitted BW + per-core cost
    l_sync_f, l_core_f, bw_f, _, ovh_f = fit_model_f(residuals_nd)
    fit_f = evaluate_model(rows, eff_macs, ovh_f, "F",
                           l_sync_f, l_core=l_core_f, bw_eff=bw_f)

    fits = [fit_a, fit_b, fit_c, fit_d, fit_db, fit_e, fit_f]
    if args.model:
        forced = [f for f in fits if f.model == args.model]
        if not forced:
            print(f"[ERROR] Model '{args.model}' not found in: "
                  f"{[f.model for f in fits]}", file=sys.stderr)
            return 1
        best = forced[0]
        print(f"[INFO] Forced model selection: {args.model}")
    else:
        best = select_best_model(fits)

    # Old model comparison (t_total_pred is in cycles)
    old_preds_us = [r.t_total_pred / CLOCK_MHZ for r in rows]
    actuals = [r.actual_us for r in rows]
    old_rho = spearman_rank_correlation(old_preds_us, actuals)
    old_mape = compute_mape(old_preds_us, actuals)

    # Per-size validation
    size_metrics = validate_by_size(rows, best.predictions)

    # Report
    print_report(eff_macs, eff_samples, fits, best, rows,
                 size_metrics, old_rho, old_mape)

    # Write output
    write_calibration_json(out_path, best, eff_macs, len(rows), gt_col, min_q)
    print(f"\n[INFO] Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
