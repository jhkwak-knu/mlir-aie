#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_dma_round.py — Fit L_DMA_ROUND coefficient and evaluate improvement.

New overhead model:
  T_overhead = (L_SYNC + L_DMA_ROUND * (SPm + SPn)) * TP_total
               + L_CORE * N_cores + L_STARTUP

Fits L_SYNC, L_DMA_ROUND, L_CORE, L_STARTUP from measurement data.
Evaluates on v9 (159 cases), R1 (139 cases), R2 (205 cases).

Usage:
    source ironenv/bin/activate
    cd test/onnx-mlir
    python3 scripts/analyze/fit_dma_round.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import (  # noqa: E402
    CalibCoeffs, load_calibration, DEFAULT_CALIB_PATH,
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
)
from cost_model import (  # noqa: E402
    total_data_bytes, Candidate, OpCase,
)
import models as _models  # noqa: E402

CLOCK_MHZ = 1500
COMP_TILES_PER_COL = 4

ROOT = Path(__file__).resolve().parents[2]  # test/onnx-mlir/


# ============================================================
# Data structures
# ============================================================
@dataclass
class MeasCase:
    """A single measurement case with all fields needed for fitting."""
    source: str  # "v9", "r1", "r2"
    case_index: int
    M: int; K: int; N: int
    SPm: int; SPn: int
    TPm: int; TPk: int; TPn: int
    TM: int; TK: int; TN: int
    num_cores: int
    tp_order_inner: int
    min_us: float
    t_total_pred_cy: float  # from CSV

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.N}"

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn

    @property
    def sp_sum(self) -> int:
        """Serialized DMA rounds = SPm + SPn."""
        return self.SPm + self.SPn

    @property
    def num_columns(self) -> int:
        return math.ceil(self.num_cores / COMP_TILES_PER_COL)

    def make_op(self) -> OpCase:
        return OpCase(M=self.M, K=self.K, N=self.N, elem_type="bf16")

    def make_cand(self) -> Candidate:
        return Candidate(
            num_cores=self.num_cores,
            num_columns=self.num_columns,
            SPm=self.SPm, SPn=self.SPn,
            TPm=self.TPm, TPk=self.TPk, TPn=self.TPn,
            TM=self.TM, TK=self.TK, TN=self.TN,
        )


# ============================================================
# Data loading
# ============================================================
def load_csv_cases(
    csv_path: Path, tc_path: Path, source: str,
) -> List[MeasCase]:
    """Load measurement cases from CSV + tc_list pair."""
    with tc_path.open() as f:
        tc_doc = json.load(f)
    tc_meta = {}
    for i, case in enumerate(tc_doc.get("cases", []), start=1):
        tc_meta[i] = case

    cases = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "PASS":
                continue
            ci = int(row["case_index"])
            meta = tc_meta.get(ci, {})
            lv = meta.get("levels", [{}])[0] if meta.get("levels") else {}
            tpo = lv.get("tpOrder", [2, 0, 1])

            cases.append(MeasCase(
                source=source,
                case_index=ci,
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
                num_cores=int(row.get("numSpm", row.get("num_cores", "0"))),
                tp_order_inner=tpo[0],
                min_us=float(row["min_us"]),
                t_total_pred_cy=float(row.get("t_total_pred", "0")),
            ))
    return cases


def load_all_data() -> Tuple[List[MeasCase], List[MeasCase], List[MeasCase]]:
    """Load v9, R1, R2 datasets."""
    v9 = load_csv_cases(
        ROOT / "out/reports/result_v9_clean.csv",
        ROOT / "out/calibration/tc_list_v3.json",
        "v9",
    )
    r1 = load_csv_cases(
        ROOT / "out/reports/result_edp_r1.csv",
        ROOT / "out/tc_list_edp_r1.json",
        "r1",
    )
    r2 = load_csv_cases(
        ROOT / "out/reports/result_edp_r2.csv",
        ROOT / "out/tc_list_edp_r2.json",
        "r2",
    )
    return v9, r1, r2


# ============================================================
# Model prediction
# ============================================================
def predict_t_total_cy(
    c: MeasCase, coeffs: CalibCoeffs, l_dma_round: float = 0.0,
) -> float:
    """Predict T_total in cycles using D+B model with optional L_DMA_ROUND."""
    op = c.make_op()
    cand = c.make_cand()

    t_comp = (c.M * c.K * c.N) / (c.num_cores * coeffs.eff_macs)
    t_comm = total_data_bytes(op, cand, c.tp_order_inner) / coeffs.bw_eff_bpc
    t_overhead = ((coeffs.l_sync_cy + l_dma_round * c.sp_sum) * c.tp_total
                  + coeffs.l_core_cy * c.num_cores
                  + coeffs.l_startup_cy)
    return t_comp + t_comm + t_overhead


def predict_with_params(
    c: MeasCase, eff_macs: float, bw: float,
    l_sync: float, l_dma_round: float, l_core: float, l_startup: float,
) -> float:
    """Predict T_total with explicit parameters (for fitting)."""
    op = c.make_op()
    cand = c.make_cand()

    t_comp = (c.M * c.K * c.N) / (c.num_cores * eff_macs)
    t_comm = total_data_bytes(op, cand, c.tp_order_inner) / bw
    t_overhead = ((l_sync + l_dma_round * c.sp_sum) * c.tp_total
                  + l_core * c.num_cores + l_startup)
    return t_comp + t_comm + t_overhead


# ============================================================
# Fitting
# ============================================================
def fit_overhead_params(
    cases: List[MeasCase], eff_macs: float, bw: float,
) -> Tuple[float, float, float, float]:
    """Fit L_SYNC, L_DMA_ROUND, L_CORE, L_STARTUP using log-space MSE.

    Fix EFF_MACS and BW (already calibrated from trace data).
    """
    # Compute measured overhead for each case
    measured_cy = []
    features = []  # (sp_sum * tp_total, tp_total, num_cores, 1)
    for c in cases:
        op = c.make_op()
        cand = c.make_cand()
        t_comp = (c.M * c.K * c.N) / (c.num_cores * eff_macs)
        t_comm = total_data_bytes(op, cand, c.tp_order_inner) / bw
        t_meas = c.min_us * CLOCK_MHZ
        t_overhead_meas = t_meas - t_comp - t_comm

        measured_cy.append(t_meas)
        features.append((
            c.sp_sum * c.tp_total,  # L_DMA_ROUND coefficient
            c.tp_total,              # L_SYNC coefficient
            c.num_cores,             # L_CORE coefficient
            1.0,                     # L_STARTUP coefficient
        ))

    measured_cy = np.array(measured_cy)
    features = np.array(features)

    # Objective: minimize sum of (log(pred) - log(meas))^2
    def objective(params):
        l_dma_round, l_sync, l_core, l_startup = params
        preds = []
        for i, c in enumerate(cases):
            op = c.make_op()
            cand = c.make_cand()
            t_comp = (c.M * c.K * c.N) / (c.num_cores * eff_macs)
            t_comm = total_data_bytes(op, cand, c.tp_order_inner) / bw
            t_ovh = ((l_sync + l_dma_round * c.sp_sum) * c.tp_total
                     + l_core * c.num_cores + l_startup)
            preds.append(t_comp + t_comm + t_ovh)

        preds = np.array(preds)
        # Clamp to avoid log(0)
        preds = np.maximum(preds, 1.0)
        log_err = np.log(preds) - np.log(measured_cy)
        return np.mean(log_err ** 2)

    # Initial guess: current v5 values, L_DMA_ROUND = 0
    x0 = [0.0, 34532.0, 7700.0, 46500.0]
    bounds = [(0, 500000), (0, 500000), (0, 500000), (0, 500000)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    l_dma_round, l_sync, l_core, l_startup = result.x
    return l_sync, l_dma_round, l_core, l_startup


# ============================================================
# Evaluation metrics
# ============================================================
def compute_metrics(
    cases: List[MeasCase], eff_macs: float, bw: float,
    l_sync: float, l_dma_round: float, l_core: float, l_startup: float,
    label: str = "",
) -> Dict[str, float]:
    """Compute rho, MAPE, Top-1 accuracy, regret for a set of cases."""
    preds = []
    meas = []
    for c in cases:
        p = predict_with_params(c, eff_macs, bw, l_sync, l_dma_round, l_core, l_startup)
        preds.append(p)
        meas.append(c.min_us * CLOCK_MHZ)

    rho = spearman_rank_correlation(preds, meas)

    # MAPE
    apes = []
    for p, m in zip(preds, meas):
        if m > 0:
            apes.append(abs(p - m) / m)
    mape = sum(apes) / len(apes) * 100 if apes else 0

    # Per-size Top-1/Top-3 and regret
    by_size = defaultdict(list)
    for i, c in enumerate(cases):
        by_size[c.size_key].append((c, preds[i], meas[i]))

    top1_correct = 0
    top3_correct = 0
    regrets = []
    sizes_total = 0

    for sk in sorted(by_size):
        group = by_size[sk]
        if len(group) < 2:
            continue
        sizes_total += 1

        # Actual best (lowest min_us)
        actual_best_time = min(m for _, _, m in group)

        # Model's predicted ranking
        sorted_by_pred = sorted(group, key=lambda x: x[1])
        model_top1_meas = sorted_by_pred[0][2]
        model_top3_meas = [x[2] for x in sorted_by_pred[:3]]

        if model_top1_meas <= actual_best_time * 1.001:
            top1_correct += 1

        if any(m <= actual_best_time * 1.001 for m in model_top3_meas):
            top3_correct += 1

        regret = (model_top1_meas - actual_best_time) / actual_best_time * 100
        regrets.append((sk, regret))

    return {
        "rho": rho,
        "mape": mape,
        "top1": top1_correct,
        "top3": top3_correct,
        "sizes": sizes_total,
        "regrets": regrets,
        "mean_regret": sum(r for _, r in regrets) / len(regrets) if regrets else 0,
        "max_regret": max(r for _, r in regrets) if regrets else 0,
    }


def print_metrics(metrics: Dict, label: str) -> None:
    """Pretty-print metrics."""
    print(f"\n  [{label}]")
    print(f"  rho={metrics['rho']:.4f}  MAPE={metrics['mape']:.1f}%")
    if metrics["sizes"] > 0:
        print(f"  Top-1={metrics['top1']}/{metrics['sizes']}  "
              f"Top-3={metrics['top3']}/{metrics['sizes']}")
        print(f"  Mean regret={metrics['mean_regret']:.1f}%  "
              f"Max regret={metrics['max_regret']:.1f}%")
    if metrics["regrets"]:
        for sk, r in metrics["regrets"]:
            print(f"    {sk}: {r:.1f}%")


# ============================================================
# Main
# ============================================================
def main():
    sep = "=" * 80

    print(f"\n{sep}")
    print("  L_DMA_ROUND Fitting & Evaluation")
    print(f"  T_overhead = (L_SYNC + L_DMA_ROUND*(SPm+SPn)) * TP_total")
    print(f"             + L_CORE * N_cores + L_STARTUP")
    print(sep)

    # Load data
    v9, r1, r2 = load_all_data()
    all_cases = v9 + r1 + r2
    print(f"\n  Data loaded: v9={len(v9)}, r1={len(r1)}, r2={len(r2)}, "
          f"total={len(all_cases)}")

    # Check SP diversity per dataset
    print(f"\n  SP diversity (unique sp_sum values per dataset):")
    for label, cases in [("v9", v9), ("r1", r1), ("r2", r2)]:
        sp_sums = sorted(set(c.sp_sum for c in cases))
        print(f"    {label}: {sp_sums}")

    # Load current calibration
    coeffs = load_calibration(DEFAULT_CALIB_PATH)
    eff_macs = coeffs.eff_macs
    bw = coeffs.bw_eff_bpc
    print(f"\n  EFF_MACS={eff_macs}, BW={bw} (fixed)")
    print(f"  Current v5: L_SYNC={coeffs.l_sync_cy}, L_CORE={coeffs.l_core_cy}, "
          f"L_STARTUP={coeffs.l_startup_cy}")

    # === Approach 1: Sweep L_DMA_ROUND with v5 params fixed ===
    print(f"\n{sep}")
    print("  Approach 1: Sweep L_DMA_ROUND [0..5000], other params fixed at v5")
    print(sep)

    sweep_values = list(range(0, 5001, 250))
    print(f"\n  {'L_DMA':>7s}  {'rho':>7s}  {'MAPE':>6s}  {'Top1':>5s}  {'Top3':>5s}  "
          f"{'MeanReg':>8s}  {'MaxReg':>8s}  Per-size regret")
    print(f"  {'-' * 90}")

    best_sweep_regret = (1e9, 0)
    for l_dma in sweep_values:
        m = compute_metrics(
            all_cases, eff_macs, bw,
            coeffs.l_sync_cy, float(l_dma), coeffs.l_core_cy, coeffs.l_startup_cy,
        )
        size_str = "  ".join(f"{sk}:{r:.0f}%" for sk, r in m["regrets"])
        print(f"  {l_dma:>7d}  {m['rho']:>7.4f}  {m['mape']:>5.1f}%  "
              f"{m['top1']:>2d}/{m['sizes']}  {m['top3']:>2d}/{m['sizes']}  "
              f"{m['mean_regret']:>7.1f}%  {m['max_regret']:>7.1f}%  {size_str}")
        if m["mean_regret"] < best_sweep_regret[0]:
            best_sweep_regret = (m["mean_regret"], l_dma)

    print(f"\n  Best mean regret: {best_sweep_regret[0]:.1f}% at L_DMA_ROUND={best_sweep_regret[1]}")

    # === Approach 2: Jointly refit L_SYNC + L_DMA_ROUND (L_CORE, L_STARTUP fixed) ===
    print(f"\n{sep}")
    print("  Approach 2: Refit L_SYNC + L_DMA_ROUND jointly (L_CORE, L_STARTUP fixed)")
    print(sep)

    l_core_fixed = coeffs.l_core_cy
    l_startup_fixed = coeffs.l_startup_cy
    measured_cy = np.array([c.min_us * CLOCK_MHZ for c in all_cases])

    # Precompute T_comp + T_comm for each case
    t_fixed = []
    for c in all_cases:
        op = c.make_op()
        cand = c.make_cand()
        tc = (c.M * c.K * c.N) / (c.num_cores * eff_macs)
        tm = total_data_bytes(op, cand, c.tp_order_inner) / bw
        t_fixed.append(tc + tm + l_core_fixed * c.num_cores + l_startup_fixed)
    t_fixed = np.array(t_fixed)

    tp_totals = np.array([c.tp_total for c in all_cases], dtype=float)
    sp_sums = np.array([c.sp_sum for c in all_cases], dtype=float)

    def obj_2param(params):
        l_sync, l_dma = params
        preds = t_fixed + (l_sync + l_dma * sp_sums) * tp_totals
        preds = np.maximum(preds, 1.0)
        return np.mean((np.log(preds) - np.log(measured_cy)) ** 2)

    # Try multiple starting points
    starts = [
        [34532, 0],
        [30000, 1000],
        [25000, 2000],
        [20000, 3000],
        [15000, 4000],
    ]
    best_2p = None
    best_2p_val = 1e18
    for x0 in starts:
        res = minimize(obj_2param, x0, method="L-BFGS-B",
                       bounds=[(0, 200000), (0, 50000)])
        if res.fun < best_2p_val:
            best_2p_val = res.fun
            best_2p = res.x

    l_sync_2p, l_dma_2p = best_2p
    print(f"\n  Fitted: L_SYNC={l_sync_2p:.1f}, L_DMA_ROUND={l_dma_2p:.1f}")
    print(f"  (L_CORE={l_core_fixed:.0f}, L_STARTUP={l_startup_fixed:.0f} fixed)")

    m_2p = compute_metrics(
        all_cases, eff_macs, bw,
        l_sync_2p, l_dma_2p, l_core_fixed, l_startup_fixed,
    )
    print_metrics(m_2p, "Approach 2 (all data)")

    # === Approach 3: Full 4-param refit with diverse starting points ===
    print(f"\n{sep}")
    print("  Approach 3: Full 4-param refit on all data (diverse starts)")
    print(sep)

    # Precompute for speed
    t_comp_comm = []
    for c in all_cases:
        op = c.make_op()
        cand = c.make_cand()
        tc = (c.M * c.K * c.N) / (c.num_cores * eff_macs)
        tm = total_data_bytes(op, cand, c.tp_order_inner) / bw
        t_comp_comm.append(tc + tm)
    t_comp_comm = np.array(t_comp_comm)
    n_cores = np.array([c.num_cores for c in all_cases], dtype=float)

    def obj_4param(params):
        l_sync, l_dma, l_core, l_startup = params
        preds = t_comp_comm + (l_sync + l_dma * sp_sums) * tp_totals + l_core * n_cores + l_startup
        preds = np.maximum(preds, 1.0)
        return np.mean((np.log(preds) - np.log(measured_cy)) ** 2)

    starts_4 = [
        [34532, 0, 7700, 46500],
        [30000, 1000, 7000, 45000],
        [25000, 2000, 5000, 40000],
        [20000, 3000, 8000, 50000],
        [10000, 5000, 10000, 30000],
        [34000, 500, 7700, 46500],
    ]
    best_4p = None
    best_4p_val = 1e18
    for x0 in starts_4:
        res = minimize(obj_4param, x0, method="L-BFGS-B",
                       bounds=[(0, 200000), (0, 50000), (0, 200000), (0, 200000)])
        if res.fun < best_4p_val:
            best_4p_val = res.fun
            best_4p = res.x

    l_sync_4, l_dma_4, l_core_4, l_startup_4 = best_4p
    print(f"\n  Fitted: L_SYNC={l_sync_4:.1f}, L_DMA_ROUND={l_dma_4:.1f}, "
          f"L_CORE={l_core_4:.1f}, L_STARTUP={l_startup_4:.1f}")

    m_4p = compute_metrics(
        all_cases, eff_macs, bw,
        l_sync_4, l_dma_4, l_core_4, l_startup_4,
    )
    print_metrics(m_4p, "Approach 3 (all data)")

    # === Approach 4: Rank-aware fitting ===
    # Maximize within-group Spearman rho instead of MSE
    print(f"\n{sep}")
    print("  Approach 4: Rank-aware L_DMA_ROUND (maximize within-group rho)")
    print(sep)

    # Group cases by (size, num_cores, tp_key) for within-group ranking
    groups_for_rank = defaultdict(list)
    for i, c in enumerate(all_cases):
        key = (c.size_key, c.num_cores,
               f"({c.TPm},{c.TPk},{c.TPn})", c.tp_order_inner)
        groups_for_rank[key].append(i)

    # Only keep groups with SP diversity (multiple sp_sum values)
    diverse_groups = {}
    for key, indices in groups_for_rank.items():
        sp_vals = set(all_cases[i].sp_sum for i in indices)
        if len(sp_vals) >= 2 and len(indices) >= 3:
            diverse_groups[key] = indices

    print(f"  Groups with SP diversity: {len(diverse_groups)} "
          f"({sum(len(v) for v in diverse_groups.values())} cases)")

    def rank_objective(l_dma):
        """Negative mean within-group Spearman rho (for minimization)."""
        rhos = []
        for key, indices in diverse_groups.items():
            preds_g = []
            meas_g = []
            for i in indices:
                c = all_cases[i]
                p = predict_with_params(
                    c, eff_macs, bw,
                    coeffs.l_sync_cy, l_dma, coeffs.l_core_cy, coeffs.l_startup_cy,
                )
                preds_g.append(p)
                meas_g.append(c.min_us)
            rho = spearman_rank_correlation(preds_g, meas_g)
            rhos.append(rho)
        return -np.mean(rhos) if rhos else 0.0

    # Sweep L_DMA_ROUND for rank objective
    print(f"\n  {'L_DMA':>7s}  {'group_rho':>10s}  {'rho_all':>8s}  "
          f"{'MAPE':>6s}  {'Top1':>5s}  {'MeanReg':>8s}  {'MaxReg':>8s}")
    print(f"  {'-' * 70}")

    best_rank_val = 1e9
    best_rank_dma = 0
    for l_dma in range(0, 5001, 250):
        rv = rank_objective(float(l_dma))
        m = compute_metrics(
            all_cases, eff_macs, bw,
            coeffs.l_sync_cy, float(l_dma), coeffs.l_core_cy, coeffs.l_startup_cy,
        )
        marker = ""
        if rv < best_rank_val:
            best_rank_val = rv
            best_rank_dma = l_dma
            marker = " <-- best rho"
        print(f"  {l_dma:>7d}  {-rv:>10.4f}  {m['rho']:>8.4f}  "
              f"{m['mape']:>5.1f}%  {m['top1']:>2d}/{m['sizes']}  "
              f"{m['mean_regret']:>7.1f}%  {m['max_regret']:>7.1f}%{marker}")

    print(f"\n  Best within-group rho at L_DMA_ROUND={best_rank_dma}")

    # Final best from rank-aware sweep
    m_rank = compute_metrics(
        all_cases, eff_macs, bw,
        coeffs.l_sync_cy, float(best_rank_dma), coeffs.l_core_cy, coeffs.l_startup_cy,
    )
    print_metrics(m_rank, f"Rank-aware (L_DMA_ROUND={best_rank_dma})")

    # === Summary ===
    print(f"\n{sep}")
    print("  Summary: All approaches on ALL data ({} cases)".format(len(all_cases)))
    print(sep)

    m_v5 = compute_metrics(
        all_cases, eff_macs, bw,
        coeffs.l_sync_cy, 0.0, coeffs.l_core_cy, coeffs.l_startup_cy,
    )

    results = [
        ("v5 (baseline)", m_v5,
         f"L_S={coeffs.l_sync_cy:.0f} L_D=0"),
        (f"Sweep (L_DMA={best_sweep_regret[1]})", compute_metrics(
            all_cases, eff_macs, bw,
            coeffs.l_sync_cy, float(best_sweep_regret[1]),
            coeffs.l_core_cy, coeffs.l_startup_cy),
         f"L_S={coeffs.l_sync_cy:.0f} L_D={best_sweep_regret[1]}"),
        ("2-param refit", m_2p,
         f"L_S={l_sync_2p:.0f} L_D={l_dma_2p:.0f}"),
        ("4-param refit", m_4p,
         f"L_S={l_sync_4:.0f} L_D={l_dma_4:.0f} L_C={l_core_4:.0f} L_U={l_startup_4:.0f}"),
        (f"Rank-aware (L_DMA={best_rank_dma})", m_rank,
         f"L_S={coeffs.l_sync_cy:.0f} L_D={best_rank_dma}"),
    ]

    print(f"\n  {'Approach':<25s}  {'rho':>7s}  {'MAPE':>6s}  {'Top1':>5s}  {'Top3':>5s}  "
          f"{'MeanR':>6s}  {'MaxR':>6s}  Params")
    print(f"  {'-' * 100}")
    for name, m, params in results:
        print(f"  {name:<25s}  {m['rho']:>7.4f}  {m['mape']:>5.1f}%  "
              f"{m['top1']:>2d}/{m['sizes']}  {m['top3']:>2d}/{m['sizes']}  "
              f"{m['mean_regret']:>5.1f}%  {m['max_regret']:>5.1f}%  {params}")

    # Per-size comparison: v5 vs best approach
    print(f"\n  Per-size regret (v5 vs best):")
    best_name, best_m, _ = min(results[1:], key=lambda x: x[1]["mean_regret"])
    print(f"  Best approach: {best_name}")
    print(f"  {'Size':<12s}  {'v5':>8s}  {'best':>8s}  {'delta':>8s}")
    print(f"  {'-' * 40}")
    v5_by = {sk: r for sk, r in m_v5["regrets"]}
    best_by = {sk: r for sk, r in best_m["regrets"]}
    for sk in sorted(set(list(v5_by.keys()) + list(best_by.keys()))):
        r5 = v5_by.get(sk, 0)
        rb = best_by.get(sk, 0)
        print(f"  {sk:<12s}  {r5:>7.1f}%  {rb:>7.1f}%  {rb-r5:>+7.1f}%")

    print()


if __name__ == "__main__":
    main()
