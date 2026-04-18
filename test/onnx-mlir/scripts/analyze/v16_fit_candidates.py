#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v16_fit_candidates.py -- Fit and compare all v16 candidate cost models.

For each candidate, runs scipy.optimize.differential_evolution with
log-space MSE objective, then evaluates MAPE/P90/rho and per-size breakdown.

Usage:
    python3 scripts/analyze/v16_fit_candidates.py \
        --result out/reports/result_v14_clean.csv \
        --tc out/tc_list_v14.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

# Resolve project paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "generate"))
from models import (  # noqa: E402
    CLOCK_MHZ,
    d_total as compute_d_total,
    total_data_bytes as compute_total_data_bytes,
    dma_ops_decomposed,
    tp_inner_value,
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EFF_MACS = 24.28       # Fixed from trace measurement
BW_BPC = 4.0           # Baseline fixed BW
ELEM_BYTES = 2         # bf16
TILES_PER_COL = 8      # XDNA2: 8 tiles per column
SEED = 42
MAXITER = 2000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CaseData:
    """Pre-computed features for one measurement case."""
    case_index: int
    SPm: int
    SPn: int
    TPm: int
    TPk: int
    TPn: int
    M: int
    K: int
    N: int
    tp_order_inner: int  # tpOrder[0]
    gt_time_us: float
    gt_energy_uj: float
    # Pre-computed features
    n_cores: int
    tp_total: int
    macs: int            # M*K*N (no *2)
    data_bytes: float
    d_total: float
    n_dma_every: float
    n_dma_reused: float
    tp_inner: int
    num_cols: int
    tp_outer: int        # TP / TP_inner

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.K}x{self.N}"


def load_cases(csv_path: Path, tc_path: Path) -> List[CaseData]:
    """Load CSV + tc_list, compute all features."""
    with tc_path.open("r", encoding="utf-8") as f:
        tc_doc = json.load(f)
    tc_cases = tc_doc["cases"]

    with csv_path.open("r", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]

    cases: List[CaseData] = []
    with io.StringIO("".join(lines)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "PASS":
                continue
            idx = int(row["case_index"])
            tc = tc_cases[idx - 1]
            lvl = tc["levels"][0]
            tp_order = lvl["tpOrder"]

            gt_time = float(row.get("batch_min_avg_us", "0"))
            gt_energy = float(row.get("batch_min_energy_per_iter_uj", "0"))
            if gt_time <= 0 or gt_energy <= 0:
                continue

            SPm, SPn = int(row["SPm"]), int(row["SPn"])
            TPm, TPk, TPn = int(row["TPm"]), int(row["TPk"]), int(row["TPn"])
            M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
            P = SPm * SPn
            TP = TPm * TPk * TPn
            inner = tp_order[0]

            data_bytes = compute_total_data_bytes(
                M, K, N, ELEM_BYTES, SPm, SPn, TPm, TPk, TPn, inner)
            d_tot = compute_d_total(SPm, SPn, P, TPm, TPk, TPn, TP, inner)
            n_every, n_reused = dma_ops_decomposed(SPm, SPn, P, inner)
            tp_inn = tp_inner_value(TPm, TPk, TPn, inner)

            cases.append(CaseData(
                case_index=idx,
                SPm=SPm, SPn=SPn,
                TPm=TPm, TPk=TPk, TPn=TPn,
                M=M, K=K, N=N,
                tp_order_inner=inner,
                gt_time_us=gt_time,
                gt_energy_uj=gt_energy,
                n_cores=P,
                tp_total=TP,
                macs=M * K * N,
                data_bytes=data_bytes,
                d_total=d_tot,
                n_dma_every=n_every,
                n_dma_reused=n_reused,
                tp_inner=tp_inn,
                num_cols=math.ceil(P / TILES_PER_COL),
                tp_outer=TP // tp_inn if tp_inn > 0 else TP,
            ))
    return cases


# ---------------------------------------------------------------------------
# Vectorized feature arrays (for fast optimization)
# ---------------------------------------------------------------------------
class FeatureArrays:
    """Pre-computed numpy arrays for all cases."""
    def __init__(self, cases: List[CaseData]):
        self.n = len(cases)
        self.gt_cy = np.array([c.gt_time_us * CLOCK_MHZ for c in cases])
        self.gt_us = np.array([c.gt_time_us for c in cases])
        self.gt_energy = np.array([c.gt_energy_uj for c in cases])
        self.macs = np.array([c.macs for c in cases], dtype=float)
        self.data_bytes = np.array([c.data_bytes for c in cases], dtype=float)
        self.n_cores = np.array([c.n_cores for c in cases], dtype=float)
        self.tp_total = np.array([c.tp_total for c in cases], dtype=float)
        self.d_total = np.array([c.d_total for c in cases], dtype=float)
        self.num_cols = np.array([c.num_cols for c in cases], dtype=float)
        self.tp_outer = np.array([c.tp_outer for c in cases], dtype=float)

        # T_comp (fixed, since eff_macs is fixed)
        self.t_comp = self.macs / (self.n_cores * EFF_MACS)
        # T_comm baseline (bw=4.0)
        self.t_comm_base = self.data_bytes / BW_BPC

        # Per-tpOrder masks
        self.mask_m = np.array([c.tp_order_inner == TP_AXIS_M for c in cases])
        self.mask_n = np.array([c.tp_order_inner == TP_AXIS_N for c in cases])
        self.mask_k = np.array([c.tp_order_inner == TP_AXIS_K for c in cases])

        # Core-squared
        self.n_cores_sq = self.n_cores ** 2

        # Sizes for per-size analysis
        self.size_keys = [c.size_key for c in cases]


# ---------------------------------------------------------------------------
# Performance model predict functions
# Each returns T_total in cycles as a numpy array
# ---------------------------------------------------------------------------

def predict_baseline(params: np.ndarray, fa: FeatureArrays) -> np.ndarray:
    """Baseline (v15): T = T_comp + T_comm + L_SYNC*TP + L_CORE*P*TP + L_DMA*D + L_STARTUP
    params: [l_sync, l_core, l_dma, l_startup]
    """
    l_sync, l_core, l_dma, l_startup = params
    overhead = (l_sync * fa.tp_total
                + l_core * fa.n_cores * fa.tp_total
                + l_dma * fa.d_total
                + l_startup)
    return fa.t_comp + fa.t_comm_base + overhead


def predict_p1_free_bw(params: np.ndarray, fa: FeatureArrays) -> np.ndarray:
    """P1: Free BW. params: [bw_fit, l_sync, l_core, l_dma, l_startup]"""
    bw_fit, l_sync, l_core, l_dma, l_startup = params
    t_comm = fa.data_bytes / bw_fit
    overhead = (l_sync * fa.tp_total
                + l_core * fa.n_cores * fa.tp_total
                + l_dma * fa.d_total
                + l_startup)
    return fa.t_comp + t_comm + overhead


def predict_p2_p_squared(params: np.ndarray, fa: FeatureArrays) -> np.ndarray:
    """P2: Superlinear sync. params: [l_sync, l_core, l_core_sq, l_dma, l_startup]"""
    l_sync, l_core, l_core_sq, l_dma, l_startup = params
    overhead = (l_sync * fa.tp_total
                + l_core * fa.n_cores * fa.tp_total
                + l_core_sq * fa.n_cores_sq * fa.tp_total
                + l_dma * fa.d_total
                + l_startup)
    return fa.t_comp + fa.t_comm_base + overhead


def predict_p3_bw_contention(params: np.ndarray, fa: FeatureArrays) -> np.ndarray:
    """P3: BW Contention. params: [bw_inv_base, bw_cont, l_sync, l_core, l_dma, l_startup]"""
    bw_inv_base, bw_cont, l_sync, l_core, l_dma, l_startup = params
    t_comm = fa.data_bytes * (bw_inv_base + bw_cont * fa.n_cores)
    overhead = (l_sync * fa.tp_total
                + l_core * fa.n_cores * fa.tp_total
                + l_dma * fa.d_total
                + l_startup)
    return fa.t_comp + t_comm + overhead


def predict_p4_per_order_dma(params: np.ndarray, fa: FeatureArrays) -> np.ndarray:
    """P4: Per-tpOrder DMA. params: [l_sync, l_core, l_dma_m, l_dma_n, l_dma_k, l_startup]"""
    l_sync, l_core, l_dma_m, l_dma_n, l_dma_k, l_startup = params
    # Apply per-axis L_DMA
    l_dma_arr = np.where(fa.mask_m, l_dma_m, np.where(fa.mask_n, l_dma_n, l_dma_k))
    overhead = (l_sync * fa.tp_total
                + l_core * fa.n_cores * fa.tp_total
                + l_dma_arr * fa.d_total
                + l_startup)
    return fa.t_comp + fa.t_comm_base + overhead


def predict_p5_column(params: np.ndarray, fa: FeatureArrays) -> np.ndarray:
    """P5: Column-aware. params: [l_sync, l_core, l_col, l_dma, l_startup]"""
    l_sync, l_core, l_col, l_dma, l_startup = params
    overhead = (l_sync * fa.tp_total
                + l_core * fa.n_cores * fa.tp_total
                + l_col * fa.num_cols * fa.tp_total
                + l_dma * fa.d_total
                + l_startup)
    return fa.t_comp + fa.t_comm_base + overhead


# ---------------------------------------------------------------------------
# Energy model predict functions
# Each returns E in uJ as a numpy array
# ---------------------------------------------------------------------------

def predict_e_baseline(params: np.ndarray, fa: FeatureArrays,
                       t_pred_us: np.ndarray) -> np.ndarray:
    """Baseline 1-G: E = (P_BASE + P_CORE*P)*T + E_DMA*D_total
    P_CORE = 40mW = 40000 uW fixed.
    params: [p_base_w, e_dma_uj]  (p_base in W for easier bounds)
    """
    p_base_w, e_dma = params
    p_base_uw = p_base_w * 1e6
    p_core_uw = 40000.0
    # (uW * us) = pJ -> convert to uJ by /1e6
    e_power_uj = (p_base_uw + p_core_uw * fa.n_cores) * t_pred_us / 1e6
    return e_power_uj + e_dma * fa.d_total


def predict_e1_free_pcore(params: np.ndarray, fa: FeatureArrays,
                          t_pred_us: np.ndarray) -> np.ndarray:
    """E1: Free P_CORE. params: [p_base_w, p_core_mw, e_dma_uj]"""
    p_base_w, p_core_mw, e_dma = params
    p_base_uw = p_base_w * 1e6
    p_core_uw = p_core_mw * 1e3
    e_power_uj = (p_base_uw + p_core_uw * fa.n_cores) * t_pred_us / 1e6
    return e_power_uj + e_dma * fa.d_total


def predict_e3_startup(params: np.ndarray, fa: FeatureArrays,
                       t_pred_us: np.ndarray) -> np.ndarray:
    """E3: Startup energy. params: [p_base_w, e_dma_uj, e_startup_uj]"""
    p_base_w, e_dma, e_startup = params
    p_base_uw = p_base_w * 1e6
    p_core_uw = 40000.0
    e_power_uj = (p_base_uw + p_core_uw * fa.n_cores) * t_pred_us / 1e6
    return e_power_uj + e_dma * fa.d_total + e_startup


def predict_e4_phase(params: np.ndarray, fa: FeatureArrays,
                     t_comp_us: np.ndarray, t_comm_us: np.ndarray,
                     t_overhead_us: np.ndarray) -> np.ndarray:
    """E4: Phase-decomposed. params: [p_comp_w, p_dma_w, p_idle_w, e_dma_uj]"""
    p_comp_w, p_dma_w, p_idle_w, e_dma = params
    e_comp_uj = p_comp_w * t_comp_us       # W * us = uJ
    e_dma_phase_uj = p_dma_w * t_comm_us
    e_idle_uj = p_idle_w * t_overhead_us
    return e_comp_uj + e_dma_phase_uj + e_idle_uj + e_dma * fa.d_total


# ---------------------------------------------------------------------------
# Optimization objective and metrics
# ---------------------------------------------------------------------------

def log_mse_objective(params, predict_fn, fa, gt_cy):
    """Log-space MSE: mean((log(pred) - log(gt))^2)."""
    pred = predict_fn(params, fa)
    # Guard against non-positive predictions
    pred = np.maximum(pred, 1.0)
    return np.mean((np.log(pred) - np.log(gt_cy)) ** 2)


def log_mse_energy_objective(params, predict_fn, fa, t_pred_us, gt_energy):
    """Log-space MSE for energy models."""
    pred = predict_fn(params, fa, t_pred_us)
    pred = np.maximum(pred, 0.001)
    return np.mean((np.log(pred) - np.log(gt_energy)) ** 2)


def log_mse_e4_objective(params, fa, t_comp_us, t_comm_us, t_overhead_us, gt_energy):
    """Log-space MSE for E4 (phase-decomposed)."""
    pred = predict_e4_phase(params, fa, t_comp_us, t_comm_us, t_overhead_us)
    pred = np.maximum(pred, 0.001)
    return np.mean((np.log(pred) - np.log(gt_energy)) ** 2)


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict:
    """Compute MAPE, P90, rho, bias."""
    ape = np.abs((gt - pred) / gt) * 100
    rho, _ = spearmanr(gt, pred)
    return {
        "mape": round(float(np.mean(ape)), 2),
        "p90": round(float(np.percentile(ape, 90)), 2),
        "rho": round(float(rho), 4),
        "bias": round(float(np.mean((pred - gt) / gt) * 100), 2),
    }


def per_size_mape(cases: List[CaseData], gt: np.ndarray, pred: np.ndarray) -> Dict:
    """Compute MAPE per problem size."""
    groups = defaultdict(list)
    for i, c in enumerate(cases):
        groups[c.size_key].append(i)

    result = {}
    for key in sorted(groups.keys()):
        idxs = groups[key]
        a = gt[idxs]
        p = pred[idxs]
        ape = np.abs((a - p) / a) * 100
        result[key] = round(float(np.mean(ape)), 1)
    return result


# ---------------------------------------------------------------------------
# Candidate definitions
# ---------------------------------------------------------------------------

@dataclass
class CandidateSpec:
    name: str
    category: str           # "perf" or "energy"
    param_names: List[str]
    bounds: List[Tuple[float, float]]
    predict_fn: Callable
    description: str


def get_perf_candidates() -> List[CandidateSpec]:
    return [
        CandidateSpec(
            name="Baseline",
            category="perf",
            param_names=["l_sync", "l_core", "l_dma", "l_startup"],
            bounds=[(100, 20000), (100, 5000), (100, 5000), (1000, 200000)],
            predict_fn=predict_baseline,
            description="v15 DMA-Refined (bw=4.0 fixed)",
        ),
        CandidateSpec(
            name="P1_FreeBW",
            category="perf",
            param_names=["bw_fit", "l_sync", "l_core", "l_dma", "l_startup"],
            bounds=[(1.0, 8.0), (100, 20000), (100, 5000), (100, 5000), (1000, 200000)],
            predict_fn=predict_p1_free_bw,
            description="Free bandwidth fitting",
        ),
        CandidateSpec(
            name="P2_P2Sync",
            category="perf",
            param_names=["l_sync", "l_core", "l_core_sq", "l_dma", "l_startup"],
            bounds=[(100, 20000), (0, 5000), (0, 200), (100, 5000), (1000, 200000)],
            predict_fn=predict_p2_p_squared,
            description="Superlinear sync (P^2*TP term)",
        ),
        CandidateSpec(
            name="P3_BWCont",
            category="perf",
            param_names=["bw_inv_base", "bw_cont", "l_sync", "l_core", "l_dma", "l_startup"],
            bounds=[(0.05, 1.0), (0.0, 0.05), (100, 20000), (100, 5000), (100, 5000), (1000, 200000)],
            predict_fn=predict_p3_bw_contention,
            description="P-dependent BW contention",
        ),
        CandidateSpec(
            name="P4_OrderDMA",
            category="perf",
            param_names=["l_sync", "l_core", "l_dma_m", "l_dma_n", "l_dma_k", "l_startup"],
            bounds=[(100, 20000), (100, 5000), (100, 10000), (100, 10000), (100, 10000), (1000, 200000)],
            predict_fn=predict_p4_per_order_dma,
            description="Per-tpOrder DMA cost split",
        ),
        CandidateSpec(
            name="P5_Column",
            category="perf",
            param_names=["l_sync", "l_core", "l_col", "l_dma", "l_startup"],
            bounds=[(100, 20000), (0, 5000), (0, 10000), (100, 5000), (1000, 200000)],
            predict_fn=predict_p5_column,
            description="Column-aware overhead",
        ),
    ]


def get_energy_candidates() -> List[CandidateSpec]:
    return [
        CandidateSpec(
            name="E_Baseline",
            category="energy",
            param_names=["p_base_w", "e_dma_uj"],
            bounds=[(1.0, 30.0), (0.1, 100.0)],
            predict_fn=predict_e_baseline,
            description="1-G (P_CORE=40mW fixed)",
        ),
        CandidateSpec(
            name="E1_FreePCORE",
            category="energy",
            param_names=["p_base_w", "p_core_mw", "e_dma_uj"],
            bounds=[(1.0, 30.0), (1.0, 500.0), (0.1, 100.0)],
            predict_fn=predict_e1_free_pcore,
            description="Free P_CORE fitting",
        ),
        CandidateSpec(
            name="E3_Startup",
            category="energy",
            param_names=["p_base_w", "e_dma_uj", "e_startup_uj"],
            bounds=[(1.0, 30.0), (0.1, 100.0), (0.1, 1000.0)],
            predict_fn=predict_e3_startup,
            description="Startup energy constant",
        ),
        # E4 (Phase-Decomposed) handled separately due to different interface
    ]


# ---------------------------------------------------------------------------
# Main fitting loop
# ---------------------------------------------------------------------------

def fit_perf_candidate(
    spec: CandidateSpec, fa: FeatureArrays,
) -> Tuple[Dict, np.ndarray]:
    """Fit one performance candidate. Returns (result_dict, pred_cy)."""
    t0 = time.time()
    result = differential_evolution(
        log_mse_objective,
        bounds=spec.bounds,
        args=(spec.predict_fn, fa, fa.gt_cy),
        seed=SEED,
        maxiter=MAXITER,
        tol=1e-8,
        atol=1e-8,
        updating="deferred",
        workers=-1,
    )
    elapsed = time.time() - t0

    pred_cy = spec.predict_fn(result.x, fa)
    pred_us = pred_cy / CLOCK_MHZ
    metrics = compute_metrics(fa.gt_us, pred_us)

    # Check bounds hit
    bounds_hit = []
    for i, (lo, hi) in enumerate(spec.bounds):
        if abs(result.x[i] - lo) < 1e-6 or abs(result.x[i] - hi) < 1e-6:
            bounds_hit.append(spec.param_names[i])

    return {
        "name": spec.name,
        "description": spec.description,
        "n_params": len(spec.param_names),
        "params": {k: round(float(v), 4) for k, v in zip(spec.param_names, result.x)},
        "metrics": metrics,
        "bounds_hit": bounds_hit,
        "nfev": result.nfev,
        "elapsed_s": round(elapsed, 1),
        "success": result.success,
    }, pred_cy


def fit_energy_candidate(
    spec: CandidateSpec, fa: FeatureArrays,
    t_pred_us: np.ndarray, label: str = "",
) -> Dict:
    """Fit one energy candidate. Returns result_dict."""
    t0 = time.time()
    result = differential_evolution(
        log_mse_energy_objective,
        bounds=spec.bounds,
        args=(spec.predict_fn, fa, t_pred_us, fa.gt_energy),
        seed=SEED,
        maxiter=MAXITER,
        tol=1e-8,
        atol=1e-8,
        updating="deferred",
        workers=-1,
    )
    elapsed = time.time() - t0

    pred_uj = spec.predict_fn(result.x, fa, t_pred_us)
    metrics = compute_metrics(fa.gt_energy, pred_uj)

    bounds_hit = []
    for i, (lo, hi) in enumerate(spec.bounds):
        if abs(result.x[i] - lo) < 1e-6 or abs(result.x[i] - hi) < 1e-6:
            bounds_hit.append(spec.param_names[i])

    name = f"{spec.name}({label})" if label else spec.name
    return {
        "name": name,
        "description": spec.description,
        "n_params": len(spec.param_names),
        "params": {k: round(float(v), 4) for k, v in zip(spec.param_names, result.x)},
        "metrics": metrics,
        "bounds_hit": bounds_hit,
        "nfev": result.nfev,
        "elapsed_s": round(elapsed, 1),
        "t_input": label,
    }


def fit_e4_phase(fa: FeatureArrays, best_perf_fn, best_perf_params,
                 label: str = "best_perf") -> Dict:
    """Fit E4 (Phase-Decomposed Power) separately."""
    # Compute component times from best perf model
    pred_cy = best_perf_fn(best_perf_params, fa)
    # For E4, we need T_comp, T_comm, T_overhead individually
    # T_comp and T_comm are the same across all perf models (eff_macs and bw fixed)
    t_comp_us = fa.t_comp / CLOCK_MHZ
    t_comm_us = fa.t_comm_base / CLOCK_MHZ
    t_overhead_us = (pred_cy - fa.t_comp - fa.t_comm_base) / CLOCK_MHZ
    # Clamp negative overhead (shouldn't happen but safety)
    t_overhead_us = np.maximum(t_overhead_us, 0.0)

    bounds = [(1.0, 30.0), (1.0, 30.0), (0.1, 20.0), (0.1, 100.0)]
    param_names = ["p_comp_w", "p_dma_w", "p_idle_w", "e_dma_uj"]

    t0 = time.time()
    result = differential_evolution(
        log_mse_e4_objective,
        bounds=bounds,
        args=(fa, t_comp_us, t_comm_us, t_overhead_us, fa.gt_energy),
        seed=SEED,
        maxiter=MAXITER,
        tol=1e-8,
        atol=1e-8,
        updating="deferred",
        workers=-1,
    )
    elapsed = time.time() - t0

    pred_uj = predict_e4_phase(result.x, fa, t_comp_us, t_comm_us, t_overhead_us)
    metrics = compute_metrics(fa.gt_energy, pred_uj)

    bounds_hit = []
    for i, (lo, hi) in enumerate(bounds):
        if abs(result.x[i] - lo) < 1e-6 or abs(result.x[i] - hi) < 1e-6:
            bounds_hit.append(param_names[i])

    return {
        "name": f"E4_Phase({label})",
        "description": "Phase-decomposed power (compute/DMA/idle)",
        "n_params": 4,
        "params": {k: round(float(v), 4) for k, v in zip(param_names, result.x)},
        "metrics": metrics,
        "bounds_hit": bounds_hit,
        "nfev": result.nfev,
        "elapsed_s": round(elapsed, 1),
        "t_input": label,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[Dict], category: str):
    """Print formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"  {category} Model Candidates Comparison")
    print(f"{'='*90}")
    print(f"{'Candidate':<25} {'Params':>6} {'MAPE':>8} {'P90':>8} {'rho':>8} {'Bias':>8} {'BoundsHit':>10}")
    print(f"{'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    baseline_mape = None
    for r in results:
        m = r["metrics"]
        bh = ",".join(r["bounds_hit"]) if r["bounds_hit"] else "-"
        if baseline_mape is None:
            baseline_mape = m["mape"]
            delta = "--"
        else:
            d = m["mape"] - baseline_mape
            delta = f"{d:+.1f}"

        print(f"{r['name']:<25} {r['n_params']:>6} {m['mape']:>7.1f}% {m['p90']:>7.1f}% "
              f"{m['rho']:>7.4f} {m['bias']:>+7.1f}% {bh:>10}")

    print()


def print_params_detail(results: List[Dict]):
    """Print fitted parameters for each candidate."""
    print(f"\n{'='*90}")
    print(f"  Fitted Parameters Detail")
    print(f"{'='*90}")
    for r in results:
        print(f"\n  {r['name']} ({r['description']}):")
        for k, v in r["params"].items():
            print(f"    {k}: {v}")
        if r["bounds_hit"]:
            print(f"    [WARNING] Bounds hit: {', '.join(r['bounds_hit'])}")


def print_per_size(cases: List[CaseData], results: List[Dict],
                   gt_arr: np.ndarray, pred_arrays: Dict[str, np.ndarray]):
    """Print per-size MAPE comparison across candidates."""
    # Get unique sizes
    sizes = sorted(set(c.size_key for c in cases))
    groups = defaultdict(list)
    for i, c in enumerate(cases):
        groups[c.size_key].append(i)

    print(f"\n{'='*90}")
    print(f"  Per-Size MAPE Comparison")
    print(f"{'='*90}")

    # Header
    names = [r["name"] for r in results]
    header = f"{'Size':<20}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print("-" * len(header))

    for size in sizes:
        idxs = groups[size]
        a = gt_arr[idxs]
        line = f"{size:<20}"
        for name in names:
            if name in pred_arrays:
                p = pred_arrays[name][idxs]
                m = float(np.mean(np.abs((a - p) / a)) * 100)
                line += f"{m:>11.1f}%"
            else:
                line += f"{'N/A':>12}"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="v16: Fit and compare candidate cost models")
    parser.add_argument("--result", required=True, help="Path to result_v14_clean.csv")
    parser.add_argument("--tc", required=True, help="Path to tc_list_v14.json")
    args = parser.parse_args()

    csv_path = Path(args.result)
    tc_path = Path(args.tc)

    # Load data
    cases = load_cases(csv_path, tc_path)
    fa = FeatureArrays(cases)
    print(f"Loaded {fa.n} cases")

    # =====================================================================
    # Part 1: Performance model candidates
    # =====================================================================
    print("\n" + "=" * 90)
    print("  FITTING PERFORMANCE MODEL CANDIDATES")
    print("=" * 90)

    perf_specs = get_perf_candidates()
    perf_results = []
    perf_preds = {}     # name -> pred_cy array
    best_perf_name = None
    best_perf_mape = float("inf")
    best_perf_fn = None
    best_perf_params = None

    for spec in perf_specs:
        print(f"\n  Fitting {spec.name} ({spec.description})...", flush=True)
        r, pred_cy = fit_perf_candidate(spec, fa)
        perf_results.append(r)
        perf_preds[spec.name] = pred_cy / CLOCK_MHZ  # store as us

        if r["metrics"]["mape"] < best_perf_mape:
            best_perf_mape = r["metrics"]["mape"]
            best_perf_name = spec.name
            best_perf_fn = spec.predict_fn
            best_perf_params = np.array([r["params"][k] for k in spec.param_names])

        print(f"    -> MAPE={r['metrics']['mape']:.1f}%, P90={r['metrics']['p90']:.1f}%, "
              f"rho={r['metrics']['rho']:.4f}, elapsed={r['elapsed_s']:.0f}s")

    # Print perf comparison
    print_comparison_table(perf_results, "Performance")
    print_params_detail(perf_results)
    print_per_size(cases, perf_results, fa.gt_us, perf_preds)

    print(f"\n  Best performance candidate: {best_perf_name} (MAPE={best_perf_mape:.1f}%)")

    # =====================================================================
    # Part 2: Energy model candidates
    # =====================================================================
    print("\n" + "=" * 90)
    print("  FITTING ENERGY MODEL CANDIDATES")
    print("=" * 90)

    # Compute T_pred from best perf model
    best_pred_cy = best_perf_fn(best_perf_params, fa)
    best_t_pred_us = best_pred_cy / CLOCK_MHZ

    energy_specs = get_energy_candidates()
    energy_results = []

    # Fit with best perf T_pred
    for spec in energy_specs:
        print(f"\n  Fitting {spec.name} with T_pred from {best_perf_name}...", flush=True)
        r = fit_energy_candidate(spec, fa, best_t_pred_us, label=best_perf_name)
        energy_results.append(r)
        print(f"    -> MAPE={r['metrics']['mape']:.1f}%, P90={r['metrics']['p90']:.1f}%, "
              f"rho={r['metrics']['rho']:.4f}")

    # E2: Oracle (T_meas)
    print(f"\n  Fitting E2_Oracle with T_meas (batch_min_avg_us)...", flush=True)
    e_baseline_spec = energy_specs[0]  # Use baseline spec for oracle
    r_oracle = fit_energy_candidate(e_baseline_spec, fa, fa.gt_us, label="T_meas")
    r_oracle["name"] = "E2_Oracle(T_meas)"
    r_oracle["description"] = "1-G with measured time (oracle)"
    energy_results.append(r_oracle)
    print(f"    -> MAPE={r_oracle['metrics']['mape']:.1f}%, P90={r_oracle['metrics']['p90']:.1f}%")

    # E1 with oracle too
    e1_spec = energy_specs[1]
    r_e1_oracle = fit_energy_candidate(e1_spec, fa, fa.gt_us, label="T_meas")
    r_e1_oracle["name"] = "E1_FreePCORE(T_meas)"
    energy_results.append(r_e1_oracle)
    print(f"\n  E1_FreePCORE with T_meas: MAPE={r_e1_oracle['metrics']['mape']:.1f}%")

    # E4: Phase-decomposed
    print(f"\n  Fitting E4_Phase with {best_perf_name}...", flush=True)
    r_e4 = fit_e4_phase(fa, best_perf_fn, best_perf_params, label=best_perf_name)
    energy_results.append(r_e4)
    print(f"    -> MAPE={r_e4['metrics']['mape']:.1f}%, P90={r_e4['metrics']['p90']:.1f}%")

    # Print energy comparison
    print_comparison_table(energy_results, "Energy")
    print_params_detail(energy_results)

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"\n  Best Performance: {best_perf_name} (MAPE={best_perf_mape:.1f}%)")

    best_e = min(energy_results, key=lambda r: r["metrics"]["mape"]
                 if "T_meas" not in r["name"] else float("inf"))
    print(f"  Best Energy (non-oracle): {best_e['name']} (MAPE={best_e['metrics']['mape']:.1f}%)")

    best_e_oracle = min(energy_results, key=lambda r: r["metrics"]["mape"])
    print(f"  Best Energy (oracle):     {best_e_oracle['name']} (MAPE={best_e_oracle['metrics']['mape']:.1f}%)")

    # Save results
    out_path = csv_path.parent / "v16_candidates.json"
    all_results = {
        "perf_candidates": perf_results,
        "energy_candidates": energy_results,
        "best_perf": best_perf_name,
        "best_energy": best_e["name"],
        "data_source": str(csv_path),
        "n_samples": fa.n,
        "eff_macs": EFF_MACS,
        "seed": SEED,
        "maxiter": MAXITER,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved results to {out_path}")


if __name__ == "__main__":
    main()
