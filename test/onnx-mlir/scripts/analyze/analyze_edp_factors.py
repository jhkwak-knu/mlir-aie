#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_edp_factors.py — Systematic factor analysis of EDP selection regret.

Analyzes R1/R2 measurement data to decompose regret into:
  1. Factor decomposition (core, SP shape, TP split, tpOrder)
  2. l_sync scaling behavior across TP_total ranges
  3. SP shape effect on performance and energy
  4. Per-DMA transfer overhead analysis
  5. tpOrder prediction accuracy

Usage:
    python3 scripts/analyze/analyze_edp_factors.py \
        --r1-result out/reports/result_edp_r1.csv \
        --r1-tc out/tc_list_edp_r1.json \
        --r2-result out/reports/result_edp_r2.csv \
        --r2-tc out/tc_list_edp_r2.json \
        --calib data/calibration.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import (  # noqa: E402
    CalibCoeffs, load_calibration, DEFAULT_CALIB_PATH,
)
from cost_model import total_data_bytes, Candidate, OpCase  # noqa: E402

CLOCK_MHZ = 1500
COMP_TILES_PER_COL = 4


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class FactorCase:
    case_index: int
    M: int; K: int; N: int
    SPm: int; SPn: int
    TPm: int; TPk: int; TPn: int
    TM: int; TK: int; TN: int
    num_cores: int
    tp_order_inner: int
    min_us: float
    avg_us: float
    npu_per_iter_uj: float
    wall_elapsed_s: float
    edp_rank: int
    edp_pred: float
    e_total_pred: float
    t_total_pred_cy: float

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.K}x{self.N}"

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn

    @property
    def sp_key(self) -> str:
        return f"({self.SPm},{self.SPn})"

    @property
    def tp_key(self) -> str:
        return f"({self.TPm},{self.TPk},{self.TPn})"

    @property
    def tpo_name(self) -> str:
        return ["M", "N", "K"][self.tp_order_inner]

    @property
    def has_energy(self) -> bool:
        return self.npu_per_iter_uj > 0 and self.wall_elapsed_s >= 0.005

    @property
    def sp_ratio(self) -> float:
        return max(self.SPm, self.SPn) / max(min(self.SPm, self.SPn), 1)

    def make_cand(self) -> Candidate:
        return Candidate(
            num_cores=self.num_cores,
            num_columns=math.ceil(self.num_cores / COMP_TILES_PER_COL),
            SPm=self.SPm, SPn=self.SPn,
            TPm=self.TPm, TPk=self.TPk, TPn=self.TPn,
            TM=self.TM, TK=self.TK, TN=self.TN,
        )

    def make_op(self) -> OpCase:
        return OpCase(M=self.M, K=self.K, N=self.N, elem_type="bf16")


def load_cases(csv_path: Path, tc_path: Path) -> List[FactorCase]:
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

            cases.append(FactorCase(
                case_index=ci,
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
                num_cores=int(row.get("numSpm", row.get("num_cores", "0"))),
                tp_order_inner=tpo[0],
                min_us=float(row["min_us"]),
                avg_us=float(row["avg_us"]),
                npu_per_iter_uj=float(row.get("npu_energy_per_iter_uj", "-1")),
                wall_elapsed_s=float(row.get("wall_elapsed_s", "0")),
                edp_rank=int(meta.get("edp_rank", 0)),
                edp_pred=float(meta.get("edp_pred", 0)),
                e_total_pred=float(meta.get("e_total_pred", 0)),
                t_total_pred_cy=float(row.get("t_total_pred", "0")),
            ))
    return cases


# ---------------------------------------------------------------------------
# Section 1: Factor Decomposition
# ---------------------------------------------------------------------------
def analyze_factor_decomposition(cases: List[FactorCase]) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  1. Factor Decomposition: Model Top-1 vs Actual Top-1")
    print(sep)

    by_size = defaultdict(list)
    for c in cases:
        by_size[c.size_key].append(c)

    print(f"\n  {'Size':<16s} {'T_reg':>7s}  {'Core':>5s}  {'SP':>8s}  {'TP':>12s}  "
          f"{'tpO':>4s}  Model_Top1 -> Actual_Top1")
    print(f"  {'-' * 74}")

    for sk in sorted(by_size):
        group = by_size[sk]
        actual_best = min(group, key=lambda c: c.min_us)
        model_best = min(group, key=lambda c: c.edp_rank if c.edp_rank > 0 else 9999)

        regret = ((model_best.min_us - actual_best.min_us) / actual_best.min_us * 100
                  if actual_best.min_us > 0 else 0)

        core_same = "=" if model_best.num_cores == actual_best.num_cores else "X"
        sp_same = "=" if model_best.sp_key == actual_best.sp_key else "X"
        tp_same = "=" if model_best.tp_key == actual_best.tp_key else "X"
        tpo_same = "=" if model_best.tp_order_inner == actual_best.tp_order_inner else "X"

        print(f"  {sk:<16s} {regret:>6.1f}%  {core_same:>5s}  {sp_same:>8s}  {tp_same:>12s}  "
              f"{tpo_same:>4s}  "
              f"{model_best.num_cores}c {model_best.sp_key} {model_best.tp_key} {model_best.tpo_name} "
              f"-> {actual_best.num_cores}c {actual_best.sp_key} {actual_best.tp_key} {actual_best.tpo_name}")

    # Stepwise regret decomposition
    print(f"\n  --- Stepwise Regret Decomposition ---")
    print(f"  For each size, fix factors one by one from model Top-1 to actual Top-1:")
    print(f"  {'Size':<16s} {'Total':>7s}  {'->fix_SP':>8s}  {'->fix_TP':>8s}  {'->fix_tpO':>8s}")
    print(f"  {'-' * 55}")

    for sk in sorted(by_size):
        group = by_size[sk]
        actual_best = min(group, key=lambda c: c.min_us)
        model_best = min(group, key=lambda c: c.edp_rank if c.edp_rank > 0 else 9999)

        total_reg = ((model_best.min_us - actual_best.min_us) / actual_best.min_us * 100
                     if actual_best.min_us > 0 else 0)

        # Find intermediate: same cores + actual SP + model TP/tpO
        step1 = None  # fix SP shape
        step2 = None  # fix TP split
        for c in group:
            if (c.num_cores == model_best.num_cores and
                c.sp_key == actual_best.sp_key and
                c.tp_key == model_best.tp_key and
                c.tp_order_inner == model_best.tp_order_inner):
                step1 = c
            if (c.num_cores == model_best.num_cores and
                c.sp_key == actual_best.sp_key and
                c.tp_key == actual_best.tp_key and
                c.tp_order_inner == model_best.tp_order_inner):
                step2 = c

        sp_reg = "n/a"
        tp_reg = "n/a"
        tpo_reg = "n/a"

        if step1 and model_best.min_us > 0:
            sp_contribution = (model_best.min_us - step1.min_us) / actual_best.min_us * 100
            sp_reg = f"{sp_contribution:+.1f}%"
        if step1 and step2:
            tp_contribution = (step1.min_us - step2.min_us) / actual_best.min_us * 100
            tp_reg = f"{tp_contribution:+.1f}%"
        if step2 and actual_best:
            tpo_contribution = (step2.min_us - actual_best.min_us) / actual_best.min_us * 100
            tpo_reg = f"{tpo_contribution:+.1f}%"

        print(f"  {sk:<16s} {total_reg:>6.1f}%  {sp_reg:>8s}  {tp_reg:>8s}  {tpo_reg:>8s}")


# ---------------------------------------------------------------------------
# Section 2: l_sync Scaling
# ---------------------------------------------------------------------------
def analyze_lsync_scaling(cases: List[FactorCase], coeffs: CalibCoeffs) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  2. l_sync Scaling: Predicted vs Measured Overhead")
    print(sep)

    if not coeffs.calibrated:
        print("  [WARN] No calibration loaded, skipping")
        return

    eff_macs = coeffs.eff_macs
    bw = coeffs.bw_eff_bpc
    l_sync = coeffs.l_sync_cy
    l_core = coeffs.l_core_cy
    l_startup = coeffs.l_startup_cy

    # Compute measured overhead for each case
    data_points = []
    for c in cases:
        op = c.make_op()
        cand = c.make_cand()
        t_comp_cy = (c.M * c.K * c.N) / (c.num_cores * eff_macs)
        t_comm_cy = total_data_bytes(op, cand, c.tp_order_inner) / bw
        t_pred_ovh_cy = l_sync * c.tp_total + l_core * c.num_cores + l_startup

        t_meas_cy = c.min_us * CLOCK_MHZ
        t_meas_ovh_cy = t_meas_cy - t_comp_cy - t_comm_cy

        data_points.append((c, t_pred_ovh_cy, t_meas_ovh_cy, t_comp_cy, t_comm_cy))

    # Group by TP_total ranges
    ranges = [(1, 1), (2, 4), (5, 16), (17, 64), (65, 256), (257, 1024),
              (1025, 4096), (4097, 100000)]

    print(f"\n  {'TP_total':>12s}  {'N':>4s}  {'pred_ovh(Kcy)':>14s}  "
          f"{'meas_ovh(Kcy)':>14s}  {'ratio':>7s}  {'pattern':>10s}")
    print(f"  {'-' * 70}")

    for lo, hi in ranges:
        pts = [(c, p, m) for c, p, m, _, _ in data_points
               if lo <= c.tp_total <= hi and m > 0]
        if not pts:
            continue

        pred_avg = sum(p for _, p, _ in pts) / len(pts)
        meas_avg = sum(m for _, _, m in pts) / len(pts)
        ratio = pred_avg / meas_avg if meas_avg > 0 else 0

        if ratio > 1.5:
            pattern = "OVER-pred"
        elif ratio < 0.7:
            pattern = "UNDER-pred"
        else:
            pattern = "OK"

        label = f"{lo}-{hi}" if hi < 100000 else f"{lo}+"
        print(f"  {label:>12s}  {len(pts):>4d}  {pred_avg/1000:>14.1f}  "
              f"{meas_avg/1000:>14.1f}  {ratio:>7.2f}  {pattern:>10s}")

    # Per-case scatter data for high TP
    print(f"\n  --- High TP_total cases (>= 64) ---")
    print(f"  {'ci':>4s} {'size':>10s} {'nc':>3s} {'TP_total':>8s}  "
          f"{'pred_ovh':>10s} {'meas_ovh':>10s} {'ratio':>7s}  "
          f"{'comp%':>6s} {'comm%':>6s} {'ovh%':>6s}")
    print(f"  {'-' * 80}")

    for c, pred_ovh, meas_ovh, t_comp, t_comm in sorted(data_points, key=lambda x: x[0].tp_total):
        if c.tp_total < 64:
            continue
        t_meas = c.min_us * CLOCK_MHZ
        ratio = pred_ovh / meas_ovh if meas_ovh > 0 else 0
        comp_pct = t_comp / t_meas * 100 if t_meas > 0 else 0
        comm_pct = t_comm / t_meas * 100 if t_meas > 0 else 0
        ovh_pct = meas_ovh / t_meas * 100 if t_meas > 0 else 0
        print(f"  {c.case_index:>4d} {c.size_key:>10s} {c.num_cores:>3d} {c.tp_total:>8d}  "
              f"{pred_ovh/1000:>9.0f}K {meas_ovh/1000:>9.0f}K {ratio:>7.2f}  "
              f"{comp_pct:>5.0f}% {comm_pct:>5.0f}% {ovh_pct:>5.0f}%")


# ---------------------------------------------------------------------------
# Section 3: SP Shape Effect
# ---------------------------------------------------------------------------
def analyze_sp_shape(cases: List[FactorCase]) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  3. SP Shape Effect (same size+cores+TP+tpOrder, different SP)")
    print(sep)

    # Group by (size, cores, TP, tpOrder) — SP is the varying factor
    groups: Dict[tuple, List[FactorCase]] = defaultdict(list)
    for c in cases:
        key = (c.size_key, c.num_cores, c.tp_key, c.tp_order_inner)
        groups[key].append(c)

    # Only keep groups with multiple SP shapes
    multi_sp = {k: v for k, v in groups.items() if len(set(c.sp_key for c in v)) >= 2}

    if not multi_sp:
        print("  No groups with multiple SP shapes found")
        return

    print(f"\n  {'Group':>45s}  {'N_sp':>4s}  {'min_us CV':>9s}  {'E CV':>7s}  "
          f"{'best_SP':>8s}  {'worst_SP':>9s}  {'gap%':>6s}")
    print(f"  {'-' * 100}")

    all_gaps = []
    all_sp_ratios_fast = []
    all_sp_ratios_slow = []

    for key in sorted(multi_sp):
        sk, nc, tp, tpo = key
        group = multi_sp[key]
        tpo_name = ["M", "N", "K"][tpo]

        # Per-SP stats
        sp_stats = defaultdict(list)
        for c in group:
            sp_stats[c.sp_key].append(c)

        times = {sp: min(c.min_us for c in cs) for sp, cs in sp_stats.items()}
        best_sp = min(times, key=times.get)
        worst_sp = max(times, key=times.get)
        gap = (times[worst_sp] - times[best_sp]) / times[best_sp] * 100

        all_min = [c.min_us for c in group]
        cv_t = (max(all_min) - min(all_min)) / min(all_min) * 100 if min(all_min) > 0 else 0

        e_vals = [c.npu_per_iter_uj for c in group if c.has_energy]
        cv_e = ""
        if len(e_vals) >= 2 and min(e_vals) > 0:
            cv_e = f"{(max(e_vals)-min(e_vals))/min(e_vals)*100:.0f}%"

        label = f"{sk} {nc}c {tp} {tpo_name}"
        print(f"  {label:>45s}  {len(sp_stats):>4d}  {cv_t:>8.1f}%  {cv_e:>7s}  "
              f"{best_sp:>8s}  {worst_sp:>9s}  {gap:>5.1f}%")

        all_gaps.append(gap)
        best_c = min(group, key=lambda c: c.min_us)
        worst_c = max(group, key=lambda c: c.min_us)
        all_sp_ratios_fast.append(best_c.sp_ratio)
        all_sp_ratios_slow.append(worst_c.sp_ratio)

    if all_gaps:
        print(f"\n  Summary: {len(all_gaps)} groups with multiple SP shapes")
        print(f"  Gap (best vs worst SP): mean={sum(all_gaps)/len(all_gaps):.1f}%, "
              f"max={max(all_gaps):.1f}%")
        print(f"  Best SP avg ratio (max/min): {sum(all_sp_ratios_fast)/len(all_sp_ratios_fast):.1f}")
        print(f"  Worst SP avg ratio (max/min): {sum(all_sp_ratios_slow)/len(all_sp_ratios_slow):.1f}")

    # SP ratio vs min_us correlation
    print(f"\n  --- SP Ratio vs Performance (all cases) ---")
    by_size_core = defaultdict(list)
    for c in cases:
        by_size_core[(c.size_key, c.num_cores)].append(c)

    for key in sorted(by_size_core):
        group = by_size_core[key]
        if len(group) < 5:
            continue
        ratios = [c.sp_ratio for c in group]
        times = [c.min_us for c in group]
        rho = spearman_rank_correlation(ratios, times)
        sk, nc = key
        print(f"  {sk} {nc:>2d}c: rho(sp_ratio, min_us)={rho:>6.3f}  (n={len(group)})")


# ---------------------------------------------------------------------------
# Section 4: Per-DMA Transfer Analysis
# ---------------------------------------------------------------------------
def compute_dma_ops(c: FactorCase) -> Tuple[int, int, int, float]:
    """Compute per-step DMA operation counts for a given configuration.

    Returns (n_lhs_ops, n_rhs_ops, n_total_ops, avg_transfer_bytes).
    """
    cols = math.ceil(c.num_cores / COMP_TILES_PER_COL)
    rows = COMP_TILES_PER_COL

    total_lhs = 0
    total_rhs = 0
    for col in range(cols):
        m_idxs = set()
        n_idxs = set()
        for i in range(rows):
            l_idx = rows * col + i
            m_idxs.add(l_idx % c.SPm)
            n_idxs.add(l_idx // c.SPm)
        total_lhs += len(m_idxs)
        total_rhs += len(n_idxs)

    total_ops = total_lhs + total_rhs
    op = c.make_op()
    cand = c.make_cand()
    total_bytes = total_data_bytes(op, cand, c.tp_order_inner) * op.elem_bytes
    avg_bytes = total_bytes / total_ops if total_ops > 0 else 0

    return total_lhs, total_rhs, total_ops, avg_bytes


def analyze_dma_transfers(cases: List[FactorCase], coeffs: CalibCoeffs) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  4. Per-DMA Transfer Overhead Analysis")
    print(sep)

    eff_macs = coeffs.eff_macs if coeffs.calibrated else 256.0
    bw = coeffs.bw_eff_bpc if coeffs.calibrated else 4.0

    # Compute DMA stats and residuals
    data = []
    for c in cases:
        n_lhs, n_rhs, n_ops, avg_bytes = compute_dma_ops(c)
        op = c.make_op()
        cand = c.make_cand()
        t_comp_cy = (c.M * c.K * c.N) / (c.num_cores * eff_macs)
        t_comm_cy = total_data_bytes(op, cand, c.tp_order_inner) / bw
        t_pred_cy = c.t_total_pred_cy
        t_meas_cy = c.min_us * CLOCK_MHZ
        residual = t_meas_cy - t_pred_cy

        data.append((c, n_lhs, n_rhs, n_ops, avg_bytes, residual, t_meas_cy))

    # Correlation: n_ops vs residual (within same size+cores)
    print(f"\n  --- DMA ops vs prediction residual (per size+cores group) ---")
    print(f"  {'Group':>20s}  {'N':>4s}  {'rho(ops,resid)':>14s}  {'rho(ops,min)':>13s}  "
          f"{'rho(avg_sz,min)':>15s}")
    print(f"  {'-' * 75}")

    by_group = defaultdict(list)
    for row in data:
        c = row[0]
        by_group[(c.size_key, c.num_cores)].append(row)

    for key in sorted(by_group):
        group = by_group[key]
        if len(group) < 5:
            continue
        ops = [r[3] for r in group]
        resids = [r[5] for r in group]
        times = [r[0].min_us for r in group]
        avg_sizes = [r[4] for r in group]

        rho_ops_resid = spearman_rank_correlation(ops, resids)
        rho_ops_time = spearman_rank_correlation(ops, times)
        rho_size_time = spearman_rank_correlation(avg_sizes, times)

        sk, nc = key
        print(f"  {sk} {nc:>2d}c{' ':>8s}  {len(group):>4d}  {rho_ops_resid:>14.3f}  "
              f"{rho_ops_time:>13.3f}  {rho_size_time:>15.3f}")


# ---------------------------------------------------------------------------
# Section 5: tpOrder Accuracy
# ---------------------------------------------------------------------------
def analyze_tporder(cases: List[FactorCase]) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  5. tpOrder Prediction Accuracy")
    print(sep)

    # Group by (size, cores, SP, TP) — tpOrder is varying
    groups: Dict[tuple, List[FactorCase]] = defaultdict(list)
    for c in cases:
        key = (c.size_key, c.num_cores, c.sp_key, c.tp_key)
        groups[key].append(c)

    multi_tpo = {k: v for k, v in groups.items()
                 if len(set(c.tp_order_inner for c in v)) >= 2}

    if not multi_tpo:
        print("  No groups with multiple tpOrders found")
        return

    correct = 0
    total = 0
    print(f"\n  {'Group':>40s}  {'tpOs':>5s}  {'pred_best':>9s}  {'meas_best':>9s}  {'match':>5s}")
    print(f"  {'-' * 75}")

    for key in sorted(multi_tpo):
        sk, nc, sp, tp = key
        group = multi_tpo[key]
        tpos = set(c.tp_order_inner for c in group)

        # Predicted best: lowest edp_rank
        pred_best = min(group, key=lambda c: c.edp_rank if c.edp_rank > 0 else 9999)
        # Measured best: lowest min_us
        meas_best = min(group, key=lambda c: c.min_us)

        match = pred_best.tp_order_inner == meas_best.tp_order_inner
        if match:
            correct += 1
        total += 1

        label = f"{sk} {nc}c {sp} {tp}"
        print(f"  {label:>40s}  {len(tpos):>5d}  {pred_best.tpo_name:>9s}  "
              f"{meas_best.tpo_name:>9s}  {'Y' if match else 'N':>5s}")

    if total > 0:
        print(f"\n  tpOrder accuracy: {correct}/{total} ({correct/total*100:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDP regret factor analysis")
    p.add_argument("--r1-result", default="", help="R1 result CSV")
    p.add_argument("--r1-tc", default="", help="R1 tc_list.json")
    p.add_argument("--r2-result", default="", help="R2 result CSV")
    p.add_argument("--r2-tc", default="", help="R2 tc_list.json")
    p.add_argument("--calib", default=str(DEFAULT_CALIB_PATH),
                   help="calibration.json")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    all_cases: List[FactorCase] = []
    if args.r1_result and args.r1_tc:
        r1 = load_cases(Path(args.r1_result), Path(args.r1_tc))
        all_cases.extend(r1)
        print(f"[INFO] R1: {len(r1)} cases loaded")
    if args.r2_result and args.r2_tc:
        r2 = load_cases(Path(args.r2_result), Path(args.r2_tc))
        all_cases.extend(r2)
        print(f"[INFO] R2: {len(r2)} cases loaded")

    if not all_cases:
        print("[ERROR] No cases loaded", file=sys.stderr)
        return 1

    coeffs = load_calibration(Path(args.calib))
    if coeffs.calibrated:
        print(f"[INFO] Calibration: eff_macs={coeffs.eff_macs}, "
              f"l_sync={coeffs.l_sync_cy}, l_core={coeffs.l_core_cy}")

    analyze_factor_decomposition(all_cases)
    analyze_lsync_scaling(all_cases, coeffs)
    analyze_sp_shape(all_cases)
    analyze_dma_transfers(all_cases, coeffs)
    analyze_tporder(all_cases)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
