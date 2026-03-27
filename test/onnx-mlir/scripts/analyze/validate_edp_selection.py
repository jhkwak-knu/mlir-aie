#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_edp_selection.py — Verify EDP-based configuration selection accuracy.

Compares model-predicted EDP ranking against actual NPU measurements.
Reports regret, Top-K accuracy, core-optimal accuracy, and Spearman rho.

Usage:
    python3 scripts/analyze/validate_edp_selection.py \
        --result out/reports/result_edp.csv \
        --tc out/tc_list_edp.json \
        --calib data/calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ranking import spearman_rank_correlation  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class EdpCase:
    """One measured case with model prediction and actual measurement."""

    def __init__(
        self,
        case_index: int,
        size_key: str,
        num_cores: int,
        edp_rank: int,
        edp_pred: float,
        e_total_pred: float,
        t_total_pred: float,
        avg_us: float,
        min_us: float,
        npu_per_iter_uj: float,
        wall_elapsed_s: float,
        config_str: str,
    ):
        self.case_index = case_index
        self.size_key = size_key
        self.num_cores = num_cores
        self.edp_rank = edp_rank
        self.edp_pred = edp_pred
        self.e_total_pred = e_total_pred
        self.t_total_pred = t_total_pred
        self.avg_us = avg_us
        self.min_us = min_us
        self.npu_per_iter_uj = npu_per_iter_uj
        self.wall_elapsed_s = wall_elapsed_s
        self.config_str = config_str

    @property
    def has_energy(self) -> bool:
        return self.npu_per_iter_uj > 0 and self.wall_elapsed_s >= 0.005

    @property
    def measured_edp(self) -> float:
        """EDP from measurement. Uses model energy as fallback if RAPL unavailable."""
        if self.has_energy:
            return self.min_us * self.npu_per_iter_uj
        # Fallback: use model energy prediction with measured time
        return self.min_us * (self.e_total_pred / 1e6)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _col(row: Dict[str, str], *keys: str) -> str:
    for k in keys:
        if k in row:
            return row[k]
    raise KeyError(f"None of {keys} found: {list(row.keys())}")


def load_data(
    csv_path: Path,
    tc_path: Path,
) -> List[EdpCase]:
    """Load result CSV and tc_list.json, merge by case_index."""

    # Load tc_list for edp_rank/edp_pred
    with tc_path.open("r", encoding="utf-8") as f:
        tc_doc = json.load(f)
    tc_meta: Dict[int, Dict[str, Any]] = {}
    for i, case in enumerate(tc_doc.get("cases", []), start=1):
        tc_meta[i] = case

    cases: List[EdpCase] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "PASS":
                continue
            ci = int(row["case_index"])
            M = int(row["M"])
            K = int(row["K"])
            N = int(row["N"])
            nc = int(_col(row, "num_cores", "numSpm"))

            meta = tc_meta.get(ci, {})
            edp_rank = int(meta.get("edp_rank", 0))
            edp_pred = float(meta.get("edp_pred", 0))
            e_total_pred = float(meta.get("e_total_pred", 0))

            npu_uj = float(_col(row, "npu_per_iter_uj",
                                "npu_energy_per_iter_uj"))

            levels = meta.get("levels", [{}])
            lv = levels[0] if levels else {}
            config = (f"{nc}c SP({lv.get('SPm','?')},{lv.get('SPn','?')}) "
                      f"TP({lv.get('TPm','?')},{lv.get('TPk','?')},{lv.get('TPn','?')})")

            cases.append(EdpCase(
                case_index=ci,
                size_key=f"{M}x{K}x{N}",
                num_cores=nc,
                edp_rank=edp_rank,
                edp_pred=edp_pred,
                e_total_pred=e_total_pred,
                t_total_pred=float(row.get("t_total_pred", "0")),
                avg_us=float(_col(row, "avg_us")),
                min_us=float(_col(row, "min_us")),
                npu_per_iter_uj=npu_uj,
                wall_elapsed_s=float(row.get("wall_elapsed_s", "0")),
                config_str=config,
            ))
    return cases


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def compute_regret(
    cases: List[EdpCase],
    use_energy: bool = True,
) -> Tuple[float, float, int, int]:
    """Compute regret of model's Top-1 selection.

    Returns (regret_pct, best_measured, model_top1_rank_in_measured, n_cases).
    """
    if not cases:
        return 0.0, 0.0, 0, 0

    if use_energy:
        measured = [(c, c.measured_edp) for c in cases]
    else:
        measured = [(c, c.min_us) for c in cases]

    # Find actual best
    measured_sorted = sorted(measured, key=lambda x: x[1])
    best_measured_val = measured_sorted[0][1]

    # Find model's Top-1 (lowest edp_rank)
    model_top1 = min(cases, key=lambda c: c.edp_rank if c.edp_rank > 0 else 9999)
    if use_energy:
        model_top1_val = model_top1.measured_edp
    else:
        model_top1_val = model_top1.min_us

    regret = ((model_top1_val - best_measured_val) / best_measured_val * 100
              if best_measured_val > 0 else 0.0)

    # Model Top-1's rank in measured ranking
    model_rank = 1
    for c, val in measured_sorted:
        if c.case_index == model_top1.case_index:
            break
        model_rank += 1

    return regret, best_measured_val, model_rank, len(cases)


def compute_top_k_accuracy(
    cases: List[EdpCase],
    k: int,
    use_energy: bool = True,
) -> bool:
    """Check if measured Top-1 is within model's Top-K."""
    if not cases:
        return False

    if use_energy:
        measured_sorted = sorted(cases, key=lambda c: c.measured_edp)
    else:
        measured_sorted = sorted(cases, key=lambda c: c.min_us)

    actual_best = measured_sorted[0]

    # Model's Top-K by edp_rank
    model_top_k = sorted(cases, key=lambda c: c.edp_rank if c.edp_rank > 0 else 9999)[:k]
    return actual_best.case_index in {c.case_index for c in model_top_k}


def compute_core_optimal(
    cases: List[EdpCase],
    use_energy: bool = True,
) -> Tuple[int, int, bool]:
    """Compare model-optimal core count vs measured-optimal core count.

    Returns (measured_opt_cores, model_opt_cores, match).
    """
    if not cases:
        return 0, 0, False

    # Best measured per core count
    core_best: Dict[int, float] = {}
    for c in cases:
        val = c.measured_edp if use_energy else c.min_us
        if c.num_cores not in core_best or val < core_best[c.num_cores]:
            core_best[c.num_cores] = val
    measured_opt = min(core_best, key=core_best.get)

    # Model's best per core count
    model_core_best: Dict[int, float] = {}
    for c in cases:
        if c.edp_rank <= 0:
            continue
        if c.num_cores not in model_core_best or c.edp_pred < model_core_best[c.num_cores]:
            model_core_best[c.num_cores] = c.edp_pred
    if model_core_best:
        model_opt = min(model_core_best, key=model_core_best.get)
    else:
        model_opt = min(cases, key=lambda c: c.edp_rank if c.edp_rank > 0 else 9999).num_cores

    return measured_opt, model_opt, measured_opt == model_opt


def compute_rho(
    cases: List[EdpCase],
    use_energy: bool = True,
) -> float:
    """Spearman rho between predicted EDP rank and measured EDP."""
    valid = [c for c in cases if c.edp_rank > 0]
    if len(valid) < 3:
        return float("nan")

    pred_ranks = [float(c.edp_rank) for c in valid]
    if use_energy:
        measured_vals = [c.measured_edp for c in valid]
    else:
        measured_vals = [c.min_us for c in valid]

    return spearman_rank_correlation(pred_ranks, measured_vals)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(cases: List[EdpCase]) -> None:
    """Print full EDP selection verification report (time/energy/EDP)."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  EDP Selection Verification Report (GT: min_us)")
    print(sep)

    # Group by size
    by_size: Dict[str, List[EdpCase]] = defaultdict(list)
    for c in cases:
        by_size[c.size_key].append(c)

    # Check which sizes have energy data
    size_has_energy: Dict[str, bool] = {}
    for sk, group in by_size.items():
        n_with_energy = sum(1 for c in group if c.has_energy)
        size_has_energy[sk] = n_with_energy >= max(3, len(group) // 2)

    total_sizes = len(by_size)

    # --- Section 1: Performance Regret (min_us) ---
    print(f"\n  --- 1. Performance Regret (T_regret: min_us) ---")
    print(f"  {'Size':<16s} {'N':>4s}  {'Top1':>4s} {'Top3':>4s} "
          f"{'Regret':>8s} {'Core':>5s} {'rho':>6s}")

    t_top1 = 0
    t_top3 = 0
    t_regrets: List[float] = []
    t_core = 0

    for sk in sorted(by_size.keys()):
        group = by_size[sk]
        regret, _, _, n = compute_regret(group, use_energy=False)
        top1 = compute_top_k_accuracy(group, 1, use_energy=False)
        top3 = compute_top_k_accuracy(group, 3, use_energy=False)
        _, _, core_match = compute_core_optimal(group, use_energy=False)
        rho = compute_rho(group, use_energy=False)
        t_regrets.append(regret)
        if top1: t_top1 += 1
        if top3: t_top3 += 1
        if core_match: t_core += 1
        print(f"  {sk:<16s} {n:>4d}  "
              f"{'Y' if top1 else 'N':>4s} {'Y' if top3 else 'N':>4s} "
              f"{regret:>7.1f}% {'Y' if core_match else 'N':>5s} {rho:>6.3f}")

    mean_t = sum(t_regrets) / len(t_regrets) if t_regrets else 0
    max_t = max(t_regrets) if t_regrets else 0
    print(f"  {'─' * 72}")
    print(f"  {'Overall':<16s} {'':>4s}  "
          f"{t_top1:>2d}/{total_sizes}  {t_top3:>2d}/{total_sizes}  "
          f"{mean_t:>7.1f}% {t_core:>3d}/{total_sizes}")

    # --- Section 2: EDP Regret (min_us * energy, where available) ---
    sizes_with_energy = [sk for sk in sorted(by_size.keys()) if size_has_energy[sk]]
    if sizes_with_energy:
        print(f"\n  --- 2. EDP Regret (min_us * energy, {len(sizes_with_energy)} sizes with RAPL) ---")
        print(f"  {'Size':<16s} {'N':>4s}  {'E_ok':>4s}  {'Top1':>4s} {'Top3':>4s} "
              f"{'Regret':>8s} {'Core':>5s} {'rho':>6s}")

        edp_top1 = 0
        edp_top3 = 0
        edp_regrets: List[float] = []
        edp_core = 0

        for sk in sizes_with_energy:
            group = by_size[sk]
            n_energy = sum(1 for c in group if c.has_energy)
            regret, _, _, _ = compute_regret(group, use_energy=True)
            top1 = compute_top_k_accuracy(group, 1, use_energy=True)
            top3 = compute_top_k_accuracy(group, 3, use_energy=True)
            _, _, core_match = compute_core_optimal(group, use_energy=True)
            rho = compute_rho(group, use_energy=True)
            edp_regrets.append(regret)
            if top1: edp_top1 += 1
            if top3: edp_top3 += 1
            if core_match: edp_core += 1
            print(f"  {sk:<16s} {len(group):>4d}  {n_energy:>4d}  "
                  f"{'Y' if top1 else 'N':>4s} {'Y' if top3 else 'N':>4s} "
                  f"{regret:>7.1f}% {'Y' if core_match else 'N':>5s} {rho:>6.3f}")

        ne = len(sizes_with_energy)
        mean_edp = sum(edp_regrets) / ne if ne else 0
        max_edp = max(edp_regrets) if edp_regrets else 0
        print(f"  {'─' * 72}")
        print(f"  {'Overall':<16s} {'':>4s}  {'':>4s}  "
              f"{edp_top1:>2d}/{ne}  {edp_top3:>2d}/{ne}  "
              f"{mean_edp:>7.1f}% {edp_core:>3d}/{ne}")

    # --- Section 3: Model Top-1 Detail (T + EDP) ---
    print(f"\n  --- 3. Model Top-1 Detail ---")
    print(f"  {'Size':<16s} {'Config':<30s} {'T_rank':>6s} {'T_reg':>7s} "
          f"{'EDP_rank':>8s} {'EDP_reg':>8s}")
    for sk in sorted(by_size.keys()):
        group = by_size[sk]
        t_regret, _, t_rank, _ = compute_regret(group, use_energy=False)
        top1_case = min(group, key=lambda c: c.edp_rank if c.edp_rank > 0 else 9999)
        if size_has_energy.get(sk):
            edp_regret, _, edp_rank, _ = compute_regret(group, use_energy=True)
            edp_str = f"{edp_rank:>8d} {edp_regret:>7.1f}%"
        else:
            edp_str = f"{'n/a':>8s} {'n/a':>8s}"
        print(f"  {sk:<16s} {top1_case.config_str:<30s} {t_rank:>6d} {t_regret:>6.1f}% {edp_str}")

    # --- Section 4: Core-Count Analysis (T + EDP) ---
    print(f"\n  --- 4. Core-Count Analysis ---")
    print(f"  {'Size':<16s} {'T_meas':>7s} {'T_model':>7s} {'T':>3s}  "
          f"{'E_meas':>7s} {'E_model':>7s} {'E':>3s}")
    for sk in sorted(by_size.keys()):
        group = by_size[sk]
        core_counts = set(c.num_cores for c in group)
        if len(core_counts) < 2:
            print(f"  {sk:<16s}  (single core count)")
            continue
        tm, tp, t_match = compute_core_optimal(group, use_energy=False)
        if size_has_energy.get(sk):
            em, ep, e_match = compute_core_optimal(group, use_energy=True)
            print(f"  {sk:<16s} {tm:>6d}c {tp:>6d}c {'Y' if t_match else 'N':>3s}  "
                  f"{em:>6d}c {ep:>6d}c {'Y' if e_match else 'N':>3s}")
        else:
            print(f"  {sk:<16s} {tm:>6d}c {tp:>6d}c {'Y' if t_match else 'N':>3s}  "
                  f"{'n/a':>7s} {'n/a':>7s} {'':>3s}")

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify EDP-based configuration selection accuracy")
    p.add_argument("--result", required=True,
                   help="Path to result CSV (from run_tc_all.sh)")
    p.add_argument("--tc", required=True,
                   help="Path to tc_list.json (with edp_rank/edp_pred)")
    p.add_argument("--calib", default="",
                   help="Path to calibration.json (optional)")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    csv_path = Path(args.result).resolve()
    tc_path = Path(args.tc).resolve()

    cases = load_data(csv_path, tc_path)
    if not cases:
        print("[ERROR] No valid cases loaded", file=sys.stderr)
        return 1

    n_with_rank = sum(1 for c in cases if c.edp_rank > 0)
    n_with_energy = sum(1 for c in cases if c.has_energy)
    print(f"[INFO] Loaded {len(cases)} cases "
          f"({n_with_rank} with edp_rank, {n_with_energy} with energy)")

    print_report(cases)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
