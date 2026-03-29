#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_empirical_cases.py — Generate a systematic measurement dataset
for empirical characterization of core scaling and energy behavior.

Generates one representative config per valid (size, cores) combination:
- Config: model's EDP Top-1 for that core count
- Iteration count: auto-calculated so wall_time >= 10ms (RAPL reliability)
- Single tc_list for overnight measurement

Usage:
    python3 scripts/analyze/prepare_empirical_cases.py \
        --calib data/calibration.json \
        --output out/tc_list_empirical.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate"))
from tiling_common import (  # noqa: E402
    DEFAULT_OP_PATH, DEFAULT_SYS_PATH, DEFAULT_CALIB_PATH,
    OpCase, SystemInfo, CalibCoeffs,
    load_op_list, load_system_info, load_calibration,
    build_tp_order,
)
from cost_model import (  # noqa: E402
    enumerate_candidates, filter_candidates, select_optimal,
    cost_result_to_tc, CostResult,
)

CLOCK_MHZ = 1500
# Minimum wall time for reliable RAPL measurement (seconds).
# 50ms provides sufficient margin for RAPL accuracy, especially after
# kernel optimizations (chess pragmas) that reduce per-iteration time.
MIN_WALL_S = 0.050
# Default warmup iterations
DEFAULT_WARMUP = 3


def estimate_iter_time_us(cr: CostResult) -> float:
    """Estimate per-iteration time in microseconds from cost model."""
    return cr.t_total / CLOCK_MHZ


def calc_iterations(est_iter_us: float, warmup: int = DEFAULT_WARMUP) -> int:
    """Calculate minimum iterations to reach MIN_WALL_S.

    Total wall time ≈ (warmup + n_iter) * per_iter_time.
    We want: (warmup + n_iter) * est_iter_us * 1e-6 >= MIN_WALL_S
    """
    min_total_iters = math.ceil(MIN_WALL_S / (est_iter_us * 1e-6))
    n_iter = max(min_total_iters - warmup, 10)  # at least 10 measured iters
    return n_iter


def main():
    p = argparse.ArgumentParser(
        description="Generate systematic empirical characterization tc_list")
    p.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    p.add_argument("--op", default=str(DEFAULT_OP_PATH))
    p.add_argument("--sys", default=str(DEFAULT_SYS_PATH))
    p.add_argument("--output", required=True, help="Output tc_list.json path")
    args = p.parse_args()

    sep = "=" * 80

    ops = load_op_list(Path(args.op))
    sys_info = load_system_info(Path(args.sys))
    coeffs = load_calibration(Path(args.calib))

    print(f"\n{sep}")
    print("  Empirical Characterization: Case Generation")
    print(f"  Min wall time for RAPL: {MIN_WALL_S*1000:.0f}ms")
    print(sep)

    tc_cases: List[Dict[str, Any]] = []
    case_idx = 0

    # Header
    print(f"\n  {'#':>3s}  {'Size':>10s}  {'Cores':>5s}  {'Cols':>4s}  "
          f"{'SP':>7s}  {'TP':>11s}  {'tpO':>3s}  "
          f"{'est_us':>8s}  {'n_iter':>6s}  {'est_wall':>9s}")
    print(f"  {'-' * 80}")

    for op in ops:
        # Enumerate and filter all candidates for this op
        all_cands = enumerate_candidates(op, sys_info)
        valid, _ = filter_candidates(all_cands, op, sys_info)

        if not valid:
            print(f"  {op.M}x{op.N}: no valid configs")
            continue

        # Rank by EDP
        ranked = select_optimal(valid, op, coeffs)

        # Find best EDP per core count
        best_per_core: Dict[int, CostResult] = {}
        for r in ranked:
            nc = r.candidate.num_cores
            if nc not in best_per_core:
                best_per_core[nc] = r

        for nc in sorted(best_per_core):
            cr = best_per_core[nc]
            c = cr.candidate
            cols = nc // 4

            # Calculate iterations for RAPL
            est_us = estimate_iter_time_us(cr)
            n_iter = calc_iterations(est_us)
            est_wall_ms = (n_iter + DEFAULT_WARMUP) * est_us / 1000

            case_idx += 1
            entry = cost_result_to_tc(op, cr, edp_rank=0)
            entry["n_iterations"] = n_iter
            entry["n_warmup"] = DEFAULT_WARMUP
            entry["empirical_category"] = "scaling"
            tc_cases.append(entry)

            tpo_name = ["M", "N", "K"][cr.tp_order]
            print(f"  {case_idx:>3d}  {op.M:>4d}x{op.N:<4d}  {nc:>5d}  {cols:>4d}  "
                  f"({c.SPm:>2d},{c.SPn:>2d})  "
                  f"({c.TPm:>2d},{c.TPk:>2d},{c.TPn:>2d})  {tpo_name:>3s}  "
                  f"{est_us:>7.0f}  {n_iter:>6d}  {est_wall_ms:>7.0f}ms")

    # Write tc_list
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tc_doc = {"cases": tc_cases}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(tc_doc, f, indent=2)

    # Summary
    print(f"\n  Total cases: {len(tc_cases)}")
    print(f"  Written to: {out_path}")

    # Per-size summary
    print(f"\n  Per-size summary:")
    by_size = defaultdict(list)
    for tc in tc_cases:
        by_size[f"{tc['M']}x{tc['N']}"].append(tc)

    total_est_s = 0
    for sk in sorted(by_size):
        cases = by_size[sk]
        cores = [tc["numCores"] for tc in cases]
        # Estimate total time for this size
        size_est_s = sum(
            (tc["n_iterations"] + tc["n_warmup"])
            * tc["t_total_pred"] / CLOCK_MHZ / 1e6
            for tc in cases
        )
        # Add ~30s per case for build overhead
        size_est_s += len(cases) * 30
        total_est_s += size_est_s
        print(f"    {sk}: {len(cases)} cases, cores={cores}, "
              f"~{size_est_s/60:.0f}min")

    print(f"\n  Estimated total time: ~{total_est_s/60:.0f}min "
          f"({total_est_s/3600:.1f}h)")
    print(f"  (includes ~30s build overhead per case)")
    print()


if __name__ == "__main__":
    main()
