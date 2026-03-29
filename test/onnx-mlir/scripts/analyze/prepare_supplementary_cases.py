#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_supplementary_cases.py — Generate supplementary measurement cases
for deeper empirical characterization.

Supplements the core 22 cases with:
  A. All valid SP shapes per (size, cores) — config sensitivity
  B. Intermediate core counts (12, 20, 24, 28) — finer scaling curve
  C. Deduplication against existing data (v9, R1, R2, empirical)

Usage:
    python3 scripts/analyze/prepare_supplementary_cases.py \
        --calib data/calibration.json \
        --existing out/reports/result_empirical.csv \
        --output out/tc_list_supplementary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
MIN_WALL_S = 0.050
DEFAULT_WARMUP = 3


def estimate_iter_time_us(cr: CostResult) -> float:
    return cr.t_total / CLOCK_MHZ


def calc_iterations(est_iter_us: float) -> int:
    min_total = math.ceil(MIN_WALL_S / (est_iter_us * 1e-6))
    return max(min_total - DEFAULT_WARMUP, 10)


def config_key(M: int, nc: int, SPm: int, SPn: int,
               TPm: int, TPk: int, TPn: int, tpOrder_inner: int) -> tuple:
    return (M, nc, SPm, SPn, TPm, TPk, TPn, tpOrder_inner)


def load_existing_configs(csv_paths: List[str]) -> Set[tuple]:
    """Load existing measured configs as a set of keys for dedup."""
    existing = set()
    for csv_path_str in csv_paths:
        csv_path = Path(csv_path_str)
        if not csv_path.is_file():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row.get("status") != "PASS":
                    continue
                M = int(row["M"])
                nc = int(row.get("numSpm", row.get("num_cores", "0")))
                SPm = int(row["SPm"])
                SPn = int(row["SPn"])
                TPm = int(row["TPm"])
                TPk = int(row["TPk"])
                TPn = int(row["TPn"])
                # tpOrder not in CSV, skip dedup on it
                existing.add((M, nc, SPm, SPn, TPm, TPk, TPn))
    return existing


def main():
    p = argparse.ArgumentParser(
        description="Generate supplementary measurement cases")
    p.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    p.add_argument("--op", default=str(DEFAULT_OP_PATH))
    p.add_argument("--sys", default=str(DEFAULT_SYS_PATH))
    p.add_argument("--existing", nargs="*", default=[],
                   help="Existing result CSVs for dedup")
    p.add_argument("--output", required=True)
    # Control what to generate
    p.add_argument("--all-sp", action="store_true",
                   help="Generate all valid SP shapes per (size, cores)")
    p.add_argument("--intermediate-cores", action="store_true",
                   help="Add intermediate core counts (12, 20, 24, 28)")
    p.add_argument("--top-k", type=int, default=0,
                   help="Top-K EDP configs per core count (0=all SP shapes)")
    p.add_argument("--exclude-cores", type=int, nargs="*", default=[],
                   help="Core counts to exclude (e.g., 24 for RUN_FAIL)")
    args = p.parse_args()
    exclude_cores = set(args.exclude_cores)

    sep = "=" * 80

    ops = load_op_list(Path(args.op))
    sys_info = load_system_info(Path(args.sys))
    coeffs = load_calibration(Path(args.calib))

    # Load existing for dedup
    existing = load_existing_configs(args.existing)
    print(f"\n{sep}")
    print("  Supplementary Cases Generation")
    print(sep)
    print(f"  Existing measured configs: {len(existing)}")

    tc_cases: List[Dict[str, Any]] = []
    case_idx = 0
    skipped_dedup = 0

    print(f"\n  {'#':>3s}  {'Cat':>5s}  {'Size':>10s}  {'Cores':>5s}  "
          f"{'SP':>7s}  {'TP':>11s}  {'tpO':>3s}  "
          f"{'est_us':>8s}  {'n_iter':>6s}")
    print(f"  {'-' * 70}")

    for op in ops:
        all_cands = enumerate_candidates(op, sys_info)
        valid, _ = filter_candidates(all_cands, op, sys_info)
        if not valid:
            continue

        ranked = select_optimal(valid, op, coeffs)

        # Group by core count
        by_cores: Dict[int, List[CostResult]] = defaultdict(list)
        for r in ranked:
            by_cores[r.candidate.num_cores].append(r)

        # Determine which core counts to include
        target_cores = set(by_cores.keys()) - exclude_cores
        if not args.intermediate_cores:
            # Only standard core counts (already in empirical set)
            target_cores = {nc for nc in target_cores if nc in (4, 8, 16, 32)}

        for nc in sorted(target_cores):
            core_results = by_cores[nc]
            is_intermediate = nc not in (4, 8, 16, 32)
            category = "mid" if is_intermediate else "sp"

            if is_intermediate:
                # For intermediate cores: only Top-1
                selection = core_results[:1]
            elif args.top_k > 0:
                selection = core_results[:args.top_k]
            else:
                # All SP shapes: one per unique (SPm, SPn)
                seen_sp = set()
                selection = []
                for r in core_results:
                    sp = (r.candidate.SPm, r.candidate.SPn)
                    if sp not in seen_sp:
                        seen_sp.add(sp)
                        selection.append(r)

            for cr in selection:
                c = cr.candidate
                # Dedup check
                dedup_key = (op.M, nc, c.SPm, c.SPn, c.TPm, c.TPk, c.TPn)
                if dedup_key in existing:
                    skipped_dedup += 1
                    continue

                est_us = estimate_iter_time_us(cr)
                n_iter = calc_iterations(est_us)

                case_idx += 1
                entry = cost_result_to_tc(op, cr, edp_rank=0)
                entry["n_iterations"] = n_iter
                entry["n_warmup"] = DEFAULT_WARMUP
                entry["empirical_category"] = category
                tc_cases.append(entry)

                # Mark as existing to avoid self-duplication
                existing.add(dedup_key)

                tpo_name = ["M", "N", "K"][cr.tp_order]
                print(f"  {case_idx:>3d}  {category:>5s}  "
                      f"{op.M:>4d}x{op.N:<4d}  {nc:>5d}  "
                      f"({c.SPm:>2d},{c.SPn:>2d})  "
                      f"({c.TPm:>2d},{c.TPk:>2d},{c.TPn:>2d})  {tpo_name:>3s}  "
                      f"{est_us:>7.0f}  {n_iter:>6d}")

    # Write tc_list
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tc_doc = {"cases": tc_cases}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(tc_doc, f, indent=2)

    # Summary
    print(f"\n  New cases: {len(tc_cases)} (skipped {skipped_dedup} existing)")
    print(f"  Written to: {out_path}")

    by_cat = defaultdict(int)
    for tc in tc_cases:
        by_cat[tc.get("empirical_category", "?")] += 1
    for cat, n in sorted(by_cat.items()):
        print(f"    {cat}: {n} cases")

    # Time estimate
    total_est_s = 0
    for tc in tc_cases:
        iter_us = tc["t_total_pred"] / CLOCK_MHZ
        total_est_s += (tc["n_iterations"] + tc["n_warmup"]) * iter_us / 1e6
        total_est_s += 30  # build overhead
    print(f"\n  Estimated time: ~{total_est_s/60:.0f}min ({total_est_s/3600:.1f}h)")
    print()


if __name__ == "__main__":
    main()
