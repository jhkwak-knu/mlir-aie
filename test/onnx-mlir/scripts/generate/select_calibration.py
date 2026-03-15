#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
select_calibration.py — Select structurally diverse calibration cases
for cross-size cost model coefficient fitting.

Selects cases across 5 groups per matrix size:
  A: Core count variation   — isolate L_config
  B: TP_total variation     — isolate L_sync
  C: tpOrder comparison     — quantify K-inner reversal
  D: SP shape comparison    — isolate spatial DMA pattern effects
  E: TP axis isolation      — separate per-axis overhead

Usage:
    python3 scripts/generate/select_calibration.py \\
        --op data/op_list.json --out out/calibration/tc_list.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_model import (                                  # noqa: E402
    Candidate, CostResult,
    enumerate_candidates, filter_candidates, evaluate_candidate,
    cost_result_to_tc,
)
from tiling_common import (                               # noqa: E402
    DEFAULT_OP_PATH, DEFAULT_SYS_PATH,
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
    OpCase, SystemInfo, CalibCoeffs,
    load_op_list, load_system_info, load_calibration, write_tc_list,
)

TARGET_CORES = [4, 8, 16, 32]
MAX_TP_SAMPLES = 8          # Group B: log-spaced TP_total samples
MAX_TPORDER_CONFIGS = 3     # Group C: number of configs to compare tpOrders
MAX_SP_SHAPES = 3           # Group D: SP shape variants per core count
TP_AXIS_NAMES = {0: "M", 1: "N", 2: "K"}


# ============================================================
# Helpers
# ============================================================
def _case_key(cr: CostResult) -> Tuple:
    """Unique identity for deduplication."""
    c = cr.candidate
    return (c.num_cores, c.SPm, c.SPn,
            c.TPm, c.TPk, c.TPn, cr.tp_order)


def _edp_best(
    op: OpCase, c: Candidate, coeffs: CalibCoeffs = None,
) -> CostResult:
    """Return the tpOrder with lowest EDP for a candidate."""
    results = [evaluate_candidate(op, c, tpo, coeffs) for tpo in (0, 1, 2)]
    return min(results, key=lambda r: r.edp)


def log_sample(values: List[int], max_count: int) -> List[int]:
    """Sample up to max_count values at log-spaced intervals from sorted list."""
    if len(values) <= max_count:
        return list(values)
    # Always include first and last
    n = max_count
    indices = set()
    indices.add(0)
    indices.add(len(values) - 1)
    # Fill remaining slots at log-spaced positions
    for i in range(1, n - 1):
        frac = i / (n - 1)
        log_idx = math.exp(frac * math.log(len(values) - 1))
        indices.add(min(round(log_idx), len(values) - 1))
    return [values[i] for i in sorted(indices)]


# ============================================================
# Group selection functions
# ============================================================
def select_group_a(
    valid: List[Candidate], op: OpCase,
    coeffs: CalibCoeffs = None,
) -> List[CostResult]:
    """Group A: Min TP_total per core count, EDP-best tpOrder.

    For each available core count, pick the candidate(s) with the
    smallest temporal iteration count. This maximizes the compute
    fraction and isolates L_config (per-core configuration cost).
    """
    results: List[CostResult] = []
    for nc in TARGET_CORES:
        subset = [c for c in valid if c.num_cores == nc]
        if not subset:
            continue
        min_tp = min(c.tp_total for c in subset)
        min_tp_candidates = [c for c in subset if c.tp_total == min_tp]
        # Keep distinct SP shapes (at most 2)
        seen_sp: Set[Tuple[int, int]] = set()
        for c in min_tp_candidates:
            sp = (c.SPm, c.SPn)
            if sp not in seen_sp:
                results.append(_edp_best(op, c, coeffs))
                seen_sp.add(sp)
            if len(seen_sp) >= 2:
                break
    return results


def select_group_b(
    valid: List[Candidate], op: OpCase, core_count: int = 4,
    coeffs: CalibCoeffs = None,
) -> List[CostResult]:
    """Group B: Fixed cores, varying TP_total with log-spaced sampling.

    Isolates L_sync by sweeping temporal iteration count while
    keeping core count constant.
    """
    subset = [c for c in valid if c.num_cores == core_count]
    if not subset:
        return []

    tp_totals = sorted(set(c.tp_total for c in subset))
    sampled = log_sample(tp_totals, MAX_TP_SAMPLES)

    results: List[CostResult] = []
    for tp in sampled:
        candidates = [c for c in subset if c.tp_total == tp]
        # Pick EDP-best among candidates with this TP_total
        best = min(
            (_edp_best(op, c, coeffs) for c in candidates),
            key=lambda r: r.edp,
        )
        results.append(best)
    return results


def select_group_c(
    valid: List[Candidate], op: OpCase,
    coeffs: CalibCoeffs = None,
) -> List[CostResult]:
    """Group C: tpOrder comparison — same config with all 3 tpOrders.

    Selects configs where TPm>1, TPk>1, TPn>1 so that all 3 tpOrders
    produce meaningfully different DMA patterns. Picks configs at
    different TP_total levels for variety.
    """
    # Candidates where all three temporal axes have splits
    triple = [c for c in valid if c.TPm > 1 and c.TPk > 1 and c.TPn > 1]
    if not triple:
        # Fallback: at least 2 axes have splits
        triple = [c for c in valid if sum(1 for t in (c.TPm, c.TPk, c.TPn) if t > 1) >= 2]
    if not triple:
        return []

    # Group by (num_cores, SPm, SPn, TPm, TPk, TPn) and pick representative configs
    config_map: Dict[Tuple, Candidate] = {}
    for c in triple:
        key = (c.num_cores, c.SPm, c.SPn, c.TPm, c.TPk, c.TPn)
        if key not in config_map:
            config_map[key] = c

    # Sort by TP_total and pick at different levels
    configs = sorted(config_map.values(), key=lambda c: c.tp_total)
    if len(configs) > MAX_TPORDER_CONFIGS:
        indices = log_sample(list(range(len(configs))), MAX_TPORDER_CONFIGS)
        configs = [configs[i] for i in indices]

    results: List[CostResult] = []
    for c in configs:
        for tpo in (TP_AXIS_M, TP_AXIS_N, TP_AXIS_K):
            results.append(evaluate_candidate(op, c, tpo, coeffs))
    return results


def select_group_d(
    valid: List[Candidate], op: OpCase,
    coeffs: CalibCoeffs = None,
) -> List[CostResult]:
    """Group D: SP shape comparison — different (SPm,SPn) at same core count.

    Tests whether spatial decomposition shape affects DMA overhead
    (e.g., (1,4) vs (2,2) vs (4,1) at 4 cores).
    """
    # Pick a core count with multiple SP shapes available
    results: List[CostResult] = []
    for nc in TARGET_CORES:
        subset = [c for c in valid if c.num_cores == nc]
        if not subset:
            continue
        sp_shapes = sorted(set((c.SPm, c.SPn) for c in subset))
        if len(sp_shapes) <= 1:
            continue

        # For each SP shape, pick min TP_total candidate with EDP-best tpOrder
        shapes_picked = 0
        for spm, spn in sp_shapes:
            sp_candidates = [c for c in subset if c.SPm == spm and c.SPn == spn]
            min_tp = min(c.tp_total for c in sp_candidates)
            best_c = min(
                (c for c in sp_candidates if c.tp_total == min_tp),
                key=lambda c: c.ws_bytes,
            )
            results.append(_edp_best(op, best_c, coeffs))
            shapes_picked += 1
            if shapes_picked >= MAX_SP_SHAPES:
                break
        # Only do this for one core count per matrix size
        break
    return results


def select_group_e(
    valid: List[Candidate], op: OpCase, core_count: int = 4,
    coeffs: CalibCoeffs = None,
) -> List[CostResult]:
    """Group E: TP axis isolation — only one TP axis > 1.

    Compares TPm-only, TPk-only, TPn-only at the same TP value
    to isolate per-axis overhead differences.
    """
    subset = [c for c in valid if c.num_cores == core_count]
    if not subset:
        return []

    # Classify single-axis temporal candidates
    m_only = [c for c in subset if c.TPm > 1 and c.TPk == 1 and c.TPn == 1]
    k_only = [c for c in subset if c.TPm == 1 and c.TPk > 1 and c.TPn == 1]
    n_only = [c for c in subset if c.TPm == 1 and c.TPk == 1 and c.TPn > 1]

    if not any([m_only, k_only, n_only]):
        return []

    # Find common TP values across axes (or closest)
    m_vals = set(c.TPm for c in m_only)
    k_vals = set(c.TPk for c in k_only)
    n_vals = set(c.TPn for c in n_only)

    # Pick TP values that appear in at least 2 axes
    all_vals = m_vals | k_vals | n_vals
    results: List[CostResult] = []
    used_vals: Set[int] = set()

    for tp_val in sorted(all_vals):
        if tp_val == 1:
            continue
        # Pick one from each axis that has this value
        picked = False
        for axis_list, axis_name in [(m_only, "TPm"), (k_only, "TPk"), (n_only, "TPn")]:
            matching = [c for c in axis_list
                        if getattr(c, axis_name) == tp_val]
            if matching:
                results.append(_edp_best(op, matching[0], coeffs))
                picked = True
        if picked:
            used_vals.add(tp_val)
        # Limit: 2 different TP values → up to 6 cases
        if len(used_vals) >= 2:
            break

    return results


# ============================================================
# Main selection
# ============================================================
def select_calibration(
    valid: List[Candidate], op: OpCase,
    coeffs: CalibCoeffs = None,
) -> List[CostResult]:
    """Combine all groups, deduplicate, sort by t_total ascending."""
    all_results: List[CostResult] = []
    group_counts: Dict[str, int] = {}

    groups = [
        ("A: core_variation",    select_group_a(valid, op, coeffs)),
        ("B: tp_variation",      select_group_b(valid, op, coeffs=coeffs)),
        ("C: tpOrder_compare",   select_group_c(valid, op, coeffs)),
        ("D: sp_shape",          select_group_d(valid, op, coeffs)),
        ("E: tp_axis_isolation", select_group_e(valid, op, coeffs=coeffs)),
    ]

    seen: Set[Tuple] = set()
    for name, results in groups:
        count = 0
        for cr in results:
            key = _case_key(cr)
            if key not in seen:
                seen.add(key)
                all_results.append(cr)
                count += 1
        group_counts[name] = count

    all_results.sort(key=lambda r: r.t_total)
    return all_results, group_counts


# ============================================================
# Reporting
# ============================================================
def print_selection_summary(
    op: OpCase,
    all_results: List[CostResult],
    group_counts: Dict[str, int],
) -> None:
    print(f"\n{'='*70}")
    print(f"Calibration selection: M={op.M}, K={op.K}, N={op.N}")
    print(f"{'='*70}")

    for name, count in group_counts.items():
        print(f"  {name:<25s}  {count:>3d} cases")
    print(f"  {'TOTAL (deduplicated)':<25s}  {len(all_results):>3d} cases")

    print(f"\n  {'#':>3s}  {'cores':>5s}  {'SP':>7s}  {'TP':>11s}  "
          f"{'tpOrd':>5s}  {'TP_tot':>6s}  {'Tile':>15s}  {'T_pred':>10s}")
    for i, cr in enumerate(all_results):
        c = cr.candidate
        print(f"  {i+1:>3d}  {c.num_cores:>5d}  ({c.SPm:>2d},{c.SPn:>2d})  "
              f"({c.TPm:>2d},{c.TPk:>2d},{c.TPn:>2d})  "
              f"    {TP_AXIS_NAMES[cr.tp_order]}  {c.tp_total:>6d}  "
              f"({c.TM:>3d},{c.TK:>3d},{c.TN:>3d})  "
              f"{cr.t_total:>10.1f}")


# ============================================================
# CLI
# ============================================================
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select structurally diverse calibration cases")
    p.add_argument("--op", default=str(DEFAULT_OP_PATH),
                   help="Path to op_list.json")
    p.add_argument("--sys", default=str(DEFAULT_SYS_PATH),
                   help="Path to xdna2_info.json")
    p.add_argument("--out", default="",
                   help="Path to write calibration tc_list.json")
    p.add_argument("--op-index", type=int, default=-1,
                   help="Run only this 0-based op index (-1 = all)")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    op_path = Path(args.op).resolve()
    sys_path = Path(args.sys).resolve()

    try:
        ops = load_op_list(op_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    try:
        sys_info = load_system_info(sys_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    coeffs = load_calibration()

    print(f"[INFO] {len(ops)} ops from {op_path}")
    if coeffs.calibrated:
        print(f"[INFO] Calibration loaded: eff_macs={coeffs.eff_macs}")
    else:
        print(f"[INFO] Calibration: not loaded (using defaults)")

    if args.op_index >= 0:
        if args.op_index >= len(ops):
            print(f"[ERROR] op-index {args.op_index} out of range",
                  file=sys.stderr)
            return 1
        targets = [(args.op_index, ops[args.op_index])]
    else:
        targets = list(enumerate(ops))

    tc_cases = []
    total_selected = 0

    for idx, op in targets:
        all_candidates = enumerate_candidates(op, sys_info)
        valid, _ = filter_candidates(all_candidates, op, sys_info)

        if not valid:
            print(f"[WARN] No valid candidates for M={op.M}, K={op.K}, N={op.N}")
            continue

        results, group_counts = select_calibration(valid, op, coeffs)
        print_selection_summary(op, results, group_counts)
        total_selected += len(results)

        for cr in results:
            tc_cases.append(cost_result_to_tc(op, cr))

    print(f"\n{'='*70}")
    print(f"Total calibration cases: {total_selected}")
    print(f"{'='*70}")

    if args.out and tc_cases:
        out_path = Path(args.out).resolve()
        write_tc_list(tc_cases, out_path)
        print(f"[INFO] Wrote {out_path} ({len(tc_cases)} cases)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
