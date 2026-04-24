#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cost_model.py — Exhaustive search + constraint filtering for spatio-temporal
parallelization on XDNA2 AIE arrays.

Stage 1: Enumerate all (numCores, SPm, SPn, TPm, TPk, TPn) candidates.
Stage 2: Apply hardware/software constraints to filter valid candidates.
Stage 3: Cost model evaluation (performance + energy).
Stage 4: EDP-based optimal selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xdna_search.cost.v16_edp import V16EdpCost, evaluate_candidate  # noqa: E402
from xdna_search.hw_constants import (  # noqa: E402
    DEFAULT_CALIB_PATH, DEFAULT_OP_PATH, DEFAULT_SYS_PATH,
)
from xdna_search.io import (  # noqa: E402
    build_metadata, load_calibration, load_op_list, load_system_info, write_tc_list,
)
from xdna_search.math_utils import build_tp_order  # noqa: E402
from xdna_search.search.constraints.filter_set import (  # noqa: E402
    DefaultFeasibility, FilterSet,
)
from xdna_search.search.enumerator import ExhaustiveEnumerator  # noqa: E402
from xdna_search.search.factory import get_searcher_factory  # noqa: E402
from xdna_search.search.selector import EdpSelector  # noqa: E402
from xdna_search.types import (  # noqa: E402
    CalibCoeffs, Candidate, CostResult, DEFAULT_COEFFS,
    FilterResult, OpCase, SystemInfo,
)

# Hardware coefficients (PEAK_MACS, BANDWIDTH_BPC, ALPHA_CYCLES, E_MAC_PJ,
# E_DRAM_PJ, P_STATIC_PJ) live in xdna_search.cost.v16_edp. Import from there
# if you need them outside this CLI.


# ============================================================
# Stage 1: Exhaustive enumeration
# ============================================================
_EXHAUSTIVE_ENUMERATOR = ExhaustiveEnumerator()


def enumerate_candidates(
    op: OpCase, sys_info: SystemInfo,
    exclude_cores: Optional[set] = None,
) -> List[Candidate]:
    """Thin wrapper around xdna_search.search.enumerator.ExhaustiveEnumerator.

    Kept for backward compatibility; see the class for the enumeration
    loop and complexity discussion.
    """
    return _EXHAUSTIVE_ENUMERATOR.generate(op, sys_info, exclude_cores=exclude_cores)


# ============================================================
# Stage 2: Constraint filters
# ============================================================
_DEFAULT_FILTER_SET = FilterSet(DefaultFeasibility())


def filter_candidates(
    candidates: List[Candidate],
    op: OpCase,
    sys_info: SystemInfo,
) -> Tuple[List[Candidate], List[FilterResult]]:
    """Thin wrapper around FilterSet(DefaultFeasibility()).apply().

    Kept for backward compatibility; constraint classes (C1Memory, ..., C5MmulShape)
    live in xdna_search.search.constraints.feasibility.
    """
    return _DEFAULT_FILTER_SET.apply(candidates, op, sys_info)


# ============================================================
# Stage 3: Cost model evaluation (moved to xdna_search.cost.v16_edp)
# ============================================================
# The module-level functions total_data_bytes / perf_* / energy_* /
# evaluate_candidate now live in xdna_search.cost.v16_edp. `evaluate_candidate`
# is re-imported above for select_optimal() below.


# ============================================================
# Stage 4: EDP-based optimal selection
# ============================================================
_EDP_SELECTOR = EdpSelector()


def select_optimal(
    valid: List[Candidate], op: OpCase,
    coeffs: Optional[CalibCoeffs] = None,
    keep_all_tporders: bool = False,
) -> List[CostResult]:
    """For each candidate, evaluate all 3 tpOrders and rank via EdpSelector.

    If keep_all_tporders=True, also returns a dict mapping candidate index
    to all 3 CostResults (used by select_tporder_verify_cases()).
    """
    scored: List[CostResult] = []
    all_tporder_results: Dict[int, List[CostResult]] = {}
    for i, c in enumerate(valid):
        results = [evaluate_candidate(op, c, tpo, coeffs) for tpo in (0, 1, 2)]
        scored.extend(results)
        if keep_all_tporders:
            all_tporder_results[i] = results

    ranked = _EDP_SELECTOR.rank(scored)
    if keep_all_tporders:
        return ranked, all_tporder_results
    return ranked


def select_tporder_verify_cases(
    all_ranked: Dict[int, List[CostResult]],
    all_tporder_data: Dict[int, Dict[int, List[CostResult]]],
    ops: list,
    n_verify: int = 19,
) -> List[Dict[str, Any]]:
    """Select tpOrder verification cases: 1 per workload, pick the candidate
    with the largest EDP ratio across 3 tpOrders. Return extra tc entries
    for the non-best tpOrders (2 per selected candidate).
    """
    verify_tcs: List[Dict[str, Any]] = []
    n_verify = min(n_verify, len(all_ranked))

    # For each workload, find the candidate with max tpOrder EDP spread
    workload_picks = []
    for op_idx, ranked in all_ranked.items():
        tpo_data = all_tporder_data.get(op_idx, {})
        if not tpo_data:
            continue
        # Build a map from candidate identity to tporder results
        # ranked contains best-per-candidate; we need to find which
        # valid-index each ranked entry maps to
        best_ratio = 0.0
        best_cand_results = None
        best_cr = None
        for valid_idx, results in tpo_data.items():
            edps = [r.edp for r in results]
            if min(edps) <= 0:
                continue
            ratio = max(edps) / min(edps)
            if ratio > best_ratio:
                best_ratio = ratio
                best_cand_results = results
                best_cr = min(results, key=lambda r: r.edp)

        if best_cand_results and best_cr:
            workload_picks.append((op_idx, best_ratio, best_cr, best_cand_results))

    # Sort by ratio descending, take top n_verify
    workload_picks.sort(key=lambda x: x[1], reverse=True)
    selected = workload_picks[:n_verify]

    for op_idx, ratio, best_cr, all_results in selected:
        op = ops[op_idx]
        best_tpo = best_cr.tp_order
        for r in all_results:
            if r.tp_order != best_tpo:
                tc = cost_result_to_tc(op, r)
                tc["tporder_verify"] = True
                verify_tcs.append(tc)
        print(f"  tpOrder verify: M{op.M}_K{op.K}_N{op.N} "
              f"SP=({best_cr.candidate.SPm},{best_cr.candidate.SPn}) "
              f"ratio={ratio:.2f}")

    return verify_tcs


# ============================================================
# Reporting
# ============================================================
def print_search_summary(
    op: OpCase,
    total: int,
    valid: List[Candidate],
    filter_results: List[FilterResult],
) -> None:
    print(f"\n{'='*60}")
    print(f"Op: M={op.M}, K={op.K}, N={op.N}, type={op.elem_type}")
    print(f"{'='*60}")
    print(f"Total enumerated:  {total}")

    for fr in filter_results:
        pct = fr.removed / fr.before * 100 if fr.before > 0 else 0
        print(f"  {fr.name:<30s}  {fr.before:>6d} -> {fr.after:>6d}  "
              f"(removed {fr.removed:>5d}, {pct:5.1f}%)")

    print(f"Valid candidates:  {len(valid)}")

    if valid:
        # Summary by num_cores
        core_counts: Dict[int, int] = {}
        for c in valid:
            core_counts[c.num_cores] = core_counts.get(c.num_cores, 0) + 1
        print(f"\nCandidates by core count:")
        for nc in sorted(core_counts):
            print(f"  {nc:>3d} cores: {core_counts[nc]:>5d} candidates")

        # Show a few examples
        print(f"\nFirst 5 candidates:")
        for c in valid[:5]:
            print(f"  cores={c.num_cores} SP=({c.SPm},{c.SPn}) "
                  f"TP=({c.TPm},{c.TPk},{c.TPn}) "
                  f"Tile=({c.TM},{c.TK},{c.TN}) ws={c.ws_bytes}B")


TP_AXIS_NAMES = {0: "M", 1: "N", 2: "K"}


def print_cost_summary(op: OpCase, ranked: List[CostResult]) -> None:
    """Print Stage 3+4 results: best candidates by core count and overall."""
    if not ranked:
        print("\n  No valid candidates to evaluate.")
        return

    print(f"\n--- Stage 3+4: Cost Model & EDP Ranking ---")

    # Best per core count
    best_by_cores: Dict[int, CostResult] = {}
    for r in ranked:
        nc = r.candidate.num_cores
        if nc not in best_by_cores:
            best_by_cores[nc] = r

    print(f"\n  Best EDP per core count:")
    print(f"  {'cores':>5s}  {'SP':>7s}  {'TP':>11s}  {'tpOrd':>5s}  "
          f"{'T_total':>12s}  {'E_total':>12s}  {'EDP':>14s}")
    for nc in sorted(best_by_cores):
        r = best_by_cores[nc]
        c = r.candidate
        print(f"  {nc:>5d}  ({c.SPm:>2d},{c.SPn:>2d})  "
              f"({c.TPm:>2d},{c.TPk:>2d},{c.TPn:>2d})  "
              f"    {TP_AXIS_NAMES[r.tp_order]}  "
              f"{r.t_total:>12.1f}  {r.e_total:>12.1f}  {r.edp:>14.1f}")

    # Overall top 5
    print(f"\n  Top 5 by EDP:")
    for i, r in enumerate(ranked[:5]):
        c = r.candidate
        print(f"  #{i+1}  cores={c.num_cores} SP=({c.SPm},{c.SPn}) "
              f"TP=({c.TPm},{c.TPk},{c.TPn}) "
              f"tpOrder={TP_AXIS_NAMES[r.tp_order]}  "
              f"T={r.t_total:.1f} E={r.e_total:.1f} EDP={r.edp:.1f}")


# ============================================================
# tc_list.json output (flat schema for the build/run pipeline)
# ============================================================
def cost_result_to_tc(
    op: OpCase, cr: CostResult, edp_rank: int = 0,
) -> Dict[str, Any]:
    """Convert a CostResult to a flat tc.json entry for the pipeline."""
    c = cr.candidate
    tp_order_full = build_tp_order(cr.tp_order)
    entry: Dict[str, Any] = {
        "M": op.M, "K": op.K, "N": op.N,
        "elemType": op.elem_type,
        "numCores": c.num_cores,
        "doubleBuffer": False,
        "t_total_pred": round(cr.t_total, 2),
        "levels": [
            {
                "SPm": c.SPm, "SPn": c.SPn,
                "TPm": c.TPm, "TPk": c.TPk, "TPn": c.TPn,
                "TM": c.TM, "TK": c.TK, "TN": c.TN,
                "tpOrder": tp_order_full,
            }
        ],
    }
    if edp_rank > 0:
        entry["edp_rank"] = edp_rank
        entry["edp_pred"] = round(cr.edp, 2)
        entry["e_total_pred"] = round(cr.e_total, 2)
    return entry


# ============================================================
# CLI
# ============================================================
def select_validation_candidates(
    ranked: List[CostResult],
    top_n: int = 0,
    per_core_top: int = 0,
    random_sample: int = 0,
) -> List[CostResult]:
    """Select validation candidates from EDP-ranked list.

    If all sampling params are 0, returns all candidates (sorted by T_total).
    Otherwise, builds a deduplicated set from:
      1. EDP Top-N overall
      2. EDP Top-K per core count
      3. Random M from remaining (seed=42 for reproducibility)
    Returns selected candidates sorted by EDP ascending.
    """
    if top_n <= 0 and per_core_top <= 0 and random_sample <= 0:
        return sorted(ranked, key=lambda r: r.t_total)

    selected_indices: set = set()

    # 1. EDP Top-N overall (ranked is already sorted by EDP ascending)
    if top_n > 0:
        for i in range(min(top_n, len(ranked))):
            selected_indices.add(i)

    # 2. Per-core-count Top-K
    if per_core_top > 0:
        core_counts: Dict[int, List[int]] = {}
        for i, r in enumerate(ranked):
            nc = r.candidate.num_cores
            if nc not in core_counts:
                core_counts[nc] = []
            core_counts[nc].append(i)
        for nc in sorted(core_counts):
            for i in core_counts[nc][:per_core_top]:
                selected_indices.add(i)

    # 3. Random sample from remaining
    if random_sample > 0:
        remaining = [i for i in range(len(ranked)) if i not in selected_indices]
        rng = random.Random(42)
        k = min(random_sample, len(remaining))
        for i in rng.sample(remaining, k):
            selected_indices.add(i)

    result = [ranked[i] for i in sorted(selected_indices)]
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exhaustive search + constraint filtering for parallelization configs")
    p.add_argument("--op", default=str(DEFAULT_OP_PATH),
                   help="Path to op_list.json")
    p.add_argument("--sys", default=str(DEFAULT_SYS_PATH),
                   help="Path to xdna2_info.json")
    p.add_argument("--out", default="",
                   help="Path to write cost results JSON (optional)")
    p.add_argument("--validate", default="",
                   help="Path to write validation tc_list.json (optional)")
    p.add_argument("--op-index", type=int, default=-1,
                   help="Run only this 0-based op index (-1 = all)")
    p.add_argument("--calib", default=str(DEFAULT_CALIB_PATH),
                   help="Path to calibration.json (empty string to skip)")
    p.add_argument("--top-n", type=int, default=0,
                   help="Validation: select EDP Top-N per size (0=all)")
    p.add_argument("--per-core-top", type=int, default=0,
                   help="Validation: select Top-K per core count per size")
    p.add_argument("--random-sample", type=int, default=0,
                   help="Validation: add N random candidates from remaining")
    p.add_argument("--tporder-verify", type=int, default=0,
                   help="Add tpOrder verification cases: N workloads x 2 extra tpOrders")
    p.add_argument("--exclude-cores", default="",
                   help="Comma-separated numCores values to exclude from enumeration "
                        "(e.g., '24' to skip all 24-core configs)")
    p.add_argument("--search", default="sm-exh",
                   choices=("sm-exh", "star-map", "naive", "timeloop"),
                   help="Searcher to use. Default 'sm-exh' preserves legacy "
                        "exhaustive behavior; 'star-map' applies STAR-Map pruning.")
    p.add_argument("--pruning-level", type=int, default=123,
                   choices=(1, 12, 123),
                   help="STAR-Map pruning level (1, 12, or 123). Only effective "
                        "when --search=star-map.")
    return p.parse_args(argv)


def process_op(
    op: OpCase, sys_info: SystemInfo,
    coeffs: Optional[CalibCoeffs] = None,
    keep_all_tporders: bool = False,
    exclude_cores: Optional[set] = None,
    searcher_name: str = "sm-exh",
    pruning_level: int = 123,
):
    """Run the full 4-component pipeline for a single op via the factory."""
    factory = get_searcher_factory(searcher_name)
    if searcher_name == "star-map":
        searcher = factory(
            sys_info, coeffs,
            pruning_level=pruning_level,
            exclude_cores=exclude_cores,
        )
    else:
        searcher = factory(sys_info, coeffs, exclude_cores=exclude_cores)
    output = searcher.search(op, keep_all_tporders=keep_all_tporders)

    print_search_summary(
        op, output.total_enumerated, output.valid, output.filter_results,
    )
    print_cost_summary(op, output.ranked)

    if keep_all_tporders:
        return output.ranked, output.all_tporder_data
    return output.ranked


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

    # Load calibration coefficients
    if args.calib:
        coeffs = load_calibration(Path(args.calib).resolve())
    else:
        coeffs = DEFAULT_COEFFS

    print(f"[INFO] {len(ops)} ops from {op_path}")
    print(f"[INFO] HW: {sys_info.total_cores} cores, "
          f"{sys_info.comp_tiles_per_col} tiles/col, "
          f"{sys_info.max_columns} cols, "
          f"{sys_info.spm_size_bytes}B/tile "
          f"(usable {sys_info.ct_usable_bytes}B)")
    if coeffs.calibrated:
        print(f"[INFO] Calibration: eff_macs={coeffs.eff_macs}, "
              f"l_sync={coeffs.l_sync_cy:.0f}, "
              f"l_pe={coeffs.l_pe_cy:.0f}, "
              f"l_startup={coeffs.l_startup_cy:.0f}")
    else:
        print(f"[INFO] Calibration: not loaded (using defaults)")

    # Select ops to process
    if args.op_index >= 0:
        if args.op_index >= len(ops):
            print(f"[ERROR] op-index {args.op_index} out of range", file=sys.stderr)
            return 1
        targets = [(args.op_index, ops[args.op_index])]
    else:
        targets = list(enumerate(ops))

    need_tporder = args.tporder_verify > 0
    if need_tporder and args.search not in ("sm-exh",):
        print(
            f"[WARN] --tporder-verify is only meaningful for --search=sm-exh; "
            f"ignoring under --search={args.search}",
            file=sys.stderr,
        )
        need_tporder = False
    searcher_label = (
        f"star-map-rule{args.pruning_level}"
        if args.search == "star-map" else args.search
    )
    print(f"[INFO] Searcher: {searcher_label}")

    exclude_cores_set = set()
    if args.exclude_cores:
        exclude_cores_set = {int(x) for x in args.exclude_cores.split(",") if x.strip()}
        print(f"[info] excluding numCores: {sorted(exclude_cores_set)}")
    all_ranked: Dict[int, List[CostResult]] = {}
    all_tporder_data: Dict[int, Dict[int, List[CostResult]]] = {}
    for idx, op in targets:
        if need_tporder:
            ranked, tpo_data = process_op(
                op, sys_info, coeffs,
                keep_all_tporders=True,
                exclude_cores=exclude_cores_set,
                searcher_name=args.search,
                pruning_level=args.pruning_level,
            )
            all_tporder_data[idx] = tpo_data
        else:
            ranked = process_op(
                op, sys_info, coeffs,
                exclude_cores=exclude_cores_set,
                searcher_name=args.search,
                pruning_level=args.pruning_level,
            )
        all_ranked[idx] = ranked

    # Write output if requested
    if args.out:
        out_path = Path(args.out).resolve()
        output = {}
        for idx, ranked in all_ranked.items():
            op = ops[idx]
            key = f"M{op.M}_K{op.K}_N{op.N}"
            output[key] = {
                "op": {"M": op.M, "K": op.K, "N": op.N, "elemType": op.elem_type},
                "num_ranked": len(ranked),
                "ranked": [
                    {
                        **asdict(r.candidate),
                        "tp_order": r.tp_order,
                        "t_comp": r.t_comp,
                        "t_comm": r.t_comm,
                        "t_overhead": r.t_overhead,
                        "t_total": r.t_total,
                        "e_dynamic_comp": r.e_dynamic_comp,
                        "e_dynamic_comm": r.e_dynamic_comm,
                        "e_static": r.e_static,
                        "e_total": r.e_total,
                        "edp": r.edp,
                    }
                    for r in ranked
                ],
            }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\n[INFO] Wrote {out_path}")

    # Write validation tc_list.json if requested
    if args.validate:
        val_path = Path(args.validate).resolve()
        tc_cases: List[Dict[str, Any]] = []
        for idx, ranked in all_ranked.items():
            op = ops[idx]
            selected = select_validation_candidates(
                ranked,
                top_n=args.top_n,
                per_core_top=args.per_core_top,
                random_sample=args.random_sample,
            )
            for cr in selected:
                edp_rank = ranked.index(cr) + 1
                tc_cases.append(cost_result_to_tc(op, cr, edp_rank=edp_rank))
            sampling = ""
            if args.top_n > 0 or args.per_core_top > 0 or args.random_sample > 0:
                sampling = (f" (top-{args.top_n} + core-top-{args.per_core_top}"
                            f" + rand-{args.random_sample}"
                            f" from {len(ranked)} total)")
            print(f"\n[INFO] Op M{op.M}_K{op.K}_N{op.N}: "
                  f"{len(selected)} validation candidates selected{sampling}")
        # Add tpOrder verification cases
        if args.tporder_verify > 0 and all_tporder_data:
            print(f"\n[INFO] Selecting tpOrder verification cases...")
            verify_tcs = select_tporder_verify_cases(
                all_ranked, all_tporder_data, ops,
                n_verify=args.tporder_verify,
            )
            tc_cases.extend(verify_tcs)
            print(f"[INFO] Added {len(verify_tcs)} tpOrder verification cases")

        meta = build_metadata(
            calib_path=Path(args.calib), coeffs=coeffs,
            searcher_name=searcher_label,
        )
        write_tc_list(tc_cases, val_path, metadata=meta)
        print(f"[INFO] Wrote {val_path} ({len(tc_cases)} cases total)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
