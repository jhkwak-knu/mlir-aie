#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
empirical_characterization.py — Core scaling & energy analysis.

Combines all measurement datasets (v9, R1, R2) to produce:
  1. Core Scaling Curves: time, energy, EDP per core count per matrix size
  2. Energy Savings vs All-Cores baseline
  3. Config Sensitivity within optimal core count
  4. Model Accuracy Summary

Usage:
    python3 scripts/analyze/empirical_characterization.py \
        --v9 out/reports/result_v9_energy.csv \
        --v9-tc out/calibration/tc_list_v3.json \
        --r1 out/reports/result_edp_r1.csv \
        --r1-tc out/tc_list_edp_r1.json \
        --r2 out/reports/result_edp_r2.csv \
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
    TP_AXIS_M, TP_AXIS_N, TP_AXIS_K,
)
from cost_model import (  # noqa: E402
    total_data_bytes, Candidate, OpCase, evaluate_candidate,
)

CLOCK_MHZ = 1500
COMP_TILES_PER_COL = 4
MIN_WALL_S_FOR_ENERGY = 0.005


# ============================================================
# Data structures
# ============================================================
@dataclass
class EmpCase:
    """Single measurement case."""
    source: str
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

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.K}x{self.N}"

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn

    @property
    def num_columns(self) -> int:
        return math.ceil(self.num_cores / COMP_TILES_PER_COL)

    @property
    def has_energy(self) -> bool:
        return self.npu_per_iter_uj > 0 and self.wall_elapsed_s >= MIN_WALL_S_FOR_ENERGY

    @property
    def edp_meas(self) -> float:
        """Measured EDP (us * uJ)."""
        if self.has_energy:
            return self.min_us * self.npu_per_iter_uj
        return float("inf")

    @property
    def config_key(self) -> str:
        return (f"SP({self.SPm},{self.SPn}) "
                f"TP({self.TPm},{self.TPk},{self.TPn}) "
                f"tpO={['M','N','K'][self.tp_order_inner]}")

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
def load_cases(csv_path: Path, tc_path: Path, source: str) -> List[EmpCase]:
    """Load measurement cases from CSV + tc_list pair."""
    if not csv_path.is_file():
        return []

    tc_meta = {}
    if tc_path.is_file():
        with tc_path.open() as f:
            tc_doc = json.load(f)
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

            cases.append(EmpCase(
                source=source,
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
            ))
    return cases


# ============================================================
# Analysis 1: Core Scaling
# ============================================================
def analyze_core_scaling(cases: List[EmpCase]) -> Dict[str, Dict[int, dict]]:
    """For each (size, cores), find the best measured time and energy.

    Returns: {size_key: {num_cores: {min_us, energy_uj, edp, n_configs, ...}}}
    """
    by_size_cores: Dict[Tuple[str, int], List[EmpCase]] = defaultdict(list)
    for c in cases:
        by_size_cores[(c.size_key, c.num_cores)].append(c)

    scaling = defaultdict(dict)
    for (sk, nc), group in sorted(by_size_cores.items()):
        best_time = min(group, key=lambda c: c.min_us)

        # Best energy (from cases with valid energy data)
        energy_cases = [c for c in group if c.has_energy]
        best_energy_uj = min((c.npu_per_iter_uj for c in energy_cases),
                             default=float("nan"))

        # Best EDP
        edp_cases = [c for c in group if c.has_energy]
        best_edp = min((c.edp_meas for c in edp_cases), default=float("nan"))
        best_edp_case = min(edp_cases, key=lambda c: c.edp_meas) if edp_cases else None

        scaling[sk][nc] = {
            "min_us": best_time.min_us,
            "best_time_config": best_time.config_key,
            "energy_uj": best_energy_uj,
            "edp": best_edp,
            "edp_config": best_edp_case.config_key if best_edp_case else "n/a",
            "n_configs": len(group),
            "n_energy": len(energy_cases),
        }

    return dict(scaling)


def print_core_scaling(scaling: Dict[str, Dict[int, dict]]) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  1. Core Scaling Curves")
    print(sep)

    for sk in sorted(scaling):
        cores_data = scaling[sk]
        print(f"\n  --- {sk} ---")
        print(f"  {'Cores':>5s}  {'Cols':>4s}  {'min_us':>10s}  {'E(uJ)':>10s}  "
              f"{'EDP':>14s}  {'speedup':>8s}  {'E_ratio':>8s}  {'N_cfg':>5s}  Config(EDP)")
        print(f"  {'-' * 85}")

        # Reference: max core count for speedup/ratio calculation
        sorted_cores = sorted(cores_data.keys())
        max_nc = max(sorted_cores)
        ref_time = cores_data[max_nc]["min_us"]
        ref_energy = cores_data[max_nc]["energy_uj"]

        edp_optimal_nc = None
        edp_optimal_val = float("inf")

        for nc in sorted_cores:
            d = cores_data[nc]
            cols = nc // COMP_TILES_PER_COL

            # Speedup vs max-cores
            speedup = ref_time / d["min_us"] if d["min_us"] > 0 else 0

            # Energy ratio vs max-cores
            e_ratio = ""
            if not math.isnan(d["energy_uj"]) and not math.isnan(ref_energy) and ref_energy > 0:
                e_ratio = f"{d['energy_uj']/ref_energy:.2f}x"

            # Track EDP optimal
            if not math.isnan(d["edp"]) and d["edp"] < edp_optimal_val:
                edp_optimal_val = d["edp"]
                edp_optimal_nc = nc

            edp_str = f"{d['edp']:.1f}" if not math.isnan(d["edp"]) else "n/a"
            e_str = f"{d['energy_uj']:.1f}" if not math.isnan(d["energy_uj"]) else "n/a"

            print(f"  {nc:>5d}  {cols:>4d}  {d['min_us']:>10.1f}  {e_str:>10s}  "
                  f"{edp_str:>14s}  {speedup:>7.2f}x  {e_ratio:>8s}  "
                  f"{d['n_configs']:>5d}  {d['edp_config']}")

        if edp_optimal_nc is not None:
            opt_d = cores_data[edp_optimal_nc]
            max_d = cores_data[max_nc]
            time_overhead = (opt_d["min_us"] / max_d["min_us"] - 1) * 100
            energy_save = ""
            if not math.isnan(opt_d["energy_uj"]) and not math.isnan(max_d["energy_uj"]):
                energy_save = f"{(1 - opt_d['energy_uj']/max_d['energy_uj'])*100:.1f}%"
            print(f"  -> EDP-optimal: {edp_optimal_nc}c "
                  f"({edp_optimal_nc//COMP_TILES_PER_COL} cols), "
                  f"time: {time_overhead:+.1f}% vs {max_nc}c, "
                  f"energy: -{energy_save} vs {max_nc}c")


# ============================================================
# Analysis 2: Energy Savings vs All-Cores
# ============================================================
def print_energy_savings(scaling: Dict[str, Dict[int, dict]]) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  2. Energy Savings: EDP-Optimal vs All-Cores (max cores)")
    print(sep)

    print(f"\n  {'Size':>10s}  {'MaxC':>5s}  {'OptC':>5s}  "
          f"{'T_opt(us)':>10s}  {'T_max(us)':>10s}  {'T_delta':>8s}  "
          f"{'E_opt(uJ)':>10s}  {'E_max(uJ)':>10s}  {'E_save':>8s}  "
          f"{'EDP_save':>8s}")
    print(f"  {'-' * 100}")

    for sk in sorted(scaling):
        cores_data = scaling[sk]
        sorted_cores = sorted(cores_data.keys())
        max_nc = max(sorted_cores)
        max_d = cores_data[max_nc]

        # Find EDP-optimal
        edp_opt_nc = max_nc
        edp_opt_val = float("inf")
        for nc in sorted_cores:
            d = cores_data[nc]
            if not math.isnan(d["edp"]) and d["edp"] < edp_opt_val:
                edp_opt_val = d["edp"]
                edp_opt_nc = nc

        opt_d = cores_data[edp_opt_nc]

        t_delta = (opt_d["min_us"] / max_d["min_us"] - 1) * 100 if max_d["min_us"] > 0 else 0

        e_save = ""
        edp_save = ""
        e_opt_str = "n/a"
        e_max_str = "n/a"
        if not math.isnan(opt_d["energy_uj"]):
            e_opt_str = f"{opt_d['energy_uj']:.1f}"
        if not math.isnan(max_d["energy_uj"]):
            e_max_str = f"{max_d['energy_uj']:.1f}"
        if not math.isnan(opt_d["energy_uj"]) and not math.isnan(max_d["energy_uj"]):
            if max_d["energy_uj"] > 0:
                e_save = f"{(1 - opt_d['energy_uj']/max_d['energy_uj'])*100:.1f}%"
            if max_d["edp"] > 0:
                edp_save = f"{(1 - opt_d['edp']/max_d['edp'])*100:.1f}%"

        print(f"  {sk:>10s}  {max_nc:>5d}  {edp_opt_nc:>5d}  "
              f"{opt_d['min_us']:>10.1f}  {max_d['min_us']:>10.1f}  {t_delta:>+7.1f}%  "
              f"{e_opt_str:>10s}  {e_max_str:>10s}  {e_save:>8s}  "
              f"{edp_save:>8s}")


# ============================================================
# Analysis 3: Config Sensitivity
# ============================================================
def print_config_sensitivity(cases: List[EmpCase], scaling: Dict[str, Dict[int, dict]]) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  3. Config Sensitivity (within EDP-optimal core count)")
    print(sep)

    # Find EDP-optimal core count per size
    edp_opt_cores = {}
    for sk in scaling:
        cores_data = scaling[sk]
        best_nc = min(cores_data, key=lambda nc: cores_data[nc]["edp"]
                      if not math.isnan(cores_data[nc]["edp"]) else float("inf"))
        edp_opt_cores[sk] = best_nc

    # Group cases by (size, optimal cores)
    by_group = defaultdict(list)
    for c in cases:
        if c.num_cores == edp_opt_cores.get(c.size_key, -1):
            by_group[c.size_key].append(c)

    print(f"\n  {'Size':>10s}  {'OptC':>5s}  {'N_cfg':>5s}  {'N_E':>4s}  "
          f"{'Best_us':>10s}  {'Worst_us':>10s}  {'T_gap':>7s}  "
          f"{'Best_EDP':>12s}  {'Worst_EDP':>12s}  {'EDP_gap':>8s}")
    print(f"  {'-' * 95}")

    for sk in sorted(by_group):
        group = by_group[sk]
        nc = edp_opt_cores[sk]

        best_time = min(c.min_us for c in group)
        worst_time = max(c.min_us for c in group)
        t_gap = (worst_time / best_time - 1) * 100 if best_time > 0 else 0

        e_group = [c for c in group if c.has_energy]
        n_e = len(e_group)
        if e_group:
            best_edp = min(c.edp_meas for c in e_group)
            worst_edp = max(c.edp_meas for c in e_group)
            edp_gap = f"{(worst_edp/best_edp - 1)*100:.0f}%"
            best_edp_s = f"{best_edp:.1f}"
            worst_edp_s = f"{worst_edp:.1f}"
        else:
            edp_gap = "n/a"
            best_edp_s = "n/a"
            worst_edp_s = "n/a"

        print(f"  {sk:>10s}  {nc:>5d}  {len(group):>5d}  {n_e:>4d}  "
              f"{best_time:>10.1f}  {worst_time:>10.1f}  {t_gap:>6.0f}%  "
              f"{best_edp_s:>12s}  {worst_edp_s:>12s}  {edp_gap:>8s}")


# ============================================================
# Analysis 4: Model Accuracy
# ============================================================
def print_model_accuracy(
    cases: List[EmpCase], scaling: Dict[str, Dict[int, dict]],
    coeffs: CalibCoeffs,
) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  4. Model Accuracy (cost model vs measured)")
    print(sep)

    if not coeffs.calibrated:
        print("  [WARN] No calibration, skipping")
        return

    # For each size, check if model selects the correct EDP-optimal core count
    # Find measured EDP-optimal core count
    meas_opt_cores = {}
    for sk in scaling:
        cores_data = scaling[sk]
        best_nc = min(cores_data, key=lambda nc: cores_data[nc]["edp"]
                      if not math.isnan(cores_data[nc]["edp"]) else float("inf"))
        meas_opt_cores[sk] = best_nc

    # For each size, get model's prediction
    print(f"\n  --- Core Selection Accuracy ---")
    print(f"  {'Size':>10s}  {'Meas_opt':>8s}  {'Model_opt':>9s}  {'Match':>5s}")
    print(f"  {'-' * 40}")

    correct = 0
    total = 0

    for c in cases:
        # Just need one case per size to get model prediction
        pass

    # Group by size and find model's optimal
    by_size = defaultdict(list)
    for c in cases:
        by_size[c.size_key].append(c)

    for sk in sorted(by_size):
        group = by_size[sk]
        if sk not in meas_opt_cores:
            continue

        # Model prediction: evaluate each case and find lowest EDP
        model_results = []
        for c in group:
            op = c.make_op()
            cand = c.make_cand()
            cr = evaluate_candidate(op, cand, c.tp_order_inner, coeffs)
            model_results.append((c, cr))

        model_best = min(model_results, key=lambda x: x[1].edp)
        model_opt_nc = model_best[0].num_cores
        meas_opt_nc = meas_opt_cores[sk]

        match = "Y" if model_opt_nc == meas_opt_nc else "N"
        if model_opt_nc == meas_opt_nc:
            correct += 1
        total += 1

        print(f"  {sk:>10s}  {meas_opt_nc:>7d}c  {model_opt_nc:>8d}c  {match:>5s}")

    if total > 0:
        print(f"\n  Core selection accuracy: {correct}/{total} "
              f"({correct/total*100:.0f}%)")

    # Top-K analysis: for each size, does model's Top-K contain measured optimal?
    print(f"\n  --- Top-K EDP Selection ---")
    print(f"  {'Size':>10s}  {'Meas_best_us':>13s}  {'Model#1_us':>11s}  "
          f"{'Regret':>8s}  {'Top3_hit':>8s}  {'Top5_hit':>8s}")
    print(f"  {'-' * 65}")

    for sk in sorted(by_size):
        group = by_size[sk]

        model_results = []
        for c in group:
            op = c.make_op()
            cand = c.make_cand()
            cr = evaluate_candidate(op, cand, c.tp_order_inner, coeffs)
            model_results.append((c, cr))

        # Sort by model EDP
        model_results.sort(key=lambda x: x[1].edp)

        # Measured best (min_us)
        meas_best_time = min(c.min_us for c in group)

        # Model Top-1 measured time
        model_top1_time = model_results[0][0].min_us

        regret = (model_top1_time / meas_best_time - 1) * 100 if meas_best_time > 0 else 0

        # Top-K: does actual best appear in model's Top-K?
        top3_times = [r[0].min_us for r in model_results[:3]]
        top5_times = [r[0].min_us for r in model_results[:5]]
        top3_hit = "Y" if meas_best_time >= min(top3_times) * 0.999 else "N"
        top5_hit = "Y" if meas_best_time >= min(top5_times) * 0.999 else "N"

        # Check if actual best case is in top-K by checking if any top-K
        # case has time within 0.1% of measured best
        top3_hit = "Y" if any(t <= meas_best_time * 1.001 for t in top3_times) else "N"
        top5_hit = "Y" if any(t <= meas_best_time * 1.001 for t in top5_times) else "N"

        print(f"  {sk:>10s}  {meas_best_time:>12.1f}  {model_top1_time:>10.1f}  "
              f"{regret:>+7.1f}%  {top3_hit:>8s}  {top5_hit:>8s}")


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Empirical characterization analysis")
    p.add_argument("--v9", default="", help="result_v9_energy.csv")
    p.add_argument("--v9-tc", default="", help="v9 tc_list")
    p.add_argument("--r1", default="", help="result_edp_r1.csv")
    p.add_argument("--r1-tc", default="", help="r1 tc_list")
    p.add_argument("--r2", default="", help="result_edp_r2.csv")
    p.add_argument("--r2-tc", default="", help="r2 tc_list")
    p.add_argument("--emp", default="", help="result_empirical.csv (additional)")
    p.add_argument("--emp-tc", default="", help="empirical tc_list")
    p.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    args = p.parse_args()

    # Load all datasets
    all_cases: List[EmpCase] = []
    datasets = [
        (args.v9, args.v9_tc, "v9"),
        (args.r1, args.r1_tc, "r1"),
        (args.r2, args.r2_tc, "r2"),
        (args.emp, args.emp_tc, "emp"),
    ]
    for csv_arg, tc_arg, src in datasets:
        if csv_arg:
            loaded = load_cases(Path(csv_arg), Path(tc_arg) if tc_arg else Path(""), src)
            all_cases.extend(loaded)
            print(f"  Loaded {src}: {len(loaded)} cases")

    print(f"  Total: {len(all_cases)} cases "
          f"({sum(1 for c in all_cases if c.has_energy)} with valid energy)")

    coeffs = load_calibration(Path(args.calib))

    # Deduplicate: if same (M, cores, SP, TP, tpOrder) appears multiple times,
    # keep the one with lowest min_us
    dedup_key = lambda c: (c.M, c.num_cores, c.SPm, c.SPn,
                           c.TPm, c.TPk, c.TPn, c.tp_order_inner)
    dedup: Dict[tuple, EmpCase] = {}
    for c in all_cases:
        key = dedup_key(c)
        if key not in dedup or c.min_us < dedup[key].min_us:
            dedup[key] = c
    cases = list(dedup.values())
    print(f"  After dedup: {len(cases)} unique configs")

    # Run analyses
    scaling = analyze_core_scaling(cases)
    print_core_scaling(scaling)
    print_energy_savings(scaling)
    print_config_sensitivity(cases, scaling)
    print_model_accuracy(cases, scaling, coeffs)

    print()


if __name__ == "__main__":
    main()
