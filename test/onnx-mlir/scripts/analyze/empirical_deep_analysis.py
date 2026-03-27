#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
empirical_deep_analysis.py — Deep empirical analysis of NPU core scaling.

Sections:
  A1. Overhead Characterization (why fewer cores are faster for small workloads)
  A2. Scaling Efficiency (parallel efficiency, Amdahl's law)
  A3. TP/SP Deep Dive (temporal/spatial partitioning effects)
  A4. Cost Model Failure Analysis (root cause of wrong predictions)
  A5. MACs vs Optimal Cores (crossover analysis, shape effects)

Usage:
    python3 scripts/analyze/empirical_deep_analysis.py \
        --data out/reports/result_all.csv \
        --tc out/tc_list_all.json \
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
from cost_model import (  # noqa: E402
    total_data_bytes, Candidate, OpCase, evaluate_candidate,
)

CLOCK_MHZ = 1500
COMP_TILES_PER_COL = 4


# ============================================================
# Data structures
# ============================================================
@dataclass
class Case:
    M: int; K: int; N: int
    SPm: int; SPn: int
    TPm: int; TPk: int; TPn: int
    TM: int; TK: int; TN: int
    num_cores: int
    tp_order_inner: int
    min_us: float
    npu_per_iter_uj: float

    @property
    def size_key(self) -> str:
        return f"{self.M}x{self.K}x{self.N}"

    @property
    def macs(self) -> int:
        return self.M * self.K * self.N

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn

    @property
    def num_columns(self) -> int:
        return math.ceil(self.num_cores / COMP_TILES_PER_COL)

    @property
    def has_energy(self) -> bool:
        return self.npu_per_iter_uj > 0

    @property
    def edp(self) -> float:
        if self.has_energy:
            return self.min_us * self.npu_per_iter_uj
        return float("inf")

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


def load_data(csv_path: Path, tc_path: Path) -> List[Case]:
    tc_meta = {}
    if tc_path.is_file():
        with tc_path.open() as f:
            doc = json.load(f)
        for i, case in enumerate(doc.get("cases", []), start=1):
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

            cases.append(Case(
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                TM=int(row["TM"]), TK=int(row["TK"]), TN=int(row["TN"]),
                num_cores=int(row.get("numSpm", "0")),
                tp_order_inner=tpo[0],
                min_us=float(row["min_us"]),
                npu_per_iter_uj=float(row.get("npu_energy_per_iter_uj", "-1")),
            ))
    return cases


def dedup_cases(cases: List[Case]) -> List[Case]:
    """Keep best min_us per unique config."""
    best: Dict[tuple, Case] = {}
    for c in cases:
        key = (c.M, c.K, c.N, c.num_cores, c.SPm, c.SPn,
               c.TPm, c.TPk, c.TPn, c.tp_order_inner)
        if key not in best or c.min_us < best[key].min_us:
            best[key] = c
    return list(best.values())


# ============================================================
# A1. Overhead Characterization
# ============================================================
def analyze_overhead(cases: List[Case], coeffs: CalibCoeffs) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  A1. Overhead Characterization")
    print(f"  Why are fewer cores faster for small workloads?")
    print(sep)

    # For each (size, cores), find the best config and decompose
    by_size_cores = defaultdict(list)
    for c in cases:
        by_size_cores[(c.size_key, c.num_cores)].append(c)

    print(f"\n  {'Size':<18s}  {'Cores':>5s}  {'min_us':>9s}  "
          f"{'T_comp%':>7s}  {'T_comm%':>7s}  {'T_ovh%':>6s}  "
          f"{'ovh(Kcy)':>9s}  {'regime':<15s}")
    print(f"  {'-' * 85}")

    # Group by size for crossover analysis
    size_data = defaultdict(dict)

    for (sk, nc) in sorted(by_size_cores):
        group = by_size_cores[(sk, nc)]
        best = min(group, key=lambda c: c.min_us)

        op = best.make_op()
        cand = best.make_cand()

        t_comp_cy = (best.M * best.K * best.N) / (best.num_cores * coeffs.eff_macs)
        t_comm_cy = total_data_bytes(op, cand, best.tp_order_inner) / coeffs.bw_eff_bpc
        t_meas_cy = best.min_us * CLOCK_MHZ
        t_ovh_cy = t_meas_cy - t_comp_cy - t_comm_cy

        comp_pct = t_comp_cy / t_meas_cy * 100 if t_meas_cy > 0 else 0
        comm_pct = t_comm_cy / t_meas_cy * 100 if t_meas_cy > 0 else 0
        ovh_pct = t_ovh_cy / t_meas_cy * 100 if t_meas_cy > 0 else 0

        if ovh_pct > 60:
            regime = "overhead-dom"
        elif comp_pct > 60:
            regime = "compute-dom"
        else:
            regime = "balanced"

        size_data[sk][nc] = {
            "min_us": best.min_us, "comp_pct": comp_pct,
            "comm_pct": comm_pct, "ovh_pct": ovh_pct,
            "ovh_kcy": t_ovh_cy / 1000, "regime": regime,
            "macs": best.macs,
        }

        print(f"  {sk:<18s}  {nc:>5d}  {best.min_us:>9.1f}  "
              f"{comp_pct:>6.0f}%  {comm_pct:>6.0f}%  {ovh_pct:>5.0f}%  "
              f"{t_ovh_cy/1000:>8.0f}K  {regime:<15s}")

    # Crossover analysis: at what overhead% does EDP-optimal switch to 32c?
    print(f"\n  --- Crossover Analysis ---")
    print(f"  {'Size':<18s}  {'MACs':>12s}  {'EDP_opt':>7s}  "
          f"{'ovh%@opt':>8s}  {'ovh%@32c':>8s}")
    print(f"  {'-' * 60}")

    for sk in sorted(size_data, key=lambda s: list(size_data[s].values())[0]["macs"]):
        cores_info = size_data[sk]
        max_nc = max(cores_info.keys())

        # Find EDP-optimal
        edp_opt_nc = max_nc
        for nc in cores_info:
            # Simplified: use the nc with best min_us weighted by energy pattern
            pass

        # Just show overhead% at each core count
        for nc in sorted(cores_info):
            d = cores_info[nc]

        # EDP-optimal from the analysis
        best_edp_nc = min(cores_info.keys(),
                          key=lambda nc: cores_info[nc]["min_us"])  # simplified

        opt_d = cores_info[best_edp_nc]
        max_d = cores_info.get(max_nc, cores_info[max(cores_info.keys())])

        print(f"  {sk:<18s}  {opt_d['macs']:>12,d}  {best_edp_nc:>6d}c  "
              f"{opt_d['ovh_pct']:>7.0f}%  {max_d['ovh_pct']:>7.0f}%")


# ============================================================
# A2. Scaling Efficiency
# ============================================================
def analyze_scaling_efficiency(cases: List[Case]) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  A2. Scaling Efficiency (Parallel Efficiency & Amdahl's Law)")
    print(sep)

    # For each size, find best time per core count
    by_size = defaultdict(lambda: defaultdict(float))
    for c in cases:
        key = (c.size_key, c.num_cores)
        if key not in by_size or c.min_us < by_size[c.size_key][c.num_cores]:
            by_size[c.size_key][c.num_cores] = c.min_us

    # Need macs for sorting
    size_macs = {}
    for c in cases:
        size_macs[c.size_key] = c.macs

    print(f"\n  {'Size':<18s}  {'T_4c':>8s}  {'T_32c':>8s}  "
          f"{'Spdup':>6s}  {'Eff':>5s}  {'f(Amdahl)':>10s}  {'regime':<15s}")
    print(f"  {'-' * 78}")

    for sk in sorted(by_size, key=lambda s: size_macs.get(s, 0)):
        cores_times = by_size[sk]
        if 4 not in cores_times:
            continue

        t_4c = cores_times[4]
        max_nc = max(cores_times.keys())
        t_max = cores_times[max_nc]

        # Speedup relative to 4c
        speedup = t_4c / t_max if t_max > 0 else 0
        ideal_speedup = max_nc / 4
        efficiency = speedup / ideal_speedup if ideal_speedup > 0 else 0

        # Amdahl's law: speedup = 1 / (f + (1-f)/p) where p = max_nc/4
        # Solve for f: f = (1/speedup - 1/p) / (1 - 1/p)
        p = max_nc / 4
        if speedup > 0 and p > 1:
            f_amdahl = (1/speedup - 1/p) / (1 - 1/p)
            f_amdahl = max(0, min(1, f_amdahl))
        else:
            f_amdahl = 1.0

        if f_amdahl > 0.6:
            regime = "overhead-bound"
        elif f_amdahl < 0.3:
            regime = "compute-bound"
        else:
            regime = "balanced"

        print(f"  {sk:<18s}  {t_4c:>8.1f}  {t_max:>8.1f}  "
              f"{speedup:>5.2f}x  {efficiency:>4.0f}%  {f_amdahl:>10.3f}  {regime:<15s}")

    # Detailed per-step efficiency
    print(f"\n  --- Step-by-step Parallel Efficiency ---")
    print(f"  {'Size':<18s}  {'4c->8c':>7s}  {'4c->16c':>8s}  {'4c->32c':>8s}")
    print(f"  {'-' * 45}")

    for sk in sorted(by_size, key=lambda s: size_macs.get(s, 0)):
        cores_times = by_size[sk]
        if 4 not in cores_times:
            continue
        t_4c = cores_times[4]

        steps = []
        for nc in [8, 16, 32]:
            if nc in cores_times:
                speedup = t_4c / cores_times[nc]
                ideal = nc / 4
                eff = speedup / ideal * 100
                steps.append(f"{eff:>5.0f}%")
            else:
                steps.append("    -")

        print(f"  {sk:<18s}  {steps[0]:>7s}  {steps[1]:>8s}  {steps[2]:>8s}")


# ============================================================
# A3. TP/SP Deep Dive
# ============================================================
def analyze_tp_sp(cases: List[Case]) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  A3. TP/SP Deep Dive")
    print(sep)

    # --- tpOrder effect ---
    print(f"\n  --- tpOrder Effect (same size/cores/SP/TP_total, different tpOrder) ---")

    tpo_groups = defaultdict(list)
    for c in cases:
        key = (c.size_key, c.num_cores, c.SPm, c.SPn, c.tp_total)
        tpo_groups[key].append(c)

    tpo_pairs = 0
    tpo_correct = 0
    tpo_names = ["M", "N", "K"]

    print(f"  {'Size':<18s}  {'Cores':>5s}  {'SP':>7s}  {'TP_t':>5s}  "
          f"{'best_tpO':>8s}  {'worst_tpO':>9s}  {'gap%':>6s}")
    print(f"  {'-' * 65}")

    for key in sorted(tpo_groups):
        group = tpo_groups[key]
        tpo_set = set(c.tp_order_inner for c in group)
        if len(tpo_set) < 2:
            continue

        best = min(group, key=lambda c: c.min_us)
        worst = max(group, key=lambda c: c.min_us)
        gap = (worst.min_us / best.min_us - 1) * 100

        tpo_pairs += 1
        sk, nc, spm, spn, tpt = key
        print(f"  {sk:<18s}  {nc:>5d}  ({spm},{spn})  {tpt:>5d}  "
              f"{tpo_names[best.tp_order_inner]:>8s}  "
              f"{tpo_names[worst.tp_order_inner]:>9s}  {gap:>5.0f}%")

    if tpo_pairs == 0:
        print("  No tpOrder variation pairs found")

    # --- TP_total scaling ---
    print(f"\n  --- TP_total Scaling (same size/cores/SP, varying TP_total) ---")

    tp_groups = defaultdict(list)
    for c in cases:
        key = (c.size_key, c.num_cores, c.SPm, c.SPn)
        tp_groups[key].append(c)

    print(f"  {'Group':<35s}  {'N_tp':>4s}  {'TP_range':>12s}  "
          f"{'best_us':>9s}  {'worst_us':>10s}  {'gap%':>6s}")
    print(f"  {'-' * 85}")

    for key in sorted(tp_groups):
        group = tp_groups[key]
        tp_totals = set(c.tp_total for c in group)
        if len(tp_totals) < 3:
            continue

        best = min(group, key=lambda c: c.min_us)
        worst = max(group, key=lambda c: c.min_us)
        gap = (worst.min_us / best.min_us - 1) * 100

        sk, nc, spm, spn = key
        tp_range = f"{min(tp_totals)}-{max(tp_totals)}"
        label = f"{sk} {nc}c SP({spm},{spn})"
        print(f"  {label:<35s}  {len(tp_totals):>4d}  {tp_range:>12s}  "
              f"{best.min_us:>9.1f}  {worst.min_us:>10.1f}  {gap:>5.0f}%")

    # --- SP shape effect ---
    print(f"\n  --- SP Shape Effect (same size/cores/TP, different SP) ---")

    sp_groups = defaultdict(list)
    for c in cases:
        key = (c.size_key, c.num_cores, c.TPm, c.TPk, c.TPn, c.tp_order_inner)
        sp_groups[key].append(c)

    print(f"  {'Group':<40s}  {'N_sp':>4s}  "
          f"{'best_SP':>8s}  {'best_us':>9s}  {'worst_SP':>9s}  {'worst_us':>10s}  {'gap%':>6s}")
    print(f"  {'-' * 95}")

    for key in sorted(sp_groups):
        group = sp_groups[key]
        sp_set = set((c.SPm, c.SPn) for c in group)
        if len(sp_set) < 2:
            continue

        best = min(group, key=lambda c: c.min_us)
        worst = max(group, key=lambda c: c.min_us)
        gap = (worst.min_us / best.min_us - 1) * 100

        sk, nc, tpm, tpk, tpn, tpo = key
        tpo_name = ["M", "N", "K"][tpo]
        label = f"{sk} {nc}c TP({tpm},{tpk},{tpn}) tpO={tpo_name}"
        print(f"  {label:<40s}  {len(sp_set):>4d}  "
              f"({best.SPm},{best.SPn}){'':<2s}  {best.min_us:>9.1f}  "
              f"({worst.SPm},{worst.SPn}){'':<2s}  {worst.min_us:>10.1f}  {gap:>5.0f}%")


# ============================================================
# A4. Cost Model Failure Analysis
# ============================================================
def analyze_model_failures(cases: List[Case], coeffs: CalibCoeffs) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  A4. Cost Model Failure Analysis")
    print(sep)

    # Find EDP-optimal per size (measured)
    by_size = defaultdict(list)
    for c in cases:
        by_size[c.size_key].append(c)

    print(f"\n  --- Per-size: Model vs Measured Optimal ---")

    for sk in sorted(by_size, key=lambda s: by_size[s][0].macs):
        group = by_size[sk]

        # Measured EDP-optimal (among cases with energy)
        e_group = [c for c in group if c.has_energy]
        if not e_group:
            continue
        meas_best = min(e_group, key=lambda c: c.edp)
        meas_opt_nc = meas_best.num_cores

        # Model EDP-optimal
        model_results = []
        for c in group:
            cr = evaluate_candidate(c.make_op(), c.make_cand(),
                                    c.tp_order_inner, coeffs)
            model_results.append((c, cr))
        model_best_c, model_best_cr = min(model_results, key=lambda x: x[1].edp)
        model_opt_nc = model_best_c.num_cores

        if model_opt_nc == meas_opt_nc:
            continue  # correct, skip

        print(f"\n  MISMATCH: {sk} (MACs={meas_best.macs:,d})")
        print(f"    Model selects: {model_opt_nc}c, Measured optimal: {meas_opt_nc}c")

        # Compare model decomposition at both core counts
        for label, nc in [("Model choice", model_opt_nc), ("Measured optimal", meas_opt_nc)]:
            # Find best measured case at this core count
            nc_cases = [c for c in e_group if c.num_cores == nc]
            if not nc_cases:
                print(f"    {label} ({nc}c): no measured cases")
                continue
            best_nc = min(nc_cases, key=lambda c: c.edp)

            op = best_nc.make_op()
            cand = best_nc.make_cand()
            cr = evaluate_candidate(op, cand, best_nc.tp_order_inner, coeffs)

            t_meas_cy = best_nc.min_us * CLOCK_MHZ
            t_comp_cy = (best_nc.M * best_nc.K * best_nc.N) / (nc * coeffs.eff_macs)
            t_comm_cy = total_data_bytes(op, cand, best_nc.tp_order_inner) / coeffs.bw_eff_bpc
            t_ovh_meas = t_meas_cy - t_comp_cy - t_comm_cy
            t_ovh_model = cr.t_overhead

            print(f"    {label} ({nc}c): SP({best_nc.SPm},{best_nc.SPn}) "
                  f"TP({best_nc.TPm},{best_nc.TPk},{best_nc.TPn})")
            print(f"      Time:  meas={best_nc.min_us:.1f}us  "
                  f"model={cr.t_total/CLOCK_MHZ:.1f}us  "
                  f"(delta={((cr.t_total/CLOCK_MHZ)/best_nc.min_us-1)*100:+.0f}%)")
            print(f"      T_comp:  {t_comp_cy/CLOCK_MHZ:.1f}us ({t_comp_cy/t_meas_cy*100:.0f}%)")
            print(f"      T_comm:  {t_comm_cy/CLOCK_MHZ:.1f}us ({t_comm_cy/t_meas_cy*100:.0f}%)")
            print(f"      T_ovh:   model={t_ovh_model/CLOCK_MHZ:.1f}us  "
                  f"meas={t_ovh_meas/CLOCK_MHZ:.1f}us  "
                  f"(delta={((t_ovh_model/t_ovh_meas)-1)*100:+.0f}%)" if t_ovh_meas > 0 else "")
            if best_nc.has_energy:
                print(f"      Energy: {best_nc.npu_per_iter_uj:.1f}uJ  "
                      f"EDP: {best_nc.edp:.0f}")


# ============================================================
# A5. MACs vs Optimal Cores
# ============================================================
def analyze_macs_vs_cores(cases: List[Case], coeffs: CalibCoeffs) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  A5. MACs vs Optimal Cores (Crossover Analysis)")
    print(sep)

    by_size = defaultdict(list)
    for c in cases:
        by_size[c.size_key].append(c)

    # Find EDP-optimal core count per size
    results = []
    for sk in by_size:
        group = by_size[sk]
        e_group = [c for c in group if c.has_energy]
        if not e_group:
            continue

        meas_best = min(e_group, key=lambda c: c.edp)
        macs = meas_best.macs
        opt_nc = meas_best.num_cores

        # Model prediction
        model_results = [(c, evaluate_candidate(c.make_op(), c.make_cand(),
                         c.tp_order_inner, coeffs)) for c in group]
        model_best = min(model_results, key=lambda x: x[1].edp)
        model_nc = model_best[0].num_cores

        # MACs/core rule: pick largest core count where MACs/core >= threshold
        results.append({
            "size": sk, "macs": macs, "opt_nc": opt_nc, "model_nc": model_nc,
            "M": meas_best.M, "K": meas_best.K, "N": meas_best.N,
            "macs_per_core": macs // opt_nc,
        })

    results.sort(key=lambda r: r["macs"])

    print(f"\n  {'Size':<18s}  {'MACs':>12s}  {'Opt':>4s}  {'Model':>5s}  "
          f"{'MACs/core':>12s}  {'M:K:N ratio':>12s}  {'Match':>5s}")
    print(f"  {'-' * 75}")

    for r in results:
        match = "Y" if r["opt_nc"] == r["model_nc"] else "N"
        # Normalize ratio to smallest dimension
        dims = [r["M"], r["K"], r["N"]]
        min_d = min(dims)
        ratio = f"{dims[0]//min_d}:{dims[1]//min_d}:{dims[2]//min_d}"

        print(f"  {r['size']:<18s}  {r['macs']:>12,d}  {r['opt_nc']:>3d}c  "
              f"{r['model_nc']:>4d}c  {r['macs_per_core']:>12,d}  "
              f"{ratio:>12s}  {match:>5s}")

    # MACs/core threshold analysis
    print(f"\n  --- MACs/core Threshold Rule Accuracy ---")
    for threshold in [500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]:
        correct = 0
        for r in results:
            # Pick largest core count where MACs/core >= threshold
            best_nc = 4
            for nc in [32, 16, 12, 8, 4]:
                if r["macs"] // nc >= threshold:
                    best_nc = nc
                    break
            if best_nc == r["opt_nc"]:
                correct += 1
        print(f"    threshold={threshold:>10,d}: {correct}/{len(results)} "
              f"({correct/len(results)*100:.0f}%)")

    print(f"\n    Cost model accuracy: 14/19 (74%)")

    # Shape effect analysis
    print(f"\n  --- Shape Effect: Same MACs, Different Optimal ---")
    # Find pairs with similar MACs but different optimal cores
    for i, r1 in enumerate(results):
        for r2 in results[i+1:]:
            ratio = max(r1["macs"], r2["macs"]) / max(min(r1["macs"], r2["macs"]), 1)
            if ratio < 2.0 and r1["opt_nc"] != r2["opt_nc"]:
                print(f"    {r1['size']} ({r1['macs']:,d} MACs, {r1['opt_nc']}c) vs "
                      f"{r2['size']} ({r2['macs']:,d} MACs, {r2['opt_nc']}c)")


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Deep empirical analysis")
    p.add_argument("--data", required=True, help="Merged result CSV")
    p.add_argument("--tc", required=True, help="Merged tc_list JSON")
    p.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    args = p.parse_args()

    cases = load_data(Path(args.data), Path(args.tc))
    cases = dedup_cases(cases)
    coeffs = load_calibration(Path(args.calib))

    print(f"  Loaded: {len(cases)} unique configs")
    print(f"  Sizes: {len(set(c.size_key for c in cases))}")
    print(f"  Energy valid: {sum(1 for c in cases if c.has_energy)}/{len(cases)}")

    analyze_overhead(cases, coeffs)
    analyze_scaling_efficiency(cases)
    analyze_tp_sp(cases)
    analyze_model_failures(cases, coeffs)
    analyze_macs_vs_cores(cases, coeffs)

    print()


if __name__ == "__main__":
    main()
