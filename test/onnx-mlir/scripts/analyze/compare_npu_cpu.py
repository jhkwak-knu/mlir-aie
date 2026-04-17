#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_npu_cpu.py -- Compare NPU optimal config vs CPU 1T/24T measurements.

For each matrix size, extracts:
  - NPU: EDP-optimal config from result CSV (best measured EDP = T * E)
  - CPU 1T: single-thread baseline (numSpm=1 in CPU CSV)
  - CPU 24T: multi-thread baseline (numSpm=24 in CPU CSV)

Computes speedup and energy efficiency ratios.

Usage:
    python3 scripts/analyze/compare_npu_cpu.py \
        --npu out/reports/result_v12.csv \
        --cpu paper/analysis/cpu_baseline_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MeasuredResult:
    """Measured performance and energy for one configuration."""
    M: int
    K: int
    N: int
    num_cores: int
    SPm: int
    SPn: int
    min_us: float          # best iteration time
    batch_min_avg_us: float  # best batch average time
    energy_per_iter_uj: float  # idle-subtracted energy per iteration
    core_energy_uj: float  # core domain energy per iteration


def _skip_comments(f):
    """Yield lines that are not comment lines (starting with #)."""
    for line in f:
        if not line.lstrip().startswith("#"):
            yield line


def load_npu_results(path: str) -> Dict[Tuple[int, int, int], MeasuredResult]:
    """Load NPU results and select EDP-optimal config per matrix size.

    EDP = batch_min_avg_us * batch_min_energy_per_iter_uj (lower is better).
    """
    # Group all configs by (M, K, N)
    by_size = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(_skip_comments(f)):
            if row.get("status") != "PASS":
                continue
            # Skip CPU rows when a combined CSV is passed (CPU rows have SPm=-1).
            try:
                if int(row.get("SPm", -1)) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            t_us = float(row.get("batch_min_avg_us", -1))
            e_uj = float(row.get("batch_min_energy_per_iter_uj", -1))
            if t_us <= 0 or e_uj <= 0:
                # Fallback to legacy columns
                t_us = float(row.get("min_us", -1))
                e_uj = float(row.get("npu_energy_per_iter_uj", -1))
                if t_us <= 0 or e_uj <= 0:
                    continue

            key = (int(row["M"]), int(row["K"]), int(row["N"]))
            core_e = float(row.get("core_energy_per_iter_uj", -1))
            by_size[key].append(MeasuredResult(
                M=key[0], K=key[1], N=key[2],
                num_cores=int(row["numSpm"]),
                SPm=int(row.get("SPm", -1)),
                SPn=int(row.get("SPn", -1)),
                min_us=float(row["min_us"]),
                batch_min_avg_us=t_us,
                energy_per_iter_uj=e_uj,
                core_energy_uj=core_e,
            ))

    # Select EDP-optimal per size
    optimal = {}
    for key, configs in by_size.items():
        best = min(configs, key=lambda c: c.batch_min_avg_us * c.energy_per_iter_uj)
        optimal[key] = best
    return optimal


def load_cpu_results(path: str) -> Tuple[
        Dict[Tuple[int, int, int], MeasuredResult],
        Dict[Tuple[int, int, int], MeasuredResult]]:
    """Load CPU results, split into 1T and 24T by numSpm column."""
    cpu_1t = {}
    cpu_24t = {}
    with open(path) as f:
        for row in csv.DictReader(_skip_comments(f)):
            if row.get("status") != "PASS":
                continue
            # Skip NPU rows when a combined CSV is passed (NPU rows have SPm>=1).
            try:
                if int(row.get("SPm", -1)) >= 1:
                    continue
            except (TypeError, ValueError):
                pass
            threads = int(row["numSpm"])
            key = (int(row["M"]), int(row["K"]), int(row["N"]))
            t_us = float(row.get("batch_min_avg_us", -1))
            e_uj = float(row.get("batch_min_energy_per_iter_uj", -1))
            if t_us <= 0:
                t_us = float(row.get("min_us", -1))
            core_e = float(row.get("core_energy_per_iter_uj", -1))
            r = MeasuredResult(
                M=key[0], K=key[1], N=key[2],
                num_cores=threads,
                SPm=-1, SPn=-1,
                min_us=float(row.get("min_us", -1)),
                batch_min_avg_us=t_us,
                energy_per_iter_uj=e_uj,
                core_energy_uj=core_e,
            )
            if threads == 1:
                cpu_1t[key] = r
            elif threads == 24:
                cpu_24t[key] = r
    return cpu_1t, cpu_24t


def fmt_ratio(cpu_val, npu_val):
    """Format speedup/efficiency ratio (cpu/npu). >1 means NPU is better."""
    if npu_val <= 0 or cpu_val <= 0:
        return "N/A"
    return f"{cpu_val / npu_val:.2f}x"


def fmt_us(val):
    """Format microseconds with appropriate precision."""
    if val < 0:
        return "N/A"
    if val < 100:
        return f"{val:.1f}"
    if val < 10000:
        return f"{val:.0f}"
    return f"{val / 1000:.1f}k"


def fmt_uj(val):
    """Format microjoules."""
    if val <= 0:
        return "N/A"
    if val < 100:
        return f"{val:.1f}"
    if val < 10000:
        return f"{val:.0f}"
    return f"{val / 1000:.1f}k"


def main():
    parser = argparse.ArgumentParser(
        description="Compare NPU optimal vs CPU 1T/24T")
    parser.add_argument("--npu", required=True, help="NPU result CSV")
    parser.add_argument("--cpu", required=True, help="CPU result CSV (50-col)")
    args = parser.parse_args()

    npu_optimal = load_npu_results(args.npu)
    cpu_1t, cpu_24t = load_cpu_results(args.cpu)

    # Collect all sizes (sorted by MACs)
    all_sizes = sorted(
        set(npu_optimal.keys()) | set(cpu_1t.keys()),
        key=lambda s: s[0] * s[1] * s[2]
    )

    # Print header
    sep = "-" * 165
    print()
    print("=" * 165)
    print("  NPU (EDP-Optimal) vs CPU 1T/24T Comparison")
    print("=" * 165)

    # Section 1: Performance (Time)
    print()
    print("  [1] Performance Comparison (batch_min_avg_us, lower is better)")
    print(sep)
    print(f"{'Size':>20s}  {'NPU(us)':>10s} {'Cores':>5s}  "
          f"{'CPU-1T(us)':>10s} {'CPU-24T(us)':>10s}  "
          f"{'1T/NPU':>8s} {'24T/NPU':>8s}  "
          f"{'Winner':>8s}")
    print(sep)

    perf_wins = {"NPU": 0, "CPU-1T": 0, "CPU-24T": 0}
    for key in all_sizes:
        m, k, n = key
        sz = f"{m}x{k}x{n}"

        npu = npu_optimal.get(key)
        c1t = cpu_1t.get(key)
        c24t = cpu_24t.get(key)

        npu_t = npu.batch_min_avg_us if npu else -1
        c1t_t = c1t.batch_min_avg_us if c1t else -1
        c24t_t = c24t.batch_min_avg_us if c24t else -1

        cores = f"{npu.num_cores}" if npu else "N/A"
        ratio_1t = fmt_ratio(c1t_t, npu_t)
        ratio_24t = fmt_ratio(c24t_t, npu_t)

        # Determine winner (lowest time)
        candidates = []
        if npu_t > 0:
            candidates.append(("NPU", npu_t))
        if c1t_t > 0:
            candidates.append(("CPU-1T", c1t_t))
        if c24t_t > 0:
            candidates.append(("CPU-24T", c24t_t))
        winner = min(candidates, key=lambda x: x[1])[0] if candidates else "N/A"
        if winner in perf_wins:
            perf_wins[winner] += 1

        print(f"{sz:>20s}  {fmt_us(npu_t):>10s} {cores:>5s}  "
              f"{fmt_us(c1t_t):>10s} {fmt_us(c24t_t):>10s}  "
              f"{ratio_1t:>8s} {ratio_24t:>8s}  "
              f"{winner:>8s}")

    print(sep)
    print(f"  Performance wins: NPU={perf_wins['NPU']}, "
          f"CPU-1T={perf_wins['CPU-1T']}, CPU-24T={perf_wins['CPU-24T']}  "
          f"(out of {len(all_sizes)} sizes)")

    # Section 2: Energy
    print()
    print("  [2] Energy Comparison (energy_per_iter_uj, idle-subtracted, lower is better)")
    print(sep)
    print(f"{'Size':>20s}  {'NPU(uJ)':>10s} {'Cores':>5s}  "
          f"{'CPU-1T(uJ)':>10s} {'CPU-24T(uJ)':>10s}  "
          f"{'1T/NPU':>8s} {'24T/NPU':>8s}  "
          f"{'Winner':>8s}")
    print(sep)

    energy_wins = {"NPU": 0, "CPU-1T": 0, "CPU-24T": 0}
    for key in all_sizes:
        m, k, n = key
        sz = f"{m}x{k}x{n}"

        npu = npu_optimal.get(key)
        c1t = cpu_1t.get(key)
        c24t = cpu_24t.get(key)

        npu_e = npu.energy_per_iter_uj if npu else -1
        c1t_e = c1t.energy_per_iter_uj if c1t else -1
        c24t_e = c24t.energy_per_iter_uj if c24t else -1

        cores = f"{npu.num_cores}" if npu else "N/A"
        ratio_1t = fmt_ratio(c1t_e, npu_e)
        ratio_24t = fmt_ratio(c24t_e, npu_e)

        candidates = []
        if npu_e > 0:
            candidates.append(("NPU", npu_e))
        if c1t_e > 0:
            candidates.append(("CPU-1T", c1t_e))
        if c24t_e > 0:
            candidates.append(("CPU-24T", c24t_e))
        winner = min(candidates, key=lambda x: x[1])[0] if candidates else "N/A"
        if winner in energy_wins:
            energy_wins[winner] += 1

        print(f"{sz:>20s}  {fmt_uj(npu_e):>10s} {cores:>5s}  "
              f"{fmt_uj(c1t_e):>10s} {fmt_uj(c24t_e):>10s}  "
              f"{ratio_1t:>8s} {ratio_24t:>8s}  "
              f"{winner:>8s}")

    print(sep)
    print(f"  Energy wins: NPU={energy_wins['NPU']}, "
          f"CPU-1T={energy_wins['CPU-1T']}, CPU-24T={energy_wins['CPU-24T']}  "
          f"(out of {len(all_sizes)} sizes)")

    # Section 3: EDP (Energy-Delay Product)
    print()
    print("  [3] EDP Comparison (T_us * E_uJ, lower is better)")
    print(sep)
    print(f"{'Size':>20s}  {'NPU EDP':>14s} {'Cores':>5s}  "
          f"{'CPU-1T EDP':>14s} {'CPU-24T EDP':>14s}  "
          f"{'1T/NPU':>8s} {'24T/NPU':>8s}  "
          f"{'Winner':>8s}")
    print(sep)

    edp_wins = {"NPU": 0, "CPU-1T": 0, "CPU-24T": 0}
    for key in all_sizes:
        m, k, n = key
        sz = f"{m}x{k}x{n}"

        npu = npu_optimal.get(key)
        c1t = cpu_1t.get(key)
        c24t = cpu_24t.get(key)

        npu_edp = (npu.batch_min_avg_us * npu.energy_per_iter_uj) if npu else -1
        c1t_edp = (c1t.batch_min_avg_us * c1t.energy_per_iter_uj) \
            if c1t and c1t.batch_min_avg_us > 0 and c1t.energy_per_iter_uj > 0 else -1
        c24t_edp = (c24t.batch_min_avg_us * c24t.energy_per_iter_uj) \
            if c24t and c24t.batch_min_avg_us > 0 and c24t.energy_per_iter_uj > 0 else -1

        cores = f"{npu.num_cores}" if npu else "N/A"

        def fmt_edp(val):
            if val <= 0:
                return "N/A"
            if val < 1e6:
                return f"{val:.0f}"
            return f"{val / 1e6:.2f}M"

        ratio_1t = fmt_ratio(c1t_edp, npu_edp)
        ratio_24t = fmt_ratio(c24t_edp, npu_edp)

        candidates = []
        if npu_edp > 0:
            candidates.append(("NPU", npu_edp))
        if c1t_edp > 0:
            candidates.append(("CPU-1T", c1t_edp))
        if c24t_edp > 0:
            candidates.append(("CPU-24T", c24t_edp))
        winner = min(candidates, key=lambda x: x[1])[0] if candidates else "N/A"
        if winner in edp_wins:
            edp_wins[winner] += 1

        print(f"{sz:>20s}  {fmt_edp(npu_edp):>14s} {cores:>5s}  "
              f"{fmt_edp(c1t_edp):>14s} {fmt_edp(c24t_edp):>14s}  "
              f"{ratio_1t:>8s} {ratio_24t:>8s}  "
              f"{winner:>8s}")

    print(sep)
    print(f"  EDP wins: NPU={edp_wins['NPU']}, "
          f"CPU-1T={edp_wins['CPU-1T']}, CPU-24T={edp_wins['CPU-24T']}  "
          f"(out of {len(all_sizes)} sizes)")

    # Section 4: Summary
    print()
    print("=" * 165)
    print("  Summary")
    print("=" * 165)

    # Geometric mean speedup/efficiency across all sizes
    import math
    t_ratios_1t = []
    t_ratios_24t = []
    e_ratios_1t = []
    e_ratios_24t = []
    edp_ratios_1t = []
    edp_ratios_24t = []

    for key in all_sizes:
        npu = npu_optimal.get(key)
        c1t = cpu_1t.get(key)
        c24t = cpu_24t.get(key)
        if not npu:
            continue

        if c1t and c1t.batch_min_avg_us > 0 and npu.batch_min_avg_us > 0:
            t_ratios_1t.append(c1t.batch_min_avg_us / npu.batch_min_avg_us)
        if c24t and c24t.batch_min_avg_us > 0 and npu.batch_min_avg_us > 0:
            t_ratios_24t.append(c24t.batch_min_avg_us / npu.batch_min_avg_us)

        if c1t and c1t.energy_per_iter_uj > 0 and npu.energy_per_iter_uj > 0:
            e_ratios_1t.append(c1t.energy_per_iter_uj / npu.energy_per_iter_uj)
        if c24t and c24t.energy_per_iter_uj > 0 and npu.energy_per_iter_uj > 0:
            e_ratios_24t.append(c24t.energy_per_iter_uj / npu.energy_per_iter_uj)

        npu_edp = npu.batch_min_avg_us * npu.energy_per_iter_uj
        if c1t and c1t.batch_min_avg_us > 0 and c1t.energy_per_iter_uj > 0:
            c1t_edp = c1t.batch_min_avg_us * c1t.energy_per_iter_uj
            edp_ratios_1t.append(c1t_edp / npu_edp)
        if c24t and c24t.batch_min_avg_us > 0 and c24t.energy_per_iter_uj > 0:
            c24t_edp = c24t.batch_min_avg_us * c24t.energy_per_iter_uj
            edp_ratios_24t.append(c24t_edp / npu_edp)

    def geo_mean(vals):
        if not vals:
            return 0
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    print(f"  Geometric mean ratios (CPU/NPU, >1 means NPU wins):")
    print(f"    Speedup  : 1T/NPU = {geo_mean(t_ratios_1t):.2f}x  "
          f"24T/NPU = {geo_mean(t_ratios_24t):.2f}x")
    print(f"    Energy   : 1T/NPU = {geo_mean(e_ratios_1t):.2f}x  "
          f"24T/NPU = {geo_mean(e_ratios_24t):.2f}x")
    print(f"    EDP      : 1T/NPU = {geo_mean(edp_ratios_1t):.2f}x  "
          f"24T/NPU = {geo_mean(edp_ratios_24t):.2f}x")

    # Crossover point
    print()
    crossover_t = None
    crossover_e = None
    for key in all_sizes:
        m, k, n = key
        npu = npu_optimal.get(key)
        c24t = cpu_24t.get(key)
        if not npu or not c24t:
            continue
        macs = m * k * n
        if npu.batch_min_avg_us < c24t.batch_min_avg_us and crossover_t is None:
            crossover_t = key
        if npu.energy_per_iter_uj > 0 and c24t.energy_per_iter_uj > 0:
            if npu.energy_per_iter_uj < c24t.energy_per_iter_uj and crossover_e is None:
                crossover_e = key

    if crossover_t:
        print(f"  Performance crossover (NPU faster than CPU-24T): "
              f"{crossover_t[0]}x{crossover_t[1]}x{crossover_t[2]}")
    if crossover_e:
        print(f"  Energy crossover (NPU more efficient than CPU-24T): "
              f"{crossover_e[0]}x{crossover_e[1]}x{crossover_e[2]}")

    print()


if __name__ == "__main__":
    main()
