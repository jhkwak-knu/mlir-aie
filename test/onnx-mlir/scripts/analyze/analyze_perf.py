#!/usr/bin/env python3
"""Analyze performance measurement results from result CSV.

Reads the merged CSV (159 cases, 6 matrix sizes: 32~1024) and produces
9 analysis sections covering execution structure, measurement reliability,
spatial/temporal parallelization effects, data reuse, kernel utilization,
optimal configuration selection, and host overhead analysis.

Usage:
    python3 scripts/analyze/analyze_perf.py --csv out/calibration/result_v7_trace.csv
    python3 scripts/analyze/analyze_perf.py --csv out/calibration/result_v7_trace.csv \
        --output out/calibration/perf_analysis_v7.txt
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from statistics import median, mean, stdev


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AXIS_NAMES = {0: "M", 1: "N", 2: "K"}

# RAPL measurement reliability threshold (5ms minimum wall time)
MIN_WALL_TIME_S = 0.005

# Ratio distribution buckets for (matmul_npu_us / avg_us)
RATIO_BUCKETS = [(0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 2.0)]


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> list:
    """Load CSV and convert numeric fields. Returns list of dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    int_fields = [
        "case_index", "numSpm", "SPm", "SPn", "TPm", "TPk", "TPn",
        "TM", "TK", "TN", "M", "K", "N", "errors", "iters", "warmup",
        "dispatch_cy", "kernel_total_cy", "ss_iter_cy", "ss_kernel_cy",
        "n_tiles", "n_valid_dispatches", "host_steps", "matmul_npu_cy",
        "missing_events",
    ]
    float_fields = [
        "t_total_pred", "avg_us", "min_us", "max_us",
        "idle_pkg_mw", "active_pkg_mw", "npu_power_mw",
        "npu_energy_uj", "npu_energy_per_iter_uj", "wall_elapsed_s",
        "dispatch_us", "kernel_pct", "ss_iter_us", "ss_kernel_util_pct",
        "gflops", "matmul_npu_us", "data_quality", "host_overhead_us",
    ]

    for r in rows:
        for k in int_fields:
            if k in r and r[k]:
                try:
                    r[k] = int(r[k])
                except ValueError:
                    r[k] = int(float(r[k]))
        for k in float_fields:
            if k in r and r[k]:
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = 0.0
        r["doubleBuffer"] = r.get("doubleBuffer", "False") == "True"
        r["pres_active"] = r.get("pres_active", "False") == "True"
        r["cores"] = r["SPm"] * r["SPn"]

    return rows


def derive_tp_order_inner(r: dict) -> str:
    """Derive tpOrder[0] (innermost temporal axis) from host_steps + pres_active.

    Returns: "M", "N", "K", "M/N" (ambiguous), or "-" (no temporal iteration).
    """
    tpm, tpk, tpn = r["TPm"], r["TPk"], r["TPn"]
    hs = r["host_steps"]

    # No temporal iteration
    if tpm == 1 and tpk == 1 and tpn == 1:
        return "-"

    candidates = []
    if tpk * tpn == hs:
        candidates.append(0)  # M inner
    if tpm * tpk == hs:
        candidates.append(1)  # N inner
    if tpm * tpn == hs:
        candidates.append(2)  # K inner

    if len(candidates) == 1:
        return AXIS_NAMES[candidates[0]]

    # Disambiguate with pres_active
    # pres required when tpOrder[0] != K AND TPk > 1
    pres = r["pres_active"]
    if pres:
        candidates = [c for c in candidates if c != 2]
    elif tpk > 1:
        candidates = [c for c in candidates if c == 2]

    if len(candidates) == 1:
        return AXIS_NAMES[candidates[0]]
    if set(candidates) == {0, 1}:
        return "M/N"
    return "/".join(AXIS_NAMES[c] for c in candidates) if candidates else "?"


def compute_gflops(r: dict) -> float:
    """Compute GFLOPS from matrix dimensions and avg_us."""
    flops = 2.0 * r["M"] * r["K"] * r["N"]
    return flops / (r["avg_us"] * 1e3) if r["avg_us"] > 0 else 0.0


def preprocess(rows: list) -> list:
    """Add derived fields to each row."""
    for r in rows:
        r["inner_axis"] = derive_tp_order_inner(r)
        r["ipd"] = r["TPm"] * r["TPk"] * r["TPn"] // max(r["host_steps"], 1)
        r["size_key"] = f"{r['M']}x{r['K']}x{r['N']}"
        r["gflops_calc"] = compute_gflops(r)
    return rows


# ---------------------------------------------------------------------------
# Section 1: Execution Structure Summary
# ---------------------------------------------------------------------------
def section1_execution_structure(rows: list) -> None:
    print("=" * 70)
    print("Section 1: Execution Structure Summary")
    print("=" * 70)
    print()
    print("One MatMul = host_steps dispatches (host loop)")
    print("  host_steps = tp[tpOrder[1]] * tp[tpOrder[2]]")
    print("One dispatch (RuntimeSequence):")
    print("  tpOrder[0] axis: tp[tpOrder[0]] iterations (NPU CoreOp)")
    print("  Per iteration: reuse-excluded data DMA + kernel execution")
    print("  Fixed data: reuse-axis operand transferred once")
    print()
    print("  tpOrder[0] | Reuse target | Per-step transfer | Fixed (1x)")
    print("  -----------|-------------|-------------------|----------")
    print("  M (0)      | RHS         | LHS, PRES, RES    | RHS")
    print("  N (1)      | LHS         | RHS, PRES, RES    | LHS")
    print("  K (2)      | (accum)     | LHS, RHS           | RES")
    print()

    by_size = defaultdict(list)
    for r in rows:
        by_size[r["M"]].append(r)

    for m in sorted(by_size):
        group = by_size[m]
        core_counts = defaultdict(int)
        hs_counts = defaultdict(int)
        inner_counts = defaultdict(int)
        pres_count = sum(1 for r in group if r["pres_active"])

        for r in group:
            core_counts[r["cores"]] += 1
            hs_counts[r["host_steps"]] += 1
            inner_counts[r["inner_axis"]] += 1

        n = len(group)
        cores_str = ", ".join(f"{c}({v})" for c, v in sorted(core_counts.items()))
        hs_str = ", ".join(f"{h}({v})" for h, v in sorted(hs_counts.items()))
        inner_str = ", ".join(
            f"{a}({v})" for a, v in sorted(inner_counts.items())
        )

        print(f"=== {m}x{m}x{m} ({n} cases) ===")
        print(f"  Cores: {cores_str}")
        print(f"  host_steps: {hs_str}")
        print(f"  tpOrder inner: {inner_str}")
        print(f"  PRES active: {pres_count}/{n}")
        print()


# ---------------------------------------------------------------------------
# Section 2: Measurement Reliability
# ---------------------------------------------------------------------------
def section2_measurement_reliability(rows: list) -> None:
    print("=" * 70)
    print("Section 2: Measurement Reliability")
    print("=" * 70)
    print()

    # 2-A: matmul_npu_us vs avg_us
    negative_overhead = [r for r in rows if r["matmul_npu_us"] > r["avg_us"]]
    n_neg = len(negative_overhead)
    n_total = len(rows)

    by_size_neg = defaultdict(int)
    for r in negative_overhead:
        by_size_neg[r["M"]] += 1

    ratios = [r["matmul_npu_us"] / r["avg_us"] for r in rows if r["avg_us"] > 0]
    neg_ratios = [
        r["matmul_npu_us"] / r["avg_us"]
        for r in negative_overhead if r["avg_us"] > 0
    ]

    print("--- matmul_npu_us vs avg_us ---")
    print(f"  Negative overhead (npu > host): {n_neg}/{n_total} "
          f"({100*n_neg/n_total:.1f}%)")
    size_str = ", ".join(
        f"{m}={by_size_neg.get(m, 0)}"
        for m in sorted(set(r["M"] for r in rows))
    )
    print(f"  By size: {size_str}")
    if ratios:
        print(f"  Median ratio (npu/host): {median(ratios):.3f} (all), ", end="")
        if neg_ratios:
            print(f"{median(neg_ratios):.3f} (negative only)")
        else:
            print("N/A (negative only)")
    print()

    # Ratio distribution
    print("  Ratio distribution (npu/host):")
    for lo, hi in RATIO_BUCKETS:
        count = sum(1 for rt in ratios if lo <= rt < hi)
        print(f"    [{lo:.1f}, {hi:.1f}): {count}")
    over2 = sum(1 for rt in ratios if rt >= 2.0)
    print(f"    [2.0, inf): {over2}")
    print()

    # Worst cases (highest ratio)
    worst = sorted(rows, key=lambda r: r["matmul_npu_us"] / max(r["avg_us"], 0.01),
                   reverse=True)[:5]
    print("  Top 5 highest npu/host ratio:")
    for r in worst:
        ratio = r["matmul_npu_us"] / max(r["avg_us"], 0.01)
        print(f"    case_{r['case_index']:03d}: ratio={ratio:.2f} "
              f"M={r['M']} cores={r['cores']} hs={r['host_steps']} "
              f"pres={r['pres_active']}")
    print()

    # 2-B: RAPL reliability
    valid_rapl = [r for r in rows if r["wall_elapsed_s"] >= MIN_WALL_TIME_S]
    neg_energy = [r for r in rows if r["npu_energy_uj"] < 0]

    print("--- RAPL Energy Reliability ---")
    print(f"  Valid (wall >= 5ms): {len(valid_rapl)}/{n_total}")
    print(f"  Negative npu_energy: {len(neg_energy)}/{n_total}")
    by_size_neg_e = defaultdict(int)
    for r in neg_energy:
        by_size_neg_e[r["M"]] += 1
    size_str = ", ".join(
        f"{m}={by_size_neg_e.get(m, 0)}"
        for m in sorted(set(r["M"] for r in rows))
    )
    print(f"  Negative energy by size: {size_str}")
    print()

    # 2-C: host_overhead analysis
    overheads = [r["host_overhead_us"] for r in rows]
    print("--- Host Overhead (avg_us - matmul_npu_us) ---")
    print(f"  Mean: {mean(overheads):.2f} us")
    print(f"  Median: {median(overheads):.2f} us")
    pos_oh = [o for o in overheads if o > 0]
    if pos_oh:
        print(f"  Positive only - Mean: {mean(pos_oh):.2f} us, "
              f"Median: {median(pos_oh):.2f} us")
    print()

    # 2-D: Trace data quality distribution
    quality_vals = [r["data_quality"] for r in rows]
    q_buckets = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 2.0)]
    print("--- Trace Data Quality Distribution ---")
    print(f"  Mean: {mean(quality_vals):.3f}, Median: {median(quality_vals):.3f}")
    for lo, hi in q_buckets:
        count = sum(1 for q in quality_vals if lo <= q < hi)
        pct = 100 * count / n_total
        print(f"    [{lo:.2f}, {hi:.2f}): {count:>4d} ({pct:>5.1f}%)")
    above2 = sum(1 for q in quality_vals if q >= 2.0)
    if above2:
        print(f"    [2.0, inf):   {above2:>4d}")

    # Quality by size
    print()
    print("  Quality by matrix size:")
    by_size_q = defaultdict(list)
    for r in rows:
        by_size_q[r["M"]].append(r["data_quality"])
    for m in sorted(by_size_q):
        vals = by_size_q[m]
        above_50 = sum(1 for q in vals if q >= 0.5)
        print(f"    {m:>5d}: median={median(vals):.3f} "
              f"mean={mean(vals):.3f} "
              f">=0.5: {above_50}/{len(vals)}")
    print()


# ---------------------------------------------------------------------------
# Section 3: Performance by Matrix Size
# ---------------------------------------------------------------------------
def section3_perf_by_size(rows: list) -> None:
    print("=" * 70)
    print("Section 3: Performance by Matrix Size")
    print("=" * 70)
    print()

    header = (f"{'Size':<14s} | {'Cases':>5s} | {'Best avg_us':>11s} | "
              f"{'Best GFLOPS':>11s} | Best Config")
    print(header)
    print("-" * len(header))

    by_size = defaultdict(list)
    for r in rows:
        by_size[r["M"]].append(r)

    for m in sorted(by_size):
        group = by_size[m]
        best = min(group, key=lambda r: r["avg_us"])
        gf = best["gflops_calc"]
        config = (f"SPm={best['SPm']} SPn={best['SPn']} "
                  f"cores={best['cores']} hs={best['host_steps']}")
        print(f"{best['size_key']:<14s} | {len(group):>5d} | "
              f"{best['avg_us']:>11.2f} | {gf:>11.2f} | {config}")

    print()

    # Also show worst and median
    print("Detailed statistics (avg_us):")
    print(f"{'Size':<14s} | {'Min':>10s} | {'Median':>10s} | {'Max':>10s} | "
          f"{'Mean':>10s} | {'StdDev':>10s}")
    print("-" * 80)
    for m in sorted(by_size):
        vals = sorted(r["avg_us"] for r in by_size[m])
        if len(vals) > 1:
            print(f"{m}x{m}x{m:<5d} | {min(vals):>10.2f} | "
                  f"{median(vals):>10.2f} | {max(vals):>10.2f} | "
                  f"{mean(vals):>10.2f} | {stdev(vals):>10.2f}")
        else:
            v = vals[0]
            print(f"{m}x{m}x{m:<5d} | {v:>10.2f} | {v:>10.2f} | "
                  f"{v:>10.2f} | {v:>10.2f} | {'N/A':>10s}")
    print()


# ---------------------------------------------------------------------------
# Section 4: Spatial Parallelization (Core Scaling)
# ---------------------------------------------------------------------------
def section4_spatial_scaling(rows: list) -> None:
    print("=" * 70)
    print("Section 4: Spatial Parallelization (Core Scaling)")
    print("=" * 70)
    print()

    by_size = defaultdict(list)
    for r in rows:
        by_size[r["M"]].append(r)

    for m in sorted(by_size):
        group = by_size[m]

        # Group by core count, take best avg_us per core count
        by_cores = defaultdict(list)
        for r in group:
            by_cores[r["cores"]].append(r)

        core_counts = sorted(by_cores)
        if len(core_counts) < 2:
            continue

        # Best avg_us per core count
        best_by_core = {}
        for nc in core_counts:
            best = min(by_cores[nc], key=lambda r: r["avg_us"])
            best_by_core[nc] = best

        base_core = core_counts[0]
        base_us = best_by_core[base_core]["avg_us"]

        print(f"=== Size={m}x{m}x{m} ===")
        print(f"  {'Cores':>5s}  {'Best avg_us':>11s}  {'Speedup':>8s}  "
              f"{'Efficiency':>10s}  {'Cases':>5s}  Config")
        for nc in core_counts:
            r = best_by_core[nc]
            speedup = base_us / r["avg_us"] if r["avg_us"] > 0 else 0
            ideal_speedup = nc / base_core
            efficiency = speedup / ideal_speedup if ideal_speedup > 0 else 0
            config = (f"SP=({r['SPm']},{r['SPn']}) "
                      f"TP=({r['TPm']},{r['TPk']},{r['TPn']})")
            print(f"  {nc:>5d}  {r['avg_us']:>11.2f}  {speedup:>7.2f}x  "
                  f"{efficiency:>10.3f}  {len(by_cores[nc]):>5d}  {config}")
        print()

    # Cross-size core scaling summary
    print("--- Cross-size Core Scaling Summary ---")
    print(f"{'Size':<14s} | {'4-core':>10s} | {'8-core':>10s} | "
          f"{'16-core':>10s} | {'32-core':>10s}")
    print("-" * 70)
    for m in sorted(by_size):
        by_cores = defaultdict(list)
        for r in by_size[m]:
            by_cores[r["cores"]].append(r)
        vals = {}
        for nc in [4, 8, 16, 32]:
            if nc in by_cores:
                vals[nc] = min(r["avg_us"] for r in by_cores[nc])
        line = f"{m}x{m}x{m:<5d} |"
        for nc in [4, 8, 16, 32]:
            if nc in vals:
                line += f" {vals[nc]:>10.2f} |"
            else:
                line += f" {'---':>10s} |"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Section 5: Temporal Parallelization (host_steps Scaling)
# ---------------------------------------------------------------------------
def section5_temporal_scaling(rows: list) -> None:
    print("=" * 70)
    print("Section 5: Temporal Parallelization (host_steps Effect)")
    print("=" * 70)
    print()

    by_size = defaultdict(list)
    for r in rows:
        by_size[r["M"]].append(r)

    for m in sorted(by_size):
        group = by_size[m]

        # Group by (cores, host_steps)
        by_core_hs = defaultdict(list)
        for r in group:
            by_core_hs[(r["cores"], r["host_steps"])].append(r)

        # Find core counts with multiple host_steps values
        core_hs_map = defaultdict(dict)
        for (nc, hs), rlist in by_core_hs.items():
            best = min(rlist, key=lambda r: r["avg_us"])
            core_hs_map[nc][hs] = best

        printed_any = False
        for nc in sorted(core_hs_map):
            hs_map = core_hs_map[nc]
            if len(hs_map) < 2:
                continue

            if not printed_any:
                print(f"=== Size={m}x{m}x{m} ===")
                printed_any = True

            print(f"  cores={nc}:")
            print(f"    {'steps':>6s}  {'avg_us':>10s}  {'per_step':>10s}  "
                  f"{'overhead/step':>13s}  inner_axis")

            base_hs = min(hs_map)
            base_per_step = hs_map[base_hs]["avg_us"] / base_hs
            for hs in sorted(hs_map):
                r = hs_map[hs]
                per_step = r["avg_us"] / hs
                if hs == base_hs:
                    oh_str = "---"
                else:
                    oh = per_step - base_per_step
                    oh_str = f"{oh:>+10.2f} us"
                print(f"    {hs:>6d}  {r['avg_us']:>10.2f}  {per_step:>10.2f}  "
                      f"{oh_str:>13s}  {r['inner_axis']}")
            print()

        if printed_any:
            print()


# ---------------------------------------------------------------------------
# Section 6: Data Reuse (tpOrder Effect)
# ---------------------------------------------------------------------------
def section6_data_reuse(rows: list) -> None:
    print("=" * 70)
    print("Section 6: Data Reuse (tpOrder[0] Effect)")
    print("=" * 70)
    print()
    print("Note: tpOrder[0] derived from host_steps + pres_active.")
    print("  'M/N' = ambiguous (TPm == TPn, cannot distinguish).")
    print("  '-'   = no temporal iteration (all TP=1).")
    print()

    # Group by (M, cores) and find cases with different inner axes
    by_size_cores = defaultdict(list)
    for r in rows:
        if r["inner_axis"] in ("-", "?"):
            continue
        by_size_cores[(r["M"], r["cores"])].append(r)

    for (m, nc) in sorted(by_size_cores):
        group = by_size_cores[(m, nc)]
        by_axis = defaultdict(list)
        for r in group:
            by_axis[r["inner_axis"]].append(r)

        if len(by_axis) < 2:
            continue

        print(f"=== Size={m}x{m}x{m}, cores={nc} ===")
        print(f"  {'Axis':>5s}  {'Cases':>5s}  {'Best avg_us':>11s}  "
              f"{'Med avg_us':>10s}  {'Med util%':>9s}  "
              f"{'Med ss_iter_cy':>14s}")

        axis_bests = {}
        for ax in sorted(by_axis):
            rlist = by_axis[ax]
            best = min(rlist, key=lambda r: r["avg_us"])
            med_avg = median(r["avg_us"] for r in rlist)
            med_util = median(r["ss_kernel_util_pct"] for r in rlist)
            med_iter = median(r["ss_iter_cy"] for r in rlist)
            axis_bests[ax] = best["avg_us"]
            print(f"  {ax:>5s}  {len(rlist):>5d}  {best['avg_us']:>11.2f}  "
                  f"{med_avg:>10.2f}  {med_util:>9.1f}  {med_iter:>14.0f}")

        if len(axis_bests) >= 2:
            best_ax = min(axis_bests, key=axis_bests.get)
            worst_ax = max(axis_bests, key=axis_bests.get)
            improvement = (axis_bests[worst_ax] - axis_bests[best_ax]) / axis_bests[worst_ax] * 100
            print(f"  Best reuse: {best_ax}-inner "
                  f"({improvement:.1f}% faster than {worst_ax}-inner)")
        print()


# ---------------------------------------------------------------------------
# Section 7: Kernel Utilization
# ---------------------------------------------------------------------------
def section7_kernel_utilization(rows: list) -> None:
    print("=" * 70)
    print("Section 7: Kernel Utilization")
    print("=" * 70)
    print()

    by_size = defaultdict(list)
    for r in rows:
        by_size[r["M"]].append(r)

    print("--- ss_kernel_util_pct distribution by size ---")
    print(f"{'Size':>6s}  {'Median':>7s}  {'Mean':>7s}  {'Min':>7s}  "
          f"{'Max':>7s}  {'StdDev':>7s}")
    for m in sorted(by_size):
        vals = [r["ss_kernel_util_pct"] for r in by_size[m]]
        if len(vals) > 1:
            print(f"{m:>6d}  {median(vals):>7.1f}  {mean(vals):>7.1f}  "
                  f"{min(vals):>7.1f}  {max(vals):>7.1f}  {stdev(vals):>7.1f}")
        else:
            print(f"{m:>6d}  {vals[0]:>7.1f}  {vals[0]:>7.1f}  "
                  f"{vals[0]:>7.1f}  {vals[0]:>7.1f}  {'N/A':>7s}")
    print()

    # Top 5 highest utilization
    sorted_by_util = sorted(rows, key=lambda r: r["ss_kernel_util_pct"],
                            reverse=True)
    print("--- Top 5 Highest Utilization ---")
    for r in sorted_by_util[:5]:
        print(f"  case_{r['case_index']:03d}: util={r['ss_kernel_util_pct']:.1f}% "
              f"M={r['M']} cores={r['cores']} hs={r['host_steps']} "
              f"inner={r['inner_axis']} avg_us={r['avg_us']:.2f}")
    print()

    # Top 5 lowest (excluding near-zero which may be measurement noise)
    valid_util = [r for r in rows if r["ss_kernel_util_pct"] > 0]
    sorted_low = sorted(valid_util, key=lambda r: r["ss_kernel_util_pct"])
    print("--- Top 5 Lowest Utilization (>0%) ---")
    for r in sorted_low[:5]:
        print(f"  case_{r['case_index']:03d}: util={r['ss_kernel_util_pct']:.1f}% "
              f"M={r['M']} cores={r['cores']} hs={r['host_steps']} "
              f"inner={r['inner_axis']} avg_us={r['avg_us']:.2f}")
    print()

    # kernel_pct analysis (overall dispatch-level kernel percentage)
    kp_vals = [r["kernel_pct"] for r in rows if r.get("kernel_pct", 0) > 0]
    if kp_vals:
        print("--- kernel_pct (dispatch-level, all tiles) ---")
        print(f"  Median: {median(kp_vals):.1f}%, Mean: {mean(kp_vals):.1f}%")
        print(f"  Min: {min(kp_vals):.1f}%, Max: {max(kp_vals):.1f}%")
        print()
        print("  kernel_pct by size:")
        by_size_kp = defaultdict(list)
        for r in rows:
            if r.get("kernel_pct", 0) > 0:
                by_size_kp[r["M"]].append(r["kernel_pct"])
        for m in sorted(by_size_kp):
            vals = by_size_kp[m]
            print(f"    {m:>5d}: median={median(vals):>5.1f}%  "
                  f"mean={mean(vals):>5.1f}%  "
                  f"range=[{min(vals):.1f}, {max(vals):.1f}]")
        print()

    # ss_kernel_util_pct vs kernel_pct comparison
    pairs = [(r["ss_kernel_util_pct"], r["kernel_pct"])
             for r in rows
             if r.get("ss_kernel_util_pct", 0) > 0 and r.get("kernel_pct", 0) > 0]
    if len(pairs) >= 3:
        su, kp = zip(*pairs)
        r_corr = _pearson(list(su), list(kp))
        print(f"  ss_kernel_util_pct vs kernel_pct correlation: r = {r_corr:.3f}")
        print()

    # Correlation with host_steps and cores
    if len(rows) > 2:
        utils = [r["ss_kernel_util_pct"] for r in rows]
        hs_vals = [math.log2(max(r["host_steps"], 1)) for r in rows]
        core_vals = [r["cores"] for r in rows]
        avg_vals = [r["avg_us"] for r in rows]

        r_hs = _pearson(utils, hs_vals)
        r_cores = _pearson(utils, core_vals)
        r_avg = _pearson(utils, avg_vals)

        print("--- Correlations ---")
        print(f"  util vs log2(host_steps): r = {r_hs:+.3f}")
        print(f"  util vs cores:            r = {r_cores:+.3f}")
        print(f"  util vs avg_us:           r = {r_avg:+.3f}")
    print()


def _pearson(x: list, y: list) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = mean(x), mean(y)
    sx = sum((xi - mx) ** 2 for xi in x)
    sy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = math.sqrt(sx * sy)
    return sxy / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Section 8: Optimal Configuration per Size
# ---------------------------------------------------------------------------
def section8_optimal_config(rows: list) -> None:
    print("=" * 70)
    print("Section 8: Optimal Configuration per Size")
    print("=" * 70)
    print()
    print("Selection criteria: lowest avg_us (host-measured)")
    print()

    header = (f"{'Size':<14s} | {'avg_us':>10s} | {'GFLOPS':>7s} | "
              f"{'Cores':>5s} | {'TM':>3s} {'TK':>3s} {'TN':>3s} | "
              f"{'tpOrd':>5s} | {'hs':>4s} | {'util%':>5s} | "
              f"{'case':>5s}")
    print(header)
    print("-" * len(header))

    by_size = defaultdict(list)
    for r in rows:
        by_size[r["M"]].append(r)

    for m in sorted(by_size):
        best = min(by_size[m], key=lambda r: r["avg_us"])
        gf = best["gflops_calc"]
        print(f"{best['size_key']:<14s} | {best['avg_us']:>10.2f} | "
              f"{gf:>7.2f} | {best['cores']:>5d} | "
              f"{best['TM']:>3d} {best['TK']:>3d} {best['TN']:>3d} | "
              f"{best['inner_axis']:>5s} | {best['host_steps']:>4d} | "
              f"{best['ss_kernel_util_pct']:>5.1f} | "
              f"{best['case_index']:>5d}")

    print()

    # Also show best per (core count, tpOrder inner) per size
    print("--- Best config per core count ---")
    print(f"{'Size':<14s} | {'Cores':>5s} | {'avg_us':>10s} | {'GFLOPS':>7s} | "
          f"SP | TP | {'inner':>5s} | {'hs':>4s}")
    print("-" * 80)

    for m in sorted(by_size):
        by_cores = defaultdict(list)
        for r in by_size[m]:
            by_cores[r["cores"]].append(r)
        for nc in sorted(by_cores):
            best = min(by_cores[nc], key=lambda r: r["avg_us"])
            gf = best["gflops_calc"]
            print(f"{best['size_key']:<14s} | {nc:>5d} | {best['avg_us']:>10.2f} | "
                  f"{gf:>7.2f} | ({best['SPm']},{best['SPn']}) | "
                  f"({best['TPm']},{best['TPk']},{best['TPn']}) | "
                  f"{best['inner_axis']:>5s} | {best['host_steps']:>4d}")
    print()


# ---------------------------------------------------------------------------
# Section 9: Host Overhead Analysis
# ---------------------------------------------------------------------------
def section9_host_overhead(rows: list) -> None:
    print("=" * 70)
    print("Section 9: Host Overhead Analysis")
    print("=" * 70)
    print()
    print("host_overhead_us = avg_us - matmul_npu_us")
    print("per_dispatch_oh = host_overhead_us / host_steps")
    print()

    # Filter to high-quality cases (positive overhead, quality >= 0.5)
    hq_rows = [r for r in rows
                if r["host_overhead_us"] > 0 and r["data_quality"] >= 0.5]
    all_oh = [r["host_overhead_us"] for r in rows if r["data_quality"] >= 0.5]

    print(f"  Cases with quality >= 0.5: {len(all_oh)}")
    print(f"  Positive overhead: {len(hq_rows)}")
    if not hq_rows:
        print("  No valid cases for analysis.")
        print()
        return

    # Per-dispatch overhead
    per_dispatch = [r["host_overhead_us"] / max(r["host_steps"], 1)
                    for r in hq_rows]
    print()
    print(f"--- Per-dispatch host overhead (us) ---")
    print(f"  Mean: {mean(per_dispatch):.2f} us")
    print(f"  Median: {median(per_dispatch):.2f} us")
    if len(per_dispatch) > 1:
        sd = stdev(per_dispatch)
        cv = sd / mean(per_dispatch) * 100 if mean(per_dispatch) > 0 else 0
        print(f"  StdDev: {sd:.2f} us, CV: {cv:.1f}%")
    print()

    # By matrix size
    print("  By matrix size:")
    by_size = defaultdict(list)
    for r in hq_rows:
        by_size[r["M"]].append(r["host_overhead_us"] / max(r["host_steps"], 1))

    print(f"  {'Size':>6s}  {'N':>4s}  {'Mean':>10s}  {'Median':>10s}  {'StdDev':>10s}")
    for m in sorted(by_size):
        vals = by_size[m]
        sd_str = f"{stdev(vals):.2f}" if len(vals) > 1 else "N/A"
        print(f"  {m:>6d}  {len(vals):>4d}  {mean(vals):>10.2f}  "
              f"{median(vals):>10.2f}  {sd_str:>10s}")
    print()

    # By core count
    print("  By core count:")
    by_cores = defaultdict(list)
    for r in hq_rows:
        by_cores[r["cores"]].append(
            r["host_overhead_us"] / max(r["host_steps"], 1))

    print(f"  {'Cores':>5s}  {'N':>4s}  {'Mean':>10s}  {'Median':>10s}")
    for nc in sorted(by_cores):
        vals = by_cores[nc]
        print(f"  {nc:>5d}  {len(vals):>4d}  {mean(vals):>10.2f}  "
              f"{median(vals):>10.2f}")
    print()

    # Overhead as fraction of avg_us
    oh_fracs = [r["host_overhead_us"] / r["avg_us"] * 100
                for r in hq_rows if r["avg_us"] > 0]
    if oh_fracs:
        print(f"--- Host overhead as % of avg_us ---")
        print(f"  Mean: {mean(oh_fracs):.1f}%, Median: {median(oh_fracs):.1f}%")
        by_size_frac = defaultdict(list)
        for r in hq_rows:
            if r["avg_us"] > 0:
                by_size_frac[r["M"]].append(
                    r["host_overhead_us"] / r["avg_us"] * 100)
        for m in sorted(by_size_frac):
            vals = by_size_frac[m]
            print(f"    {m:>5d}: median={median(vals):.1f}%  mean={mean(vals):.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze performance measurement results."
    )
    parser.add_argument("--csv", required=True, help="Path to result CSV")
    parser.add_argument("--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    rows = load_data(args.csv)
    rows = [r for r in rows if r.get("status") == "PASS"]
    rows = preprocess(rows)

    if not rows:
        print("ERROR: No PASS rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    def _run_analysis(rows_):
        print("Performance Analysis Report")
        print(f"CSV: {args.csv}")
        print(f"Cases: {len(rows_)} (PASS only)")
        print()

        section1_execution_structure(rows_)
        section2_measurement_reliability(rows_)
        section3_perf_by_size(rows_)
        section4_spatial_scaling(rows_)
        section5_temporal_scaling(rows_)
        section6_data_reuse(rows_)
        section7_kernel_utilization(rows_)
        section8_optimal_config(rows_)
        section9_host_overhead(rows_)

    if args.output:
        old_stdout = sys.stdout
        try:
            with open(args.output, "w", encoding="utf-8") as out_file:
                sys.stdout = out_file
                _run_analysis(rows)
        finally:
            sys.stdout = old_stdout
        print(f"Output written to: {args.output}")
    else:
        _run_analysis(rows)


if __name__ == "__main__":
    main()
