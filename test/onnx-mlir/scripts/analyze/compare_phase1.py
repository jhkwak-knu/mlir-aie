#!/usr/bin/env python3
"""Compare Phase 1 (post-optimization) vs pre-Phase 1 measurement data.

Analyzes performance changes from chess pragmas, pres optimization,
and step_time measurement additions.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_csv(path):
    """Load CSV and return list of dicts, filtering to PASS only."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "PASS":
                rows.append(row)
    return rows


def config_key(row):
    """Create a unique config key from tiling parameters."""
    return (
        int(row["SPm"]), int(row["SPn"]),
        int(row["TPm"]), int(row["TPk"]), int(row["TPn"]),
        int(row["TM"]), int(row["TK"]), int(row["TN"]),
        int(row["M"]), int(row["K"]), int(row["N"]),
    )


def size_key(row):
    return f"{row['M']}x{row['K']}x{row['N']}"


def load_tporder(tc_path):
    """Load tpOrder from tc_list.json, keyed by config."""
    with open(tc_path) as f:
        data = json.load(f)
    cases = data["cases"] if isinstance(data, dict) else data
    result = {}
    for c in cases:
        lev = c["levels"][0]
        key = (
            lev["SPm"], lev["SPn"],
            lev["TPm"], lev["TPk"], lev["TPn"],
            lev["TM"], lev["TK"], lev["TN"],
            c["M"], c["K"], c["N"],
        )
        result[key] = lev.get("tpOrder", [2, 0, 1])
    return result


def percentile(values, pct):
    """Simple percentile without numpy."""
    s = sorted(values)
    idx = int(len(s) * pct / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def stdev(values):
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description="Compare Phase 1 vs pre-Phase 1")
    ap.add_argument("--old", required=True, help="Pre-Phase 1 result CSV")
    ap.add_argument("--new", required=True, help="Phase 1 result CSV")
    ap.add_argument("--tc", required=True, help="tc_list.json with tpOrder")
    args = ap.parse_args()

    old_rows = load_csv(args.old)
    new_rows = load_csv(args.new)
    tporders = load_tporder(args.tc)

    # Index by config key
    old_by_key = {}
    for r in old_rows:
        k = config_key(r)
        old_min = float(r["min_us"])
        if k not in old_by_key or old_min < old_by_key[k]["min_us"]:
            old_by_key[k] = {"min_us": old_min, "row": r}

    new_by_key = {}
    for r in new_rows:
        k = config_key(r)
        new_min = float(r["min_us"])
        if k not in new_by_key or new_min < new_by_key[k]["min_us"]:
            new_by_key[k] = {"min_us": new_min, "row": r}

    # Common configs
    common_keys = set(old_by_key.keys()) & set(new_by_key.keys())
    print(f"{'=' * 70}")
    print(f"Phase 1 Before/After Comparison")
    print(f"{'=' * 70}")
    print(f"Old PASS configs: {len(old_by_key)}")
    print(f"New PASS configs: {len(new_by_key)}")
    print(f"Common configs:   {len(common_keys)}")
    print()

    # ===== 1-A: Overall performance change =====
    print(f"{'=' * 70}")
    print("1-A. Overall Performance Change (min_us)")
    print(f"{'=' * 70}")

    ratios = []  # new/old ratio (<1 = faster)
    speedups = []  # old/new (>1 = faster)
    diffs = []  # old - new (positive = faster)

    for k in common_keys:
        old_t = old_by_key[k]["min_us"]
        new_t = new_by_key[k]["min_us"]
        if old_t > 0:
            ratios.append(new_t / old_t)
            speedups.append(old_t / new_t)
            diffs.append(old_t - new_t)

    improved = sum(1 for r in ratios if r < 0.99)
    degraded = sum(1 for r in ratios if r > 1.01)
    unchanged = len(ratios) - improved - degraded

    print(f"\nConfigs improved (>1% faster):  {improved}/{len(ratios)} ({100*improved/len(ratios):.1f}%)")
    print(f"Configs degraded (>1% slower):  {degraded}/{len(ratios)} ({100*degraded/len(ratios):.1f}%)")
    print(f"Configs unchanged (+/-1%):      {unchanged}/{len(ratios)} ({100*unchanged/len(ratios):.1f}%)")

    print(f"\nSpeedup (old/new) statistics:")
    print(f"  Mean:   {sum(speedups)/len(speedups):.4f}x")
    print(f"  Median: {median(speedups):.4f}x")
    print(f"  Min:    {min(speedups):.4f}x (worst degradation)")
    print(f"  Max:    {max(speedups):.4f}x (best improvement)")
    print(f"  P10:    {percentile(speedups, 10):.4f}x")
    print(f"  P90:    {percentile(speedups, 90):.4f}x")
    print(f"  Stdev:  {stdev(speedups):.4f}")

    # ===== 1-A2: By matrix size =====
    print(f"\n{'=' * 70}")
    print("1-A2. Speedup by Matrix Size")
    print(f"{'=' * 70}")

    by_size = defaultdict(list)
    for k in common_keys:
        old_t = old_by_key[k]["min_us"]
        new_t = new_by_key[k]["min_us"]
        row = new_by_key[k]["row"]
        sz = size_key(row)
        if old_t > 0:
            by_size[sz].append(old_t / new_t)

    # Sort by M*K*N
    def sort_size(sz):
        parts = sz.split("x")
        return int(parts[0]) * int(parts[1]) * int(parts[2])

    print(f"\n{'Size':<20} {'N':>5} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print("-" * 65)
    for sz in sorted(by_size.keys(), key=sort_size):
        vals = by_size[sz]
        print(f"{sz:<20} {len(vals):>5} {sum(vals)/len(vals):>8.4f} {median(vals):>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}")

    # ===== 1-A3: By core count =====
    print(f"\n{'=' * 70}")
    print("1-A3. Speedup by Core Count")
    print(f"{'=' * 70}")

    by_cores = defaultdict(list)
    for k in common_keys:
        old_t = old_by_key[k]["min_us"]
        new_t = new_by_key[k]["min_us"]
        row = new_by_key[k]["row"]
        cores = int(row["numSpm"])
        if old_t > 0:
            by_cores[cores].append(old_t / new_t)

    print(f"\n{'Cores':>6} {'N':>5} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    for c in sorted(by_cores.keys()):
        vals = by_cores[c]
        print(f"{c:>6} {len(vals):>5} {sum(vals)/len(vals):>8.4f} {median(vals):>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}")

    # ===== 1-B: Energy change =====
    print(f"\n{'=' * 70}")
    print("1-B. Energy Change (npu_energy_per_iter_uj)")
    print(f"{'=' * 70}")

    energy_ratios = []
    energy_by_size = defaultdict(list)

    for k in common_keys:
        old_row = old_by_key[k]["row"]
        new_row = new_by_key[k]["row"]
        old_e = float(old_row.get("npu_energy_per_iter_uj", -1))
        new_e = float(new_row.get("npu_energy_per_iter_uj", -1))
        if old_e > 0 and new_e > 0:
            ratio = new_e / old_e
            energy_ratios.append(ratio)
            sz = size_key(new_row)
            energy_by_size[sz].append(ratio)

    if energy_ratios:
        e_improved = sum(1 for r in energy_ratios if r < 0.99)
        e_degraded = sum(1 for r in energy_ratios if r > 1.01)
        print(f"\nValid energy comparisons: {len(energy_ratios)}")
        print(f"Energy reduced (>1% less): {e_improved}/{len(energy_ratios)} ({100*e_improved/len(energy_ratios):.1f}%)")
        print(f"Energy increased (>1% more): {e_degraded}/{len(energy_ratios)} ({100*e_degraded/len(energy_ratios):.1f}%)")
        print(f"\nEnergy ratio (new/old) -- <1 = less energy:")
        print(f"  Mean:   {sum(energy_ratios)/len(energy_ratios):.4f}")
        print(f"  Median: {median(energy_ratios):.4f}")

        print(f"\n{'Size':<20} {'N':>5} {'Mean':>8} {'Median':>8}")
        print("-" * 45)
        for sz in sorted(energy_by_size.keys(), key=sort_size):
            vals = energy_by_size[sz]
            print(f"{sz:<20} {len(vals):>5} {sum(vals)/len(vals):>8.4f} {median(vals):>8.4f}")

    # ===== 1-B2: EDP comparison (new data has step_min_us) =====
    print(f"\n{'=' * 70}")
    print("1-B2. EDP Change (min_us * energy_per_iter)")
    print(f"{'=' * 70}")

    edp_ratios = []
    for k in common_keys:
        old_row = old_by_key[k]["row"]
        new_row = new_by_key[k]["row"]
        old_e = float(old_row.get("npu_energy_per_iter_uj", -1))
        new_e = float(new_row.get("npu_energy_per_iter_uj", -1))
        old_t = old_by_key[k]["min_us"]
        new_t = new_by_key[k]["min_us"]
        if old_e > 0 and new_e > 0 and old_t > 0 and new_t > 0:
            old_edp = old_t * old_e
            new_edp = new_t * new_e
            edp_ratios.append(new_edp / old_edp)

    if edp_ratios:
        edp_improved = sum(1 for r in edp_ratios if r < 0.99)
        print(f"\nEDP reduced: {edp_improved}/{len(edp_ratios)} ({100*edp_improved/len(edp_ratios):.1f}%)")
        print(f"EDP ratio (new/old) -- <1 = better:")
        print(f"  Mean:   {sum(edp_ratios)/len(edp_ratios):.4f}")
        print(f"  Median: {median(edp_ratios):.4f}")

    # ===== 1-C: Detailed analysis =====
    print(f"\n{'=' * 70}")
    print("1-C1. Top 10 Most Improved Configs")
    print(f"{'=' * 70}")

    paired = []
    for k in common_keys:
        old_t = old_by_key[k]["min_us"]
        new_t = new_by_key[k]["min_us"]
        row = new_by_key[k]["row"]
        if old_t > 0:
            paired.append({
                "key": k, "old": old_t, "new": new_t,
                "speedup": old_t / new_t, "row": row,
            })

    paired.sort(key=lambda x: x["speedup"], reverse=True)
    print(f"\n{'#':>3} {'Size':<20} {'Cores':>5} {'SP':>6} {'Old_us':>12} {'New_us':>12} {'Speedup':>8}")
    print("-" * 75)
    for i, p in enumerate(paired[:10]):
        r = p["row"]
        sp = f"{r['SPm']}x{r['SPn']}"
        print(f"{i+1:>3} {size_key(r):<20} {r['numSpm']:>5} {sp:>6} {p['old']:>12.1f} {p['new']:>12.1f} {p['speedup']:>8.4f}x")

    print(f"\n{'=' * 70}")
    print("1-C2. Top 10 Most Degraded Configs")
    print(f"{'=' * 70}")
    paired.sort(key=lambda x: x["speedup"])
    print(f"\n{'#':>3} {'Size':<20} {'Cores':>5} {'SP':>6} {'Old_us':>12} {'New_us':>12} {'Speedup':>8}")
    print("-" * 75)
    for i, p in enumerate(paired[:10]):
        r = p["row"]
        sp = f"{r['SPm']}x{r['SPn']}"
        print(f"{i+1:>3} {size_key(r):<20} {r['numSpm']:>5} {sp:>6} {p['old']:>12.1f} {p['new']:>12.1f} {p['speedup']:>8.4f}x")

    # ===== 1-C3: By tpOrder =====
    print(f"\n{'=' * 70}")
    print("1-C3. Speedup by tpOrder (innermost axis)")
    print(f"{'=' * 70}")

    axis_names = {0: "M-inner", 1: "N-inner", 2: "K-inner"}
    by_tporder = defaultdict(list)
    for k in common_keys:
        old_t = old_by_key[k]["min_us"]
        new_t = new_by_key[k]["min_us"]
        tp = tporders.get(k)
        if tp and old_t > 0:
            inner = tp[0]
            by_tporder[inner].append(old_t / new_t)

    print(f"\n{'tpOrder[0]':<12} {'N':>5} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    for axis in sorted(by_tporder.keys()):
        vals = by_tporder[axis]
        name = axis_names.get(axis, f"axis{axis}")
        print(f"{name:<12} {len(vals):>5} {sum(vals)/len(vals):>8.4f} {median(vals):>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}")

    # ===== 1-C4: By pres requirement =====
    print(f"\n{'=' * 70}")
    print("1-C4. Speedup by PRES Requirement")
    print(f"{'=' * 70}")
    print("  (pres needed when tpOrder[0] != 2 AND TPk > 1)")

    by_pres = {"needs_pres": [], "no_pres": []}
    for k in common_keys:
        old_t = old_by_key[k]["min_us"]
        new_t = new_by_key[k]["min_us"]
        tp = tporders.get(k)
        row = new_by_key[k]["row"]
        tpk = int(row["TPk"])
        if tp and old_t > 0:
            needs_pres = tp[0] != 2 and tpk > 1
            cat = "needs_pres" if needs_pres else "no_pres"
            by_pres[cat].append(old_t / new_t)

    print(f"\n{'Category':<15} {'N':>5} {'Mean':>8} {'Median':>8}")
    print("-" * 40)
    for cat in ["needs_pres", "no_pres"]:
        vals = by_pres[cat]
        if vals:
            print(f"{cat:<15} {len(vals):>5} {sum(vals)/len(vals):>8.4f} {median(vals):>8.4f}")

    # ===== 1-C5: New step_min_us analysis =====
    print(f"\n{'=' * 70}")
    print("1-C5. step_min_us vs min_us (New Data Only)")
    print(f"{'=' * 70}")
    print("  step_min_us includes memcpy+sync overhead, min_us is dispatch-only")

    overhead_ratios = []
    overhead_by_size = defaultdict(list)
    for k in new_by_key:
        row = new_by_key[k]["row"]
        min_us = float(row.get("min_us", 0))
        step_min = float(row.get("step_min_us", 0))
        if min_us > 0 and step_min > 0:
            oh = (step_min - min_us) / min_us
            overhead_ratios.append(oh)
            overhead_by_size[size_key(row)].append(oh)

    if overhead_ratios:
        print(f"\nHost overhead ratio (step_min - min) / min:")
        print(f"  Mean:   {100*sum(overhead_ratios)/len(overhead_ratios):.2f}%")
        print(f"  Median: {100*median(overhead_ratios):.2f}%")
        print(f"  Max:    {100*max(overhead_ratios):.2f}%")

        print(f"\n{'Size':<20} {'N':>5} {'Mean%':>8} {'Median%':>8}")
        print("-" * 45)
        for sz in sorted(overhead_by_size.keys(), key=sort_size):
            vals = overhead_by_size[sz]
            print(f"{sz:<20} {len(vals):>5} {100*sum(vals)/len(vals):>8.2f} {100*median(vals):>8.2f}")

    # ===== Summary =====
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Common configs compared: {len(common_keys)}")
    print(f"Overall speedup: {sum(speedups)/len(speedups):.4f}x (mean), {median(speedups):.4f}x (median)")
    if energy_ratios:
        print(f"Energy change: {sum(energy_ratios)/len(energy_ratios):.4f}x (mean, <1=better)")
    if edp_ratios:
        print(f"EDP change: {sum(edp_ratios)/len(edp_ratios):.4f}x (mean, <1=better)")


if __name__ == "__main__":
    main()
