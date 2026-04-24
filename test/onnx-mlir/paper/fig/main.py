#!/usr/bin/env python3
"""
main.py — Generate all paper figures and tables.

Usage:
    python main.py                          # Generate everything
    python main.py --only F5 F6 T8          # Generate specific items
    python main.py --cal path/to/calib.json # Custom calibration file
    python main.py --csv path/to/result.csv --json path/to/tc_list.json

Output goes to ./output/ (PDF + PNG for figures, .tex + .csv for tables).

† marks in T8 indicate configurations where the framework-chosen config
  was not found in measurement data; predicted values were used instead.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Generate only specified items (e.g., F5 F6 T8)")
    parser.add_argument("--cal", type=Path, default=None,
                        help="Path to calibration JSON")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Path to result CSV")
    parser.add_argument("--json", type=Path, default=None,
                        help="Path to tc_list JSON")
    parser.add_argument("--outdir", type=Path, default=Path("output"),
                        help="Output directory")
    args = parser.parse_args()

    # ── Import after argparse so --help is fast ──
    from config import load_config, DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON
    from data_loader import load_all
    from baselines import compute_all_baselines
    from figures import figure_f5, figure_f6, figure_f7, figure_f8
    from tables import table_t8, table_t9, table_t10, table_t11, table_t12

    # ── Config ──
    cal_path = args.cal or DEFAULT_CALIBRATION_PATH
    csv_path = args.csv or DEFAULT_RESULT_CSV
    json_path = args.json or DEFAULT_TC_LIST_JSON
    output_dir = args.outdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Calibration: {cal_path}")
    print(f"Result CSV:  {csv_path}")
    print(f"TC List:     {json_path}")
    print(f"Output:      {output_dir}")
    print()

    cfg = load_config(cal_path, csv_path, json_path)

    # ── Load data ──
    t0 = time.time()
    print("Loading data...")
    df, groups = load_all(cfg)
    print(f"  {len(df)} rows, {len(groups)} workloads ({time.time()-t0:.1f}s)")

    # ── Baselines ──
    t0 = time.time()
    print("Computing baselines (GT, Naive-Max, Framework)...")
    baselines = compute_all_baselines(groups, cfg)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # Print summary
    print("\n── Baseline Summary ──")
    n_charm_diff = 0   # count workloads where CHARM picked a different config
    n_timeloop_diff = 0  # count workloads where Timeloop picked ≠ Naive-Max
    n_timeloop_pmax = 0  # count workloads where Timeloop picked P_max
    for wl in cfg.WORKLOADS:
        bl = baselines.get(wl)
        if not bl:
            continue
        m, k, n = wl
        label = f"{m}×{k}×{n}"
        gt_p = bl.gt.P if bl.gt else "?"
        naive_edp = bl.naive_max.measured_edp if bl.naive_max and not np.isnan(bl.naive_max.measured_edp) else None
        gt_edp = bl.gt.measured_edp if bl.gt and not np.isnan(bl.gt.measured_edp) else None
        fw_in_meas = "✓" if bl.framework and bl.framework.in_measurements else "†"

        reduction = ""
        if gt_edp and naive_edp and naive_edp > 0:
            r = (1 - gt_edp / naive_edp) * 100
            reduction = f" ({r:+.0f}%)"

        fw_p = bl.framework.P if bl.framework else "?"
        charm_p = bl.charm_spk1.P if bl.charm_spk1 else "?"
        timeloop_p = bl.timeloop.P if bl.timeloop else "?"
        # Flag when CHARM picks a different point than Naive-Max (variability check)
        charm_vs_naive = ""
        if bl.charm_spk1 and bl.naive_max:
            same = (
                bl.charm_spk1.P == bl.naive_max.P
                and bl.charm_spk1.SPm == bl.naive_max.SPm
                and bl.charm_spk1.SPn == bl.naive_max.SPn
                and bl.charm_spk1.TPm == bl.naive_max.TPm
                and bl.charm_spk1.TPk == bl.naive_max.TPk
                and bl.charm_spk1.TPn == bl.naive_max.TPn
                and bl.charm_spk1.tpOrder_inner == bl.naive_max.tpOrder_inner
            )
            if not same:
                charm_vs_naive = " [CHARM≠Naive]"
                n_charm_diff += 1
        # Flag when Timeloop picks a different point than Naive-Max
        timeloop_vs_naive = ""
        if bl.timeloop and bl.naive_max:
            same = (
                bl.timeloop.P == bl.naive_max.P
                and bl.timeloop.SPm == bl.naive_max.SPm
                and bl.timeloop.SPn == bl.naive_max.SPn
                and bl.timeloop.TPm == bl.naive_max.TPm
                and bl.timeloop.TPk == bl.naive_max.TPk
                and bl.timeloop.TPn == bl.naive_max.TPn
                and bl.timeloop.tpOrder_inner == bl.naive_max.tpOrder_inner
            )
            if not same:
                timeloop_vs_naive = " [TL≠Naive]"
                n_timeloop_diff += 1
            if bl.timeloop.P == cfg.hw.P_total:
                n_timeloop_pmax += 1
        print(
            f"  {label:>20s}  GT:P={gt_p:<3} FW:P={fw_p:<3}{fw_in_meas}  "
            f"Naive:P={bl.naive_max.P if bl.naive_max else '?':<3}  "
            f"CHARM:P={charm_p:<3}{charm_vs_naive}  "
            f"TL:P={timeloop_p:<3}{timeloop_vs_naive}  EDP_red={reduction}"
        )
    print(f"\n  CHARM-CDSE picked a different config than Naive-Max in "
          f"{n_charm_diff}/{len(cfg.WORKLOADS)} workloads.")
    print(f"  Timeloop picked a different config than Naive-Max in "
          f"{n_timeloop_diff}/{len(cfg.WORKLOADS)} workloads "
          f"(Timeloop picked P={cfg.hw.P_total} in {n_timeloop_pmax}).")

    # ── Determine what to generate ──
    all_items = {"F5", "F6", "F7", "F8", "T8", "T9", "T10", "T11", "T12"}
    items = set(args.only) if args.only else all_items
    invalid = items - all_items
    if invalid:
        print(f"\nWarning: Unknown items {invalid}, skipping.")
        items -= invalid

    # ── Generate ──
    print(f"\n── Generating: {sorted(items)} ──")
    generated = []

    if "T8" in items:
        t0 = time.time()
        out = table_t8(groups, cfg, output_dir)
        print(f"  T8: {out} ({time.time()-t0:.1f}s)")
        generated.append(out)

    if "T9" in items:
        t0 = time.time()
        out = table_t9(df, cfg, output_dir)
        print(f"  T9: {out} ({time.time()-t0:.1f}s)")
        generated.append(out)

    if "T10" in items:
        t0 = time.time()
        out = table_t10(df, cfg, output_dir)
        print(f"  T10: {out} ({time.time()-t0:.1f}s)")
        generated.append(out)

    if "T11" in items:
        t0 = time.time()
        out = table_t11(baselines, cfg, output_dir)
        print(f"  T11: {out} ({time.time()-t0:.1f}s)")
        generated.append(out)

    if "T12" in items:
        t0 = time.time()
        out = table_t12(cfg, output_dir, groups)
        print(f"  T12: {out} ({time.time()-t0:.1f}s)")
        generated.append(out)

    if "F5" in items:
        t0 = time.time()
        out = figure_f5(df, cfg, output_dir)
        print(f"  F5: {out} ({time.time()-t0:.1f}s)")
        generated.append(out)

    if "F6" in items:
        t0 = time.time()
        out = figure_f6(groups, baselines, cfg, output_dir)
        print(f"  F6: {out} ({time.time()-t0:.1f}s)")
        generated.append(out)

    if "F7" in items:
        t0 = time.time()
        out = figure_f7(baselines, cfg, output_dir)
        print(f"  F7: {out} ({time.time()-t0:.1f}s)")
        generated.append(out)

    if "F8" in items:
        t0 = time.time()
        out = figure_f8(output_dir, cfg=cfg, groups=groups, baselines=baselines)
        print(f"  F8: {out} ({time.time()-t0:.1f}s)")
        generated.append(out)

    print(f"\n✅ Generated {len(generated)} items in {output_dir}/")


if __name__ == "__main__":
    main()
