#!/usr/bin/env python3
"""Batch re-analyze archived trace data with improved PRES separation and ipd=1 support.

Reads archived case directories (each containing trace.json + tc.json),
runs the updated analyze_trace pipeline, and produces result_v3.csv with
additional quality columns.

Usage:
    python3 reanalyze_batch.py \
        --archive out/calibration/archive_v2 \
        --output out/calibration/result_v3.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Allow importing sibling modules when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_trace import (
    parse_trace_json,
    load_tc_json,
    detect_all_transfers,
    apply_timestamp_corrections,
    build_dispatch_template,
    validate_tiles,
    compute_combined_res,
    compute_summary_dict,
    XDNA2_DEFAULT_CLOCK_MHZ,
)
from predict_trace_events import (
    compare_predicted_vs_actual,
    compute_data_quality,
)


# ---------------------------------------------------------------------------
# CSV columns: tiling config + cost model + benchmark + energy + trace + derived
# ---------------------------------------------------------------------------
TILING_COLUMNS = [
    "case_index", "numSpm", "SPm", "SPn", "TPm", "TPk", "TPn",
    "TM", "TK", "TN", "M", "K", "N", "doubleBuffer", "t_total_pred",
]

BENCHMARK_COLUMNS = [
    "status", "errors", "iters", "warmup",
    "avg_us", "min_us", "max_us",
]

ENERGY_COLUMNS = [
    "idle_pkg_mw", "active_pkg_mw", "npu_power_mw",
    "npu_energy_uj", "npu_energy_per_iter_uj", "wall_elapsed_s",
]

TRACE_COLUMNS = [
    "dispatch_cy", "dispatch_us", "kernel_total_cy", "kernel_pct",
    "ss_iter_cy", "ss_iter_us", "ss_kernel_cy", "ss_kernel_util_pct",
    "ss_method", "gflops", "n_tiles", "n_valid_dispatches",
    "host_steps", "matmul_npu_us", "matmul_npu_cy",
]

QUALITY_COLUMNS = [
    "pres_active", "data_quality", "missing_events",
]

DERIVED_COLUMNS = [
    "host_overhead_us",
]


def find_case_dirs(archive_root: Path) -> list[tuple[int, Path]]:
    """Find case_NNN directories and return sorted (index, path) pairs."""
    cases = []
    for entry in archive_root.iterdir():
        if entry.is_dir() and entry.name.startswith("case_"):
            try:
                idx = int(entry.name.split("_")[1])
                cases.append((idx, entry))
            except (IndexError, ValueError):
                continue
    return sorted(cases, key=lambda x: x[0])


def load_original_result(case_dir: Path) -> dict:
    """Load original result row from case directory (result.json or tc.json metadata).

    Maps result.json key 'iterations' to CSV column name 'iters'.
    """
    result_path = case_dir / "result.json"
    if not result_path.is_file():
        return {}
    with open(result_path) as f:
        data = json.load(f)
    # result.json uses 'iterations'; CSV column is 'iters'
    if "iterations" in data and "iters" not in data:
        data["iters"] = data.pop("iterations")
    return data


def analyze_case(case_dir: Path, n_measured: int = 10,
                 clock_mhz: float = XDNA2_DEFAULT_CLOCK_MHZ) -> dict:
    """Analyze a single archived case directory.

    Returns a dict with all summary metrics + quality info, or None on failure.
    """
    trace_path = case_dir / "trace.json"
    tc_path = case_dir / "tc.json"

    if not trace_path.is_file() or not tc_path.is_file():
        return {"error": f"Missing trace.json or tc.json in {case_dir}"}

    tc = load_tc_json(str(tc_path))
    tiles = parse_trace_json(str(trace_path))

    if not tiles:
        return {"error": f"No tile data in {trace_path}"}

    # Run improved analysis pipeline
    detect_all_transfers(tiles, tc)

    # Calibrate cross-tile timestamps using Start timer_values
    hw_timers = []
    trace_raw_path = case_dir / "trace_raw.txt"
    if trace_raw_path.is_file():
        hw_timers = apply_timestamp_corrections(tiles, str(trace_raw_path))

    template = build_dispatch_template(tc)

    lvl = tc["levels"][0]
    inner_axis = lvl["tpOrder"][0]
    tp_map = {0: lvl["TPm"], 1: lvl["TPn"], 2: lvl["TPk"]}
    iters_per_dispatch = tp_map[inner_axis]

    # Adaptive n_dispatches: capture all dispatches from n_measured iterations
    host_steps = tp_map[lvl["tpOrder"][1]] * tp_map[lvl["tpOrder"][2]]
    n_dispatches = n_measured * host_steps

    tile_validations = validate_tiles(tiles, template, n_dispatches)

    # Filter to active tiles
    active_set = {tv.tile_name for tv in tile_validations if tv.has_data}
    active_tiles = [td for td in tiles if td.tile_name in active_set]
    active_validations = [tv for tv in tile_validations if tv.has_data]

    if not active_tiles:
        return {"error": f"No active tiles in {trace_path}"}

    combined_res = compute_combined_res(active_tiles)

    # Compute summary (with iteration-aware matmul_npu_us)
    summary = compute_summary_dict(
        active_tiles, combined_res, tc,
        iters_per_dispatch, n_dispatches, clock_mhz,
        template, active_validations, hw_timers)

    # Event prediction comparison (uses n_measured for expected total events)
    event_report = compare_predicted_vs_actual(tc, tiles, n_measured)
    quality = compute_data_quality(event_report)
    missing = sum(max(0, g) for r in event_report for g in r["gaps"].values())

    summary["data_quality"] = round(quality, 4)
    summary["missing_events"] = missing

    return summary


def build_csv_row(case_idx: int, tc: dict, summary: dict,
                  original: dict) -> dict:
    """Build a CSV row combining tiling config, benchmark, energy, trace, and derived."""
    lvl = tc["levels"][0]

    # Tiling config from tc.json
    row = {
        "case_index": case_idx,
        "numSpm": lvl["SPm"] * lvl["SPn"],
        "SPm": lvl["SPm"],
        "SPn": lvl["SPn"],
        "TPm": lvl["TPm"],
        "TPk": lvl["TPk"],
        "TPn": lvl["TPn"],
        "TM": lvl["TM"],
        "TK": lvl["TK"],
        "TN": lvl["TN"],
        "M": tc["M"],
        "K": tc["K"],
        "N": tc["N"],
        "doubleBuffer": lvl.get("doubleBuffer", False),
        "t_total_pred": tc.get("t_total_pred", ""),
    }

    # Benchmark columns from result.json
    for col in BENCHMARK_COLUMNS:
        row[col] = original.get(col, "")

    # Energy columns from result.json
    for col in ENERGY_COLUMNS:
        row[col] = original.get(col, "")

    # Trace analysis + quality columns
    trace_and_quality = TRACE_COLUMNS + QUALITY_COLUMNS
    if "error" in summary:
        row["status"] = f"trace_error: {summary['error']}"
        for col in trace_and_quality:
            if col not in row:
                row[col] = ""
    else:
        for col in trace_and_quality:
            row[col] = summary.get(col, "")

    # Derived: host_overhead_us = avg_us - matmul_npu_us
    try:
        avg = float(original.get("avg_us", -1))
        npu = float(summary.get("matmul_npu_us", -1))
        if avg > 0 and npu > 0:
            row["host_overhead_us"] = round(avg - npu, 2)
        else:
            row["host_overhead_us"] = ""
    except (ValueError, TypeError):
        row["host_overhead_us"] = ""

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Batch re-analyze archived traces with PRES split + ipd=1 support")
    parser.add_argument("--archive", required=True,
                        help="Path to archive directory (e.g., out/calibration/archive_v2)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (e.g., out/calibration/result_v3.csv)")
    parser.add_argument("--n-measured", type=int, default=10,
                        help="Number of measured iterations per case (default: 10)")
    parser.add_argument("--clock-mhz", type=float,
                        default=XDNA2_DEFAULT_CLOCK_MHZ,
                        help=f"NPU clock MHz (default: {XDNA2_DEFAULT_CLOCK_MHZ})")
    parser.add_argument("--json-dir", default=None,
                        help="Directory to write per-case trace_summary_v3.json")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_dir():
        print(f"ERROR: Archive directory not found: {archive}", file=sys.stderr)
        sys.exit(1)

    cases = find_case_dirs(archive)
    if not cases:
        print(f"ERROR: No case_NNN directories in {archive}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(cases)} cases in {archive}", file=sys.stderr)

    # Process each case
    all_columns = (TILING_COLUMNS + BENCHMARK_COLUMNS + ENERGY_COLUMNS
                    + TRACE_COLUMNS + QUALITY_COLUMNS + DERIVED_COLUMNS)
    rows = []
    n_ok = 0
    n_error = 0
    n_neg_ss = 0
    n_zero_ss = 0

    for case_idx, case_dir in cases:
        tc_path = case_dir / "tc.json"
        if not tc_path.is_file():
            print(f"  case_{case_idx:03d}: SKIP (no tc.json)", file=sys.stderr)
            n_error += 1
            continue

        tc = load_tc_json(str(tc_path))
        original = load_original_result(case_dir)
        summary = analyze_case(case_dir, args.n_measured, args.clock_mhz)

        row = build_csv_row(case_idx, tc, summary, original)
        rows.append(row)

        # Track quality
        ss_iter = summary.get("ss_iter_cy", 0)
        if "error" in summary:
            n_error += 1
            status = "ERROR"
        elif ss_iter < 0:
            n_neg_ss += 1
            status = "NEG_SS"
        elif ss_iter == 0:
            n_zero_ss += 1
            status = "ZERO_SS"
        else:
            n_ok += 1
            status = "OK"

        method = summary.get("ss_method", "N/A")
        pres = summary.get("pres_active", False)
        quality = summary.get("data_quality", 0)
        print(f"  case_{case_idx:03d}: {status:8s} ss_iter={ss_iter:>10} "
              f"method={method:16s} pres={str(pres):5s} quality={quality:.1%}",
              file=sys.stderr)

        # Write per-case JSON summary if requested
        if args.json_dir and "error" not in summary:
            json_dir = Path(args.json_dir)
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / f"case_{case_idx:03d}_v3.json"
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"\nResults written to {out_path}", file=sys.stderr)
    print(f"  Total: {len(rows)}", file=sys.stderr)
    print(f"  OK: {n_ok}  Negative ss_iter: {n_neg_ss}  "
          f"Zero ss_iter: {n_zero_ss}  Errors: {n_error}", file=sys.stderr)


if __name__ == "__main__":
    main()
