#!/usr/bin/env python3
"""Filter original result CSV to produce a clean subset for re-calibration.

Exclusion criteria (a case is EXCLUDED if ANY applies):
  1. Time anomaly: remeasurement exists AND is >=10% faster than original
     (original was a time outlier; keep original value but drop the case).
  2. Energy anomaly: idle_pkg_mw >= 3000 or missing (RAPL noise too high).

Both checks apply independently: a case must pass BOTH to be kept.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


ANOM_TIME_DELTA = -0.10  # remeasure <= -10% vs original
ANOM_IDLE_THRESHOLD_MW = 3000.0


def _iter_data_rows(path: Path):
    """Yield DictReader rows, skipping metadata lines (starting with #)."""
    with open(path) as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        yield row, reader.fieldnames


def _make_key(row):
    return tuple(row[c] for c in (
        "M", "K", "N", "numSpm", "SPm", "SPn",
        "TPm", "TPk", "TPn", "TM", "TK", "TN"))


def build_remeasure_map(remeas_csv: Path):
    m = {}
    for row, _ in _iter_data_rows(remeas_csv):
        if row.get("status") != "PASS":
            continue
        t = float(row.get("batch_min_avg_us", -1))
        if t > 0:
            m[_make_key(row)] = t
    return m


def classify(orig_row, remeas_map):
    """Return (is_time_anom, is_energy_anom)."""
    idle = float(orig_row.get("idle_pkg_mw", -1))
    t_orig = float(orig_row.get("batch_min_avg_us", -1))

    is_time_anom = False
    key = _make_key(orig_row)
    if key in remeas_map and t_orig > 0:
        t_new = remeas_map[key]
        if t_new > 0:
            delta = (t_new - t_orig) / t_orig
            if delta <= ANOM_TIME_DELTA:
                is_time_anom = True

    is_energy_anom = (idle <= 0) or (idle >= ANOM_IDLE_THRESHOLD_MW)
    return is_time_anom, is_energy_anom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, type=Path)
    ap.add_argument("--remeas", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-time", type=Path, default=None,
                    help="Separate output with time filter only "
                         "(energy-anomaly kept).")
    args = ap.parse_args()

    remeas_map = build_remeasure_map(args.remeas)
    print(f"Remeasure cases indexed: {len(remeas_map)}")

    # Load original, filter, write
    rows_all = []
    fieldnames = None
    for row, fns in _iter_data_rows(args.orig):
        rows_all.append(row)
        if fieldnames is None:
            fieldnames = fns

    pass_rows = [r for r in rows_all if r.get("status") == "PASS"]
    print(f"Original PASS rows: {len(pass_rows)}")

    clean, t_anom, e_anom, both = [], 0, 0, 0
    time_clean_only = []
    for row in pass_rows:
        ta, ea = classify(row, remeas_map)
        if ta and ea:
            both += 1
        elif ta:
            t_anom += 1
        elif ea:
            e_anom += 1
        else:
            clean.append(row)
        if not ta:
            time_clean_only.append(row)

    print(f"  Time-only anomaly: {t_anom}")
    print(f"  Energy-only anomaly: {e_anom}")
    print(f"  Both: {both}")
    print(f"  Clean (kept): {len(clean)}")

    # Derive wall_elapsed_s in batch mode from n_batches*n_inner*step_time.
    # calibrate_energy.py requires wall_elapsed_s for window-filtering.
    def derive_wall(row):
        try:
            nb = int(row.get("n_batches", "0"))
            ni = int(row.get("n_inner", "0"))
            t_us = float(row.get("batch_min_avg_us", "-1"))
            if nb > 0 and ni > 0 and t_us > 0:
                row["wall_elapsed_s"] = f"{nb * ni * t_us / 1e6:.6f}"
        except (ValueError, TypeError):
            pass
        return row

    clean = [derive_wall(r) for r in clean]
    time_clean_only = [derive_wall(r) for r in time_clean_only]

    # Write clean subset (only standard columns, drop padded -1's)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(clean)
    print(f"\nWrote {len(clean)} rows -> {args.out}")

    if args.out_time is not None:
        with open(args.out_time, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(time_clean_only)
        print(f"Wrote {len(time_clean_only)} rows (time filter only) "
              f"-> {args.out_time}")


if __name__ == "__main__":
    main()
