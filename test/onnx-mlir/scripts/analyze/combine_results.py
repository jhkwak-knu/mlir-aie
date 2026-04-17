#!/usr/bin/env python3
"""Combine NPU and CPU measurement CSVs into a single canonical snapshot.

Preserves the 50-column schema used by run_tc_all.sh and
measure_cpu_energy.py. NPU rows keep their original case_index.
CPU rows are re-indexed to 10000+idx (1T) and 10100+idx (24T) to avoid
collision with NPU case_index. Provenance (source files, git commit,
CPU mode, date, caveats) is written as comment lines at the top.

NPU vs CPU distinction in the combined file follows the existing
convention: NPU rows have SPm>=1, CPU rows have SPm=-1. The updated
compare_npu_cpu.py filters by SPm to pick the right subset when the
combined file is passed as both --npu and --cpu.
"""

import argparse
import csv
import datetime
import subprocess
from pathlib import Path


def _read_csv_rows(path):
    """Return (rows, fieldnames). Skip comment lines and duplicate-header /
    short / non-PASS rows defensively."""
    with open(path) as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames
    rows = []
    for r in reader:
        # Drop the None-key entry caused by extra trailing commas.
        r.pop(None, None)
        # Skip duplicate-header rows (first cell equals "case_index").
        if r.get("case_index") == "case_index":
            continue
        # Skip rows missing required fields or not PASS.
        if r.get("status") != "PASS":
            continue
        try:
            int(r["case_index"]); int(r["M"]); int(r["K"]); int(r["N"])
        except (TypeError, ValueError, KeyError):
            continue
        # Ensure every canonical field is present (fill missing with -1).
        for k in fieldnames:
            if r.get(k) is None:
                r[k] = "-1"
        rows.append(r)
    return rows, fieldnames


def _git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npu", required=True, help="NPU result CSV")
    ap.add_argument("--cpu", required=True, help="CPU baseline CSV")
    ap.add_argument("--out", required=True, help="Combined output CSV")
    ap.add_argument("--cpu-mode-tag", default="upcast_fp32",
                    help="CPU measurement mode tag for provenance header")
    ap.add_argument("--caveats", default="",
                    help="Optional caveat note for provenance header")
    args = ap.parse_args()

    npu_rows, npu_cols = _read_csv_rows(args.npu)
    cpu_rows, cpu_cols = _read_csv_rows(args.cpu)

    # Dedupe NPU rows by case_index (keep LAST — retries override earlier).
    _seen = {}
    for r in npu_rows:
        _seen[r["case_index"]] = r
    npu_rows = list(_seen.values())

    # Sanity: same column schema
    if npu_cols != cpu_cols:
        raise SystemExit(
            f"Schema mismatch:\n  NPU cols: {npu_cols}\n  CPU cols: {cpu_cols}")

    # Re-index CPU rows (1T: 10001..., 24T: 10101...)
    cnt_1t = 0
    cnt_24t = 0
    for r in cpu_rows:
        th = int(r["numSpm"])
        if th == 1:
            cnt_1t += 1
            r["case_index"] = str(10000 + cnt_1t)
        elif th == 24:
            cnt_24t += 1
            r["case_index"] = str(10100 + cnt_24t)
        else:
            raise SystemExit(f"Unexpected CPU numSpm={th} in {args.cpu}")

    # Write combined CSV with provenance header
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    header_lines = [
        f"# combined_results snapshot",
        f"# source_npu: {args.npu}",
        f"# source_cpu: {args.cpu}",
        f"# npu_rows: {len(npu_rows)}",
        f"# cpu_rows: {len(cpu_rows)} (1T={cnt_1t}, 24T={cnt_24t})",
        f"# git_commit: {commit}",
        f"# cpu_mode: {args.cpu_mode_tag}",
        f"# combined_at: {now}",
        f"# convention: NPU rows SPm>=1, CPU rows SPm=-1; CPU numSpm encodes thread count",
        f"# case_index remap: CPU 1T -> 10001+, CPU 24T -> 10101+; NPU kept as source",
    ]
    if args.caveats:
        header_lines.append(f"# caveats: {args.caveats}")

    with open(out_path, "w", newline="") as f:
        for ln in header_lines:
            f.write(ln + "\n")
        writer = csv.DictWriter(f, fieldnames=npu_cols)
        writer.writeheader()
        for r in npu_rows:
            writer.writerow(r)
        for r in cpu_rows:
            writer.writerow(r)

    print(f"Wrote {out_path}")
    print(f"  NPU rows: {len(npu_rows)}")
    print(f"  CPU rows: {len(cpu_rows)} (1T={cnt_1t}, 24T={cnt_24t})")
    print(f"  git_commit: {commit}")


if __name__ == "__main__":
    main()
