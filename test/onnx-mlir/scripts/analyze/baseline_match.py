#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_match.py — Measurement-side baseline analysis over result.csv.

Ported from the measurement-matching half of paper/fig/baselines.py:
  - find_gt_from_csv(df, workload)          measurement argmin-EDP (GT)
  - find_pmax_gt_from_csv(df, workload, P)  GT constrained to P = P_max
  - match_predicted_to_measured(cases, df)  join a tc_list.json pick with
                                            the measured row for that config

The selection (algorithm) side of paper/fig's find_framework_optimal /
find_charm_cdse / find_timeloop lives in xdna_search searchers; THIS script
handles ONLY measured-data lookup, which is independent of which searcher
produced the candidate.

Typical usage in our pipeline:
  1. cost_model.py --search <X>  -> tc_list_<X>.json
  2. run_tc_all.sh tc_list_<X>.json -> result.csv
  3. baseline_match.py --csv result.csv --op op_list.json --out match.csv
     (produces GT / P_max-GT columns; optional --tc-list to match a pick)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


WorkloadKey = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# CSV loading — minimal pandas-free parser
# ---------------------------------------------------------------------------

def _coerce(val: str) -> Any:
    """Attempt int -> float -> str conversion for a CSV cell."""
    if val == "" or val is None:
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def load_result_csv(path: Path) -> List[Dict[str, Any]]:
    """Parse result.csv into a list of row dicts (no pandas dependency).

    Handles two header styles:
      - v11: header is the very first line; comments are data rows.
      - v12+: comments start with '#'; header is the first non-comment line.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    header_idx = 0
    for i, line in enumerate(raw_lines):
        if line.strip().startswith("case_index"):
            header_idx = i
            break
    header = [c.strip() for c in raw_lines[header_idx].strip().split(",")]
    n_cols = len(header)

    for line in raw_lines[header_idx + 1:]:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("case_index"):
            continue
        vals = s.split(",")[:n_cols]
        if len(vals) < n_cols:
            continue
        row = {h: _coerce(v) for h, v in zip(header, vals)}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Derived metrics: time_us, energy_uj, edp_measured
# ---------------------------------------------------------------------------

def _pick_time_us(row: Dict[str, Any]) -> Optional[float]:
    """Prefer batch_min_avg_us when available (matches paper/fig semantics)."""
    bm = row.get("batch_min_avg_us")
    if isinstance(bm, (int, float)) and bm > 0:
        return float(bm)
    mu = row.get("min_us")
    if isinstance(mu, (int, float)) and mu > 0:
        return float(mu)
    return None


def _pick_energy_uj(row: Dict[str, Any]) -> Optional[float]:
    bm = row.get("batch_min_energy_per_iter_uj")
    if isinstance(bm, (int, float)) and bm > 0:
        return float(bm)
    npu = row.get("npu_energy_per_iter_uj")
    if isinstance(npu, (int, float)) and npu > 0:
        return float(npu)
    return None


def annotate_metrics(rows: List[Dict[str, Any]]) -> None:
    """Add time_us / energy_uj / edp_measured keys in-place."""
    for r in rows:
        t = _pick_time_us(r)
        e = _pick_energy_uj(r)
        r["time_us"] = t
        r["energy_uj"] = e
        if t is not None and e is not None:
            r["edp_measured"] = t * e
        else:
            r["edp_measured"] = None


def group_by_workload(
    rows: List[Dict[str, Any]],
) -> Dict[WorkloadKey, List[Dict[str, Any]]]:
    groups: Dict[WorkloadKey, List[Dict[str, Any]]] = {}
    for r in rows:
        try:
            key = (int(r["M"]), int(r["K"]), int(r["N"]))
        except (KeyError, TypeError, ValueError):
            continue
        groups.setdefault(key, []).append(r)
    return groups


# ---------------------------------------------------------------------------
# Measurement-side baselines
# ---------------------------------------------------------------------------

_CONFIG_FIELDS = (
    "numSpm", "SPm", "SPn", "TPm", "TPk", "TPn", "tpOrder_inner",
)


def _row_config_tuple(row: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(int(row[k]) for k in _CONFIG_FIELDS)
    except (KeyError, TypeError, ValueError):
        return None


def find_gt_from_csv(
    workload_rows: List[Dict[str, Any]], workload: WorkloadKey,
) -> Optional[Dict[str, Any]]:
    """Measurement-side ground truth: argmin(measured EDP) over all rows.

    Returns a dict with config + measured values, or None when no row has
    valid (positive) measured EDP.
    """
    valid = [r for r in workload_rows if r.get("edp_measured") is not None]
    if not valid:
        return None
    best = min(valid, key=lambda r: r["edp_measured"])
    return _baseline_dict(best, workload, label="gt")


def find_pmax_gt_from_csv(
    workload_rows: List[Dict[str, Any]], workload: WorkloadKey,
    p_max: int = 32,
) -> Optional[Dict[str, Any]]:
    """P_max-GT: measurement argmin EDP among rows with numSpm == p_max."""
    valid = [
        r for r in workload_rows
        if r.get("edp_measured") is not None and r.get("numSpm") == p_max
    ]
    if not valid:
        return None
    best = min(valid, key=lambda r: r["edp_measured"])
    return _baseline_dict(best, workload, label="pmax_gt")


def _baseline_dict(
    row: Dict[str, Any], workload: WorkloadKey, *, label: str,
) -> Dict[str, Any]:
    M, K, N = workload
    return {
        "label": label,
        "M": M, "K": K, "N": N,
        "P": int(row.get("numSpm", 0)),
        "SPm": int(row.get("SPm", 0)),
        "SPn": int(row.get("SPn", 0)),
        "TPm": int(row.get("TPm", 0)),
        "TPk": int(row.get("TPk", 0)),
        "TPn": int(row.get("TPn", 0)),
        "tpOrder_inner": int(row.get("tpOrder_inner", 0)),
        "time_us": row.get("time_us"),
        "energy_uj": row.get("energy_uj"),
        "edp_measured": row.get("edp_measured"),
    }


def match_predicted_to_measured(
    cases: List[Dict[str, Any]],
    rows_by_workload: Dict[WorkloadKey, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Join each tc_list.json case with the measured row for its config.

    Returns one dict per case, carrying the original predicted fields plus
    `measured_time_us` / `measured_energy_uj` / `measured_edp` (None when
    the exact (P, SPm, SPn, TPm, TPk, TPn, tpOrder_inner) combination is
    absent from the measurement data — no fallback applied).
    """
    out: List[Dict[str, Any]] = []
    for case in cases:
        key: WorkloadKey = (int(case["M"]), int(case["K"]), int(case["N"]))
        level = case["levels"][0]
        want = (
            int(case["numCores"]),
            int(level["SPm"]), int(level["SPn"]),
            int(level["TPm"]), int(level["TPk"]), int(level["TPn"]),
            int(level["tpOrder"][0]),
        )
        wl_rows = rows_by_workload.get(key, [])
        hit = None
        for r in wl_rows:
            rc = _row_config_tuple(r)
            if rc == want:
                hit = r
                break
        enriched = dict(case)
        enriched["measured_time_us"] = hit.get("time_us") if hit else None
        enriched["measured_energy_uj"] = hit.get("energy_uj") if hit else None
        enriched["measured_edp"] = hit.get("edp_measured") if hit else None
        enriched["measured_match"] = hit is not None
        out.append(enriched)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_workloads_from_op_list(path: Path) -> List[WorkloadKey]:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    return [
        (int(c["M"]), int(c["K"]), int(c["N"]))
        for c in doc.get("cases", [])
    ]


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Compute measurement-side baselines (GT / P_max-GT) "
                    "and optionally match a tc_list.json pick to measured rows.",
    )
    p.add_argument("--csv", required=True, type=Path,
                   help="Path to result.csv from run_tc_all.sh")
    p.add_argument("--op", required=True, type=Path,
                   help="Path to op_list.json defining workloads of interest")
    p.add_argument("--out", required=True, type=Path,
                   help="Output CSV path (one row per workload per baseline)")
    p.add_argument("--p-max", type=int, default=32,
                   help="P value treated as the vendor/P_max baseline (default 32)")
    p.add_argument("--tc-list", type=Path, default=None,
                   help="Optional tc_list.json. When given, emits one extra row "
                        "per case with its measured match.")
    args = p.parse_args(argv)

    try:
        rows = load_result_csv(args.csv)
    except Exception as e:
        print(f"[ERROR] failed to read {args.csv}: {e}", file=sys.stderr)
        return 1
    annotate_metrics(rows)
    groups = group_by_workload(rows)

    workloads = _load_workloads_from_op_list(args.op)
    print(f"[INFO] {len(rows)} rows, {len(groups)} workloads in CSV; "
          f"{len(workloads)} workloads in op_list")

    output: List[Dict[str, Any]] = []
    for wl in workloads:
        wl_rows = groups.get(wl, [])
        if not wl_rows:
            print(f"[WARN] no rows for workload {wl}")
            continue
        gt = find_gt_from_csv(wl_rows, wl)
        pmax = find_pmax_gt_from_csv(wl_rows, wl, p_max=args.p_max)
        if gt:
            output.append(gt)
        if pmax:
            output.append(pmax)

    if args.tc_list:
        with args.tc_list.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        cases = doc.get("cases", [])
        matched = match_predicted_to_measured(cases, groups)
        for m in matched:
            # flatten the single-level SP/TP into the output row
            lvl = m["levels"][0]
            output.append({
                "label": "pick",
                "M": m["M"], "K": m["K"], "N": m["N"],
                "P": m["numCores"],
                "SPm": lvl["SPm"], "SPn": lvl["SPn"],
                "TPm": lvl["TPm"], "TPk": lvl["TPk"], "TPn": lvl["TPn"],
                "tpOrder_inner": lvl["tpOrder"][0],
                "time_us": m.get("measured_time_us"),
                "energy_uj": m.get("measured_energy_uj"),
                "edp_measured": m.get("measured_edp"),
                "measured_match": m.get("measured_match"),
            })

    _write_csv(output, args.out)
    print(f"[INFO] wrote {args.out} ({len(output)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
