#!/usr/bin/env python3
"""
Analyze cost model ranking accuracy by comparing predicted T_total
against actual NPU execution time using Spearman rank correlation.

Usage:
    python3 scripts/analyze_ranking.py --csv out/reports/result.csv
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple


def spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient between two lists."""
    n = len(x)
    if n < 2:
        return float("nan")

    def rank(vals: List[float]) -> List[float]:
        """Assign average ranks, handling ties."""
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)

    # Pearson correlation on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = sum((a - mean_rx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_ry) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze cost model ranking accuracy")
    parser.add_argument("--csv", required=True, help="Path to result CSV")
    parser.add_argument("--group-by-cores", action="store_true",
                        help="Also compute correlation per core count")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        return 1

    # Parse CSV: need t_total_pred (predicted) and avg_us (actual)
    rows: List[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("status", "")
            t_pred = row.get("t_total_pred", "-")
            avg_us = row.get("avg_us", "-1")

            if status != "PASS" or t_pred in ("-", "") or avg_us in ("-1", "-", ""):
                continue

            try:
                rows.append({
                    "case": int(row["case_index"]),
                    "cores": int(row["numSpm"]),
                    "SPm": int(row["SPm"]),
                    "SPn": int(row["SPn"]),
                    "TPm": int(row["TPm"]),
                    "TPk": int(row["TPk"]),
                    "TPn": int(row["TPn"]),
                    "t_pred": float(t_pred),
                    "avg_us": float(avg_us),
                })
            except (ValueError, KeyError):
                continue

    if len(rows) < 2:
        print(f"[ERROR] Need at least 2 PASS cases with predictions, got {len(rows)}")
        return 1

    # Sort by predicted time for display
    rows.sort(key=lambda r: r["t_pred"])

    # Overall Spearman correlation
    preds = [r["t_pred"] for r in rows]
    actuals = [r["avg_us"] for r in rows]
    rho = spearman_rank_correlation(preds, actuals)

    print(f"\n{'='*80}")
    print(f"  Cost Model Ranking Analysis: {len(rows)} candidates")
    print(f"{'='*80}")
    print(f"\n  Spearman rank correlation (rho): {rho:.4f}")
    print(f"  (1.0 = perfect agreement, -1.0 = completely inverted)")

    # Show top/bottom by predicted vs actual rank
    # Assign actual ranks
    actual_sorted = sorted(range(len(rows)), key=lambda i: actuals[i])
    actual_rank = [0] * len(rows)
    for rank_pos, idx in enumerate(actual_sorted):
        actual_rank[idx] = rank_pos + 1

    pred_sorted = sorted(range(len(rows)), key=lambda i: preds[i])
    pred_rank = [0] * len(rows)
    for rank_pos, idx in enumerate(pred_sorted):
        pred_rank[idx] = rank_pos + 1

    print(f"\n  {'#':>3s}  {'cores':>5s}  {'SP':>7s}  {'TP':>11s}  "
          f"{'T_pred':>10s}  {'T_actual':>10s}  {'pred_rank':>9s}  {'act_rank':>8s}  {'diff':>5s}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*7}  {'-'*11}  "
          f"{'-'*10}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*5}")

    for i, r in enumerate(rows):
        pidx = preds.index(r["t_pred"])
        pr = pred_rank[pidx]
        ar = actual_rank[pidx]
        diff = pr - ar
        print(f"  {i+1:>3d}  {r['cores']:>5d}  ({r['SPm']:>2d},{r['SPn']:>2d})  "
              f"({r['TPm']:>2d},{r['TPk']:>2d},{r['TPn']:>2d})  "
              f"{r['t_pred']:>10.1f}  {r['avg_us']:>10.1f}  "
              f"{pr:>9d}  {ar:>8d}  {diff:>+5d}")

    # Actual top 5 vs predicted top 5
    actual_top5_idx = actual_sorted[:min(5, len(rows))]
    pred_top5_idx = pred_sorted[:min(5, len(rows))]

    print(f"\n  Predicted top 5 (fastest):")
    for rank_pos, idx in enumerate(pred_top5_idx):
        r = rows[idx]
        print(f"    #{rank_pos+1}  cores={r['cores']} SP=({r['SPm']},{r['SPn']}) "
              f"TP=({r['TPm']},{r['TPk']},{r['TPn']}) "
              f"T_pred={r['t_pred']:.1f}  T_actual={r['avg_us']:.1f}us "
              f"(actual_rank={actual_rank[idx]})")

    print(f"\n  Actual top 5 (fastest):")
    for rank_pos, idx in enumerate(actual_top5_idx):
        r = rows[idx]
        print(f"    #{rank_pos+1}  cores={r['cores']} SP=({r['SPm']},{r['SPn']}) "
              f"TP=({r['TPm']},{r['TPk']},{r['TPn']}) "
              f"T_actual={r['avg_us']:.1f}us  T_pred={r['t_pred']:.1f} "
              f"(pred_rank={pred_rank[idx]})")

    # Per core-count analysis
    if args.group_by_cores:
        core_counts = sorted(set(r["cores"] for r in rows))
        print(f"\n  Per core-count Spearman correlation:")
        print(f"  {'cores':>5s}  {'n':>4s}  {'rho':>8s}")
        for nc in core_counts:
            sub = [r for r in rows if r["cores"] == nc]
            if len(sub) < 2:
                print(f"  {nc:>5d}  {len(sub):>4d}  {'n/a':>8s}")
                continue
            sub_preds = [r["t_pred"] for r in sub]
            sub_acts = [r["avg_us"] for r in sub]
            sub_rho = spearman_rank_correlation(sub_preds, sub_acts)
            print(f"  {nc:>5d}  {len(sub):>4d}  {sub_rho:>8.4f}")

    print(f"\n{'='*80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
