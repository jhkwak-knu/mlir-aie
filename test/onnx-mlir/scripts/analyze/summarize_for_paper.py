#!/usr/bin/env python3
"""Generate summary tables for the paper from measurement results.

Reads result_all.csv and produces:
  - summary_by_size.csv: per-size optimal config and EDP savings
  - edp_optimal_cores.csv: EDP-optimal core count per size
  - model_accuracy.csv: cost model accuracy metrics
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def load_results(path: str) -> list[dict]:
    """Load result CSV, returning only PASS rows with numeric fields parsed."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "PASS":
                continue
            try:
                parsed = {
                    "M": int(row["M"]),
                    "K": int(row["K"]),
                    "N": int(row["N"]),
                    "numSpm": int(row["numSpm"]),
                    "SPm": int(row["SPm"]),
                    "SPn": int(row["SPn"]),
                    "TPm": int(row["TPm"]),
                    "TPk": int(row["TPk"]),
                    "TPn": int(row["TPn"]),
                    "TM": int(row["TM"]),
                    "TK": int(row["TK"]),
                    "TN": int(row["TN"]),
                    "min_us": float(row["min_us"]),
                    "avg_us": float(row["avg_us"]),
                    "t_total_pred": float(row["t_total_pred"]),
                }
                # Energy fields may be missing or invalid
                npu_e = row.get("npu_energy_per_iter_uj", "")
                wall_s = row.get("wall_elapsed_s", "")
                if npu_e and wall_s:
                    parsed["npu_energy_per_iter_uj"] = float(npu_e)
                    parsed["wall_elapsed_s"] = float(wall_s)
                else:
                    parsed["npu_energy_per_iter_uj"] = float("nan")
                    parsed["wall_elapsed_s"] = 0.0
                rows.append(parsed)
            except (ValueError, KeyError):
                continue
    return rows


def compute_macs(m: int, k: int, n: int) -> int:
    """MACs for a single matmul: 2*M*K*N."""
    return 2 * m * k * n


def compute_bytes(m: int, k: int, n: int) -> int:
    """Total DRAM bytes transferred: (A + B + C) * elem_bytes, bf16=2."""
    return (m * k + k * n + m * n) * 2


def compute_edp(time_us: float, energy_uj: float) -> float:
    """Energy-Delay Product: E * T (uJ * us)."""
    if math.isnan(energy_uj) or energy_uj <= 0:
        return float("inf")
    return energy_uj * time_us


def group_by_size(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Group rows by (M, K, N)."""
    groups = {}
    for r in rows:
        key = (r["M"], r["K"], r["N"])
        groups.setdefault(key, []).append(r)
    return groups


def find_optimal(configs: list[dict], metric: str = "edp") -> dict:
    """Find the optimal config by the given metric."""
    if metric == "edp":
        valid = [c for c in configs if not math.isinf(compute_edp(c["min_us"], c["npu_energy_per_iter_uj"]))]
        if not valid:
            # Fall back to time-only
            return min(configs, key=lambda c: c["min_us"])
        return min(valid, key=lambda c: compute_edp(c["min_us"], c["npu_energy_per_iter_uj"]))
    elif metric == "time":
        return min(configs, key=lambda c: c["min_us"])
    else:
        raise ValueError(f"Unknown metric: {metric}")


def find_32core_best(configs: list[dict]) -> dict | None:
    """Find the best 32-core config (by EDP if energy valid, else by time)."""
    c32 = [c for c in configs if c["numSpm"] == 32]
    if not c32:
        return None
    return find_optimal(c32, "edp")


def generate_summary_by_size(groups: dict[tuple, list[dict]], out_path: str):
    """Generate summary_by_size.csv."""
    fieldnames = [
        "M", "K", "N", "MACs", "bytes",
        "opt_cores", "opt_SPm", "opt_SPn",
        "opt_TPm", "opt_TPk", "opt_TPn",
        "opt_TM", "opt_TK", "opt_TN",
        "opt_min_us", "opt_energy_uj", "opt_edp",
        "ref32_min_us", "ref32_energy_uj", "ref32_edp",
        "edp_savings_pct", "time_diff_pct", "energy_savings_pct",
        "gflops_opt", "n_configs",
    ]
    rows_out = []
    for key in sorted(groups.keys()):
        configs = groups[key]
        m, k, n = key
        macs = compute_macs(m, k, n)
        nbytes = compute_bytes(m, k, n)

        opt = find_optimal(configs, "edp")
        opt_edp = compute_edp(opt["min_us"], opt["npu_energy_per_iter_uj"])

        ref32 = find_32core_best(configs)

        ref32_min = ref32["min_us"] if ref32 else opt["min_us"]
        ref32_e = ref32["npu_energy_per_iter_uj"] if ref32 else opt["npu_energy_per_iter_uj"]
        ref32_edp = compute_edp(ref32_min, ref32_e)

        edp_save = 0.0
        if not math.isinf(ref32_edp) and ref32_edp > 0:
            edp_save = (1.0 - opt_edp / ref32_edp) * 100.0

        time_diff = (opt["min_us"] / ref32_min - 1.0) * 100.0 if ref32_min > 0 else 0.0
        e_save = 0.0
        if not math.isnan(ref32_e) and ref32_e > 0:
            e_save = (1.0 - opt["npu_energy_per_iter_uj"] / ref32_e) * 100.0

        gflops = macs / opt["min_us"] / 1e3 if opt["min_us"] > 0 else 0.0

        rows_out.append({
            "M": m, "K": k, "N": n,
            "MACs": macs, "bytes": nbytes,
            "opt_cores": opt["numSpm"],
            "opt_SPm": opt["SPm"], "opt_SPn": opt["SPn"],
            "opt_TPm": opt["TPm"], "opt_TPk": opt["TPk"], "opt_TPn": opt["TPn"],
            "opt_TM": opt["TM"], "opt_TK": opt["TK"], "opt_TN": opt["TN"],
            "opt_min_us": round(opt["min_us"], 2),
            "opt_energy_uj": round(opt["npu_energy_per_iter_uj"], 2)
                if not math.isnan(opt["npu_energy_per_iter_uj"]) else "",
            "opt_edp": round(opt_edp, 2) if not math.isinf(opt_edp) else "",
            "ref32_min_us": round(ref32_min, 2),
            "ref32_energy_uj": round(ref32_e, 2) if not math.isnan(ref32_e) else "",
            "ref32_edp": round(ref32_edp, 2) if not math.isinf(ref32_edp) else "",
            "edp_savings_pct": round(edp_save, 1),
            "time_diff_pct": round(time_diff, 1),
            "energy_savings_pct": round(e_save, 1),
            "gflops_opt": round(gflops, 1),
            "n_configs": len(configs),
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  summary_by_size.csv: {len(rows_out)} sizes written to {out_path}")
    return rows_out


def generate_edp_optimal_cores(groups: dict[tuple, list[dict]], out_path: str):
    """Generate edp_optimal_cores.csv with core-level EDP comparison."""
    fieldnames = [
        "M", "K", "N", "MACs",
        "edp_opt_cores", "edp_opt_cols",
        "time_vs_32c_pct", "energy_save_pct", "edp_save_pct",
        "category",
    ]
    rows_out = []
    for key in sorted(groups.keys()):
        configs = groups[key]
        m, k, n = key
        macs = compute_macs(m, k, n)

        opt = find_optimal(configs, "edp")
        opt_cores = opt["numSpm"]
        opt_cols = opt_cores // 4  # 4 rows per column

        ref32 = find_32core_best(configs)
        ref32_min = ref32["min_us"] if ref32 else opt["min_us"]
        ref32_e = ref32["npu_energy_per_iter_uj"] if ref32 else float("nan")
        ref32_edp = compute_edp(ref32_min, ref32_e)

        opt_edp = compute_edp(opt["min_us"], opt["npu_energy_per_iter_uj"])

        time_vs = (opt["min_us"] / ref32_min - 1.0) * 100.0 if ref32_min > 0 else 0.0
        e_save = 0.0
        if not math.isnan(ref32_e) and ref32_e > 0:
            e_save = (1.0 - opt["npu_energy_per_iter_uj"] / ref32_e) * 100.0
        edp_save = 0.0
        if not math.isinf(ref32_edp) and ref32_edp > 0:
            edp_save = (1.0 - opt_edp / ref32_edp) * 100.0

        # Categorize
        if m == k == n:
            if macs < 2_000_000:
                cat = "square_small"
            elif macs < 100_000_000:
                cat = "square_medium"
            else:
                cat = "square_large"
        else:
            if k <= 64:
                cat = "attention_head"
            elif n >= 3072 or k >= 3072:
                cat = "ffn_layer"
            else:
                cat = "projection"

        rows_out.append({
            "M": m, "K": k, "N": n, "MACs": macs,
            "edp_opt_cores": opt_cores,
            "edp_opt_cols": opt_cols,
            "time_vs_32c_pct": round(time_vs, 1),
            "energy_save_pct": round(e_save, 1),
            "edp_save_pct": round(edp_save, 1),
            "category": cat,
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  edp_optimal_cores.csv: {len(rows_out)} sizes written to {out_path}")
    return rows_out


def generate_model_accuracy(groups: dict[tuple, list[dict]], calib: dict, out_path: str):
    """Generate model_accuracy.csv with calibration model metrics."""
    fitted = calib.get("fitted_from", {})
    energy_fitted = calib.get("energy", {}).get("fitted_from", {})

    rows_out = [
        {
            "model": "Performance (v7 DMA-add)",
            "n_samples": fitted.get("n_samples", ""),
            "spearman_rho": fitted.get("spearman_rho", ""),
            "mape_pct": fitted.get("mape_pct", ""),
            "edp_core_accuracy": fitted.get("edp_core_accuracy", ""),
            "edp_regret_mean_pct": fitted.get("edp_regret_mean_pct", ""),
            "edp_regret_max_pct": fitted.get("edp_regret_max_pct", ""),
            "rank_inversions": fitted.get("rank_inversions", ""),
        },
        {
            "model": "Energy (T-B)",
            "n_samples": energy_fitted.get("n_samples", ""),
            "spearman_rho": energy_fitted.get("spearman_rho", ""),
            "mape_pct": energy_fitted.get("mape_pct", ""),
            "edp_core_accuracy": "",
            "edp_regret_mean_pct": "",
            "edp_regret_max_pct": "",
            "rank_inversions": "",
        },
    ]

    fieldnames = [
        "model", "n_samples", "spearman_rho", "mape_pct",
        "edp_core_accuracy", "edp_regret_mean_pct", "edp_regret_max_pct",
        "rank_inversions",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  model_accuracy.csv: {len(rows_out)} models written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper summary tables")
    parser.add_argument("--result", required=True, help="Path to result_all.csv")
    parser.add_argument("--calib", required=True, help="Path to calibration.json")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    rows = load_results(args.result)
    print(f"  {len(rows)} PASS rows loaded")

    with open(args.calib) as f:
        calib = json.load(f)

    groups = group_by_size(rows)
    print(f"  {len(groups)} unique sizes")

    print("\nGenerating tables:")
    generate_summary_by_size(groups, str(out_dir / "summary_by_size.csv"))
    generate_edp_optimal_cores(groups, str(out_dir / "edp_optimal_cores.csv"))
    generate_model_accuracy(groups, calib, str(out_dir / "model_accuracy.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()
