#!/usr/bin/env python3
"""
search_bench.py — Benchmark wall-clock search time for STAR-Map pruning rules.

Measures the end-to-end time required to:
    (1) enumerate all feasible configurations
    (2) find the cost-model argmin using the canonical tie-break key
        (edp, P, SPm, SPn, TPm, TPk, TPn, tpOrder_inner)

for four methods per workload:

    exhaustive          # enumerate_configs (no pruning)
    rule1               # enumerate_configs_pruned(rule1=True, rule2=False, rule3=False)
    rule12              # rule1=True, rule2=True, rule3=False
    rule123             # rule1=True, rule2=True, rule3=True  (final STAR-Map)

Methodology (mirrors the paper's dual-repeat-min for T/E):
    - Warmup: K calls (default 3) to stabilize CPU / caches
    - Measurement: N independent calls (default 10), min wall-time
    - gc.disable() around the measurement region, gc.collect() before
    - time.perf_counter_ns() for sub-microsecond resolution

Run on the same host used for T/E measurements, with:
    * CPU governor = performance (boost disabled), C2/C3 disabled
    * taskset -c 0  (pin to one core)
    * No other heavy load on the machine

Usage (from fig/):
    python search_bench.py
    python search_bench.py --warmup 5 --repeats 20
    python search_bench.py --cal data/calibration.json --out output/T13_search.csv
    python search_bench.py --only 128 128 128 384 384 384   # subset
    python search_bench.py --emit-tex                       # also write .tex
"""

from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    Config,
    load_config,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_RESULT_CSV,
    DEFAULT_TC_LIST_JSON,
)
from baselines import enumerate_configs, enumerate_configs_pruned
from cost_model import MappingConfig, edp


# ─── Canonical tie-break argmin ───────────────────────────────────────────────
def _argmin_canonical(configs: List[MappingConfig], cfg: Config) -> MappingConfig:
    """Pick the cost-model optimal config using the canonical tie-break key
    defined in baselines.find_framework_optimal."""
    def _key(mc: MappingConfig):
        return (
            edp(mc, cfg),
            mc.P, mc.SPm, mc.SPn,
            mc.TPm, mc.TPk, mc.TPn,
            mc.tpOrder_inner,
        )
    return min(configs, key=_key)


# ─── Single-method timing primitive ───────────────────────────────────────────
def _time_one_method(
    search_fn: Callable[[], List[MappingConfig]],
    cfg: Config,
    warmup: int,
    repeats: int,
) -> Tuple[int, List[int], int]:
    """Time (search_fn() + _argmin_canonical) repeatedly.

    Returns
    -------
    min_ns   : int                 minimum wall-time in nanoseconds
    all_ns   : list[int]           all per-call wall-times (for diagnostics)
    n_cfg    : int                 number of configurations enumerated

    Methodology
    -----------
    warmup calls are discarded. Between warmup and measurement, we run
    gc.collect() once and disable GC for the measurement window. The
    minimum is reported (as in the paper's performance measurement),
    matching the assumption that wall-time ≥ true compute time and the
    minimum is the least-contaminated observation.
    """
    # ── Warmup ──
    configs = None
    for _ in range(warmup):
        configs = search_fn()
        if configs:
            _ = _argmin_canonical(configs, cfg)

    # Sanity: search_fn must yield at least one config after warmup
    if not configs:
        return 0, [], 0

    # ── Measurement (gc off) ──
    gc.collect()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    samples: List[int] = []
    n_cfg = 0
    try:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            cs = search_fn()
            _ = _argmin_canonical(cs, cfg)
            t1 = time.perf_counter_ns()
            samples.append(t1 - t0)
            n_cfg = len(cs)
    finally:
        if gc_was_enabled:
            gc.enable()

    return min(samples), samples, n_cfg


# ─── Benchmark one workload across all methods ────────────────────────────────
def bench_workload(
    M: int, K: int, N: int,
    cfg: Config,
    warmup: int = 3,
    repeats: int = 10,
    verbose: bool = False,
) -> Dict:
    """Benchmark one (M, K, N) workload across four methods.

    Returns a flat dict keyed by '{method}_{field}' so it is ready to
    be assembled into a DataFrame row.
    """
    methods: Dict[str, Callable[[], List[MappingConfig]]] = {
        "exhaustive": lambda: enumerate_configs(M, K, N, cfg),
        "rule1":      lambda: enumerate_configs_pruned(
            M, K, N, cfg, rule1=True,  rule2=False, rule3=False),
        "rule12":     lambda: enumerate_configs_pruned(
            M, K, N, cfg, rule1=True,  rule2=True,  rule3=False),
        "rule123":    lambda: enumerate_configs_pruned(
            M, K, N, cfg, rule1=True,  rule2=True,  rule3=True),
    }

    row: Dict = {"M": M, "K": K, "N": N}

    for name, fn in methods.items():
        min_ns, samples, n_cfg = _time_one_method(fn, cfg, warmup, repeats)
        row[f"{name}_min_ns"]    = int(min_ns)
        row[f"{name}_min_us"]    = min_ns / 1_000.0
        row[f"{name}_min_ms"]    = min_ns / 1_000_000.0
        row[f"{name}_n_configs"] = int(n_cfg)
        # Extra observability: median / mean / std of samples
        if samples:
            arr = np.asarray(samples, dtype=np.int64)
            row[f"{name}_median_ns"] = int(np.median(arr))
            row[f"{name}_mean_ns"]   = int(np.mean(arr))
            row[f"{name}_std_ns"]    = int(np.std(arr, ddof=0))
            row[f"{name}_cv"]        = (
                float(np.std(arr, ddof=0) / np.mean(arr)) if np.mean(arr) > 0 else 0.0
            )

        if verbose:
            print(f"    {name:<10s} min={min_ns/1e6:8.3f}ms  "
                  f"CV={row[f'{name}_cv']*100:5.2f}%  "
                  f"n_cfg={n_cfg}")

    # ── Derived quantities ──
    for rule in ("rule1", "rule12", "rule123"):
        exh_ns = row["exhaustive_min_ns"]
        rn_ns  = row[f"{rule}_min_ns"]
        exh_n  = row["exhaustive_n_configs"]
        rn_n   = row[f"{rule}_n_configs"]
        row[f"{rule}_speedup"]   = (exh_ns / rn_ns) if rn_ns > 0 else float("nan")
        row[f"{rule}_reduction"] = (1.0 - rn_n / exh_n) if exh_n > 0 else float("nan")

    return row


# ─── Top-level driver ─────────────────────────────────────────────────────────
def run_bench(
    cfg: Config,
    workloads: Optional[List[Tuple[int, int, int]]] = None,
    warmup: int = 3,
    repeats: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the search-time benchmark across all (or a subset of) workloads."""
    wls = workloads if workloads is not None else cfg.WORKLOADS

    rows = []
    t_start = time.time()
    for i, (M, K, N) in enumerate(wls):
        if verbose:
            label = f"{M}x{K}x{N}"
            print(f"[{i+1:2d}/{len(wls)}] {label:<20s}  ", end="", flush=True)
        t0 = time.time()
        row = bench_workload(M, K, N, cfg, warmup=warmup, repeats=repeats,
                             verbose=False)
        rows.append(row)
        if verbose:
            print(
                f"exh={row['exhaustive_min_ms']:8.2f}ms  "
                f"r123={row['rule123_min_ms']:8.2f}ms  "
                f"sp={row['rule123_speedup']:6.1f}x  "
                f"red={row['rule123_reduction']*100:5.1f}%  "
                f"({time.time()-t0:.1f}s)"
            )
    if verbose:
        print(f"\nTotal benchmark wall-time: {time.time()-t_start:.1f}s")

    return pd.DataFrame(rows)


# ─── Summary (for console + body-text numbers) ────────────────────────────────
def summarize(df: pd.DataFrame) -> Dict:
    """Compute aggregate statistics suitable for the paper's body text."""
    def _gmean(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        return float(np.exp(np.log(x).mean())) if x.size else float("nan")

    out = {}
    for rule in ("rule1", "rule12", "rule123"):
        sp = df[f"{rule}_speedup"].to_numpy()
        rd = df[f"{rule}_reduction"].to_numpy() * 100.0
        out[f"{rule}_speedup_gmean"] = _gmean(sp)
        out[f"{rule}_speedup_min"]   = float(np.nanmin(sp))
        out[f"{rule}_speedup_max"]   = float(np.nanmax(sp))
        out[f"{rule}_reduction_mean"] = float(np.nanmean(rd))
        out[f"{rule}_reduction_min"]  = float(np.nanmin(rd))
        out[f"{rule}_reduction_max"]  = float(np.nanmax(rd))
    out["total_exhaustive_configs"] = int(df["exhaustive_n_configs"].sum())
    out["total_rule123_configs"]    = int(df["rule123_n_configs"].sum())
    out["total_rule123_reduction"]  = (
        1.0 - out["total_rule123_configs"] / out["total_exhaustive_configs"]
    ) * 100.0
    return out


# ─── LaTeX emitter ────────────────────────────────────────────────────────────
def emit_tex(df: pd.DataFrame, out_path: Path) -> None:
    """Write a LaTeX tabularx for T13 (one row per workload).

    Columns: Workload | |Ω| | t_exh [ms] | |Ω_{r123}| | t_{r123} [ms] | Speedup.
    """
    lines: List[str] = []
    lines.append(r"\begin{tabularx}{\textwidth}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Workload & $|\Omega|$ & $t_{\text{exh}}$ [ms] & "
        r"$|\Omega_{\text{r123}}|$ & $t_{\text{r123}}$ [ms] & Speedup \\"
    )
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        lines.append(
            f"{int(r['M'])}$\\times${int(r['K'])}$\\times${int(r['N'])} & "
            f"{int(r['exhaustive_n_configs'])} & "
            f"{r['exhaustive_min_ms']:.2f} & "
            f"{int(r['rule123_n_configs'])} & "
            f"{r['rule123_min_ms']:.2f} & "
            f"{r['rule123_speedup']:.1f}$\\times$ \\\\"
        )

    lines.append(r"\midrule")
    s = summarize(df)
    lines.append(
        f"\\textbf{{Total / Geomean}} & "
        f"{s['total_exhaustive_configs']} & -- & "
        f"{s['total_rule123_configs']} & -- & "
        f"{s['rule123_speedup_gmean']:.1f}$\\times$ \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append("")
    lines.append(
        r"% Caption: Search-time benchmark. $|\Omega|$: number of configurations"
    )
    lines.append(
        r"%   enumerated. $t$: min wall-time (warmup=K, min-of-N); see"
    )
    lines.append(
        r"%   search_bench.py for methodology. r123 = STAR-Map final"
    )
    lines.append(
        r"%   (Rule~1+2+3, $K=3$). Speedup = $t_{\text{exh}} / t_{\text{r123}}$."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def _parse_only(only: Optional[List[int]]) -> Optional[List[Tuple[int, int, int]]]:
    """--only 128 128 128 384 384 384  →  [(128,128,128), (384,384,384)]"""
    if not only:
        return None
    if len(only) % 3 != 0:
        raise SystemExit("--only requires triplets of integers (M K N ...)")
    return [tuple(only[i:i + 3]) for i in range(0, len(only), 3)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Search-time benchmark for STAR-Map pruning rules."
    )
    ap.add_argument("--cal",     type=Path, default=None,
                    help="Path to calibration JSON (default: data/calibration.json)")
    ap.add_argument("--csv",     type=Path, default=None,
                    help="Path to result CSV (default: data/result_v14.csv)")
    ap.add_argument("--json",    type=Path, default=None,
                    help="Path to tc_list JSON (default: data/tc_list_v14.json)")
    ap.add_argument("--warmup",  type=int,  default=3,  help="Warmup calls (default: 3)")
    ap.add_argument("--repeats", type=int,  default=10, help="Measurement calls (default: 10)")
    ap.add_argument("--out",     type=Path,
                    default=Path("output/T13_search_time.csv"),
                    help="Output CSV path")
    ap.add_argument("--emit-tex", action="store_true",
                    help="Also emit LaTeX tabularx at <out>.tex")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="Restrict to given workloads (triplets: M K N M K N ...)")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-workload progress")
    args = ap.parse_args()

    cal_path  = args.cal  or DEFAULT_CALIBRATION_PATH
    csv_path  = args.csv  or DEFAULT_RESULT_CSV
    json_path = args.json or DEFAULT_TC_LIST_JSON

    # ── Header ──
    print("=" * 60)
    print("STAR-Map Search-Time Benchmark")
    print("=" * 60)
    print(f"Host:        {platform.node()} / {platform.machine()} / "
          f"Python {sys.version.split()[0]}")
    print(f"Calibration: {cal_path}")
    print(f"CSV:         {csv_path}")
    print(f"TC list:     {json_path}")
    print(f"Warmup:      {args.warmup}")
    print(f"Repeats:     {args.repeats}  (min-of-N reported)")
    print(f"Output CSV:  {args.out}")
    print()

    cfg = load_config(cal_path, csv_path, json_path)
    wls = _parse_only(args.only)

    df = run_bench(cfg, workloads=wls,
                   warmup=args.warmup, repeats=args.repeats,
                   verbose=(not args.quiet))

    # ── Save ──
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved CSV:  {args.out}")

    if args.emit_tex:
        tex_path = args.out.with_suffix(".tex")
        emit_tex(df, tex_path)
        print(f"Saved TeX:  {tex_path}")

    # ── Summary ──
    s = summarize(df)
    print("\n── Summary (for paper body text) ──")
    print(f"  Total configs: {s['total_exhaustive_configs']} → "
          f"{s['total_rule123_configs']}  "
          f"({s['total_rule123_reduction']:.1f}% reduction)")
    for rule, label in [("rule1", "Rule 1    "),
                        ("rule12", "Rule 1+2  "),
                        ("rule123", "Rule 1+2+3")]:
        print(f"  {label}  speedup: "
              f"geomean {s[f'{rule}_speedup_gmean']:6.1f}×   "
              f"min {s[f'{rule}_speedup_min']:6.1f}×   "
              f"max {s[f'{rule}_speedup_max']:6.1f}×    "
              f"(reduction mean {s[f'{rule}_reduction_mean']:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
