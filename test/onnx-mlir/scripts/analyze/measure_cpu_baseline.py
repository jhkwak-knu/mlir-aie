#!/usr/bin/env python3
"""Measure CPU baseline for matrix multiplication (bf16 via numpy).

Runs matmul on the host CPU using numpy (backed by OpenBLAS/MKL) for
each matrix size in the op_list. Results are saved to a CSV file for
comparison with NPU measurements.

Note: NumPy does not natively support bf16 matmul, so we use fp32
and note the comparison caveat. The CPU time serves as a baseline
to demonstrate NPU acceleration benefit, not an exact apples-to-apples
comparison.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


# Number of repetitions for each size
N_REPEATS = 5
# Minimum total time per size to ensure reliable measurement
MIN_TOTAL_S = 0.5
# Warmup iterations
N_WARMUP = 3


def measure_matmul_cpu(m: int, k: int, n: int) -> dict:
    """Measure CPU matmul time for a given size.

    Returns dict with timing statistics.
    """
    # Use fp32 since numpy doesn't support bf16 matmul efficiently
    # This gives the CPU its best chance (fp32 is well-optimized)
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    # Warmup
    for _ in range(N_WARMUP):
        _ = a @ b

    # Determine iteration count for reliable timing
    t0 = time.perf_counter()
    _ = a @ b
    t1 = time.perf_counter()
    single_s = t1 - t0

    n_iters = max(N_REPEATS, int(MIN_TOTAL_S / max(single_s, 1e-9)))

    times_us = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        c = a @ b
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    # Force use of result to prevent compiler optimization
    _ = c[0, 0]

    macs = 2 * m * k * n
    min_us = min(times_us)
    avg_us = sum(times_us) / len(times_us)
    max_us = max(times_us)
    gflops = macs / min_us / 1e3 if min_us > 0 else 0.0

    return {
        "M": m, "K": k, "N": n,
        "MACs": macs,
        "n_iters": n_iters,
        "dtype": "fp32",
        "avg_us": round(avg_us, 2),
        "min_us": round(min_us, 2),
        "max_us": round(max_us, 2),
        "gflops": round(gflops, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure CPU matmul baseline")
    parser.add_argument("--op", required=True, help="Path to op_list_paper.json")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--threads", type=int, default=0,
                        help="Number of threads (0=all, 1=single)")
    args = parser.parse_args()

    with open(args.op) as f:
        op_data = json.load(f)
    cases = op_data["cases"]

    # Show numpy/BLAS info
    print(f"NumPy version: {np.__version__}")
    try:
        config = np.show_config(mode="dicts")
        if isinstance(config, dict):
            blas = config.get("Build Dependencies", {}).get("blas", {})
            print(f"BLAS: {blas.get('name', 'unknown')} {blas.get('version', '')}")
    except Exception:
        print("BLAS info: check np.show_config()")

    if args.threads > 0:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        print(f"Threads limited to: {args.threads}")
    else:
        print("Threads: all available")

    print(f"\nMeasuring {len(cases)} matrix sizes...")
    print(f"{'Size':>20s}  {'min_us':>10s}  {'avg_us':>10s}  {'GFLOPS':>8s}  {'iters':>6s}")
    print("-" * 62)

    results = []
    for case in cases:
        m, k, n = case["M"], case["K"], case["N"]
        result = measure_matmul_cpu(m, k, n)
        results.append(result)
        size_str = f"{m}x{k}x{n}"
        print(f"{size_str:>20s}  {result['min_us']:>10.2f}  {result['avg_us']:>10.2f}  "
              f"{result['gflops']:>8.1f}  {result['n_iters']:>6d}")

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["M", "K", "N", "MACs", "n_iters", "dtype",
                  "avg_us", "min_us", "max_us", "gflops"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {out_path}")

    # Summary: NPU comparison hint
    print("\n--- For NPU comparison ---")
    print("Load paper/analysis/summary_by_size.csv and compare min_us columns.")
    print("Speedup = cpu_min_us / npu_opt_min_us")


if __name__ == "__main__":
    main()
