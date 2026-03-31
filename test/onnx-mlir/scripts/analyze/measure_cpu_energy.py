#!/usr/bin/env python3
"""Measure CPU matmul baseline with RAPL energy measurement.

Measures both single-thread and multi-thread execution time and energy
for each matrix size. Uses RAPL (package-0 and core domains) for
energy measurement.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

# RAPL paths
RAPL_PKG = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_CORE = "/sys/class/powercap/intel-rapl:0:0/energy_uj"

# Minimum wall time for reliable RAPL reading (seconds)
# 100ms minimum ensures RAPL resolution is sufficient
MIN_WALL_S = 0.10
N_WARMUP = 5
# Idle measurement: N samples, take median for noise robustness
# (Matches NPU host.cpp: N_IDLE_SAMPLES=5, IDLE_SAMPLE_WINDOW_S=0.2)
N_IDLE_SAMPLES = 5
IDLE_SAMPLE_WINDOW_S = 0.2


def read_rapl(path):
    """Read RAPL energy counter in microjoules."""
    with open(path) as f:
        return int(f.read().strip())


def measure_matmul(np, m, k, n, n_threads):
    """Measure matmul time and energy for given size and thread count.

    Automatically adjusts iteration count so total wall time >= MIN_WALL_S.
    """
    # Set thread count before any computation
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)

    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    # Warmup
    for _ in range(N_WARMUP):
        _ = a @ b

    # Estimate single iteration time (average of 3 runs for stability)
    est_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = a @ b
        t1 = time.perf_counter()
        est_times.append(t1 - t0)
    single_s = min(est_times)

    # Calculate iterations needed for reliable RAPL measurement
    # Need total wall time >= MIN_WALL_S * 2 (double margin for safety)
    n_iters = max(20, int(MIN_WALL_S * 2.0 / max(single_s, 1e-9)))

    # Measure idle power: N samples with median for noise robustness
    # Matches NPU host.cpp methodology (N_IDLE_SAMPLES, median selection)
    idle_pkg_samples = []
    idle_core_samples = []
    for _ in range(N_IDLE_SAMPLES):
        pkg0 = read_rapl(RAPL_PKG)
        core0 = read_rapl(RAPL_CORE)
        time.sleep(IDLE_SAMPLE_WINDOW_S)
        pkg1 = read_rapl(RAPL_PKG)
        core1 = read_rapl(RAPL_CORE)
        dt = IDLE_SAMPLE_WINDOW_S
        idle_pkg_samples.append((pkg1 - pkg0) / (dt * 1000))   # mW
        idle_core_samples.append((core1 - core0) / (dt * 1000))
    idle_pkg_samples.sort()
    idle_core_samples.sort()
    idle_pkg_mw = idle_pkg_samples[N_IDLE_SAMPLES // 2]
    idle_core_mw = idle_core_samples[N_IDLE_SAMPLES // 2]
    idle_wall = IDLE_SAMPLE_WINDOW_S  # per-sample window for scaling

    # Measure active: run n_iters iterations, measure total time + energy
    times_us = []

    pkg_before = read_rapl(RAPL_PKG)
    core_before = read_rapl(RAPL_CORE)
    wall_start = time.perf_counter()

    for _ in range(n_iters):
        t0 = time.perf_counter()
        c = a @ b
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    wall_end = time.perf_counter()
    pkg_after = read_rapl(RAPL_PKG)
    core_after = read_rapl(RAPL_CORE)

    # Prevent optimization
    _ = c[0, 0]

    wall_s = wall_end - wall_start
    total_pkg_uj = pkg_after - pkg_before
    total_core_uj = core_after - core_before

    # Subtract idle energy: idle_mw * wall_s * 1000 = idle energy in uJ
    # (Matches NPU host.cpp: npu_energy_uj = active_pkg_uj - idle_pkg_mw * wall_s * 1000)
    active_pkg_uj = total_pkg_uj - idle_pkg_mw * wall_s * 1000
    active_core_uj = total_core_uj - idle_core_mw * wall_s * 1000

    # Per-iteration energy
    pkg_per_iter_uj = total_pkg_uj / n_iters
    core_per_iter_uj = total_core_uj / n_iters
    active_pkg_per_iter_uj = max(0, active_pkg_uj / n_iters)
    active_core_per_iter_uj = max(0, active_core_uj / n_iters)

    macs = 2 * m * k * n
    min_us = min(times_us)
    avg_us = sum(times_us) / len(times_us)
    gflops = macs / min_us / 1e3 if min_us > 0 else 0.0

    # Power during active period
    active_pkg_mw = total_pkg_uj / (wall_s * 1000) if wall_s > 0 else 0
    active_core_mw = total_core_uj / (wall_s * 1000) if wall_s > 0 else 0

    return {
        "M": m, "K": k, "N": n,
        "MACs": macs,
        "threads": n_threads,
        "n_iters": n_iters,
        "min_us": round(min_us, 2),
        "avg_us": round(avg_us, 2),
        "gflops": round(gflops, 1),
        "wall_s": round(wall_s, 6),
        "idle_pkg_mw": round(idle_pkg_mw, 1),
        "idle_core_mw": round(idle_core_mw, 1),
        "active_pkg_mw": round(active_pkg_mw, 1),
        "active_core_mw": round(active_core_mw, 1),
        "pkg_per_iter_uj": round(pkg_per_iter_uj, 2),
        "core_per_iter_uj": round(core_per_iter_uj, 2),
        "active_pkg_per_iter_uj": round(active_pkg_per_iter_uj, 2),
        "active_core_per_iter_uj": round(active_core_per_iter_uj, 2),
    }


FIELDNAMES = [
    "M", "K", "N", "MACs", "threads", "n_iters",
    "min_us", "avg_us", "gflops", "wall_s",
    "idle_pkg_mw", "idle_core_mw",
    "active_pkg_mw", "active_core_mw",
    "pkg_per_iter_uj", "core_per_iter_uj",
    "active_pkg_per_iter_uj", "active_core_per_iter_uj",
]


def run_single_thread_batch(cases):
    """Run measurements in-process (threads already set before import)."""
    import numpy as np
    results = []
    for case in cases:
        m, k, n = case["M"], case["K"], case["N"]
        result = measure_matmul(np, m, k, n, 1)
        results.append(result)
        sz = f"{m}x{k}x{n}"
        print(f"{sz:>20s} {result['min_us']:>10.2f} {result['gflops']:>8.1f} "
              f"{result['active_pkg_mw']:>8.1f} {result['active_core_mw']:>8.1f} "
              f"{result['pkg_per_iter_uj']:>10.2f} {result['core_per_iter_uj']:>10.2f} "
              f"{result['n_iters']:>6d}")
    return results


def run_multithread_batch(op_path, n_threads, out_tmp):
    """Run multi-thread measurements in a subprocess.

    OpenBLAS binds thread count at import time, so a fresh process
    with the correct env vars is required.
    """
    import subprocess
    import sys

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["OPENBLAS_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)

    # Run this same script with --_worker flag
    cmd = [
        sys.executable, __file__,
        "--op", op_path,
        "--out", out_tmp,
        "--_worker", str(n_threads),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=False)
    if proc.returncode != 0:
        print(f"ERROR: subprocess for {n_threads}T exited with {proc.returncode}")
        return []

    # Read results from tmp CSV
    results = []
    with open(out_tmp) as f:
        for row in csv.DictReader(f):
            # Convert numeric fields back
            parsed = {}
            for k, v in row.items():
                if k in ("M", "K", "N", "MACs", "threads", "n_iters"):
                    parsed[k] = int(v)
                else:
                    parsed[k] = float(v)
            results.append(parsed)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure CPU matmul with RAPL energy")
    parser.add_argument("--op", required=True, help="Path to op_list_paper.json")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--_worker", type=int, default=0,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Worker mode: run a single thread config and write CSV
    if args._worker > 0:
        import numpy as np
        with open(args.op) as f:
            cases = json.load(f)["cases"]

        print(f"  [worker] Threads: {args._worker}")
        print(f"{'Size':>20s} {'min_us':>10s} {'GFLOPS':>8s} {'pkg_mW':>8s} {'core_mW':>8s} "
              f"{'pkg/iter':>10s} {'core/iter':>10s} {'iters':>6s}")
        print("-" * 85)

        results = []
        for case in cases:
            m, k, n = case["M"], case["K"], case["N"]
            result = measure_matmul(np, m, k, n, args._worker)
            results.append(result)
            sz = f"{m}x{k}x{n}"
            print(f"{sz:>20s} {result['min_us']:>10.2f} {result['gflops']:>8.1f} "
                  f"{result['active_pkg_mw']:>8.1f} {result['active_core_mw']:>8.1f} "
                  f"{result['pkg_per_iter_uj']:>10.2f} {result['core_per_iter_uj']:>10.2f} "
                  f"{result['n_iters']:>6d}")

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(results)
        return

    # Main mode
    # Set 1T before importing numpy
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import numpy as np

    with open(args.op) as f:
        op_data = json.load(f)
    cases = op_data["cases"]

    print(f"NumPy version: {np.__version__}")
    try:
        config = np.show_config(mode="dicts")
        if isinstance(config, dict):
            blas = config.get("Build Dependencies", {}).get("blas", {})
            print(f"BLAS: {blas.get('name', 'unknown')} {blas.get('version', '')}")
    except Exception:
        pass

    # Check RAPL access
    try:
        read_rapl(RAPL_PKG)
        print("RAPL: accessible (pkg + core)")
    except PermissionError:
        print("ERROR: RAPL not readable. "
              "Run: sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj")
        return

    all_results = []

    # 1T: run in-process (already set to 1 thread)
    print(f"\n{'=' * 80}")
    print(f"  Threads: 1 (in-process)")
    print(f"{'=' * 80}")
    print(f"{'Size':>20s} {'min_us':>10s} {'GFLOPS':>8s} {'pkg_mW':>8s} {'core_mW':>8s} "
          f"{'pkg/iter':>10s} {'core/iter':>10s} {'iters':>6s}")
    print("-" * 85)
    results_1t = run_single_thread_batch(cases)
    all_results.extend(results_1t)

    # 24T: run in subprocess (fresh process with 24 threads)
    print(f"\n{'=' * 80}")
    print(f"  Threads: 24 (subprocess)")
    print(f"{'=' * 80}")
    tmp_path = str(Path(args.out).parent / "_tmp_24t.csv")
    results_24t = run_multithread_batch(args.op, 24, tmp_path)
    all_results.extend(results_24t)

    # Clean up temp file
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    # Write combined CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
