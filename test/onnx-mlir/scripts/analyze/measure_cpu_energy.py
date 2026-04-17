#!/usr/bin/env python3
"""Measure CPU bf16 matmul baseline with RAPL energy measurement.

Uses PyTorch bfloat16 matmul on CPU (hardware-accelerated via AVX-512
BF16/VNNI on Zen4+) to match the NPU data type. Measures both
single-thread and multi-thread execution time and energy for each matrix
size. Uses RAPL (package-0 and core domains) for energy measurement.

Output CSV uses the same 50-column format as NPU result CSV
(run_tc_all.sh) for direct comparison. NPU-specific columns are
filled with -1.
"""

import argparse
import csv
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

# RAPL paths
RAPL_PKG = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_CORE = "/sys/class/powercap/intel-rapl:0:0/energy_uj"

# Idle measurement: N samples, take median for noise robustness
# (Matches NPU host.cpp: N_IDLE_SAMPLES=10, IDLE_SAMPLE_WINDOW_S=0.2)
N_IDLE_SAMPLES = 10
IDLE_SAMPLE_WINDOW_S = 0.2

# Per-batch bracket idle: matches NPU host.cpp (5 samples x 300ms)
N_BRACKET_IDLE_SAMPLES = 5
BRACKET_IDLE_WINDOW_S = 0.3

# Double-loop min parameters (matches NPU host.cpp v12 methodology)
N_WARMUP = 10
N_BATCHES = 5
TARGET_BATCH_WALL_S = 1.0
MIN_INNER = 10

# NPU-compatible 50-column CSV header (matches run_tc_all.sh line 129)
NPU_FIELDNAMES = [
    "case_index", "numSpm", "SPm", "SPn", "TPm", "TPk", "TPn",
    "TM", "TK", "TN", "M", "K", "N", "doubleBuffer", "t_total_pred",
    "status", "errors", "iters", "warmup",
    "avg_us", "min_us", "max_us",
    "step_avg_us", "step_min_us", "step_max_us",
    "trace_dispatch_us", "trace_kern_pct", "trace_gflops",
    "host_overhead_us", "ss_iter_cy", "ss_kernel_cy",
    "idle_pkg_mw", "active_pkg_mw", "npu_power_mw",
    "npu_energy_uj", "npu_energy_per_iter_uj", "wall_elapsed_s",
    "host_steps", "matmul_npu_us",
    "n_batches", "n_inner", "batch_min_avg_us",
    "batch_min_energy_per_iter_uj", "idle_post_mw",
    "bracket_idle_mean_mw", "batch_energy_cv_pct", "batch_step_cv_pct",
    "batch_best_wall_s", "batch_best_active_uj", "core_energy_per_iter_uj",
]


def read_rapl(path):
    """Read RAPL energy counter in microjoules."""
    with open(path) as f:
        return int(f.read().strip())


def _measure_idle_median(window_s, n_samples):
    """Measure idle power (mW) using median method.

    Returns (pkg_mw, core_mw).
    """
    pkg_samples = []
    core_samples = []
    for _ in range(n_samples):
        pkg0 = read_rapl(RAPL_PKG)
        core0 = read_rapl(RAPL_CORE)
        time.sleep(window_s)
        pkg1 = read_rapl(RAPL_PKG)
        core1 = read_rapl(RAPL_CORE)
        pkg_samples.append((pkg1 - pkg0) / (window_s * 1000))
        core_samples.append((core1 - core0) / (window_s * 1000))
    pkg_samples.sort()
    core_samples.sort()
    return pkg_samples[n_samples // 2], core_samples[n_samples // 2]


def _measure_bracket_idle(window_s, n_samples):
    """Measure idle power (mW) using bracket method.

    Returns (pkg_mw, core_mw). Uses median for robustness.
    """
    return _measure_idle_median(window_s, n_samples)


def measure_matmul(torch, m, k, n, n_threads):
    """Measure bf16 matmul time and energy using double-loop min methodology.

    Uses PyTorch bfloat16 matmul on CPU (hardware-accelerated via
    AVX-512 BF16/VNNI on Zen4+). Matches NPU host.cpp v12: warmup outside
    outer loop with min-based n_inner sizing, per-batch bracket idle
    (before+after), target wall 1.0s, post-measurement idle drift detection,
    CV metrics.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)

    a = torch.randn(m, k, dtype=torch.bfloat16).contiguous()
    b = torch.randn(k, n, dtype=torch.bfloat16).contiguous()
    # Mode control via env var: CPU_BASELINE_MODE
    #   "bf16" (default): native bf16 matmul (uses AVX-512 BF16 VNNI when available)
    #   "upcast_fp32": cast bf16 inputs to fp32 and run fp32 matmul
    #       (models "CPU without BF16 VNNI": bf16-precision data but fp32 compute)
    mode = os.environ.get("CPU_BASELINE_MODE", "bf16")
    if mode == "upcast_fp32":
        a = a.float().contiguous()
        b = b.float().contiguous()

    # Session-level idle (median of N samples)
    idle_pkg_mw, idle_core_mw = _measure_idle_median(
        IDLE_SAMPLE_WINDOW_S, N_IDLE_SAMPLES)

    # Warmup (N_WARMUP iterations, use MIN for n_inner sizing)
    single_s_min = 1e18
    for _ in range(N_WARMUP):
        t0 = time.perf_counter()
        c = torch.matmul(a, b)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        if elapsed < single_s_min:
            single_s_min = elapsed
    if single_s_min <= 0 or single_s_min >= 1e18:
        single_s_min = 1e-9

    # Dynamically determine inner iteration count
    n_inner = max(MIN_INNER, math.ceil(TARGET_BATCH_WALL_S / single_s_min))

    # Double-loop min: K batches x N inner iterations
    best_batch_avg_us = 1e18
    best_batch_energy_pkg_uj = 1e18
    best_batch_energy_core_uj = 1e18
    all_iter_times = []

    # Per-batch accumulators for diagnostics
    batch_bracket_idles = []  # (pkg_mw, core_mw) per batch
    batch_wall_times = []
    batch_raw_pkg_ujs = []  # raw RAPL delta (before idle subtraction)
    batch_energy_per_iter_list = []
    batch_active_pkg_mws = []

    # Best-energy batch tracking
    best_energy_wall_s = -1.0
    best_energy_raw_pkg_uj = -1.0

    for batch in range(N_BATCHES):
        # Per-batch bracket idle: BEFORE
        brk_before_pkg, brk_before_core = _measure_bracket_idle(
            BRACKET_IDLE_WINDOW_S, N_BRACKET_IDLE_SAMPLES)

        pkg_before = read_rapl(RAPL_PKG)
        core_before = read_rapl(RAPL_CORE)
        batch_wall_start = time.perf_counter()

        batch_time_total = 0.0
        for _ in range(n_inner):
            t0 = time.perf_counter()
            c = torch.matmul(a, b)
            t1 = time.perf_counter()
            iter_us = (t1 - t0) * 1e6
            batch_time_total += iter_us
            all_iter_times.append(iter_us)

        batch_wall_end = time.perf_counter()
        pkg_after = read_rapl(RAPL_PKG)
        core_after = read_rapl(RAPL_CORE)

        # Per-batch bracket idle: AFTER
        brk_after_pkg, brk_after_core = _measure_bracket_idle(
            BRACKET_IDLE_WINDOW_S, N_BRACKET_IDLE_SAMPLES)

        batch_wall_s = batch_wall_end - batch_wall_start
        batch_avg_us = batch_time_total / n_inner

        # Bracket idle = mean of before and after
        batch_idle_pkg = (brk_before_pkg + brk_after_pkg) / 2.0
        batch_idle_core = (brk_before_core + brk_after_core) / 2.0
        batch_bracket_idles.append(batch_idle_pkg)
        batch_wall_times.append(batch_wall_s)

        # Raw RAPL delta and active power
        raw_pkg_uj = float(pkg_after - pkg_before)
        batch_raw_pkg_ujs.append(raw_pkg_uj)
        if batch_wall_s > 0:
            batch_active_pkg_mws.append(raw_pkg_uj / batch_wall_s / 1000.0)

        # Batch energy per iteration (bracket-idle-subtracted)
        batch_pkg_uj = raw_pkg_uj - batch_idle_pkg * batch_wall_s * 1000
        batch_core_uj = float(core_after - core_before) - \
            batch_idle_core * batch_wall_s * 1000
        batch_pkg_per_iter = batch_pkg_uj / n_inner
        batch_core_per_iter = batch_core_uj / n_inner
        batch_energy_per_iter_list.append(batch_pkg_per_iter)

        # Independent min selection: time and energy separately
        if batch_avg_us < best_batch_avg_us:
            best_batch_avg_us = batch_avg_us
        if batch_pkg_per_iter < best_batch_energy_pkg_uj:
            best_batch_energy_pkg_uj = batch_pkg_per_iter
            best_energy_wall_s = batch_wall_s
            best_energy_raw_pkg_uj = raw_pkg_uj
        if batch_core_per_iter < best_batch_energy_core_uj:
            best_batch_energy_core_uj = batch_core_per_iter

    # Prevent optimization
    _ = c[0, 0].item()

    # Post-measurement idle for drift detection (matches NPU host.cpp)
    idle_post_pkg, _ = _measure_idle_median(
        IDLE_SAMPLE_WINDOW_S, N_IDLE_SAMPLES)

    # Bracket idle mean across all batches
    bracket_idle_mean_mw = -1.0
    if batch_bracket_idles:
        bracket_idle_mean_mw = sum(batch_bracket_idles) / len(batch_bracket_idles)

    # Batch energy CV (coefficient of variation %)
    batch_energy_cv_pct = -1.0
    if len(batch_energy_per_iter_list) >= 2:
        e_mean = sum(batch_energy_per_iter_list) / len(batch_energy_per_iter_list)
        e_var = sum((v - e_mean) ** 2 for v in batch_energy_per_iter_list) / \
            (len(batch_energy_per_iter_list) - 1)
        if e_mean > 0:
            batch_energy_cv_pct = (e_var ** 0.5) / e_mean * 100.0

    # Batch step time CV (using wall times as proxy, same as NPU host.cpp)
    batch_step_cv_pct = -1.0
    if len(batch_wall_times) >= 2:
        w_mean = sum(batch_wall_times) / len(batch_wall_times)
        w_var = sum((v - w_mean) ** 2 for v in batch_wall_times) / \
            (len(batch_wall_times) - 1)
        if w_mean > 0:
            batch_step_cv_pct = (w_var ** 0.5) / w_mean * 100.0

    # Active pkg mW from best-time batch
    active_pkg_mw = -1.0
    if batch_active_pkg_mws:
        # Use the batch with best avg time
        best_idx = batch_wall_times.index(min(batch_wall_times)) \
            if batch_wall_times else 0
        if best_idx < len(batch_active_pkg_mws):
            active_pkg_mw = batch_active_pkg_mws[best_idx]

    # Aggregate statistics
    macs = 2 * m * k * n
    total_iters = N_BATCHES * n_inner
    min_us = min(all_iter_times)
    max_us = max(all_iter_times)
    avg_us = sum(all_iter_times) / len(all_iter_times)
    gflops = macs / best_batch_avg_us / 1e3 if best_batch_avg_us > 0 else 0.0

    return {
        "M": m, "K": k, "N": n,
        "MACs": macs,
        "threads": n_threads,
        "n_iters": total_iters,
        "n_inner": n_inner,
        "min_us": round(min_us, 2),
        "max_us": round(max_us, 2),
        "avg_us": round(avg_us, 2),
        "gflops": round(gflops, 1),
        "idle_pkg_mw": round(idle_pkg_mw, 1),
        "idle_core_mw": round(idle_core_mw, 1),
        "active_pkg_mw": round(active_pkg_mw, 1),
        "pkg_per_iter_uj": round(max(0, best_batch_energy_pkg_uj), 2),
        "core_per_iter_uj": round(max(0, best_batch_energy_core_uj), 2),
        "batch_min_avg_us": round(best_batch_avg_us, 2),
        "idle_post_mw": round(idle_post_pkg, 1),
        "bracket_idle_mean_mw": round(bracket_idle_mean_mw, 1),
        "batch_energy_cv_pct": round(batch_energy_cv_pct, 4),
        "batch_step_cv_pct": round(batch_step_cv_pct, 4),
        "batch_best_wall_s": round(best_energy_wall_s, 6),
        "batch_best_active_uj": round(best_energy_raw_pkg_uj, 1),
    }


def _to_npu_csv_row(case_index, result):
    """Convert measure_matmul() result to NPU-compatible 50-column row.

    CPU-specific mapping:
    - numSpm encodes thread count (1 or 24)
    - step_* equals compute time (no memcpy/sync for CPU)
    - NPU-specific columns (trace, tiling, npu_power) are -1
    """
    threads = result["threads"]
    return OrderedDict([
        ("case_index", case_index),
        ("numSpm", threads),  # thread count encoded here
        ("SPm", -1),
        ("SPn", -1),
        ("TPm", -1),
        ("TPk", -1),
        ("TPn", -1),
        ("TM", -1),
        ("TK", -1),
        ("TN", -1),
        ("M", result["M"]),
        ("K", result["K"]),
        ("N", result["N"]),
        ("doubleBuffer", -1),
        ("t_total_pred", -1),
        ("status", "PASS"),
        ("errors", 0),
        ("iters", result["n_iters"]),
        ("warmup", N_WARMUP),
        ("avg_us", result["avg_us"]),
        ("min_us", result["min_us"]),
        ("max_us", result["max_us"]),
        # For CPU: step = compute (no memcpy/sync overhead)
        ("step_avg_us", result["avg_us"]),
        ("step_min_us", result["min_us"]),
        ("step_max_us", result["max_us"]),
        ("trace_dispatch_us", -1),
        ("trace_kern_pct", -1),
        ("trace_gflops", -1),
        ("host_overhead_us", -1),
        ("ss_iter_cy", -1),
        ("ss_kernel_cy", -1),
        ("idle_pkg_mw", result["idle_pkg_mw"]),
        ("active_pkg_mw", result["active_pkg_mw"]),
        ("npu_power_mw", -1),
        ("npu_energy_uj", -1),
        ("npu_energy_per_iter_uj", -1),
        ("wall_elapsed_s", result["batch_best_wall_s"]),
        ("host_steps", -1),
        ("matmul_npu_us", result["batch_min_avg_us"]),
        ("n_batches", N_BATCHES),
        ("n_inner", result["n_inner"]),
        ("batch_min_avg_us", result["batch_min_avg_us"]),
        ("batch_min_energy_per_iter_uj", result["pkg_per_iter_uj"]),
        ("idle_post_mw", result["idle_post_mw"]),
        ("bracket_idle_mean_mw", result["bracket_idle_mean_mw"]),
        ("batch_energy_cv_pct", result["batch_energy_cv_pct"]),
        ("batch_step_cv_pct", result["batch_step_cv_pct"]),
        ("batch_best_wall_s", result["batch_best_wall_s"]),
        ("batch_best_active_uj", result["batch_best_active_uj"]),
        ("core_energy_per_iter_uj", result["core_per_iter_uj"]),
    ])


def run_single_thread_batch(cases, start_index=1):
    """Run measurements in-process (threads already set before import)."""
    import torch
    results = []
    for i, case in enumerate(cases):
        m, k, n_val = case["M"], case["K"], case["N"]
        result = measure_matmul(torch, m, k, n_val, 1)
        row = _to_npu_csv_row(start_index + i, result)
        results.append(row)
        sz = f"{m}x{k}x{n_val}"
        print(f"{sz:>20s}  min={result['min_us']:>10.2f}us  "
              f"avg={result['batch_min_avg_us']:>10.2f}us  "
              f"GFLOPS={result['gflops']:>6.1f}  "
              f"E/iter={result['pkg_per_iter_uj']:>10.2f}uJ  "
              f"idle={result['idle_pkg_mw']:>8.1f}mW  "
              f"n_inner={result['n_inner']:>6d}")
    return results


def run_multithread_batch(op_path, n_threads, out_tmp, start_index=1):
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

    cmd = [
        sys.executable, __file__,
        "--op", op_path,
        "--out", out_tmp,
        "--_worker", str(n_threads),
        "--_start-index", str(start_index),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=False)
    if proc.returncode != 0:
        print(f"ERROR: subprocess for {n_threads}T exited with {proc.returncode}")
        return []

    # Read results from tmp CSV (already in NPU 50-column format)
    results = []
    with open(out_tmp) as f:
        for row_dict in csv.DictReader(f):
            results.append(OrderedDict(
                (k, row_dict[k]) for k in NPU_FIELDNAMES
            ))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure CPU matmul with RAPL energy (NPU-compatible CSV)")
    parser.add_argument("--op", required=True, help="Path to op_list_paper.json")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--_worker", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_start-index", type=int, default=1,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Worker mode: run a single thread config and write CSV
    if args._worker > 0:
        import torch
        with open(args.op) as f:
            cases = json.load(f)["cases"]

        print(f"  [worker] Threads: {args._worker}")
        print(f"{'Size':>20s}  {'min_us':>12s}  {'batch_avg':>12s}  "
              f"{'GFLOPS':>6s}  {'E/iter_uJ':>12s}  "
              f"{'idle_mW':>8s}  {'n_inner':>6s}")
        print("-" * 90)

        rows = []
        for i, case in enumerate(cases):
            m, k, n_val = case["M"], case["K"], case["N"]
            result = measure_matmul(torch, m, k, n_val, args._worker)
            row = _to_npu_csv_row(args._start_index + i, result)
            rows.append(row)
            sz = f"{m}x{k}x{n_val}"
            print(f"{sz:>20s}  min={result['min_us']:>10.2f}us  "
                  f"avg={result['batch_min_avg_us']:>10.2f}us  "
                  f"GFLOPS={result['gflops']:>6.1f}  "
                  f"E/iter={result['pkg_per_iter_uj']:>10.2f}uJ  "
                  f"idle={result['idle_pkg_mw']:>8.1f}mW  "
                  f"n_inner={result['n_inner']:>6d}")

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=NPU_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        return

    # Main mode
    # Set 1T before importing torch
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    with open(args.op) as f:
        op_data = json.load(f)
    cases = op_data["cases"]

    print(f"PyTorch version: {torch.__version__}")
    print(f"  torch.get_num_threads()={torch.get_num_threads()}")
    print(f"  mkldnn_available={torch.backends.mkldnn.is_available()}")
    try:
        bf16_ok = torch.cpu._is_avx512_bf16_supported()
        print(f"  AVX512_BF16 supported: {bf16_ok}")
    except Exception:
        pass

    # bf16 matmul sanity check (relative error < 5% vs fp32 reference)
    _a = torch.randn(64, 64, dtype=torch.bfloat16)
    _b = torch.randn(64, 64, dtype=torch.bfloat16)
    _c_bf = torch.matmul(_a, _b).float()
    _c_fp = torch.matmul(_a.float(), _b.float())
    _rel = ((_c_bf - _c_fp).abs() / (_c_fp.abs() + 1e-6)).mean().item()
    print(f"  bf16 sanity: mean relative error vs fp32 = {_rel*100:.3f}%")

    # Check RAPL access
    try:
        read_rapl(RAPL_PKG)
        print("RAPL: accessible (pkg + core)")
    except PermissionError:
        print("ERROR: RAPL not readable. "
              "Run: sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj")
        return

    n_cases = len(cases)
    all_rows = []

    # 1T: run in-process (already set to 1 thread)
    print(f"\n{'=' * 90}")
    print(f"  Threads: 1 (in-process), {n_cases} cases")
    print(f"  Batch mode: K={N_BATCHES}, target_wall={TARGET_BATCH_WALL_S}s")
    print(f"{'=' * 90}")
    print(f"{'Size':>20s}  {'min_us':>12s}  {'batch_avg':>12s}  "
          f"{'GFLOPS':>6s}  {'E/iter_uJ':>12s}  "
          f"{'idle_mW':>8s}  {'n_inner':>6s}")
    print("-" * 90)
    rows_1t = run_single_thread_batch(cases, start_index=1)
    all_rows.extend(rows_1t)

    # 24T: run in subprocess (fresh process with 24 threads)
    print(f"\n{'=' * 90}")
    print(f"  Threads: 24 (subprocess), {n_cases} cases")
    print(f"  Batch mode: K={N_BATCHES}, target_wall={TARGET_BATCH_WALL_S}s")
    print(f"{'=' * 90}")
    tmp_path = str(Path(args.out).parent / "_tmp_24t.csv")
    rows_24t = run_multithread_batch(
        args.op, 24, tmp_path, start_index=n_cases + 1)
    all_rows.extend(rows_24t)

    # Clean up temp file
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    # Write combined CSV with NPU-compatible 50-column format
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NPU_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults written to {out_path}")
    print(f"  Total rows: {len(all_rows)} ({n_cases} x 1T + {n_cases} x 24T)")
    print(f"  CSV columns: {len(NPU_FIELDNAMES)} (NPU-compatible)")


if __name__ == "__main__":
    main()
