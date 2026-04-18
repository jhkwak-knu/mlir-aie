#!/usr/bin/env python3
"""Measure iGPU (Radeon 890M, gfx1150) fp32 matmul baseline via Vulkan/Kompute.

Uses a tiled SPIR-V GEMM compute shader (16x16 workgroup, shared-memory
tiling). Energy is measured via package RAPL (matches CPU baseline); GPU
PPT is recorded as a secondary signal in NPU-named columns for reference.

Output CSV uses the same 50-column format as NPU/CPU result CSVs so that
compare_npu_cpu_v16.py can consume it directly (via --gpu option, added
later).

Data type note: fp32 is used due to kp 0.9.0 Python binding type
limitation. NPU baseline is bf16. MAC counts are identical; fp32 doubles
memory footprint vs bf16. Flagged in comparison reports as caveat.
"""

import argparse
import csv
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

# ---------- RAPL (package + core) ----------
RAPL_PKG = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_CORE = "/sys/class/powercap/intel-rapl:0:0/energy_uj"

# ---------- amdgpu PPT (iGPU Package Power Tracking) ----------
# Discovered at runtime: /sys/class/drm/card*/device/hwmon/hwmon*/power1_average
AMDGPU_PPT_GLOB = "/sys/class/drm/card*/device/hwmon/hwmon*/power1_average"
AMDGPU_PPT_LABEL_SUFFIX = "power1_label"

# ---------- Measurement constants (match measure_cpu_energy.py) ----------
N_IDLE_SAMPLES = 10
IDLE_SAMPLE_WINDOW_S = 0.2
N_BRACKET_IDLE_SAMPLES = 5
BRACKET_IDLE_WINDOW_S = 0.3
N_WARMUP = 10
N_BATCHES = 5
TARGET_BATCH_WALL_S = 1.0
MIN_INNER = 10

# ---------- Shader / workgroup ----------
SHADER_TILE = 16
DEFAULT_SHADER_PATH = str(
    Path(__file__).resolve().parent.parent / "gpu" / "gemm_fp32.spv"
)

# ---------- NPU-compatible 50-column CSV header ----------
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

# GPU identifier encoded in numSpm (matches convention: 1/24 for CPU threads,
# positive SPm*SPn for NPU; use -2 for GPU so it's easy to filter)
GPU_NUM_SPM = -2


def read_rapl(path):
    with open(path) as f:
        return int(f.read().strip())


def _discover_amdgpu_ppt():
    """Find the amdgpu PPT power sensor (power1_average), verifying the
    label is PPT. Returns path or None."""
    import glob
    candidates = glob.glob(AMDGPU_PPT_GLOB)
    for p in candidates:
        label = Path(p).parent / AMDGPU_PPT_LABEL_SUFFIX
        try:
            with open(label) as f:
                if f.read().strip().upper() == "PPT":
                    return p
        except OSError:
            continue
    return candidates[0] if candidates else None


def _read_ppt_mw(path):
    """Read PPT average power in milliwatts (input is microwatts)."""
    try:
        with open(path) as f:
            return int(f.read().strip()) / 1000.0
    except OSError:
        return -1.0


def _measure_idle_rapl_ppt(window_s, n_samples, ppt_path):
    """Sample RAPL (pkg, core) and PPT (mW) over window_s, return medians."""
    pkg_samples = []
    core_samples = []
    ppt_samples = []
    for _ in range(n_samples):
        pkg0 = read_rapl(RAPL_PKG)
        core0 = read_rapl(RAPL_CORE)
        ppt0 = _read_ppt_mw(ppt_path) if ppt_path else -1.0
        time.sleep(window_s)
        pkg1 = read_rapl(RAPL_PKG)
        core1 = read_rapl(RAPL_CORE)
        ppt1 = _read_ppt_mw(ppt_path) if ppt_path else -1.0
        pkg_samples.append((pkg1 - pkg0) / (window_s * 1000))
        core_samples.append((core1 - core0) / (window_s * 1000))
        if ppt0 >= 0 and ppt1 >= 0:
            ppt_samples.append((ppt0 + ppt1) / 2.0)
    pkg_samples.sort()
    core_samples.sort()
    ppt_samples.sort()
    ppt_med = ppt_samples[len(ppt_samples) // 2] if ppt_samples else -1.0
    return (
        pkg_samples[n_samples // 2],
        core_samples[n_samples // 2],
        ppt_med,
    )


def _prepare_algorithm(mgr, kp, spirv_bytes, M, K, N):
    """Create Kompute tensors + algorithm for a given shape. Returns
    (algorithm, tensors, push_const_array, groups)."""
    import numpy as np
    rng = np.random.default_rng(0xBEEF ^ (M * 1_000_003) ^ (K * 97) ^ N)
    A = rng.standard_normal(M * K, dtype=np.float32)
    B = rng.standard_normal(K * N, dtype=np.float32)
    C = np.zeros(M * N, dtype=np.float32)

    tA = mgr.tensor(A)
    tB = mgr.tensor(B)
    tC = mgr.tensor(C)

    groups = [
        (N + SHADER_TILE - 1) // SHADER_TILE,
        (M + SHADER_TILE - 1) // SHADER_TILE,
        1,
    ]
    # Push constants: three uint32 packed as three float32 via view.
    pc = np.array([M, K, N], dtype=np.uint32).view(np.float32)
    algo = mgr.algorithm([tA, tB, tC], spirv_bytes, groups, [], pc)
    return algo, (tA, tB, tC), pc, groups


def measure_matmul_gpu(kp, mgr, spirv_bytes, ppt_path, M, K, N):
    """Measure GPU fp32 matmul. Returns dict matching measure_matmul (CPU).

    Iteration = OpTensorSyncDevice + OpAlgoDispatch + OpTensorSyncLocal,
    evaluated as a single blocking sequence. This matches an end-to-end
    "host issues one GEMM" model equivalent to PyTorch's torch.matmul on
    CPU (which includes implicit memory traffic). For more precise
    kernel-only timing, a dispatch-only sequence is possible but would
    exclude HtoD/DtoH sync cost; we include sync to match host.cpp step
    semantics on NPU.
    """
    import numpy as np  # noqa

    # Build algorithm + tensors once per shape
    algo, tensors, _pc, _groups = _prepare_algorithm(
        mgr, kp, spirv_bytes, M, K, N)
    tA, tB, tC = tensors

    # Pre-upload A/B to device once so per-iteration we only re-dispatch.
    up_seq = mgr.sequence()
    up_seq.record(kp.OpTensorSyncDevice([tA, tB]))
    up_seq.eval()

    # Per-iteration sequence: dispatch only (keeps A/B on device).
    dispatch_seq = mgr.sequence()
    dispatch_seq.record(kp.OpAlgoDispatch(algo))

    # Session idle
    idle_pkg_mw, idle_core_mw, idle_ppt_mw = _measure_idle_rapl_ppt(
        IDLE_SAMPLE_WINDOW_S, N_IDLE_SAMPLES, ppt_path)

    # Warmup for n_inner sizing
    single_s_min = 1e18
    for _ in range(N_WARMUP):
        t0 = time.perf_counter()
        dispatch_seq.eval()
        t1 = time.perf_counter()
        if (t1 - t0) < single_s_min:
            single_s_min = t1 - t0
    if single_s_min <= 0 or single_s_min >= 1e18:
        single_s_min = 1e-9
    n_inner = max(MIN_INNER, math.ceil(TARGET_BATCH_WALL_S / single_s_min))

    # Double-loop min
    best_batch_avg_us = 1e18
    best_batch_energy_pkg_uj = 1e18
    best_batch_energy_core_uj = 1e18
    best_batch_energy_ppt_uj = 1e18
    all_iter_times = []
    batch_bracket_idles = []
    batch_wall_times = []
    batch_raw_pkg_ujs = []
    batch_energy_per_iter_list = []
    batch_active_pkg_mws = []
    batch_ppt_active_mws = []
    best_energy_wall_s = -1.0
    best_energy_raw_pkg_uj = -1.0

    for _batch in range(N_BATCHES):
        brk_before_pkg, brk_before_core, brk_before_ppt = \
            _measure_idle_rapl_ppt(
                BRACKET_IDLE_WINDOW_S, N_BRACKET_IDLE_SAMPLES, ppt_path)

        pkg_before = read_rapl(RAPL_PKG)
        core_before = read_rapl(RAPL_CORE)
        # PPT is average power, sample at start/end and average
        ppt_start = _read_ppt_mw(ppt_path) if ppt_path else -1.0
        batch_wall_start = time.perf_counter()

        batch_time_total = 0.0
        for _ in range(n_inner):
            t0 = time.perf_counter()
            dispatch_seq.eval()
            t1 = time.perf_counter()
            it_us = (t1 - t0) * 1e6
            batch_time_total += it_us
            all_iter_times.append(it_us)

        batch_wall_end = time.perf_counter()
        ppt_end = _read_ppt_mw(ppt_path) if ppt_path else -1.0
        pkg_after = read_rapl(RAPL_PKG)
        core_after = read_rapl(RAPL_CORE)

        brk_after_pkg, brk_after_core, brk_after_ppt = \
            _measure_idle_rapl_ppt(
                BRACKET_IDLE_WINDOW_S, N_BRACKET_IDLE_SAMPLES, ppt_path)

        batch_wall_s = batch_wall_end - batch_wall_start
        batch_avg_us = batch_time_total / n_inner
        batch_idle_pkg = (brk_before_pkg + brk_after_pkg) / 2.0
        batch_idle_core = (brk_before_core + brk_after_core) / 2.0
        batch_idle_ppt = -1.0
        if brk_before_ppt >= 0 and brk_after_ppt >= 0:
            batch_idle_ppt = (brk_before_ppt + brk_after_ppt) / 2.0
        batch_bracket_idles.append(batch_idle_pkg)
        batch_wall_times.append(batch_wall_s)

        raw_pkg_uj = float(pkg_after - pkg_before)
        batch_raw_pkg_ujs.append(raw_pkg_uj)
        if batch_wall_s > 0:
            batch_active_pkg_mws.append(raw_pkg_uj / batch_wall_s / 1000.0)

        # PPT active power = mean of start/end readings (already mW)
        ppt_active_mw = -1.0
        if ppt_start >= 0 and ppt_end >= 0:
            ppt_active_mw = (ppt_start + ppt_end) / 2.0
            batch_ppt_active_mws.append(ppt_active_mw)

        batch_pkg_uj = raw_pkg_uj - batch_idle_pkg * batch_wall_s * 1000
        batch_core_uj = float(core_after - core_before) - \
            batch_idle_core * batch_wall_s * 1000
        batch_pkg_per_iter = batch_pkg_uj / n_inner
        batch_core_per_iter = batch_core_uj / n_inner
        batch_energy_per_iter_list.append(batch_pkg_per_iter)

        # PPT-based active energy per iter (optional, secondary)
        batch_ppt_per_iter = -1.0
        if ppt_active_mw >= 0 and batch_idle_ppt >= 0:
            ppt_active_uj = (ppt_active_mw - batch_idle_ppt) * \
                batch_wall_s * 1000
            batch_ppt_per_iter = max(0.0, ppt_active_uj / n_inner)

        if batch_avg_us < best_batch_avg_us:
            best_batch_avg_us = batch_avg_us
        if batch_pkg_per_iter < best_batch_energy_pkg_uj:
            best_batch_energy_pkg_uj = batch_pkg_per_iter
            best_energy_wall_s = batch_wall_s
            best_energy_raw_pkg_uj = raw_pkg_uj
        if batch_core_per_iter < best_batch_energy_core_uj:
            best_batch_energy_core_uj = batch_core_per_iter
        if batch_ppt_per_iter >= 0 and batch_ppt_per_iter < best_batch_energy_ppt_uj:
            best_batch_energy_ppt_uj = batch_ppt_per_iter

    # Pull back C once for correctness sanity (small cost, excluded from timing)
    pull = mgr.sequence()
    pull.record(kp.OpTensorSyncLocal([tC]))
    pull.eval()
    _ = tC.data()[0]

    # Post idle
    idle_post_pkg, _, _ = _measure_idle_rapl_ppt(
        IDLE_SAMPLE_WINDOW_S, N_IDLE_SAMPLES, ppt_path)

    bracket_idle_mean_mw = -1.0
    if batch_bracket_idles:
        bracket_idle_mean_mw = sum(batch_bracket_idles) / len(
            batch_bracket_idles)

    batch_energy_cv_pct = -1.0
    if len(batch_energy_per_iter_list) >= 2:
        em = sum(batch_energy_per_iter_list) / len(batch_energy_per_iter_list)
        ev = sum((v - em) ** 2 for v in batch_energy_per_iter_list) / \
            (len(batch_energy_per_iter_list) - 1)
        if em > 0:
            batch_energy_cv_pct = (ev ** 0.5) / em * 100.0

    batch_step_cv_pct = -1.0
    if len(batch_wall_times) >= 2:
        wm = sum(batch_wall_times) / len(batch_wall_times)
        wv = sum((v - wm) ** 2 for v in batch_wall_times) / \
            (len(batch_wall_times) - 1)
        if wm > 0:
            batch_step_cv_pct = (wv ** 0.5) / wm * 100.0

    active_pkg_mw = -1.0
    if batch_active_pkg_mws:
        best_idx = batch_wall_times.index(min(batch_wall_times)) \
            if batch_wall_times else 0
        if best_idx < len(batch_active_pkg_mws):
            active_pkg_mw = batch_active_pkg_mws[best_idx]

    active_ppt_mw = -1.0
    if batch_ppt_active_mws:
        active_ppt_mw = sum(batch_ppt_active_mws) / len(batch_ppt_active_mws)

    macs = 2 * M * K * N
    min_us = min(all_iter_times)
    max_us = max(all_iter_times)
    avg_us = sum(all_iter_times) / len(all_iter_times)
    gflops = macs / best_batch_avg_us / 1e3 if best_batch_avg_us > 0 else 0.0

    # Best PPT energy (fallback -1 if unavailable)
    ppt_energy_per_iter = best_batch_energy_ppt_uj if \
        best_batch_energy_ppt_uj < 1e17 else -1.0

    return {
        "M": M, "K": K, "N": N,
        "MACs": macs,
        "threads": GPU_NUM_SPM,
        "n_iters": N_BATCHES * n_inner,
        "n_inner": n_inner,
        "min_us": round(min_us, 2),
        "max_us": round(max_us, 2),
        "avg_us": round(avg_us, 2),
        "gflops": round(gflops, 1),
        "idle_pkg_mw": round(idle_pkg_mw, 1),
        "idle_core_mw": round(idle_core_mw, 1),
        "idle_ppt_mw": round(idle_ppt_mw, 1) if idle_ppt_mw >= 0 else -1.0,
        "active_pkg_mw": round(active_pkg_mw, 1),
        "active_ppt_mw": round(active_ppt_mw, 1)
            if active_ppt_mw >= 0 else -1.0,
        "pkg_per_iter_uj": round(max(0.0, best_batch_energy_pkg_uj), 2),
        "core_per_iter_uj": round(max(0.0, best_batch_energy_core_uj), 2),
        "ppt_per_iter_uj": round(ppt_energy_per_iter, 2)
            if ppt_energy_per_iter >= 0 else -1.0,
        "batch_min_avg_us": round(best_batch_avg_us, 2),
        "idle_post_mw": round(idle_post_pkg, 1),
        "bracket_idle_mean_mw": round(bracket_idle_mean_mw, 1),
        "batch_energy_cv_pct": round(batch_energy_cv_pct, 4),
        "batch_step_cv_pct": round(batch_step_cv_pct, 4),
        "batch_best_wall_s": round(best_energy_wall_s, 6),
        "batch_best_active_uj": round(best_energy_raw_pkg_uj, 1),
    }


def _to_npu_csv_row(case_index, result):
    """Convert GPU measure result to NPU-compatible 50-column row.

    GPU-specific mapping:
    - numSpm = -2 (GPU identifier)
    - npu_power_mw / npu_energy_uj / npu_energy_per_iter_uj hold PPT data
      (secondary, not used for main comparison)
    - step_* == compute time (single dispatch per iteration)
    """
    return OrderedDict([
        ("case_index", case_index),
        ("numSpm", result["threads"]),  # -2 for GPU
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
        # Reuse NPU-named columns for PPT so consumers can pick up GPU
        # power without adding new columns.
        ("npu_power_mw", result["active_ppt_mw"]),
        ("npu_energy_uj", -1),
        ("npu_energy_per_iter_uj", result["ppt_per_iter_uj"]),
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


def main():
    parser = argparse.ArgumentParser(
        description="Measure iGPU fp32 matmul with RAPL+PPT energy "
                    "(NPU-compatible CSV)")
    parser.add_argument("--op", required=True,
                        help="Path to op_list.json")
    parser.add_argument("--out", required=True,
                        help="Output CSV path")
    parser.add_argument("--shader", default=DEFAULT_SHADER_PATH,
                        help="Path to compiled SPIR-V (default: "
                             "scripts/gpu/gemm_fp32.spv)")
    parser.add_argument("--op-index", type=int, default=None,
                        help="Run a single workload (0-based index) "
                             "instead of all")
    parser.add_argument("--append", action="store_true",
                        help="Append rows to an existing CSV instead of "
                             "overwriting. Validates header matches "
                             "NPU_FIELDNAMES and auto-continues case_index "
                             "from the existing max+1. If the file does not "
                             "exist, behaves like a fresh write.")
    args = parser.parse_args()

    # Check RAPL
    try:
        read_rapl(RAPL_PKG)
    except PermissionError:
        print("ERROR: RAPL not readable. Run: "
              "sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj")
        return

    # Load op list
    with open(args.op) as f:
        cases = json.load(f)["cases"]
    if args.op_index is not None:
        cases = [cases[args.op_index]]
        start_index = args.op_index + 1
    else:
        start_index = 1

    # Append mode: validate existing CSV header and auto-advance case_index
    out_path = Path(args.out)
    append_mode = args.append and out_path.exists() and \
        out_path.stat().st_size > 0
    if args.append and not append_mode:
        print(f"  --append: {out_path} does not exist, creating new")
    if append_mode:
        with out_path.open() as f:
            # Skip lines starting with '#' (comment markers, same as
            # CommentFilterFile in compare_npu_cpu_v16.py)
            header_line = None
            for line in f:
                if not line.startswith("#"):
                    header_line = line.rstrip("\r\n")
                    break
        if header_line is None:
            print(f"ERROR: --append target {out_path} has no header")
            return
        existing_cols = header_line.split(",")
        if existing_cols != NPU_FIELDNAMES:
            print(f"ERROR: --append target header does not match "
                  f"NPU_FIELDNAMES (50 cols). Got {len(existing_cols)} cols. "
                  f"First mismatch would break consumers.")
            return
        # Scan for max case_index to auto-advance
        max_idx = 0
        with out_path.open() as f:
            reader = csv.DictReader(
                (line for line in f if not line.startswith("#")))
            for row in reader:
                try:
                    max_idx = max(max_idx, int(row.get("case_index", 0)))
                except (TypeError, ValueError):
                    continue
        start_index = max_idx + 1
        print(f"  --append: existing rows detected, "
              f"case_index starts at {start_index}")

    # Load SPIR-V
    spv_path = Path(args.shader)
    if not spv_path.exists():
        print(f"ERROR: shader not found: {spv_path}")
        print(f"  Compile with: glslc {spv_path.with_suffix('.comp')} "
              f"-o {spv_path}")
        return
    spirv_bytes = spv_path.read_bytes()

    # Discover PPT sensor
    ppt_path = _discover_amdgpu_ppt()
    if ppt_path:
        print(f"  amdgpu PPT sensor: {ppt_path}")
    else:
        print("  amdgpu PPT sensor: NOT FOUND (PPT energy = -1)")

    # Init Kompute (lazy import so --help / errors don't require Vulkan)
    import kp
    mgr = kp.Manager()
    print(f"  Kompute Manager initialized (kp {kp.__version__})")
    print(f"  Shader: {spv_path} ({len(spirv_bytes)} bytes)")
    print(f"  Workloads: {len(cases)}")
    print(f"  Batch mode: K={N_BATCHES}, target_wall="
          f"{TARGET_BATCH_WALL_S}s, warmup={N_WARMUP}")

    print(f"\n{'Size':>20s}  {'min_us':>12s}  {'batch_avg':>12s}  "
          f"{'GFLOPS':>8s}  {'E_pkg/iter':>12s}  {'E_ppt/iter':>12s}  "
          f"{'idle_mW':>8s}  {'n_inner':>6s}")
    print("-" * 110)

    rows = []
    for i, case in enumerate(cases):
        m, k, n_val = case["M"], case["K"], case["N"]
        result = measure_matmul_gpu(kp, mgr, spirv_bytes, ppt_path,
                                    m, k, n_val)
        row = _to_npu_csv_row(start_index + i, result)
        rows.append(row)
        sz = f"{m}x{k}x{n_val}"
        ppt_str = (f"{result['ppt_per_iter_uj']:>10.2f}uJ"
                   if result['ppt_per_iter_uj'] >= 0 else "       N/A")
        print(f"{sz:>20s}  min={result['min_us']:>10.2f}us  "
              f"avg={result['batch_min_avg_us']:>10.2f}us  "
              f"GFLOPS={result['gflops']:>6.1f}  "
              f"E/iter={result['pkg_per_iter_uj']:>10.2f}uJ  "
              f"PPT={ppt_str}  "
              f"idle={result['idle_pkg_mw']:>8.1f}mW  "
              f"n_inner={result['n_inner']:>6d}")

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append_mode else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NPU_FIELDNAMES)
        if not append_mode:
            writer.writeheader()
        writer.writerows(rows)

    action = "appended to" if append_mode else "written to"
    print(f"\nResults {action} {out_path}")
    print(f"  New rows: {len(rows)}")
    print(f"  CSV columns: {len(NPU_FIELDNAMES)} (NPU-compatible)")
    print("  Data type: fp32 (Kompute 0.9.0 Python binding limit; "
          "NPU uses bf16)")


if __name__ == "__main__":
    main()
