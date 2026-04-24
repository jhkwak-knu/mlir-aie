#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_npu_cpu_v16.py -- Compare NPU (model-selected) vs CPU 1T/24T (+ GPU).

Unlike compare_npu_cpu.py which selects the measured EDP-optimal config, this
script selects the NPU configuration predicted as EDP-optimal by the v16 cost
model (DMA-Bottleneck + Power-Time-Byte), then looks up the corresponding
measured values.

Resolution modes:
  - exact:    model-predicted config exists in measured NPU data
  - fallback: predicted config absent -> pick same-core-count measurement
              with closest model-predicted EDP

With --gpu, iGPU (Radeon 890M, fp32) measurements are appended to each row.
Note: NPU uses bf16, CPU uses bf16, GPU uses fp32 (Kompute 0.9.0 binding
limitation); MACs are identical across.

Usage:
    python3 scripts/analyze/compare_npu_cpu_v16.py \\
        --op data/op_list.json \\
        --calib data/calibration.json \\
        --npu out/reports/result_v14_clean.csv \\
        --tc-list out/tc_list_v14.json \\
        --cpu out/reports/cpu_baseline_v14.csv \\
        [--gpu out/reports/gpu_baseline_v14.csv] \\
        --out-csv out/reports/cpu_npu_combined_v14.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow importing from scripts/generate/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "generate"))

from xdna_search.hw_constants import (  # noqa: E402
    DEFAULT_CALIB_PATH, DEFAULT_OP_PATH, DEFAULT_SYS_PATH,
)
from xdna_search.io import (  # noqa: E402
    CommentFilterFile,
    load_calibration, load_op_list, load_system_info,
)
from xdna_search.types import (  # noqa: E402
    CalibCoeffs, Candidate, CostResult, OpCase, SystemInfo,
)
from xdna_search.cost.v16_edp import evaluate_candidate  # noqa: E402
from cost_model import (  # noqa: E402
    enumerate_candidates, filter_candidates, select_optimal,
)

CLOCK_MHZ = 1500  # XDNA2 clock for cycle -> us conversion


# ------------------------------------------------------------
# Data containers
# ------------------------------------------------------------

@dataclass
class NpuMeasurement:
    """Measured NPU configuration row (from result_v14_clean.csv)."""
    case_index: int
    M: int
    K: int
    N: int
    num_cores: int
    SPm: int
    SPn: int
    TPm: int
    TPk: int
    TPn: int
    TM: int
    TK: int
    TN: int
    tp_order: int  # innermost axis (0/1/2), from tc_list_v14.json
    batch_min_avg_us: float
    batch_min_energy_per_iter_uj: float
    min_us: float
    core_energy_per_iter_uj: float

    @property
    def edp(self) -> float:
        return self.batch_min_avg_us * self.batch_min_energy_per_iter_uj


@dataclass
class CpuMeasurement:
    """Measured CPU row (1T or 24T)."""
    M: int
    K: int
    N: int
    threads: int
    batch_min_avg_us: float
    batch_min_energy_per_iter_uj: float

    @property
    def edp(self) -> float:
        return self.batch_min_avg_us * self.batch_min_energy_per_iter_uj


@dataclass
class GpuMeasurement:
    """Measured iGPU row (numSpm == -2)."""
    M: int
    K: int
    N: int
    batch_min_avg_us: float
    batch_min_energy_per_iter_uj: float  # RAPL package bracket-idle subtracted
    ppt_energy_per_iter_uj: float  # amdgpu PPT-based (secondary)
    ppt_active_mw: float  # amdgpu PPT active power (secondary)

    @property
    def edp(self) -> float:
        return self.batch_min_avg_us * self.batch_min_energy_per_iter_uj


@dataclass
class ComparisonRow:
    """One row of the output CSV: per-workload NPU(model) vs CPU comparison."""
    workload_idx: int
    M: int
    K: int
    N: int
    # NPU model-predicted config
    npu_num_cores: int
    npu_SPm: int
    npu_SPn: int
    npu_TPm: int
    npu_TPk: int
    npu_TPn: int
    npu_tp_order: int
    # NPU model-predicted metrics (cycles -> us)
    npu_t_pred_us: float
    npu_e_pred_uj: float
    npu_edp_pred: float
    # NPU measured metrics (looked up after selection)
    npu_t_meas_us: float
    npu_e_meas_uj: float
    npu_edp_meas: float
    npu_resolution: str
    # CPU 1T
    cpu_1t_t_us: float
    cpu_1t_e_uj: float
    cpu_1t_edp: float
    # CPU 24T
    cpu_24t_t_us: float
    cpu_24t_e_uj: float
    cpu_24t_edp: float
    # GPU (iGPU fp32)
    gpu_t_us: float
    gpu_e_uj: float  # RAPL package energy per iter (idle subtracted)
    gpu_edp: float
    gpu_ppt_e_uj: float  # PPT-based energy (secondary)
    gpu_ppt_mw: float  # PPT active power (secondary)
    # Ratios (CPU/NPU, >1 = NPU wins)
    speedup_1t: float
    speedup_24t: float
    energy_ratio_1t: float
    energy_ratio_24t: float
    edp_ratio_1t: float
    edp_ratio_24t: float
    # GPU ratios (GPU/NPU, >1 = NPU wins)
    speedup_gpu: float
    energy_ratio_gpu: float
    edp_ratio_gpu: float


# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------

def load_tc_list(path: Path) -> Dict[int, int]:
    """Return {case_index: tp_order}. tp_order = levels[0].tpOrder[0] (innermost)."""
    with path.open() as f:
        doc = json.load(f)
    idx_to_tpo: Dict[int, int] = {}
    # case_index in the CSV is 1-based and matches the order in tc_list.cases.
    for i, case in enumerate(doc.get("cases", []), start=1):
        levels = case.get("levels", [])
        if not levels:
            continue
        tp_order_list = levels[0].get("tpOrder")
        if not tp_order_list:
            continue
        idx_to_tpo[i] = int(tp_order_list[0])
    return idx_to_tpo


def load_npu_measurements(
    csv_path: Path, tc_map: Dict[int, int],
) -> Dict[Tuple[int, int, int], List[NpuMeasurement]]:
    """Group NPU PASS rows by (M,K,N). Joins tpOrder via case_index -> tc_list."""
    by_size: Dict[Tuple[int, int, int], List[NpuMeasurement]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(CommentFilterFile(f))
        for row in reader:
            if row.get("status") != "PASS":
                continue
            try:
                spm = int(row["SPm"])
                if spm <= 0:
                    continue  # skip CPU-merged rows
            except (TypeError, ValueError):
                continue
            case_idx = int(row["case_index"])
            tp_order = tc_map.get(case_idx)
            if tp_order is None:
                continue
            t_us = float(row.get("batch_min_avg_us", -1))
            e_uj = float(row.get("batch_min_energy_per_iter_uj", -1))
            if t_us <= 0 or e_uj <= 0:
                continue
            key = (int(row["M"]), int(row["K"]), int(row["N"]))
            by_size[key].append(NpuMeasurement(
                case_index=case_idx,
                M=key[0], K=key[1], N=key[2],
                num_cores=int(row["numSpm"]),
                SPm=spm,
                SPn=int(row["SPn"]),
                TPm=int(row["TPm"]),
                TPk=int(row["TPk"]),
                TPn=int(row["TPn"]),
                TM=int(row["TM"]),
                TK=int(row["TK"]),
                TN=int(row["TN"]),
                tp_order=tp_order,
                batch_min_avg_us=t_us,
                batch_min_energy_per_iter_uj=e_uj,
                min_us=float(row.get("min_us", -1)),
                core_energy_per_iter_uj=float(row.get("core_energy_per_iter_uj", -1)),
            ))
    return by_size


def load_cpu_measurements(
    csv_path: Path,
) -> Tuple[Dict[Tuple[int, int, int], CpuMeasurement],
           Dict[Tuple[int, int, int], CpuMeasurement]]:
    """Load CPU CSV, split into 1T and 24T by numSpm field."""
    cpu_1t: Dict[Tuple[int, int, int], CpuMeasurement] = {}
    cpu_24t: Dict[Tuple[int, int, int], CpuMeasurement] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(CommentFilterFile(f))
        for row in reader:
            if row.get("status") != "PASS":
                continue
            try:
                spm = int(row.get("SPm", -1))
                if spm >= 1:
                    continue  # skip NPU rows
            except (TypeError, ValueError):
                pass
            try:
                threads = int(row["numSpm"])
            except (TypeError, ValueError):
                continue
            key = (int(row["M"]), int(row["K"]), int(row["N"]))
            t_us = float(row.get("batch_min_avg_us", -1))
            e_uj = float(row.get("batch_min_energy_per_iter_uj", -1))
            if t_us <= 0 or e_uj <= 0:
                continue
            m = CpuMeasurement(
                M=key[0], K=key[1], N=key[2],
                threads=threads,
                batch_min_avg_us=t_us,
                batch_min_energy_per_iter_uj=e_uj,
            )
            if threads == 1:
                cpu_1t[key] = m
            elif threads == 24:
                cpu_24t[key] = m
    return cpu_1t, cpu_24t


def load_gpu_measurements(
    csv_path: Path,
) -> Dict[Tuple[int, int, int], GpuMeasurement]:
    """Load GPU CSV rows where numSpm == -2 (GPU identifier)."""
    by_size: Dict[Tuple[int, int, int], GpuMeasurement] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(CommentFilterFile(f))
        for row in reader:
            if row.get("status") != "PASS":
                continue
            try:
                num_spm = int(row.get("numSpm", 0))
            except (TypeError, ValueError):
                continue
            if num_spm != -2:
                continue
            try:
                key = (int(row["M"]), int(row["K"]), int(row["N"]))
            except (TypeError, ValueError, KeyError):
                continue
            t_us = float(row.get("batch_min_avg_us", -1))
            e_uj = float(row.get("batch_min_energy_per_iter_uj", -1))
            ppt_e = float(row.get("npu_energy_per_iter_uj", -1))
            ppt_mw = float(row.get("npu_power_mw", -1))
            if t_us <= 0 or e_uj <= 0:
                continue
            by_size[key] = GpuMeasurement(
                M=key[0], K=key[1], N=key[2],
                batch_min_avg_us=t_us,
                batch_min_energy_per_iter_uj=e_uj,
                ppt_energy_per_iter_uj=ppt_e,
                ppt_active_mw=ppt_mw,
            )
    return by_size


# ------------------------------------------------------------
# Model-based optimal selection + measurement lookup
# ------------------------------------------------------------

def select_npu_optimal(
    op: OpCase, sys_info: SystemInfo, coeffs: CalibCoeffs,
) -> Optional[CostResult]:
    """Run cost_model enumeration + filtering + EDP ranking, return the top-1."""
    all_cands = enumerate_candidates(op, sys_info)
    valid, _ = filter_candidates(all_cands, op, sys_info)
    if not valid:
        return None
    ranked = select_optimal(valid, op, coeffs)
    return ranked[0] if ranked else None


def _config_key(obj) -> Tuple[int, int, int, int, int, int, int]:
    """(SPm, SPn, TPm, TPk, TPn, num_cores, tp_order). Used for exact match."""
    if isinstance(obj, CostResult):
        c = obj.candidate
        return (c.SPm, c.SPn, c.TPm, c.TPk, c.TPn, c.num_cores, obj.tp_order)
    # NpuMeasurement
    return (obj.SPm, obj.SPn, obj.TPm, obj.TPk, obj.TPn, obj.num_cores, obj.tp_order)


def resolve_measurement(
    op: OpCase, best: CostResult, coeffs: CalibCoeffs,
    measurements: List[NpuMeasurement],
) -> Tuple[Optional[NpuMeasurement], str]:
    """Find measured NPU row for the model-predicted best config.

    Resolution order:
      1. Exact match by (SPm, SPn, TPm, TPk, TPn, num_cores, tp_order).
      2. Fallback (same cores): same num_cores, closest predicted EDP.
      3. Fallback (any cores): any measurement, closest predicted EDP.
         Used when the target core count has no measurements at all.
    """
    if not measurements:
        return None, "no_measurement"

    # 1. Exact match
    target = _config_key(best)
    for m in measurements:
        if _config_key(m) == target:
            return m, "exact"

    def _predicted_edp(m: NpuMeasurement) -> float:
        # Reconstruct a Candidate to feed into the cost model.
        cand = Candidate(
            num_cores=m.num_cores,
            num_columns=max(1, math.ceil(m.num_cores / 4)),  # 4 tiles/col on xdna2
            SPm=m.SPm, SPn=m.SPn,
            TPm=m.TPm, TPk=m.TPk, TPn=m.TPn,
            TM=m.TM, TK=m.TK, TN=m.TN,
        )
        res = evaluate_candidate(op, cand, m.tp_order, coeffs)
        return res.edp

    # 2. Fallback: same core count, closest model-predicted EDP
    same_cores = [m for m in measurements if m.num_cores == best.candidate.num_cores]
    if same_cores:
        closest = min(same_cores, key=lambda m: abs(_predicted_edp(m) - best.edp))
        return closest, "fallback"

    # 3. Fallback: any core count, closest model-predicted EDP
    closest = min(measurements, key=lambda m: abs(_predicted_edp(m) - best.edp))
    return closest, "fallback_xcore"


# ------------------------------------------------------------
# Build comparison rows
# ------------------------------------------------------------

def _ratio(cpu_val: float, npu_val: float) -> float:
    """CPU/NPU; >1 means NPU wins. Returns -1 when inputs invalid."""
    if cpu_val <= 0 or npu_val <= 0:
        return -1.0
    return cpu_val / npu_val


def build_row(
    idx: int,
    op: OpCase,
    best: CostResult,
    npu_meas: Optional[NpuMeasurement],
    resolution: str,
    cpu_1t: Optional[CpuMeasurement],
    cpu_24t: Optional[CpuMeasurement],
    gpu: Optional[GpuMeasurement] = None,
) -> ComparisonRow:
    c = best.candidate
    # Model-predicted (cycles -> us, pJ -> uJ)
    t_pred_us = best.t_total / CLOCK_MHZ
    e_pred_uj = best.e_total / 1e6
    edp_pred = t_pred_us * e_pred_uj

    # Measured
    if npu_meas is not None:
        t_meas = npu_meas.batch_min_avg_us
        e_meas = npu_meas.batch_min_energy_per_iter_uj
        edp_meas = npu_meas.edp
    else:
        t_meas = e_meas = edp_meas = -1.0

    # CPU
    c1t_t = cpu_1t.batch_min_avg_us if cpu_1t else -1.0
    c1t_e = cpu_1t.batch_min_energy_per_iter_uj if cpu_1t else -1.0
    c1t_edp = cpu_1t.edp if cpu_1t else -1.0

    c24t_t = cpu_24t.batch_min_avg_us if cpu_24t else -1.0
    c24t_e = cpu_24t.batch_min_energy_per_iter_uj if cpu_24t else -1.0
    c24t_edp = cpu_24t.edp if cpu_24t else -1.0

    # GPU
    gpu_t = gpu.batch_min_avg_us if gpu else -1.0
    gpu_e = gpu.batch_min_energy_per_iter_uj if gpu else -1.0
    gpu_edp = gpu.edp if gpu else -1.0
    gpu_ppt_e = gpu.ppt_energy_per_iter_uj if gpu else -1.0
    gpu_ppt_mw = gpu.ppt_active_mw if gpu else -1.0

    return ComparisonRow(
        workload_idx=idx,
        M=op.M, K=op.K, N=op.N,
        npu_num_cores=c.num_cores,
        npu_SPm=c.SPm, npu_SPn=c.SPn,
        npu_TPm=c.TPm, npu_TPk=c.TPk, npu_TPn=c.TPn,
        npu_tp_order=best.tp_order,
        npu_t_pred_us=t_pred_us,
        npu_e_pred_uj=e_pred_uj,
        npu_edp_pred=edp_pred,
        npu_t_meas_us=t_meas,
        npu_e_meas_uj=e_meas,
        npu_edp_meas=edp_meas,
        npu_resolution=resolution,
        cpu_1t_t_us=c1t_t,
        cpu_1t_e_uj=c1t_e,
        cpu_1t_edp=c1t_edp,
        cpu_24t_t_us=c24t_t,
        cpu_24t_e_uj=c24t_e,
        cpu_24t_edp=c24t_edp,
        gpu_t_us=gpu_t,
        gpu_e_uj=gpu_e,
        gpu_edp=gpu_edp,
        gpu_ppt_e_uj=gpu_ppt_e,
        gpu_ppt_mw=gpu_ppt_mw,
        speedup_1t=_ratio(c1t_t, t_meas),
        speedup_24t=_ratio(c24t_t, t_meas),
        energy_ratio_1t=_ratio(c1t_e, e_meas),
        energy_ratio_24t=_ratio(c24t_e, e_meas),
        edp_ratio_1t=_ratio(c1t_edp, edp_meas),
        edp_ratio_24t=_ratio(c24t_edp, edp_meas),
        speedup_gpu=_ratio(gpu_t, t_meas),
        energy_ratio_gpu=_ratio(gpu_e, e_meas),
        edp_ratio_gpu=_ratio(gpu_edp, edp_meas),
    )


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

CSV_FIELDNAMES = [
    "workload_idx", "M", "K", "N",
    "npu_num_cores", "npu_SPm", "npu_SPn",
    "npu_TPm", "npu_TPk", "npu_TPn", "npu_tp_order",
    "npu_t_pred_us", "npu_e_pred_uj", "npu_edp_pred",
    "npu_t_meas_us", "npu_e_meas_uj", "npu_edp_meas",
    "npu_resolution",
    "cpu_1t_t_us", "cpu_1t_e_uj", "cpu_1t_edp",
    "cpu_24t_t_us", "cpu_24t_e_uj", "cpu_24t_edp",
    "gpu_t_us", "gpu_e_uj", "gpu_edp",
    "gpu_ppt_e_uj", "gpu_ppt_mw",
    "speedup_1t", "speedup_24t",
    "energy_ratio_1t", "energy_ratio_24t",
    "edp_ratio_1t", "edp_ratio_24t",
    "speedup_gpu", "energy_ratio_gpu", "edp_ratio_gpu",
]


def write_csv(rows: List[ComparisonRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in rows:
            d = {k: getattr(r, k) for k in CSV_FIELDNAMES}
            # Round floats for readability
            for k, v in d.items():
                if isinstance(v, float):
                    d[k] = round(v, 4)
            writer.writerow(d)


def _fmt(v: float, decimals: int = 2) -> str:
    if v is None or v <= 0:
        return "N/A"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.2f}k"
    return f"{v:.{decimals}f}"


def _fmt_ratio(v: float) -> str:
    if v <= 0:
        return "N/A"
    return f"{v:.2f}x"


def _geo_mean(vals: List[float]) -> float:
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def print_report(rows: List[ComparisonRow], include_gpu: bool = False) -> None:
    if not rows:
        print("No rows to report.")
        return
    sep = "-" * 155
    hdr_extra_1 = f" {'GPU(us)':>10} {'GPU/NPU':>8}" if include_gpu else ""
    hdr_extra_e = f" {'GPU(uJ)':>10} {'GPU/NPU':>8}" if include_gpu else ""
    hdr_extra_edp = f" {'GPU EDP':>12} {'GPU/NPU':>8}" if include_gpu else ""
    print()
    print("=" * 155)
    label_xpu = "CPU 1T/24T" + (" + GPU" if include_gpu else "")
    print(f"  NPU (v16 model-selected) vs {label_xpu} Comparison")
    print("=" * 155)

    # [1] Performance
    print()
    print("  [1] Performance (batch_min_avg_us, lower is better)")
    print(sep)
    print(f"{'Size':>18} {'NPU(us)':>10} {'Cores':>5} {'Res':>9} "
          f"{'CPU-1T':>10} {'CPU-24T':>10}{hdr_extra_1} "
          f"{'1T/NPU':>8} {'24T/NPU':>8} {'Winner':>8}")
    print(sep)
    winner_set = {"NPU": 0, "CPU-1T": 0, "CPU-24T": 0}
    if include_gpu:
        winner_set["GPU"] = 0
    for r in rows:
        sz = f"{r.M}x{r.K}x{r.N}"
        cand = [("NPU", r.npu_t_meas_us), ("CPU-1T", r.cpu_1t_t_us),
                ("CPU-24T", r.cpu_24t_t_us)]
        if include_gpu:
            cand.append(("GPU", r.gpu_t_us))
        cand = [c for c in cand if c[1] > 0]
        winner = min(cand, key=lambda c: c[1])[0] if cand else "N/A"
        if winner in winner_set:
            winner_set[winner] += 1
        extra = (f" {_fmt(r.gpu_t_us):>10} {_fmt_ratio(r.speedup_gpu):>8}"
                 if include_gpu else "")
        print(f"{sz:>18} {_fmt(r.npu_t_meas_us):>10} {r.npu_num_cores:>5} "
              f"{r.npu_resolution:>9} "
              f"{_fmt(r.cpu_1t_t_us):>10} {_fmt(r.cpu_24t_t_us):>10}"
              f"{extra} "
              f"{_fmt_ratio(r.speedup_1t):>8} {_fmt_ratio(r.speedup_24t):>8} "
              f"{winner:>8}")
    print(sep)
    wins_str = ", ".join(f"{k}={v}" for k, v in winner_set.items())
    print(f"  Wins: {wins_str} (of {len(rows)})")

    # [2] Energy
    print()
    print("  [2] Energy (batch_min_energy_per_iter_uj, lower is better)")
    print(sep)
    print(f"{'Size':>18} {'NPU(uJ)':>10} {'Cores':>5} {'Res':>9} "
          f"{'CPU-1T':>10} {'CPU-24T':>10}{hdr_extra_e} "
          f"{'1T/NPU':>8} {'24T/NPU':>8} {'Winner':>8}")
    print(sep)
    ewins = {"NPU": 0, "CPU-1T": 0, "CPU-24T": 0}
    if include_gpu:
        ewins["GPU"] = 0
    for r in rows:
        sz = f"{r.M}x{r.K}x{r.N}"
        cand = [("NPU", r.npu_e_meas_uj), ("CPU-1T", r.cpu_1t_e_uj),
                ("CPU-24T", r.cpu_24t_e_uj)]
        if include_gpu:
            cand.append(("GPU", r.gpu_e_uj))
        cand = [c for c in cand if c[1] > 0]
        winner = min(cand, key=lambda c: c[1])[0] if cand else "N/A"
        if winner in ewins:
            ewins[winner] += 1
        extra = (f" {_fmt(r.gpu_e_uj):>10} {_fmt_ratio(r.energy_ratio_gpu):>8}"
                 if include_gpu else "")
        print(f"{sz:>18} {_fmt(r.npu_e_meas_uj):>10} {r.npu_num_cores:>5} "
              f"{r.npu_resolution:>9} "
              f"{_fmt(r.cpu_1t_e_uj):>10} {_fmt(r.cpu_24t_e_uj):>10}"
              f"{extra} "
              f"{_fmt_ratio(r.energy_ratio_1t):>8} "
              f"{_fmt_ratio(r.energy_ratio_24t):>8} "
              f"{winner:>8}")
    print(sep)
    wins_str = ", ".join(f"{k}={v}" for k, v in ewins.items())
    print(f"  Wins: {wins_str} (of {len(rows)})")

    # [3] EDP
    print()
    print("  [3] EDP (T_us * E_uJ, lower is better)")
    print(sep)
    print(f"{'Size':>18} {'NPU EDP':>12} {'Cores':>5} {'Res':>9} "
          f"{'CPU-1T EDP':>12} {'CPU-24T EDP':>12}{hdr_extra_edp} "
          f"{'1T/NPU':>8} {'24T/NPU':>8} {'Winner':>8}")
    print(sep)
    edpwins = {"NPU": 0, "CPU-1T": 0, "CPU-24T": 0}
    if include_gpu:
        edpwins["GPU"] = 0
    for r in rows:
        sz = f"{r.M}x{r.K}x{r.N}"
        cand = [("NPU", r.npu_edp_meas), ("CPU-1T", r.cpu_1t_edp),
                ("CPU-24T", r.cpu_24t_edp)]
        if include_gpu:
            cand.append(("GPU", r.gpu_edp))
        cand = [c for c in cand if c[1] > 0]
        winner = min(cand, key=lambda c: c[1])[0] if cand else "N/A"
        if winner in edpwins:
            edpwins[winner] += 1
        extra = (f" {_fmt(r.gpu_edp):>12} {_fmt_ratio(r.edp_ratio_gpu):>8}"
                 if include_gpu else "")
        print(f"{sz:>18} {_fmt(r.npu_edp_meas):>12} {r.npu_num_cores:>5} "
              f"{r.npu_resolution:>9} "
              f"{_fmt(r.cpu_1t_edp):>12} {_fmt(r.cpu_24t_edp):>12}"
              f"{extra} "
              f"{_fmt_ratio(r.edp_ratio_1t):>8} "
              f"{_fmt_ratio(r.edp_ratio_24t):>8} "
              f"{winner:>8}")
    print(sep)
    wins_str = ", ".join(f"{k}={v}" for k, v in edpwins.items())
    print(f"  Wins: {wins_str} (of {len(rows)})")

    # [4] Model accuracy
    print()
    print("  [4] Model accuracy (predicted vs measured, NPU)")
    print(sep)
    print(f"{'Size':>18} {'Res':>9} "
          f"{'T_pred':>10} {'T_meas':>10} {'T_err%':>8} "
          f"{'E_pred':>10} {'E_meas':>10} {'E_err%':>8} "
          f"{'EDP_err%':>10}")
    print(sep)
    for r in rows:
        sz = f"{r.M}x{r.K}x{r.N}"

        def _err(pred, meas):
            if pred <= 0 or meas <= 0:
                return -1
            return (pred - meas) / meas * 100.0

        t_err = _err(r.npu_t_pred_us, r.npu_t_meas_us)
        e_err = _err(r.npu_e_pred_uj, r.npu_e_meas_uj)
        edp_err = _err(r.npu_edp_pred, r.npu_edp_meas)
        print(f"{sz:>18} {r.npu_resolution:>9} "
              f"{_fmt(r.npu_t_pred_us):>10} {_fmt(r.npu_t_meas_us):>10} "
              f"{t_err:>7.1f}% "
              f"{_fmt(r.npu_e_pred_uj):>10} {_fmt(r.npu_e_meas_uj):>10} "
              f"{e_err:>7.1f}% "
              f"{edp_err:>9.1f}%")

    # [5] Summary
    print()
    print("=" * 135)
    print("  Summary")
    print("=" * 135)
    exact_n = sum(1 for r in rows if r.npu_resolution == "exact")
    fb_same_n = sum(1 for r in rows if r.npu_resolution == "fallback")
    fb_xcore_n = sum(1 for r in rows if r.npu_resolution == "fallback_xcore")
    none_n = len(rows) - exact_n - fb_same_n - fb_xcore_n
    print(f"  Resolution: exact={exact_n}, "
          f"fallback_same_cores={fb_same_n}, "
          f"fallback_xcore={fb_xcore_n}, "
          f"unmatched={none_n}")

    print(f"  Geometric mean ratios (other/NPU, >1 = NPU wins):")
    geo_entries = [
        ("1T ", "speedup_1t", "energy_ratio_1t", "edp_ratio_1t"),
        ("24T", "speedup_24t", "energy_ratio_24t", "edp_ratio_24t"),
    ]
    if any(r.gpu_t_us > 0 for r in rows):
        geo_entries.append(
            ("GPU", "speedup_gpu", "energy_ratio_gpu", "edp_ratio_gpu"))
    for label, tkey, ekey, edpkey in geo_entries:
        t_geo = _geo_mean([getattr(r, tkey) for r in rows])
        e_geo = _geo_mean([getattr(r, ekey) for r in rows])
        edp_geo = _geo_mean([getattr(r, edpkey) for r in rows])
        print(f"    {label}: speedup={t_geo:.2f}x  "
              f"energy={e_geo:.2f}x  EDP={edp_geo:.2f}x")

    # Crossover: first size where NPU wins over each baseline
    sorted_rows = sorted(rows, key=lambda r: r.M * r.K * r.N)
    cross_t = next((r for r in sorted_rows if r.speedup_24t > 1), None)
    cross_edp = next((r for r in sorted_rows if r.edp_ratio_24t > 1), None)
    if cross_t:
        print(f"  Perf crossover (NPU > CPU-24T): "
              f"{cross_t.M}x{cross_t.K}x{cross_t.N}")
    if cross_edp:
        print(f"  EDP crossover  (NPU > CPU-24T): "
              f"{cross_edp.M}x{cross_edp.K}x{cross_edp.N}")
    cross_t_gpu = next((r for r in sorted_rows if r.speedup_gpu > 1), None)
    cross_edp_gpu = next((r for r in sorted_rows if r.edp_ratio_gpu > 1), None)
    if cross_t_gpu:
        print(f"  Perf crossover (NPU > GPU):     "
              f"{cross_t_gpu.M}x{cross_t_gpu.K}x{cross_t_gpu.N}")
    if cross_edp_gpu:
        print(f"  EDP crossover  (NPU > GPU):     "
              f"{cross_edp_gpu.M}x{cross_edp_gpu.K}x{cross_edp_gpu.N}")
    print()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare NPU (v16 model-selected) vs CPU 1T/24T")
    parser.add_argument("--op", type=Path, default=DEFAULT_OP_PATH,
                        help="op_list.json")
    parser.add_argument("--sys", type=Path, default=DEFAULT_SYS_PATH,
                        help="xdna2_info.json")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH,
                        help="calibration.json (v16)")
    parser.add_argument("--npu", type=Path, required=True,
                        help="NPU result CSV (e.g. result_v14_clean.csv)")
    parser.add_argument("--tc-list", type=Path, required=True,
                        help="tc_list_v14.json (case_index -> tpOrder)")
    parser.add_argument("--cpu", type=Path, required=True,
                        help="CPU baseline CSV (e.g. cpu_baseline_v14.csv)")
    parser.add_argument("--gpu", type=Path, default=None,
                        help="Optional iGPU baseline CSV "
                             "(e.g. gpu_baseline_v14.csv)")
    parser.add_argument("--out-csv", type=Path, required=True,
                        help="Output combined CSV")
    parser.add_argument("--op-index", type=int, default=-1,
                        help="If >=0, process only this workload (0-based)")
    args = parser.parse_args()

    ops = load_op_list(args.op)
    sys_info = load_system_info(args.sys)
    coeffs = load_calibration(args.calib)
    tc_map = load_tc_list(args.tc_list)
    npu_by_size = load_npu_measurements(args.npu, tc_map)
    cpu_1t, cpu_24t = load_cpu_measurements(args.cpu)
    gpu_by_size: Dict[Tuple[int, int, int], GpuMeasurement] = {}
    if args.gpu is not None:
        gpu_by_size = load_gpu_measurements(args.gpu)

    summary = (f"Loaded: {len(ops)} workloads, "
               f"calibration={coeffs.perf_model}/{coeffs.energy_model}, "
               f"NPU sizes={len(npu_by_size)}, "
               f"CPU 1T/24T={len(cpu_1t)}/{len(cpu_24t)}")
    if args.gpu is not None:
        summary += f", GPU={len(gpu_by_size)}"
    print(summary)

    rows: List[ComparisonRow] = []
    for i, op in enumerate(ops):
        if args.op_index >= 0 and i != args.op_index:
            continue
        key = (op.M, op.K, op.N)
        npu_measurements = npu_by_size.get(key, [])

        best = select_npu_optimal(op, sys_info, coeffs)
        if best is None:
            print(f"[warn] {op.M}x{op.K}x{op.N}: no valid candidates, skip")
            continue

        npu_meas, resolution = resolve_measurement(
            op, best, coeffs, npu_measurements)

        row = build_row(
            idx=i, op=op,
            best=best, npu_meas=npu_meas, resolution=resolution,
            cpu_1t=cpu_1t.get(key), cpu_24t=cpu_24t.get(key),
            gpu=gpu_by_size.get(key),
        )
        rows.append(row)

    write_csv(rows, args.out_csv)
    print(f"CSV written: {args.out_csv}")

    print_report(rows, include_gpu=bool(gpu_by_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
