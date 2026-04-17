#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tiling_common.py — Shared constants, data structures, I/O, and math utilities
for the XDNA2 spatio-temporal tiling pipeline.

Used by: cost_model.py, (future scripts)
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Path constants
# ============================================================
THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parents[2]           # test/onnx-mlir/
REPO_ROOT = THIS_FILE.parents[4]           # mlir-aie/
DATA_DIR  = ROOT_DIR / "data"
OUT_DIR   = ROOT_DIR / "out"

DEFAULT_OP_PATH    = DATA_DIR / "op_list.json"
DEFAULT_SYS_PATH   = REPO_ROOT / "include" / "onnx" / "Target" / "XDNA2" / "xdna2_info.json"
DEFAULT_CALIB_PATH = DATA_DIR / "calibration.json"


# ============================================================
# Hardware / SW constants
# ============================================================

# SW overhead (stack, heap, reserved) subtracted from HW tile memory.
CTILE_RESERVED_BYTES = 4 * 1024

ELEM_SIZE_MAP = {
    "f16": 2, "bf16": 2, "f32": 4,
    "i8": 1, "i16": 2, "i32": 4,
    "ui8": 1, "ui16": 2, "ui32": 4,
}

# bf16 mmul<4,8,8> intrinsic shape (aie2p).
# 2x2 expansion requires TM % (2*MMUL_R), TN % (2*MMUL_T).
MMUL_R, MMUL_S, MMUL_T = 4, 8, 8

# tpOrder axis indices (innermost temporal loop axis).
TP_AXIS_M, TP_AXIS_N, TP_AXIS_K = 0, 1, 2


# ============================================================
# Data structures
# ============================================================
@dataclass
class OpCase:
    """Single matrix multiplication specification."""
    M: int
    K: int
    N: int
    elem_type: str = "bf16"

    @property
    def elem_bytes(self) -> int:
        return ELEM_SIZE_MAP.get(self.elem_type.lower(), 4)


@dataclass
class SystemInfo:
    """Hardware parameters loaded from xdna2_info.json."""
    total_cores: int
    comp_tiles_per_col: int
    max_columns: int
    spm_size_bytes: int
    mem_tile_mem_bytes: int

    @property
    def ct_usable_bytes(self) -> int:
        return self.spm_size_bytes - CTILE_RESERVED_BYTES


@dataclass
class CalibCoeffs:
    """Cost model coefficients loaded from calibration.json."""
    eff_macs: float       # Effective MACs/cycle/tile
    bw_eff_bpc: float     # Effective DMA bandwidth (bytes/cycle)
    l_sync_cy: float      # Per-temporal-step sync cost (cycles)
    l_core_cy: float      # Per-core setup cost (cycles)
    l_startup_cy: float   # One-time NPU startup cost (cycles)
    calibrated: bool      # True if loaded from file, False if defaults
    # Legacy v6-v8: alpha/beta scaling factors (kept for backward compat).
    # v9+: alpha=1.0, beta=1.0 (both fixed; T_comp and T_comm use physical values).
    perf_alpha: float = 1.0   # Deprecated in v9 (always 1.0)
    perf_beta: float = 1.0    # Deprecated in v9 (always 1.0)
    # v7 DMA-add: per-DMA-descriptor setup cost per temporal iteration.
    # N_dma = SPm+SPn (K-inner), SPm+2N (M-inner), SPn+2N (N-inner).
    # Balanced SP minimizes N_dma (AM-GM: SPm+SPn >= 2*sqrt(N_cores)).
    l_dma_cy: float = 0.0     # Per-DMA-op per-iteration cost (0 = v6 compat)
    # v9 Core-Sync: per-core per-iteration barrier cost.
    # T_overhead = L_SYNC*TP + L_CORE*P*TP + L_DMA*N_dma*TP + L_STARTUP
    # Performance model variant (DMA count structure).
    # "Core-Sync" = N_dma*TP (v9 baseline). "DMA-Refined" = D_total (v13+).
    perf_model: str = "Core-Sync"
    # Energy calibration (v3)
    energy_model: str = ""           # E-A, E-B, E-C, E-D (empty = not calibrated)
    energy_params: Dict[str, float] = None  # Model-specific fitted parameters
    energy_calibrated: bool = False  # True if energy section exists in calibration.json

    def __post_init__(self):
        if self.energy_params is None:
            self.energy_params = {}


# Pre-calibration fallback: matches original hardcoded constants in cost_model.py
DEFAULT_COEFFS = CalibCoeffs(
    eff_macs=256.0,       # PEAK_MACS
    bw_eff_bpc=4.0,       # BANDWIDTH_BPC
    l_sync_cy=20.0,       # ALPHA_CYCLES
    l_core_cy=0.0,
    l_startup_cy=0.0,
    calibrated=False,
)


# ============================================================
# I/O
# ============================================================
def load_op_list(path: Path) -> List[OpCase]:
    """Read op_list.json and return OpCase list from the 'cases' array."""
    if not path.is_file():
        raise FileNotFoundError(f"op_list.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    cases = doc.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Invalid op_list.json: missing 'cases' array")
    return [
        OpCase(
            M=int(c["M"]), K=int(c["K"]), N=int(c["N"]),
            elem_type=str(c.get("elemType", "bf16")),
        )
        for c in cases
    ]


def load_system_info(path: Path) -> SystemInfo:
    """Read xdna2_info.json and return SystemInfo."""
    if not path.is_file():
        raise FileNotFoundError(f"system info not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    sys_obj = doc["system"]
    device = sys_obj.get("device", {})
    spm_levels = sys_obj.get("spm_levels", [])
    if not spm_levels:
        raise ValueError("empty spm_levels")
    return SystemInfo(
        total_cores=int(sys_obj["total_cores"]),
        comp_tiles_per_col=int(device.get("comp_tiles_per_col", 4)),
        max_columns=int(device.get("max_columns", 8)),
        spm_size_bytes=int(spm_levels[0]["spm_size_bytes"]),
        mem_tile_mem_bytes=int(device.get("mem_tile_mem_bytes", 524288)),
    )


def load_calibration(path: Path = DEFAULT_CALIB_PATH) -> CalibCoeffs:
    """Load calibration.json. Returns DEFAULT_COEFFS if file is missing.

    Supports v2 (perf only) and v3 (perf + energy).
    """
    if not path.is_file():
        return DEFAULT_COEFFS
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    # Energy calibration (v3)
    energy_section = doc.get("energy", {})
    energy_model = energy_section.get("model", "")
    energy_params = energy_section.get("params", {})
    energy_calibrated = bool(energy_model)

    return CalibCoeffs(
        eff_macs=float(doc["eff_macs"]),
        bw_eff_bpc=float(doc["bw_eff_bpc"]),
        l_sync_cy=float(doc["l_sync_cy"]),
        l_core_cy=float(doc.get("l_core_cy", doc.get("l_sync2_cy", 0.0))),
        l_startup_cy=float(doc.get("l_startup_cy", 0)),
        calibrated=True,
        perf_alpha=float(doc.get("perf_alpha", 1.0)),
        perf_beta=float(doc.get("perf_beta", 1.0)),
        l_dma_cy=float(doc.get("l_dma_cy", 0.0)),
        perf_model=str(doc.get("model", "Core-Sync")),
        energy_model=energy_model,
        energy_params=energy_params,
        energy_calibrated=energy_calibrated,
    )


class CommentFilterFile:
    """Wraps a file object to skip lines starting with '#'.

    Use with csv.DictReader to transparently handle metadata comments:
        with open(path) as f:
            reader = csv.DictReader(CommentFilterFile(f))
    """

    def __init__(self, f):
        self._f = f
        self.metadata: List[str] = []

    def __iter__(self):
        for line in self._f:
            if line.startswith("#"):
                self.metadata.append(line.rstrip())
            else:
                yield line

    def __next__(self):
        for line in self._f:
            if line.startswith("#"):
                self.metadata.append(line.rstrip())
                continue
            return line
        raise StopIteration

    def readline(self):
        """Support csv.reader which calls readline()."""
        return next(self, "")


def atomic_write_json(obj: Any, out_path: Path) -> None:
    """Write JSON atomically to prevent partial writes on crash."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".tmp", delete=False,
            dir=str(out_path.parent), encoding="utf-8",
        ) as tmp:
            json.dump(obj, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(text: str, out_path: Path) -> None:
    """Write text atomically to prevent partial writes on crash."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".logtmp", delete=False,
            dir=str(out_path.parent), encoding="utf-8",
        ) as tmp:
            tmp.write(text)
            tmp.flush()
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _git_commit_short() -> str:
    """Get current git commit hash (short), or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_metadata(
    calib_path: Optional[Path] = None,
    coeffs: Optional["CalibCoeffs"] = None,
) -> Dict[str, Any]:
    """Build metadata dict for tc_list.json and result CSV traceability."""
    meta: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit_short(),
    }
    if calib_path:
        meta["calibration_file"] = str(calib_path.name)
    if coeffs:
        meta["perf_model"] = coeffs.__class__.__name__
        meta["perf_version"] = 9
        meta["bw_eff_bpc"] = coeffs.bw_eff_bpc
        meta["energy_model"] = coeffs.energy_model or "none"
    return meta


def write_tc_list(
    tc_cases: List[Dict[str, Any]], out_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write tc_list.json atomically with optional metadata."""
    doc: Dict[str, Any] = {}
    if metadata:
        doc["metadata"] = metadata
    doc["cases"] = tc_cases
    atomic_write_json(doc, out_path)


# ============================================================
# Math utilities
# ============================================================
def divisors(n: int) -> List[int]:
    """All positive divisors of n, sorted ascending."""
    ds = set()
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            ds.add(d)
            ds.add(n // d)
    return sorted(ds)


def factor_pairs(n: int) -> List[Tuple[int, int]]:
    """All (a, b) pairs with a * b == n, sorted by a."""
    return [(d, n // d) for d in divisors(n)]


def ws_bytes(TM: int, TK: int, TN: int, elem_bytes: int) -> int:
    """Working-set size for one compute tile: A(TM*TK) + B(TK*TN) + C(TM*TN)."""
    return elem_bytes * (TM * TK + TK * TN + TM * TN)


# ============================================================
# tpOrder utility
# ============================================================
def build_tp_order(winner_axis: int) -> List[int]:
    """
    Build full tpOrder from the innermost (winner) axis.
    Returns [winner, middle, outermost] where remaining axes follow K>M>N priority.
    """
    default_order = [TP_AXIS_K, TP_AXIS_M, TP_AXIS_N]
    return [winner_axis] + [ax for ax in default_order if ax != winner_axis]
