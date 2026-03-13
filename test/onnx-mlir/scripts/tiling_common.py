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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Path constants
# ============================================================
THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parents[1]           # test/onnx-mlir/
REPO_ROOT = THIS_FILE.parents[3]           # mlir-aie/
DATA_DIR  = ROOT_DIR / "data"
OUT_DIR   = ROOT_DIR / "out"

DEFAULT_OP_PATH  = DATA_DIR / "op_list.json"
DEFAULT_SYS_PATH = REPO_ROOT / "include" / "onnx" / "Target" / "XDNA2" / "xdna2_info.json"


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


def write_tc_list(tc_cases: List[Dict[str, Any]], out_path: Path) -> None:
    """Write tc_list.json atomically for the build/run pipeline."""
    atomic_write_json({"cases": tc_cases}, out_path)


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
