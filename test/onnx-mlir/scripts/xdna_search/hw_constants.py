"""Hardware / path constants for the XDNA2 spatio-temporal tiling pipeline.

Physical memory sizes, intrinsic shapes, temporal-axis encoding, and the
canonical filesystem locations for op_list.json / xdna2_info.json /
calibration.json. Moved from scripts/generate/tiling_common.py without any
value change.
"""

from __future__ import annotations

from pathlib import Path

# Path constants (identical parent computation as the former tiling_common.py;
# this file sits at scripts/xdna_search/hw_constants.py so parents[2] is still
# test/onnx-mlir and parents[4] is mlir-aie).
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[2]            # test/onnx-mlir/
REPO_ROOT = THIS_FILE.parents[4]           # mlir-aie/
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "out"

DEFAULT_OP_PATH = DATA_DIR / "op_list.json"
DEFAULT_SYS_PATH = REPO_ROOT / "include" / "onnx" / "Target" / "XDNA2" / "xdna2_info.json"
DEFAULT_CALIB_PATH = DATA_DIR / "calibration.json"


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
