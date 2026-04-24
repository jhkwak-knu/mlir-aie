"""xdna_search — unified search + cost library for the XDNA2 tiling pipeline.

Single canonical import target for data classes, I/O helpers, math
utilities, cost functions, and searchers. Downstream scripts (cost_model
CLI, analyze/*, paper/search_bench) import from this package.
"""

from xdna_search.hw_constants import (
    CTILE_RESERVED_BYTES,
    DATA_DIR,
    DEFAULT_CALIB_PATH,
    DEFAULT_OP_PATH,
    DEFAULT_SYS_PATH,
    ELEM_SIZE_MAP,
    MMUL_R,
    MMUL_S,
    MMUL_T,
    OUT_DIR,
    REPO_ROOT,
    ROOT_DIR,
    TP_AXIS_K,
    TP_AXIS_M,
    TP_AXIS_N,
)
from xdna_search.io import (
    CommentFilterFile,
    _git_commit_short,
    atomic_write_json,
    atomic_write_text,
    build_metadata,
    load_calibration,
    load_op_list,
    load_system_info,
    write_tc_list,
)
from xdna_search.math_utils import (
    build_tp_order,
    divisors,
    factor_pairs,
    spearman_rank_correlation,
    ws_bytes,
)
from xdna_search.types import (
    CalibCoeffs,
    DEFAULT_COEFFS,
    OpCase,
    SystemInfo,
)

__all__ = [
    # types
    "OpCase", "SystemInfo", "CalibCoeffs", "DEFAULT_COEFFS",
    # hw_constants
    "CTILE_RESERVED_BYTES", "ELEM_SIZE_MAP",
    "MMUL_R", "MMUL_S", "MMUL_T",
    "TP_AXIS_M", "TP_AXIS_N", "TP_AXIS_K",
    "ROOT_DIR", "REPO_ROOT", "DATA_DIR", "OUT_DIR",
    "DEFAULT_OP_PATH", "DEFAULT_SYS_PATH", "DEFAULT_CALIB_PATH",
    # io
    "load_op_list", "load_system_info", "load_calibration",
    "CommentFilterFile", "atomic_write_json", "atomic_write_text",
    "_git_commit_short", "build_metadata", "write_tc_list",
    # math_utils
    "divisors", "factor_pairs", "ws_bytes",
    "build_tp_order", "spearman_rank_correlation",
]
