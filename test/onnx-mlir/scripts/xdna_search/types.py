"""Data classes shared across the search/cost library.

Moved from scripts/generate/tiling_common.py without any semantic change.
Keeping original names (OpCase, SystemInfo, CalibCoeffs, DEFAULT_COEFFS) so
downstream consumers import the same identifiers via the shim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from xdna_search.hw_constants import CTILE_RESERVED_BYTES, ELEM_SIZE_MAP


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
class Candidate:
    """A single spatio-temporal parallelization configuration.

    Spatial: SPm × SPn compute tiles covering the M/N axes.
    Temporal: TPm × TPk × TPn sequential iterations per tile.
    Tile sizes (TM, TK, TN) are derived from (M, K, N) / (SP * TP); set to 0
    when that division does not land on an integer (constraint C2/C3 will
    then reject the candidate).
    """
    num_cores: int
    num_columns: int
    SPm: int
    SPn: int
    TPm: int
    TPk: int
    TPn: int
    TM: int
    TK: int
    TN: int
    ws_bytes: int = 0
    # Optional innermost temporal axis hint. When set (0=M, 1=N, 2=K) the
    # Searcher evaluates only that tpOrder instead of all three. Used by
    # pruned / baseline enumerators that already commit to one inner axis.
    inner_axis: Optional[int] = None

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn


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
class CostResult:
    """Cost model evaluation result for a (candidate, tp_order) pair."""
    candidate: Candidate
    tp_order: int           # 0=M, 1=N, 2=K (innermost temporal axis)
    # Performance (cycles)
    t_comp: float
    t_comm: float
    t_overhead: float
    t_total: float
    # Energy (pJ)
    e_dynamic_comp: float
    e_dynamic_comm: float
    e_static: float
    e_total: float
    # Energy-Delay Product (pJ * cycles)
    edp: float


@dataclass
class FilterResult:
    """Tracks how many candidates each constraint removes."""
    name: str
    before: int
    after: int

    @property
    def removed(self) -> int:
        return self.before - self.after


@dataclass
class CalibCoeffs:
    """Cost model coefficients loaded from calibration.json."""
    eff_macs: float       # Effective MACs/cycle/tile
    bw_eff_bpc: float     # Effective DMA bandwidth (bytes/cycle)
    l_sync_cy: float      # Per-temporal-step sync cost (cycles)
    l_pe_cy: float        # Per-PE (per-tile) per-iteration barrier cost (cycles)
    l_startup_cy: float   # One-time NPU startup cost (cycles)
    calibrated: bool      # True if loaded from file, False if defaults
    # Legacy v6-v8: alpha/beta scaling factors (kept for backward compat).
    # v9+: alpha=1.0, beta=1.0 (both fixed; T_comp and T_comm use physical values).
    perf_alpha: float = 1.0   # Deprecated in v9 (always 1.0)
    perf_beta: float = 1.0    # Deprecated in v9 (always 1.0)
    # v7 DMA-add: per-DMA-descriptor setup cost per temporal iteration.
    l_dma_cy: float = 0.0     # Per-DMA-op per-iteration cost (0 = v6 compat)
    # v16 DMA-Bottleneck: per-descriptor setup cost (used in max(L_SETUP, avg_tile/BW)).
    l_setup_cy: float = 0.0   # Per-descriptor DMA setup overhead (cycles)
    # Performance model variant (DMA count structure).
    # "Core-Sync" = N_dma*TP (v9 baseline). "DMA-Refined" = D_total (v13+).
    # "DMA-Bottleneck" = D*max(L_SETUP, avg_tile/BW) (v16).
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
    l_pe_cy=0.0,
    l_startup_cy=0.0,
    calibrated=False,
)
