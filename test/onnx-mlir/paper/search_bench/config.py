"""
config.py — Configuration loader for paper figure generation.

Loads calibration coefficients from an external JSON file so that
re-calibration only requires replacing the JSON, not editing Python code.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# ─── Default paths ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
# Single source of truth: reuse the main calibration file under test/onnx-mlir/data/.
# Flat v16 structure is mapped onto {hw, perf, energy} in load_config().
MAIN_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_CALIBRATION_PATH = MAIN_DATA_DIR / "calibration.json"
DEFAULT_RESULT_CSV = DATA_DIR / "result_v14.csv"
DEFAULT_TC_LIST_JSON = DATA_DIR / "tc_list_v14.json"
DEFAULT_CPU_ENERGY_CSV = DATA_DIR / "cpu_npu_combined_v13.csv"


# ─── Data classes ────────────────────────────────────────────────────────────
@dataclass
class HWConfig:
    P_total: int = 32
    G_size: int = 4          # column gating granularity
    L_mem_bytes: int = 61440  # 60KB usable
    elem_bytes: int = 2       # bf16
    P_set: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 32])
    TM_align: int = 8
    TK_align: int = 8
    TN_align: int = 16
    clock_mhz: float = 1500.0


@dataclass
class PerfCoeffs:
    """v9 Core-Sync performance model coefficients."""
    eff_macs: float = 24.28      # MACs/cycle/tile (measured)
    bw_eff_bpc: float = 4.0      # bytes/cycle effective bandwidth
    L_SYNC: float = 0.0          # per-iteration sync cost (cycles)
    L_PE: float = 0.0          # per-PE per-iteration barrier (cycles)
    L_DMA: float = 0.0           # per-DMA-descriptor setup (cycles)
    L_STARTUP: float = 0.0       # one-time dispatch overhead (cycles)


@dataclass
class EnergyCoeffs:
    """v16 4-term energy model coefficients."""
    P_PE: float = 0.0      # aggregate per-PE active power (mW)
    E_BYTE: float = 0.0      # per-byte data-movement energy (µJ/B)
    E_STARTUP: float = 0.0   # one-time dispatch / startup energy (µJ)
    P_BASE: float = 0.0      # system base power, core-count independent (W)


@dataclass
class Config:
    hw: HWConfig = field(default_factory=HWConfig)
    perf: PerfCoeffs = field(default_factory=PerfCoeffs)
    energy: EnergyCoeffs = field(default_factory=EnergyCoeffs)
    result_csv: Path = DEFAULT_RESULT_CSV
    tc_list_json: Path = DEFAULT_TC_LIST_JSON
    cpu_energy_csv: Path = DEFAULT_CPU_ENERGY_CSV

    # Ordered workload list (MACs ascending) for consistent figure ordering
    WORKLOADS: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (32, 32, 32),       # 66K
        (64, 64, 64),       # 524K
        (128, 64, 128),     # 2M
        (128, 128, 128),    # 4M
        (256, 64, 256),     # 8M
        (256, 256, 256),    # 34M
        (512, 64, 512),     # 34M
        (32, 768, 768),     # 38M
        (384, 384, 384),    # 113M
        (128, 768, 768),    # 151M
        (512, 512, 512),    # 268M
        (128, 1024, 1024),  # 268M
        (256, 1024, 1024),  # 537M
        (128, 768, 3072),   # 604M
        (128, 3072, 768),   # 604M
        # (768, 768, 768),    # 906M  — excluded (high regret)
        (256, 1024, 4096),  # 2.1G
        (1024, 1024, 1024), # 2.1G
        (2048, 2048, 2048), # 17.2G
    ])

    WORKLOAD_DESC: dict = field(default_factory=lambda: {
        (32, 32, 32):       "Minimal size",
        (64, 64, 64):       "Small square",
        (128, 64, 128):     "Attention score",
        (128, 128, 128):    "Medium square",
        (256, 64, 256):     "Attention",
        (256, 256, 256):    "Medium square",
        (512, 64, 512):     "Large attention",
        (32, 768, 768):     "Edge batch=1",
        (384, 384, 384):    "Crossover point",
        (128, 768, 768):    "BERT QKV proj",
        (512, 512, 512):    "Large square",
        (128, 1024, 1024):  "GPT proj",
        (256, 1024, 1024):  "Large proj",
        (128, 768, 3072):   "FFN up-proj",
        (128, 3072, 768):   "FFN down-proj",
        # (768, 768, 768):    "BERT hidden",  — excluded
        (256, 1024, 4096):  "Very large proj",
        (1024, 1024, 1024): "Large square",
        (2048, 2048, 2048): "Very large",
    })

    @property
    def workload_labels(self) -> List[str]:
        """Short labels for each workload, e.g. '128×768×768'."""
        return [f"{m}×{k}×{n}" for m, k, n in self.WORKLOADS]


def _map_flat_v16(raw: dict) -> Tuple[PerfCoeffs, EnergyCoeffs]:
    """Map the main flat v16 calibration.json to search_bench coefficient objects.

    Main schema: cycle-based perf keys at top level (l_*_cy, eff_macs, bw_eff_bpc)
    and energy.params in micro-units (uW, uJ, uJ/B).
    search_bench schema: cycle-based perf (same) + energy in mW/W/uJ/uJ-per-byte.
    """
    perf = PerfCoeffs(
        eff_macs=float(raw["eff_macs"]),
        bw_eff_bpc=float(raw["bw_eff_bpc"]),
        L_SYNC=float(raw["l_sync_cy"]),
        L_PE=float(raw["l_pe_cy"]),
        L_DMA=float(raw["l_dma_cy"]),
        L_STARTUP=float(raw["l_startup_cy"]),
    )
    ep = raw["energy"]["params"]
    energy = EnergyCoeffs(
        P_PE=float(ep["p_pe_uw"]) / 1.0e3,    # uW -> mW
        E_BYTE=float(ep["e_byte_uj_per_byte"]),   # uJ/B (already)
        E_STARTUP=float(ep["e_startup_uj"]),      # uJ
        P_BASE=float(ep["p_base_uw"]) / 1.0e6,    # uW -> W
    )
    return perf, energy


def load_config(
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    result_csv: Path = DEFAULT_RESULT_CSV,
    tc_list_json: Path = DEFAULT_TC_LIST_JSON,
    cpu_energy_csv: Path = DEFAULT_CPU_ENERGY_CSV,
) -> Config:
    """Load configuration from calibration JSON + data file paths.

    Accepts the main flat v16 schema (test/onnx-mlir/data/calibration.json)
    as the single source of truth. Also accepts the legacy {hw, perf, energy}
    nested schema for standalone paper runs.
    """
    with open(calibration_path) as f:
        raw = json.load(f)

    # Flat v16 schema: top-level l_sync_cy etc. + energy.params block
    if "l_sync_cy" in raw and "params" in raw.get("energy", {}):
        perf, energy = _map_flat_v16(raw)
        hw = HWConfig()  # xdna2 defaults are authoritative
        return Config(
            hw=hw, perf=perf, energy=energy,
            result_csv=result_csv, tc_list_json=tc_list_json,
            cpu_energy_csv=cpu_energy_csv,
        )

    # Legacy nested schema {hw, perf, energy}
    def _clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    hw = HWConfig(**_clean(raw.get("hw", {})))
    perf = PerfCoeffs(**_clean(raw.get("perf", {})))
    energy = EnergyCoeffs(**_clean(raw.get("energy", {})))

    return Config(
        hw=hw, perf=perf, energy=energy,
        result_csv=result_csv, tc_list_json=tc_list_json,
        cpu_energy_csv=cpu_energy_csv,
    )
