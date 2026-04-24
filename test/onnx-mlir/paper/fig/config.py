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
DEFAULT_CALIBRATION_PATH = DATA_DIR / "calibration.json"
DEFAULT_RESULT_CSV = DATA_DIR / "result_v14.csv"
DEFAULT_TC_LIST_JSON = DATA_DIR / "tc_list_v14.json"
DEFAULT_CPU_ENERGY_CSV = DATA_DIR / "cpu_npu_gpu_combined_v14.csv"
DEFAULT_SEARCH_BENCH_CSV = DATA_DIR / "T13_search_time.csv"


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
    L_CORE: float = 0.0          # per-core per-iteration barrier (cycles)
    L_DMA: float = 0.0           # per-DMA-descriptor setup (cycles)
    L_STARTUP: float = 0.0       # one-time dispatch overhead (cycles)


@dataclass
class EnergyCoeffs:
    """v16 4-term energy model coefficients."""
    P_CORE: float = 0.0      # aggregate per-core active power (mW)
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
        (32, 32, 32):       "Minimal synthetic square",
        (64, 64, 64):       "Small synthetic square",
        (128, 64, 128):     r"Attention score $QK^\top$ ($\mathrm{seq}=128$, $d_{\mathrm{head}}=64$)",
        (128, 128, 128):    "Small synthetic square",
        (256, 64, 256):     r"Attention score $QK^\top$ ($\mathrm{seq}=256$, $d_{\mathrm{head}}=64$)",
        (256, 256, 256):    "Medium synthetic square",
        (512, 64, 512):     r"Attention score $QK^\top$ ($\mathrm{seq}=512$, $d_{\mathrm{head}}=64$)",
        (32, 768, 768):     r"BERT-base QKV proj. (edge, $\mathrm{seq}=32$, $d_{\mathrm{model}}=768$)",
        (384, 384, 384):    "Medium synthetic square",
        (128, 768, 768):    r"BERT-base QKV proj. ($\mathrm{seq}=128$, $d_{\mathrm{model}}=768$)",
        (512, 512, 512):    "Large synthetic square",
        (128, 1024, 1024):  r"GPT-style QKV proj. ($\mathrm{seq}=128$, $d_{\mathrm{model}}=1024$)",
        (256, 1024, 1024):  r"GPT-style QKV proj. ($\mathrm{seq}=256$, $d_{\mathrm{model}}=1024$)",
        (128, 768, 3072):   r"BERT-base FFN up-proj. ($\mathrm{seq}=128$, $768\to3072$)",
        (128, 3072, 768):   r"BERT-base FFN down-proj. ($\mathrm{seq}=128$, $3072\to768$)",
        # (768, 768, 768):    "BERT hidden",  — excluded
        (256, 1024, 4096):  r"GPT-style FFN up-proj. ($\mathrm{seq}=256$, $1024\to4096$)",
        (1024, 1024, 1024): "Large synthetic square",
        (2048, 2048, 2048): "Very large synthetic square",
    })

    @property
    def workload_labels(self) -> List[str]:
        """Short labels for each workload, e.g. '128×768×768'."""
        return [f"{m}×{k}×{n}" for m, k, n in self.WORKLOADS]


def load_config(
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    result_csv: Path = DEFAULT_RESULT_CSV,
    tc_list_json: Path = DEFAULT_TC_LIST_JSON,
    cpu_energy_csv: Path = DEFAULT_CPU_ENERGY_CSV,
) -> Config:
    """Load configuration from calibration JSON + data file paths."""
    with open(calibration_path) as f:
        raw = json.load(f)

    hw_raw = raw.get("hw", {})
    perf_raw = raw.get("perf", {})
    energy_raw = raw.get("energy", {})

    # Filter out _comment / _note / _model keys
    def _clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    hw = HWConfig(**_clean(hw_raw))
    perf = PerfCoeffs(**_clean(perf_raw))
    energy = EnergyCoeffs(**_clean(energy_raw))

    return Config(
        hw=hw, perf=perf, energy=energy,
        result_csv=result_csv, tc_list_json=tc_list_json,
        cpu_energy_csv=cpu_energy_csv,
    )
