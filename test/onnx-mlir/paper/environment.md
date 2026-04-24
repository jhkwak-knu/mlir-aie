# Test Environment

## Hardware

| Component | Specification |
|-----------|--------------|
| **CPU** | AMD Ryzen AI 9 HX 370 (Strix Point, Zen 5) |
| **CPU Cores** | 12C / 24T, max boost 5.157 GHz |
| **RAM** | 32 GB DDR5 |
| **NPU** | AMD XDNA2 (PCI Device ID 0x17f0, rev 10) |
| **NPU Config** | 8 columns x 4 rows = 32 AIE cores |
| **NPU Core Memory** | 64 KB scratchpad per core |
| **NPU MemTile** | 512 KB per column |
| **NPU Clock** | 1.5 GHz |
| **GPU** | AMD Radeon 890M (integrated) |

## Software

| Component | Version |
|-----------|---------|
| **OS** | Ubuntu 24.04.4 LTS |
| **Kernel** | 6.14.0-37-generic |
| **XRT** | 2.20.0 (build 2025-07-20) |
| **NPU Driver** | amdxdna 2.20.0 |
| **Python** | 3.12.3 |
| **NumPy** | 1.26.4 |
| **SciPy** | 1.17.1 |
| **MLIR-AIE** | Custom fork, branch `energy-cost-model` |
| **aie-opt** | Custom build (2025-03-20) |

## Measurement Configuration

| Parameter | Value |
|-----------|-------|
| **Data Type** | bf16 (bfloat16) |
| **Kernel** | mmul<4,8,8> with 2x2 expansion (24.28 MACs/cycle/core) |
| **Buffering** | Single buffer (double-buffering disabled) |
| **Iterations** | Dynamic (target ~10 ms total per config) |
| **Warmup** | 3 iterations |
| **Time Metric** | host-side `min_us` via XRT |
| **Energy Metric** | RAPL-based, idle-corrected (`npu_energy_per_iter_uj`) |
| **Energy Filter** | `wall_elapsed_s >= 0.005` (RAPL resolution guarantee) |
| **NPU Timeout** | 60 seconds |

## Measurement Methodology

### Time (Performance)
- Each configuration runs `n_warmup` warmup iterations followed by `n_iterations` timed iterations.
- `n_iterations` is dynamically set so that total execution exceeds ~10 ms.
- `min_us` (minimum latency) is used as the ground truth to minimize OS scheduling noise.

### Energy
- System-level energy is measured via Intel RAPL (`/sys/class/powercap/`).
- Idle package power is sampled before each run and subtracted to isolate NPU energy.
- Only runs with `wall_elapsed_s >= 0.005` are considered valid (RAPL temporal resolution).
- Per-iteration energy: `npu_energy_per_iter_uj = (active_energy - idle_energy) / n_iterations`.

### Reproducibility
- All measurement scripts are in `scripts/run/` and `scripts/analyze/`.
- Test case lists are in `out/tc_list_*.json`.
- Raw results are in `out/reports/result_*.csv`.
- Cost model coefficients are in `data/calibration.json` (v7 DMA-add).
