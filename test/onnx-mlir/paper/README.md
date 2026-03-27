# Paper Data Package

Measurement data and analysis artifacts for energy-efficient spatio-temporal
configuration selection on AMD Ryzen AI NPU (XDNA2).

## Directory Structure

```
paper/
├── README.md              # This file
├── environment.md         # Test environment (hardware, software, methodology)
├── data/
│   ├── result_all.csv     # Unified measurement results (490 configs, 483 PASS)
│   ├── calibration.json   # Cost model coefficients (v7 DMA-add + T-B energy)
│   ├── op_list_paper.json # 19 matrix sizes with source annotations
│   └── tc_list_all.json   # 490 test configurations (symlink to out/)
├── analysis/
│   ├── summary_by_size.csv    # Per-size optimal config summary (19 rows)
│   ├── edp_optimal_cores.csv  # EDP-optimal core count per size
│   ├── model_accuracy.csv     # Cost model accuracy metrics
│   └── cpu_baseline.csv       # CPU comparison results
└── figures/                   # Paper figures (generated)
```

## Dataset Overview

| Property | Value |
|----------|-------|
| Matrix sizes | 19 (9 square + 10 non-square) |
| Total configs | 490 |
| Successful (PASS) | 483 (98.6%) |
| Energy valid | ~480 (99.4%) |
| Deduplicated (for calibration) | 452 |
| Core counts | 4, 8, 12, 16, 24, 32 |
| Data type | bf16 |
| Batch size | 1 |

## Matrix Sizes

### Square
32, 64, 128, 256, 384, 512, 768, 1024, 2048

### Non-square (Transformer layers)
| Size (MxKxN) | DNN Layer |
|--------------|-----------|
| 128x64x128 | Attention head (d_head=64) |
| 256x64x256 | Attention (seq=256) |
| 512x64x512 | Attention (seq=512) |
| 32x768x768 | Edge inference projection |
| 128x768x768 | QKV projection |
| 128x768x3072 | FFN up-projection |
| 128x3072x768 | FFN down-projection |
| 128x1024x1024 | GPT projection |
| 256x1024x1024 | Large projection |
| 256x1024x4096 | Large FFN layer |

## Cost Models

### Performance Model (v7 DMA-add)
```
T = alpha * T_comp + beta * T_comm
  + L_SYNC * TP_total
  + L_DMA * N_dma_per_step * TP_total
  + L_CORE * N_cores + L_STARTUP
```
- Spearman rho = 0.887, MAPE = 17.8%
- EDP core accuracy: 16/19 (84%), mean regret = 8.2%

### Energy Model (T-B)
```
E = E_MAC * MACs + E_DRAM * bytes + (P_BASE + P_CORE * N_cores) * T
```
- Spearman rho = 0.986, MAPE = 29.7%

## Regenerating Results

```bash
# From test/onnx-mlir/ directory:
source ../ironenv/bin/activate
source ../utils/env_setup.sh ../install

# Generate summary tables
python3 scripts/analyze/summarize_for_paper.py \
  --result paper/data/result_all.csv \
  --tc paper/data/tc_list_all.json \
  --calib paper/data/calibration.json \
  --out-dir paper/analysis/

# Run CPU baseline
python3 scripts/analyze/measure_cpu_baseline.py \
  --op paper/data/op_list_paper.json \
  --out paper/analysis/cpu_baseline.csv
```
