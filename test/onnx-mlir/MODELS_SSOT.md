# Single Source of Truth (SSOT) for Model Formulas

## Overview

All performance and energy cost model formulas are now centralized in `scripts/generate/models.py` as the canonical implementation. This eliminates formula duplication across multiple files and ensures consistency in model calculations.

## Architecture

```
models.py (Single Source of Truth)
├── dma_ops_per_step(spm, spn, num_cores, tp_order)
├── total_data_bytes(M, K, N, elem_bytes, spm, spn, tpm, tpk, tpn, tp_order)
├── PerfModel
│   ├── components_v9() → (T_comp, T_comm, T_overhead)
│   └── predict_v9() → T_total
└── EnergyModel
    ├── predict(model_name, params, features) [dispatcher]
    ├── predict_te() [canonical T-E formula]
    ├── predict_tb(), predict_ta(), ... [legacy models]
    └── predict_*_optim() [scipy-optimized versions]
```

## Delegation Flow

### Cost Model Pipeline
```
cost_model.py (public API)
├── evaluate_candidate()
│   └── perf_overhead() → models.PerfModel.components_v9()
├── perf_comm() → total_data_bytes() → models.total_data_bytes()
└── energy_total_calibrated() → models.EnergyModel.predict()
```

### Calibration Pipeline
```
calibrate_energy.py
├── compute_features() → models.PerfModel.components_v9()
└── predict_* → models.EnergyModel.predict_*_optim()

recalibrate_model.py
└── _dma_ops_per_step() → models.dma_ops_per_step()

analyze_energy_basis.py
└── energy prediction → models.EnergyModel.predict()
```

## Key Models

### Performance Model v9 (Core-Sync)
```
T_total = T_comp + T_comm + L_SYNC*TP + L_SYNC2*P*TP + L_DMA*N_dma*TP + L_STARTUP

Where:
- T_comp = MACs / (eff_macs * P)
- T_comm = bytes / bw_eff_bpc
- TP = temporal product (sum of temporal dimensions)
- P = spatial product (num_cores)
- L_SYNC, L_SYNC2, L_DMA, L_STARTUP = calibration coefficients
- N_dma = DMA operations per iteration
```

### Energy Model T-E (Theoretical + Power)
```
E_total = e_mac*MACs + e_dram*bytes + e_dma*N_dma*TP + e_sync*TP + (p_base + p_core*P)*T_total_us

Where:
- e_mac = energy per MAC (pJ)
- e_dram = energy per DRAM byte (pJ)
- e_dma = energy per DMA operation (uJ)
- e_sync = energy per sync operation (uJ)
- p_base = baseline power (uW)
- p_core = per-core power (uW)
- T_total_us = total time in microseconds
```

## Configuration Loading

`tiling_common.py` extended with energy model support:

```python
CalibCoeffs(
    # Performance coefficients (existing)
    eff_macs=24.28,
    bw_eff_bpc=4.0,
    l_sync_cy=4081,
    l_sync2_cy=1266,
    l_dma_cy=1890,
    l_startup_cy=62303,
    
    # Energy model (new)
    energy_model="T-E",
    energy_params={
        "e_mac_pj": 20.4634,
        "e_dram_pj": 727.4622,
        "e_dma_uj": 0.446966,
        "e_sync_uj": 89.4143,
        "p_base_uw": 1276000.7,
        "p_core_uw": 112583.6
    },
    energy_calibrated=True
)
```

## Data Files

### Current
- `data/calibration.json` — v9 Core-Sync + T-E model (active)

### Legacy
- `data/legacy/calibration_v9_lB.json` — v9 Core-Sync + L-B model (backup)

## Backward Compatibility

### v8 Fallback
When `l_sync2_cy == 0`, the v9 formula automatically falls back to v8 behavior:
```
T_overhead = L_SYNC*TP + L_STARTUP  (v8 path, ignoring L_SYNC2 and L_DMA)
```

### Legacy Analysis Scripts
Scripts designed for v8 data continue to work due to the fallback mechanism.

## How to Update Models

**To update a model formula or coefficient:**

1. Modify only in `scripts/generate/models.py` (appropriate function/class method)
2. Re-run `calibrate.py` or `recalibrate_model.py` to propagate changes
3. All dependent scripts automatically use the updated formula:
   - cost_model.py → tc_list.json
   - calibrate_energy.py → parameter fitting
   - analyze_energy_basis.py → analysis reports

**No changes needed in:**
- cost_model.py (uses models.py)
- calibrate_energy.py (uses models.py)
- analyze_energy_basis.py (uses models.py)
- Or any other analysis script

## Validation

All changes validated in Phase 6:
- ✓ models.py successfully unified from 4 files
- ✓ Numerical results match baseline (pre-refactoring)
- ✓ All scripts import and run without errors
- ✓ T-E energy model parameters correctly loaded
- ✓ End-to-end pipeline functional

## References

- Task 10: "최신 연구 내용에 맞춤 환경 정리" (commit 8691c4be)
- Methodology Section 4: Cost Model Formulation (Notion)
- Phase 6 Validation Report (Task 10 results)
