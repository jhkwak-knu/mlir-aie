# Cost Model for Spatio-Temporal Parallelization on XDNA2 NPU

## 1. Overview

This document describes the cost model used to estimate the execution time and energy
consumption of matrix multiplication on AMD Ryzen AI NPU (XDNA2). The cost model drives
two key decisions:

1. **Configuration selection** — Choose the optimal spatio-temporal parallelization
   configuration (SPm, SPn, TPm, TPk, TPn, tpOrder) by ranking candidates via
   Energy-Delay Product (EDP).
2. **Calibration** — Fit model coefficients against NPU hardware measurements to
   improve prediction accuracy across matrix sizes.

### Source Files

| File | Role |
|------|------|
| `scripts/generate/cost_model.py` | Cost estimation functions, candidate evaluation |
| `scripts/generate/tiling_common.py` | Shared constants (memory limits, tile shapes) |
| `scripts/analyze/calibrate.py` | Coefficient fitting from NPU measurements |
| `scripts/analyze/analyze_ranking.py` | Spearman rank correlation evaluation |
| `data/calibration.json` | Fitted coefficients (output of calibrate.py) |

---

## 2. Cost Model Formulation

### 2.1 Performance Model (Cycles)

The total execution time for a single matrix multiplication is modeled as:

```
T_total = T_comp + T_dma + T_overhead
```

| Component | Formula | Description |
|-----------|---------|-------------|
| T_comp | M x K x N / (N_cores x R_peak) | Parallel compute time |
| T_dma | D_total / BW_eff | Data transfer time |
| T_overhead | alpha x TP_total | Pipeline drain/fill per temporal step |

Where:
- **N_cores** = SPm x SPn (number of compute tiles)
- **R_peak** = 256 MACs/cycle/core (XDNA2 BF16 peak throughput)
- **BW_eff** = 4 bytes/cycle (NoC stream bandwidth)
- **alpha** = 20 cycles (pipeline overhead constant)
- **TP_total** = TPm x TPk x TPn (total temporal iterations)
- **D_total** = total data transfer volume in bytes (reuse-aware)

### 2.2 Data Transfer Volume (D_total)

D_total depends on the innermost temporal axis (`tpOrder[0]`), which determines
data reuse patterns. The innermost axis operand stays in tile memory and is reused;
outer axis operands must be reloaded each iteration.

| tpOrder[0] | Reused Operand | LHS (A) | RHS (B) | OUT (C) |
|------------|---------------|---------|---------|---------|
| M (axis 0) | RHS | M x K x TPn | K x N | 2 x M x N x TPk |
| N (axis 1) | LHS | M x K | K x N x TPm | 2 x M x N x TPk |
| K (axis 2) | OUT | M x K x TPn | K x N x TPm | 2 x M x N |

```
D_total = (LHS + RHS + OUT) x elem_bytes
```

The factor 2 in the OUT term accounts for both reading partial results (pres) and
writing results back, except when `tpOrder[0] = K` where local accumulation
eliminates pres reads.

### 2.3 Energy Model (pJ)

```
E_total  = E_comp + E_comm + E_static
E_comp   = M x K x N x E_MAC            (E_MAC   = 0.2 pJ/MAC)
E_comm   = D_total  x E_DRAM            (E_DRAM  = 40  pJ/byte)
E_static = N_cores  x P_STATIC x T_total (P_STATIC = 27 pJ/cycle/tile)
```

### 2.4 Optimization Objective

```
EDP = T_total x E_total   (Energy-Delay Product, lower is better)
```

EDP balances speed and energy efficiency. A configuration that is fast but
energy-hungry or slow but efficient will both score poorly.

---

## 3. Calibrated Cost Model

The design-time cost model uses theoretical peak values (R_peak=256, BW_eff=4,
alpha=20). Calibration replaces these with empirically fitted coefficients.

### 3.1 Calibrated Formula (Model D+B, Selected)

```
T_total = T_comp + T_dma + T_sync + T_core + T_startup

T_comp    = (M x K x N) / (N_cores x EFF_MACS)
T_dma     = D_total / BW_EFF
T_sync    = L_SYNC    x TP_total
T_core    = L_CORE    x N_cores
T_startup = L_STARTUP
```

| Parameter | Meaning | How Determined |
|-----------|---------|---------------|
| EFF_MACS | Effective MACs/cycle per core | Median of (TM x TK x TN) / ss_kernel_cy from trace |
| BW_EFF | Effective DMA bandwidth (bytes/cycle) | Fixed at 4.0 B/cycle |
| L_SYNC | Per-temporal-step synchronization cost | OLS regression (dominant term) |
| L_CORE | Per-core fixed setup cost | 2D grid search minimizing MAPE |
| L_STARTUP | One-time initialization cost | 2D grid search minimizing MAPE |

### 3.2 Fitting Procedure

**Input**: result.csv (batch runner output with NPU timing and trace metrics) +
tc_list.json (tiling configuration including tpOrder).

**Phase A — EFF_MACS**:
For each case with kernel trace data (ss_kernel_cy > 0), compute:
```
eff_macs_i = (TM x TK x TN) / ss_kernel_cy
```
Take the median across all cases as the global EFF_MACS.

**Phase B — BW_EFF**:
Fixed at 4.0 bytes/cycle (NoC stream bandwidth) for models A through D+B.
Models E and F attempt to fit this parameter from data (see Section 4).

**Phase C — Overhead Fitting**:
Compute residual for each case:
```
residual = T_actual_cy - T_comp - T_dma
```
where `T_actual_cy = min_us x CLOCK_MHZ` (1500 MHz).

Fit overhead coefficients via OLS regression and/or grid search on the
residuals (see Section 4). For the selected Model D+B, L_SYNC is fitted
via OLS at each grid point, while L_CORE and L_STARTUP are found by 2D
grid search minimizing MAPE.

---

## 4. Model Candidates

Seven candidate models were evaluated. All share the same T_comp formula;
they differ in how overhead and DMA bandwidth are modeled.

### 4.1 Model Definitions

#### Model A — Baseline (1 free variable)
```
T = T_comp + D_total/BW_FIXED + L_SYNC x TP_total
```
Simplest model. One fitted parameter (L_SYNC) captures all per-iteration overhead.
BW is fixed at 4.0 B/cycle.

**Fitting**: 1-variable OLS (no intercept)
```
residual = L_SYNC x tp_total
L_SYNC = sum(tp_total_i x res_i) / sum(tp_total_i^2)
```

#### Model B — Startup Cost (2 free variables)
```
T = T_comp + D_total/BW_FIXED + L_SYNC x TP_total + L_STARTUP
```
Adds a constant startup overhead (intercept term) to capture one-time
initialization cost independent of temporal iterations.

**Fitting**: Standard OLS with intercept (y = ax + b)

#### Model C — Host Call Cost (2 free variables)
```
T = T_comp + D_total/BW_FIXED + L_SYNC x TP_total + L_CONFIG x N_host_calls
```
Separates NPU-internal sync (L_SYNC) from host-side dispatch overhead (L_CONFIG).
N_host_calls = tpOrder[1] axis TP x tpOrder[2] axis TP.

**Fitting**: 2-variable OLS via Cramer's rule (no intercept)

**Universality concern**: N_host_calls depends on the host-side loop structure,
which is implementation-specific rather than algorithm-intrinsic.

#### Model D — Per-Core Setup Cost (2 free variables)
```
T = T_comp + D_total/BW_FIXED + L_SYNC x TP_total + L_CORE x N_cores
```
Adds a per-core fixed cost to capture tile configuration, DMA BD initialization,
and lock setup that scale with the number of active compute tiles.

**Fitting**: 2-variable OLS via Cramer's rule (no intercept)
```
| sum(tp^2)    sum(tp x nc) |   | L_SYNC |   | sum(tp x res) |
| sum(tp x nc) sum(nc^2)    | x | L_CORE | = | sum(nc x res) |
```

#### Model D+B — Per-Core Cost + Startup (3 free variables, Selected)
```
T = T_comp + D_total/BW_FIXED + L_SYNC x TP_total + L_CORE x N_cores + L_STARTUP
```
Combines Model D's per-core cost with Model B's one-time startup overhead.
This decomposition captures three distinct overhead sources:
- **L_SYNC x TP_total**: Per-iteration DMA reconfiguration and synchronization
- **L_CORE x N_cores**: Per-core tile initialization, lock setup, BD programming
- **L_STARTUP**: Core-count-independent one-time cost (NPU wakeup, instruction transfer)

When Model B is fitted alone via OLS, L_STARTUP absorbs the per-core cost
(since both are approximately constant per case), yielding an inflated
intercept. By explicitly separating L_CORE, the startup term shrinks to a
physically reasonable value.

**Fitting**: Hybrid OLS + grid search
1. For each (L_CORE, L_STARTUP) grid point, subtract both from residuals
2. Fit L_SYNC via OLS on the adjusted residuals (always ~34,794)
3. Evaluate full-model MAPE
4. Select the (L_CORE, L_STARTUP) pair minimizing MAPE
5. Two-pass grid: coarse (2K x 10K steps) then fine (500 x 2K steps)

#### Model E — Fitted DMA Bandwidth (2 free variables)
```
T = T_comp + D_total/BW_FIT + L_SYNC x TP_total
```
Instead of fixing BW at 4.0 B/cycle, fits the effective bandwidth from data.
This allows the model to capture DMA inefficiencies (per-transfer setup,
NoC contention, small-transfer penalties).

**Fitting**: 2-variable OLS on y = T_actual - T_comp (no intercept)
```
y = (1/BW_FIT) x data_bytes + L_SYNC x tp_total
```
Variables: inv_bw = 1/BW_FIT, L_SYNC. BW_FIT = 1/inv_bw.

#### Model F — Fitted BW + Per-Core Cost (3 free variables)
```
T = T_comp + D_total/BW_FIT + L_SYNC x TP_total + L_CORE x N_cores
```
Combines Models D and E: fits DMA bandwidth and per-core setup cost simultaneously.

**Fitting**: 3-variable OLS via Cramer's rule (3x3 linear system)
```
| s11  s12  s13 |   | 1/BW  |   | s1y |
| s12  s22  s23 | x | L_SYNC| = | s2y |    where x1=data, x2=tp, x3=N_cores
| s13  s23  s33 |   | L_CORE|   | s3y |
```

### 4.2 Variable Universality

All model variables are grounded in hardware specs or algorithm parameters,
ensuring transferability across NPU generations.

| Variable | Meaning | Source |
|----------|---------|--------|
| EFF_MACS | Per-tile effective throughput | HW vector unit spec |
| BW_EFF / BW_FIT | Effective DMA bandwidth | NoC bandwidth spec |
| L_SYNC | Per-iteration sync overhead | Inherent to any iterative NPU |
| L_CORE | Per-core setup cost | Inherent to any multi-core NPU |
| L_STARTUP | One-time initialization cost | Inherent to any NPU dispatch |
| N_cores | Active compute tiles | Algorithm parameter (SPm x SPn) |
| TP_total | Total temporal iterations | Algorithm parameter (TPm x TPk x TPn) |

Note: Model C's `N_host_calls` variable depends on the host-side dispatch
structure, which may change with implementation. It is excluded from the
universality-preferred set.

### 4.3 Model Selection Criteria

Models are compared on two metrics:
- **Spearman rho**: Rank correlation between predicted and actual times.
  Higher is better. Primary criterion for configuration selection accuracy.
- **MAPE**: Mean Absolute Percentage Error. Lower is better.
  Secondary criterion for absolute prediction accuracy.

Selection rule (prefer simpler model):
```
1. Start with Model A (simplest)
2. Accept a more complex model only if:
   - rho improvement > 0.02, OR
   - rho within 0.02 AND MAPE improvement > 5 percentage points
```

---

## 5. Coefficient Values

### 5.1 Design-Time Coefficients (cost_model.py)

These are the original theoretical values used before calibration:

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| PEAK_MACS (R_peak) | 256 | MACs/cycle/core | XDNA2 BF16 spec (16x16 systolic) |
| BANDWIDTH_BPC (BW_eff) | 4 | bytes/cycle | NoC stream width |
| ALPHA_CYCLES (alpha) | 20 | cycles | Pipeline fill/drain estimate |
| CLOCK_MHZ | 1500 | MHz | Derived from TOPS spec |

### 5.2 Phase 1 Fitted Coefficients (per-size, 2-step regression)

Fitted from individual matrix sizes using 2-step linear regression:
Step 1 — L_CONFIG from spatial-only cases (TP=1,1,1) vs N_cores.
Step 2 — L_SYNC from 4-core temporal cases vs TP_total.

| Size | L_SYNC (cy) | L_CONFIG (cy) | Spearman rho | N |
|------|-------------|---------------|-------------|---|
| 32x32x32 | 55,767 | 9,771 | 0.9479 | 177 |
| 256x256x256 | varies by tpOrder | varies by tpOrder | 0.7276* | 43 |

*Using 32x32 coefficients on 256x256 data.

**Key finding**: Coefficients fitted on 32x32x32 do NOT transfer to 256x256x256.
This motivated the cross-size calibration effort (Phase 2).

### 5.3 Cross-Size Calibrated Coefficients (calibrate.py, 159 cases)

Fitted from 159 cases across 6 matrix sizes (32 to 1024), using the Phase A/C
procedure described in Section 3.2.

#### EFF_MACS (Phase A)

| Metric | Value |
|--------|-------|
| EFF_MACS | 23.15 MACs/cycle |
| N samples | 134 (with valid kernel trace) |
| Range | [0.0, 26.2] MACs/cycle |

Compared to the theoretical peak of 256 MACs/cycle, the effective rate is ~9%.
This reflects the mmul<4,8,8> kernel's 2x2 expansion pattern which achieves
~51 MACs/cycle for the kernel alone, but trace measurement includes kernel
dispatch overhead, reducing the effective rate.

#### Overhead Model Comparison (Phase C)

Models A~D and E~F use unconstrained OLS, which suffers from multicollinearity
between TP_total, N_cores, and data_bytes. Model D+B uses a hybrid approach
(OLS for L_SYNC + MAPE-minimizing grid search for L_CORE and L_STARTUP) to
avoid this issue.

| Model | L_SYNC | L_STARTUP | L_CONFIG | L_CORE | BW_EFF | rho | MAPE |
|-------|--------|-----------|----------|--------|--------|-----|------|
| A | 34,794 | 0 | 0 | 0 | 4.00 (fixed) | 0.9679 | 45.5% |
| B (OLS) | 34,786 | 1,752,318 | 0 | 0 | 4.00 (fixed) | 0.9679 | 244.4% |
| C | 6,412 | 0 | 1,802,550 | 0 | 4.00 (fixed) | 0.9626 | 410.9% |
| D (OLS) | 34,791 | 0 | 0 | 181,239 | 4.00 (fixed) | 0.8876 | 177.1% |
| **D+B** | **34,794** | **76,000** | 0 | **8,500** | **4.00 (fixed)** | **0.9729** | **32.6%** |
| E | 5,620 | 0 | 0 | 0 | 0.05 (fitted) | 0.9037 | 744.5% |
| F | 4,982 | 0 | 0 | -10,215,133 | 0.05 (fitted) | 0.7976 | 10,408.6% |

**Selected: Model D+B** — highest Spearman rho (0.9729) and lowest MAPE (32.6%).

Note: Models B and D with unconstrained OLS produce physically unreasonable
coefficients. The constrained D+B approach finds per-component values that are
both physically interpretable and statistically superior.

#### calibration.json (Final Output)

```json
{
  "version": 2,
  "target": "xdna2",
  "model": "D+B",
  "eff_macs": 23.15,
  "bw_eff_bpc": 4.0,
  "l_sync_cy": 34794,
  "l_startup_cy": 76000,
  "l_config_cy": 0,
  "l_core_cy": 8500,
  "clock_mhz": 1500,
  "fitted_from": {
    "n_samples": 159,
    "spearman_rho": 0.9729,
    "mape_pct": 32.6,
    "ground_truth": "min_us"
  }
}
```

#### Final Coefficient Summary

| Parameter | Value | Unit | Physical Meaning |
|-----------|-------|------|-----------------|
| EFF_MACS | 23.15 | MACs/cycle/core | Effective kernel throughput (includes dispatch overhead) |
| BW_EFF | 4.0 | bytes/cycle | NoC DMA stream bandwidth (fixed) |
| L_SYNC | 34,794 | cycles (23.2 us) | Per-temporal-step DMA reconfiguration + synchronization |
| L_CORE | 8,500 | cycles (5.7 us) | Per-core tile initialization, lock + BD setup |
| L_STARTUP | 76,000 | cycles (50.7 us) | One-time NPU wakeup + instruction transfer |
| CLOCK | 1,500 | MHz | XDNA2 tile clock frequency |

---

## 6. Evaluation Results

### 6.1 Design-Time vs Calibrated (Overall)

| Metric | Design-Time (alpha=20) | Model A | **Model D+B** |
|--------|----------------------|---------|--------------|
| Spearman rho | 0.8965 | 0.9679 (+0.07) | **0.9729** (+0.08) |
| MAPE | 88.1% | 45.5% (-42.6pp) | **32.6%** (-55.5pp) |
| tp=1 MAPE | — | 76.0% | **20.4%** |
| tp>1 MAPE | — | 40.6% | **34.9%** |
| N samples | 159 | 159 | 159 |

Model D+B improves over Model A in both rank ordering (rho +0.005) and
absolute accuracy (MAPE -12.9pp). The most dramatic improvement is in
tp=1 cases (76% to 20%) where L_STARTUP and L_CORE terms now capture
the fixed overhead that Model A cannot represent.

### 6.2 Per-Size Breakdown

| Size | N | Old rho | D+B rho | Old MAPE | A MAPE | D+B MAPE |
|------|---|---------|---------|----------|--------|----------|
| 32x32x32 | 16 | 0.9113 | 0.9142 | 99.0% | 54.1% | **29.9%** |
| 64x64x64 | 27 | 0.8323 | **0.9492** | 97.3% | 57.1% | **27.1%** |
| 128x128x128 | 29 | 0.7807 | **0.8775** | 92.5% | 54.2% | **30.8%** |
| 256x256x256 | 32 | 0.9299 | 0.8579 | 83.5% | 39.3% | **36.5%** |
| 512x512x512 | 28 | 0.9304 | 0.9173 | 81.6% | 37.9% | **35.2%** |
| 1024x1024x1024 | 27 | 0.8549 | **0.9381** | 79.7% | 34.6% | **34.1%** |

**Observations**:
- MAPE is uniformly 27~36% across all sizes (vs 34~57% for Model A).
- Rho improves significantly for small sizes (64x64 +0.12, 128x128 +0.10)
  where the L_CORE term differentiates multi-core configurations.
- 256x256 rho slightly degrades (0.93→0.86) — the K-inner performance
  reversal effect (see Section 7.1) is not captured by any model.

### 6.3 Per-Core-Count Breakdown

| Cores | N | A MAPE | D+B MAPE | Improvement |
|-------|---|--------|----------|-------------|
| 4 | 118 | 43.3% | **34.5%** | -8.8pp |
| 8 | 14 | 52.4% | **31.1%** | -21.3pp |
| 16 | 13 | 48.8% | **21.8%** | -27.0pp |
| 32 | 14 | 53.7% | **32.1%** | -21.6pp |

Multi-core configurations see the largest improvement. The 8~32 core cases
had 48~54% MAPE under Model A because the per-core overhead was entirely
unmodeled. Model D+B reduces this to 22~32%.

### 6.4 Worst-Case Predictions

| Case | Cores | SP | TP | T_pred (us) | T_actual (us) | Error |
|------|-------|----|----|-------------|--------------|-------|
| 156 | 4 | (2,2) | (64,128,32) | 6,122,668 | 3,123,991 | 96.0% |
| 129 | 4 | (2,2) | (32,64,16) | 765,485 | 404,399 | 89.3% |
| 157 | 4 | (4,1) | (32,128,64) | 6,122,668 | 3,272,894 | 87.1% |
| 126 | 8 | (8,1) | (2,4,4) | 2,457 | 10,911 | 77.5% |

The top worst cases are now dominated by **high-TP cases** (156, 129, 157)
where the model overestimates by ~2x. This suggests per-step overhead
decreases at very high iteration counts (DMA BD caching, instruction reuse).
The tp=1 cases (formerly 89% error) have dropped to <70% error.

### 6.5 Analysis of Rejected Models

**Model B (OLS, L_STARTUP)**: Unconstrained OLS yields L_STARTUP = 1.75M
cycles (~1.2 ms), inflated because the intercept absorbs per-core costs.
MAPE = 244%. The same L_STARTUP mechanism works in D+B because L_CORE
separates the core-proportional component.

**Model C (L_CONFIG x N_host_calls)**: N_host_calls is highly correlated
with TP_total, causing multicollinearity. Additionally, N_host_calls is
implementation-dependent (host-side loop structure), reducing universality.

**Model D (OLS, L_CORE x N_cores)**: Unconstrained 2-variable OLS
produces L_CORE = 181,239 (120 us/core, physically unreasonable).
N_cores and TP_total are inversely correlated in the dataset, causing
the OLS to confound per-core cost with per-iteration cost. The
MAPE-minimizing grid search in D+B avoids this by constraining both
variables to physically plausible ranges.

**Models E/F (fitted BW)**: BW_FIT = 0.05 B/cycle — unrealistically low
(80x slower than the theoretical 4 B/cycle). Severe multicollinearity
between `data_bytes` and `tp_total` prevents the OLS solver from
separating DMA time from sync overhead.

---

## 7. Known Limitations and Future Work

### 7.1 Current Limitations

1. **Non-linear temporal overhead**: The per-step cost is not truly constant.
   At very high TP_total (>4096), the model overestimates by ~50% — likely
   due to DMA BD caching and instruction reuse reducing per-step overhead.
   A power-law model (L_SYNC x TP^beta, beta < 1) could capture this,
   but would complicate the cost function for EDP optimization.

2. **Serial compute-DMA**: The current implementation runs compute and DMA
   sequentially (no double buffering). Kernel utilization is 0.3~11%.
   The cost model assumes serial execution, which is correct for now but
   will need revision when double buffering is implemented.

3. **K-inner performance reversal**: For sizes >= 256, K-inner tpOrder
   is empirically slower than M/N-inner despite having fewer DMA ops and
   less total data transfer. Root cause is not fully understood (see
   cost_model_validation.md for investigation details). The cost model
   does not capture this effect.

4. **EFF_MACS variability**: The fitted EFF_MACS (23.15) includes kernel
   dispatch overhead mixed into the trace measurement, making it lower
   than the pure kernel throughput (51.4 MACs/cycle from mmul<4,8,8>).

### 7.2 Future Improvements

1. **Double buffering**: Implementing compute-DMA overlap will fundamentally
   change the cost model from `T = T_comp + T_dma + T_sync` to
   `T = max(T_comp, T_dma) + T_sync`, requiring re-calibration.

2. **Per-tpOrder coefficients**: Fitting separate L_SYNC values for
   K-inner vs M/N-inner could capture the K-inner penalty, at the cost
   of model simplicity.

3. **Piecewise BW model**: Using different BW_eff for small vs large
   transfers could address the DMA-bound case inaccuracy.

4. **Diminishing per-step cost**: Investigate sub-linear L_SYNC behavior
   at high TP_total. If confirmed, a log or power-law term could reduce
   worst-case error from 96% to a more reasonable range.

---

## Appendix A: Execution Environment

- **Target**: AMD Ryzen AI NPU (XDNA2, Strix/Strix Halo)
- **Clock**: 1500 MHz (derived from 100 TOPS / (32 tiles x 256 MACs/tile x 2 / tile_clock))
- **Compute tiles**: Up to 32 (4 per column, 8 columns)
- **Tile memory**: 64 KB per tile (60 KB usable after stack/heap reservation)
- **Kernel**: mmul<4,8,8> with 2x2 expansion (bf16, B transposed)
- **Driver**: amdxdna (timeout=60s)
- **Measurement**: min_us from warmup=3, iterations=10 runs

## Appendix B: Calibration Command

```bash
cd test/onnx-mlir
python3 scripts/analyze/calibrate.py \
  --csv out/calibration/result.csv \
  --tc out/calibration/tc_list.json \
  --output data/calibration.json
```

Output includes:
- Phase A: EFF_MACS with distribution statistics
- Phase C: 7-model comparison table (rho, MAPE, fitted coefficients)
- Per-size validation (rho and MAPE breakdown)
- Top 10 worst-case predictions
