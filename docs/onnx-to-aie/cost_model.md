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

### 5.3 Cross-Size Calibrated Coefficients v2 (calibrate.py, 159 cases)

Initial cross-size calibration (v2). EFF_MACS fitted from trace, overhead via
OLS + grid search. Ground truth: min_us.

| Model | L_SYNC | L_STARTUP | L_CORE | BW_EFF | rho | MAPE |
|-------|--------|-----------|--------|--------|-----|------|
| **D+B** | **34,794** | **76,000** | **8,500** | **4.00** | **0.9729** | **32.6%** |

### 5.4 Trace-Based Measurement Analysis

NPU trace 데이터의 신뢰도를 체계적으로 분석하여 v5 캘리브레이션의 기반을 마련했다.

#### 5.4.1 Host NPU Time (min_us, avg_us)

Host chrono로 측정하는 `runKernel()` 실행 시간. 10회 측정, 3회 warmup.

| Size | N | (max-min)/avg | min/avg | 판정 |
|------|---|---------------|---------|------|
| 32 | 16 | 51.4% | 0.863 | 짧은 실행 시간으로 인한 자연적 분산 |
| 64 | 27 | 41.6% | 0.801 | |
| 128 | 29 | 41.8% | 0.808 | |
| 256 | 32 | 23.4% | 0.925 | 양호 |
| 512 | 28 | 7.7% | 0.977 | 우수 |
| 1024 | 27 | 5.7% | 0.967 | 우수 |

**결론**: min_us, avg_us 모두 신뢰 가능. 소규모 행렬의 높은 분산은 실행 시간이 짧아
시스템 노이즈가 지배하기 때문이며, 비용 모델 ground truth로 적합.

#### 5.4.2 Trace Time (dispatch_cy, ss_iter_cy, matmul_npu_us)

NPU trace에서 추출한 시간 지표들의 신뢰도 분석.

**hw_timer (dispatch_cy)**: BROADCAST_15 Start 명령의 하드웨어 카운터값.

```
dispatch N Start → [NPU 실행] → Stop → [host idle: memcpy+sync] → dispatch N+1 Start
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        hw_timer가 이 구간도 포함!
```

실증 데이터로 확인:

| Case | dispatch_us | min_us | dispatch < min? | idle_us |
|------|-------------|--------|-----------------|---------|
| 001 (32x32) | 93.8 | 114.7 | YES | 72.2 |
| 084 (256x256) | 620.4 | 578.6 | **NO** | 56.8 |

dispatch_cy에 host turnaround (~56-72 us)이 포함되므로 NPU-only 시간으로 사용 불가.

**ss_iter_cy (kernel-to-kernel gap)**: Dispatch 내 연속 커널 시작 간격.
trace 이벤트 timestamp 기반으로 NPU-only에 가깝지만, `ss_iter * ipd * hs`로
matmul_npu_us를 계산하면 outlier가 평균을 왜곡하는 문제가 있다.

**ss_kernel_cy**: 커널 이벤트 페어링 (start -> end). 가장 신뢰할 수 있는 지표.

| 지표 | NPU-only? | 신뢰도 | 용도 |
|------|-----------|--------|------|
| min_us | 아니오 (XRT overhead 포함) | 높음 | Ground truth |
| dispatch_cy | 아니오 (host idle 포함) | 중간 | 참고용 |
| ss_iter_cy | 예 (근사) | 중간 | 구성요소 분석 |
| ss_kernel_cy | 예 (정확) | 높음 | EFF_MACS 직접 측정 |

#### 5.4.3 _group_boundaries() 버그 수정

`apply_timestamp_corrections()`에서 dispatch 경계를 감지하는 `_group_boundaries()` 함수에
hw_timer 값 비교 로직 버그가 있었다.

**문제**: 연속 dispatch의 hw_timer 값이 유사하면 (실행 시간이 비슷한 반복 dispatch)
1000-cycle threshold에 의해 서로 다른 dispatch가 하나로 병합됨.

```
tile(2,1) hw_timer 값:
[1]  118,710  ← dispatch 0
[2]  416,147  ← dispatch 1 (outlier)
[3]  118,726  ← dispatch 2
[4]  119,457  ← |119457-118726|=731 < 1000 → dispatch 2에 병합! (BUG)
...
[10] 118,706  ← 병합
```

결과: 10개 dispatch → 4개 그룹 → ss_iter가 3개 값으로만 계산 → outlier 과대영향.

**수정**: hw_timer 값 대신 accumulated timer 간격으로 비교.
같은 dispatch 경계의 중복 Start만 병합하고, 다른 dispatch의 Start는 분리.

```python
# Before (bug): hw_timer 값 비교
if not groups or abs(hw - groups[-1][1]) > 1000:

# After (fix): accumulated timer 간격 비교
if not groups or (acc - groups[-1][0]) > 1000:
```

#### 5.4.4 EFF_MACS Trace 직접 측정

ss_kernel_cy로부터 순수 커널 효율을 직접 측정:

```
EFF_MACS = (TM x TK x TN) / ss_kernel_cy
```

| 필터 | N | EFF_MACS median |
|------|---|-----------------|
| 전체 | 159 | 23.31 |
| dq >= 0.5 | 136 | **24.28** |
| dq >= 0.75 | 122 | 24.25 |
| dq >= 0.99 | 103 | 24.22 |

dq >= 0.5에서 안정적으로 수렴. 타일 크기별 편차가 있으나 (TK=8: ~11, TK=128: ~25),
단일 대표값으로 24.28 MACs/cy를 사용.

#### 5.4.5 구성요소 분해 결과

trace에서 확인된 실행 시간 구성 비율 (min_us 기준):

| Size | Compute% | DMA% | Overhead% | OH_us |
|------|----------|------|-----------|-------|
| 32 | 0.2% | 53.9% | 46.0% | 75.6 |
| 64 | 0.8% | 69.2% | 28.9% | 78.5 |
| 128 | 4.6% | 70.4% | 20.7% | 59.1 |
| 256 | 23.0% | 41.9% | 25.5% | 93.7 |
| 512 | 22.2% | 54.7% | 15.3% | 310.3 |

소규모 행렬은 DMA + overhead가 지배적이고, 256+ 크기에서 compute 비중이 20%+.

### 5.5 Cross-Size Calibrated Coefficients v5 (Current)

Trace 분석 결과를 반영한 최종 캘리브레이션. v2 대비 변경점:

1. **EFF_MACS = 24.28 (trace 고정)**: ss_kernel_cy에서 직접 측정한 값을 fitting 대신
   고정값으로 사용. 커널 dispatch overhead가 제거되어 v2의 23.15보다 정확.

2. **archive tc.json 사용**: tc_list.json 재생성으로 인한 불일치 해결. archive_v3의
   per-case tc.json에서 tpOrder를 로드.

3. **3단계 ultra-fine 그리드 서치**: coarse(2000/10000) → fine(500/2000) →
   ultra-fine(100/500)로 L_CORE/L_STARTUP 최적화 정밀도 향상.

4. **Ground truth = min_us**: 모든 159건 사용 (trace quality 필터 불필요).

#### Coefficient Evolution (v2 → v5)

| Parameter | v2 | v5 | Change | 원인 |
|-----------|------|------|--------|------|
| EFF_MACS | 23.15 (fitted) | **24.28** (trace fixed) | +4.9% | 순수 커널 효율 직접 측정 |
| L_SYNC | 34,794 | **34,532** | -0.8% | EFF_MACS 변경에 따른 잔차 재조정 |
| L_CORE | 8,500 | **7,700** | -9.4% | ultra-fine grid로 정밀 최적화 |
| L_STARTUP | 76,000 | **46,500** | -38.8% | 동일 |

#### calibration.json v5

```json
{
  "version": 5,
  "model": "D+B",
  "eff_macs": 24.28,
  "bw_eff_bpc": 4.0,
  "l_sync_cy": 34532,
  "l_startup_cy": 46500,
  "l_core_cy": 7700,
  "clock_mhz": 1500,
  "fitted_from": {
    "n_samples": 159,
    "spearman_rho": 0.9765,
    "mape_pct": 35.0,
    "ground_truth": "min_us",
    "eff_macs_source": "trace ss_kernel_cy median (136 cases, dq>=0.5)",
    "eff_macs_fixed": true,
    "tc_source": "archive_v3"
  }
}
```

#### Final Coefficient Summary

| Parameter | Value | Unit | Physical Meaning |
|-----------|-------|------|-----------------|
| EFF_MACS | 24.28 | MACs/cycle/core | Pure kernel throughput (trace-measured, no dispatch overhead) |
| BW_EFF | 4.0 | bytes/cycle | NoC DMA stream bandwidth (fixed) |
| L_SYNC | 34,532 | cycles (23.0 us) | Per-temporal-step DMA reconfiguration + synchronization |
| L_CORE | 7,700 | cycles (5.1 us) | Per-core tile initialization, lock + BD setup |
| L_STARTUP | 46,500 | cycles (31.0 us) | One-time NPU wakeup + instruction transfer |
| CLOCK | 1,500 | MHz | XDNA2 tile clock frequency |

---

## 6. Evaluation Results (v5)

### 6.1 Overall Metrics

| Metric | Design-Time | D+B v2 | **D+B v5** |
|--------|-------------|--------|------------|
| Spearman rho | 0.8965 | 0.9729 | **0.9765** |
| MAPE | 88.1% | 32.6% | **35.0%** |
| MdAPE | — | — | **33.2%** |
| Bias% | — | — | +12.1% |
| Top-1 selection | — | — | **3/6 (50%)** |
| Top-3 selection | — | — | **6/6 (100%)** |
| N samples | 159 | 159 | 159 |

v5는 v2 대비 rho 개선 (+0.004)되었으나 MAPE가 약간 증가 (+2.4pp).
이는 EFF_MACS를 trace 직접 측정값(24.28)으로 고정한 결과이며, T_comp의 정확도가
향상된 대신 overhead 항이 더 많은 분산을 흡수하기 때문이다.

**핵심 개선**: Top-3 selection 100% 달성 — 모든 행렬 크기에서 모델이 예측한
최적 config이 실측 Top-3 안에 포함된다.

### 6.2 Per-Size Breakdown

| Size | N | rho | MAPE | MdAPE | Bias% |
|------|---|-----|------|-------|-------|
| 32x32x32 | 16 | 0.9098 | 30.4% | 26.7% | -19.0% |
| 64x64x64 | 27 | **0.9722** | 24.5% | 22.0% | -5.8% |
| 128x128x128 | 29 | **0.9774** | 27.4% | 28.3% | +3.9% |
| 256x256x256 | 32 | **0.9826** | 37.7% | 32.5% | +28.7% |
| 512x512x512 | 28 | 0.9186 | 40.5% | 37.6% | +24.1% |
| 1024x1024x1024 | 27 | 0.9072 | 47.1% | 52.0% | +24.8% |

**Observations**:
- 64~256 크기에서 rho 0.97+ (v2 대비 크게 개선)
- 소규모 행렬(32, 64)은 under-predict (bias -19%, -6%): overhead 비중이 높아
  모델이 실제보다 낮게 예측
- 대규모 행렬(256+)은 over-predict (bias +24~29%): T_dma/T_sync 과대추정

### 6.3 Per-Core-Count Breakdown

| Cores | N | MAPE | Bias% |
|-------|---|------|-------|
| 4 | 118 | 40.9% | +18.6% |
| 8 | 14 | **15.2%** | +5.8% |
| 16 | 13 | **14.8%** | -9.2% |
| 32 | 14 | 23.0% | -17.1% |

8~16 코어에서 MAPE 15% 수준으로 우수한 정확도.

### 6.4 Optimal Configuration Selection

| Size | N | Top-1 | Top-3 | Regret% |
|------|---|-------|-------|---------|
| 32x32x32 | 16 | no | **YES** | +10.4% |
| 64x64x64 | 27 | no | **YES** | +14.1% |
| 128x128x128 | 29 | no | **YES** | +12.8% |
| 256x256x256 | 32 | **YES** | **YES** | 0.0% |
| 512x512x512 | 28 | **YES** | **YES** | 0.0% |
| 1024x1024x1024 | 27 | **YES** | **YES** | 0.0% |

Top-1: 3/6 (50%), **Top-3: 6/6 (100%)**

256+ 크기에서 Top-1 정확도 100%. 소규모에서의 10~14% regret은 overhead 지배적
환경에서 config 간 차이가 작기 때문이다.

### 6.5 Component Dominance

| Size | Compute% | DMA% | Overhead% |
|------|----------|------|-----------|
| 32 | 0.2% | 1.4% | **98.5%** |
| 64 | 0.9% | 3.6% | **95.6%** |
| 128 | 5.1% | 10.8% | 84.1% |
| 256 | 14.6% | 18.6% | 66.8% |
| 512 | 15.4% | 24.7% | 59.9% |
| 1024 | 15.8% | 27.0% | 57.2% |

소규모 행렬에서는 T_overhead가 98%+를 차지하여 L_SYNC/L_CORE/L_STARTUP의 정확도가
전체 예측을 결정한다. 256+ 크기에서 compute/DMA 비중이 증가하며 EFF_MACS와 BW_EFF의
역할이 커진다.

---

## 7. Known Limitations and Future Work

### 7.1 Current Limitations

1. **Non-linear temporal overhead**: The per-step cost is not truly constant.
   At very high TP_total (>4096), the model overestimates by ~50% — likely
   due to DMA BD caching and instruction reuse reducing per-step overhead.

2. **Serial compute-DMA**: The current implementation runs compute and DMA
   sequentially (no double buffering). Kernel utilization is 0.3~11%.
   The cost model assumes serial execution, which is correct for now but
   will need revision when double buffering is implemented.

3. **EFF_MACS tile-size dependency**: Single EFF_MACS (24.28) cannot capture
   tile-size-dependent kernel efficiency. TK=8 tiles achieve only ~11 MACs/cy
   while TK=128+ achieves ~25 MACs/cy. This causes systematic under/over-
   prediction for specific tile configurations.

4. **Trace matmul_npu_us limitations**: Trace-derived NPU-only time has
   structural issues: (a) dispatch_cy includes host turnaround, (b) ss_iter
   outliers distort mean-based aggregation, (c) matmul_npu_us is unsuitable
   as calibration ground truth. min_us remains the reliable ground truth.

5. **Large-size over-prediction bias**: 256+ sizes have +24~29% systematic
   bias. T_dma model (fixed 4 B/cy) may overestimate for large contiguous
   transfers that achieve higher effective bandwidth.

### 7.2 Future Improvements

1. **Double buffering**: Implementing compute-DMA overlap will change
   `T = T_comp + T_dma + T_sync` to `T = max(T_comp, T_dma) + T_sync`.

2. **EFF_MACS(TK) function**: Per-TK kernel efficiency to capture tile-size
   dependency. Could reduce MAPE significantly for mixed tile configurations.

3. **Trace-based overhead decomposition**: Use per-dispatch fixed overhead
   (~115K cy from trace) to refine overhead model structure (separate
   per-dispatch from per-iteration overhead).

4. **Piecewise BW model**: Size-dependent DMA bandwidth to address large-size
   over-prediction bias.

---

## Appendix A: Execution Environment

- **Target**: AMD Ryzen AI NPU (XDNA2, Strix/Strix Halo)
- **Clock**: 1500 MHz (derived from 100 TOPS / (32 tiles x 256 MACs/tile x 2 / tile_clock))
- **Compute tiles**: Up to 32 (4 per column, 8 columns)
- **Tile memory**: 64 KB per tile (60 KB usable after stack/heap reservation)
- **Kernel**: mmul<4,8,8> with 2x2 expansion (bf16, B transposed)
- **Driver**: amdxdna (timeout=60s)
- **Measurement**: min_us from warmup=3, iterations=10 runs

## Appendix B: Calibration Commands

```bash
cd test/onnx-mlir

# v5 calibration (trace-fixed EFF_MACS + archive tpOrder)
python3 scripts/analyze/calibrate.py \
  --csv out/calibration/result_v8_trace.csv \
  --tc-archive out/calibration/archive_v3 \
  --ground-truth min_us \
  --model D+B \
  --fix-eff-macs 24.28 \
  --output data/calibration.json

# Validation with charts
python3 scripts/analyze/validate_perf_model.py \
  --csv out/calibration/result_v8_trace.csv \
  --tc-archive out/calibration/archive_v3 \
  --calib data/calibration.json \
  --ground-truth min_us \
  --plot out/calibration/plots_v8_final
```

### Source Files

| File | Role |
|------|------|
| `scripts/analyze/calibrate.py` | D+B coefficient fitting (7-model comparison) |
| `scripts/analyze/validate_perf_model.py` | 9-section validation report + charts |
| `scripts/analyze/analyze_trace.py` | NPU trace analysis, ss_iter/dispatch timing |
| `scripts/analyze/reanalyze_batch.py` | Batch re-analysis of archived trace data |
| `scripts/analyze/predict_trace_events.py` | Trace data quality scoring |
| `scripts/analyze/analyze_perf.py` | 9-section performance analysis report |
| `data/calibration.json` | Fitted coefficients (v5) |
