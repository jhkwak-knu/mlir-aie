# NPU Reconfiguration 측정 처리 분석 노트

## Context

논문에 사용된 최신 실험 환경/데이터에서 **NPU reconfiguration(재구성) 비용이 성능 측정(시간·에너지)과
calibration, 그리고 모델 단위 실험에 어떻게(포함/제외) 반영되는지**를 코드 레벨로 파악한 결과를
정리한 문서다. 논문 원고와 나란히 놓고 서술 방향(포함으로 쓸지, caveat로 쓸지)을 논의하기 위한 자료.

논문에 포함된 실험은 두 종류:
- **kernel-level**: per-op(단일 matmul) 마이크로벤치 -- `result_v14.csv`
- **모델 단위(end-to-end)**: MLP(MNIST), BERT-Small

---

## 0. 실험 환경 (paper/environment.md)

### Hardware
- CPU: AMD Ryzen AI 9 HX 370 (Strix Point, Zen 5), 12C/24T, max 5.157 GHz
- RAM: 32 GB DDR5
- NPU: AMD XDNA2 (PCI 0x17f0, rev 10), 8 columns x 4 rows = 32 AIE cores, core scratchpad 64 KB,
  MemTile 512 KB/column, clock 1.5 GHz
- GPU: AMD Radeon 890M (integrated)

### Software
- OS Ubuntu 24.04.4 LTS, Kernel 6.14.0-37-generic
- XRT 2.20.0 (2025-07-20), amdxdna 2.20.0
- Python 3.12.3, NumPy 1.26.4, SciPy 1.17.1
- MLIR-AIE custom fork, branch `energy-cost-model`, aie-opt 2025-03-20

### 측정 공통 설정
- Data type: bf16, Kernel: mmul<4,8,8> 2x2 expansion (24.28 MACs/cycle/core)
- Single buffer (double-buffering disabled)
- 시간 지표: host-side `min_us` (XRT), 에너지: RAPL package, idle-corrected (per-iteration)
- Energy filter: `wall_elapsed_s >= 0.005`

---

## 1. 최신 실험 데이터 버전

| 항목 | 파일 / 값 |
|---|---|
| kernel-level 원시 데이터 | `test/onnx-mlir/out/reports/result_v14.csv` (1290 PASS rows, 18 shapes) |
| kernel-level figure 데이터 | `test/onnx-mlir/paper/fig/data/result_v14.csv` |
| NPU vs CPU/GPU (per-op) | `paper/fig/data/cpu_npu_combined_v14.csv`, `cpu_npu_gpu_combined_v14.csv` (각 19 workload) |
| calibration 계수 | `test/onnx-mlir/data/calibration.json` = **v16**, `_data_base: result_v14.csv` |
| 모델 단위 실험 (별도 폴더) | `mlir-aie/experiments/` |
| - MLP(MNIST) | `experiments/results/mlp_512_512_bs32/` (784-512-512-16, bs 32) |
| - BERT-Small | `experiments/results/bert_small_L128_bs1/` (4 layers, d=512, ffn=2048, seq=128, 최신 2026-04-25) |

> 즉 측정 데이터는 v14, cost model 계수는 그 v14에 피팅된 v16.

---

## 2. Reconfiguration 개념: 두 가지를 구분

`init_xrt_load_kernel` (`runtime_lib/test_lib/test_utils.cpp:124-165`) 기준:
`device(0)` -> `xrt::xclbin(file)` -> `register_xclbin` -> `xrt::hw_context(device, uuid)` -> `xrt::kernel(context, name)`

### (a) 콜드 xclbin 로드/등록 -- 1회성
| 단계 | 성격 |
|---|---|
| `xrt::xclbin(file)` | 파일 I/O (소프트웨어) |
| `register_xclbin` | 드라이버 등록/검증 (소프트웨어) |
| **`xrt::hw_context(device, uuid)`** | **NPU 컬럼 파티션 예약 + 구성(CDO) 컨테이너 확립** |
| `xrt::kernel(...)` | 핸들 (소프트웨어) |

- "순수 소프트웨어 준비"는 부분적으로만 맞다: 파일·등록·핸들은 소프트웨어지만, `hw_context` 생성은
  NPU 구성을 확립하는 단계다.
- AIE 어레이에 CDO를 물리적으로 써넣는 시점이 hw_context 생성 때(eager)인지 첫 submit 때(lazy)인지는
  드라이버 내부 동작이라 코드로 단정 불가. **어느 쪽이든 이 코드에서는 warmup 중에 끝나 측정에서 제외됨.**

### (b) 런타임 레이어 전환 재구성 -- 반복적
- 여러 hw_context가 동시에 살아 있고, NPU 물리 어레이(8열)는 동시에 올릴 수 있는 구성이 제한적.
- 레이어 A 다음 다른 xclbin의 레이어 B를 submit할 때 B 구성이 컬럼에 상주하지 않으면 드라이버가
  B의 CDO를 다시 써넣어(reprogram) 스와핑 후 실행. 파일 I/O·등록 없이 **어레이 재프로그래밍만** 반복.
- `ctx.kernel(...)` submit 시점에 발생, `run.wait()`가 완료까지 블록.

**사용자가 말한 reconfiguration = (b) 런타임 전환에 가까움.**

> Caveat: (b)가 매 전환마다 실제로 전체 재프로그램을 유발하는지, 두 구성이 동시 상주해 스위칭이 거의
> 공짜인지는 드라이버 스케줄링에 달렸고 코드로 단정 불가. 확실한 건 "발생 시 어느 창에 잡히느냐".

---

## 3. kernel-level 실험 (result_v14) 측정 구조

호스트: `test/onnx-mlir/src/host.cpp`

### 콜드 로드/등록은 측정 루프 밖 1회
- device open + xclbin 로드 + register + hw_context + kernel: `host.cpp:162-164` (`init_xrt_load_kernel`)
- buffer object 할당: `host.cpp:167-192`
- instruction sequence 로드/업로드: `host.cpp:152-153`, `197-199`
- instruction BO sync (1회): `host.cpp:315-316` ("Sync instruction buffer once")

### 측정 구간 (`runOneIteration`, host.cpp:488-505)
```
memcpy(A,B) -> sync -> runKernel()[dispatch+wait]
  min_us      = runKernel의 t1-t0 = dispatch + wait만 (498-501)
  step_min_us = memcpy + sync + dispatch + wait (493-502)
```
- 에너지 RAPL 창: batch inner loop만 감쌈 (`host.cpp:557` start ~ `600` end)
- 에너지·시간 모두 per-iteration 평균 (`batch_energy_per_iter = npu_uj / n_inner`, `host.cpp:608`)
- warmup은 batch loop 밖, cold-start 폐기 (`host.cpp:509-519`)

### reconfiguration 처리
- **테스트 케이스 1개 = config 1개 = xclbin 1개.** 단일 커널을 n_inner번 반복 실행.
- 반복 사이 구성이 바뀌지 않음 -> **런타임 전환(b)이라는 사건 자체가 없음** (제외가 아니라 부재).
- 콜드 로드(a)는 루프 밖 1회 + per-iteration amortize로 제외.
- cost model의 `L_STARTUP`(45086 cy ~30us) / `E_STARTUP`(278.7 uJ)은 **dispatch당 startup**
  (sync barrier + DMA 파이프 채움)이지 전환 비용이 아님 (반복 내내 구성 동일).

**결론: kernel-level 값 = NPU가 이미 그 연산에 맞게 구성된 상태의 순수 steady-state per-op 비용.
전환 비용 없음.**

---

## 4. Calibration (calibration.json v16)

- perf: `eff_macs=24.28`, `bw_eff_bpc=4.0`, `L_SYNC=4200`, `L_CORE=1896`, `L_DMA=3109`, `L_STARTUP=45086` cy
- energy: `P_CORE=17.9mW`, `E_BYTE=0.0012uJ/B`, `E_STARTUP=278.7uJ`, `P_BASE=10.85W`
- 피팅 대상 컬럼: 시간 `batch_min_avg_us`(dispatch+wait per-iter), 에너지 `batch_min_energy_per_inference_uj`
  (`v16_fit_candidates.py:113-114`) -- 둘 다 reconfiguration을 amortize/제외한 값.
- **`l_config_cy: 0`** -- 재구성(config) 항을 넣을 스키마 슬롯이 있으나 명시적으로 0 (미사용).
- 적용부 `paper/fig/cost_model.py:171-208`: `t_overhead = L_SYNC*TP + L_CORE*P*TP + L_STARTUP`,
  `E = P_CORE*P*t + E_BYTE*bytes + E_STARTUP + P_BASE*t`.

**결론: calibration에 reconfiguration 항 없음. L_STARTUP/E_STARTUP은 per-dispatch startup.**

---

## 5. 모델 단위 실험 (MLP, BERT-Small) 측정 구조

- 오케스트레이터: `experiments/scripts/measure.py` (자체 측정 안 함, runner subprocess 호출만;
  Python 타이머 `measure.py:179-186`는 로깅용, 보고값 아님)
- 실측: 컴파일된 runner (`src/mlp_runner/`, `src/distilbert_runner/`)
- 공유 측정 코드: `src/mlp_runner/measurement.cpp`, `xrt_dispatcher.cpp`, `rapl.*`
- config: `warmup_iterations=10`, `outer_batches=5`, `inner_target_seconds=0.5`,
  RAPL package+core (`configs/bert_small_L128_bs1.json` measurement 블록)

### 콜드 로드는 캐시 + warmup으로 측정 제외
- `XrtDispatcher::getOrLoad` (`xrt_dispatcher.cpp:98-165`)는 **xclbin_path 기준 캐시**.
  캐시 미스일 때만 `init_xrt_load_kernel` (129-137)로 로드. 캐시 히트는 즉시 반환 (100-101).
- kernel/insts는 사전 컴파일 (`compile_kernels.py`), 레이어별 xclbin.
- warmup 10회가 전 레이어 dispatch -> 모든 xclbin이 측정 배치 전에 캐시됨.
- **-> 디스크->NPU 콜드 로드/등록 비용은 모든 측정에서 제외.**

### 측정 구간 (`instrumentedForward`, measurement.cpp:32-93)
- 모델 시간: `t0`(44, layer loop 전) ~ `t1`(86, softmax 후) -> `sample.time_us`(89).
  **forward pass 전체의 실제 wall-clock** (모든 레이어 dispatch + ReLU + softmax + 레이어 간 간격 포함)
- per-layer 시간: `lt0`(53) ~ `dispatch(L,...)`(55) ~ `lt1`(57) -> `ps.time_us`(63).
  dispatch 전체 (bf16 변환 + staging + memcpy + sync + runKernel + decode)를 감쌈
- 모델 에너지: batch RAPL 창 (`measurement.cpp:127` `batch_e0` ~ `155` `batch_e1`),
  idle 차감 후 `/n_inner` (167-174). per-layer 에너지: `l0`~`l1` RAPL delta (51-59, 64)
- 모델 총량 = **직접 측정한 whole-model wall / whole-batch energy** (레이어 합 아님).
  레이어 합은 host 잔차·gemm fraction 계산에만 사용.

### reconfiguration 처리 (런타임 전환)
- 각 레이어 `ctx.kernel(...)+run.wait()` (`xrt_dispatcher.cpp:261-270, 296`)가
  per-layer 타이머·모델 타이머·RAPL 창 **모두 안**에 위치.
- **-> 런타임 전환(b)이 발생한다면 그 시간·에너지는 측정 창 안에 구조적으로 포함되고,
  각 레이어 dispatch(= Kernel 파트)와 모델 총량에 반영됨. 매 추론마다 발생하므로 amortize 안 됨.**

### 논문 분해 (`publication_plots.py:174-224`, `figure_te_decomposition`)
```
모델 시간 = Host overhead + Kernel(sum of per-layer dispatches)
  t_kernel = sum(kernel_rows time_us_min)          (212)
  t_host   = max(0, t_model - t_kernel)            (213)  <- ReLU/softmax/layernorm 등 non-GEMM
  에너지는 시간 비례 배분: e_host = e_model * t_host/t_model  (214)  <- per-layer RAPL 너무 거침
```
- 런타임 전환 비용은 (발생 시) **Kernel 파트에 흡수**, Host overhead 아님.
- "Host overhead"는 `analyze.py:216`에 명시: "ReLU, softmax, dispatch latency" 등 non-GEMM.

### 결과 파일 컬럼 (measurements.csv, measurement.cpp:240-245)
```
measurement_type(model|kernel), setter, backend, layer, batch_idx, n_inner,
time_us_min, time_us_mean, energy_uj_min_package, wall_s,
idle_power_pre_mw, idle_power_post_mw, batch_energy_cv_pct, batch_time_cv_pct
[+ gemm_type, layer_idx, sub_type  (bert/distilbert)]
```
- **reconfiguration / xclbin-load / switch 전용 컬럼 없음.**

---

## 6. 종합 표

| 항목 | 콜드 로드/등록 (a) | 런타임 레이어 전환 (b) |
|---|---|---|
| kernel-level (result_v14) | 루프 밖 1회 + amortize, **제외** | **부재** (단일 커널 반복, 전환 없음) |
| MLP / BERT-Small (모델) | warmup 캐시, **제외** | **측정 창 안에 포함** (Kernel 파트에 흡수, 모델 총량 반영) |
| calibration (v16) | 미포함 | 미포함 (`l_config_cy: 0`) |
| 전용 정량화 컬럼/항 | 없음 | 없음 |

---

## 7. 증거의 성격 (중요)

런타임 전환이 kernel에 포함된다는 것에 대해 두 가지를 구분:

1. **측정 창이 전환을 구조적으로 감싼다 -> 코드로 증명 가능 (명확).**
   per-layer 타이머(measurement.cpp:53-63)와 RAPL 창이 `run.wait()`(xrt_dispatcher.cpp:261-270)을
   포함 -> submit~완료 사이 드라이버 동기 재구성은 무엇이든 잡힘.

2. **전환 비용이 실제로 0이 아니게 발생/크기 -> 분리 입증 안 됨.**
   전용 컬럼·카운터 없음. 매 전환 재프로그램 여부는 하드웨어/드라이버 동작이라 코드로 확인 불가.
   "모델 in-context per-layer" vs "result_v14 격리" 비교도 host측 변환/decode가 지배해 깨끗한 분리 불가.

**정확한 서술: "런타임 전환 비용은 측정 창에 구조적으로 포함되나(발생 시 모델 총량·Kernel 파트 반영),
현재 실험은 그것을 분리 측정하거나 크기를 입증하지 않는다."**

---

## 8. 추가 Caveat

- **에너지는 RAPL package(CPU 패키지) 에너지 - idle** 이지 직접 NPU 전력 레일이 아님.
  NPU 측 reconfiguration 에너지는 host/package 활동을 유발하는 만큼만 반영됨.
- kernel-level `min_us`(dispatch+wait)와 모델 runner의 per-layer 시간(host 변환+decode 포함)은
  스코프가 달라 직접 비교 불가.
- (DistilBERT는 논문 범위 밖이나 참고: per-GEMM 타이머가 A/B bf16 변환 제외, NPU 오프로드는 4 GEMM 종류만,
  attention_score/context·classifier head(M=1)는 CPU ggml 폴백.)

---

## 핵심 파일:라인 인덱스

- 환경: `test/onnx-mlir/paper/environment.md`
- kernel-level 측정: `test/onnx-mlir/src/host.cpp:152-199, 315-316, 488-505, 545-648`
- calibration: `test/onnx-mlir/data/calibration.json`, `paper/fig/cost_model.py:171-208`,
  `scripts/analyze/v16_fit_candidates.py:113-114`
- 콜드 로드 구현: `runtime_lib/test_lib/test_utils.cpp:124-165`
- 모델 측정: `experiments/scripts/measure.py:179-186`, `src/mlp_runner/measurement.cpp:32-93, 121-231, 240-245`
- NPU dispatch/캐시: `src/mlp_runner/xrt_dispatcher.cpp:98-165, 261-270, 337-341`
- 모델 분해: `experiments/scripts/analyze.py:15-23, 216`, `scripts/publication_plots.py:174-224`
- config: `experiments/configs/bert_small_L128_bs1.json`, `mlp_512_512_bs32.json`
