# NPU Reconfiguration 비용 측정 -- 실험 설정 및 결과 (시간 전용)

> 목적: 논문 서술 판단을 위해, **레이어 전환(runtime reconfiguration) 비용을 어떻게 측정
> 구간을 설정하고 어떻게 측정했는지**, 그리고 그 결과를 상세히 정리한다.
> 관련 분석 노트(측정이 아니라 "기존 실험이 이 비용을 어떻게 처리하는가"): `reconfiguration_measurement_notes.md`.
> **본 문서는 시간(Δt)만 다룬다. 에너지(ΔE)는 RAPL 권한 문제로 미측정 (§9).**

---

## 0. 한눈에 (TL;DR)

- **측정값(per-switch 재구성 비용)**: config 크기(numCores)에 **선형 비례**.
  - 4코어 = **276 µs**, 16코어 = **1161 µs**, 32코어(full array) = **2193 µs**.
  - 선형 회귀: **Δt ≈ 68.2 µs × numCores + 27** (즉 코어당 약 68 µs, R² ≈ 1.0).
- **co-residence 없음**: 8컬럼 안에 동시 상주 가능한 작은 config(4코어)도 **매 전환마다 reprogram**을 지불.
  → 드라이버는 레이어 전환마다 NPU를 다시 프로그래밍한다.
- **실제 모델 대조**(measurements.csv의 실제 NPU dispatch × 그 footprint의 Δt):
  - **BERT-Small (STAR-Map)**: ~36.1 ms/추론 = 측정 모델 시간(262.8 ms)의 **약 13.7%**.
  - **MLP (STAR-Map)**: ~2.0 ms/추론 = 측정 모델 시간(10.5 ms)의 **약 19.1%**.
- **STAR-Map이 setter 중 재구성 비율이 가장 낮다**(BERT 13.7% vs max_p/timeloop/charm 15.7~19.8%).
  STAR-Map이 더 작은 footprint를 고르기 때문 → 사용자 논증에 유리한 뉘앙스(§8).
- 이 값은 **이미 측정된 모델 시간에 포함**되어 있다(추가 오버헤드가 아니라 그 일부).
- 이 비용은 **현재 런타임 구조(레이어마다 별도 xclbin/hw_context, co-residence 미사용)의 산물**이다.

---

## 1. 실험 환경

### Hardware
- CPU: AMD Ryzen AI 9 HX 370 (Strix Point, Zen 5), 12C/24T
- **NPU: AMD XDNA2, 8 columns × 4 rows = 32 AIE cores, MemTile 512 KB/col, core scratchpad 64 KB, 1.5 GHz**
- RAM: 32 GB DDR5

### Software
- OS Ubuntu 24.04.4 LTS, Kernel 6.14.0-37
- **XRT 2.20.0, amdxdna 2.20.0** (드라이버가 hw_context/reprogram을 담당하는 핵심 소프트웨어)
- MLIR-AIE fork, branch `energy-cost-model`

### 측정 시점 드라이버 상태 (본 실험 특이사항)
- `amdxdna timeout_in_sec = 2` (sudo 불가로 60 미설정). **마이크로벤치 dispatch가 빠르므로(<3 ms) 무해**.
- 데이터 타입 bf16, kernel `mmul<4,8,8>` 2×2 expansion (기존 실험과 동일).

---

## 2. 측정 대상: reconfiguration이란, 어디서 발생하나

**런타임 레이어 전환 재구성(b)**: 여러 config(xclbin)가 각자 hw_context를 갖고 살아 있을 때,
직전과 **다른 config**를 submit하면 그 config가 컬럼에 상주하지 않는 한 드라이버가 해당 config의
CDO/instruction을 NPU 어레이에 **다시 써넣어(reprogram)** 실행한다.

- 발생 지점: `kernel(...)` submit 후 `run.wait()` **내부**(드라이버 동기 재프로그램).
- 실제 모델도 동일 구조: 레이어마다 별도 xclbin을 로드해 per-xclbin hw_context를 만들고
  (`src/mlp_runner/xrt_dispatcher.cpp:98-165`), 추론 시 레이어 순서대로 각 context에 dispatch
  (`xrt_dispatcher.cpp:261-270`). 즉 **모델의 매 레이어 전환 = 본 측정이 재현하는 그 전환**이다.

콜드 xclbin 로드(a, 디스크→등록→hw_context 최초 생성)는 **본 측정 대상이 아니다** — warmup으로 제외(§6).

---

## 3. 측정 원리: 차등(differential) 설계

핵심 아이디어: **같은 GEMM shape의 서로 다른 tiling config**를 대상으로,
config를 바꾸지 않는 실행과 매번 바꾸는 실행의 **dispatch+wait 시간 차이**를 낸다.
전환 이외의 모든 요소(호스트 XRT 오버헤드, DMA, 연산, per-dispatch startup)는 두 실행에서
**동일하므로 차이에서 상쇄**되고, 남는 것은 전환(reprogram) 비용뿐이다.

| 스케줄 | dispatch 순서 | 측정 내용 |
|---|---|---|
| **baseline** b_i | config i → i → i … (전환 없음) | 연산 + per-dispatch startup (reprogram 없음) |
| **alternating** a_d | config 0 → 1 → … → d−1 → 0 … (매번 전환) | 연산 + startup + **reprogram** |

**per-switch 재구성 비용**: `Δt(d) = a_d − mean(b_0 … b_{d−1})`

- baseline과 alternating을 **한 프로세스에서 back-to-back**으로 측정 → thermal/드라이버 상태 통제.
- 본 실험은 d=2(가장 깨끗한 최소 구성). config 0과 1을 번갈아 dispatch.

### 왜 이 차이가 순수 전환 비용인가
- alternating의 한 dispatch = (직전과 다른 config로 전환) + (그 config의 한 chunk 연산).
- baseline의 한 dispatch = (같은 config, 전환 없음) + (그 config의 한 chunk 연산).
- 두 실행에서 **연산·데이터 이동·호스트 경로가 동일** → 빼면 전환분만 남는다.
- alternating은 config 0/1을 반반 dispatch하므로 baseline도 두 config 평균을 뺀다.

---

## 4. 측정 구간 (가장 중요) -- 정확히 무엇을 언제 쟀나

측정 하니스: `src/host_reconfig.cpp` (기존 `src/host.cpp`의 측정 로직을 이식·축소).

### (1) 타이머가 감싸는 것: `runKernel()`만
```cpp
auto t0 = high_resolution_clock::now();
runKernel(pick(iter));      // = kernel(opcode, bo_instr, size, bo_inA, bo_inB, bo_outC[, bo_inC]) + run.wait()
auto t1 = high_resolution_clock::now();
double us = duration<micro>(t1 - t0);   // <-- 이 창이 per-dispatch 측정값
```
- **오직 `kernel(...)` submit + `run.wait()`** 만 t0..t1 안에 있다 = **dispatch + wait**.
- 이는 기존 커널벤치 `result_v14`의 `min_us`(dispatch+wait only)와 **동일한 스코프**다.
- **드라이버 reprogram은 `run.wait()` 내부에서 완료**되므로 전환이 일어나면 반드시 이 창에 잡힌다.

### (2) 창 밖으로 뺀 것: 입력 staging (핵심 설계)
- 입력 버퍼(bo_inA/inB/inC)와 instruction 버퍼(bo_instr)는 **측정 루프 시작 전에 1회만**
  zero로 채우고 `sync(BO_TO_DEVICE)` (`host_reconfig.cpp` loadConfig). 루프 안에서는 memcpy/sync를
  **하지 않는다**.
- 근거: **matmul dispatch 시간은 입력 데이터 값과 무관**(고정 함수, 데이터 의존 분기 없음).
  타이밍은 config와 버퍼 크기에만 의존한다. 따라서 값은 0이어도 무방하고, staging을 루프 밖에 두어
  **호스트 변환·복사·sync 잡음을 측정에서 완전히 배제**한다.
- 결과: 창 = 순수 device dispatch+wait. host.cpp의 정교한 tile staging/decode/verify는 **불필요**해
  하니스가 대폭 단순해졌다(원본 ~647줄 → ~330줄).

### (3) dispatch 단위와 전환 대응
- 한 `runKernel()` = 한 chunk에 대한 한 번의 kernel 호출.
- baseline: 같은 config를 반복 → 전환 0.
- alternating: 매 호출이 직전과 다른 config → **매 호출이 정확히 1회 전환**.
- 실제 모델에서는 한 GEMM이 여러 chunk(totalSteps) dispatch를 하지만 전환은 **첫 dispatch 1회**뿐이다.
  즉 본 측정의 "1 dispatch당 전환 비용"이 모델의 "레이어(GEMM)당 전환 비용"에 1:1로 대응한다(§8).

---

## 5. 통계 방법론 (batch-min, host.cpp 패턴 이식)

`measureSchedule()` (`host_reconfig.cpp`):
1. **warmup** `n_warmup=20`회 dispatch → 콜드 로드/최초 프로그래밍 폐기, 최소 warmup 시간으로 n_inner 산정.
2. **n_batches=5** 외부 배치. 각 배치는 목표 `target_batch_wall_s=0.5s`를 채울 때까지 내부 루프
   (n_inner 동적 결정). 배치별 per-dispatch 평균 시간 기록.
3. **배치 최소값 선택**: 배치 평균 중 **최솟값**을 대표값(`per_dispatch_us_min`)으로 — OS 스케줄링
   잡음이 가장 적은 배치. 배치 간 **CV(변동계수)**로 안정성 보고.
4. (에너지 활성 시) 배치 RAPL 창 + idle-bracket 보정 — 본 실험은 미사용.

측정 안정성: 모든 실행에서 **time_cv < 4%** (대부분 < 1%). 재현성 높음.

---

## 6. Config 생성 및 로드

### 생성 (`scripts/reconfig/build_configs.py`)
1. `scripts/generate/cost_model.py`로 대상 (M,K,N)의 **유효 tiling config 전수 열거**.
2. **numCores(footprint) 필터** 후 **EDP rank 상위 K개 distinct tiling** 선택(= cost-model setter가
   선호할 config).
3. 각 config를 `experiments/scripts/compile_kernels.py`의 `build_xclbin`으로 컴파일,
   `out/reconfig/<shape_tag>/cfg{i}/{final.xclbin,insts.bin,tc.json}`로 복사. `manifest.json` 기록.

### 로드 (`host_reconfig.cpp` loadConfig)
- device(0) 1회 open, config마다 `init_xrt_load_kernel`로 **별도 xclbin 등록 + 별도 hw_context**
  생성(각 kernel 핸들이 자기 hw_context를 살려둠). → K개 config가 동시에 살아 있는 상태.
- **이는 실제 모델 runner(`xrt_dispatcher.cpp:128-137`)와 동일한 다중 hw_context 패턴**이다.

### 왜 full-array(32코어)가 강제 reprogram인가
- numCores=32 = 8열×4행 전체 점유 → 두 config가 절대 동시 상주 불가 → 매 전환 full reprogram(최악 상한).
- 작은 footprint(4·16코어)도 측정해 **co-residence 여부를 직접 검증**(§7에서 "없음" 확인).

---

## 7. 결과: footprint sweep (원시 데이터)

원시 CSV: `out/reports/reconfig_v1.csv`. 각 행 = 한 (shape, depth, schedule).

| shape (GEMM) | footprint | baseline 평균 | alternating | **Δt (per-switch)** | µs/core | time_cv |
|---|---|---|---|---|---|---|
| 128×128×64 (attention_context) | **4코어** (2컬럼) | 81.5 µs | 357.6 µs | **276.1 µs** | 69.0 | <1.7% |
| 128×512×512 (attention qkv/output) | **16코어** (4-8컬럼) | 303.3 µs | 1464.7 µs | **1161.4 µs** | 72.6 | <4% |
| 128×512×512 (full array) | **32코어** (8컬럼) | 308.1 µs | 2501.0 µs | **2192.9 µs** | 68.5 | <0.7% |

### 해석
1. **선형성**: Δt ≈ **68.2 µs × numCores + 27** (세 점 회귀, 거의 원점 통과). 재구성 비용은
   재프로그래밍하는 **코어(타일) 수에 비례** — CDO/instruction을 더 많은 타일에 써넣을수록 비쌈.
2. **연산 대비 비율**: 전환 비용이 steady-state 연산의 **3.4×(4코어) ~ 7.1×(32코어)**. 큰 config일수록
   연산은 병렬화로 빨라지지만 전환은 코어 수에 비례해 커져 **전환:연산 비율이 악화**.
3. **co-residence 없음(중요)**: 4코어 config 두 개는 함께 써도 6컬럼(<8)이라 동시 상주가 물리적으로
   가능한데도, alternating(357.6)이 baseline(81.5)보다 **276 µs 더** 든다. 즉 드라이버는
   **fit 여부와 무관하게 매 전환마다 reprogram**한다. → "작은 config면 전환이 공짜"라는 희망은 성립 안 함.

---

## 8. 모델 대조 (새 측정 없이, 실제 dispatch 가중)

**N_switch의 ground truth는 measurements.csv의 실제 NPU dispatch**(distinct `(layer, layer_idx)`
kernel 행)이다. 각 dispatch의 footprint(numCores)는 `configurations.json`에서 GEMM 타입명으로 조회.
이 방식은 **CPU 폴백 GEMM(attention_score/context)을 자동 제외**하고 **분리된 GEMM(Q/K/V)을 개별
계수**한다. (레이어(GEMM)당 전환 1회 = §4-(3).)

### BERT-Small (L=128, bs=1, 4 layers) -- 실제 dispatch 기준
측정된 NPU dispatch = **레이어당 6개**: Q(16), K(16), V(16), attention_output(16),
ffn_expand(32), ffn_compress(32). **attention_score/context는 NPU에 없음(CPU ggml 폴백)이라 제외.**
qkv는 configurations.json엔 1타입이지만 실제로는 Q/K/V 3개 GEMM.

- 레이어당 전환 합 = 4×1161.4(16코어 4개) + 2×2192.9(32코어 2개) = **9,031.4 µs**
- × 4 layers = 24 dispatch = **36,126 µs ≈ 36.1 ms/추론**
- 측정된 모델 시간(star_map) = **262,830 µs** → **재구성 ≈ 13.74%**

### MLP (784-512-512-16, bs=32, 3 layers)
측정된 NPU dispatch = fc1(16), fc2(8), fc3(4).

- 전환 합 = 1161.4 + **573.6*** + 276.1 = **2,011 µs ≈ 2.0 ms/추론**
  (*fc2=8코어는 미실측, 선형 fit `68.2×8+27=573.6 µs`로 보간)
- 측정된 모델 시간(star_map) = **10,523.8 µs** → **재구성 ≈ 19.11%**

> MLP 비율이 더 큰 이유: 모델 자체가 작고 빨라 고정적 전환 비용의 비중이 큼.

### setter 간 비교 (STAR-Map이 유리한 근거)
| 모델 | STAR-Map | max_p | timeloop | charm_cdse |
|---|---|---|---|---|
| BERT-Small 재구성 % | **13.7%** | 19.8% | 19.7% | 15.7% |
| MLP 재구성 % | **19.1%** | 37.1% | 36.5% | 33.9% |

- **STAR-Map이 모든 setter 중 재구성 비율이 가장 낮다.** 다른 setter는 대부분 full-array(32코어)를
  골라 전환마다 최대 reprogram(2193 µs)을 내는 반면, STAR-Map은 작은 footprint를 섞어 골라
  전환 비용까지 낮춘다. → "STAR-Map의 per-layer 선택이 (재구성 포함) 전체적으로 유리"에 부합.
- 단, 절대값은 여전히 13~19%로 무시할 수준은 아님(§9 참고).

산출 스크립트: `scripts/reconfig/analyze_reconfig.py` (figure 데이터 `paper/fig/data/reconfig_v1.csv`,
`paper/fig/data/reconfig_model_context.csv`).

---

## 9. Caveat / 해석 주의점 (서술 시 반드시 고려)

1. **이미 포함된 비용**: 모델 시간 262.8 ms(BERT)는 이 재구성을 **이미 포함**한다(측정 창 안에 있음,
   `reconfiguration_measurement_notes.md` §5). 36 ms는 262.8 ms에 더해지는 게 아니라 그중 일부.
2. **"per-layer 선택이 유리"는 별개 문제**: 위 11~19%는 재구성 **비용**이다. per-layer 최적 config
   선택의 **이득**이 이 비용을 초과하는지는 **고정-config 대조 실험**(현재 scope 밖)이 있어야 증명된다.
3. **현재 런타임의 산물**: 레이어마다 별도 xclbin/hw_context를 만들고 co-residence를 안 하는 현
   구조가 매 전환 reprogram을 유발한다. 여러 레이어를 한 xclbin에 묶거나 co-residence를 쓰는 런타임은
   이 비용을 줄일 수 있다 → "per-layer config가 나쁘다"가 아니라 **최적화 여지**로 해석 가능.
4. **전환 vs 단순 context-switch**: Δt는 두 hw_context 간 전환 시 device 측에서 발생하는 추가 시간
   전부다. 코어 수 선형성(reprogram이 타일 수에 비례)은 이것이 순수 XRT 호스트 오버헤드가 아니라
   **어레이 재프로그래밍**임을 강하게 시사하나, 드라이버 내부 동작을 코드로 단정하진 못한다.
5. **d=2만 측정**: 로테이션 깊이 d=3,4(더 많은 config 순환)는 미측정. co-residence가 없으므로
   d를 늘려도 per-switch는 동일할 것으로 예상되나 확인은 추가 실험 필요.
6. **에너지 미측정**: RAPL `energy_uj`가 root 전용(0400)이라 ΔE는 -1. 필요 시 RAPL 읽기 권한 부여 후
   동일 하니스로 즉시 측정 가능(에너지 창은 배치 RAPL + idle-bracket, host.cpp와 동일).
7. **MLP fc2(8코어)는 보간값**. 정밀히 하려면 8코어 config를 실측하면 됨.
8. **shape 대표성**: 전환 비용은 코어 수에 비례하고 shape 의존은 약함(같은 footprint면 유사). 다만
   instruction 길이 차이에 따른 2차 효과는 미검증.

---

## 10. 재현 정보

### 파일
- 하니스: `test/onnx-mlir/src/host_reconfig.cpp` (+ `CMakeLists.txt`의 `HOST_SRC`, `Makefile`의 `host_reconfig` 타깃)
- config 빌드: `scripts/reconfig/build_configs.py`
- 실행: `scripts/reconfig/run_reconfig.sh`
- 분석: `scripts/reconfig/analyze_reconfig.py`
- 원시 결과: `out/reports/reconfig_v1.csv`
- config 아티팩트: `out/reconfig/<shape_tag>/cfg{i}/`, `manifest.json`

### 명령어 (재현)
```bash
source test/onnx-mlir/scripts/setup_env.sh
cd test/onnx-mlir
# 1) config 빌드 (footprint별 K개 xclbin)
python3 scripts/reconfig/build_configs.py --shape 128 512 512 \
        --shape-tag qkv_128x512x512_p16 --depth 2 --footprint 16
# 2) 측정 (baseline + alternating, 시간)
bash scripts/reconfig/run_reconfig.sh qkv_128x512x512_p16 "2"
# 3) 분석 (Δt + 모델 대조)
python3 scripts/reconfig/analyze_reconfig.py
```

### 측정된 config (manifest 기준)
- 4코어: (128×128×64) SPm2×SPn2, SPm1×SPn4
- 16코어: (128×512×512) SPm4×SPn4, SPm2×SPn8
- 32코어: (128×512×512) SPm4×SPn8, SPm2×SPn16
