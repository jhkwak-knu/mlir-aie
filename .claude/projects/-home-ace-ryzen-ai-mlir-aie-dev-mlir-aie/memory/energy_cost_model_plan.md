---
name: energy_cost_model_plan
description: NPU 에너지 비용 모델 검증/캘리브레이션 계획 및 구현 상태
type: project
---

## NPU 에너지 비용 모델 — 구현 현황

**브랜치**: `energy-cost-model` (spatio-temporal-cost-model 기반)

### 구현 완료 파일 (Phase 0-4)

| # | 파일 | 역할 | 상태 |
|---|------|------|------|
| 1 | `scripts/analyze/energy_meter.py` | NPU/RAPL/GPU 3계층 전력 측정 라이브러리 | 완료 |
| 2 | `scripts/analyze/validate_energy_measurement.py` | Phase 0 측정 방법론 검증 (0-A~0-E) | 완료 |
| 3 | `scripts/analyze/run_energy_sweep.py` | Phase 2 에너지 수집 자동화 | 완료 |
| 4 | `scripts/analyze/validate_energy_model.py` | Phase 3/5 모델 검증 + 교차 검증 | 완료 |
| 5 | `scripts/analyze/calibrate_energy.py` | Phase 4 에너지 계수 fitting (4모델) | 완료 |

### 수정 완료 파일

| 파일 | 변경 | 상태 |
|------|------|------|
| `tiling_common.py` | CalibCoeffs에 energy 필드 3개 추가, load_calibration v3 지원 | 완료 |
| `cost_model.py` | 에너지 함수에 coeffs 전달, energy_total_calibrated() 추가 | 완료 |

### 센서 감지 결과 (2026-03-16)
- NPU ioctl: N/A (NPU 비활성 시)
- RAPL: root 필요
- GPU hwmon: 사용 가능 (amdgpu)
- xrt-smi: idle 시 N/A (활성 시 미확인)

**Why:** 에너지 효율 최적화가 프로젝트 목표. 현재 에너지 모델은 Horowitz 2014 이론 상수만 사용.
**How to apply:** Phase 0-A (xrt-smi 활성 워크로드 중 전력) 실험부터 시작. sudo 필요 시 RAPL fallback.
