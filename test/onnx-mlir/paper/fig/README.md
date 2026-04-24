# Paper Figure Generator — Setup Guide (Windows)

## 셋업 (1회)

```cmd
cd "C:\Users\<사용자명>\Desktop\Ryzen_AI\EDP_Opt\fig"

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

## 실행

```cmd
cd "C:\Users\<사용자명>\Desktop\Ryzen_AI\EDP_Opt\fig"
.venv\Scripts\activate

python main.py

python main.py --only F6 T8

python main.py --csv new_result.csv --json new_tc_list.json --cal new_calib.json
```

## PowerShell 사용 시

```powershell
cd "C:\Users\<사용자명>\Desktop\Ryzen_AI\EDP_Opt\fig"

# 최초 1회: 스크립트 실행 정책 허용 (관리자 권한 불필요)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py
```

## 재측정 후 업데이트

1. `calibration.json` 계수값 교체
2. CSV/JSON 파일 교체
3. `python main.py` 재실행

## 출력물 (output\ 폴더)

| 파일 | 내용 |
|------|------|
| F5_pred_vs_measured.pdf/png | 예측 vs 실측 산점도 (T, E) |
| F6_edp_landscape.pdf/png | ★ EDP Landscape Small Multiples |
| F7_t_vs_e_decomposition.pdf/png | GT vs Naive-Max: T 증가 vs E 절감 |
| T5_workloads.tex/csv | 워크로드 목록 |
| T6_perf_accuracy.tex | 성능 모델 계수 + 정확도 |
| T7_energy_accuracy.tex | 에너지 모델 계수 + 정확도 |
| T8_edp_reduction.tex/csv | EDP 절감 정량화 |

## 참고

- Python >= 3.9 필요
- 가상 환경 비활성화: `deactivate`
- T8의 `†`: Framework 선택 config이 측정 데이터에 없어 예측값 사용
