#!/usr/bin/env bash
# Structural verification: run all 24 structurally distinct parallelization cases.
# Usage: bash scripts/run_structural_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEST_DIR="$ROOT_DIR/test/onnx-mlir"
TC_JSON="$TEST_DIR/out/tc.json"
RESULT_CSV="$TEST_DIR/out/reports/structural_test_result.csv"

mkdir -p "$(dirname "$RESULT_CSV")"

# CSV header
echo "case,SPm,SPn,numCores,numCols,tpOrder,TPk,pres,M,K,N,build_status,run_status,avg_us" > "$RESULT_CSV"

# Each case: SPm SPn numCores tpOrder0 tpOrder1 tpOrder2 TPk M K N description
CASES=(
  # 1-col: SPm=1 (LHS broadcast)
  "1  1 4  4  2 0 1  1  16 16 64   LHS_broadcast_spatial_only"
  "2  1 4  4  2 0 1  2  16 32 64   LHS_broadcast_K_loop"
  "3  1 4  4  0 2 1  2  16 32 64   LHS_broadcast_M_pres"
  "4  1 4  4  1 2 0  2  16 32 64   LHS_broadcast_N_pres"
  # 1-col: SPm=2 (symmetric multicast)
  "5  2 2  4  2 0 1  1  32 16 32   symmetric_multicast_spatial_only"
  "6  2 2  4  2 0 1  2  32 32 32   symmetric_multicast_K_loop"
  "7  2 2  4  0 2 1  2  32 32 32   symmetric_multicast_M_pres"
  "8  2 2  4  1 2 0  2  32 32 32   symmetric_multicast_N_pres"
  # 1-col: SPm=4 (RHS broadcast)
  "9  4 1  4  2 0 1  1  64 16 16   RHS_broadcast_spatial_only"
  "10 4 1  4  2 0 1  2  64 32 16   RHS_broadcast_K_loop"
  "11 4 1  4  0 2 1  2  64 32 16   RHS_broadcast_M_pres"
  "12 4 1  4  1 2 0  2  64 32 16   RHS_broadcast_N_pres"
  # 2-col: SPm=1
  "13 1 8  8  2 0 1  1  16 16 128  2col_LHS_broadcast_spatial"
  "14 1 8  8  0 2 1  2  16 32 128  2col_LHS_broadcast_M_pres"
  # 2-col: SPm=2
  "15 2 4  8  2 0 1  1  32 16 64   2col_symmetric_spatial"
  "16 2 4  8  0 2 1  2  32 32 64   2col_symmetric_M_pres"
  # 2-col: SPm=4
  "17 4 2  8  2 0 1  1  64 16 32   2col_RHS_broadcast_spatial"
  "18 4 2  8  0 2 1  2  64 32 32   2col_RHS_broadcast_M_pres"
  # 2-col: SPm=8 (SPm > compTilesPerCol)
  "19 8 1  8  2 0 1  1  128 16 16  SPm8_all_unicast_spatial"
  "20 8 1  8  0 2 1  2  128 32 16  SPm8_all_unicast_M_pres"
  # 4-col: SPm=2
  "21 2 8  16 2 0 1  1  32 16 128  4col_symmetric_spatial"
  "22 2 8  16 0 2 1  2  32 32 128  4col_symmetric_M_pres"
  # 3-col: SPm=3 (asymmetric)
  "23 3 4  12 2 0 1  1  48 16 64   asymmetric_SPm3_spatial"
  "24 3 4  12 0 2 1  2  48 32 64   asymmetric_SPm3_M_pres"
)

TOTAL=${#CASES[@]}
PASS_COUNT=0
FAIL_COUNT=0

for entry in "${CASES[@]}"; do
  read -r CASE_NUM SPm SPn CORES TP0 TP1 TP2 TPk M K N DESC <<< "$entry"

  # Compute TPm and TPn (always 1 for structural test)
  TPm=1
  TPn=1
  TM=16
  TK=16
  TN=16

  echo ""
  echo "================================================================"
  echo "Case $CASE_NUM/$TOTAL: SP=($SPm,$SPn) cores=$CORES tpOrder=[$TP0,$TP1,$TP2] TPk=$TPk M=${M}xK=${K}xN=${N}"
  echo "  Description: $DESC"
  echo "================================================================"

  # Write tc.json
  cat > "$TC_JSON" <<EOF
{
  "M": $M, "K": $K, "N": $N, "elemType": "bf16",
  "numCores": $CORES, "doubleBuffer": false,
  "levels": [{
    "SPm": $SPm, "SPn": $SPn, "TPm": $TPm, "TPk": $TPk, "TPn": $TPn,
    "TM": $TM, "TK": $TK, "TN": $TN, "tpOrder": [$TP0, $TP1, $TP2]
  }]
}
EOF

  BUILD_STATUS="FAIL"
  RUN_STATUS="FAIL"
  AVG_US="-"

  # Build
  cd "$TEST_DIR"
  make clean > /dev/null 2>&1 || true
  if bash scripts/gen_onnx_matmul_mlir.sh out/tc.json > /dev/null 2>&1 && \
     make > /dev/null 2>&1; then
    BUILD_STATUS="PASS"
    echo "  Build: PASS"

    # Run
    if make run > /dev/null 2>&1; then
      if grep -q 'PASS!' out/logs/log.txt 2>/dev/null; then
        RUN_STATUS="PASS"
        AVG_US=$(grep 'Avg NPU time' out/logs/log.txt 2>/dev/null | sed 's/.*: \(.*\)us\./\1/' || echo "-")
        echo "  Run:   PASS (Avg ${AVG_US}us)"
        PASS_COUNT=$((PASS_COUNT + 1))
      else
        MISMATCH=$(grep -o '[0-9]* mismatches' out/logs/log.txt 2>/dev/null || echo "unknown error")
        echo "  Run:   FAIL ($MISMATCH)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
      fi
    else
      echo "  Run:   FAIL (execution error)"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
  else
    echo "  Build: FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    # Capture error for debugging
    if [ -f out/logs/log.txt ]; then
      tail -5 out/logs/log.txt 2>/dev/null || true
    fi
  fi

  echo "$CASE_NUM,$SPm,$SPn,$CORES,$(($CORES/4)),[$TP0.$TP1.$TP2],$TPk,$([ $TPk -gt 1 ] && [ $TP0 -ne 2 ] && echo Yes || echo No),$M,$K,$N,$BUILD_STATUS,$RUN_STATUS,$AVG_US" >> "$RESULT_CSV"
done

echo ""
echo "================================================================"
echo "SUMMARY: $PASS_COUNT/$TOTAL PASS, $FAIL_COUNT/$TOTAL FAIL"
echo "Results: $RESULT_CSV"
echo "================================================================"
