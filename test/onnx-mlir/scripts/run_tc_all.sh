#!/usr/bin/env bash
# run_tc_all.sh - Iterate all cases in tc_list.json, build+run each,
#                 append results to result.csv, then clean before next case.
#
# Usage:
#   ./run_tc_all.sh [-i INPUT_JSON] [-o OUTPUT_JSON] [-r RESULT_CSV]
#     -i: input file   (default: data/tc_list.json)
#     -o: tc.json path (default: out/tc.json)
#     -r: result csv   (default: out/reports/result.csv)

set -uo pipefail  # intentionally NOT using -e to continue on errors

# resolve dirs and common paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

# defaults under new tree
INPUT_JSON="$TC_LIST"                  # data/tc_list.json
OUTPUT_JSON="$OUT_DIR/tc.json"         # out/tc.json
RESULT_CSV="$REPORTS_DIR/result.csv"   # out/reports/result.csv

usage() {
  cat <<EOF
run_tc_all.sh - Run ALL testcases from JSON and append results to CSV.

Options:
  -i FILE   Input JSON list (default: $INPUT_JSON)
  -o FILE   Per-case extracted JSON (default: $OUTPUT_JSON)
  -r FILE   Result CSV path (default: $RESULT_CSV)
  -h        Help
EOF
  exit 1
}

while getopts ":i:o:r:h" opt; do
  case "$opt" in
    i) INPUT_JSON="$OPTARG" ;;
    o) OUTPUT_JSON="$OPTARG" ;;
    r) RESULT_CSV="$OPTARG" ;;
    h) usage ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

GEN_SCRIPT="$SCRIPTS_DIR/gen_onnx_matmul_mlir.sh"
CLEAN_SCRIPT="$SCRIPTS_DIR/clean_tc_all.sh"
MAKE_DIR="$ROOT_DIR"                   # Makefile at repo root
LOG_FILE="$LOGS_DIR/log.txt"

# deps
command -v jq   >/dev/null 2>&1 || { echo "error: 'jq' required"; exit 127; }
command -v make >/dev/null 2>&1 || { echo "error: 'make' required"; exit 127; }

# input check
[ -f "$INPUT_JSON" ] || { echo "error: input file not found: $INPUT_JSON" >&2; exit 2; }

TOTAL_CASES=$(jq '.cases | length' "$INPUT_JSON")
if [ "$TOTAL_CASES" -le 0 ]; then
  echo "error: 'cases' array is empty." >&2
  exit 3
fi
echo "Found $TOTAL_CASES cases in $INPUT_JSON"

# CSV header
if [ ! -f "$RESULT_CSV" ]; then
  mkdir -p "$(dirname "$RESULT_CSV")"
  echo "case_index,numLevel,TM,TK,TN,MemTM,MemTK,MemTN,M,K,N,numLastSpm,doubleBuffer,status,errors" > "$RESULT_CSV"
fi

# helper to read numeric (default 0)
read_num() { jq -r "$1 // 0" "$OUTPUT_JSON"; }
# helper to read bool -> true/false 문자열
read_bool_str() { jq -r "if $1 == true then \"true\" else \"false\" end" "$OUTPUT_JSON"; }

for (( idx=1; idx<=TOTAL_CASES; idx++ )); do
  echo "=== Case #$idx / $TOTAL_CASES ==="

  # 1) extract case -> tc.json
  tmp_out="$(mktemp)"
  if ! jq --indent 4 ".cases[$((idx-1))]" "$INPUT_JSON" > "$tmp_out"; then
    echo "warn: failed to extract case #$idx"
    echo "$idx,0,0,0,0,0,0,0,0,0,0,0,false,EXTRACT_FAIL,-1" >> "$RESULT_CSV"
    rm -f "$tmp_out"
    # cleanup then continue
    if [ -x "$CLEAN_SCRIPT" ]; then bash "$CLEAN_SCRIPT" || true; fi
    continue
  fi
  mv "$tmp_out" "$OUTPUT_JSON"

  # 2) read fields
  numLevel=$(read_num '.numLevel')
  TM=$(read_num '.TM'); TK=$(read_num '.TK'); TN=$(read_num '.TN')
  MemTM=$(read_num '.MemTM'); MemTK=$(read_num '.MemTK'); MemTN=$(read_num '.MemTN')
  M=$(read_num '.M');   K=$(read_num '.K');   N=$(read_num '.N')
  numLastSpm=$(read_num '.numLastSpm')
  DB_STR=$(read_bool_str '.doubleBuffer')

  for v in numLevel TM TK TN MemTM MemTK MemTN M K N numLastSpm; do
    val="${!v}"
    if ! [[ "$val" =~ ^-?[0-9]+$ ]]; then
      echo "warn: $v not integer (got: $val); marking as PARSE_FAIL"
      echo "$idx,$numLevel,$TM,$TK,$TN,$MemTM,$MemTK,$MemTN,$M,$K,$N,$numLastSpm,$DB_STR,PARSE_FAIL,-1" >> "$RESULT_CSV"
      [ -x "$CLEAN_SCRIPT" ] && bash "$CLEAN_SCRIPT" || true
      continue 2
    fi
  done

  echo "Params: numLevel=$numLevel | TM=$TM TK=$TK TN=$TN | MemTM=$MemTM MemTK=$MemTK MemTN=$MemTN | M=$M K=$K N=$N | numLastSpm=$numLastSpm | doubleBuffer=$DB_STR"

  # 3) generate MLIR
  if [ -x "$GEN_SCRIPT" ]; then
    echo "running: $GEN_SCRIPT $M $K $N"
    if ! "$GEN_SCRIPT" "$M" "$K" "$N"; then
      echo "warn: generator failed"
      echo "$idx,$numLevel,$TM,$TK,$TN,$MemTM,$MemTK,$MemTN,$M,$K,$N,$numLastSpm,$DB_STR,GEN_FAIL,-1" >> "$RESULT_CSV"
      [ -x "$CLEAN_SCRIPT" ] && bash "$CLEAN_SCRIPT" || true
      continue
    fi
  elif [ -f "$GEN_SCRIPT" ]; then
    echo "running: bash $GEN_SCRIPT $M $K $N"
    if ! bash "$GEN_SCRIPT" "$M" "$K" "$N"; then
      echo "warn: generator failed"
      echo "$idx,$numLevel,$TM,$TK,$TN,$MemTM,$MemTK,$MemTN,$M,$K,$N,$numLastSpm,$DB_STR,GEN_FAIL,-1" >> "$RESULT_CSV"
      [ -x "$CLEAN_SCRIPT" ] && bash "$CLEAN_SCRIPT" || true
      continue
    fi
  else
    echo "warn: generator script not found: $GEN_SCRIPT"
    echo "$idx,$numLevel,$TM,$TK,$TN,$MemTM,$MemTK,$MemTN,$M,$K,$N,$numLastSpm,$DB_STR,GEN_MISSING,-1" >> "$RESULT_CSV"
    [ -x "$CLEAN_SCRIPT" ] && bash "$CLEAN_SCRIPT" || true
    continue
  fi

  # 4) build & run (Makefile 'run' target). Pass macros for host.cpp.
  CPPDEFS="-DM_SIZE=$M -DK_SIZE=$K -DN_SIZE=$N"
  echo "make -C \"$MAKE_DIR\" run CPPDEFS=\"$CPPDEFS\""
  if ! make -C "$MAKE_DIR" run CPPDEFS="$CPPDEFS"; then
    echo "warn: make run failed"
    echo "$idx,$numLevel,$TM,$TK,$TN,$MemTM,$MemTK,$MemTN,$M,$K,$N,$numLastSpm,$DB_STR,RUN_FAIL,-1" >> "$RESULT_CSV"
    [ -x "$CLEAN_SCRIPT" ] && bash "$CLEAN_SCRIPT" || true
    continue
  fi

  # 5) parse log.txt
  STATUS="FAIL"; ERRORS=-1
  if [ -f "$LOG_FILE" ]; then
    if grep -q 'PASS!' "$LOG_FILE"; then
      STATUS="PASS"; ERRORS=0
    else
      last_mis="$(grep -Eo '[0-9]+ mismatches\.' "$LOG_FILE" | tail -n1 | grep -Eo '^[0-9]+')"
      if [ -n "${last_mis:-}" ]; then ERRORS="$last_mis"; fi
    fi
  else
    STATUS="NO_LOG"; ERRORS=-1
  fi

  # 6) append to CSV
  echo "$idx,$numLevel,$TM,$TK,$TN,$MemTM,$MemTK,$MemTN,$M,$K,$N,$numLastSpm,$DB_STR,$STATUS,$ERRORS" >> "$RESULT_CSV"
  echo "Result: case #$idx -> $STATUS (errors=$ERRORS) appended to $RESULT_CSV"

  # 7) clean before next case
  if [ -x "$CLEAN_SCRIPT" ]; then
    echo "cleaning via $CLEAN_SCRIPT ..."
    bash "$CLEAN_SCRIPT" || true
  else
    echo "clean script not found; fallback to 'make clean'"
    make -C "$MAKE_DIR" clean || true
  fi

done

echo "All cases done. Results aggregated in: $RESULT_CSV"
