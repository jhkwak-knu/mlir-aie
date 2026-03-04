#!/usr/bin/env bash
# run_tc_all.sh - Iterate all cases in tc_list.json, build+run each,
#                 append results to result.csv, then clean before next case.
#
# Usage:
#   ./run_tc_all.sh [-i INPUT_JSON] [-o OUTPUT_JSON] [-r RESULT_CSV] [-n INDEX]
#     -i: input file   (default: out/tc_list.json)
#     -o: tc.json path (default: out/tc.json)
#     -r: result csv   (default: out/reports/result.csv)
#     -n: 1-based case index to run only that single case

set -uo pipefail  # intentionally NOT using -e to continue on errors

# resolve dirs and common paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

# defaults under new tree
INPUT_JSON="$TC_LIST"                  # out/tc_list.json
OUTPUT_JSON="$OUT_DIR/tc.json"         # out/tc.json
RESULT_CSV="$REPORTS_DIR/result.csv"   # out/reports/result.csv
SINGLE_IDX=""                          # optional: run only this 1-based index

usage() {
  cat <<EOF
run_tc_all.sh - Run testcases from JSON and append results to CSV.

Options:
  -i FILE   Input JSON list (default: $INPUT_JSON)
  -o FILE   Per-case extracted JSON (default: $OUTPUT_JSON)
  -r FILE   Result CSV path (default: $RESULT_CSV)
  -n INDEX  Run only the INDEX-th case (1-based)
  -h        Help
EOF
  exit 1
}

while getopts ":i:o:r:n:h" opt; do
  case "$opt" in
    i) INPUT_JSON="$OPTARG" ;;
    o) OUTPUT_JSON="$OPTARG" ;;
    r) RESULT_CSV="$OPTARG" ;;
    n) SINGLE_IDX="$OPTARG" ;;
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
command -v jq   &>/dev/null || { echo "error: 'jq' required" >&2; exit 127; }
command -v make &>/dev/null || { echo "error: 'make' required" >&2; exit 127; }

# input check
[[ -f "$INPUT_JSON" ]] || { echo "error: input file not found: $INPUT_JSON" >&2; exit 2; }

TOTAL_CASES=$(jq '.cases | length' "$INPUT_JSON")
if [[ "$TOTAL_CASES" -le 0 ]]; then
  echo "error: 'cases' array is empty." >&2
  exit 3
fi

# range decide (single-case support)
if [[ -n "${SINGLE_IDX:-}" ]]; then
  if ! [[ "$SINGLE_IDX" =~ ^[0-9]+$ ]] || [[ "$SINGLE_IDX" -lt 1 ]] || [[ "$SINGLE_IDX" -gt "$TOTAL_CASES" ]]; then
    echo "error: invalid -n index: $SINGLE_IDX (valid range: 1..$TOTAL_CASES)" >&2
    exit 4
  fi
  START_IDX="$SINGLE_IDX"
  END_IDX="$SINGLE_IDX"
  echo "Found $TOTAL_CASES cases in $INPUT_JSON (running only case #$SINGLE_IDX)"
else
  START_IDX=1
  END_IDX="$TOTAL_CASES"
  echo "Found $TOTAL_CASES cases in $INPUT_JSON"
fi

# CSV header (+upgrade if old header exists)
NEW_HEADER="case_index,numSpm,SPm,SPn,TPm,TPk,TPn,TM,TK,TN,M,K,N,doubleBuffer,status,errors,iters,warmup,avg_us,min_us,max_us"
if [[ ! -f "$RESULT_CSV" ]]; then
  mkdir -p "$(dirname "$RESULT_CSV")"
  echo "$NEW_HEADER" > "$RESULT_CSV"
else
  CUR_HEADER="$(head -n1 "$RESULT_CSV" || true)"
  if ! echo "$CUR_HEADER" | grep -q 'avg_us'; then
    echo "info: upgrading result CSV header (adding timing columns) ..."
    tmp_csv="$(mktemp)"
    echo "$NEW_HEADER" > "$tmp_csv"
    # Append old rows with five additional timing fields defaulted to -1.
    # tail failure is benign (CSV may not exist yet); only awk failure is an error.
    tail -n +2 "$RESULT_CSV" 2>/dev/null | awk -F',' '{print $0",-1,-1,-1,-1,-1"}' >> "$tmp_csv"
    if [[ ${PIPESTATUS[1]} -ne 0 ]]; then
      echo "error: awk failed during CSV header upgrade; aborting to preserve original" >&2
      rm -f "$tmp_csv"
      exit 1
    fi
    mv "$tmp_csv" "$RESULT_CSV"
  fi
fi

# helper to read numeric (default 0)
read_num() { jq -r "$1 // 0" "$OUTPUT_JSON"; }
# helper to read bool -> true/false string
read_bool_str() { jq -r "if $1 == true then \"true\" else \"false\" end" "$OUTPUT_JSON"; }

for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
  echo "=== Case #$idx / $TOTAL_CASES ==="

  # 1) extract case -> tc.json
  tmp_out="$(mktemp)"
  if ! jq --indent 4 ".cases[$((idx-1))]" "$INPUT_JSON" > "$tmp_out"; then
    echo "warn: failed to extract case #$idx"
    echo "$idx,0,0,0,0,0,0,0,0,0,0,0,0,false,EXTRACT_FAIL,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
    rm -f "$tmp_out"
    # cleanup then continue
    if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
    continue
  fi
  mkdir -p "$(dirname "$OUTPUT_JSON")"
  mv "$tmp_out" "$OUTPUT_JSON"

  # 2) read fields
  numSpm=$(read_num '.levels[0].numSpm')
  SPm=$(read_num '.levels[0].SPm'); SPn=$(read_num '.levels[0].SPn')
  TPm=$(read_num '.levels[0].TPm'); TPk=$(read_num '.levels[0].TPk'); TPn=$(read_num '.levels[0].TPn')
  TM=$(read_num '.levels[0].TM'); TK=$(read_num '.levels[0].TK'); TN=$(read_num '.levels[0].TN')
  M=$(read_num '.M');   K=$(read_num '.K');   N=$(read_num '.N')
  DB_STR=$(read_bool_str '.doubleBuffer')

  for v in numSpm SPm SPn TPm TPk TPn TM TK TN M K N; do
    val="${!v}"
    if ! [[ "$val" =~ ^-?[0-9]+$ ]]; then
      echo "warn: $v not integer (got: $val); marking as PARSE_FAIL"
      echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,PARSE_FAIL,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
      if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
      continue 2
    fi
  done

  echo "Params: numSpm=$numSpm | SPm=$SPm SPn=$SPn | TPm=$TPm TPk=$TPk TPn=$TPn | TM=$TM TK=$TK TN=$TN | M=$M K=$K N=$N | doubleBuffer=$DB_STR"

  # 3) generate MLIR
  if [[ -f "$GEN_SCRIPT" ]]; then
    echo "running: bash $GEN_SCRIPT $M $K $N"
    if ! bash "$GEN_SCRIPT" "$M" "$K" "$N"; then
      echo "warn: generator failed"
      echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,GEN_FAIL,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
      if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
      continue
    fi
  else
    echo "warn: generator script not found: $GEN_SCRIPT"
    echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,GEN_MISSING,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
    if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
    continue
  fi

  # 4) build & run (Makefile 'run' target). Pass macros for host.cpp.
  CPPDEFS="-DM_SIZE=$M -DK_SIZE=$K -DN_SIZE=$N"
  echo "make -C \"$MAKE_DIR\" run CPPDEFS=\"$CPPDEFS\""
  if ! make -C "$MAKE_DIR" run CPPDEFS="$CPPDEFS"; then
    echo "warn: make run failed"
    echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,RUN_FAIL,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
    # if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
    continue
  fi

  # 5) parse log.txt
  STATUS="FAIL"; ERRORS=-1
  ITERS=-1; WARMUP=-1; AVG_US=-1; MIN_US=-1; MAX_US=-1

  if [[ -f "$LOG_FILE" ]]; then
    # PASS/FAIL & mismatches
    if grep -q 'PASS!' "$LOG_FILE"; then
      STATUS="PASS"; ERRORS=0
    else
      last_mis="$(grep -Eo '[0-9]+ mismatches\.' "$LOG_FILE" | tail -n1 | grep -Eo '^[0-9]+')"
      if [[ -n "${last_mis:-}" ]]; then ERRORS="$last_mis"; fi
    fi

    # Iterations/Warmup: "Number of iterations: 5 (warmup iterations: 3)"
    it_line="$(grep -m1 -E '^Number of iterations:' "$LOG_FILE" || true)"
    if [[ -n "$it_line" ]]; then
      ITERS="$(echo "$it_line"  | grep -Eo 'Number of iterations:\s*[0-9]+' | awk '{print $NF}')"
      # Extract warmup within brackets
      WARMUP="$(echo "$it_line" | grep -Eo '\(warmup iterations:\s*[0-9]+' | grep -Eo '[0-9]+' || true)"
      : "${ITERS:=-1}"; : "${WARMUP:=-1}"
    fi

    # Avg/Min/Max NPU time: "Avg NPU time: 1279.2us."
    avg_line="$(grep -m1 -E '^Avg NPU time:' "$LOG_FILE" || true)"
    if [[ -n "$avg_line" ]]; then
      AVG_US="$(echo "$avg_line" | sed -E 's/.*Avg NPU time:\s*([0-9]+(\.[0-9]+)?)us.*/\1/')"
    fi
    min_line="$(grep -m1 -E '^Min NPU time:' "$LOG_FILE" || true)"
    if [[ -n "$min_line" ]]; then
      MIN_US="$(echo "$min_line" | sed -E 's/.*Min NPU time:\s*([0-9]+(\.[0-9]+)?)us.*/\1/')"
    fi
    max_line="$(grep -m1 -E '^Max NPU time:' "$LOG_FILE" || true)"
    if [[ -n "$max_line" ]]; then
      MAX_US="$(echo "$max_line" | sed -E 's/.*Max NPU time:\s*([0-9]+(\.[0-9]+)?)us.*/\1/')"
    fi

    # Check number format (Return -1 if missing)
    for v in AVG_US MIN_US MAX_US; do
      val="${!v}"
      if ! [[ "$val" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf -v "$v" '%s' -1
      fi
    done
  else
    STATUS="NO_LOG"; ERRORS=-1
  fi

  # 6) append to CSV
  echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,$STATUS,$ERRORS,$ITERS,$WARMUP,$AVG_US,$MIN_US,$MAX_US" >> "$RESULT_CSV"
  echo "Result: case #$idx -> $STATUS (errors=$ERRORS, avg=${AVG_US}us, min=${MIN_US}us, max=${MAX_US}us) appended to $RESULT_CSV"

  # 7) clean before next case
  if [[ -z "${SINGLE_IDX:-}" ]]; then
    if [[ -f "$CLEAN_SCRIPT" ]]; then
      echo "cleaning via $CLEAN_SCRIPT ..."
      bash "$CLEAN_SCRIPT" || true
    else
      echo "clean script not found; fallback to 'make clean'"
      make -C "$MAKE_DIR" clean || true
    fi
  fi

done

echo "All cases done. Results aggregated in: $RESULT_CSV"
