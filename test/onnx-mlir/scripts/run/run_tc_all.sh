#!/usr/bin/env bash
# run_tc_all.sh - Iterate all cases in tc_list.json, build+run each,
#                 append results to result.csv, then clean before next case.
#
# Usage:
#   ./run_tc_all.sh [-i INPUT_JSON] [-o OUTPUT_JSON] [-r RESULT_CSV] [-n INDEX] [-s START] [-t] [-a DIR]
#     -i: input file   (default: out/tc_list.json)
#     -o: tc.json path (default: out/tc.json)
#     -r: result csv   (default: out/reports/result.csv)
#     -n: 1-based case index to run only that single case
#     -s: 1-based start index to resume from (skips earlier cases)
#     -t: enable NPU trace collection
#     -a: archive per-case artifacts to DIR/case_NNN/

set -uo pipefail  # intentionally NOT using -e to continue on errors

# resolve dirs and common paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../common.sh"

# defaults under new tree
INPUT_JSON="$TC_LIST"                  # out/tc_list.json
OUTPUT_JSON="$OUT_DIR/tc.json"         # out/tc.json
RESULT_CSV="$REPORTS_DIR/result.csv"   # out/reports/result.csv
SINGLE_IDX=""                          # optional: run only this 1-based index
START_FROM=""                          # optional: resume from this 1-based index
TRACE_ENABLED=""                       # optional: enable NPU trace collection
TRACE_SZ_DEFAULT=1048576               # 1MB trace buffer
ARCHIVE_DIR=""                         # optional: archive per-case artifacts

usage() {
  cat <<EOF
run_tc_all.sh - Run testcases from JSON and append results to CSV.

Options:
  -i FILE   Input JSON list (default: $INPUT_JSON)
  -o FILE   Per-case extracted JSON (default: $OUTPUT_JSON)
  -r FILE   Result CSV path (default: $RESULT_CSV)
  -n INDEX  Run only the INDEX-th case (1-based)
  -s START  Resume from the START-th case (1-based, appends to existing CSV)
  -t        Enable NPU trace collection and analysis
  -a DIR    Archive per-case artifacts to DIR/case_NNN/ (preserves all trace data)
  -h        Help
EOF
  exit 1
}

while getopts ":i:o:r:n:s:a:th" opt; do
  case "$opt" in
    i) INPUT_JSON="$OPTARG" ;;
    o) OUTPUT_JSON="$OPTARG" ;;
    r) RESULT_CSV="$OPTARG" ;;
    n) SINGLE_IDX="$OPTARG" ;;
    s) START_FROM="$OPTARG" ;;
    a) ARCHIVE_DIR="$OPTARG" ;;
    t) TRACE_ENABLED=1 ;;
    h) usage ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

GEN_SCRIPT="$SCRIPTS_DIR/generate/gen_onnx_matmul_mlir.sh"
CLEAN_SCRIPT="$SCRIPTS_DIR/run/clean_tc_all.sh"
MAKE_DIR="$ROOT_DIR"                   # Makefile at repo root
LOG_FILE="$LOGS_DIR/log.txt"

# Trace-related paths (mlir-aie repo root is two levels above test/onnx-mlir)
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
PARSE_TRACE="$REPO_ROOT/programming_examples/utils/parse_trace.py"
ANALYZE_TRACE="$SCRIPTS_DIR/analyze/analyze_trace.py"
TRACE_RAW="$LOGS_DIR/trace_raw.txt"
TRACE_JSON="$LOGS_DIR/trace.json"
TRACE_SUMMARY="$LOGS_DIR/trace_summary.json"
AIE_MLIR="$BUILD_DIR/aie.mlir"

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

# range decide (single-case and resume support)
if [[ -n "${SINGLE_IDX:-}" ]]; then
  if ! [[ "$SINGLE_IDX" =~ ^[0-9]+$ ]] || [[ "$SINGLE_IDX" -lt 1 ]] || [[ "$SINGLE_IDX" -gt "$TOTAL_CASES" ]]; then
    echo "error: invalid -n index: $SINGLE_IDX (valid range: 1..$TOTAL_CASES)" >&2
    exit 4
  fi
  START_IDX="$SINGLE_IDX"
  END_IDX="$SINGLE_IDX"
  echo "Found $TOTAL_CASES cases in $INPUT_JSON (running only case #$SINGLE_IDX)"
elif [[ -n "${START_FROM:-}" ]]; then
  if ! [[ "$START_FROM" =~ ^[0-9]+$ ]] || [[ "$START_FROM" -lt 1 ]] || [[ "$START_FROM" -gt "$TOTAL_CASES" ]]; then
    echo "error: invalid -s index: $START_FROM (valid range: 1..$TOTAL_CASES)" >&2
    exit 4
  fi
  START_IDX="$START_FROM"
  END_IDX="$TOTAL_CASES"
  echo "Found $TOTAL_CASES cases in $INPUT_JSON (resuming from case #$START_FROM)"
else
  START_IDX=1
  END_IDX="$TOTAL_CASES"
  echo "Found $TOTAL_CASES cases in $INPUT_JSON"
fi

# CSV header (+upgrade if old header exists)
NEW_HEADER="case_index,numSpm,SPm,SPn,TPm,TPk,TPn,TM,TK,TN,M,K,N,doubleBuffer,t_total_pred,status,errors,iters,warmup,avg_us,min_us,max_us,trace_dispatch_us,trace_kern_pct,trace_gflops,host_overhead_us,ss_iter_cy,ss_kernel_cy,idle_pkg_mw,active_pkg_mw,npu_power_mw,npu_energy_uj,npu_energy_per_iter_uj,wall_elapsed_s,host_steps,matmul_npu_us"
NUM_COLUMNS=36
if [[ ! -f "$RESULT_CSV" ]]; then
  mkdir -p "$(dirname "$RESULT_CSV")"
  echo "$NEW_HEADER" > "$RESULT_CSV"
else
  CUR_HEADER="$(head -n1 "$RESULT_CSV" || true)"
  if ! echo "$CUR_HEADER" | grep -q 'trace_dispatch_us'; then
    echo "info: upgrading result CSV header (adding trace columns) ..."
    tmp_csv="$(mktemp)"
    echo "$NEW_HEADER" > "$tmp_csv"
    # Determine how many columns exist to pad correctly.
    n_cols=$(echo "$CUR_HEADER" | awk -F',' '{print NF}')
    # Target: NUM_COLUMNS columns. Pad missing columns with -1.
    n_pad=$((NUM_COLUMNS - n_cols))
    if [[ $n_pad -gt 0 ]]; then
      pad=$(printf ',-1%.0s' $(seq 1 $n_pad))
      tail -n +2 "$RESULT_CSV" 2>/dev/null | awk -v p="$pad" -F',' '{print $0 p}' >> "$tmp_csv"
    else
      tail -n +2 "$RESULT_CSV" 2>/dev/null >> "$tmp_csv"
    fi
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

  # Remove stale trace files from previous case to prevent cross-contamination
  rm -f "$TRACE_RAW" "$TRACE_JSON" "$TRACE_SUMMARY"

  # 1) extract case -> tc.json
  tmp_out="$(mktemp)"
  if ! jq --indent 4 ".cases[$((idx-1))]" "$INPUT_JSON" > "$tmp_out"; then
    echo "warn: failed to extract case #$idx"
    echo "$idx,0,0,0,0,0,0,0,0,0,0,0,0,false,-,EXTRACT_FAIL,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
    rm -f "$tmp_out"
    # cleanup then continue
    if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
    continue
  fi
  mkdir -p "$(dirname "$OUTPUT_JSON")"
  mv "$tmp_out" "$OUTPUT_JSON"

  # Inject "trace": true into tc.json when trace collection is enabled
  # so that ONNXToAIE pass emits TraceFlowOps in the generated MLIR
  if [[ -n "$TRACE_ENABLED" ]]; then
    jq '. + {"trace": true}' "$OUTPUT_JSON" > "${OUTPUT_JSON}.tmp" \
      && mv "${OUTPUT_JSON}.tmp" "$OUTPUT_JSON"
  fi

  # 2) read fields (levels[] array with tiling params at levels[0])
  numSpm=$(read_num '.numCores')
  SPm=$(read_num '.levels[0].SPm'); SPn=$(read_num '.levels[0].SPn')
  TPm=$(read_num '.levels[0].TPm'); TPk=$(read_num '.levels[0].TPk'); TPn=$(read_num '.levels[0].TPn')
  TM=$(read_num '.levels[0].TM'); TK=$(read_num '.levels[0].TK'); TN=$(read_num '.levels[0].TN')
  M=$(read_num '.M');   K=$(read_num '.K');   N=$(read_num '.N')
  DB_STR=$(read_bool_str '.doubleBuffer')
  T_PRED=$(jq -r '.t_total_pred // "-"' "$OUTPUT_JSON")

  for v in numSpm SPm SPn TPm TPk TPn TM TK TN M K N; do
    val="${!v}"
    if ! [[ "$val" =~ ^-?[0-9]+$ ]]; then
      echo "warn: $v not integer (got: $val); marking as PARSE_FAIL"
      echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,$T_PRED,PARSE_FAIL,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
      if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
      continue 2
    fi
  done

  echo "Params: numSpm=$numSpm | SPm=$SPm SPn=$SPn | TPm=$TPm TPk=$TPk TPn=$TPn | TM=$TM TK=$TK TN=$TN | M=$M K=$K N=$N | doubleBuffer=$DB_STR"

  # 3) generate MLIR
  if [[ -f "$GEN_SCRIPT" ]]; then
    echo "running: bash $GEN_SCRIPT $OUTPUT_JSON"
    if ! bash "$GEN_SCRIPT" "$OUTPUT_JSON"; then
      echo "warn: generator failed"
      echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,$T_PRED,GEN_FAIL,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
      if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
      continue
    fi
  else
    echo "warn: generator script not found: $GEN_SCRIPT"
    echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,$T_PRED,GEN_MISSING,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
    if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
    continue
  fi

  # 4) build & run (JSON_OUTPUT enables structured result parsing via jq)
  JSON_RESULT="$LOGS_DIR/result.json"
  TRACE_MAKE_ARGS=""
  if [[ -n "$TRACE_ENABLED" ]]; then
    # Trace buffer must match MLIR-generated size:
    # TRACE_PER_STREAM(256KB) * compTilesPerCol(4) * 2(core+mem) * numCols
    # numCols = numSpm / compTilesPerCol = numSpm / 4
    TRACE_NUM_COLS=$(( numSpm / 4 ))
    TRACE_SZ=$(( 262144 * 4 * 2 * TRACE_NUM_COLS ))
    TRACE_MAKE_ARGS="TRACE_SZ=$TRACE_SZ TRACE_FILE=$TRACE_RAW"
  fi
  echo "make -C \"$MAKE_DIR\" run JSON_OUTPUT=$JSON_RESULT $TRACE_MAKE_ARGS"
  if ! make -C "$MAKE_DIR" run JSON_OUTPUT="$JSON_RESULT" $TRACE_MAKE_ARGS; then
    echo "warn: make run failed"
    echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,$T_PRED,RUN_FAIL,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1" >> "$RESULT_CSV"
    # if [[ -f "$CLEAN_SCRIPT" ]]; then bash "$CLEAN_SCRIPT" || true; fi
    continue
  fi

  # 5) parse results — prefer JSON (deterministic) over grep (fragile)
  STATUS="FAIL"; ERRORS=-1
  ITERS=-1; WARMUP=-1; AVG_US=-1; MIN_US=-1; MAX_US=-1
  IDLE_PKG_MW=-1; ACTIVE_PKG_MW=-1; NPU_POWER_MW=-1
  NPU_ENERGY_UJ=-1; NPU_ENERGY_PER_ITER_UJ=-1; WALL_ELAPSED_S=-1

  if [[ -f "$JSON_RESULT" ]]; then
    # Structured JSON written by host --json-output; no regex needed.
    STATUS="$(jq -r '.status // "FAIL"' "$JSON_RESULT")"
    ERRORS="$(jq -r '.errors // -1' "$JSON_RESULT")"
    ITERS="$(jq -r '.iterations // -1' "$JSON_RESULT")"
    WARMUP="$(jq -r '.warmup // -1' "$JSON_RESULT")"
    AVG_US="$(jq -r '.avg_us // -1' "$JSON_RESULT")"
    MIN_US="$(jq -r '.min_us // -1' "$JSON_RESULT")"
    MAX_US="$(jq -r '.max_us // -1' "$JSON_RESULT")"
    # Energy fields (host RAPL measurement)
    IDLE_PKG_MW="$(jq -r '.idle_pkg_mw // -1' "$JSON_RESULT")"
    ACTIVE_PKG_MW="$(jq -r '.active_pkg_mw // -1' "$JSON_RESULT")"
    NPU_POWER_MW="$(jq -r '.npu_power_mw // -1' "$JSON_RESULT")"
    NPU_ENERGY_UJ="$(jq -r '.npu_energy_uj // -1' "$JSON_RESULT")"
    NPU_ENERGY_PER_ITER_UJ="$(jq -r '.npu_energy_per_iter_uj // -1' "$JSON_RESULT")"
    WALL_ELAPSED_S="$(jq -r '.wall_elapsed_s // -1' "$JSON_RESULT")"
  elif [[ -f "$LOG_FILE" ]]; then
    # Fallback: grep-based log parsing for backward compatibility.
    if grep -q 'PASS!' "$LOG_FILE"; then
      STATUS="PASS"; ERRORS=0
    else
      last_mis="$(grep -Eo '[0-9]+ mismatches\.' "$LOG_FILE" | tail -n1 | grep -Eo '^[0-9]+')"
      if [[ -n "${last_mis:-}" ]]; then ERRORS="$last_mis"; fi
    fi

    it_line="$(grep -m1 -E '^Number of iterations:' "$LOG_FILE" || true)"
    if [[ -n "$it_line" ]]; then
      ITERS="$(echo "$it_line"  | grep -Eo 'Number of iterations:\s*[0-9]+' | awk '{print $NF}')"
      WARMUP="$(echo "$it_line" | grep -Eo '\(warmup iterations:\s*[0-9]+' | grep -Eo '[0-9]+' || true)"
      : "${ITERS:=-1}"; : "${WARMUP:=-1}"
    fi

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

    for v in AVG_US MIN_US MAX_US; do
      val="${!v}"
      if ! [[ "$val" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf -v "$v" '%s' -1
      fi
    done
  else
    STATUS="NO_LOG"; ERRORS=-1
  fi

  # 6) trace post-processing
  TRACE_DISPATCH_US=-1; TRACE_KERN_PCT=-1; TRACE_GFLOPS=-1; HOST_OVERHEAD_US=-1
  SS_ITER_CY=-1; SS_KERNEL_CY=-1
  HOST_STEPS=-1; MATMUL_NPU_US=-1

  if [[ -n "$TRACE_ENABLED" && -f "$TRACE_RAW" ]]; then
    # Step 1: parse raw hex trace -> trace.json (Perfetto format)
    if python3 "$PARSE_TRACE" --input "$TRACE_RAW" --mlir "$AIE_MLIR" \
         --output "$TRACE_JSON" 2>"$LOGS_DIR/parse_trace_err.txt"; then

      # Step 2: analyze trace -> summary JSON
      if python3 "$ANALYZE_TRACE" --trace "$TRACE_JSON" --tc "$OUTPUT_JSON" \
           --trace-raw "$TRACE_RAW" \
           --n-dispatches 0 \
           --json-summary "$TRACE_SUMMARY" 2>"$LOGS_DIR/analyze_trace_err.txt"; then

        TRACE_DISPATCH_US=$(jq -r '.dispatch_us // -1' "$TRACE_SUMMARY")
        TRACE_KERN_PCT=$(jq -r '.kernel_pct // -1' "$TRACE_SUMMARY")
        TRACE_GFLOPS=$(jq -r '.gflops // -1' "$TRACE_SUMMARY")

        SS_ITER_CY=$(jq -r '.ss_iter_cy // -1' "$TRACE_SUMMARY")
        SS_KERNEL_CY=$(jq -r '.ss_kernel_cy // -1' "$TRACE_SUMMARY")

        HOST_STEPS=$(jq -r '.host_steps // -1' "$TRACE_SUMMARY")
        MATMUL_NPU_US=$(jq -r '.matmul_npu_us // -1' "$TRACE_SUMMARY")

        # Host overhead = host chrono time - NPU trace matmul time (per-matmul units)
        if [[ "$AVG_US" != "-1" && "$MATMUL_NPU_US" != "-1" ]]; then
          HOST_OVERHEAD_US=$(python3 -c "print(round($AVG_US - $MATMUL_NPU_US, 2))")
        fi
      else
        echo "warn: analyze_trace.py failed for case #$idx"
      fi
    else
      echo "warn: parse_trace.py failed for case #$idx"
    fi
  fi

  # 7) append to CSV
  echo "$idx,$numSpm,$SPm,$SPn,$TPm,$TPk,$TPn,$TM,$TK,$TN,$M,$K,$N,$DB_STR,$T_PRED,$STATUS,$ERRORS,$ITERS,$WARMUP,$AVG_US,$MIN_US,$MAX_US,$TRACE_DISPATCH_US,$TRACE_KERN_PCT,$TRACE_GFLOPS,$HOST_OVERHEAD_US,$SS_ITER_CY,$SS_KERNEL_CY,$IDLE_PKG_MW,$ACTIVE_PKG_MW,$NPU_POWER_MW,$NPU_ENERGY_UJ,$NPU_ENERGY_PER_ITER_UJ,$WALL_ELAPSED_S,$HOST_STEPS,$MATMUL_NPU_US" >> "$RESULT_CSV"
  echo "Result: case #$idx -> $STATUS (errors=$ERRORS, avg=${AVG_US}us, min=${MIN_US}us, max=${MAX_US}us) appended to $RESULT_CSV"

  # 7b) archive per-case artifacts before cleanup
  if [[ -n "$ARCHIVE_DIR" ]]; then
    CASE_DIR="$ARCHIVE_DIR/case_$(printf '%03d' "$idx")"
    mkdir -p "$CASE_DIR"
    cp -f "$OUTPUT_JSON"     "$CASE_DIR/tc.json"             2>/dev/null || true
    cp -f "$AIE_MLIR"        "$CASE_DIR/aie.mlir"            2>/dev/null || true
    cp -f "$LOG_FILE"        "$CASE_DIR/log.txt"             2>/dev/null || true
    cp -f "$JSON_RESULT"     "$CASE_DIR/result.json"         2>/dev/null || true
    # Only copy trace files when trace was actually captured (non-empty raw)
    if [[ -s "$TRACE_RAW" ]]; then
      cp -f "$TRACE_RAW"       "$CASE_DIR/trace_raw.txt"       2>/dev/null || true
      cp -f "$TRACE_JSON"      "$CASE_DIR/trace.json"          2>/dev/null || true
      cp -f "$TRACE_SUMMARY"   "$CASE_DIR/trace_summary.json"  2>/dev/null || true
    fi
  fi

  # 8) clean before next case
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
