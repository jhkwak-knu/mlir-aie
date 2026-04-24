#!/usr/bin/env bash
# run_tc_chunks.sh — Workload-chunked measurement orchestrator.
#
# Iterates tc_list_v14.json one workload (shape) at a time, performs an
# idle pre-check, runs run_tc_all.sh for the shape's contiguous case
# range, validates the resulting CSV rows (idle, CV), and sleeps
# between chunks to let the system cool down.
#
# Usage:
#   ./run_tc_chunks.sh [-i INPUT_JSON] [-r RESULT_CSV] [-l LOG]
#                      [-c COOLDOWN_SEC] [--resume N] [--dry-run]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_JSON="$ONNX_DIR/out/tc_list_v14.json"
RESULT_CSV="$ONNX_DIR/out/reports/result_v14.csv"
LOG="$ONNX_DIR/out/reports/result_v14_chunks.log"
COOLDOWN_SEC=120
RESUME_CHUNK=1
DRY_RUN=0

# Validation thresholds (per chunk, median-only; max is reported but not gating).
# Tightened 2026-04-16 based on observed clean-run distributions.
IDLE_PRE_MAX_MW=3500
IDLE_MED_MAX_MW=3500
ECV_MED_MAX=5.0
SCV_MED_MAX=3.0
MAX_RETRIES=2
RETRY_COOLDOWN_SEC=150

# ---------- CLI ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) INPUT_JSON="$2"; shift 2 ;;
    -r) RESULT_CSV="$2"; shift 2 ;;
    -l) LOG="$2"; shift 2 ;;
    -c) COOLDOWN_SEC="$2"; shift 2 ;;
    --resume) RESUME_CHUNK="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '1,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$(dirname "$RESULT_CSV")" "$(dirname "$LOG")"

# Activate venv + PATH (skip sudo-requiring bits of setup_env.sh so this
# runs cleanly in non-interactive background shells).
MLIR_AIE_ROOT="$(cd "$ONNX_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$MLIR_AIE_ROOT/ironenv/bin/activate" 2>/dev/null || true
# shellcheck disable=SC1091
source "$MLIR_AIE_ROOT/utils/env_setup.sh" "$MLIR_AIE_ROOT/install" >/dev/null 2>&1 || true

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[%s] %s\n' "$ts" "$*" | tee -a "$LOG"
}

# ---------- Environment checks ----------
check_env() {
  if ! [[ -r /sys/class/powercap/intel-rapl:0/energy_uj ]]; then
    log "ERROR: RAPL pkg not readable"; return 1
  fi
  local t
  t="$(cat /sys/module/amdxdna/parameters/timeout_in_sec 2>/dev/null || echo 0)"
  if [[ "$t" != "60" ]]; then
    log "WARN: amdxdna timeout=$t (expected 60); continuing"
  fi
  rm -f "$ONNX_DIR/out/tc.json"
  rm -rf "$ONNX_DIR/out/build"
  return 0
}

# Measure idle pkg power (mW) as median of 5 x 0.2s samples.
measure_idle_mw() {
  python3 - <<'PY'
import time
P = "/sys/class/powercap/intel-rapl:0/energy_uj"
samples = []
for _ in range(5):
    with open(P) as f: a = int(f.read())
    time.sleep(0.2)
    with open(P) as f: b = int(f.read())
    samples.append((b - a) / (0.2 * 1000.0))
samples.sort()
print(f"{samples[len(samples)//2]:.0f}")
PY
}

# Wait until idle < IDLE_PRE_MAX_MW, max 3 x 60s attempts.
wait_for_cool_idle() {
  local attempt=1
  while (( attempt <= 3 )); do
    local m
    m="$(measure_idle_mw)"
    log "  idle_probe attempt $attempt: ${m} mW"
    if (( m < IDLE_PRE_MAX_MW )); then
      return 0
    fi
    log "  idle > $IDLE_PRE_MAX_MW mW, sleeping 60s"
    sleep 60
    attempt=$((attempt + 1))
  done
  log "  idle probe failed 3x, proceeding anyway"
  return 1
}

# ---------- Chunk plan ----------
# Returns lines: "shape_key|start|end|n_cases"
plan_chunks() {
  python3 - "$INPUT_JSON" <<'PY'
import json, sys
from collections import OrderedDict
p = sys.argv[1]
d = json.load(open(p))
cases = d["cases"] if isinstance(d, dict) else d
by = OrderedDict()
for i, c in enumerate(cases, 1):
    k = (c["M"], c["K"], c["N"])
    if k not in by:
        by[k] = [i, i]
    else:
        by[k][1] = i
# Output chunks sorted by workload size (smallest first).
ordered = sorted(by.items(), key=lambda kv: kv[0][0] * kv[0][1] * kv[0][2])
for k, (s, e) in ordered:
    shape = f"{k[0]}x{k[1]}x{k[2]}"
    print(f"{shape}|{s}|{e}|{e - s + 1}")
PY
}

# Validate last N rows of RESULT_CSV (where N = expected chunk size).
validate_chunk() {
  local shape="$1" start="$2" end="$3"
  python3 - "$RESULT_CSV" "$shape" "$start" "$end" "$IDLE_MED_MAX_MW" \
    "$ECV_MED_MAX" "$SCV_MED_MAX" <<'PY'
import csv, sys, statistics
path, shape, s, e, idle_med_max, ecv_med_max, scv_med_max = sys.argv[1:]
s, e = int(s), int(e)
idle_med_max = float(idle_med_max)
ecv_med_max = float(ecv_med_max)
scv_med_max = float(scv_med_max)
mshape = tuple(int(x) for x in shape.split("x"))
with open(path) as f:
    rows = list(csv.DictReader(l for l in f if not l.lstrip().startswith("#")))
# Pick last occurrence per case_index within chunk range and matching shape.
keep = {}
for r in rows:
    if r.get("status") != "PASS":
        continue
    try:
        ci = int(r["case_index"])
        if not (s <= ci <= e): continue
        if (int(r["M"]), int(r["K"]), int(r["N"])) != mshape: continue
    except Exception:
        continue
    keep[ci] = r
rows = list(keep.values())
if not rows:
    print("FAIL no_rows")
    sys.exit(1)
def col(name):
    out = []
    for r in rows:
        v = r.get(name)
        if v is None or v in ("", "-1", "-1.0"): continue
        try: out.append(float(v))
        except: pass
    return out
idle = col("idle_pkg_mw")
ecv  = col("batch_energy_cv_pct")
scv  = col("batch_step_cv_pct")
probs = []
if idle and statistics.median(idle) > idle_med_max:
    probs.append(f"idle_med={statistics.median(idle):.0f}>{idle_med_max:.0f}")
if ecv and statistics.median(ecv) > ecv_med_max:
    probs.append(f"ecv_med={statistics.median(ecv):.2f}>{ecv_med_max}")
if scv and statistics.median(scv) > scv_med_max:
    probs.append(f"scv_med={statistics.median(scv):.2f}>{scv_med_max}")
n_pass = sum(1 for r in rows)
def fmt(vals, prec=0):
    if not vals: return "N/A"
    return f"{min(vals):.{prec}f}/{statistics.median(vals):.{prec}f}/{max(vals):.{prec}f}"
print(f"SUMMARY n_pass={n_pass} idle={fmt(idle,0)}mW ecv={fmt(ecv,2)}% scv={fmt(scv,2)}%")
if probs:
    print("FAIL " + " ".join(probs)); sys.exit(1)
print("PASS"); sys.exit(0)
PY
}

# Run one chunk (with retries).
run_chunk() {
  local n="$1" shape="$2" start="$3" end="$4" total="$5"
  local attempt=1
  while (( attempt <= MAX_RETRIES + 1 )); do
    log "chunk $n/$total [$shape] attempt $attempt: idx $start..$end"
    wait_for_cool_idle
    bash "$SCRIPT_DIR/run_tc_all.sh" \
      -i "$INPUT_JSON" -r "$RESULT_CSV" -s "$start" -e "$end" \
      >>"$LOG" 2>&1 || true
    local v_out
    v_out="$(validate_chunk "$shape" "$start" "$end" 2>&1 || true)"
    echo "$v_out" | tee -a "$LOG" >/dev/null
    log "  validate: $v_out"
    if echo "$v_out" | grep -q "^PASS$"; then
      return 0
    fi
    if (( attempt <= MAX_RETRIES )); then
      log "  retry after ${RETRY_COOLDOWN_SEC}s cooldown"
      sleep "$RETRY_COOLDOWN_SEC"
    fi
    attempt=$((attempt + 1))
  done
  log "chunk $n [$shape] FAILED after $MAX_RETRIES retries — continuing"
  return 1
}

# ---------- Main ----------
log "=== run_tc_chunks start ==="
log "input=$INPUT_JSON  csv=$RESULT_CSV  cooldown=${COOLDOWN_SEC}s  resume=$RESUME_CHUNK"

mapfile -t CHUNKS < <(plan_chunks)
TOTAL="${#CHUNKS[@]}"
log "planned $TOTAL chunks"

if (( DRY_RUN == 1 )); then
  log "DRY RUN:"
  for i in "${!CHUNKS[@]}"; do
    n=$((i + 1))
    IFS='|' read -r shape start end ncases <<< "${CHUNKS[$i]}"
    log "  chunk $n: $shape  idx $start..$end  ($ncases cases)"
  done
  exit 0
fi

check_env || { log "ENV CHECK FAILED"; exit 1; }

FAILED=()
for i in "${!CHUNKS[@]}"; do
  n=$((i + 1))
  (( n < RESUME_CHUNK )) && continue
  IFS='|' read -r shape start end ncases <<< "${CHUNKS[$i]}"
  if ! run_chunk "$n" "$shape" "$start" "$end" "$TOTAL"; then
    FAILED+=("$n:$shape")
  fi
  if (( n < TOTAL )); then
    log "cooldown ${COOLDOWN_SEC}s"
    sleep "$COOLDOWN_SEC"
  fi
done

log "=== run_tc_chunks done ==="
if (( ${#FAILED[@]} > 0 )); then
  log "FAILED chunks: ${FAILED[*]}"
  exit 2
fi
exit 0
