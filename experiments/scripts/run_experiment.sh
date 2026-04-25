#!/usr/bin/env bash
# Single-shot orchestrator for the MLP / BERT / DistilBERT EDP experiments.
#
# Reads `model.type` from <config.json> and runs the full pipeline:
#   1. extract_shapes.py  -> <results_dir>/shapes.json
#   2. generate_configs.py -> <results_dir>/configurations.json
#   3. compile_kernels.py -> <results_dir>/kernel_binaries/<shape>_<setter>/
#   4. cmake --build src/<mlp_runner|distilbert_runner>/build
#   5. measure.py --backend <npu|cpu>
#   6. analyze.py + (for bert/distilbert) analyze_distilbert_step6.py
#
# Assumes `source test/onnx-mlir/scripts/setup_env.sh --measure` has been
# sourced in the parent shell so PATH / amdxdna timeout / RAPL access are
# already configured. Aborts on the first failure.
#
# Usage:
#   experiments/scripts/run_experiment.sh <config.json> [options]
#
# Options:
#   --backend npu|cpu             passed to measure.py (default: npu)
#   --skip-compile                skip Step 3 (kernel compile) — reuse cache
#   --skip-build                  skip Step 4 (cmake build of the runner)
#   --skip-measure                skip Step 5 (e.g. compile-only run)
#   --skip-analyze                skip Step 6 (aggregation)
#   --inter-setter-idle-seconds N forwarded to measure.py (default: 30)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config.json> [--backend npu|cpu] [--skip-compile] [--skip-build] [--skip-measure] [--skip-analyze] [--inter-setter-idle-seconds N]" >&2
  exit 1
fi

CONFIG="$1"
shift

BACKEND="npu"
SKIP_COMPILE=0
SKIP_BUILD=0
SKIP_MEASURE=0
SKIP_ANALYZE=0
IDLE_SECONDS="30"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --skip-compile) SKIP_COMPILE=1; shift;;
    --skip-build) SKIP_BUILD=1; shift;;
    --skip-measure) SKIP_MEASURE=1; shift;;
    --skip-analyze) SKIP_ANALYZE=1; shift;;
    --inter-setter-idle-seconds) IDLE_SECONDS="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Python is required for everything; we use the same interpreter throughout
# so that ironenv / PYTHONPATH stay consistent across stages.
PY="${PYTHON:-python}"

# Pull model.type and results_dir from the config without depending on jq.
MODEL_TYPE="$("$PY" - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print((cfg.get("model") or {}).get("type", ""))
PY
)"

if [[ -z "$MODEL_TYPE" ]]; then
  echo "config $CONFIG: missing model.type" >&2
  exit 1
fi

case "$MODEL_TYPE" in
  mlp)              RUNNER_DIR="src/mlp_runner" ;;
  distilbert|bert)  RUNNER_DIR="src/distilbert_runner" ;;
  *) echo "unsupported model.type='$MODEL_TYPE' (expected mlp/distilbert/bert)" >&2; exit 1;;
esac

RESULTS_DIR="$("$PY" - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
cfg = json.load(open(sys.argv[1]))
rd = (cfg.get("output") or {}).get("results_dir")
if rd:
    print(Path(rd).expanduser())
else:
    print(Path("experiments/results") / Path(sys.argv[1]).stem)
PY
)"

echo "=== run_experiment.sh ==="
echo "config        : $CONFIG"
echo "model.type    : $MODEL_TYPE"
echo "runner_dir    : $RUNNER_DIR"
echo "results_dir   : $RESULTS_DIR"
echo "backend       : $BACKEND"
echo "skip_build    : $SKIP_BUILD"
echo "skip_measure  : $SKIP_MEASURE"
echo "skip_analyze  : $SKIP_ANALYZE"
echo

echo "[1/6] extract_shapes"
"$PY" experiments/scripts/extract_shapes.py --config "$CONFIG"

echo "[2/6] generate_configs"
"$PY" experiments/scripts/generate_configs.py --config "$CONFIG"

if [[ "$SKIP_COMPILE" -eq 0 ]]; then
  echo "[3/6] compile_kernels"
  "$PY" experiments/scripts/compile_kernels.py --config "$CONFIG"
else
  echo "[3/6] compile_kernels SKIPPED"
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[4/6] cmake build $RUNNER_DIR"
  if [[ ! -d "$RUNNER_DIR/build" ]]; then
    cmake -S "$RUNNER_DIR" -B "$RUNNER_DIR/build"
  fi
  cmake --build "$RUNNER_DIR/build" -j
else
  echo "[4/6] cmake build SKIPPED"
fi

if [[ "$SKIP_MEASURE" -eq 1 ]]; then
  echo "[5/6] measure SKIPPED"
  echo "done (stopped before measurement)."
  exit 0
fi

echo "[5/6] measure --backend $BACKEND --inter-setter-idle-seconds $IDLE_SECONDS"
"$PY" experiments/scripts/measure.py \
    --config "$CONFIG" \
    --backend "$BACKEND" \
    --inter-setter-idle-seconds "$IDLE_SECONDS"

if [[ "$SKIP_ANALYZE" -eq 1 ]]; then
  echo "[6/6] analyze SKIPPED"
  echo "done (stopped before analysis)."
  exit 0
fi

echo "[6/6] analyze"
"$PY" experiments/scripts/analyze.py \
    --measurements-csv "$RESULTS_DIR/measurements.csv" \
    --output-dir "$RESULTS_DIR"

# bert/distilbert get the additional setter-by-setter analysis
# (gemm_type decomposition + STAR-Map vs max_p comparison).
if [[ "$MODEL_TYPE" == "distilbert" || "$MODEL_TYPE" == "bert" ]]; then
  echo "[6/6] analyze_distilbert_step6"
  "$PY" experiments/scripts/analyze_distilbert_step6.py \
      --results-dir "$RESULTS_DIR"
fi

echo
echo "done. results in $RESULTS_DIR"
