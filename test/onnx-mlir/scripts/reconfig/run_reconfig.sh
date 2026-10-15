#!/usr/bin/env bash
# Run the reconfiguration differential microbenchmark over a rotation-depth
# sweep for one already-built config set.
#
# Prereq:
#   source test/onnx-mlir/scripts/setup_env.sh
#   python3 scripts/reconfig/build_configs.py --shape M K N --shape-tag TAG --depth DMAX
#
# Usage:
#   bash scripts/reconfig/run_reconfig.sh <shape-tag> [dmax] [depths...]
# Example:
#   bash scripts/reconfig/run_reconfig.sh bert_qkv_128x512x512 4 "2 3 4"
set -euo pipefail

ONNX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ONNX_DIR"

SHAPE_TAG="${1:?usage: run_reconfig.sh <shape-tag> [depths...]}"
DEPTHS="${2:-2 3 4}"
CONFIG_DIR="out/reconfig/${SHAPE_TAG}"
BENCH="out/build/reconfig_bench"
OUT_CSV="out/reports/reconfig_v1.csv"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

# --- Pre-test environment checks (CLAUDE.md hardware protocol) ---
echo "== pre-flight =="
if ! cat /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1; then
  echo "WARNING: RAPL not readable; energy columns will be -1" >&2
fi
TO=$(cat /sys/module/amdxdna/parameters/timeout_in_sec 2>/dev/null || echo "?")
echo "amdxdna timeout_in_sec = ${TO}"
[ -d "$CONFIG_DIR" ] || { echo "missing config dir: $CONFIG_DIR (run build_configs.py first)"; exit 1; }

# --- Build the reconfig host LAST (build_configs.py's make clean wipes out/build) ---
echo "== building reconfig host =="
make host_reconfig >/dev/null

mkdir -p out/reports
echo "== sweep depths: $DEPTHS  (run_tag=$RUN_TAG) =="
for d in $DEPTHS; do
  echo "--- depth $d ---"
  "$BENCH" \
    --config-dir "$CONFIG_DIR" \
    --depth "$d" \
    --shape-tag "$SHAPE_TAG" \
    --run-tag "$RUN_TAG" \
    --out-csv "$OUT_CSV" \
    --n-batches 5 \
    --target-batch-wall-s 0.5 \
    --n-warmup 20
done

echo "== done -> $OUT_CSV =="
tail -n +1 "$OUT_CSV" | tail -20
