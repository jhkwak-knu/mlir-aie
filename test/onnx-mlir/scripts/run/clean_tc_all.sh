#!/usr/bin/env bash
# Clean per-case artifacts in the new tree layout.

set -euo pipefail

# resolve dirs and load common paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../common.sh"

# 1) build clean (Makefile lives at repo root)
if [[ -f "$ROOT_DIR/Makefile" ]]; then
  make -C "$ROOT_DIR" clean || true
fi

# 2) remove per-case files
rm -f "$OUT_DIR/tc.json"
rm -f "$MLIR_DIR/onnx_matmul.mlir"

# 3) (optional) clear last log to avoid stale parsing
rm -f "$LOGS_DIR/log.txt"

# 4) remove trace artifacts
rm -f "$LOGS_DIR/trace_raw.txt"
rm -f "$LOGS_DIR/trace.json"
rm -f "$LOGS_DIR/trace_summary.json"
rm -f "$LOGS_DIR/parse_trace_err.txt"
rm -f "$LOGS_DIR/analyze_trace_err.txt"
