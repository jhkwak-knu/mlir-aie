#!/usr/bin/env bash
# Usage: ./gen_onnx_matmul_mlir.sh M K N [OUT=out/mlir/onnx_matmul.mlir]

set -euo pipefail

# resolve dirs and load common paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 M K N [OUT]" >&2
  exit 1
fi

M="$1"; K="$2"; N="$3"
OUT="${4:-$MLIR_DIR/onnx_matmul.mlir}"

is_pos_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

if ! is_pos_int "$M" || ! is_pos_int "$K" || ! is_pos_int "$N"; then
  echo "Error: M, K, N must be positive integers." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<MLIR
func.func @test_matmul(%arg0: tensor<${M}x${K}xf32>, %arg1: tensor<${K}x${N}xf32>) -> tensor<${M}x${N}xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<${M}x${K}xf32>, tensor<${K}x${N}xf32>) -> tensor<${M}x${N}xf32>
  return %0 : tensor<${M}x${N}xf32>
}
MLIR

echo "Wrote $OUT"
