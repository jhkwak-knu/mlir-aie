#!/usr/bin/env bash
# Usage: ./gen_onnx_matmul_mlir.sh TC_JSON [OUT=out/mlir/onnx_matmul.mlir]

set -euo pipefail

# resolve dirs and load common paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../common.sh"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 TC_JSON [OUT]" >&2
  exit 1
fi

TC_JSON_PATH="$1"
OUT="${2:-$MLIR_DIR/onnx_matmul.mlir}"

if [[ ! -f "$TC_JSON_PATH" ]]; then
  echo "Error: tc.json not found: $TC_JSON_PATH" >&2
  exit 1
fi

is_pos_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

M=$(jq '.M'                            "$TC_JSON_PATH")
K=$(jq '.K'                            "$TC_JSON_PATH")
N=$(jq '.N'                            "$TC_JSON_PATH")
ELEM_TYPE=$(jq -r '.elemType // "f32"' "$TC_JSON_PATH")

if ! is_pos_int "$M" || ! is_pos_int "$K" || ! is_pos_int "$N"; then
  echo "Error: M, K, N must be positive integers (got M=$M K=$K N=$N)." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<MLIR
func.func @test_matmul(%arg0: tensor<${M}x${K}x${ELEM_TYPE}>, %arg1: tensor<${K}x${N}x${ELEM_TYPE}>) -> tensor<${M}x${N}x${ELEM_TYPE}> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<${M}x${K}x${ELEM_TYPE}>, tensor<${K}x${N}x${ELEM_TYPE}>) -> tensor<${M}x${N}x${ELEM_TYPE}>
  return %0 : tensor<${M}x${N}x${ELEM_TYPE}>
}
MLIR

echo "Wrote $OUT"
