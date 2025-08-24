#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$THIS_DIR/.." && pwd)"

SRC_DIR="$ROOT_DIR/src"
SCRIPTS_DIR="$ROOT_DIR/scripts"
DATA_DIR="$ROOT_DIR/data"
CONFIG_DIR="$ROOT_DIR/config"
OUT_DIR="$ROOT_DIR/out"
BUILD_DIR="$OUT_DIR/build"
ONNX_DIR="$OUT_DIR/onnx"
MLIR_DIR="$OUT_DIR/mlir"
LOGS_DIR="$OUT_DIR/logs"
REPORTS_DIR="$OUT_DIR/reports"

TC_LIST="$DATA_DIR/tc_list.json"
REF_DIR="$DATA_DIR/ref"

mkdir -p "$BUILD_DIR" "$ONNX_DIR" "$MLIR_DIR" "$LOGS_DIR" "$REPORTS_DIR"
