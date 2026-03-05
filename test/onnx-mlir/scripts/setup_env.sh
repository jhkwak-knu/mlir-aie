#!/usr/bin/env bash
set -uo pipefail

# Set parameters
MLIR_AIE_ROOT="$HOME/ryzen_ai/mlir-aie-dev/mlir-aie"
IRONENV_ACTIVATE="$MLIR_AIE_ROOT/ironenv/bin/activate"
ENV_SETUP_SH="$MLIR_AIE_ROOT/utils/env_setup.sh"
MLIR_AIE_INSTALL="$MLIR_AIE_ROOT/install"

AMD_XDNA_TIMEOUT_PATH="/sys/module/amdxdna/parameters/timeout_in_sec"
AMD_XDNA_TIMEOUT_SEC=60

# Activate ironenv (affects current shell because this script is sourced)
if [[ ! -f "$IRONENV_ACTIVATE" ]]; then
  echo "[error] activate not found: $IRONENV_ACTIVATE" >&2
  return 1 2>/dev/null || exit 1
fi
source "$IRONENV_ACTIVATE"

# Run env setup
if [[ ! -f "$ENV_SETUP_SH" ]]; then
  echo "[error] env_setup.sh not found: $ENV_SETUP_SH" >&2
  return 1 2>/dev/null || exit 1
fi
source "$ENV_SETUP_SH" "$MLIR_AIE_INSTALL"
echo

# Set amdxdna timeout (seconds)
if [[ ! -f "$AMD_XDNA_TIMEOUT_PATH" ]]; then
  echo "[error] timeout parameter file not found: $AMD_XDNA_TIMEOUT_PATH" >&2
  return 1 2>/dev/null || exit 1
fi
echo "[info] amdxdna timeout (before): $(cat "$AMD_XDNA_TIMEOUT_PATH") sec"

if ! echo "$AMD_XDNA_TIMEOUT_SEC" | sudo tee "$AMD_XDNA_TIMEOUT_PATH" >/dev/null; then
  echo "[error] failed to write $AMD_XDNA_TIMEOUT_SEC to $AMD_XDNA_TIMEOUT_PATH (need sudo/root?)" >&2
  return 1 2>/dev/null || exit 1
fi
echo "[info] amdxdna timeout (after):  $(cat "$AMD_XDNA_TIMEOUT_PATH") sec"
echo

# Print the information of virtual env
echo "[info] Activated: ${VIRTUAL_ENV:-"(unknown)"}"
echo "[info] PWD (after): $(pwd)"
command -v python || true
python --version || true
