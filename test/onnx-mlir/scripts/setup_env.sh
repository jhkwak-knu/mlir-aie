#!/usr/bin/env bash
set -uo pipefail

# Usage: source setup_env.sh [--measure]
#   --measure: additionally apply measurement_env settings (CPU governor, freq, C-states, etc.)

_SETUP_MEASURE=0
if [[ "${1:-}" == "--measure" ]]; then
  _SETUP_MEASURE=1
  shift
fi

# Set parameters
MLIR_AIE_ROOT="$HOME/ryzen_ai/mlir-aie-dev/mlir-aie"
IRONENV_ACTIVATE="$MLIR_AIE_ROOT/ironenv/bin/activate"
ENV_SETUP_SH="$MLIR_AIE_ROOT/utils/env_setup.sh"
MLIR_AIE_INSTALL="$MLIR_AIE_ROOT/install"

AMD_XDNA_TIMEOUT_PATH="/sys/module/amdxdna/parameters/timeout_in_sec"
AMD_XDNA_TIMEOUT_SEC=60

RAPL_ENERGY_PATH="/sys/class/powercap/intel-rapl:0/energy_uj"

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

# Grant read access to RAPL energy counters (resets on reboot)
RAPL_CORE_PATH="/sys/class/powercap/intel-rapl:0:0/energy_uj"
for _rp in "$RAPL_ENERGY_PATH" "$RAPL_CORE_PATH"; do
  if [[ -f "$_rp" ]]; then
    if [[ ! -r "$_rp" ]]; then
      echo "[info] granting read access to $_rp ..."
      if ! sudo chmod o+r "$_rp"; then
        echo "[warn] failed to chmod $_rp; energy measurement may be unavailable" >&2
      fi
    fi
    echo "[info] RAPL counter readable: $_rp"
  else
    echo "[warn] RAPL counter not found: $_rp" >&2
  fi
done
echo

# Print the information of virtual env
echo "[info] Activated: ${VIRTUAL_ENV:-"(unknown)"}"
echo "[info] PWD (after): $(pwd)"
command -v python || true
python --version || true

# Apply measurement environment if --measure was passed
MEASUREMENT_ENV_SH="$MLIR_AIE_ROOT/test/onnx-mlir/scripts/run/measurement_env.sh"
if [[ "$_SETUP_MEASURE" -eq 1 ]]; then
  echo
  if [[ -f "$MEASUREMENT_ENV_SH" ]]; then
    sudo bash "$MEASUREMENT_ENV_SH" apply
  else
    echo "[warn] measurement_env.sh not found: $MEASUREMENT_ENV_SH" >&2
  fi
else
  # Always show current status for awareness
  if [[ -f "$MEASUREMENT_ENV_SH" ]]; then
    echo
    bash "$MEASUREMENT_ENV_SH" status
  fi
fi
unset _SETUP_MEASURE
