#!/usr/bin/env bash
# Structural verification — thin wrapper around run_tc_all.sh
# Uses data/structural_cases.json (24 structurally distinct parallelization cases).
#
# Usage: bash scripts/run/run_structural_test.sh [-t] [-n INDEX]
#   -t        Enable NPU trace collection
#   -n INDEX  Run only the INDEX-th case (1-based)
#   All other options are forwarded to run_tc_all.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../common.sh"

exec bash "$SCRIPT_DIR/run_tc_all.sh" \
  -i "$DATA_DIR/structural_cases.json" \
  -r "$REPORTS_DIR/structural_test_result.csv" \
  "$@"
