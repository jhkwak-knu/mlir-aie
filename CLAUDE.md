# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLIR-AIE is an MLIR-based toolchain targeting AMD Ryzen AI NPUs (AI Engine tiles). This fork adds an **ONNX-to-AIE compilation pass** (`--convert-onnx-to-aie`) that lowers ONNX dialect operations to AIE/AIEX dialect with spatio-temporal tiling for matrix multiplication.

Active branch: `dev`

### Project Goals

1. **Candidate generation & cost estimation** -- Enumerate valid spatio-temporal parallelization configurations for matrix multiplication and compute their estimated cost.
2. **Energy-efficient configuration selection** -- Choose the optimal parallelization setting based on cost (energy efficiency).
3. **MLIR-based compilation** -- Compile the selected configuration to AIE/AIEX dialect using the `--convert-onnx-to-aie` pass (MLIR-AIE toolchain).
4. **Hardware validation & benchmarking** -- Execute on a real Ryzen AI NPU (XDNA2) to verify correctness and measure performance.

---

## Critical Rules

### NPU measurement
- Before running NPU tests or batches, use the `npu-batch-run` skill (pre-checks, first-case
  validation, monitoring, hang recovery).
- An NPU hang shows as `onnx_matmul` in `D` (disk sleep) state; only a reboot recovers it.
  Stop and tell the user instead of retrying.

### Result versioning
Every measurement batch must be traceable:
- Name result files with version/phase: `result_phase1.csv`, `result_v8.csv`
- Record the git commit hash at measurement time
- Record the tc_list file used (with config count)
- Archive previous results before overwriting (never delete raw data)

---

## Environment Setup

Before running any build or test, activate the environment (must be **sourced**, not executed):

```bash
source test/onnx-mlir/scripts/setup_env.sh
```

This activates `ironenv` (Python venv), runs `utils/env_setup.sh <install_dir>` (sets `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`), and verifies the `amdxdna` TDR timeout is 60000 ms (set at module load via `/etc/modprobe.d/amdxdna.conf`). Install dir is `<repo>/install` (repo root is derived from the script location). XRT must be sourced first (`source /opt/xilinx/xrt/setup.sh`); the sudo steps (timeout check, RAPL chmod) need a password, so run `setup_env.sh` yourself.

---

## Build & Test Commands (ONNX-to-AIE Pass)

All commands run from `test/onnx-mlir/`.

Laptop memory limit: each Chess core compile in `aiecc` peaks at ~6-7GB. Keep `aiecc -j 2`
(Makefile default); `-j 12` froze the machine. Rebuilding the compiler: `ninja -C build -j 12` under `nice -n 19`.

### Step 1 -- Generate test case list
```bash
# Every valid candidate (for cost-model validation sweeps)
python3 scripts/generate/cost_model.py --op data/op_list.json --tc-list out/tc_list.json

# Top-1 per workload only (final pick)
python3 scripts/generate/cost_model.py --op data/op_list.json --top-n 1 --tc-list out/tc_list.json

# Dry-run (in-memory enumeration + rank, no file output):
python3 scripts/generate/cost_model.py --op data/op_list.json --op-index 0
```
Notes:
- `--search {sm-exh, star-map, charm, timeloop}` selects the searcher (default `sm-exh`).
- `--validate` is a deprecated alias of `--tc-list` (still accepted, prints a warning).

### Step 2 -- Run all test cases (batch)
```bash
bash scripts/run/run_tc_all.sh -i out/tc_list.json -r out/reports/result.csv
```

### Step 3 -- Run a single test case (by 1-based index)
```bash
bash scripts/run/run_tc_all.sh -i out/tc_list.json -n 1
# Artifacts kept in out/ for inspection (no auto-clean in single-case mode)
```

### Manual single-case flow
```bash
# 1. Put one case in out/tc.json, then generate ONNX MLIR from it (M/K/N read from tc.json)
bash scripts/generate/gen_onnx_matmul_mlir.sh out/tc.json

# 2. Full build + run (reads out/tc.json for tiling config)
make run

# 3. Build only (xclbin)
make

# 4. Rebuild host only
make host

# 5. Clean build artifacts
make clean
```

### Debug flags (add to Makefile DEBUGS or pass directly)
```bash
aie-opt out/mlir/onnx_matmul.mlir --convert-onnx-to-aie \
  --debug-system-info --debug-tile-param --debug-aie-placement
```

---

## Architecture: ONNXToAIE Pass

**Pass registration**: `onnx::createConvertONNXToAIEPass()` on `func::FuncOp`, registered via `Passes.td` -> `aie-opt`.

**Tiling config input**: The pass reads `tc.json` at compile time to load spatio-temporal tiling parameters (see Key Implementation Notes for path details).

**AIE MLIR generation order**:
`DeviceOp` -> `TileOp` -> `BufferOp`/`LockOp` -> `FlowOp` (packet routing) -> `DmaOp` (MemTileDMAOp/MemOp/DMABdOp) -> `CoreOp`/`CallOp` (with `SCF ForOp`) -> `RuntimeSequenceOp`/`NpuDmaMemcpyNdOp`

**Spatio-temporal parallelization**:
- **Spatial**: `SPm x SPn` compute tiles handle M/N axis splits (`M0 = M/SPm`, `N0 = N/SPn`)
- **Temporal**: `TPm x TPk x TPn` sequential iterations per tile, innermost tile = `TM x TK x TN`
- **Memory constraint**: `elem_bytes x (TM*TK + TK*TN + TM*TN) <= 60KB` (CTILE_MEM_LIMIT)
- **tpOrder**: `[axis0, axis1, axis2]` -- axis indices `0=M, 1=N, 2=K` -- controls loop nesting order selected by data-reuse analysis in `cost_model.py`

**Packet routing**: Static packet IDs are pre-allocated per tile to avoid packet filtering bugs. Output quadrant IDs map: `1,2,4,8 -> 0,1,2,3`.

---

## Test Data & References

- `data/op_list.json` -- input operations (M, K, N, elemType=f32)
- `data/refs/mlir/` -- reference AIE MLIR for 1-4 core configurations
- `data/refs/tc_list_ver1/` -- reference tc_list.json per matrix size
- `data/refs/result_ver1/` -- reference result.csv (correctness + timing baseline)

Result CSV (39 columns since commit 7c984731):
- `min_us`: NPU dispatch+wait only (cost model calibration)
- `step_min_us`: memcpy+sync+dispatch (matches energy measurement scope, use for EDP)

---

## Key Implementation Notes

- `aie-opt` is the modified version from this repo's `tools/aie-opt/`, not the installed system one. Ensure `$PATH` from `env_setup.sh` points to the local install.
- `xdna2_info.json` target is XDNA2 (Ryzen AI Strix/Strix Halo/Krackan); `NPU2=1` is set in environment for these devices.
- Double-buffering is defined in `TileParam` but currently disabled (`doubleBuffer=false`) in test case generation.
- Kernel function signature: `void extern_kernel(bfloat16* A, bfloat16* B, bfloat16* C, uint32_t N_ROW, uint32_t N_COL, uint32_t N_DEP, bool acc)` -- B is transposed.

---

## Git Workflow

- Active branch: `dev`
- Result CSV columns changed at commit 7c984731 (39 columns with step_time)
