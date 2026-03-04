# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goals

1. **Candidate generation & cost estimation** — Enumerate valid spatio-temporal parallelization configurations for matrix multiplication and compute their estimated cost.
2. **Energy-efficient configuration selection** — Choose the optimal parallelization setting based on cost (energy efficiency).
3. **MLIR-based compilation** — Compile the selected configuration to AIE/AIEX dialect using the `--convert-onnx-to-aie` pass (MLIR-AIE toolchain).
4. **Hardware validation & benchmarking** — Execute on a real Ryzen AI NPU (XDNA2) to verify correctness and measure performance.

---

## Repository Overview

MLIR-AIE is an MLIR-based toolchain targeting AMD Ryzen™ AI NPUs (AI Engine tiles). This fork adds an **ONNX-to-AIE compilation pass** (`--convert-onnx-to-aie`) that lowers ONNX dialect operations to AIE/AIEX dialect with spatio-temporal tiling for matrix multiplication.

Active branch: `spatio-temporal_parallelization`

---

## Environment Setup

Before running any build or test, activate the environment (must be **sourced**, not executed):

```bash
source test/onnx-mlir/scripts/setup_env.sh
```

This activates `ironenv` (Python venv), runs `utils/env_setup.sh <install_dir>` (sets `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`), and sets the `amdxdna` driver timeout to 60 seconds. Install dir is `$HOME/ryzen_ai/mlir-aie-dev/mlir-aie/install`.

---

## Build & Test Commands (ONNX-to-AIE Pass)

All commands run from `test/onnx-mlir/`.

### Step 1 – Generate test case list
```bash
python3 scripts/gen_tc_list.py --op data/op_list.json --out out/tc_list.json
# Dry-run (validate only, no output):
python3 scripts/gen_tc_list.py --op data/op_list.json --dry-run
```

### Step 2 – Run all test cases (batch)
```bash
bash scripts/run_tc_all.sh -i out/tc_list.json -r out/reports/result.csv
```

### Step 3 – Run a single test case (by 1-based index)
```bash
bash scripts/run_tc_all.sh -i out/tc_list.json -n 1
# Artifacts kept in out/ for inspection (no auto-clean in single-case mode)
```

### Manual single-case flow
```bash
# 1. Generate ONNX MLIR (M K N)
bash scripts/gen_onnx_matmul_mlir.sh 64 64 64

# 2. Full build + run (reads out/tc.json for tiling config)
make run CPPDEFS="-DM_SIZE=64 -DK_SIZE=64 -DN_SIZE=64"

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

## Added Source Structure

```
include/onnx/
├── Conversion/
│   ├── ONNXToAIE/ONNXToAIE.h   # Pass entry point declaration
│   ├── Passes.h / Passes.td     # Pass registration (TableGen)
├── Dialect/ONNX/IR/             # ONNX dialect (MatMul, minimal OpSet-22)
└── Target/XDNA2/xdna2_info.json # Hardware config (32 SPMs, 64KB each)

lib/Conversion/ONNXToAIE/
└── ONNXToAIE.cpp                # Main pass implementation (~700+ lines)

lib/Dialect/ONNX/IR/
├── ONNXDialect.cpp              # Dialect registration
└── ONNXOps.cpp                  # Op implementations

tools/aie-opt/aie-opt.cpp        # Registers --convert-onnx-to-aie flag

test/onnx-mlir/
├── Makefile / CMakeLists.txt    # Build pipeline
├── src/kernel.cc                # AIE compute kernel (C++ MatMul)
├── src/host.cpp                 # XRT host orchestrator (~647 lines)
├── scripts/                     # Shell/Python automation
└── data/op_list.json            # Input: list of (M,K,N,elemType) ops
```

---

## Architecture: ONNXToAIE Pass

**Pass registration**: `onnx::createConvertONNXToAIEPass()` on `func::FuncOp`, registered via `Passes.td` → `aie-opt`.

**Tiling config input**: The pass reads `test/onnx-mlir/out/tc.json` at compile time (hardcoded path, line 307 of `ONNXToAIE.cpp`) to load spatio-temporal tiling parameters.

**Key data structures in ONNXToAIE.cpp**:
- `SystemInfo` – hardware config loaded from `xdna2_info.json` (SPM levels, core count)
- `TileParam` – tiling config: `SPm`, `SPn` (spatial), `TPm`, `TPk`, `TPn` (temporal), `TM`/`TK`/`TN` (tile sizes), `tpOrder` (loop order)
- `AieTile`, `AieBuf`, `AieDma`, `AiePacket` – hardware abstraction structs

**AIE MLIR generation order**:
`DeviceOp` → `TileOp` → `BufferOp`/`LockOp` → `FlowOp` (packet routing) → `DmaOp` (MemTileDMAOp/MemOp/DMABdOp) → `CoreOp`/`CallOp` (with `SCF ForOp`) → `RuntimeSequenceOp`/`NpuDmaMemcpyNdOp` (AIEX)

**Spatio-temporal parallelization**:
- **Spatial**: `SPm × SPn` compute tiles handle M/N axis splits (`M0 = M/SPm`, `N0 = N/SPn`)
- **Temporal**: `TPm × TPk × TPn` sequential iterations per tile, innermost tile = `TM × TK × TN`
- **Memory constraint**: `elem_bytes × (TM×TK + TK×TN + TM×TN) ≤ 60KB` (CTILE_MEM_LIMIT)
- **tpOrder**: `[axis0, axis1, axis2]` — axis indices `0=M, 1=N, 2=K` — controls loop nesting order selected by data-reuse analysis in `gen_tc_list.py`

**Packet routing**: Static packet IDs are pre-allocated per tile to avoid packet filtering bugs. Output quadrant IDs map: `1,2,4,8 → 0,1,2,3`.

---

## Test Data & References

- `data/op_list.json` – input operations (M, K, N, elemType=f32)
- `data/refs/mlir/` – reference AIE MLIR for 1–4 core configurations
- `data/refs/tc_list_ver1/` – reference tc_list.json per matrix size
- `data/refs/result_ver1/` – reference result.csv (correctness + timing baseline)

Result CSV columns: `case_index, numSpm, SPm, SPn, TPm, TPk, TPn, TM, TK, TN, M, K, N, doubleBuffer, status, errors, iters, warmup, avg_us, min_us, max_us`

---

## Key Implementation Notes

- The `tc.json` path in `ONNXToAIE.cpp:307` is hardcoded — must be set correctly before `aie-opt` runs.
- `aie-opt` is the modified version from this repo's `tools/aie-opt/`, not the installed system one. Ensure `$PATH` from `env_setup.sh` points to the local install.
- `xdna2_info.json` target is XDNA2 (Ryzen AI Strix/Strix Halo/Krackan); `NPU2=1` is set in environment for these devices.
- Double-buffering is defined in `TileParam` but currently disabled (`doubleBuffer=false`) in test case generation.
- Kernel function signature: `void extern_kernel(float* A, float* B, float* C, uint32_t N_ROW, uint32_t N_COL, uint32_t N_DEP, bool acc)` — B is transposed.
