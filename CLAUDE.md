# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**At the start of every session, read `.claude/research-sync.md` for the
Notion-based research synchronization workflow (page IDs, access rules,
task status transitions).**

## Project Overview

MLIR-AIE is an MLIR-based toolchain targeting AMD Ryzen AI NPUs (AI Engine tiles). This fork adds an **ONNX-to-AIE compilation pass** (`--convert-onnx-to-aie`) that lowers ONNX dialect operations to AIE/AIEX dialect with spatio-temporal tiling for matrix multiplication.

Active branch: `energy-cost-model`

### Project Goals

1. **Candidate generation & cost estimation** -- Enumerate valid spatio-temporal parallelization configurations for matrix multiplication and compute their estimated cost.
2. **Energy-efficient configuration selection** -- Choose the optimal parallelization setting based on cost (energy efficiency).
3. **MLIR-based compilation** -- Compile the selected configuration to AIE/AIEX dialect using the `--convert-onnx-to-aie` pass (MLIR-AIE toolchain).
4. **Hardware validation & benchmarking** -- Execute on a real Ryzen AI NPU (XDNA2) to verify correctness and measure performance.

---

## Critical Rules

Code style, testing, security, and git workflow rules are inherited from
the user-level CLAUDE.md. Below are **project-specific additions only**.

### Hardware Test Execution Protocol

1. **Pre-test environment check** -- Before running any NPU test batch:
   - Verify RAPL is accessible: `cat /sys/class/powercap/intel-rapl:0/energy_uj`
   - Verify NPU driver timeout: `cat /sys/module/amdxdna/parameters/timeout`
   - Verify no stale `out/tc.json` or `out/build/` from previous runs
   - Close unnecessary applications (browser, IDE) to reduce measurement noise

2. **First-case validation** -- After the first test case completes:
   - Check status is PASS (not RUN_FAIL or timeout)
   - Verify `min_us` and `step_min_us` are reasonable (not -1 or extreme)
   - Verify energy fields are populated (not all -1)
   - If any issue, stop the batch and investigate before continuing

3. **Progress monitoring** -- During long batch runs:
   - Periodically check `tail -5 <result_csv>` for recent statuses
   - Watch for consecutive RUN_FAIL (may indicate driver/NPU hang)
   - If >3 consecutive failures, stop and restart NPU driver or reboot
   - Compare `log.txt` modification time vs current time to detect hangs
   - If `onnx_matmul` process is in `D` (disk sleep) state for >5 min, it is hung

4. **Hang detection and recovery** -- NPU hangs manifest as `D` (disk sleep) state:
   - Check: `pstree -p <batch_pid>` to find `onnx_matmul` PID
   - Check: `cat /proc/<pid>/status | grep State` -- `D (disk sleep)` = hung
   - NPU PCI device: `0000:c5:00.1` (PCI_ID=1022:17F0, driver=amdxdna)
   - `D` state is uninterruptible -- `kill -9`, PCI reset, driver rmmod are all ineffective
   - **Only reliable recovery is reboot**
   - After reboot, resume with `run_tc_all.sh -s <next_case>` (preserves existing CSV)
   - Prevention: ensure `timeout_in_sec` is set to 60 via `setup_env.sh` before batch start
   - Note: timeout_in_sec=60 does NOT prevent all hangs -- some D states bypass driver timeout

5. **Autonomous batch execution** -- When running batches via Claude Code:
   - **Before start**: verify RAPL readable, NPU timeout set, no stale artifacts
   - **First-case validation**: wait for case #1 to finish, check all fields per item 2
   - **Periodic monitoring**: check progress every 10-15 min via background commands
   - **Report**: alert user immediately on any FAIL, RUN_FAIL, or hang detection

6. **Result versioning** -- Every measurement batch must be traceable:
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

This activates `ironenv` (Python venv), runs `utils/env_setup.sh <install_dir>` (sets `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`), and sets the `amdxdna` driver timeout to 60 seconds. Install dir is `$HOME/ryzen_ai/mlir-aie-dev/mlir-aie/install`.

---

## Build & Test Commands (ONNX-to-AIE Pass)

All commands run from `test/onnx-mlir/`.

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
# 1. Generate ONNX MLIR (M K N)
bash scripts/generate/gen_onnx_matmul_mlir.sh 64 64 64

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

## File Structure

```
include/onnx/
├── Conversion/
│   ├── ONNXToAIE/ONNXToAIE.h   # Pass entry point declaration
│   ├── Passes.h / Passes.td     # Pass registration (TableGen)
├── Dialect/ONNX/IR/             # ONNX dialect (MatMul, minimal OpSet-22)
└── Target/XDNA2/xdna2_info.json # Hardware config (32 SPMs, 64KB each)

lib/Conversion/ONNXToAIE/
├── ONNXToAIE.cpp                # Pass entry + registration
├── TileParam.h / TileParam.cpp  # TileParam, SystemInfo, tc.json loading
├── AiePlacement.h / AiePlacement.cpp  # Tile placement, buffer/DMA allocation
├── AieEmitter.cpp               # MLIR op emission
└── CMakeLists.txt

lib/Dialect/ONNX/IR/
├── ONNXDialect.cpp              # Dialect registration
└── ONNXOps.cpp                  # Op implementations

tools/aie-opt/aie-opt.cpp        # Registers --convert-onnx-to-aie flag

test/onnx-mlir/
├── Makefile / CMakeLists.txt    # Build pipeline
├── src/kernel.cc                # AIE compute kernel (C++ MatMul)
├── src/host.cpp                 # XRT host orchestrator (~647 lines)
├── scripts/                     # Shell/Python automation
└── data/                        # Test data (see Test Data & References)
```

---

## Architecture: ONNXToAIE Pass

**Pass registration**: `onnx::createConvertONNXToAIEPass()` on `func::FuncOp`, registered via `Passes.td` -> `aie-opt`.

**Tiling config input**: The pass reads `tc.json` at compile time to load spatio-temporal tiling parameters (see Key Implementation Notes for path details).

**Key data structures in ONNXToAIE.cpp**:
- `SystemInfo` -- hardware config loaded from `xdna2_info.json` (SPM levels, core count)
- `TileParam` -- tiling config: `SPm`, `SPn` (spatial), `TPm`, `TPk`, `TPn` (temporal), `TM`/`TK`/`TN` (tile sizes), `tpOrder` (loop order)
- `AieTile`, `AieBuf`, `AieDma`, `AiePacket` -- hardware abstraction structs

**AIE MLIR generation order**:
`DeviceOp` -> `TileOp` -> `BufferOp`/`LockOp` -> `FlowOp` (packet routing) -> `DmaOp` (MemTileDMAOp/MemOp/DMABdOp) -> `CoreOp`/`CallOp` (with `SCF ForOp`) -> `RuntimeSequenceOp`/`NpuDmaMemcpyNdOp` (AIEX)

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

Result CSV columns (39 total, since commit 7c984731):
`case_index, numSpm, SPm, SPn, TPm, TPk, TPn, TM, TK, TN, M, K, N, doubleBuffer, t_total_pred, status, errors, iters, warmup, avg_us, min_us, max_us, step_avg_us, step_min_us, step_max_us, trace_dispatch_us, trace_kern_pct, trace_gflops, host_overhead_us, ss_iter_cy, ss_kernel_cy, idle_pkg_mw, active_pkg_mw, npu_power_mw, npu_energy_uj, npu_energy_per_iter_uj, wall_elapsed_s, host_steps, matmul_npu_us`

- `min_us`: NPU dispatch+wait only (cost model calibration)
- `step_min_us`: memcpy+sync+dispatch (matches energy measurement scope, use for EDP)

---

## Key Implementation Notes

- The `tc.json` path is loaded via `--tile-param-json` CLI option (`ONNXToAIE.cpp:47`), parsed in `TileParam.cpp`.
- `aie-opt` is the modified version from this repo's `tools/aie-opt/`, not the installed system one. Ensure `$PATH` from `env_setup.sh` points to the local install.
- `xdna2_info.json` target is XDNA2 (Ryzen AI Strix/Strix Halo/Krackan); `NPU2=1` is set in environment for these devices.
- Double-buffering is defined in `TileParam` but currently disabled (`doubleBuffer=false`) in test case generation.
- Kernel function signature: `void extern_kernel(bfloat16* A, bfloat16* B, bfloat16* C, uint32_t N_ROW, uint32_t N_COL, uint32_t N_DEP, bool acc)` -- B is transposed.

---

## Available Commands

- `/tdd` - Test-driven development workflow
- `/plan` - Create implementation plan
- `/code-review` - Review code quality
- `/build-fix` - Fix build errors

---

## Git Workflow

Git conventions (conventional commits, PR workflow) are inherited from
the user-level CLAUDE.md. Project-specific notes:

- Active branch: `energy-cost-model` (will merge into `dev`)
- Result CSV columns changed at commit 7c984731 (39 columns with step_time)
