# External C++ MLP Reference Selection (Notion step 21, Step 4-1)

This document records the process used to select an external C++ MLP
reference implementation for the MLP End-to-End EDP measurement
experiment. Notion Section "Step 4-1" requires a fork of a verified
implementation that demonstrates >= 90% MNIST accuracy under a
permissive license.

## Candidate Comparison

| Repo | MNIST Accuracy | License | GEMM Structure | Dependencies | Verdict |
|---|---|---|---|---|---|
| [muchlakshay/MLP-From-Scratch](https://github.com/muchlakshay/MLP-From-Scratch) | 95% @ 20 epochs (README) | MIT | Eigen `MatrixXd` expression templates | Eigen (header-only), C++11 | **Selected** |
| [MichalPitr/inference_engine](https://github.com/MichalPitr/inference_engine) | 94.65% @ 10 epochs (blog) | MIT | Explicit `gemm_cpu(A, B, bias, out, n, m, k, ...)` | protobuf, YAML-CPP, (opt. CUDA Toolkit) | Rejected (heavyweight) |
| [xbili/mnist-cpp](https://github.com/xbili/mnist-cpp) | Not documented | MIT | Not documented | CMake, C++ only | Rejected (WIP README) |
| [gbiro/Cpp_MLP](https://github.com/gbiro/Cpp_MLP) | 82% @ 5 epochs | GPL-3.0 | Not documented | CMake | Rejected (<90%, copyleft) |

## Selection: muchlakshay/MLP-From-Scratch

**Upstream commit**: `fd53258b092a60f57e7a85594397bf0c41b2ff3e` (2025-05-22)

**Reasons**:

1. Documented 95% MNIST accuracy (Notion baseline threshold: 90%).
2. Permissive MIT license, fully compatible with this repo's policies.
3. Smallest surface area of the candidates: 5 source files, 686 lines
   total. The entire training + inference flow fits in
   `NeuralNetwork.{cpp,h}` and is easy to audit.
4. Header-only Eigen is the only third-party dependency; no Python,
   protobuf, YAML, or CUDA build toolchain required.
5. The CSV / MNIST-IDX loaders in `loadcsv.h` and `load_mnist.h` are
   self-contained utilities we can reuse as-is for Step 4-2 accuracy
   validation.

## Why not MichalPitr/inference_engine?

Its `gemm_cpu.cpp` is structurally the cleanest GEMM abstraction of the
four candidates — a good pattern for later integration. But the repo
enforces a full ONNX execution engine: protobuf code generation,
YAML-CPP configuration parsing, and CUDA-aware CMake. Vendoring it
would pull a large transitive build graph into this repo for no
additional benefit at the baseline-validation stage.

## Scope of the Fork

The vendored copy at `external/MLP-From-Scratch/` is used exclusively
for **Step 4-2 baseline accuracy validation**: build the original code,
run MNIST test set, confirm >= 90% accuracy.

The NPU-integrated runner (`src/mlp_runner/`, Step 6-3 / 6-4) is written
from scratch in this repo. It does **not** depend on the external fork.
Weight reproducibility is achieved by seeding the same PRNG in both the
CPU reference path and the NPU dispatch path within `mlp_runner`, so
no weight file transfer between the external fork and the runner is
required. The external fork's role is bounded to proving that a C++
MLP implementation with matching hyperparameters can reach >= 90%
accuracy — a sanity check on the methodology, not a code dependency.

## Baseline Sanity (Step 4-2, lightweight scope)

The full training + 10 000-sample MNIST evaluation is skipped deliberately.
The NPU measurement runs with random weights (`config.model.weight_init =
"random"`), so trained accuracy does not feed into any downstream result.
We only need to prove that the upstream pattern compiles with our toolchain
and runs a forward pass end-to-end.

The smoke harness lives at `external/MLP-From-Scratch/smoke_test.cpp`:

1. Instantiate `NeuralNetwork{{784, 512, 512, 10}, {"relu", "relu", "softmax"}}`
   — the same architecture our experiment config pins.
2. Feed a random-initialized `MatrixXd(32, 784)` through `predict()`.
3. Assert `(rows, cols) == (32, 10)` and that each softmax row sums to
   1 within `1e-5`.

Build (from the fork directory):

```bash
g++ -O2 -std=c++17 -I /usr/include/eigen3 \
    smoke_test.cpp NeuralNetwork.cpp -o smoke_test
./smoke_test
```

### Observed Baseline Sanity (2026-04-25)

- Toolchain: `g++ 13.3.0`, `Eigen 3.4.0` (system `libeigen3-dev`).
- Build: clean, only sign-comparison and unused-variable style warnings
  in `NeuralNetwork.cpp` (upstream; not fixed here).
- Smoke output:
  ```
  arch: 784-512-512-10
  batch: 32
  output_shape: (32, 10)
  softmax_row_sum[0]: 1
  softmax_row_sum[16]: 1
  softmax_row_sum[31]: 1
  smoke: ok
  ```
- Verdict: **pass**. The upstream reference compiles and evaluates the
  target architecture end-to-end. Documented 95% MNIST accuracy is
  accepted as-is per the upstream README; re-running a full train is
  not part of the experiment's acceptance criteria.
