//===- gemm_cpu.h - Row-major float32 reference GEMM ------------*- C++ -*-===//
//
// Minimal C = A * B implementation used as:
//   * the numerical reference against which the NPU dispatch is compared
//     (Step 6-3 / 6-4 numerical verification),
//   * the CPU fallback when the runner is invoked with --cpu-only.
//
// Shapes (all row-major, float32):
//   A : [M, K], B : [K, N], C : [M, N]
//
// No SIMD / no BLAS: three nested loops. The runner is measurement-driven
// against an NPU, so the CPU path only needs to be correct, not fast.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlp_runner {

/// C[m, n] = sum_k A[m, k] * B[k, n]. No bias, no alpha/beta; matches the
/// shape generate_configs.py emits.
void gemmCpu(const float* A, const float* B, float* C, int M, int K, int N);

/// In-place ReLU over a length-n contiguous buffer.
void relu(float* x, int n);

/// Row-wise softmax for [batch, classes] buffer, numerically stable.
void softmaxRowWise(float* logits, int batch, int classes);

/// Slice the first `keep_cols` columns from a [batch, full_cols] row-major
/// buffer into `dst` of shape [batch, keep_cols]. Useful for the
/// padded-N output layer (e.g. NPU-facing 16 -> MNIST 10).
void sliceColumns(const float* src, float* dst, int batch,
                  int full_cols, int keep_cols);

}  // namespace mlp_runner
