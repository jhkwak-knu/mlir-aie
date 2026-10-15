//===- gemm_cpu.cpp - Reference GEMM + activations --------------*- C++ -*-===//

#include "gemm_cpu.h"

#include <algorithm>
#include <cmath>

namespace mlp_runner {

void gemmCpu(const float* A, const float* B, float* C, int M, int K, int N) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = acc;
    }
  }
}

void relu(float* x, int n) {
  for (int i = 0; i < n; ++i) {
    if (x[i] < 0.0f) x[i] = 0.0f;
  }
}

void softmaxRowWise(float* logits, int batch, int classes) {
  for (int m = 0; m < batch; ++m) {
    float* row = logits + m * classes;
    // Subtract max for numerical stability.
    float max_v = row[0];
    for (int c = 1; c < classes; ++c) {
      if (row[c] > max_v) max_v = row[c];
    }
    float sum = 0.0f;
    for (int c = 0; c < classes; ++c) {
      row[c] = std::exp(row[c] - max_v);
      sum += row[c];
    }
    float inv = 1.0f / sum;
    for (int c = 0; c < classes; ++c) row[c] *= inv;
  }
}

void sliceColumns(const float* src, float* dst, int batch,
                  int full_cols, int keep_cols) {
  for (int m = 0; m < batch; ++m) {
    std::copy(src + m * full_cols, src + m * full_cols + keep_cols,
              dst + m * keep_cols);
  }
}

}  // namespace mlp_runner
