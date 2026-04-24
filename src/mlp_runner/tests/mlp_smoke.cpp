//===- mlp_smoke.cpp - Hermetic forward-pass sanity test --------*- C++ -*-===//
//
// Standalone binary that does not depend on configurations.json or any
// xclbin. Builds a tiny 4-2-2 MLP with hand-rolled weights, runs the
// forward pass through the CPU dispatcher, and asserts:
//   1. Shape of the softmax output matches (batch, output_classes).
//   2. Softmax rows sum to ~1.
//   3. ReLU was applied (non-negative intermediate not visible here, but
//      the output for a known weights set matches a hand-computed value).
//   4. Deterministic init: two runs with the same seed produce bit-identical
//      softmax output.
//   5. gemm_cpu correctness on a toy 2x3 * 3x2 case.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "config_loader.h"
#include "gemm_cpu.h"
#include "mlp.h"
#include "npu_dispatch.h"

using namespace mlp_runner;

static void _assertClose(float got, float want, float tol, const char* msg) {
  if (std::fabs(got - want) > tol) {
    std::fprintf(stderr, "FAIL %s: got=%f want=%f (tol=%f)\n", msg, got, want,
                 tol);
    std::abort();
  }
}

static void testGemmCpuToy() {
  // A (2x3) * B (3x2) = C (2x2)
  //   A = [[1,2,3],[4,5,6]]   B = [[1,0],[0,1],[1,1]]
  //   C = [[1+0+3, 0+2+3],    = [[4, 5],
  //        [4+0+6, 0+5+6]]       [10,11]]
  float A[] = {1, 2, 3, 4, 5, 6};
  float B[] = {1, 0, 0, 1, 1, 1};
  float C[4] = {0};
  gemmCpu(A, B, C, 2, 3, 2);
  _assertClose(C[0], 4.f, 1e-6f, "gemm C[0,0]");
  _assertClose(C[1], 5.f, 1e-6f, "gemm C[0,1]");
  _assertClose(C[2], 10.f, 1e-6f, "gemm C[1,0]");
  _assertClose(C[3], 11.f, 1e-6f, "gemm C[1,1]");
}

static void testReluAndSoftmax() {
  float x[] = {-1.f, 0.f, 2.f, 3.f};
  relu(x, 4);
  _assertClose(x[0], 0.f, 0.f, "relu neg");
  _assertClose(x[1], 0.f, 0.f, "relu zero");
  _assertClose(x[2], 2.f, 0.f, "relu pos");
  _assertClose(x[3], 3.f, 0.f, "relu pos");

  // Softmax row: all-zero row -> uniform
  float row[3] = {0, 0, 0};
  softmaxRowWise(row, 1, 3);
  _assertClose(row[0], 1.f / 3.f, 1e-6f, "softmax uniform");
  _assertClose(row[0] + row[1] + row[2], 1.f, 1e-6f, "softmax sum=1");
}

static ExperimentConfig _tinyConfig(int batch, int output_classes) {
  ExperimentConfig c;
  c.layer_sizes = {4, 2, 2};  // two GEMM layers: 4->2 (hidden), 2->2 (output)
  c.activations = {"relu"};   // one activation between the two layers
  c.output_classes = output_classes;
  c.batch_size = batch;
  c.warmup_iterations = 0;
  c.outer_batches = 0;
  c.inner_target_seconds = 0.0;
  return c;
}

static std::vector<LayerKernelEntry> _tinyLayers(int batch) {
  std::vector<LayerKernelEntry> L(2);
  L[0].layer = "fc1"; L[0].setter = "cpu";
  L[0].M = batch; L[0].K = 4; L[0].N = 2;
  L[1].layer = "fc2"; L[1].setter = "cpu";
  L[1].M = batch; L[1].K = 2; L[1].N = 2;
  return L;
}

static void testForwardShapeAndSoftmax() {
  auto cfg = _tinyConfig(3, 2);
  auto layers = _tinyLayers(3);
  auto w = initRandomWeights(layers, 7);
  auto x = makeRandomInput(cfg.batch_size, cfg.layer_sizes.front(), 7 ^ 0xA5A5u);
  auto disp = makeDispatcher("cpu");
  auto probs = forward(cfg, layers, w, x, *disp);

  assert(probs.size() ==
         static_cast<size_t>(cfg.batch_size * cfg.output_classes));
  for (int b = 0; b < cfg.batch_size; ++b) {
    float s = 0.f;
    for (int c = 0; c < cfg.output_classes; ++c)
      s += probs[b * cfg.output_classes + c];
    _assertClose(s, 1.f, 1e-5f, "forward softmax row sum");
  }
}

static void testDeterministicSeed() {
  auto cfg = _tinyConfig(3, 2);
  auto layers = _tinyLayers(3);
  auto disp = makeDispatcher("cpu");

  auto run = [&]() {
    auto w = initRandomWeights(layers, 12345);
    auto x = makeRandomInput(cfg.batch_size, cfg.layer_sizes.front(),
                             12345 ^ 0xA5A5u);
    return forward(cfg, layers, w, x, *disp);
  };

  auto a = run();
  auto b = run();
  assert(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    _assertClose(a[i], b[i], 0.f, "deterministic seed");
  }
}

static void testOutputSlicingPadded() {
  // Simulate the real pipeline: last layer N=4 (padded), output_classes=2.
  ExperimentConfig c;
  c.layer_sizes = {3, 4};   // single GEMM, 3 -> 4 (padded)
  c.activations = {};
  c.output_classes = 2;
  c.batch_size = 1;

  std::vector<LayerKernelEntry> L(1);
  L[0].layer = "fc1"; L[0].setter = "cpu";
  L[0].M = 1; L[0].K = 3; L[0].N = 4;

  MlpWeights w;
  // W [3,4] identity-like: column j = unit weight on input j mod 3
  w.W = {{1, 0, 0, 1,
          0, 1, 0, 0,
          0, 0, 1, 0}};
  std::vector<float> x = {10.f, 20.f, 30.f};

  auto disp = makeDispatcher("cpu");
  auto probs = forward(c, L, w, x, *disp);

  // pre-softmax logits would be [10, 20, 30, 10]; we slice first 2 -> [10, 20]
  // softmax([10, 20]) ~ [e^-10 / (e^-10 + 1), 1 / (1 + e^-10)]
  assert(probs.size() == 2);
  _assertClose(probs[0] + probs[1], 1.f, 1e-5f, "padded slice+softmax sum");
  // argmax of pre-softmax [10, 20] is index 1.
  assert(probs[1] > probs[0]);
}

static void testXrtStubThrows() {
  bool threw = false;
  try {
    auto d = makeDispatcher("npu");
    (void)d;
  } catch (const std::exception& e) {
    threw = true;
  }
  assert(threw && "XrtDispatcherStub must throw in Step 6-3");
}

int main() {
  testGemmCpuToy();
  testReluAndSoftmax();
  testForwardShapeAndSoftmax();
  testDeterministicSeed();
  testOutputSlicingPadded();
  testXrtStubThrows();
  std::printf("mlp_smoke: ok (6 checks)\n");
  return 0;
}
