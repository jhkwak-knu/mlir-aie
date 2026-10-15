// Minimal sanity check for the vendored MLP reference (Notion step 21, Step 4-2).
//
// We do not need to re-derive the 95% MNIST accuracy: the experiment consumes
// this code only to confirm that the upstream pattern compiles with our
// toolchain and runs a forward pass end-to-end. Training + 10 000-sample
// evaluation is intentionally skipped because:
//   1. the NPU measurement runs with RANDOM weights (config.model.weight_init
//      = "random"), so trained accuracy is not load-bearing for our data,
//   2. the upstream README already documents 95% after 20 epochs, and
//   3. a full train + eval adds multiple minutes of wall time per run with
//      zero information gain for the measurement pipeline.
//
// Smoke check: instantiate the same 784-512-512-10 MLP architecture our
// experiment config pins, feed a random batch of 32 samples through
// predict(), and assert that the output matrix has the expected shape and
// that each softmax row sums to ~1.
//
// Build (from this directory):
//   g++ -O2 -std=c++17 -I /usr/include/eigen3 \
//       smoke_test.cpp NeuralNetwork.cpp -o smoke_test

#include "NeuralNetwork.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    constexpr int kBatch = 32;
    constexpr int kInput = 784;
    constexpr int kHidden = 512;
    constexpr int kOutput = 10;

    NeuralNetwork nn{
        {kInput, kHidden, kHidden, kOutput},
        {"relu", "relu", "softmax"},
    };

    MatrixXd X = MatrixXd::Random(kBatch, kInput);
    MatrixXd pred = nn.predict(X);

    std::cout << "arch: " << kInput << "-" << kHidden << "-" << kHidden
              << "-" << kOutput << "\n";
    std::cout << "batch: " << kBatch << "\n";
    std::cout << "output_shape: (" << pred.rows() << ", " << pred.cols()
              << ")\n";

    assert(pred.rows() == kBatch && "output rows must equal batch size");
    assert(pred.cols() == kOutput && "output cols must equal num classes");

    // Softmax rows must sum to ~1.0. Pick a few rows as spot checks.
    for (int i : {0, kBatch / 2, kBatch - 1}) {
        double row_sum = pred.row(i).sum();
        std::cout << "softmax_row_sum[" << i << "]: " << row_sum << "\n";
        assert(std::abs(row_sum - 1.0) < 1e-5 &&
               "softmax row must sum to 1 within 1e-5");
    }

    std::cout << "smoke: ok\n";
    return 0;
}
