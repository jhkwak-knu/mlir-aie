//===- rapl.h - RAPL energy helpers -----------------------------*- C++ -*-===//
//
// Thin wrappers around /sys/class/powercap/intel-rapl:0/energy_uj so the
// measurement layer can read package (and optionally core) energy, compute
// wrap-safe deltas, and sample an idle-power baseline without copying the
// host.cpp batch-mode code verbatim.
//
// Constants and defaults mirror host.cpp (Step 6-4 explicitly reuses that
// protocol): MAX_ENERGY_RANGE_UJ = 65 532 610 987 uJ, idle window defaults
// to 0.2 s x 10 samples for the baseline, 0.3 s x 5 samples for per-batch
// bracketing.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mlp_runner {

constexpr int64_t RAPL_MAX_ENERGY_RANGE_UJ = 65532610987LL;

constexpr double IDLE_SAMPLE_WINDOW_S_DEFAULT = 0.2;
constexpr int    N_IDLE_SAMPLES_DEFAULT       = 10;
constexpr double BRACKET_IDLE_WINDOW_S_DEFAULT = 0.3;
constexpr int    N_BRACKET_IDLE_SAMPLES_DEFAULT = 5;

/// Canonical sysfs paths. Callers pass these (or custom paths) to RaplReader
/// so tests can inject synthetic counters.
extern const char* kRaplPackagePath;
extern const char* kRaplCorePath;

/// Wrap-safe RAPL delta: if the counter rolled over, add MAX_ENERGY_RANGE_UJ.
int64_t raplDelta(int64_t after, int64_t before);

/// Reads a single RAPL counter file. Returns 0 on any I/O failure so callers
/// can detect unavailability without exceptions. Public so tests can verify
/// behavior against fake sysfs files.
int64_t readRaplEnergyUj(const char* path);

/// Sample idle power over `n_samples` windows of `window_s` seconds each,
/// return the median power in mW. Median suppresses OS scheduling spikes.
double measureIdlePowerMw(const char* rapl_path,
                          int n_samples = N_IDLE_SAMPLES_DEFAULT,
                          double window_s = IDLE_SAMPLE_WINDOW_S_DEFAULT);

/// Convenience sample with smaller defaults suitable for per-batch brackets.
double measureBracketIdlePowerMw(
    const char* rapl_path,
    int n_samples = N_BRACKET_IDLE_SAMPLES_DEFAULT,
    double window_s = BRACKET_IDLE_WINDOW_S_DEFAULT);

/// Mean (and sample-CV in %) over a set of values. Returns (mean, cv_pct).
/// cv_pct is 0 when n < 2 or mean == 0.
struct MeanCv {
  double mean;
  double cv_pct;
};
MeanCv computeMeanCv(const std::vector<double>& xs);

}  // namespace mlp_runner
