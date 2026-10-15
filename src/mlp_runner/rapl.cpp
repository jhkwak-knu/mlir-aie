//===- rapl.cpp - Implementations -------------------------------*- C++ -*-===//

#include "rapl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

namespace mlp_runner {

const char* kRaplPackagePath = "/sys/class/powercap/intel-rapl:0/energy_uj";
const char* kRaplCorePath    = "/sys/class/powercap/intel-rapl:0:0/energy_uj";

int64_t raplDelta(int64_t after, int64_t before) {
  int64_t d = after - before;
  if (d < 0) d += RAPL_MAX_ENERGY_RANGE_UJ;
  return d;
}

int64_t readRaplEnergyUj(const char* path) {
  FILE* fp = std::fopen(path, "r");
  if (!fp) return 0;
  int64_t val = 0;
  if (std::fscanf(fp, "%ld", &val) != 1) val = 0;
  std::fclose(fp);
  return val;
}

double measureIdlePowerMw(const char* rapl_path, int n_samples, double window_s) {
  std::vector<double> samples;
  samples.reserve(n_samples);
  for (int s = 0; s < n_samples; ++s) {
    int64_t e0 = readRaplEnergyUj(rapl_path);
    auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(window_s));
    auto t1 = std::chrono::steady_clock::now();
    int64_t e1 = readRaplEnergyUj(rapl_path);
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double pwr_mw = (elapsed > 0.0)
        ? static_cast<double>(raplDelta(e1, e0)) / elapsed / 1000.0
        : 0.0;
    samples.push_back(pwr_mw);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];  // median
}

double measureBracketIdlePowerMw(const char* rapl_path, int n_samples,
                                 double window_s) {
  // Bracket variant uses mean (not median) per host.cpp line 584-590 behavior.
  double sum_mw = 0.0;
  for (int s = 0; s < n_samples; ++s) {
    int64_t e0 = readRaplEnergyUj(rapl_path);
    auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(window_s));
    auto t1 = std::chrono::steady_clock::now();
    int64_t e1 = readRaplEnergyUj(rapl_path);
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > 0.0)
      sum_mw += static_cast<double>(raplDelta(e1, e0)) / elapsed / 1000.0;
  }
  return n_samples > 0 ? sum_mw / n_samples : 0.0;
}

MeanCv computeMeanCv(const std::vector<double>& xs) {
  if (xs.empty()) return {0.0, 0.0};
  double sum = 0.0;
  for (double v : xs) sum += v;
  double mean = sum / static_cast<double>(xs.size());
  if (xs.size() < 2 || mean == 0.0) return {mean, 0.0};
  double sq = 0.0;
  for (double v : xs) {
    double d = v - mean;
    sq += d * d;
  }
  double stddev = std::sqrt(sq / static_cast<double>(xs.size() - 1));
  return {mean, 100.0 * stddev / std::fabs(mean)};
}

}  // namespace mlp_runner
