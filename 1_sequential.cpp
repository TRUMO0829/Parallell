// g++ -O2 -std=c++17 -pthread 1_sequential.cpp -o 1_sequential
// ./1_sequential
//
// Sequential LSD radix sort: 4 passes over 8-bit digits of uint32 keys.
// Single-threaded baseline used as the reference for speedup of the
// pthread / OpenMP / CUDA versions.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// One LSD pass: count -> prefix-sum -> scatter, on the byte at `shift`.
static void counting_sort_pass(const std::vector<uint32_t> &in,
                               std::vector<uint32_t> &out,
                               int shift)
{
  constexpr int B = 256;            // one bucket per byte value (0..255)
  std::size_t n = in.size();

  // Phase 1 — Histogram: count elements per bucket.
  uint32_t count[B] = {};
  for (std::size_t i = 0; i < n; ++i)
    ++count[(in[i] >> shift) & 0xFF];

  // Phase 2 — Exclusive prefix-sum: starting position of each bucket in output.
  uint32_t prefix[B];
  prefix[0] = 0;
  for (int k = 1; k < B; ++k)
    prefix[k] = prefix[k - 1] + count[k - 1];

  // Phase 3 — Scatter: place each element at its bucket position (stable).
  for (std::size_t i = 0; i < n; ++i) {
    int bucket = (in[i] >> shift) & 0xFF;
    out[prefix[bucket]++] = in[i];
  }
}

// Full sort: 4 passes over each byte (LSD-first); buffers alternate.
// Even number of passes -> result lands back in `data`.
void radix_sort(std::vector<uint32_t> &data) {
  std::size_t n = data.size();
  if (n < 2)
    return;

  std::vector<uint32_t> tmp(n);

  counting_sort_pass(data, tmp, 0);
  counting_sort_pass(tmp, data, 8);
  counting_sort_pass(data, tmp, 16);
  counting_sort_pass(tmp, data, 24);
}

// Test data: deterministic uniform random uint32 values from a given seed.
std::vector<uint32_t> random_data(std::size_t n, uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
  std::vector<uint32_t> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

// Correctness check: verify the array is non-decreasing.
bool is_sorted_check(const std::vector<uint32_t> &v) {
  for (std::size_t i = 1; i < v.size(); ++i)
    if (v[i] < v[i - 1])
      return false;
  return true;
}

// Op-count model shared across all 4 implementations:
//   PASSES * (4N + B)  =  count(1) + scatter(3) per element + prefix-sum(B) per pass
// Used to derive a unified MOPS metric for fair comparison.
static constexpr int RADIX_PASSES = 4;
static constexpr int RADIX_B      = 256;

static long long radix_total_ops(std::size_t n) {
  return static_cast<long long>(RADIX_PASSES) * (4LL * static_cast<long long>(n) + RADIX_B);
}

// Per-run benchmark output (unified CSV schema across all impls).
struct Result {
  std::size_t n;
  int         threads;
  double      execution_ms;
  double      computation_ms;
  double      transfer_ms;
  long long   transfer_bytes;
  long long   total_ops;
  double      performance_mops;
  bool        correct;
};

// Run `runs` timed trials with fresh data per run, return median time.
// One untimed warmup primes caches and the allocator.
Result benchmark(std::size_t n, int runs = 20) {
  std::vector<double> times;
  times.reserve(runs);
  bool ok = true;

  // Warmup (untimed): hot caches, allocator pre-warmed.
  {
    auto data = random_data(n, 0);
    radix_sort(data);
  }

  for (int r = 0; r < runs; ++r) {
    auto data = random_data(n, r * 1337 + 7);   // fresh data per run

    auto t0 = std::chrono::high_resolution_clock::now();
    radix_sort(data);
    auto t1 = std::chrono::high_resolution_clock::now();

    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    if (!is_sorted_check(data))
      ok = false;
  }

  std::sort(times.begin(), times.end());
  double median_ms = times[runs / 2];

  Result res;
  res.n                = n;
  res.threads          = 1;
  res.execution_ms     = median_ms;
  res.computation_ms   = median_ms;       // CPU: no host/device split
  res.transfer_ms      = 0.0;
  res.transfer_bytes   = 0;
  res.total_ops        = radix_total_ops(n);
  res.performance_mops = static_cast<double>(res.total_ops) / (median_ms * 1000.0);
  res.correct          = ok;
  return res;
}

int main() {
  // Sweep over four problem sizes; print table and write CSV.
  const std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000, 10'000'000};
  const int RUNS = 20;

  std::vector<Result> results;
  results.reserve(sizes.size());

  std::cout << "\n  Sequential LSD Radix Sort (Base-256, uint32_t)\n";
  std::cout << "  " << std::string(72, '-') << "\n";
  std::cout << std::setw(12) << "Elements"
            << std::setw(14) << "Exec (ms)"
            << std::setw(14) << "Compute (ms)"
            << std::setw(14) << "Xfer (ms)"
            << std::setw(14) << "Perf (MOPS)"
            << std::setw(10) << "Correct"
            << "\n";
  std::cout << "  " << std::string(72, '-') << "\n";

  for (auto n : sizes) {
    auto r = benchmark(n, RUNS);
    results.push_back(r);
    std::cout << std::setw(12) << r.n
              << std::setw(14) << std::fixed << std::setprecision(3) << r.execution_ms
              << std::setw(14) << std::fixed << std::setprecision(3) << r.computation_ms
              << std::setw(14) << std::fixed << std::setprecision(3) << r.transfer_ms
              << std::setw(14) << std::fixed << std::setprecision(2) << r.performance_mops
              << std::setw(10) << (r.correct ? "YES" : "NO!") << "\n";
  }
  std::cout << "  " << std::string(72, '-') << "\n\n";
  std::cout << "  (each result is median of " << RUNS << " runs)\n\n";

  // Write CSV using the unified schema shared with all 4 implementations.
  const std::string csv_path = "radix_sort_stats.csv";
  std::ofstream csv(csv_path);
  if (!csv) {
    std::cerr << "Could not open " << csv_path << " for writing.\n";
    return 1;
  }
  csv << "implementation,N,threads,execution_ms,computation_ms,transfer_ms,"
         "transfer_bytes,total_ops,performance_mops,sorted_ok\n";
  for (auto &r : results)
    csv << "sequential," << r.n << "," << r.threads << ","
        << std::fixed << std::setprecision(4) << r.execution_ms << ","
        << std::fixed << std::setprecision(4) << r.computation_ms << ","
        << std::fixed << std::setprecision(4) << r.transfer_ms << ","
        << r.transfer_bytes << ","
        << r.total_ops << ","
        << std::fixed << std::setprecision(4) << r.performance_mops << ","
        << (r.correct ? "true" : "false") << "\n";
  csv.close();
  std::cout << "  Stats written to radix_sort_stats.csv\n\n";

  return 0;
}
