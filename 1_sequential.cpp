 // g++ -O2 -std=c++17 -pthread 1_sequential.cpp -o 1_sequential
// ./1_sequential

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

static void counting_sort_pass(const std::vector<uint32_t> &in,
                               std::vector<uint32_t> &out,
                               int shift) 
{
  constexpr int B = 256; 
  std::size_t n = in.size();

  uint32_t count[B] = {};
  for (std::size_t i = 0; i < n; ++i)
    ++count[(in[i] >> shift) & 0xFF];

  uint32_t prefix[B];
  prefix[0] = 0;
  for (int k = 1; k < B; ++k)
    prefix[k] = prefix[k - 1] + count[k - 1];

  for (std::size_t i = 0; i < n; ++i) {
    int bucket = (in[i] >> shift) & 0xFF;
    out[prefix[bucket]++] = in[i];
  }
}

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

std::vector<uint32_t> random_data(std::size_t n, uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
  std::vector<uint32_t> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

bool is_sorted_check(const std::vector<uint32_t> &v) {
  for (std::size_t i = 1; i < v.size(); ++i)
    if (v[i] < v[i - 1])
      return false;
  return true;
}

struct Result {
  std::size_t n;
  double time_ms;
  double throughput_meps;
  bool correct;
};

Result benchmark(std::size_t n, int runs = 20) {
  std::vector<double> times;
  times.reserve(runs);
  bool ok = true;

  for (int r = 0; r < runs; ++r) {
    auto data = random_data(n, /*seed=*/r * 1337 + 7);

    auto t0 = std::chrono::high_resolution_clock::now();
    radix_sort(data);
    auto t1 = std::chrono::high_resolution_clock::now();

    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    if (!is_sorted_check(data))
      ok = false;
  }

  std::sort(times.begin(), times.end());
  double median_ms = times[runs / 2];   // use median, not mean
  double meps = (static_cast<double>(n) / 1e6) / (median_ms / 1000.0);

  return {n, median_ms, meps, ok};
}

int main() {
  const std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000};
  const int RUNS = 20;

  std::vector<Result> results;
  results.reserve(sizes.size());

  std::cout << "\n  Sequential LSD Radix Sort (Base-256, uint32_t)\n";
  std::cout << "  " << std::string(56, '-') << "\n";
  std::cout << std::setw(12) << "Elements" << std::setw(16) << "Avg time (ms)"
            << std::setw(20) << "Throughput (M el/s)" << std::setw(10)
            << "Correct"
            << "\n";
  std::cout << "  " << std::string(56, '-') << "\n";

  for (auto n : sizes) {
    auto r = benchmark(n, RUNS);
    results.push_back(r);
    std::cout << std::setw(12) << r.n << std::setw(16) << std::fixed
              << std::setprecision(3) << r.time_ms << std::setw(20)
              << std::fixed << std::setprecision(2) << r.throughput_meps
              << std::setw(10) << (r.correct ? "YES" : "NO!") << "\n";
  }
  std::cout << "  " << std::string(56, '-') << "\n\n";
  std::cout << "  (each result averaged over " << RUNS << " runs)\n\n";

  // ── Write CSV ──────────────────────────────
  const std::string csv_path = "radix_sort_stats.csv";
  std::ofstream csv(csv_path);
  if (!csv) {
    std::cerr << "Could not open " << csv_path << " for writing.\n";
    return 1;
  }
  csv << "elements,avg_time_ms,throughput_meps,correct\n";
  for (auto &r : results)
    csv << r.n << "," << std::fixed << std::setprecision(4) << r.time_ms << ","
        << std::fixed << std::setprecision(4) << r.throughput_meps << ","
        << (r.correct ? "true" : "false") << "\n";
  csv.close();
  std::cout << "  Stats written to radix_sort_stats.csv\n\n";

  return 0;
}