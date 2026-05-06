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

// Нэг LSD алхам: тоолох -> урьдчилсан нийлбэр -> тараах (`shift` орон дээр).
static void counting_sort_pass(const std::vector<uint32_t> &in,
                               std::vector<uint32_t> &out,
                               int shift)
{
  constexpr int B = 256;            // байт тус бүрд нэг хувин (0..255)
  std::size_t n = in.size();

  // Үе 1 — Гистограм: хувин бүрт оногдох элементийн тоог тоолно.
  uint32_t count[B] = {};
  for (std::size_t i = 0; i < n; ++i)
    ++count[(in[i] >> shift) & 0xFF];

  // Үе 2 — Exclusive prefix-sum: хувин бүрийн эхлэх индексийг тооцно.
  uint32_t prefix[B];
  prefix[0] = 0;
  for (int k = 1; k < B; ++k)
    prefix[k] = prefix[k - 1] + count[k - 1];

  // Үе 3 — Тараах: элемент бүрийг өөрийн хувины байрлалд хадгална (тогтвортой).
  for (std::size_t i = 0; i < n; ++i) {
    int bucket = (in[i] >> shift) & 0xFF;
    out[prefix[bucket]++] = in[i];
  }
}

// Бүрэн цэгцлэлт: 4 байт тус бүрд алхам хийнэ; буферууд ээлжилнэ.
// Тэгш тооны алхамын дараа (4) үр дүн `data`-д буцаж ирнэ.
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

// Туршилтын өгөгдөл: seed-ээс хамаарсан жигд тархалттай uint32 утгууд.
std::vector<uint32_t> random_data(std::size_t n, uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
  std::vector<uint32_t> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

// Зөв эсэхийг шалгах: массив өсөх дарааллаар байгаа эсэх.
bool is_sorted_check(const std::vector<uint32_t> &v) {
  for (std::size_t i = 1; i < v.size(); ++i)
    if (v[i] < v[i - 1])
      return false;
  return true;
}

// 4 хэрэгжүүлэлтэд адил үйлдлийн загвар:
//   PASSES * (4N + B) = тоолох(1) + тараах(3) элемент тутамд + prefix-sum(B)
// MOPS-ийн утгыг шударга харьцуулахад ашиглана.
static constexpr int RADIX_PASSES = 4;
static constexpr int RADIX_B      = 256;

static long long radix_total_ops(std::size_t n) {
  return static_cast<long long>(RADIX_PASSES) * (4LL * static_cast<long long>(n) + RADIX_B);
}

// Туршилтын үр дүнгийн бүтэц (бүх хэрэгжүүлэлтэд адил CSV форматтай).
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

// `runs` удаа хэмжсэн дунд (median) хугацааг буцаана.
// Эхэнд хугацаа хэмжихгүй 1 удаа warmup хийж кэшийг халаана.
Result benchmark(std::size_t n, int runs = 20) {
  std::vector<double> times;
  times.reserve(runs);
  bool ok = true;

  // Урьдчилсан халаалт: кэш ба allocator-ийг бэлтгэнэ.
  {
    auto data = random_data(n, 0);
    radix_sort(data);
  }

  for (int r = 0; r < runs; ++r) {
    auto data = random_data(n, r * 1337 + 7);   // run бүрд шинэ өгөгдөл

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
  res.computation_ms   = median_ms;       // CPU: host/device хуваагдалгүй
  res.transfer_ms      = 0.0;
  res.transfer_bytes   = 0;
  res.total_ops        = radix_total_ops(n);
  res.performance_mops = static_cast<double>(res.total_ops) / (median_ms * 1000.0);
  res.correct          = ok;
  return res;
}

int main() {
  // 4 өөр хэмжээ дээр давтан туршиж, хүснэгт хэвлэн CSV-д бичнэ.
  const std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000, 10'000'000};
  const int RUNS = 20;

  std::vector<Result> results;
  results.reserve(sizes.size());

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
  // std::cout << "  (each result is median of " << RUNS << " runs)\n\n";

  // Бүх 4 хэрэгжүүлэлтэд адил CSV форматаар үр дүнг бичнэ.
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

  return 0;
}
