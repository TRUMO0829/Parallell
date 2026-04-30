// clang++ -std=c++17 -O2 -pthread -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -o 3_openmp 3_openmp.cpp
// ./3_openmp
//
// Parallel LSD radix sort using OpenMP. Same algorithm as the pthread version
// but expressed with OpenMP directives. All three phases of each pass run
// inside ONE parallel region (cheaper than three) and rely on OpenMP's
// implicit barriers between #pragma omp for / single blocks.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr int B      = 256;   // buckets (one per byte value)
static constexpr int PASSES = 4;     // 4 x 8 bits = 32-bit keys

// One LSD pass over byte `shift`, parallelized with OpenMP.
// Caller pre-zeros local_hist; offsets is filled in Phase 2.
static void omp_counting_pass(
    const uint32_t* __restrict__ src,
          uint32_t* __restrict__ dst,
    std::size_t n,
    int         shift,
    int         T,                   // thread count for this region
    uint32_t*   local_hist,          // T x B
    uint32_t*   offsets)             // T x B
{
    // Open ONE parallel region; all 3 phases share its team of T threads.
    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        uint32_t* hist = local_hist + tid * B;   // this thread's histogram row

        // Phase 1 — Parallel histogram (static schedule).
        // schedule(static) gives every thread a fixed contiguous slice; the
        // SAME slice is reused in Phase 3 so the scatter has no overlap.
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i)
            ++hist[(src[i] >> shift) & 0xFF];
        // implicit barrier here

        // Phase 2 — Sequential prefix-sum on a single thread.
        // Parallelizing 256 entries is slower than running them on one core;
        // other threads wait at the implicit barrier after the single block.
        #pragma omp single
        {
            // Sum local histograms across threads -> bucket totals.
            uint32_t total[B] = {};
            for (int t = 0; t < T; ++t)
                for (int b = 0; b < B; ++b)
                    total[b] += local_hist[t * B + b];

            // Exclusive scan -> global start of each bucket.
            uint32_t gstart[B];
            gstart[0] = 0;
            for (int b = 1; b < B; ++b)
                gstart[b] = gstart[b - 1] + total[b - 1];

            // offsets[t][b] = position where thread t writes bucket b first.
            for (int b = 0; b < B; ++b) {
                uint32_t pos = gstart[b];
                for (int t = 0; t < T; ++t) {
                    offsets[t * B + b] = pos;
                    pos += local_hist[t * B + b];
                }
            }
        }
        // implicit barrier here -- all threads now see offsets[]

        // Phase 3 — Parallel scatter, SAME static partition as Phase 1.
        // Each thread writes only to slots it counted -> no race, sort stable.
        uint32_t* off = offsets + tid * B;
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const int bkt = (src[i] >> shift) & 0xFF;
            dst[off[bkt]++] = src[i];
        }
        // implicit barrier here
    }
}

// Driver: run 4 passes, alternating buffers. Result lands in `data` (4 swaps).
void radix_sort_omp(std::vector<uint32_t>& data, int T)
{
    const std::size_t n = data.size();
    if (n < 2) return;

    std::vector<uint32_t> buf(n);
    std::vector<uint32_t> local_hist(T * B);
    std::vector<uint32_t> offsets(T * B);

    uint32_t* src = data.data();
    uint32_t* dst = buf.data();

    for (int pass = 0; pass < PASSES; ++pass) {
        std::fill(local_hist.begin(), local_hist.end(), 0u);   // zero before each pass

        omp_counting_pass(src, dst, n, pass * 8, T,
                          local_hist.data(), offsets.data());

        std::swap(src, dst);
    }
}

// Test data: deterministic uniform random uint32 values from a given seed.
static std::vector<uint32_t> random_data(std::size_t n, uint32_t seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Correctness check: verify the array is non-decreasing.
static bool is_sorted_check(const std::vector<uint32_t>& v)
{
    for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i] < v[i - 1]) return false;
    return true;
}

// Same op-count model as the sequential baseline (see 1_sequential.cpp).
static long long radix_total_ops(std::size_t n) {
    return static_cast<long long>(PASSES) * (4LL * static_cast<long long>(n) + B);
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

// Run `runs` timed trials at thread count T, return median time + correctness.
static Result benchmark(std::size_t n, int T, int runs = 20)
{
    std::vector<double> times;
    times.reserve(runs);
    bool ok = true;

    // Warmup (untimed): hot caches, OpenMP thread pool spun up.
    {
        auto data = random_data(n, 0u);
        radix_sort_omp(data, T);
    }

    for (int r = 0; r < runs; ++r) {
        auto data = random_data(n, r * 1337u + 7u);   // fresh data per run
        double t0 = omp_get_wtime();
        radix_sort_omp(data, T);
        double t1 = omp_get_wtime();

        times.push_back((t1 - t0) * 1000.0);
        if (!is_sorted_check(data)) ok = false;
    }

    std::sort(times.begin(), times.end());
    double median_ms = times[runs / 2];

    Result res;
    res.n                = n;
    res.threads          = T;
    res.execution_ms     = median_ms;
    res.computation_ms   = median_ms;       // CPU: no host/device split
    res.transfer_ms      = 0.0;
    res.transfer_bytes   = 0;
    res.total_ops        = radix_total_ops(n);
    res.performance_mops = static_cast<double>(res.total_ops) / (median_ms * 1000.0);
    res.correct          = ok;
    return res;
}

int main()
{
    // Sweep over (problem size x thread count).
    const std::vector<std::size_t> sizes   = {10'000, 100'000, 1'000'000, 10'000'000};
    const std::vector<int>         threads = {1, 2, 4, 8, 16};
    const int RUNS = 20;

    std::cout << "\n  Host logical CPUs : " << omp_get_max_threads() << "\n";
    std::cout << "\n  Parallel LSD Radix Sort (Base-256, OpenMP, uint32_t)\n";
    std::cout << "  " << std::string(78, '-') << "\n";
    std::cout << std::setw(12) << "Elements"
              << std::setw(10) << "Threads"
              << std::setw(14) << "Exec (ms)"
              << std::setw(14) << "Compute (ms)"
              << std::setw(14) << "Xfer (ms)"
              << std::setw(14) << "Perf (MOPS)"
              << std::setw(10) << "Correct"
              << "\n";
    std::cout << "  " << std::string(78, '-') << "\n";

    std::vector<Result> all_results;

    for (auto n : sizes) {
        bool first_row = true;
        for (auto T : threads) {
            auto r = benchmark(n, T, RUNS);
            all_results.push_back(r);

            std::cout << std::setw(12) << (first_row ? std::to_string(r.n) : "")
                      << std::setw(10) << r.threads
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.execution_ms
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.computation_ms
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.transfer_ms
                      << std::setw(14) << std::fixed << std::setprecision(2) << r.performance_mops
                      << std::setw(10) << (r.correct ? "YES" : "NO!")
                      << "\n";
            first_row = false;
        }
        std::cout << "  " << std::string(78, '-') << "\n";
    }

    std::cout << "\n  (each result is median of " << RUNS << " runs)\n\n";

    // Write CSV using the unified schema shared with all 4 implementations.
    const std::string csv_path = "radix_sort_omp_stats.csv";
    std::ofstream csv(csv_path);
    if (!csv) { std::cerr << "Cannot open CSV.\n"; return 1; }

    csv << "implementation,N,threads,execution_ms,computation_ms,transfer_ms,"
           "transfer_bytes,total_ops,performance_mops,sorted_ok\n";
    for (auto& r : all_results)
        csv << "openmp," << r.n << "," << r.threads << ","
            << std::fixed << std::setprecision(4) << r.execution_ms << ","
            << std::fixed << std::setprecision(4) << r.computation_ms << ","
            << std::fixed << std::setprecision(4) << r.transfer_ms << ","
            << r.transfer_bytes << ","
            << r.total_ops << ","
            << std::fixed << std::setprecision(4) << r.performance_mops << ","
            << (r.correct ? "true" : "false") << "\n";
    csv.close();

    std::cout << "  Stats written to radix_sort_omp_stats.csv\n\n";
    return 0;
}
