// =============================================================================
// 3_openmp.cpp
//
// OpenMP parallel version of the LSD radix sort from radix_sort_sequential.cpp.
//
// Strategy: SIMPLE.
//   - Phase (1) histogram     — PARALLEL via OpenMP array reduction
//   - Phase (2) prefix sum    — serial (only R = 10 entries; not worth it)
//   - Phase (3) distribute    — serial (parallelizing it requires per-thread
//                               offsets; deferred to keep the code readable)
//
// Strong-scaling study: thread count is varied across {1, 2, 4, 8, 16} via
// omp_set_num_threads, and the same N = {10k, 100k, 1M} sweep is run for
// each thread count. Output CSV has one row per (num_threads, N) pair
// (15 rows total). Same RNG seed 12345 and per-N driver structure as the
// sequential and stdthread versions so timings are directly comparable.
//
// References:
//   - CLRS, 3rd ed., §8.3 — radix sort outer loop & stability requirement
//   - Sedgewick & Wayne, Algorithms 4e, §5.1 LSD.java — three-phase shape
//   - OpenMP API Specification 5.2 (https://www.openmp.org/spec-html/5.2/openmp.html)
//       * `parallel for` work-sharing construct
//       * `reduction(+: array[:length])` array reduction (OpenMP 4.5+)
//   - OpenMP API Examples 5.2.1 — array-reduction patterns
//   - Haichuan Wang, "A faster OpenMP Radix Sort implementation" (UIUC CS484)
//     https://haichuanwang.wordpress.com/2014/05/26/...
//     — recommends parallelizing all three phases for best speedup; we only
//        do phase (1) here as the chosen "simple" baseline.
//
// Build: clang++ -std=c++17 -O2 -pthread -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -o radix_sort_omp 3_openmp.cpp
// Run:     ./radix_sort_omp
// =============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <omp.h>   // OpenMP runtime API (omp_set_num_threads, etc.)

// Stable counting sort on the decimal digit at place value `exp` (1, 10, 100, …).
// Identical to the sequential version EXCEPT phase (1) is parallelized.
static void counting_sort_by_digit(std::vector<uint32_t>& a,
                                   std::vector<uint32_t>& aux,
                                   uint64_t exp) {
    constexpr int R = 10;          // base 10
    int count[R + 1] = {0};        // same offset trick as Sedgewick: count[d+1]
    const std::size_t n = a.size();

    // ---- (1) histogram  — PARALLEL ---------------------------------------
    // OpenMP gives each thread its own private copy of count[0..R], runs the
    // loop in parallel, and after the parallel region sums all the private
    // copies into the shared `count`. No race conditions, no atomics.
    //
    // The array-reduction syntax `count[:R+1]` (OpenMP 4.5+) is documented in
    // the OpenMP API Specification §2.21.5 and is the idiomatic way to do a
    // parallel histogram.
    #pragma omp parallel for reduction(+:count[:R+1])
    for (std::size_t i = 0; i < n; ++i) {
        int d = static_cast<int>((a[i] / exp) % R);
        count[d + 1]++;
    }

    // ---- (2) prefix sum — SERIAL -----------------------------------------
    // Only R = 10 elements; parallelizing this would cost more in overhead
    // than it saves. The CLRS Ch. 8.2 "cumulative counts" step.
    for (int r = 0; r < R; ++r) {
        count[r + 1] += count[r];
    }

    // ---- (3) distribute — SERIAL -----------------------------------------
    // Parallelizing this safely requires per-thread, per-bucket write offsets
    // (Wang, 2014). Kept serial here per the chosen "simple" strategy.
    for (std::size_t i = 0; i < n; ++i) {
        int d = static_cast<int>((a[i] / exp) % R);
        aux[count[d]++] = a[i];
    }

    a.swap(aux);   // Sedgewick's pointer-swap optimization
}

// LSD radix sort, base 10, for unsigned 32-bit integers.
// Outer loop is identical to the sequential version (CLRS RADIX-SORT).
void radix_sort_lsd(std::vector<uint32_t>& a) {
    if (a.size() < 2) return;

    const uint32_t max_val = *std::max_element(a.begin(), a.end());

    std::vector<uint32_t> aux(a.size());

    for (uint64_t exp = 1; max_val / exp > 0; exp *= 10) {
        counting_sort_by_digit(a, aux, exp);
    }
}

// =============================================================================
// Benchmark driver — strong-scaling sweep.
//   Outer loop: thread count T in {1, 2, 4, 8, 16}  (via omp_set_num_threads)
//   Inner loop: N in {10k, 100k, 1M}
//   => 5 × 3 = 15 CSV rows.
//
// CSV columns:
//   implementation,N,num_threads,execution_ms,computation_ms,transfer_ms,
//   transfer_bytes,total_ops,performance_mops,sorted_ok
//
// Metric definitions (OpenMP, base-10 LSD on uint32):
//   execution_ms     = wall-clock around radix_sort_lsd
//   computation_ms   = same as execution_ms
//   transfer_ms      = 0
//   transfer_bytes   = 0
//   total_ops        = D * (5N + R)
//   performance_mops = total_ops / (execution_ms * 1000)
// =============================================================================
int main() {
    using clock = std::chrono::steady_clock;

    constexpr int R = 10;

    const char* CSV_PATH = "results_openmp.csv";
    const std::size_t sizes[]         = { 10'000, 100'000, 1'000'000 };
    const int         thread_counts[] = { 1, 2, 4, 8, 16 };

    std::ofstream csv(CSV_PATH, std::ios::trunc);
    csv << "implementation,N,num_threads,execution_ms,computation_ms,transfer_ms,"
           "transfer_bytes,total_ops,performance_mops,sorted_ok\n";

    std::cout << "implementation = openmp (base-10 LSD, uint32, strong-scaling sweep)\n";
    std::cout << "------------------------------------------------------------\n";

    for (int num_threads : thread_counts) {
        omp_set_num_threads(num_threads);
        std::cout << "\n=== num_threads = " << num_threads << " ===\n";

        for (std::size_t N : sizes) {
            // ---- generate random test data (same seed as the other versions) ---
            std::vector<uint32_t> data(N);
            std::mt19937 rng(12345);
            std::uniform_int_distribution<uint32_t> dist(
                0, std::numeric_limits<uint32_t>::max());
            for (std::size_t i = 0; i < N; ++i) data[i] = dist(rng);

            // independent reference copy: catches both order violations AND
            // element corruption from a buggy parallel scatter.
            std::vector<uint32_t> ref = data;

            // ---- time the parallel radix sort ---------------------------------
            auto t1 = clock::now();
            radix_sort_lsd(data);
            auto t2 = clock::now();
            double execution_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double computation_ms = execution_ms;
            double transfer_ms    = 0.0;
            std::uint64_t transfer_bytes = 0;

            std::sort(ref.begin(), ref.end());        // reference oracle

            // ---- digit count D for the actual max value -----------------------
            const uint32_t max_val = ref.back();
            int D = 0;
            for (uint64_t exp = 1; max_val / exp > 0; exp *= R) ++D;
            if (D == 0) D = 1;

            const std::uint64_t total_ops =
                static_cast<std::uint64_t>(D) *
                (5ull * static_cast<std::uint64_t>(N) + static_cast<std::uint64_t>(R));
            const double performance_mops =
                execution_ms > 0.0
                    ? static_cast<double>(total_ops) / (execution_ms * 1000.0)
                    : 0.0;

            const bool sorted_ok = std::is_sorted(data.begin(), data.end()) && (data == ref);

            std::cout << "N=" << N
                      << "  Threads=" << num_threads
                      << "  Execution=" << execution_ms << " ms"
                      << "  Computation=" << computation_ms << " ms"
                      << "  Transfer=" << transfer_ms << " ms"
                      << "  Operations=" << total_ops
                      << "  Performance=" << performance_mops << " MOPS"
                      << "  Sorted=" << (sorted_ok ? "true" : "false") << "\n";

            csv << "openmp," << N << "," << num_threads << ","
                << execution_ms << "," << computation_ms << "," << transfer_ms << ","
                << transfer_bytes << "," << total_ops << "," << performance_mops << ","
                << (sorted_ok ? "true" : "false") << "\n";

            if (!sorted_ok) {
                std::cerr << "ERROR: output mismatched reference for N=" << N
                          << ", threads=" << num_threads << "\n";
                return 1;
            }
        }
    }

    csv.close();
    std::cout << "\nWrote " << CSV_PATH << "\n";
    return 0;
}