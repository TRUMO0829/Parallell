// =============================================================================
// radix_sort_openmp.cpp
//
// OpenMP parallel version of the LSD radix sort from radix_sort_sequential.cpp.
//
// Strategy: SIMPLE.
//   - Phase (1) histogram     — PARALLEL via OpenMP array reduction
//   - Phase (2) prefix sum    — serial (only R = 10 entries; not worth it)
//   - Phase (3) distribute    — serial (parallelizing it requires per-thread
//                               offsets; deferred to keep the code readable)
//
// Everything else (algorithm, base 10, uint32 type, N, RNG seed, driver,
// std::sort verification) is identical to the sequential version, so the
// timings are directly comparable.
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
// Build:   g++ -O2 -std=c++17 -fopenmp radix_sort_openmp.cpp -o radix_sort_omp
// Run:     ./radix_sort_omp
// =============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
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
// Benchmark driver — same N, same seed, same structure as sequential version
// so the timing numbers are directly comparable.
// =============================================================================
int main() {
    using clock = std::chrono::steady_clock;

    // ---- HARDCODED thread count -------------------------------------------
    constexpr int NUM_THREADS = 4;
    omp_set_num_threads(NUM_THREADS);

    constexpr std::size_t N = 1'000'000;

    // ---- generate random test data (same seed as sequential -> same data) -
    std::vector<uint32_t> data(N);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(
        0, std::numeric_limits<uint32_t>::max());
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
    std::vector<uint32_t> ref = data;

    // ---- time the parallel radix sort -------------------------------------
    auto t1 = clock::now();
    radix_sort_lsd(data);
    auto t2 = clock::now();
    double ms_radix = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ---- time std::sort on the same input, as a reference -----------------
    auto t3 = clock::now();
    std::sort(ref.begin(), ref.end());
    auto t4 = clock::now();
    double ms_std = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // ---- verify correctness -----------------------------------------------
    const bool sorted_ok = std::is_sorted(data.begin(), data.end());
    const bool matches   = (data == ref);
    const bool ok        = sorted_ok && matches;

    std::cout << "N             = " << N           << "\n";
    std::cout << "Threads (OMP) = " << NUM_THREADS << "\n";
    std::cout << "Radix sort    = " << ms_radix    << " ms\n";
    std::cout << "std::sort     = " << ms_std      << " ms\n";
    std::cout << "is_sorted     = " << (sorted_ok ? "true" : "false") << "\n";
    std::cout << "matches std   = " << (matches   ? "true" : "false") << "\n";
    std::cout << "Result        = " << (ok ? "OK" : "MISMATCH") << "\n";

    return ok ? 0 : 1;
}