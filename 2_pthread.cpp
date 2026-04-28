// =============================================================================
// 2_pthread.cpp
//
// C++ std::thread parallel version of LSD radix sort, base 10, unsigned 32-bit.
//
// Strategy: FULL parallelization (per the project plan).
//   Phase (1) histogram    — PARALLEL: per-thread local histograms, no sharing
//   Phase (2) offsets      — SERIAL : compute each thread's private write
//                            positions (a 2D prefix sum over [bucket, thread]);
//                            this REPLACES the simple "cumulates" step from the
//                            sequential version
//   Phase (3) distribute   — PARALLEL: each thread writes to disjoint output
//                            regions using its precomputed offsets — no races,
//                            no atomics
//
// Threading model: spawn-and-join per phase per pass. Threads are created
// fresh for each phase (~10 ms work per phase per digit), simplest to read.
//
// Strong-scaling study: thread count is varied across {1, 2, 4, 8, 16} and
// the same N = {10k, 100k, 1M} sweep is run for each thread count. Output
// CSV has one row per (num_threads, N) pair (15 rows total). Same RNG seed
// (12345) and per-N driver structure as the sequential and OpenMP versions
// so the timings are directly comparable.
//
// References:
//   - CLRS, 3rd ed., §8.3 — radix sort outer loop & stability requirement
//   - Sedgewick & Wayne, Algorithms 4e, §5.1 LSD.java — three-phase shape
//   - A. Williams, "C++ Concurrency in Action", 2nd ed. (Manning 2019) —
//       * Ch. 2 (managing std::thread, RAII, join semantics)
//       * Ch. 8 (designing concurrent code: data partitioning patterns)
//   - cppreference.com — <thread>, std::thread::join, std::thread::hardware_concurrency
//   - OpenGenus, "Parallel Radix Sort handling positive & negative numbers in C++"
//     (https://iq.opengenus.org/parallel-radix-sort/) — LSD pattern with
//     std::thread.
//   - H. Wang, "A faster OpenMP Radix Sort implementation" (UIUC CS484) —
//     argument for parallelizing all three phases. The algorithmic structure
//     transfers directly to std::thread; only the threading API differs.
//
// Build:  g++ -O2 -std=c++17 -pthread 2_pthread.cpp -o radix_sort_thr
// Run:    ./radix_sort_thr
// =============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <vector>

// ---- Configuration ---------------------------------------------------------
constexpr int R = 10;                  // base 10 (one decimal digit per pass)
// Thread count is now a runtime parameter (passed through `radix_sort_lsd`)
// so the strong-scaling driver can sweep it across {1, 2, 4, 8, 16}.

// ---- Parallel stable counting sort on the digit at place value `exp` -------
//
// 2D offset trick (Phase 2): instead of a single 1D cumulative-counts array
// (sequential code), we compute offset[t][b] = the index where thread t
// should start writing its keys with digit b. The layout is:
//
//       bucket 0           bucket 1           bucket 2     ...
//   [t0][t1][t2][t3] | [t0][t1][t2][t3] | [t0][t1][t2][t3] | ...
//   ^                  ^                  ^
//   global_start[0]    global_start[1]    global_start[2]
//
// Each thread t's writes for bucket b land in a disjoint sub-region, so
// no synchronization is needed during distribute.
static void counting_sort_by_digit(std::vector<uint32_t>& a,
                                   std::vector<uint32_t>& aux,
                                   uint64_t exp,
                                   int num_threads) {
    const std::size_t n = a.size();

    // chunk boundaries: [chunk_start(t), chunk_end(t)) for each thread
    auto chunk_start = [&](int t) -> std::size_t { return (n *  t     ) / num_threads; };
    auto chunk_end   = [&](int t) -> std::size_t { return (n * (t + 1)) / num_threads; };

    // Per-thread local histograms (no sharing -> no races, no atomics).
    std::vector<std::array<int, R>> local(num_threads);
    for (auto& h : local) h.fill(0);

    // ---- (1) histogram — PARALLEL ----------------------------------------
    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const std::size_t s = chunk_start(t);
                const std::size_t e = chunk_end(t);
                auto& h = local[t];
                for (std::size_t i = s; i < e; ++i) {
                    int d = static_cast<int>((a[i] / exp) % R);
                    h[d]++;
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    // ---- (2) per-thread, per-bucket write offsets — SERIAL ---------------
    // Walk buckets in order; within each bucket, walk threads in order.
    // This is what makes the sort STABLE: within (thread t, bucket b) keys
    // are written left-to-right, and thread t's region for bucket b sits
    // strictly before thread (t+1)'s region for the same bucket.
    std::vector<std::array<int, R>> offset(num_threads);
    {
        int running = 0;
        for (int b = 0; b < R; ++b) {
            for (int t = 0; t < num_threads; ++t) {
                offset[t][b] = running;
                running += local[t][b];
            }
        }
        // assert: running == n  (sanity)
        assert(static_cast<std::size_t>(running) == n);
    }

    // ---- (3) distribute — PARALLEL ---------------------------------------
    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const std::size_t s = chunk_start(t);
                const std::size_t e = chunk_end(t);
                std::array<int, R> off = offset[t];   // private copy, mutate
                for (std::size_t i = s; i < e; ++i) {
                    int d = static_cast<int>((a[i] / exp) % R);
                    aux[off[d]++] = a[i];
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    a.swap(aux);  // Sedgewick's pointer-swap optimization
}

// LSD radix sort, base 10, for unsigned 32-bit integers.
// Outer loop is identical to the sequential and OpenMP versions.
void radix_sort_lsd(std::vector<uint32_t>& a, int num_threads) {
    if (a.size() < 2) return;

    const uint32_t max_val = *std::max_element(a.begin(), a.end());

    std::vector<uint32_t> aux(a.size());

    for (uint64_t exp = 1; max_val / exp > 0; exp *= 10) {
        counting_sort_by_digit(a, aux, exp, num_threads);
    }
}

// =============================================================================
// Benchmark driver — strong-scaling sweep.
//   Outer loop: thread count T in {1, 2, 4, 8, 16}
//   Inner loop: N in {10k, 100k, 1M}
//   => 5 × 3 = 15 CSV rows.
//
// CSV columns:
//   implementation,N,num_threads,execution_ms,computation_ms,transfer_ms,
//   transfer_bytes,total_ops,performance_mops,sorted_ok
//
// Metric definitions (std::thread, base-10 LSD on uint32, T threads):
//   execution_ms     = wall-clock around radix_sort_lsd
//   computation_ms   = same as execution_ms (no host/device split)
//   transfer_ms      = 0
//   transfer_bytes   = 0
//   total_ops        = D * (5N + R)
//   performance_mops = total_ops / (execution_ms * 1000)
// =============================================================================
int main() {
    using clock = std::chrono::steady_clock;

    const char* CSV_PATH = "results_stdthread.csv";
    const std::size_t sizes[]         ={ 10'000, 100'000, 1'000'000 };
    const int         thread_counts[] = { 1, 2, 4, 8, 16 };

    std::ofstream csv(CSV_PATH, std::ios::trunc);
    csv << "implementation,N,num_threads,execution_ms,computation_ms,transfer_ms,"
           "transfer_bytes,total_ops,performance_mops,sorted_ok\n";

    std::cout << "implementation = stdthread (base-10 LSD, uint32, strong-scaling sweep)\n";
    std::cout << "HW concurrency = " << std::thread::hardware_concurrency() << "\n";
    std::cout << "------------------------------------------------------------\n";

    for (int num_threads : thread_counts) {
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
            radix_sort_lsd(data, num_threads);
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

            csv << "stdthread," << N << "," << num_threads << ","
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