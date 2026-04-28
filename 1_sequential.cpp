// =============================================================================
// 1_sequential.cpp
//
// Sequential LSD (Least Significant Digit) radix sort, base 10.
// Sorts std::vector<uint32_t> using stable counting sort as the per-digit
// subroutine.
//
// References:
//   - CLRS, 3rd ed., Section 8.3 ("Radix sort") — algorithmic structure
//     (stable counting sort applied digit-by-digit, LSD to MSD).
//   - Sedgewick & Wayne, Algorithms 4e, §5.1 (LSD.java) — the
//     histogram / prefix-sum / distribute structure used here.
//   - GeeksforGeeks "C++ Program for Radix Sort" — base-10 reference,
//     used only as a sanity check.
//
// Build:   g++ -O2 -std=c++17 1_sequential.cpp -o radix_sort_seq
// Run:     ./radix_sort_seq
//
// NOTE on later parallelization:
//   The per-digit subroutine `counting_sort_by_digit` is split into three
//   explicit phases — (1) histogram, (2) prefix sum, (3) distribute. This
//   is the structure the std::thread and OpenMP variants parallelize. The
//   CUDA variant uses the same three-phase shape per pass but with base 256
//   (4 passes) instead of base 10 (10 passes).
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

static void counting_sort_by_digit(std::vector<uint32_t>& a,
                                   std::vector<uint32_t>& aux,
                                   uint64_t exp) {
    constexpr int R = 10;         
    int count[R + 1] = {0};   
    const std::size_t n = a.size();

    // ---- (1) histogram: count how many keys have each digit value ----------
    for (std::size_t i = 0; i < n; ++i) {
        int d = static_cast<int>((a[i] / exp) % R);
        count[d + 1]++;
    }

    // ---- (2) prefix sum: count[d] becomes the start index for digit d ------
    for (int r = 0; r < R; ++r) {
        count[r + 1] += count[r];
    }

    // ---- (3) distribute: scatter each key to its bucket, advancing the -----
    for (std::size_t i = 0; i < n; ++i) {
        int d = static_cast<int>((a[i] / exp) % R);
        aux[count[d]++] = a[i];
    }

    a.swap(aux);
}

void radix_sort_lsd(std::vector<uint32_t>& a) {
    if (a.size() < 2) return;

    const uint32_t max_val = *std::max_element(a.begin(), a.end());

    std::vector<uint32_t> aux(a.size());

    for (uint64_t exp = 1; max_val / exp > 0; exp *= 10) {
        counting_sort_by_digit(a, aux, exp);
    }
}

// =============================================================================
// Benchmark driver — runs N = 10k, 100k, 1M and writes one CSV row per N.
// CSV columns:
//   implementation,N,execution_ms,computation_ms,transfer_ms,transfer_bytes,
//   total_ops,performance_mops,sorted_ok
//
// Metric definitions (sequential, base-10 LSD on uint32):
//   execution_ms     = wall-clock around radix_sort_lsd (no transfer phase)
//   computation_ms   = same as execution_ms (no host/device split)
//   transfer_ms      = 0   (everything is in main memory)
//   transfer_bytes   = 0
//   total_ops        = D * (5N + R) per the project's operation-counting
//                      convention: 2N (count) + R (prefix sum) + 3N (scatter)
//                      per digit pass, times D digit passes.
//                      R = 10 (radix), D = #base-10 digits in max(data).
//   performance_mops = total_ops / (execution_ms * 1000)
// =============================================================================
int main() {
    using clock = std::chrono::steady_clock;

    constexpr int R = 10;
    const char* CSV_PATH = "results_sequential.csv";
    const std::size_t sizes[] = { 10'000, 100'000, 1'000'000 };

    std::ofstream csv(CSV_PATH, std::ios::trunc);
    csv << "implementation,N,num_threads,execution_ms,computation_ms,transfer_ms,"
           "transfer_bytes,total_ops,performance_mops,sorted_ok\n";

    std::cout << "implementation = sequential (base-10 LSD, uint32)\n";
    std::cout << "------------------------------------------------------------\n";

    for (std::size_t N : sizes) {
        // ---- generate random test data (fixed seed -> reproducible runs) ---
        std::vector<uint32_t> data(N);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<uint32_t> dist(
            0, std::numeric_limits<uint32_t>::max());
        for (std::size_t i = 0; i < N; ++i) data[i] = dist(rng);

        // independent reference copy: catches both order violations AND
        // element corruption (e.g., a buggy scatter that overwrites values
        // and still leaves the result `is_sorted`).
        std::vector<uint32_t> ref = data;

        // ---- time the radix sort ------------------------------------------
        auto t1 = clock::now();
        radix_sort_lsd(data);
        auto t2 = clock::now();
        double execution_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double computation_ms = execution_ms;     // no transfer phase
        double transfer_ms    = 0.0;
        std::uint64_t transfer_bytes = 0;

        std::sort(ref.begin(), ref.end());        // reference oracle

        // ---- digit count D for the actual max value -----------------------
        const uint32_t max_val = ref.back();      // already sorted, so max is last
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
                  << "  Execution=" << execution_ms << " ms"
                  << "  Computation=" << computation_ms << " ms"
                  << "  Transfer=" << transfer_ms << " ms"
                  << "  Operations=" << total_ops
                  << "  Performance=" << performance_mops << " MOPS"
                  << "  Sorted=" << (sorted_ok ? "true" : "false") << "\n";

        csv << "sequential," << N << ",1,"
            << execution_ms << "," << computation_ms << "," << transfer_ms << ","
            << transfer_bytes << "," << total_ops << "," << performance_mops << ","
            << (sorted_ok ? "true" : "false") << "\n";

        if (!sorted_ok) {
            std::cerr << "ERROR: output mismatched reference for N=" << N << "\n";
            return 1;
        }
    }

    csv.close();
    std::cout << "Wrote " << CSV_PATH << "\n";
    return 0;
}