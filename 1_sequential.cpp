// g++ -O2 -std=c++17 1_sequential.cpp -o 1_sequential
// ./1_sequential
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

static constexpr int NUM_BUCKETS = 1 << 8;  

static void counting_sort_by_digit(std::vector<uint32_t>& a,
                                   std::vector<uint32_t>& aux,
                                   uint32_t shift) {
    int count[NUM_BUCKETS + 1] = {0};
    const std::size_t n = a.size();

    // ---- (1) histogram: count how many keys have each byte value -----------
    for (std::size_t i = 0; i < n; ++i) {
        int d = static_cast<int>((a[i] >> shift) & 0xFFu);
        count[d + 1]++;
    }

    // ---- (2) prefix sum: count[d] becomes the start index for bucket d -----
    for (int r = 0; r < NUM_BUCKETS; ++r) {
        count[r + 1] += count[r];
    }

    // ---- (3) distribute: scatter each key to its bucket, advancing the -----
    for (std::size_t i = 0; i < n; ++i) {
        int d = static_cast<int>((a[i] >> shift) & 0xFFu);
        aux[count[d]++] = a[i];
    }

    a.swap(aux);
}

void radix_sort_lsd(std::vector<uint32_t>& a) {
    if (a.size() < 2) return;

    std::vector<uint32_t> aux(a.size());

    for (uint32_t shift = 0; shift < 32; shift += 8) {
        counting_sort_by_digit(a, aux, shift);
    }
}

int main() {
    using clock = std::chrono::steady_clock;

    const char* CSV_PATH = "results_sequential.csv";
    const std::size_t sizes[] = { 10'000, 100'000, 1'000'000 };

    std::ofstream csv(CSV_PATH, std::ios::trunc);
    csv << "implementation,N,num_threads,execution_ms,computation_ms,transfer_ms,"
           "transfer_bytes,total_ops,performance_mops,sorted_ok\n";

    std::cout << "implementation = sequential (base-256 LSD, uint32, 4 passes)\n";
    std::cout << "------------------------------------------------------------\n";

    for (std::size_t N : sizes) {
        std::vector<uint32_t> data(N);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<uint32_t> dist(
            0, std::numeric_limits<uint32_t>::max());
        for (std::size_t i = 0; i < N; ++i) data[i] = dist(rng);

        std::vector<uint32_t> ref = data;

        // ---- warm-up: run once, discard (warms cache, CPU freq) -----------
        {
            std::vector<uint32_t> warmup = data;
            radix_sort_lsd(warmup);
        }

        // ---- time the radix sort ------------------------------------------
        auto t1 = clock::now();
        radix_sort_lsd(data);
        auto t2 = clock::now();
        double execution_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double computation_ms = execution_ms;     // no transfer phase
        double transfer_ms    = 0.0;
        std::uint64_t transfer_bytes = 0;

        std::sort(ref.begin(), ref.end());        // reference oracle

        // uint32_t always needs exactly 4 byte-passes; no max_val scan needed.
        constexpr int D = 4;

        const std::uint64_t total_ops =
            static_cast<std::uint64_t>(D) *
            (5ull * static_cast<std::uint64_t>(N) + static_cast<std::uint64_t>(NUM_BUCKETS));
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