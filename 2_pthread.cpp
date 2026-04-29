// g++ -O2 -std=c++17 -pthread 2_pthread.cpp -o 2_pthread
// ./2_pthread

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

static constexpr int NUM_BUCKETS = 1 << 8;   // 256 buckets (8-bit radix)

static void counting_sort_by_digit(std::vector<uint32_t>& a,
                                   std::vector<uint32_t>& aux,
                                   uint32_t shift,
                                   int num_threads) {
    const std::size_t n = a.size();

    auto chunk_start = [&](int t) -> std::size_t { return (n *  t     ) / num_threads; };
    auto chunk_end   = [&](int t) -> std::size_t { return (n * (t + 1)) / num_threads; };

    std::vector<std::array<int, NUM_BUCKETS>> local(num_threads);
    for (auto& h : local) h.fill(0);

    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const std::size_t s = chunk_start(t);
                const std::size_t e = chunk_end(t);
                auto& h = local[t];
                for (std::size_t i = s; i < e; ++i) {
                    int d = static_cast<int>((a[i] >> shift) & 0xFFu);
                    h[d]++;
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    std::vector<std::array<int, NUM_BUCKETS>> offset(num_threads);
    {
        int running = 0;
        for (int b = 0; b < NUM_BUCKETS; ++b) {
            for (int t = 0; t < num_threads; ++t) {
                offset[t][b] = running;
                running += local[t][b];
            }
        }
        assert(static_cast<std::size_t>(running) == n);
    }

    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const std::size_t s = chunk_start(t);
                const std::size_t e = chunk_end(t);
                std::array<int, NUM_BUCKETS> off = offset[t];   // private copy, mutate
                for (std::size_t i = s; i < e; ++i) {
                    int d = static_cast<int>((a[i] >> shift) & 0xFFu);
                    aux[off[d]++] = a[i];
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    a.swap(aux);
}

void radix_sort_lsd(std::vector<uint32_t>& a, int num_threads) {
    if (a.size() < 2) return;

    std::vector<uint32_t> aux(a.size());

    // uint32_t = 4 bytes => exactly 4 LSD passes (shift = 0, 8, 16, 24).
    for (uint32_t shift = 0; shift < 32; shift += 8) {
        counting_sort_by_digit(a, aux, shift, num_threads);
    }
}

int main() {
    using clock = std::chrono::steady_clock;

    const char* CSV_PATH = "results_stdthread.csv";
    const std::size_t sizes[]         ={ 10'000, 100'000, 1'000'000 };
    const int         thread_counts[] = { 1, 2, 4, 8, 16 };

    std::ofstream csv(CSV_PATH, std::ios::trunc);
    csv << "implementation,N,num_threads,execution_ms,computation_ms,transfer_ms,"
           "transfer_bytes,total_ops,performance_mops,sorted_ok\n";

    std::cout << "implementation = stdthread (base-256 LSD, uint32, 4 passes, strong-scaling sweep)\n";
    std::cout << "HW concurrency = " << std::thread::hardware_concurrency() << "\n";
    std::cout << "------------------------------------------------------------\n";

    for (int num_threads : thread_counts) {
        std::cout << "\n=== num_threads = " << num_threads << " ===\n";

        for (std::size_t N : sizes) {
            std::vector<uint32_t> data(N);
            std::mt19937 rng(12345);
            std::uniform_int_distribution<uint32_t> dist(
                0, std::numeric_limits<uint32_t>::max());
            for (std::size_t i = 0; i < N; ++i) data[i] = dist(rng);

            std::vector<uint32_t> ref = data;

            // warm-up: run once, discard (warms cache, CPU freq, thread state)
            {
                std::vector<uint32_t> warmup = data;
                radix_sort_lsd(warmup, num_threads);
            }

            auto t1 = clock::now();
            radix_sort_lsd(data, num_threads);
            auto t2 = clock::now();
            double execution_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double computation_ms = execution_ms;
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