// clang++ -std=c++17 -O2 -pthread -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -o 3_openmp 3_openmp.cpp
// ./3_openmp

/**
 * Parallel LSD Radix Sort — Base 256, OpenMP
 *
 * Sources used:
 *   [1] Haichuan Wang (Intel Research):
 *       "A faster OpenMP Radix Sort implementation" (2014)
 *       https://haichuanwang.wordpress.com/2014/05/26/
 *       → Parallelise all three phases; use schedule(static) consistently;
 *         keep prefix-sum sequential inside omp single (256 entries is tiny).
 *
 *   [2] Dmitry Vyukov (1024cores.net):
 *       "Parallel Radix Sort"
 *       https://www.1024cores.net/home/parallel-computing/radix-sort
 *       → Store T independent local histograms (T×B matrix) to eliminate
 *         any critical section during counting.
 *
 *   [3] PiotrSypek/pradsort (GitHub)
 *       https://github.com/PiotrSypek/pradsort
 *       → LSD-first, base-256 uint32_t reference implementation.
 *
 * ┌─ Algorithm (one pass over byte `shift` of each uint32_t) ────────────────┐
 * │                                                                           │
 * │  Phase 1 — parallel count                                                │
 * │    Each thread independently fills its own 256-bucket histogram over     │
 * │    its static chunk.  No critical section: thread t writes only to       │
 * │    local_hist[t][0..255].                                                │
 * │    Directive: #pragma omp for schedule(static)  [implicit barrier]       │
 * │                                                                           │
 * │  Phase 2 — sequential prefix-sum  (omp single)                          │
 * │    One thread computes global totals from the T×256 matrix, builds an    │
 * │    exclusive prefix-sum of bucket totals, then fills per-thread scatter  │
 * │    offsets.  Running this on one thread is *faster* than parallelising   │
 * │    it: the prefix-sum touches only 256 values (< 1 KB).  [Wang 2014]    │
 * │    Directive: #pragma omp single  [implicit barrier on exit]             │
 * │                                                                           │
 * │  Phase 3 — parallel scatter                                              │
 * │    Each thread scatters its static chunk to dst[] using its own row of   │
 * │    offsets.  schedule(static) with no chunk-size guarantees the SAME     │
 * │    partition as Phase 1, so every thread writes exactly to the slots     │
 * │    it counted — no race conditions, stability preserved.  [Wang 2014]   │
 * │    Directive: #pragma omp for schedule(static)  [implicit barrier]       │
 * │                                                                           │
 * │  Buffer swap (main thread) then repeat for passes 1-3.                  │
 * └───────────────────────────────────────────────────────────────────────────┘
 *
 * After 4 passes (even number of swaps) the sorted result is back in `data`.
 *
 * Complexity : O(d·(n + B))   work       d = 4 passes, B = 256 buckets
 *              O(d·(n/T + B)) span        T = thread count
 * Extra space: O(n + T·B)
 */

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
 
 // ════════════════════════════════════════════════════════════════════════════
 //  Constants
 // ════════════════════════════════════════════════════════════════════════════
 static constexpr int B      = 256;   // buckets (one per byte value)
 static constexpr int PASSES = 4;     // 4 × 8 bits = 32-bit keys
 
 // ════════════════════════════════════════════════════════════════════════════
 //  Core: one pass of counting sort over byte `shift`
 //  Called from radix_sort_omp() with src/dst already set for this pass.
 // ════════════════════════════════════════════════════════════════════════════
 static void omp_counting_pass(
     const uint32_t* __restrict__ src,
           uint32_t* __restrict__ dst,
     std::size_t n,
     int         shift,
     int         T,           // thread count this parallel region will use
     uint32_t*   local_hist,  // T × B, zeroed before entry
     uint32_t*   offsets)     // T × B, output
 {
     // ── All three phases live inside ONE parallel region ─────────────────
     // Opening the region once amortises the fork/join overhead over all
     // three phases.  [Recommendation: Wang 2014, Vyukov 1024cores]
     #pragma omp parallel num_threads(T)
     {
         const int tid = omp_get_thread_num();
         uint32_t* hist = local_hist + tid * B;   // this thread's histogram row
 
         // ── Phase 1: parallel count ───────────────────────────────────────
         // schedule(static) with no chunk → OpenMP divides [0,n) into T
         // contiguous, equal-sized (±1) pieces deterministically.
         // Thread tid processes the SAME indices in Phase 3 below.  [Wang 2014]
         #pragma omp for schedule(static)
         for (std::size_t i = 0; i < n; ++i)
             ++hist[(src[i] >> shift) & 0xFF];
         // implicit barrier at end of omp-for
 
         // ── Phase 2: prefix-sum  (single thread) ─────────────────────────
         // Sequential over 256 values: faster than launching a parallel
         // reduction here.  All other threads wait at the implicit barrier
         // that follows omp single.  [Wang 2014]
         #pragma omp single
         {
             // Step A – global total per bucket
             uint32_t total[B] = {};
             for (int t = 0; t < T; ++t)
                 for (int b = 0; b < B; ++b)
                     total[b] += local_hist[t * B + b];
 
             // Step B – exclusive prefix-sum of totals → global bucket starts
             uint32_t gstart[B];
             gstart[0] = 0;
             for (int b = 1; b < B; ++b)
                 gstart[b] = gstart[b - 1] + total[b - 1];
 
             // Step C – per-thread scatter offsets
             // offsets[t][b] = position in dst[] where thread t begins
             //                 writing elements of bucket b.
             for (int b = 0; b < B; ++b) {
                 uint32_t pos = gstart[b];
                 for (int t = 0; t < T; ++t) {
                     offsets[t * B + b] = pos;
                     pos += local_hist[t * B + b];
                 }
             }
         }
         // implicit barrier after omp single — all threads see offsets[]
 
         // ── Phase 3: parallel scatter ─────────────────────────────────────
         // Each thread scatters its OWN static chunk (same partition as Phase 1)
         // using its own offset row → zero overlap, no race condition.
         // Iterating i in ascending order preserves input-order within each
         // bucket → sort remains stable.  [Vyukov, 1024cores]
         uint32_t* off = offsets + tid * B;
         #pragma omp for schedule(static)
         for (std::size_t i = 0; i < n; ++i) {
             const int bkt = (src[i] >> shift) & 0xFF;
             dst[off[bkt]++] = src[i];
         }
         // implicit barrier at end of omp-for
     }
     // All threads rejoined — dst[] holds the pass result.
 }
 
 // ════════════════════════════════════════════════════════════════════════════
 //  Public entry point
 // ════════════════════════════════════════════════════════════════════════════
 void radix_sort_omp(std::vector<uint32_t>& data, int T)
 {
     const std::size_t n = data.size();
     if (n < 2) return;
 
     std::vector<uint32_t> buf(n);          // alternating output buffer
     std::vector<uint32_t> local_hist(T * B);
     std::vector<uint32_t> offsets(T * B);
 
     uint32_t* src = data.data();
     uint32_t* dst = buf.data();
 
     for (int pass = 0; pass < PASSES; ++pass) {
         // Zero local histograms before each pass (cheap: T × 256 entries)
         std::fill(local_hist.begin(), local_hist.end(), 0u);
 
         omp_counting_pass(src, dst, n, pass * 8, T,
                           local_hist.data(), offsets.data());
 
         std::swap(src, dst);
         // After swap:
         //   pass 0: src=buf,  dst=data  (result of pass 0 now in buf)
         //   pass 1: src=data, dst=buf
         //   pass 2: src=buf,  dst=data
         //   pass 3: src=data, dst=buf
         // 4 swaps (even) → src ends on data.data() → result is in data. ✓
     }
 }
 
 // ════════════════════════════════════════════════════════════════════════════
 //  Helpers
 // ════════════════════════════════════════════════════════════════════════════
 static std::vector<uint32_t> random_data(std::size_t n, uint32_t seed = 42)
 {
     std::mt19937 rng(seed);
     std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
     std::vector<uint32_t> v(n);
     for (auto& x : v) x = dist(rng);
     return v;
 }
 
 static bool is_sorted_check(const std::vector<uint32_t>& v)
 {
     for (std::size_t i = 1; i < v.size(); ++i)
         if (v[i] < v[i - 1]) return false;
     return true;
 }
 
 // ════════════════════════════════════════════════════════════════════════════
 //  Benchmark
 // ════════════════════════════════════════════════════════════════════════════
 // Op model shared with all 4 impls: PASSES * (4*N + B)
 static long long radix_total_ops(std::size_t n) {
     return static_cast<long long>(PASSES) * (4LL * static_cast<long long>(n) + B);
 }

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

 static Result benchmark(std::size_t n, int T, int runs = 20)
 {
     std::vector<double> times;
     times.reserve(runs);
     bool ok = true;

     // warmup (untimed)
     {
         auto data = random_data(n, 0u);
         radix_sort_omp(data, T);
     }

     for (int r = 0; r < runs; ++r) {
         auto data = random_data(n, r * 1337u + 7u);
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
     res.computation_ms   = median_ms;   // CPU: no host/device split
     res.transfer_ms      = 0.0;
     res.transfer_bytes   = 0;
     res.total_ops        = radix_total_ops(n);
     res.performance_mops = static_cast<double>(res.total_ops) / (median_ms * 1000.0);
     res.correct          = ok;
     return res;
 }
 
 // ════════════════════════════════════════════════════════════════════════════
 //  Main
 // ════════════════════════════════════════════════════════════════════════════
 int main()
 {
     const std::vector<std::size_t> sizes   = {10'000, 100'000, 1'000'000, 10'000'000};
     const std::vector<int>         threads = {1, 2, 4, 8, 16};
     const int RUNS = 20;
 
     // Report available hardware threads
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

     // ── Write CSV (unified schema) ────────────────────────────────────────
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