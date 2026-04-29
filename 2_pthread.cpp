
// g++ -O2 -std=c++17 -pthread 2_pthread.cpp -o 2_pthread
// ./2_pthread

 #include <pthread.h>

 #include <algorithm>
 #include <chrono>
 #include <cstdint>
 #include <cstring>
 #include <fstream>
 #include <iomanip>
 #include <iostream>
 #include <random>
 #include <string>
 #include <vector>
 
 static constexpr int B      = 256;  
 static constexpr int PASSES = 4;

 // macOS does not implement pthread_barrier_t (optional in POSIX).
 struct PthreadBarrier {
     pthread_mutex_t mtx{};
     pthread_cond_t  cv{};
     int             arrived = 0;
     int             trip    = 0;
     const int       n;

     explicit PthreadBarrier(int thread_count) : n(thread_count) {
         pthread_mutex_init(&mtx, nullptr);
         pthread_cond_init(&cv, nullptr);
     }

     ~PthreadBarrier() {
         pthread_cond_destroy(&cv);
         pthread_mutex_destroy(&mtx);
     }

     PthreadBarrier(const PthreadBarrier&)            = delete;
     PthreadBarrier& operator=(const PthreadBarrier&) = delete;

     void wait() {
         pthread_mutex_lock(&mtx);
         const int my_trip = trip;
         if (++arrived == n) {
             arrived = 0;
             ++trip;
             pthread_cond_broadcast(&cv);
             pthread_mutex_unlock(&mtx);
         } else {
             while (my_trip == trip)
                 pthread_cond_wait(&cv, &mtx);
             pthread_mutex_unlock(&mtx);
         }
     }
 };

 struct SortCtx {
     uint32_t*    src;    
     uint32_t*    dst;    
     std::size_t  n;      
     int          T;      
 
     uint32_t*    local_hist; 
 
     uint32_t*    offsets;  
 
     int          pass;

     PthreadBarrier* barrier;
 };
 
 struct ThreadArg { SortCtx* ctx; int tid; };
 
 //  Worker thread
 static void* radix_worker(void* raw)
 {
     ThreadArg* ta  = reinterpret_cast<ThreadArg*>(raw);
     SortCtx*   ctx = ta->ctx;
     const int  tid = ta->tid;
     const int  T   = ctx->T;
     const std::size_t n = ctx->n;
 
     // ── Compute this thread's contiguous chunk ─────────────────────
     const std::size_t chunk = (n + T - 1) / T;
     const std::size_t lo    = static_cast<std::size_t>(tid) * chunk;
     const std::size_t hi    = std::min(lo + chunk, n);
 
     for (int pass = 0; pass < PASSES; ++pass) {
         const int shift = pass * 8;
 
         // ── Phase 1 : local histogram ─────────────────────────────
         uint32_t* hist = ctx->local_hist + tid * B;
         std::fill(hist, hist + B, 0u);
         for (std::size_t i = lo; i < hi; ++i)
             ++hist[(ctx->src[i] >> shift) & 0xFF];
 
         ctx->barrier->wait();
 
         // ── Phase 2 : global prefix-sum  (thread 0 only) ──────────
         if (tid == 0) {
             // Step A: total elements per bucket across all threads
             uint32_t total[B] = {};
             for (int t = 0; t < T; ++t)
                 for (int b = 0; b < B; ++b)
                     total[b] += ctx->local_hist[t * B + b];
 
             // Step B: exclusive prefix-sum of totals → global start per bucket
             uint32_t gstart[B];
             gstart[0] = 0;
             for (int b = 1; b < B; ++b)
                 gstart[b] = gstart[b - 1] + total[b - 1];
 
             // Step C: per-thread offset = global_start[b]
             //         + (local counts of bucket b from earlier threads)
             for (int b = 0; b < B; ++b) {
                 uint32_t pos = gstart[b];
                 for (int t = 0; t < T; ++t) {
                     ctx->offsets[t * B + b] = pos;
                     pos += ctx->local_hist[t * B + b];
                 }
             }
        }

        ctx->barrier->wait();

         // ── Phase 3 : scatter to output buffer ────────────────────
         uint32_t* off = ctx->offsets + tid * B;
         for (std::size_t i = lo; i < hi; ++i) {
             const int bkt = (ctx->src[i] >> shift) & 0xFF;
             ctx->dst[off[bkt]++] = ctx->src[i];
         }
 
         ctx->barrier->wait();
 
         // ── Swap buffers (thread 0) then notify others ─────────────
         if (tid == 0) std::swap(ctx->src, ctx->dst);
 
         ctx->barrier->wait();
     }
     return nullptr;
 }
 
 //  Public entry point
 void parallel_radix_sort(std::vector<uint32_t>& data, int T)
 {
     const std::size_t n = data.size();
     if (n < 2) return;
     T = std::max(1, T);
 
     std::vector<uint32_t> buf(n);   // output buffer (alternates with data)
 
     std::vector<uint32_t> local_hist(T * B, 0);
     std::vector<uint32_t> offsets(T * B, 0);
 
     SortCtx ctx;
     ctx.src        = data.data();
     ctx.dst        = buf.data();
     ctx.n          = n;
     ctx.T          = T;
     ctx.local_hist = local_hist.data();
     ctx.offsets    = offsets.data();
    ctx.pass       = 0;
     PthreadBarrier barrier(T);
     ctx.barrier    = &barrier;

     std::vector<pthread_t>  threads(T);
     std::vector<ThreadArg>  args(T);
 
     for (int t = 0; t < T; ++t) {
         args[t] = {&ctx, t};
         pthread_create(&threads[t], nullptr, radix_worker, &args[t]);
     }
    for (int t = 0; t < T; ++t)
         pthread_join(threads[t], nullptr);

}
 
 //  Helpers
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
 
 //  Benchmark
 // Op model shared with all 4 impls: PASSES * (4*N + B)
 //   1 op/elem (histogram) + 3 ops/elem (scatter) + B per pass (prefix sum)
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

     // warmup (untimed) — caches and thread spawn cost
     {
         auto data = random_data(n, 0u);
         parallel_radix_sort(data, T);
     }

     for (int r = 0; r < runs; ++r) {
         auto data = random_data(n, r * 1337u + 7u);
         auto t0   = std::chrono::high_resolution_clock::now();
         parallel_radix_sort(data, T);
         auto t1   = std::chrono::high_resolution_clock::now();

         times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
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

 //  Main
 int main()
 {
     const std::vector<std::size_t> sizes   = {10'000, 100'000, 1'000'000, 10'000'000};
     const std::vector<int>         threads = {1, 2, 4, 8, 16};
     const int RUNS = 40;
 
     // ── Console header ────────────────────────────────────────────
     std::cout << "\n  Parallel LSD Radix Sort (Base-256, pthreads, uint32_t)\n";
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
         bool first = true;
         for (auto T : threads) {
             auto r = benchmark(n, T, RUNS);
             all_results.push_back(r);

             std::cout << std::setw(12) << (first ? std::to_string(r.n) : "")
                       << std::setw(10) << r.threads
                       << std::setw(14) << std::fixed << std::setprecision(3) << r.execution_ms
                       << std::setw(14) << std::fixed << std::setprecision(3) << r.computation_ms
                       << std::setw(14) << std::fixed << std::setprecision(3) << r.transfer_ms
                       << std::setw(14) << std::fixed << std::setprecision(2) << r.performance_mops
                       << std::setw(10) << (r.correct ? "YES" : "NO!")
                       << "\n";
             first = false;
         }
         std::cout << "  " << std::string(78, '-') << "\n";
     }

     std::cout << "\n  (each result is median of " << RUNS << " runs)\n\n";

     // ── Write CSV (unified schema) ────────────────────────────────
     const std::string csv_path = "radix_sort_pthread_stats.csv";
     std::ofstream csv(csv_path);
     if (!csv) {
         std::cerr << "Could not open " << csv_path << "\n";
         return 1;
     }
     csv << "implementation,N,threads,execution_ms,computation_ms,transfer_ms,"
            "transfer_bytes,total_ops,performance_mops,sorted_ok\n";
     for (auto& r : all_results)
         csv << "pthread," << r.n << "," << r.threads << ","
             << std::fixed << std::setprecision(4) << r.execution_ms << ","
             << std::fixed << std::setprecision(4) << r.computation_ms << ","
             << std::fixed << std::setprecision(4) << r.transfer_ms << ","
             << r.transfer_bytes << ","
             << r.total_ops << ","
             << std::fixed << std::setprecision(4) << r.performance_mops << ","
             << (r.correct ? "true" : "false") << "\n";
     csv.close();

     std::cout << "  Stats written to radix_sort_pthread_stats.csv\n\n";
     return 0;
 }