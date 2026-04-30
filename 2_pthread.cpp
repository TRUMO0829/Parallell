// g++ -O2 -std=c++17 -pthread 2_pthread.cpp -o 2_pthread
// ./2_pthread
//
// Parallel LSD radix sort using POSIX threads.
// Each thread owns a contiguous chunk of the array; barriers synchronize
// the three phases of every pass: local histogram -> global prefix-sum
// (thread 0 only) -> parallel scatter.

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

static constexpr int B      = 256;   // buckets (one per byte value)
static constexpr int PASSES = 4;     // 4 x 8 bits = 32-bit keys

// macOS does not implement pthread_barrier_t (it's optional in POSIX).
// Roll a portable barrier using a mutex + condition variable + "trip" counter.
struct PthreadBarrier {
    pthread_mutex_t mtx{};
    pthread_cond_t  cv{};
    int             arrived = 0;
    int             trip    = 0;     // generation counter; flips each release
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

    // Block until all `n` threads have called wait().
    void wait() {
        pthread_mutex_lock(&mtx);
        const int my_trip = trip;
        if (++arrived == n) {
            // Last thread: reset, advance trip, wake everyone.
            arrived = 0;
            ++trip;
            pthread_cond_broadcast(&cv);
            pthread_mutex_unlock(&mtx);
        } else {
            // Wait until trip advances (release).
            while (my_trip == trip)
                pthread_cond_wait(&cv, &mtx);
            pthread_mutex_unlock(&mtx);
        }
    }
};

// Shared state passed to every worker thread.
struct SortCtx {
    uint32_t*    src;          // input buffer this pass
    uint32_t*    dst;          // output buffer this pass
    std::size_t  n;
    int          T;            // thread count
    uint32_t*    local_hist;   // T x B matrix (one histogram row per thread)
    uint32_t*    offsets;      // T x B matrix (per-thread scatter starts)
    int          pass;
    PthreadBarrier* barrier;
};

struct ThreadArg { SortCtx* ctx; int tid; };

// Worker: processes its chunk through 4 passes, with 4 barriers per pass.
static void* radix_worker(void* raw)
{
    ThreadArg* ta  = reinterpret_cast<ThreadArg*>(raw);
    SortCtx*   ctx = ta->ctx;
    const int  tid = ta->tid;
    const int  T   = ctx->T;
    const std::size_t n = ctx->n;

    // Each thread owns a contiguous chunk [lo, hi) of the input.
    const std::size_t chunk = (n + T - 1) / T;
    const std::size_t lo    = static_cast<std::size_t>(tid) * chunk;
    const std::size_t hi    = std::min(lo + chunk, n);

    for (int pass = 0; pass < PASSES; ++pass) {
        const int shift = pass * 8;

        // Phase 1 — Local histogram (parallel, no contention).
        // Thread t writes only to its own row local_hist[t][..].
        uint32_t* hist = ctx->local_hist + tid * B;
        std::fill(hist, hist + B, 0u);
        for (std::size_t i = lo; i < hi; ++i)
            ++hist[(ctx->src[i] >> shift) & 0xFF];

        ctx->barrier->wait();

        // Phase 2 — Global prefix-sum (thread 0 only; T-1 others wait).
        // Builds per-thread offsets so the parallel scatter has no overlap.
        if (tid == 0) {
            // Sum local histograms across threads -> bucket totals.
            uint32_t total[B] = {};
            for (int t = 0; t < T; ++t)
                for (int b = 0; b < B; ++b)
                    total[b] += ctx->local_hist[t * B + b];

            // Exclusive scan -> global start of each bucket in dst.
            uint32_t gstart[B];
            gstart[0] = 0;
            for (int b = 1; b < B; ++b)
                gstart[b] = gstart[b - 1] + total[b - 1];

            // offsets[t][b] = position where thread t writes bucket b first.
            for (int b = 0; b < B; ++b) {
                uint32_t pos = gstart[b];
                for (int t = 0; t < T; ++t) {
                    ctx->offsets[t * B + b] = pos;
                    pos += ctx->local_hist[t * B + b];
                }
            }
        }

        ctx->barrier->wait();

        // Phase 3 — Scatter (parallel; each thread writes its own slots).
        uint32_t* off = ctx->offsets + tid * B;
        for (std::size_t i = lo; i < hi; ++i) {
            const int bkt = (ctx->src[i] >> shift) & 0xFF;
            ctx->dst[off[bkt]++] = ctx->src[i];
        }

        ctx->barrier->wait();

        // Swap buffers for the next pass (thread 0 only).
        if (tid == 0) std::swap(ctx->src, ctx->dst);

        ctx->barrier->wait();
    }
    return nullptr;
}

// Entry point: allocate buffers, spawn T workers, join them.
void parallel_radix_sort(std::vector<uint32_t>& data, int T)
{
    const std::size_t n = data.size();
    if (n < 2) return;
    T = std::max(1, T);

    std::vector<uint32_t> buf(n);                  // alternating output buffer
    std::vector<uint32_t> local_hist(T * B, 0);    // T independent histograms
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

    // Warmup (untimed): hot caches and amortize thread-spawn cost.
    {
        auto data = random_data(n, 0u);
        parallel_radix_sort(data, T);
    }

    for (int r = 0; r < runs; ++r) {
        auto data = random_data(n, r * 1337u + 7u);   // fresh data per run
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
    const int RUNS = 40;

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

    // Write CSV using the unified schema shared with all 4 implementations.
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
