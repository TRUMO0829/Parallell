//   nvcc -O3 -std=c++17 -arch=sm_120 4_cuda.cu -o 4_cuda
//   ./4_cuda


#include <cuda_runtime.h>
#include <cub/cub.cuh>

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

static constexpr int NUM_BUCKETS        = 256;    // base-256
static constexpr int PASSES             = 4;      // 4 × 8 bits = 32 bits
static constexpr int HIST_BLOCK_THREADS = 256;    // threads/block in histogram
static constexpr int ITEMS_PER_BLOCK    = 8192;   // chunk size per block

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << "  at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

//  Phase 1 — block-level histogram
__global__ void histogram_kernel(const uint32_t* __restrict__ src,
                                  uint32_t* __restrict__ g_hist,
                                  int n, int shift, int num_blocks)
{
    __shared__ uint32_t s_hist[NUM_BUCKETS];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Init shared histogram (256 entries, 256 threads → one each)
    if (tid < NUM_BUCKETS) s_hist[tid] = 0u;
    __syncthreads();

    // Each block processes ITEMS_PER_BLOCK contiguous source elements.
    // Threads stride by HIST_BLOCK_THREADS so reads from src are coalesced.
    const int block_start = bid * ITEMS_PER_BLOCK;
    const int block_end   = min(block_start + ITEMS_PER_BLOCK, n);

    for (int i = block_start + tid; i < block_end; i += HIST_BLOCK_THREADS) {
        const int byte = (src[i] >> shift) & 0xFF;
        atomicAdd(&s_hist[byte], 1u);   // shared-mem atomic — fast on modern HW
    }
    __syncthreads();

    // Write block's 256 counts to global memory in BUCKET-MAJOR layout:
    //     g_hist[bucket * num_blocks + bid]
    // (The single ExclusiveSum below relies on this layout being correct.)
    if (tid < NUM_BUCKETS)
        g_hist[tid * num_blocks + bid] = s_hist[tid];
}

//  Phase 3 — block-level scatter (1 thread per block, stable)
__global__ void scatter_kernel(const uint32_t* __restrict__ src,
                                uint32_t* __restrict__ dst,
                                const uint32_t* __restrict__ g_offsets,
                                int n, int shift, int num_blocks)
{
    const int bid = blockIdx.x;

    // Load this block's 256 offsets from global. May spill to local memory
    // (which is L1-cached) if it doesn't fit in registers — that's fine here.
    uint32_t off[NUM_BUCKETS];
    #pragma unroll 1
    for (int b = 0; b < NUM_BUCKETS; ++b)
        off[b] = g_offsets[b * num_blocks + bid];

    const int block_start = bid * ITEMS_PER_BLOCK;
    const int block_end   = min(block_start + ITEMS_PER_BLOCK, n);

    // Walk in source order, post-increment for stability.
    for (int i = block_start; i < block_end; ++i) {
        const uint32_t v   = src[i];
        const int      bkt = (v >> shift) & 0xFF;
        dst[off[bkt]++]    = v;
    }
}

//  Device-side state + sort orchestration
struct GpuSortBuffers {
    uint32_t* d_keys         = nullptr;
    uint32_t* d_buf          = nullptr;
    uint32_t* d_hist         = nullptr;
    void*     d_scan_tmp     = nullptr;
    size_t    scan_tmp_bytes = 0;
    int       n              = 0;
    int       num_blocks     = 0;
};

static GpuSortBuffers allocate_buffers(int n)
{
    GpuSortBuffers b;
    b.n          = n;
    b.num_blocks = (n + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
    const int hist_size = b.num_blocks * NUM_BUCKETS;

    CUDA_CHECK(cudaMalloc(&b.d_keys, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.d_buf,  n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.d_hist, hist_size * sizeof(uint32_t)));

    // Query CUB for required scan temp storage, then allocate it once.
    cub::DeviceScan::ExclusiveSum(
        nullptr, b.scan_tmp_bytes, b.d_hist, b.d_hist, hist_size);
    CUDA_CHECK(cudaMalloc(&b.d_scan_tmp, b.scan_tmp_bytes));

    return b;
}

static void free_buffers(GpuSortBuffers& b)
{
    if (b.d_keys)     cudaFree(b.d_keys);
    if (b.d_buf)      cudaFree(b.d_buf);
    if (b.d_hist)     cudaFree(b.d_hist);
    if (b.d_scan_tmp) cudaFree(b.d_scan_tmp);
    b = {};
}

static void radix_sort_gpu(GpuSortBuffers& b)
{
    if (b.n < 2) return;

    const int n = b.n;
    const int G = b.num_blocks;
    const int hist_size = G * NUM_BUCKETS;

    uint32_t* src = b.d_keys;
    uint32_t* dst = b.d_buf;

    for (int pass = 0; pass < PASSES; ++pass) {
        const int shift = pass * 8;

        // Phase 1
        histogram_kernel<<<G, HIST_BLOCK_THREADS>>>(src, b.d_hist, n, shift, G);

        // Phase 2 (in-place; counts → exclusive prefix sums = offsets)
        cub::DeviceScan::ExclusiveSum(
            b.d_scan_tmp, b.scan_tmp_bytes,
            b.d_hist, b.d_hist, hist_size);

        // Phase 3
        scatter_kernel<<<G, 1>>>(src, dst, b.d_hist, n, shift, G);

        std::swap(src, dst);
    }
    // 4 swaps (even) → src ends back at b.d_keys → result is in d_keys ✓
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

// Stronger correctness check: compare to std::sort byte-for-byte.
// Catches sorts that produce monotone-but-not-equivalent output.
static bool deep_equal_to_std_sort(std::vector<uint32_t> v_got,
                                    const std::vector<uint32_t>& original)
{
    std::vector<uint32_t> gold = original;
    std::sort(gold.begin(), gold.end());
    if (v_got.size() != gold.size()) return false;
    return std::memcmp(v_got.data(), gold.data(),
                       v_got.size() * sizeof(uint32_t)) == 0;
}

// Op model shared with all 4 impls:  PASSES * (4*N + B)
static long long radix_total_ops(std::size_t n) {
    return static_cast<long long>(PASSES) *
           (4LL * static_cast<long long>(n) + NUM_BUCKETS);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Result struct + benchmark
// ═══════════════════════════════════════════════════════════════════════════
struct Result {
    std::size_t n;
    int         threads;          // we use #blocks (each block = 1 logical worker in scatter)
    double      execution_ms;     // wall-clock, host-side, includes H↔D transfers
    double      computation_ms;   // GPU kernel time only (cudaEvent)
    double      transfer_ms;      // H→D + D→H
    long long   transfer_bytes;
    long long   total_ops;
    double      performance_mops;
    bool        correct;
};

static Result benchmark(std::size_t n, int runs = 20)
{
    std::vector<double> exec_t, compute_t, transfer_t;
    exec_t.reserve(runs); compute_t.reserve(runs); transfer_t.reserve(runs);
    bool ok = true;

    auto bufs = allocate_buffers(static_cast<int>(n));

    cudaEvent_t evK0, evK1;
    CUDA_CHECK(cudaEventCreate(&evK0));
    CUDA_CHECK(cudaEventCreate(&evK1));

    // Warmup (untimed): hot caches, JIT, CUB temp alloc paths
    {
        auto data = random_data(n, 0u);
        CUDA_CHECK(cudaMemcpy(bufs.d_keys, data.data(),
                              n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        radix_sort_gpu(bufs);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int r = 0; r < runs; ++r) {
        auto data = random_data(n, r * 1337u + 7u);
        auto original = data; // for deep correctness check

        auto wall0 = std::chrono::high_resolution_clock::now();

        auto t_h2d_0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(bufs.d_keys, data.data(),
                              n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        auto t_h2d_1 = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaEventRecord(evK0));
        radix_sort_gpu(bufs);
        CUDA_CHECK(cudaEventRecord(evK1));
        CUDA_CHECK(cudaEventSynchronize(evK1));
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, evK0, evK1));

        auto t_d2h_0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(data.data(), bufs.d_keys,
                              n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        auto t_d2h_1 = std::chrono::high_resolution_clock::now();

        auto wall1 = std::chrono::high_resolution_clock::now();

        const double total_ms =
            std::chrono::duration<double, std::milli>(wall1 - wall0).count();
        const double h2d_ms =
            std::chrono::duration<double, std::milli>(t_h2d_1 - t_h2d_0).count();
        const double d2h_ms =
            std::chrono::duration<double, std::milli>(t_d2h_1 - t_d2h_0).count();

        exec_t.push_back(total_ms);
        compute_t.push_back(kernel_ms);
        transfer_t.push_back(h2d_ms + d2h_ms);

        // Strong correctness check (vs std::sort) on every run
        if (!deep_equal_to_std_sort(data, original)) ok = false;
    }

    cudaEventDestroy(evK0);
    cudaEventDestroy(evK1);

    auto median = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    Result res;
    res.n                = n;
    res.threads          = bufs.num_blocks;  // analogous to T in pthread/OpenMP
    res.execution_ms     = median(exec_t);
    res.computation_ms   = median(compute_t);
    res.transfer_ms      = median(transfer_t);
    res.transfer_bytes   = 2LL * static_cast<long long>(n) * sizeof(uint32_t);
    res.total_ops        = radix_total_ops(n);
    res.performance_mops = static_cast<double>(res.total_ops)
                            / (res.computation_ms * 1000.0);
    res.correct          = ok;

    free_buffers(bufs);
    return res;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════
int main()
{
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "\n  Device: " << prop.name
              << "  (compute " << prop.major << "." << prop.minor
              << ", " << prop.multiProcessorCount << " SMs, "
              << (prop.totalGlobalMem >> 30) << " GB)\n";

    const std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000, 10'000'000};
    const int RUNS = 20;

    std::cout << "\n  CUDA LSD Radix Sort (base-256, uint32_t, Path B)\n";
    std::cout << "  " << std::string(86, '-') << "\n";
    std::cout << std::setw(12) << "Elements"
              << std::setw(10) << "Blocks"
              << std::setw(14) << "Exec (ms)"
              << std::setw(14) << "Compute (ms)"
              << std::setw(14) << "Xfer (ms)"
              << std::setw(14) << "Perf (MOPS)"
              << std::setw(10) << "Correct" << "\n";
    std::cout << "  " << std::string(86, '-') << "\n";

    std::vector<Result> all;
    for (auto n : sizes) {
        auto r = benchmark(n, RUNS);
        all.push_back(r);
        std::cout << std::setw(12) << r.n
                  << std::setw(10) << r.threads
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.execution_ms
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.computation_ms
                  << std::setw(14) << std::fixed << std::setprecision(3) << r.transfer_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.performance_mops
                  << std::setw(10) << (r.correct ? "YES" : "NO!") << "\n";
    }
    std::cout << "  " << std::string(86, '-') << "\n";
    std::cout << "\n  (each result is median of " << RUNS << " runs;\n"
              << "   correctness verified vs std::sort byte-for-byte each run)\n\n";

    // Unified CSV schema (matches your other 3 impls)
    const std::string csv_path = "radix_sort_cuda_stats.csv";
    std::ofstream csv(csv_path);
    if (!csv) { std::cerr << "Cannot open " << csv_path << "\n"; return 1; }

    csv << "implementation,N,threads,execution_ms,computation_ms,transfer_ms,"
           "transfer_bytes,total_ops,performance_mops,sorted_ok\n";
    for (auto& r : all) {
        csv << "cuda," << r.n << "," << r.threads << ","
            << std::fixed << std::setprecision(4) << r.execution_ms << ","
            << std::fixed << std::setprecision(4) << r.computation_ms << ","
            << std::fixed << std::setprecision(4) << r.transfer_ms << ","
            << r.transfer_bytes << ","
            << r.total_ops << ","
            << std::fixed << std::setprecision(4) << r.performance_mops << ","
            << (r.correct ? "true" : "false") << "\n";
    }
    csv.close();
    std::cout << "  Stats written to " << csv_path << "\n\n";

    return 0;
}
