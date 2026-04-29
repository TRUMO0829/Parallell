// gpu_radix_sort_pc.cu
// ============================================================================
// GPU LSD radix sort for desktop RTX GPUs (Blackwell · Ada · Ampere)
// Based on Harada & Howes, "Introduction to GPU Radix Sort" (AMD).
//
// Algorithm: 4 bits per pass · RADIX = 16 bins · 8 passes for uint32_t
//
// Per pass:
//   1. k_block_histogram  – each block builds a 16-bin histogram of its tile
//      in shared memory and writes it column-major to global memory.
//   2. k_global_scan      – single-thread exclusive prefix sum over the
//      (bin × block) table → global write offsets for every (bin, block) pair.
//   3. k_scatter          – each block writes its tile to the correct global
//      positions in stable order (thread 0 assigns sequential per-bin ranks,
//      all threads write in parallel).
//
// After 8 passes (even) the alternating buffers leave the result in buf_a.
//
// ============================================================================
// BUILD
// ============================================================================
//
//   ── Linux / WSL ──────────────────────────────────────────────────────────
//   # CUDA 12.4+: auto-detect your GPU's SM (recommended)
//   nvcc -O3 -std=c++17 -arch=native -o gpu_radix_sort gpu_radix_sort_pc.cu
//
//   # RTX 5080 / Blackwell (sm_120), explicit target:
//   nvcc -O3 -std=c++17 -arch=sm_120 -o gpu_radix_sort gpu_radix_sort_pc.cu
//
//   # RTX 4090 / Ada Lovelace (sm_89):
//   nvcc -O3 -std=c++17 -arch=sm_89  -o gpu_radix_sort gpu_radix_sort_pc.cu
//
//   # RTX 3090 / Ampere (sm_86):
//   nvcc -O3 -std=c++17 -arch=sm_86  -o gpu_radix_sort gpu_radix_sort_pc.cu
//
//   ── Windows (Developer Command Prompt / PowerShell) ───────────────────────
//   nvcc -O3 -std=c++17 -arch=native -o gpu_radix_sort.exe gpu_radix_sort_pc.cu
//
//   ── Don't know your SM? ───────────────────────────────────────────────────
//   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
//   # Then use  -arch=sm_<major><minor>  (e.g. 12.0 → -arch=sm_120)
//
// ============================================================================
// RUN
// ============================================================================
//
//   ./gpu_radix_sort                   # full benchmark sweep (N = 10k…10M)
//   ./gpu_radix_sort 1000000           # single N
//   ./gpu_radix_sort 1000000 42 20     # N, seed, timed-runs
//   ./gpu_radix_sort 0                 # trigger full sweep explicitly
//
// CSV output columns (same format as the CPU versions):
//   implementation, N, median_exec_ms, median_compute_ms, median_transfer_ms,
//   transfer_bytes, performance_mops, sorted_ok
//
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <cassert>
#include <cuda_runtime.h>

// ── error-checking macro ─────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s  at %s:%d\n",                     \
                    cudaGetErrorString(_err), __FILE__, __LINE__);              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ── tuning constants ─────────────────────────────────────────────────────────
#define BITS_PER_PASS   4
#define RADIX           (1 << BITS_PER_PASS)    // 16
#define NUM_PASSES      (32 / BITS_PER_PASS)    // 8
#define BLOCK_SIZE      256                     // threads per block
#define TILE_SIZE       2048                    // elements per block
                                                // (↑ doubled vs Colab version;
                                                //  better BW utilisation on
                                                //  high-VRAM desktop GPUs)

// ============================================================================
// Kernel 1: per-block histogram
// Each block counts the 4-bit digit of its TILE_SIZE elements and writes a
// 16-entry histogram column-major: block_hist[bin * num_blocks + bid].
// ============================================================================
__global__ void k_block_histogram(const uint32_t* __restrict__ in,
                                         uint32_t* __restrict__ block_hist,
                                   int n, int shift, int num_blocks)
{
    __shared__ uint32_t lhist[RADIX];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid < RADIX) lhist[tid] = 0;
    __syncthreads();

    int base = bid * TILE_SIZE;
    for (int i = tid; i < TILE_SIZE; i += BLOCK_SIZE) {
        int gi = base + i;
        if (gi < n) {
            uint32_t bin = (in[gi] >> shift) & (RADIX - 1);
            atomicAdd(&lhist[bin], 1u);
        }
    }
    __syncthreads();

    if (tid < RADIX)
        block_hist[tid * num_blocks + bid] = lhist[tid];
}

// ============================================================================
// Kernel 2: global exclusive prefix scan
// Converts the (bin × block) histogram table into global write offsets.
// Single-thread launch; at TILE_SIZE=2048 and N=10M the table is only
// RADIX × ceil(N/TILE_SIZE) = 16 × 4883 ≈ 78 K entries — negligible cost.
// ============================================================================
__global__ void k_global_scan(uint32_t* hist, int total)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t sum = 0;
        for (int i = 0; i < total; i++) {
            uint32_t v = hist[i];
            hist[i]    = sum;
            sum       += v;
        }
    }
}

// ============================================================================
// Kernel 3: stable scatter
// Thread 0 assigns sequential per-bin local ranks for the entire tile (this
// guarantees stability — essential for LSD correctness).  All threads then
// write in parallel using  global_offset[bin] + local_rank[i]  as destination.
// ============================================================================
__global__ void k_scatter(const uint32_t* __restrict__ in,
                                uint32_t* __restrict__ out,
                          const uint32_t* __restrict__ prefix,
                          int n, int shift, int num_blocks)
{
    __shared__ uint32_t s_goff[RADIX];
    __shared__ uint32_t s_rank[TILE_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Load this block's global base offset for each bin
    if (tid < RADIX)
        s_goff[tid] = prefix[tid * num_blocks + bid];
    __syncthreads();

    // Thread 0: sequential rank assignment (maintains stability)
    if (tid == 0) {
        uint32_t cnt[RADIX] = {};
        int base = bid * TILE_SIZE;
        for (int i = 0; i < TILE_SIZE; i++) {
            int gi = base + i;
            if (gi < n) {
                uint32_t bin = (in[gi] >> shift) & (RADIX - 1);
                s_rank[i]    = cnt[bin]++;
            }
        }
    }
    __syncthreads();

    // All threads: parallel write to global memory
    int base = bid * TILE_SIZE;
    for (int i = tid; i < TILE_SIZE; i += BLOCK_SIZE) {
        int gi = base + i;
        if (gi < n) {
            uint32_t v   = in[gi];
            uint32_t bin = (v >> shift) & (RADIX - 1);
            out[s_goff[bin] + s_rank[i]] = v;
        }
    }
}

// ============================================================================
// Host driver: one full sort (all 8 passes)
// Returns the pointer that currently holds the sorted data
// (always d_buf_a after an even number of passes).
// ============================================================================
static uint32_t* run_sort(uint32_t* d_buf_a, uint32_t* d_buf_b,
                           uint32_t* d_hist,
                           int n, int num_blocks, int hist_size)
{
    uint32_t* d_in  = d_buf_a;
    uint32_t* d_out = d_buf_b;

    for (int pass = 0; pass < NUM_PASSES; pass++) {
        int shift = pass * BITS_PER_PASS;

        CUDA_CHECK(cudaMemsetAsync(d_hist, 0,
                   (size_t)hist_size * sizeof(uint32_t)));

        k_block_histogram<<<num_blocks, BLOCK_SIZE>>>(
            d_in, d_hist, n, shift, num_blocks);

        k_global_scan<<<1, 1>>>(d_hist, hist_size);

        k_scatter<<<num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, d_hist, n, shift, num_blocks);

        std::swap(d_in, d_out);
    }

    return d_in;   // after 8 (even) swaps this equals d_buf_a
}

// ============================================================================
// Benchmark helper: run RUNS timed trials, return median times in ms.
// ============================================================================
struct BenchResult {
    double exec_ms;
    double compute_ms;
    double transfer_ms;
    bool   sorted_ok;
};

static BenchResult benchmark(const std::vector<uint32_t>& h_in,
                              int n, int runs)
{
    int num_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    int hist_size  = RADIX * num_blocks;

    // Device allocations
    uint32_t *d_a = nullptr, *d_b = nullptr, *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,    (size_t)n         * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_b,    (size_t)n         * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hist, (size_t)hist_size * sizeof(uint32_t)));

    // CUDA events for timing
    cudaEvent_t e0, e1, ec0, ec1, eh0, eh1, ed0, ed1;
    CUDA_CHECK(cudaEventCreate(&e0));   CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&ec0));  CUDA_CHECK(cudaEventCreate(&ec1));
    CUDA_CHECK(cudaEventCreate(&eh0));  CUDA_CHECK(cudaEventCreate(&eh1));
    CUDA_CHECK(cudaEventCreate(&ed0));  CUDA_CHECK(cudaEventCreate(&ed1));

    // Warmup run (not measured) — amortises JIT and allocator overhead
    {
        CUDA_CHECK(cudaMemcpy(d_a, h_in.data(),
                   (size_t)n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        run_sort(d_a, d_b, d_hist, n, num_blocks, hist_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> v_exec(runs), v_compute(runs), v_transfer(runs);
    std::vector<uint32_t> h_out(n);
    bool sorted_ok = true;

    for (int r = 0; r < runs; r++) {
        CUDA_CHECK(cudaEventRecord(e0));

        CUDA_CHECK(cudaEventRecord(eh0));
        CUDA_CHECK(cudaMemcpy(d_a, h_in.data(),
                   (size_t)n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(eh1));

        CUDA_CHECK(cudaEventRecord(ec0));
        uint32_t* d_res = run_sort(d_a, d_b, d_hist, n, num_blocks, hist_size);
        CUDA_CHECK(cudaEventRecord(ec1));

        CUDA_CHECK(cudaEventRecord(ed0));
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_res,
                   (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ed1));

        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));

        float exec_ms=0, comp_ms=0, h2d_ms=0, d2h_ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&exec_ms, e0,  e1));
        CUDA_CHECK(cudaEventElapsedTime(&comp_ms, ec0, ec1));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  eh0, eh1));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  ed0, ed1));

        v_exec[r]     = exec_ms;
        v_compute[r]  = comp_ms;
        v_transfer[r] = h2d_ms + d2h_ms;

        if (r == 0)
            sorted_ok = std::is_sorted(h_out.begin(), h_out.end());
    }

    // Median
    auto median_f = [](std::vector<float>& v) -> double {
        std::sort(v.begin(), v.end());
        int n = (int)v.size();
        return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0
                             : (double)v[n/2];
    };

    BenchResult res;
    res.exec_ms     = median_f(v_exec);
    res.compute_ms  = median_f(v_compute);
    res.transfer_ms = median_f(v_transfer);
    res.sorted_ok   = sorted_ok;

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));  CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaEventDestroy(e0));   CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(ec0));  CUDA_CHECK(cudaEventDestroy(ec1));
    CUDA_CHECK(cudaEventDestroy(eh0));  CUDA_CHECK(cudaEventDestroy(eh1));
    CUDA_CHECK(cudaEventDestroy(ed0));  CUDA_CHECK(cudaEventDestroy(ed1));

    return res;
}

// ============================================================================
// Print GPU info at startup
// ============================================================================
static void print_device_info()
{
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr,
        "GPU: %s  |  SM %d.%d  |  %d SMs  |  %.1f GB VRAM  "
        "|  %.1f GB/s peak BW\n",
        prop.name,
        prop.major, prop.minor,
        prop.multiProcessorCount,
        (double)prop.totalGlobalMem / (1 << 30),
        2.0 * prop.memoryClockRate * 1e3 *
            prop.memoryBusWidth / 8.0 / 1e9);
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv)
{
    print_device_info();

    // Defaults
    int single_n = 0;       // 0 = sweep mode
    int seed     = 42;
    int runs     = 20;

    if (argc >= 2) single_n = std::atoi(argv[1]);
    if (argc >= 3) seed     = std::atoi(argv[2]);
    if (argc >= 4) runs     = std::atoi(argv[3]);

    // Benchmark targets
    std::vector<int> sizes;
    if (single_n > 0) {
        sizes = {single_n};
    } else {
        // Full sweep
        sizes = {10000, 100000, 1000000, 5000000, 10000000};
    }

    // CSV header
    printf("implementation,N,median_exec_ms,median_compute_ms,"
           "median_transfer_ms,transfer_bytes,performance_mops,sorted_ok\n");

    for (int n : sizes) {
        // Generate input once per N
        std::vector<uint32_t> h_in(n);
        {
            std::mt19937 rng(seed);
            for (int i = 0; i < n; i++) h_in[i] = rng();
        }

        BenchResult r = benchmark(h_in, n, runs);

        long long transfer_bytes = 2LL * n * (long long)sizeof(uint32_t);
        // 8 passes × ~1.5 effective ops/element (histogram + scatter)
        // + small global scan overhead
        long long total_ops = 12LL * n + 8192LL * NUM_PASSES;
        double mops = (double)total_ops / (r.exec_ms * 1000.0);

        printf("cuda_pc,%d,%.4f,%.4f,%.4f,%lld,%.2f,%s\n",
               n,
               r.exec_ms, r.compute_ms, r.transfer_ms,
               transfer_bytes,
               mops,
               r.sorted_ok ? "true" : "false");

        fflush(stdout);
    }

    return 0;
}
