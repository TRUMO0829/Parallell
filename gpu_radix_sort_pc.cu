// nvcc -O3 -std=c++17 -arch=native -o gpu_radix_sort gpu_radix_sort_pc.cu
// ./gpu_radix_sort

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s  at %s:%d\n",                       \
                    cudaGetErrorString(_err), __FILE__, __LINE__);              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define BITS_PER_PASS   4
#define RADIX           (1 << BITS_PER_PASS)   
#define NUM_PASSES      (32 / BITS_PER_PASS)   
#define BLOCK_SIZE      256                    
#define TILE_SIZE       2048                   

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

__global__ void k_scatter(const uint32_t* __restrict__ in,
                                uint32_t* __restrict__ out,
                          const uint32_t* __restrict__ prefix,
                          int n, int shift, int num_blocks)
{
    __shared__ uint32_t s_goff[RADIX];      
    __shared__ uint32_t s_rank[TILE_SIZE];  

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid < RADIX)
        s_goff[tid] = prefix[tid * num_blocks + bid];
    __syncthreads();

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

    return d_in;
}

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

    uint32_t *d_a = nullptr, *d_b = nullptr, *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,    (size_t)n         * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_b,    (size_t)n         * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hist, (size_t)hist_size * sizeof(uint32_t)));

    cudaEvent_t e0, e1, ec0, ec1, eh0, eh1, ed0, ed1;
    CUDA_CHECK(cudaEventCreate(&e0));   CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&ec0));  CUDA_CHECK(cudaEventCreate(&ec1));
    CUDA_CHECK(cudaEventCreate(&eh0));  CUDA_CHECK(cudaEventCreate(&eh1));
    CUDA_CHECK(cudaEventCreate(&ed0));  CUDA_CHECK(cudaEventCreate(&ed1));

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

    CUDA_CHECK(cudaFree(d_a));  CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaEventDestroy(e0));   CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(ec0));  CUDA_CHECK(cudaEventDestroy(ec1));
    CUDA_CHECK(cudaEventDestroy(eh0));  CUDA_CHECK(cudaEventDestroy(eh1));
    CUDA_CHECK(cudaEventDestroy(ed0));  CUDA_CHECK(cudaEventDestroy(ed1));

    return res;
}

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

int main(int argc, char** argv)
{
    print_device_info();

    int single_n = 0;
    int seed     = 42;
    int runs     = 20;

    if (argc >= 2) single_n = std::atoi(argv[1]);
    if (argc >= 3) seed     = std::atoi(argv[2]);
    if (argc >= 4) runs     = std::atoi(argv[3]);

    std::vector<int> sizes;
    if (single_n > 0) {
        sizes = {single_n};
    } else {
        sizes = {10000, 100000, 1000000, 5000000, 10000000};
    }

    printf("implementation,N,median_exec_ms,median_compute_ms,"
           "median_transfer_ms,transfer_bytes,performance_mops,sorted_ok\n");

    for (int n : sizes) {
        std::vector<uint32_t> h_in(n);
        {
            std::mt19937 rng(seed);
            for (int i = 0; i < n; i++) h_in[i] = rng();
        }

        BenchResult r = benchmark(h_in, n, runs);

        long long transfer_bytes = 2LL * n * (long long)sizeof(uint32_t);

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
