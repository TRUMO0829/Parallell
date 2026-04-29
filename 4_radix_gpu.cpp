%%writefile radix_cuda.cu
//    !nvcc -O3 -std=c++17 -o radix_cuda radix_cuda.cu
//    !./radix_cuda

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#define BLOCKS   8                       
#define THREADS  256                     
#define BINS     256                     
#define PASSES   4                       
#define SCAN_N   (BLOCKS * BINS)         

__global__ void count_kernel(const uint32_t *in,
                             uint32_t *block_hist,
                             int n, uint32_t shift)
{
    __shared__ uint32_t local_hist[BINS];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid < BINS) local_hist[tid] = 0;
    __syncthreads();

    int chunk = (n + BLOCKS - 1) / BLOCKS;
    int start = bid * chunk;
    int end   = (start + chunk < n) ? (start + chunk) : n;

    for (int i = start + tid; i < end; i += THREADS) {
        uint32_t bin = (in[i] >> shift) & 0xFFu;
        atomicAdd(&local_hist[bin], 1u);
    }
    __syncthreads();

    if (tid < BINS) {
        block_hist[tid * BLOCKS + bid] = local_hist[tid];
    }
}

//  KERNEL 2 & 3: Blelloch
__global__ void upsweep_kernel(uint32_t *data, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i   = (tid + 1) * stride - 1;
    if (i < SCAN_N) {
        data[i] += data[i - stride / 2];
    }
}

__global__ void downsweep_kernel(uint32_t *data, int stride)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int i    = (tid + 1) * stride - 1;
    if (i < SCAN_N) {
        int left   = i - stride / 2;
        uint32_t t = data[left];
        data[left] = data[i];
        data[i]   += t;
    }
}

//  KERNEL 4: reorder_kernel
__global__ void reorder_kernel(const uint32_t *in,
                               uint32_t *out,
                               uint32_t *off,
                               int n, uint32_t shift)
{
    if (threadIdx.x != 0) return;
    int bid = blockIdx.x;

    int chunk = (n + BLOCKS - 1) / BLOCKS;
    int start = bid * chunk;
    int end   = (start + chunk < n) ? (start + chunk) : n;

    for (int i = start; i < end; i++) {
        uint32_t v   = in[i];
        uint32_t bin = (v >> shift) & 0xFFu;
        uint32_t pos = off[bin * BLOCKS + bid]++;
        out[pos] = v;
    }
}

static void blelloch_exclusive_scan(uint32_t *d_data)
{
    for (int stride = 2; stride <= SCAN_N; stride <<= 1) {
        int num = SCAN_N / stride;
        int tpb = (num < THREADS) ? num : THREADS;
        int blk = (num + tpb - 1) / tpb;
        upsweep_kernel<<<blk, tpb>>>(d_data, stride);
    }
    cudaMemset(d_data + SCAN_N - 1, 0, sizeof(uint32_t));
    for (int stride = SCAN_N; stride >= 2; stride >>= 1) {
        int num = SCAN_N / stride;
        int tpb = (num < THREADS) ? num : THREADS;
        int blk = (num + tpb - 1) / tpb;
        downsweep_kernel<<<blk, tpb>>>(d_data, stride);
    }
}

struct Timing {
    float ms_total;
    float ms_kernel;
    float ms_h2d;
    float ms_d2h;
};

Timing run_radix_cuda(const uint32_t *host_in, uint32_t *host_out, int N, bool print)
{
    size_t size_in = (size_t)N * sizeof(uint32_t);
    uint32_t *d_A = nullptr, *d_B = nullptr, *d_off = nullptr;
    cudaEvent_t e_total_s, e_total_e, e_gpu_s, e_gpu_e, e_h2d_s, e_h2d_e, e_d2h_s, e_d2h_e;

    cudaEventCreate(&e_total_s); cudaEventCreate(&e_total_e);
    cudaEventCreate(&e_gpu_s);   cudaEventCreate(&e_gpu_e);
    cudaEventCreate(&e_h2d_s);   cudaEventCreate(&e_h2d_e);
    cudaEventCreate(&e_d2h_s);   cudaEventCreate(&e_d2h_e);

    cudaEventRecord(e_total_s);
    cudaMalloc(&d_A, size_in);
    cudaMalloc(&d_B, size_in);
    cudaMalloc(&d_off, SCAN_N * sizeof(uint32_t));

    cudaEventRecord(e_h2d_s);
    cudaMemcpy(d_A, host_in, size_in, cudaMemcpyHostToDevice);
    cudaEventRecord(e_h2d_e);

    cudaEventRecord(e_gpu_s);
    for (int pass = 0; pass < PASSES; pass++) {
        uint32_t shift = (uint32_t)(pass * 8);
        cudaMemset(d_off, 0, SCAN_N * sizeof(uint32_t));
        count_kernel<<<BLOCKS, THREADS>>>(d_A, d_off, N, shift);
        blelloch_exclusive_scan(d_off);
        reorder_kernel<<<BLOCKS, THREADS>>>(d_A, d_B, d_off, N, shift);
        uint32_t *tmp = d_A; d_A = d_B; d_B = tmp;
    }
    cudaEventRecord(e_gpu_e);

    cudaEventRecord(e_d2h_s);
    cudaMemcpy(host_out, d_A, size_in, cudaMemcpyDeviceToHost);
    cudaEventRecord(e_d2h_e);

    cudaEventRecord(e_total_e);
    cudaEventSynchronize(e_total_e);

    float ms_total, ms_kernel, ms_h2d, ms_d2h;
    cudaEventElapsedTime(&ms_total,  e_total_s, e_total_e);
    cudaEventElapsedTime(&ms_kernel, e_gpu_s,   e_gpu_e);
    cudaEventElapsedTime(&ms_h2d,    e_h2d_s,   e_h2d_e);
    cudaEventElapsedTime(&ms_d2h,    e_d2h_s,   e_d2h_e);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_off);
    cudaEventDestroy(e_total_s); cudaEventDestroy(e_total_e);
    return Timing{ ms_total, ms_kernel, ms_h2d, ms_d2h };
}

int main()
{
    int sizes[] = { 10000, 100000, 1000000 };

    FILE* csv = fopen("results_cuda.csv", "w");
    if (!csv) { fprintf(stderr, "ERROR: cannot open results_cuda.csv\n"); return 1; }
    fprintf(csv, "implementation,N,execution_ms,computation_ms,transfer_ms,"
                 "transfer_bytes,total_ops,performance_mops,sorted_ok\n");

    printf("implementation = cuda (base-256 LSD, uint32, %d passes, %d blocks x %d threads)\n",
           PASSES, BLOCKS, THREADS);
    printf("------------------------------------------------------------\n");

    for (int s = 0; s < 3; s++) {
        int N = sizes[s];
        std::vector<uint32_t> input(N), output(N);
        std::mt19937 rng(12345);                                  // unified seed
        std::uniform_int_distribution<uint32_t> dist;
        for (int i = 0; i < N; i++) input[i] = dist(rng);

        std::vector<uint32_t> ref(input);
        std::sort(ref.begin(), ref.end());

        Timing t = run_radix_cuda(input.data(), output.data(), N, false);
        bool ok = std::is_sorted(output.begin(), output.end()) && (output == ref);

        const double execution_ms   = t.ms_total;
        const double computation_ms = t.ms_kernel;
        const double transfer_ms    = t.ms_h2d + t.ms_d2h;
        const uint64_t transfer_bytes = 2ull * (uint64_t)N * sizeof(uint32_t);

        const int D_gpu = PASSES;
        const int R_gpu = BINS;
        const uint64_t total_ops =
            (uint64_t)D_gpu * (5ull * (uint64_t)N + (uint64_t)R_gpu);
        const double performance_mops =
            execution_ms > 0.0
                ? (double)total_ops / (execution_ms * 1000.0)
                : 0.0;

        printf("N=%d  Execution=%.3f ms  Computation=%.3f ms  Transfer=%.3f ms (H2D=%.3f D2H=%.3f)"
               "  TransferBytes=%llu  Operations=%llu  Performance=%.3f MOPS  Sorted=%s\n",
               N, execution_ms, computation_ms, transfer_ms,
               t.ms_h2d, t.ms_d2h,
               (unsigned long long)transfer_bytes,
               (unsigned long long)total_ops,
               performance_mops,
               ok ? "true" : "false");

        fprintf(csv, "cuda,%d,%.6f,%.6f,%.6f,%llu,%llu,%.6f,%s\n",
                N, execution_ms, computation_ms, transfer_ms,
                (unsigned long long)transfer_bytes,
                (unsigned long long)total_ops,
                performance_mops,
                ok ? "true" : "false");

        if (!ok) {
            fprintf(stderr, "ERROR: output not sorted for N=%d\n", N);
            fclose(csv);
            return 1;
        }
    }

    fclose(csv);
    printf("Wrote results_cuda.csv\n");
    return 0;
}