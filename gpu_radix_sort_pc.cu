// nvcc -O3 -std=c++17 -arch=native -o gpu_radix_sort gpu_radix_sort_pc.cu
// ./gpu_radix_sort
//
// CUDA дээрх LSD radix sort: uint32 түлхүүрийг 4-битийн оронгоор 8 удаа цэгцэлнэ
// (RADIX = 16 хувин). Tile-ийн хэмжээ 2048 — CUDA блок бүр өөрийн 2048 элементийн
// tile-ийг боловсруулна, олон блок зэрэг ажиллана. Алхам бүрд 3 kernel:
// блокийн гистограм -> глобал scan -> тогтвортой scatter.
// 8 алхам (тэгш) дууссаны дараа үр дүн d_buf_a-д буцаж ирнэ.

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

// CUDA API дуудлагыг боож, алдаа гарвал тодорхой мэдээлэлтэйгээр зогсооно.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s  at %s:%d\n",                       \
                    cudaGetErrorString(_err), __FILE__, __LINE__);              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// 4-битийн орон (16 хувин) — блокийн гистограм shared memory-д таарна.
// 2048 элемент/tile нь desktop RTX картанд тохиромжтой хэмжээ.
#define BITS_PER_PASS   4
#define RADIX           (1 << BITS_PER_PASS)    // 16
#define NUM_PASSES      (32 / BITS_PER_PASS)    // 8 алхам (32-бит)
#define BLOCK_SIZE      256                     // CUDA блок дахь утасны тоо
#define TILE_SIZE       2048                    // блокт ноогдох элементийн тоо

// Kernel 1 — Блокийн гистограм.
// Блок бүр өөрийн tile-ийн 16 хувинт гистограмыг shared memory-д atomicAdd-аар
// бүтээж, дараа нь global memory-д column-major форматаар бичнэ:
// block_hist[bin * num_blocks + bid]. Column-major байх нь Kernel 2-ийн
// глобал scan дараалсан санах ой уншихад хэрэгтэй.
__global__ void k_block_histogram(const uint32_t* __restrict__ in,
                                         uint32_t* __restrict__ block_hist,
                                   int n, int shift, int num_blocks)
{
    __shared__ uint32_t lhist[RADIX];   // 16 тоологч (shared memory-д)

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Shared гистограмыг тэглэнэ.
    if (tid < RADIX) lhist[tid] = 0;
    __syncthreads();

    // Бүх 256 утас энэ блокийн 2048 элементийн tile-аар алхаж,
    // shared memory-д atomicAdd хийнэ (хямд atomics, глобал contention байхгүй).
    int base = bid * TILE_SIZE;
    for (int i = tid; i < TILE_SIZE; i += BLOCK_SIZE) {
        int gi = base + i;
        if (gi < n) {
            uint32_t bin = (in[gi] >> shift) & (RADIX - 1);
            atomicAdd(&lhist[bin], 1u);
        }
    }
    __syncthreads();

    // Утас 0..15 тус бүр нэг хувины тоог global memory-д бичнэ.
    if (tid < RADIX)
        block_hist[tid * num_blocks + bid] = lhist[tid];
}

// Kernel 2 — Глобал exclusive prefix-scan (нэг утас).
// (bin x block) тооны хүснэгтийг бичих offset болгож хувиргана: hist[i] нь
// "i-р (bin, block) хосын dst-д эхлэх байрлал" болно.
// N=10M үед хүснэгт нийт 16 x 4883 ~ 78K тул нэг утсан scan нь parallel
// reduction launch хийхээс хурдан.
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

// Kernel 3 — Тогтвортой scatter.
// Алхам A: блок бүрийн утас 0 өөрийн 2048 элементийг дараалуулан гүйж
//          элемент тус бүрд "хувин доторх дугаар" (rank) ононо
//          (оролтын дараалал тогтвортой -> LSD-д заавал).
// Алхам B: 256 утас бүгд dst[s_goff[bin] + s_rank[i]] руу зэрэг бичнэ.
__global__ void k_scatter(const uint32_t* __restrict__ in,
                                uint32_t* __restrict__ out,
                          const uint32_t* __restrict__ prefix,
                          int n, int shift, int num_blocks)
{
    __shared__ uint32_t s_goff[RADIX];      // хувин бүрийн глобал offset
    __shared__ uint32_t s_rank[TILE_SIZE];  // элемент бүрийн хувин доторх дугаар

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Энэ блокийн 16 хувин тус бүрийн глобал offset-ыг prefix-ээс уншина.
    if (tid < RADIX)
        s_goff[tid] = prefix[tid * num_blocks + bid];
    __syncthreads();

    // Алхам A — Дараалсан rank оноох (тогтвортой).
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

    // Алхам B — Глобал санах ой руу зэрэг бичих.
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

// Драйвер: 8 алхам, buf_a / buf_b ээлжилнэ. Цэгцлэгдсэн үр дүнг
// хадгалсан буферийн заагчийг буцаана (тэгш алхамын дараа d_buf_a).
static uint32_t* run_sort(uint32_t* d_buf_a, uint32_t* d_buf_b,
                           uint32_t* d_hist,
                           int n, int num_blocks, int hist_size)
{
    uint32_t* d_in  = d_buf_a;
    uint32_t* d_out = d_buf_b;

    for (int pass = 0; pass < NUM_PASSES; pass++) {
        int shift = pass * BITS_PER_PASS;

        // Алхам бүрийн өмнө гистограмын хүснэгтийг тэглэнэ.
        CUDA_CHECK(cudaMemsetAsync(d_hist, 0,
                   (size_t)hist_size * sizeof(uint32_t)));

        // Алхам бүрд 3 kernel; kernel-ийн зааг нь глобал барьер болно.
        k_block_histogram<<<num_blocks, BLOCK_SIZE>>>(
            d_in, d_hist, n, shift, num_blocks);

        k_global_scan<<<1, 1>>>(d_hist, hist_size);

        k_scatter<<<num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, d_hist, n, shift, num_blocks);

        std::swap(d_in, d_out);
    }

    return d_in;
}

// Run тус бүрийн хугацаа: нийт exec, цэвэр compute (3 kernel), PCIe transfer.
struct BenchResult {
    double exec_ms;
    double compute_ms;
    double transfer_ms;
    bool   sorted_ok;
};

// Туршилт: device буфер хуваарилж, `runs` удаа (H2D + sort + D2H) мөчлөгийг
// CUDA event ашиглан мс-ийн нарийвчлалаар хэмжинэ. 1 удаа warmup ажиллуулж
// JIT компайл болон allocator-ыг бэлдэнэ.
static BenchResult benchmark(const std::vector<uint32_t>& h_in,
                              int n, int runs)
{
    int num_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    int hist_size  = RADIX * num_blocks;

    // Device хуваарилалт: 2 түлхүүр буфер (ping-pong) + гистограмын хүснэгт.
    uint32_t *d_a = nullptr, *d_b = nullptr, *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,    (size_t)n         * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_b,    (size_t)n         * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hist, (size_t)hist_size * sizeof(uint32_t)));

    // Дэд үе тус бүрийн нарийн хугацаа хэмжих CUDA event-үүд.
    cudaEvent_t e0, e1, ec0, ec1, eh0, eh1, ed0, ed1;
    CUDA_CHECK(cudaEventCreate(&e0));   CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&ec0));  CUDA_CHECK(cudaEventCreate(&ec1));
    CUDA_CHECK(cudaEventCreate(&eh0));  CUDA_CHECK(cudaEventCreate(&eh1));
    CUDA_CHECK(cudaEventCreate(&ed0));  CUDA_CHECK(cudaEventCreate(&ed1));

    // Урьдчилсан халаалт (хугацаа хэмжихгүй): JIT компайл, allocator бэлтгэл.
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
        // Нийт wall-clock хугацаа.
        CUDA_CHECK(cudaEventRecord(e0));

        // H2D дамжуулалт (host -> device).
        CUDA_CHECK(cudaEventRecord(eh0));
        CUDA_CHECK(cudaMemcpy(d_a, h_in.data(),
                   (size_t)n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(eh1));

        // Цэвэр GPU compute (8 алхам x 3 kernel).
        CUDA_CHECK(cudaEventRecord(ec0));
        uint32_t* d_res = run_sort(d_a, d_b, d_hist, n, num_blocks, hist_size);
        CUDA_CHECK(cudaEventRecord(ec1));

        // D2H дамжуулалт (device -> host).
        CUDA_CHECK(cudaEventRecord(ed0));
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_res,
                   (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ed1));

        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));   // бүх GPU ажил дуустал хүлээх

        // Энэ run-ийн өнгөрсөн хугацааг уншина.
        float exec_ms=0, comp_ms=0, h2d_ms=0, d2h_ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&exec_ms, e0,  e1));
        CUDA_CHECK(cudaEventElapsedTime(&comp_ms, ec0, ec1));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  eh0, eh1));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  ed0, ed1));

        v_exec[r]     = exec_ms;
        v_compute[r]  = comp_ms;
        v_transfer[r] = h2d_ms + d2h_ms;

        // Нэг л удаа зөв эсэхийг шалгана (хямд бөгөөд run бүрд давтахгүй).
        if (r == 0)
            sorted_ok = std::is_sorted(h_out.begin(), h_out.end());
    }

    // Median нь outlier-аас (clock jitter гэх мэт) хамгаалдаг.
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

    // Device санах ой ба event-үүдийг чөлөөлнө.
    CUDA_CHECK(cudaFree(d_a));  CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaEventDestroy(e0));   CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(ec0));  CUDA_CHECK(cudaEventDestroy(ec1));
    CUDA_CHECK(cudaEventDestroy(eh0));  CUDA_CHECK(cudaEventDestroy(eh1));
    CUDA_CHECK(cudaEventDestroy(ed0));  CUDA_CHECK(cudaEventDestroy(ed1));

    return res;
}

// Аль GPU ашиглаж байгаа болон онолын дээд санах ойн bandwidth-ыг хэвлэнэ.
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

    // CLI: N, seed, runs (default: бүрэн sweep, 42, 20 run).
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

    // CSV header stdout руу (хэрэгтэй бол файл руу redirect хий).
    printf("implementation,N,median_exec_ms,median_compute_ms,"
           "median_transfer_ms,transfer_bytes,performance_mops,sorted_ok\n");

    for (int n : sizes) {
        // N тус бүрд нэг л санамсаргүй өгөгдөл (бүх run-д дахин ашиглана).
        std::vector<uint32_t> h_in(n);
        {
            std::mt19937 rng(seed);
            for (int i = 0; i < n; i++) h_in[i] = rng();
        }

        BenchResult r = benchmark(h_in, n, runs);

        // PCIe traffic = N * 4B H2D + N * 4B D2H = 2 * N * sizeof(uint32_t).
        long long transfer_bytes = 2LL * n * (long long)sizeof(uint32_t);

        // Үйлдлийн загвар: ~12 ops/elem (histogram + scatter) x 8 алхам +
        // глобал scan-ийн жижиг overhead. MOPS-аар тайлагнана.
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
