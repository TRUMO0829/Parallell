// nvcc  -O3 -std=c++17           -o cuda   radix_cuda.cu
//
// CUDA дээр LSD Radix Sort — 3 кернель × 4 дамжлага.
//   1) histogram_kernel: блок тус бүрд локал гистограмм (shared mem + atomicAdd)
//   2) scan_kernel:      (блок × бункет) глобал scatter offset тооцоо
//   3) scatter_kernel:   тогтвортой scatter — блок дотор thread 0 цуваа бичинэ
//
// Дизайн тэмдэглэл: техник даалгаварт «per-thread atomicAdd shared prefix дээр»
// гэсэн боловч тэр аргачлал блок дотор стабильность алдаж, LSD радиксыг
// буруу болгодог (N=1M үед is_sorted эвдэрнэ). Тиймээс блок тус бүрд
// thread 0-р цуваа scatter хийж бүтэн стабиль байдлыг хадгалсан —
// «хамгийн энгийн зөв» сонголт.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

constexpr int RADIX  = 256;
constexpr int PASSES = 4;
constexpr int BLOCK  = 256;   // 256 thread/block, 256 элемент/block

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA алдаа: %s\n", cudaGetErrorString(e_)); exit(1); } \
    } while(0)

// ── Кернель 1: блок тус бүрийн 256-бункетийн гистограмм ──
// Shared memory дээр atomicAdd → бүхэл блокын дотор зэрэгцээ тоолно.
__global__ void histogram_kernel(const uint32_t* __restrict__ in,
                                  uint32_t* __restrict__ block_hist,
                                  int N, int shift) {
    __shared__ unsigned int s_hist[RADIX];
    int tid = threadIdx.x;
    s_hist[tid] = 0;
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < N) {
        unsigned int bin = (in[gid] >> shift) & 0xFFu;
        atomicAdd(&s_hist[bin], 1u);
    }
    __syncthreads();

    // 256 thread × 1 элемент → бүх 256 бункетийн утгыг глобал руу гаргана
    block_hist[blockIdx.x * RADIX + tid] = s_hist[tid];
}

// ── Кернель 2: (блок × бункет) scatter offset тооцоо ──
// offsets[block * RADIX + bin] = bin_prefix[bin] + Σ_{prev блокуудын bin тоо}
// 1 блок, 256 thread — thread t нь bin t-г хариуцна.
__global__ void scan_kernel(const uint32_t* __restrict__ block_hist,
                             uint32_t* __restrict__ offsets, int num_blocks) {
    int bin = threadIdx.x;
    __shared__ uint32_t bin_total[RADIX];
    __shared__ uint32_t bin_prefix[RADIX];

    // Алхам 1: тухайн bin-ийн багана дээр блокоор exclusive prefix sum
    uint32_t sum = 0;
    for (int b = 0; b < num_blocks; ++b) {
        offsets[b * RADIX + bin] = sum;
        sum += block_hist[b * RADIX + bin];
    }
    bin_total[bin] = sum;  // тухайн bin-ийн нийт тоо
    __syncthreads();

    // Алхам 2: нэг утсаар bin_total дээгүүр exclusive prefix sum
    if (bin == 0) {
        uint32_t s = 0;
        for (int i = 0; i < RADIX; ++i) {
            bin_prefix[i] = s;
            s += bin_total[i];
        }
    }
    __syncthreads();

    // Алхам 3: bin_prefix-ийг тухайн bin-ийн бүх блокт нэмэх
    uint32_t bp = bin_prefix[bin];
    for (int b = 0; b < num_blocks; ++b) {
        offsets[b * RADIX + bin] += bp;
    }
}

// ── Кернель 3: тогтвортой scatter ──
// Блок тус бүрд thread 0 цуваа бичинэ → бункет дотор оролтын дараалал хадгалагдана.
__global__ void scatter_kernel(const uint32_t* __restrict__ in,
                                uint32_t* __restrict__ out,
                                const uint32_t* __restrict__ offsets,
                                int N, int shift) {
    __shared__ uint32_t s_off[RADIX];
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;

    // Энэ блокт зориулсан 256 offset-ийг shared memory руу ачаалах
    s_off[tid] = offsets[blockIdx.x * RADIX + tid];
    __syncthreads();

    if (tid == 0) {
        int block_end = min(block_start + (int)blockDim.x, N);
        for (int i = block_start; i < block_end; ++i) {
            uint32_t v = in[i];
            int bin = (v >> shift) & 0xFF;
            out[s_off[bin]++] = v;
        }
    }
}

static double median_of_4(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return (v[1] + v[2]) / 2.0;
}

int main() {
    const std::vector<int> sizes = {10000, 100000, 1000000};

    for (int N : sizes) {
        // Бүх хувилбарт ижил тоонууд (seed = 42)
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> dist;
        std::vector<uint32_t> origin(N);
        for (auto& x : origin) x = dist(rng);

        int num_blocks = (N + BLOCK - 1) / BLOCK;

        // ── GPU дээр буфер хуваарилах (5 удаагийн ажиллалтад 1 удаа) ──
        uint32_t *d_in, *d_out, *d_hist, *d_off;
        CUDA_CHECK(cudaMalloc(&d_in,   N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_out,  N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_hist, (size_t)num_blocks * RADIX * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_off,  (size_t)num_blocks * RADIX * sizeof(uint32_t)));

        std::vector<double> total_times, gpu_times, mv_times;
        bool ok = true;

        for (int run = 0; run < 5; ++run) {
            // SAXPY жишээтэй адил 4 хос cudaEvent: total / gpu / move1 / move2
            cudaEvent_t e_total_s, e_total_e, e_gpu_s, e_gpu_e;
            cudaEvent_t e_mv1_s,   e_mv1_e,   e_mv2_s, e_mv2_e;
            cudaEventCreate(&e_total_s); cudaEventCreate(&e_total_e);
            cudaEventCreate(&e_gpu_s);   cudaEventCreate(&e_gpu_e);
            cudaEventCreate(&e_mv1_s);   cudaEventCreate(&e_mv1_e);
            cudaEventCreate(&e_mv2_s);   cudaEventCreate(&e_mv2_e);

            cudaEventRecord(e_total_s);

            // ── H2D дамжуулалт ──
            cudaEventRecord(e_mv1_s);
            CUDA_CHECK(cudaMemcpy(d_in, origin.data(),
                                  N * sizeof(uint32_t), cudaMemcpyHostToDevice));
            cudaEventRecord(e_mv1_e);

            // ── 4 дамжлагын кернель (буферийн ping-pong) ──
            cudaEventRecord(e_gpu_s);
            uint32_t *cur_in = d_in, *cur_out = d_out;
            for (int pass = 0; pass < PASSES; ++pass) {
                int shift = pass * 8;
                histogram_kernel<<<num_blocks, BLOCK>>>(cur_in, d_hist, N, shift);
                scan_kernel<<<1, RADIX>>>(d_hist, d_off, num_blocks);
                scatter_kernel<<<num_blocks, BLOCK>>>(cur_in, cur_out, d_off, N, shift);
                std::swap(cur_in, cur_out);
            }
            cudaEventRecord(e_gpu_e);
            // 4 swap → cur_in нь d_in (анхны байрлал) руу буцаж заана

            // ── D2H дамжуулалт ──
            std::vector<uint32_t> result(N);
            cudaEventRecord(e_mv2_s);
            CUDA_CHECK(cudaMemcpy(result.data(), cur_in,
                                  N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            cudaEventRecord(e_mv2_e);

            cudaEventRecord(e_total_e);
            cudaEventSynchronize(e_total_e);

            float t_total, t_gpu, t_mv1, t_mv2;
            cudaEventElapsedTime(&t_total, e_total_s, e_total_e);
            cudaEventElapsedTime(&t_gpu,   e_gpu_s,   e_gpu_e);
            cudaEventElapsedTime(&t_mv1,   e_mv1_s,   e_mv1_e);
            cudaEventElapsedTime(&t_mv2,   e_mv2_s,   e_mv2_e);

            if (run > 0) {
                total_times.push_back(t_total);
                gpu_times.push_back(t_gpu);
                mv_times.push_back(t_mv1 + t_mv2);
            }
            if (!std::is_sorted(result.begin(), result.end())) ok = false;

            cudaEventDestroy(e_total_s); cudaEventDestroy(e_total_e);
            cudaEventDestroy(e_gpu_s);   cudaEventDestroy(e_gpu_e);
            cudaEventDestroy(e_mv1_s);   cudaEventDestroy(e_mv1_e);
            cudaEventDestroy(e_mv2_s);   cudaEventDestroy(e_mv2_e);
        }

        cudaFree(d_in); cudaFree(d_out); cudaFree(d_hist); cudaFree(d_off);

        double t_ms  = median_of_4(total_times);
        double g_ms  = median_of_4(gpu_times);
        double mv_ms = median_of_4(mv_times);
        double perf  = (double)N / (1000.0 * t_ms);
        long long bytes = 2LL * N * (long long)sizeof(uint32_t);  // 1 H2D + 1 D2H

        printf("=== CUDA, N = %d ===\n", N);
        printf("Гүйцэтгэлийн хугацаа (Execution time): %.3f мс\n", t_ms);
        printf("Тооцооллын хугацаа (Computation):       %.3f мс\n", g_ms);
        printf("Дамжуулалтын хугацаа (Data transfer):   %.3f мс\n", mv_ms);
        printf("Нийт үйлдэл (Total operations):         %lld\n", 20LL * N);
        printf("Дамжуулсан өгөгдөл (Bytes transferred): %lld\n", bytes);
        printf("Гүйцэтгэл (Achievable performance):     %.2f M elem/s\n", perf);
        printf("Шалгалт: %s\n\n", ok ? "Зөв" : "Буруу");
    }
    return 0;
}
