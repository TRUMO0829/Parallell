// g++   -O3 -std=c++17 -fopenmp  -o omp    radix_omp.cpp
//
// OpenMP ашигласан LSD Radix Sort.
// Дамжлага бүрд нэг #pragma omp parallel бүс — гистограмм / prefix / scatter.
// std::thread хувилбартай яг адил алгоритм, side-by-side харьцуулахад тохиромжтой.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <array>
#include <omp.h>

constexpr int RADIX  = 256;
constexpr int PASSES = 4;

// ── OpenMP параллел LSD Radix Sort ──
static void radix_sort_omp(std::vector<uint32_t>& data, int T) {
    int N = (int)data.size();
    std::vector<uint32_t> buffer(N);
    uint32_t* in_ptr  = data.data();
    uint32_t* out_ptr = buffer.data();

    // Утас тус бүрийн локал гистограмм + scatter offset
    // Тэмдэглэл: reduction(+:hist[:256]) ашиглавал глобал гистограмм гарна,
    // харин scatter-д утас бүрийн тус тусын тоологч хэрэгтэй (стабиль scatter)
    // тул тэр тоологчийг анх scatter-д бэлтгэх зорилготой шууд локал-аар тооцно.
    std::vector<std::array<uint32_t, RADIX>> local_hist(T);
    std::vector<std::array<uint32_t, RADIX>> local_offset(T);

    for (int pass = 0; pass < PASSES; ++pass) {
        int shift = pass * 8;

        #pragma omp parallel num_threads(T)
        {
            int tid = omp_get_thread_num();

            // 1. Локал гистограмм — өөрийн талбарт write
            local_hist[tid].fill(0);
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i)
                local_hist[tid][(in_ptr[i] >> shift) & 0xFF]++;
            // omp for-ын дараа автомат барьер

            // 2. Нэг утсаар глобал prefix sum + утас тус бүрийн scatter offset
            #pragma omp single
            {
                std::array<uint32_t, RADIX> total{};
                for (int t = 0; t < T; ++t)
                    for (int b = 0; b < RADIX; ++b)
                        total[b] += local_hist[t][b];

                std::array<uint32_t, RADIX> bin_prefix{};
                uint32_t s = 0;
                for (int b = 0; b < RADIX; ++b) {
                    bin_prefix[b] = s;
                    s += total[b];
                }
                std::array<uint32_t, RADIX> running = bin_prefix;
                for (int t = 0; t < T; ++t) {
                    for (int b = 0; b < RADIX; ++b) {
                        local_offset[t][b] = running[b];
                        running[b] += local_hist[t][b];
                    }
                }
            }
            // omp single-ийн дараа автомат барьер

            // 3. Локал scatter — schedule(static) Phase 1-тэй ижил хуваарьтай
            // тул утас бүр Phase 1 дэх ижил элементүүдийг ижил дарааллаар уншина
            // → стабиль scatter (бункет дотор оролтын дараалал хадгалагдана)
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                uint32_t v = in_ptr[i];
                int b = (v >> shift) & 0xFF;
                out_ptr[local_offset[tid][b]++] = v;
            }
        }

        std::swap(in_ptr, out_ptr);
    }
    // 4 удаа сольсон тул in_ptr нь буцаж data.data() руу заана
}

static double median_of_4(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return (v[1] + v[2]) / 2.0;
}

int main() {
    const std::vector<int> sizes = {10000, 100000, 1000000};

    // Утасны тоог тогтворжуулах — local_hist хүртээмжтэй байх баталгаа
    int T = omp_get_max_threads();
    if (T <= 0) T = 4;

    for (int N : sizes) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> dist;
        std::vector<uint32_t> origin(N);
        for (auto& x : origin) x = dist(rng);

        std::vector<double> times;
        bool ok = true;
        for (int run = 0; run < 5; ++run) {
            std::vector<uint32_t> work = origin;
            auto t0 = std::chrono::high_resolution_clock::now();
            radix_sort_omp(work, T);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (run > 0) times.push_back(ms);
            if (!std::is_sorted(work.begin(), work.end())) ok = false;
        }
        double t_ms = median_of_4(times);
        double perf = (double)N / (1000.0 * t_ms);

        printf("=== OpenMP (T=%d), N = %d ===\n", T, N);
        printf("Гүйцэтгэлийн хугацаа (Execution time): %.3f мс\n", t_ms);
        printf("Тооцооллын хугацаа (Computation):       %.3f мс\n", t_ms);
        printf("Дамжуулалтын хугацаа (Data transfer):   %.3f мс\n", 0.0);
        printf("Нийт үйлдэл (Total operations):         %lld\n", 20LL * N);
        printf("Дамжуулсан өгөгдөл (Bytes transferred): %d\n", 0);
        printf("Гүйцэтгэл (Achievable performance):     %.2f M elem/s\n", perf);
        printf("Шалгалт: %s\n\n", ok ? "Зөв" : "Буруу");
    }
    return 0;
}
