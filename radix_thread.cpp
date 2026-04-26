// g++   -O3 -std=c++20 -pthread  -o thr    radix_thread.cpp
//
// std::thread + std::barrier (C++20) ашигласан LSD Radix Sort.
// Утсуудыг 1 удаа үүсгээд бүх 4 дамжлагыг тэдгээр дотор гүйцэтгэнэ —
// дамжлага бүрд утас үүсгэх/устгах overhead үгүй.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <array>
#include <thread>
#include <barrier>

constexpr int RADIX  = 256;
constexpr int PASSES = 4;

// ── Параллел LSD Radix Sort ──
// T утас, std::barrier ашиглан 4 дамжлагыг синхрончилно.
static void radix_sort_threaded(std::vector<uint32_t>& data, int T) {
    int N = (int)data.size();
    std::vector<uint32_t> buffer(N);
    uint32_t* in_ptr  = data.data();
    uint32_t* out_ptr = buffer.data();

    // Утас тус бүрийн локал гистограмм ба scatter offset
    // (false sharing бага байх — 1024B/мөр)
    std::vector<std::array<uint32_t, RADIX>> local_hist(T);
    std::vector<std::array<uint32_t, RADIX>> local_offset(T);

    // Үе шат бүрийн хооронд бүх T утсыг зэрэгцүүлэх барьер
    std::barrier sync(T);

    auto worker = [&](int tid) {
        // Энэ утсанд оноосон зэргэлдээ хэсэг (chunk)
        int chunk = N / T;
        int start = tid * chunk;
        int end   = (tid == T - 1) ? N : start + chunk;

        for (int pass = 0; pass < PASSES; ++pass) {
            int shift = pass * 8;

            // 1. Локал гистограмм — өөрийн талбарт write, мөргөлдөөн үгүй
            local_hist[tid].fill(0);
            for (int i = start; i < end; ++i)
                local_hist[tid][(in_ptr[i] >> shift) & 0xFF]++;

            // ──── Барьер #1: бүх локал гистограмм бэлэн ────
            sync.arrive_and_wait();

            // 2. Утас 0 — глобал prefix sum + утас тус бүрийн scatter offset
            // local_offset[t][b] = bin_prefix[b] + Σ_{t' < t} local_hist[t'][b]
            // → стабиль scatter (бункет дотор оролтын дараалал хадгалагдана)
            if (tid == 0) {
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

            // ──── Барьер #2: scatter offset бэлэн, бүх утас уншиж болно ────
            sync.arrive_and_wait();

            // 3. Локал scatter — өөрийн offset-уудыг ашиглан out_ptr руу бичих
            auto& off = local_offset[tid];
            for (int i = start; i < end; ++i) {
                uint32_t v = in_ptr[i];
                int b = (v >> shift) & 0xFF;
                out_ptr[off[b]++] = v;
            }

            // ──── Барьер #3: scatter дууссан, буфер солих аюулгүй ────
            sync.arrive_and_wait();

            // 4. Утас 0 буферүүдийг солино
            if (tid == 0) std::swap(in_ptr, out_ptr);

            // ──── Барьер #4: солилт бүх утсанд харагдсан, дараагийн дамжлага ────
            sync.arrive_and_wait();
        }
    };

    // Утсуудыг 1 удаа үүсгэх — main мөн tid=0 болж оролцоно (нэг утас хэмнэгдэнэ)
    std::vector<std::thread> threads;
    threads.reserve(T - 1);
    for (int t = 1; t < T; ++t) threads.emplace_back(worker, t);
    worker(0);
    for (auto& th : threads) th.join();
    // 4 удаа сольсон тул in_ptr нь буцаад data.data() руу заана — copy-back хэрэггүй
}

static double median_of_4(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return (v[1] + v[2]) / 2.0;
}

int main() {
    const std::vector<int> sizes = {10000, 100000, 1000000};

    // hardware_concurrency дэмжлэггүй системд 4-р fallback
    int T = (int)std::thread::hardware_concurrency();
    if (T == 0) T = 4;

    for (int N : sizes) {
        // Бүх хувилбарт ижил тоонууд (seed = 42)
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> dist;
        std::vector<uint32_t> origin(N);
        for (auto& x : origin) x = dist(rng);

        std::vector<double> times;
        bool ok = true;
        for (int run = 0; run < 5; ++run) {
            std::vector<uint32_t> work = origin;
            auto t0 = std::chrono::high_resolution_clock::now();
            radix_sort_threaded(work, T);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (run > 0) times.push_back(ms);
            if (!std::is_sorted(work.begin(), work.end())) ok = false;
        }
        double t_ms = median_of_4(times);
        double perf = (double)N / (1000.0 * t_ms);

        printf("=== std::thread (Multi-thread, T=%d), N = %d ===\n", T, N);
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
