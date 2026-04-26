// g++   -O3 -std=c++17           -o seq    radix_seq.cpp
//
// Цуваа LSD Radix Sort — суурь хувилбар (T_seq).
// Бусад параллел хувилбаруудтай харьцуулах хурдны баазлайн (baseline).
// Алгоритм: 8 бит/дамжлага × 4 дамжлага, RADIX = 256.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <array>

constexpr int RADIX  = 256;  // нэг алхам = 8 бит
constexpr int PASSES = 4;    // 32 бит / 8 = 4 дамжлага

// ── Цуваа radix sort ──
// Хоёр буфер (data, buffer)-ийг ээлжлэн солиод 4 дамжлага хийнэ.
// 4 удаа сольсон тул эцэст нь in_ptr нь буцаж data.data() руу заана —
// тиймээс гадагш хуулах шаардлагагүй.
static void radix_sort_seq(std::vector<uint32_t>& data) {
    int N = (int)data.size();
    std::vector<uint32_t> buffer(N);
    uint32_t* in_ptr  = data.data();
    uint32_t* out_ptr = buffer.data();

    for (int pass = 0; pass < PASSES; ++pass) {
        int shift = pass * 8;

        // 1. 256 бункетийн гистограмм тоолох
        std::array<uint32_t, RADIX> hist{};
        for (int i = 0; i < N; ++i)
            hist[(in_ptr[i] >> shift) & 0xFF]++;

        // 2. Эксклюзив prefix sum — бункет тус бүрийн эхлэх индекс
        std::array<uint32_t, RADIX> offset{};
        uint32_t s = 0;
        for (int b = 0; b < RADIX; ++b) {
            offset[b] = s;
            s += hist[b];
        }

        // 3. Тогтвортой scatter (stable) — оролтын дарааллыг хадгална
        for (int i = 0; i < N; ++i) {
            uint32_t v = in_ptr[i];
            int b = (v >> shift) & 0xFF;
            out_ptr[offset[b]++] = v;
        }

        // 4. Дараагийн дамжлагад зориулж буферүүдийг солих
        std::swap(in_ptr, out_ptr);
    }
}

// 5 удаагийн ажиллалтаас warmup-ыг хасч, үлдсэн 4 утгын дунд утга
static double median_of_4(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return (v[1] + v[2]) / 2.0;
}

int main() {
    const std::vector<int> sizes = {10000, 100000, 1000000};

    for (int N : sizes) {
        // ── Тогтмол үрээр өгөгдөл бэлтгэх — бүх хувилбар ижил тоо хүлээн авна ──
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> dist;
        std::vector<uint32_t> origin(N);
        for (auto& x : origin) x = dist(rng);

        // ── 5 удаа ажиллуулна, эхний (warmup) хэмжилтийг хасна ──
        std::vector<double> times;
        bool ok = true;
        for (int run = 0; run < 5; ++run) {
            std::vector<uint32_t> work = origin;
            auto t0 = std::chrono::high_resolution_clock::now();
            radix_sort_seq(work);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (run > 0) times.push_back(ms);
            if (!std::is_sorted(work.begin(), work.end())) ok = false;
        }
        double t_ms = median_of_4(times);
        double perf = (double)N / (1000.0 * t_ms);  // M элемент/секунд

        // ── Үзүүлэлтүүдийг тогтсон форматаар хэвлэх ──
        printf("=== Цуваа (Sequential), N = %d ===\n", N);
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
