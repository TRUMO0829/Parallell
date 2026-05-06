// clang++ -std=c++17 -O2 -pthread -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -o 3_openmp 3_openmp.cpp
// ./3_openmp

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr int B      = 256;   // хувин (байт тус бүрд)
static constexpr int PASSES = 4;     // 4 x 8 бит = 32-битийн түлхүүр

// `shift` орон дээр нэг LSD алхам — OpenMP-ээр зэрэгцүүлсэн.
// Дуудагч талаас local_hist-ийг тэглэсэн байх ёстой; offsets-ийг үе 2-д бөглөнө.
static void omp_counting_pass(
    const uint32_t* __restrict__ src,
          uint32_t* __restrict__ dst,
    std::size_t n,
    int         shift,
    int         T,                   // энэ region-ы утасны тоо
    uint32_t*   local_hist,          // T x B
    uint32_t*   offsets)             // T x B
{
    // НЭГ parallel region; 3 үе бүгд ижил T утасны багт ажиллана.
    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        uint32_t* hist = local_hist + tid * B;   // энэ утасны гистограм мөр

        // Үе 1 — Зэрэгцээ гистограм (static schedule).
        // schedule(static) утас бүрд тогтмол дараалсан хэсэг өгнө; ИЖИЛ
        // хэсэг нь Үе 3-д давтагдан scatter давхцалгүй болно.
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i)
            ++hist[(src[i] >> shift) & 0xFF];
        // implicit barrier (omp for-ын төгсгөлд)

        // Үе 2 — Дараалсан prefix-sum (нэг утас).
        // 256 утгыг parallel-аар хийх нь нэг утсаар хийхээс удаан;
        // бусад утсууд omp single-ын дараах barrier дээр хүлээнэ.
        #pragma omp single
        {
            // Утсуудын локал гистограмыг нийлүүлж нийт тоог олно.
            uint32_t total[B] = {};
            for (int t = 0; t < T; ++t)
                for (int b = 0; b < B; ++b)
                    total[b] += local_hist[t * B + b];

            // Exclusive scan -> хувин бүрийн глобал эхлэх индекс.
            uint32_t gstart[B];
            gstart[0] = 0;
            for (int b = 1; b < B; ++b)
                gstart[b] = gstart[b - 1] + total[b - 1];

            // offsets[t][b] = утас t хувин b-д эхэлж бичих байрлал.
            for (int b = 0; b < B; ++b) {
                uint32_t pos = gstart[b];
                for (int t = 0; t < T; ++t) {
                    offsets[t * B + b] = pos;
                    pos += local_hist[t * B + b];
                }
            }
        }
        // implicit barrier — бүх утас одоо offsets[]-ыг харна

        // Үе 3 — Зэрэгцээ scatter, Үе 1-тэй ИЖИЛ static хуваалт.
        // Утас бүр зөвхөн өөрийн тоолсон нүднүүддээ бичнэ -> race үгүй,
        // дараалал тогтвортой.
        uint32_t* off = offsets + tid * B;
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const int bkt = (src[i] >> shift) & 0xFF;
            dst[off[bkt]++] = src[i];
        }
        // implicit barrier
    }
}

// Драйвер: 4 алхам, буферууд ээлжилнэ. Үр дүн `data`-д буцаж ирнэ (4 swap).
void radix_sort_omp(std::vector<uint32_t>& data, int T)
{
    const std::size_t n = data.size();
    if (n < 2) return;

    std::vector<uint32_t> buf(n);
    std::vector<uint32_t> local_hist(T * B);
    std::vector<uint32_t> offsets(T * B);

    uint32_t* src = data.data();
    uint32_t* dst = buf.data();

    for (int pass = 0; pass < PASSES; ++pass) {
        std::fill(local_hist.begin(), local_hist.end(), 0u);   // алхам бүрийн өмнө тэглэх

        omp_counting_pass(src, dst, n, pass * 8, T,
                          local_hist.data(), offsets.data());

        std::swap(src, dst);
    }
}

// Туршилтын өгөгдөл: seed-ээс хамаарсан жигд тархалттай uint32 утгууд.
static std::vector<uint32_t> random_data(std::size_t n, uint32_t seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Зөв эсэхийг шалгах: массив өсөх дарааллаар байгаа эсэх.
static bool is_sorted_check(const std::vector<uint32_t>& v)
{
    for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i] < v[i - 1]) return false;
    return true;
}

// Sequential-тай ижил үйлдлийн загвар (1_sequential.cpp дотор тайлбар бий).
static long long radix_total_ops(std::size_t n) {
    return static_cast<long long>(PASSES) * (4LL * static_cast<long long>(n) + B);
}

// Туршилтын үр дүнгийн бүтэц (бүх хэрэгжүүлэлтэд адил CSV форматтай).
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

// T утсаар `runs` удаа хэмжиж дунд (median) хугацаа буцаана.
static Result benchmark(std::size_t n, int T, int runs = 20)
{
    std::vector<double> times;
    times.reserve(runs);
    bool ok = true;

    // Урьдчилсан халаалт: кэш ба OpenMP утасны pool-ийг бэлтгэнэ.
    {
        auto data = random_data(n, 0u);
        radix_sort_omp(data, T);
    }

    for (int r = 0; r < runs; ++r) {
        auto data = random_data(n, r * 1337u + 7u);   // run бүрд шинэ өгөгдөл
        double t0 = omp_get_wtime();
        radix_sort_omp(data, T);
        double t1 = omp_get_wtime();

        times.push_back((t1 - t0) * 1000.0);
        if (!is_sorted_check(data)) ok = false;
    }

    std::sort(times.begin(), times.end());
    double median_ms = times[runs / 2];

    Result res;
    res.n                = n;
    res.threads          = T;
    res.execution_ms     = median_ms;
    res.computation_ms   = median_ms;       // CPU: host/device хуваагдалгүй
    res.transfer_ms      = 0.0;
    res.transfer_bytes   = 0;
    res.total_ops        = radix_total_ops(n);
    res.performance_mops = static_cast<double>(res.total_ops) / (median_ms * 1000.0);
    res.correct          = ok;
    return res;
}

int main()
{
    // (Хэмжээ x утасны тоо)-гоор давтан туршина.
    const std::vector<std::size_t> sizes   = {10'000, 100'000, 1'000'000, 10'000'000};
    const std::vector<int>         threads = {1, 2, 4, 8, 16};
    const int RUNS = 20;

    // std::cout << "\n  Host logical CPUs : " << omp_get_max_threads() << "\n";
    // std::cout << "\n  Parallel LSD Radix Sort (Base-256, OpenMP, uint32_t)\n";
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
        bool first_row = true;
        for (auto T : threads) {
            auto r = benchmark(n, T, RUNS);
            all_results.push_back(r);

            std::cout << std::setw(12) << (first_row ? std::to_string(r.n) : "")
                      << std::setw(10) << r.threads
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.execution_ms
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.computation_ms
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.transfer_ms
                      << std::setw(14) << std::fixed << std::setprecision(2) << r.performance_mops
                      << std::setw(10) << (r.correct ? "YES" : "NO!")
                      << "\n";
            first_row = false;
        }
        std::cout << "  " << std::string(78, '-') << "\n";
    }

    // std::cout << "\n  (each result is median of " << RUNS << " runs)\n\n";

    // Бүх 4 хэрэгжүүлэлтэд адил CSV форматаар үр дүнг бичнэ.
    const std::string csv_path = "radix_sort_omp_stats.csv";
    std::ofstream csv(csv_path);
    if (!csv) { std::cerr << "Cannot open CSV.\n"; return 1; }

    csv << "implementation,N,threads,execution_ms,computation_ms,transfer_ms,"
           "transfer_bytes,total_ops,performance_mops,sorted_ok\n";
    for (auto& r : all_results)
        csv << "openmp," << r.n << "," << r.threads << ","
            << std::fixed << std::setprecision(4) << r.execution_ms << ","
            << std::fixed << std::setprecision(4) << r.computation_ms << ","
            << std::fixed << std::setprecision(4) << r.transfer_ms << ","
            << r.transfer_bytes << ","
            << r.total_ops << ","
            << std::fixed << std::setprecision(4) << r.performance_mops << ","
            << (r.correct ? "true" : "false") << "\n";
    csv.close();

    // std::cout << "  Stats written to radix_sort_omp_stats.csv\n\n";
    return 0;
}
