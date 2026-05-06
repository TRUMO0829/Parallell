// g++ -O2 -std=c++17 -pthread 2_pthread.cpp -o 2_pthread
// ./2_pthread

#include <pthread.h>

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

static constexpr int B      = 256;   // хувин (байт тус бүрд)
static constexpr int PASSES = 4;     // 4 x 8 бит = 32-битийн түлхүүр

// macOS дээр pthread_barrier_t байхгүй (POSIX-д заавал биш) тул
// mutex + condition variable + "trip" тоологч ашиглан бариер бичсэн.
struct PthreadBarrier {
    pthread_mutex_t mtx{};
    pthread_cond_t  cv{};
    int             arrived = 0;
    int             trip    = 0;     // үе ахих тоологч; release бүрд нэмэгдэнэ
    const int       n;

    explicit PthreadBarrier(int thread_count) : n(thread_count) {
        pthread_mutex_init(&mtx, nullptr);
        pthread_cond_init(&cv, nullptr);
    }

    ~PthreadBarrier() {
        pthread_cond_destroy(&cv);
        pthread_mutex_destroy(&mtx);
    }

    PthreadBarrier(const PthreadBarrier&)            = delete;
    PthreadBarrier& operator=(const PthreadBarrier&) = delete;

    // Бүх `n` утас wait() дуудах хүртэл хүлээнэ.
    void wait() {
        pthread_mutex_lock(&mtx);
        const int my_trip = trip;
        if (++arrived == n) {
            // Сүүлчийн утас: тоологч цэвэрлээд, trip ахиулж бүгдийг сэрээнэ.
            arrived = 0;
            ++trip;
            pthread_cond_broadcast(&cv);
            pthread_mutex_unlock(&mtx);
        } else {
            // trip өөрчлөгдөх хүртэл хүлээнэ.
            while (my_trip == trip)
                pthread_cond_wait(&cv, &mtx);
            pthread_mutex_unlock(&mtx);
        }
    }
};

// Бүх утсанд дамжих ерөнхий контекст.
struct SortCtx {
    uint32_t*    src;          // одоогийн алхамын оролт
    uint32_t*    dst;          // одоогийн алхамын гаралт
    std::size_t  n;
    int          T;            // утасны тоо
    uint32_t*    local_hist;   // T x B матриц (утас тус бүрийн гистограм мөр)
    uint32_t*    offsets;      // T x B матриц (утас бүрийн scatter эхлэл)
    int          pass;
    PthreadBarrier* barrier;
};

struct ThreadArg { SortCtx* ctx; int tid; };

// Утас: өөрийн хэсгийг 4 алхамаар цэгцэлнэ; алхам бүрд 4 удаа барьер.
static void* radix_worker(void* raw)
{
    ThreadArg* ta  = reinterpret_cast<ThreadArg*>(raw);
    SortCtx*   ctx = ta->ctx;
    const int  tid = ta->tid;
    const int  T   = ctx->T;
    const std::size_t n = ctx->n;

    // Утас бүр [lo, hi) дараалсан хэсэгт ажиллана.
    const std::size_t chunk = (n + T - 1) / T;
    const std::size_t lo    = static_cast<std::size_t>(tid) * chunk;
    const std::size_t hi    = std::min(lo + chunk, n);

    for (int pass = 0; pass < PASSES; ++pass) {
        const int shift = pass * 8;

        // Үе 1 — Локал гистограм (зэрэгцээ, давхцалгүй).
        // Утас t зөвхөн өөрийн local_hist[t][..] мөрд бичнэ.
        uint32_t* hist = ctx->local_hist + tid * B;
        std::fill(hist, hist + B, 0u);
        for (std::size_t i = lo; i < hi; ++i)
            ++hist[(ctx->src[i] >> shift) & 0xFF];

        ctx->barrier->wait();

        // Үе 2 — Глобал prefix-sum (зөвхөн утас 0; бусад нь хүлээнэ).
        // Утас бүрийн scatter offset-ыг бэлтгэж давхцлыг арилгана.
        if (tid == 0) {
            // Утсуудын локал гистограмыг нийлүүлж нийт тоог олно.
            uint32_t total[B] = {};
            for (int t = 0; t < T; ++t)
                for (int b = 0; b < B; ++b)
                    total[b] += ctx->local_hist[t * B + b];

            // Exclusive scan -> хувин бүрийн глобал эхлэх индекс.
            uint32_t gstart[B];
            gstart[0] = 0;
            for (int b = 1; b < B; ++b)
                gstart[b] = gstart[b - 1] + total[b - 1];

            // offsets[t][b] = утас t хувин b-д эхэлж бичих байрлал.
            for (int b = 0; b < B; ++b) {
                uint32_t pos = gstart[b];
                for (int t = 0; t < T; ++t) {
                    ctx->offsets[t * B + b] = pos;
                    pos += ctx->local_hist[t * B + b];
                }
            }
        }

        ctx->barrier->wait();

        // Үе 3 — Тараах (зэрэгцээ; утас бүр өөрт оногдсон нүднүүддээ бичнэ).
        uint32_t* off = ctx->offsets + tid * B;
        for (std::size_t i = lo; i < hi; ++i) {
            const int bkt = (ctx->src[i] >> shift) & 0xFF;
            ctx->dst[off[bkt]++] = ctx->src[i];
        }

        ctx->barrier->wait();

        // Дараагийн алхамын өмнө буферуудыг солино (зөвхөн утас 0).
        if (tid == 0) std::swap(ctx->src, ctx->dst);

        ctx->barrier->wait();
    }
    return nullptr;
}

// Үндсэн орц: буфер хуваарилж, T утас үүсгэж, дуусахыг хүлээнэ.
void parallel_radix_sort(std::vector<uint32_t>& data, int T)
{
    const std::size_t n = data.size();
    if (n < 2) return;
    T = std::max(1, T);

    std::vector<uint32_t> buf(n);                  // ээлжилж ажиллах буфер
    std::vector<uint32_t> local_hist(T * B, 0);    // утас тус бүрийн гистограм
    std::vector<uint32_t> offsets(T * B, 0);

    SortCtx ctx;
    ctx.src        = data.data();
    ctx.dst        = buf.data();
    ctx.n          = n;
    ctx.T          = T;
    ctx.local_hist = local_hist.data();
    ctx.offsets    = offsets.data();
    ctx.pass       = 0;
    PthreadBarrier barrier(T);
    ctx.barrier    = &barrier;

    std::vector<pthread_t>  threads(T);
    std::vector<ThreadArg>  args(T);

    for (int t = 0; t < T; ++t) {
        args[t] = {&ctx, t};
        pthread_create(&threads[t], nullptr, radix_worker, &args[t]);
    }
    for (int t = 0; t < T; ++t)
        pthread_join(threads[t], nullptr);
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

    // Урьдчилсан халаалт: кэш ба утас үүсгэх зардлыг хальсална.
    {
        auto data = random_data(n, 0u);
        parallel_radix_sort(data, T);
    }

    for (int r = 0; r < runs; ++r) {
        auto data = random_data(n, r * 1337u + 7u);   // run бүрд шинэ өгөгдөл
        auto t0   = std::chrono::high_resolution_clock::now();
        parallel_radix_sort(data, T);
        auto t1   = std::chrono::high_resolution_clock::now();

        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
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
    const int RUNS = 40;

    // std::cout << "\n  Parallel LSD Radix Sort (Base-256, pthreads, uint32_t)\n";
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
        bool first = true;
        for (auto T : threads) {
            auto r = benchmark(n, T, RUNS);
            all_results.push_back(r);

            std::cout << std::setw(12) << (first ? std::to_string(r.n) : "")
                      << std::setw(10) << r.threads
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.execution_ms
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.computation_ms
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.transfer_ms
                      << std::setw(14) << std::fixed << std::setprecision(2) << r.performance_mops
                      << std::setw(10) << (r.correct ? "YES" : "NO!")
                      << "\n";
            first = false;
        }
        std::cout << "  " << std::string(78, '-') << "\n";
    }

    // Бүх 4 хэрэгжүүлэлтэд адил CSV форматаар үр дүнг бичнэ.
    const std::string csv_path = "radix_sort_pthread_stats.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Could not open " << csv_path << "\n";
        return 1;
    }
    csv << "implementation,N,threads,execution_ms,computation_ms,transfer_ms,"
           "transfer_bytes,total_ops,performance_mops,sorted_ok\n";
    for (auto& r : all_results)
        csv << "pthread," << r.n << "," << r.threads << ","
            << std::fixed << std::setprecision(4) << r.execution_ms << ","
            << std::fixed << std::setprecision(4) << r.computation_ms << ","
            << std::fixed << std::setprecision(4) << r.transfer_ms << ","
            << r.transfer_bytes << ","
            << r.total_ops << ","
            << std::fixed << std::setprecision(4) << r.performance_mops << ","
            << (r.correct ? "true" : "false") << "\n";
    csv.close();

    return 0;
}
