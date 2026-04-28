/*
 * ============================================================
 *  Radix Sort - Параллел харьцуулалт
 *  1) Цуваа (sequential)
 *  2) std::thread ашигласан параллел (pthreads - C хувилбар)
 *  3) OpenMP ашигласан параллел
 *
 *  Хэмжих хэмжээ: 10,000 / 100,000 / 1,000,000 элемент
 * ============================================================
 *  Хөрвүүлэх:
 *    gcc -O2 -fopenmp -pthread -o radix_sort radix_sort_parallel.c
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>

/* ──────────────────────────────────────────
   Тохиргоо
   ────────────────────────────────────────── */
#define BITS          32          /* тоонуудын бит өргөн                */
#define RADIX         256         /* 8 бит нэг алхам → 256 бункет       */
#define PASSES        4           /* 4 дамжлага × 8 бит = 32 бит        */
#define MAX_THREADS   8           /* pthread дэд утгын тоо               */

/* ──────────────────────────────────────────
   Туслах функцүүд
   ────────────────────────────────────────── */

/* Санамсаргүй тоо дүүргэх */
void fill_random(unsigned int *arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = (unsigned int)rand();
}

/* Масиваас хуулах */
void copy_array(unsigned int *dst, const unsigned int *src, int n) {
    memcpy(dst, src, n * sizeof(unsigned int));
}

/* Зөв эрэмбэлсэн эсэхийг шалгах */
int is_sorted(unsigned int *arr, int n) {
    for (int i = 1; i < n; i++)
        if (arr[i] < arr[i-1]) return 0;
    return 1;
}

/* Цаг хэмжих – секундээр */
double get_time_sec(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) +
           (end->tv_nsec - start->tv_nsec) * 1e-9;
}

/* ══════════════════════════════════════════
   3. OPENMP RADIX SORT
      #pragma omp parallel ашиглана
   ══════════════════════════════════════════ */
void radix_sort_openmp(unsigned int *arr, int n, int num_threads) {
    unsigned int *tmp = (unsigned int *)malloc(n * sizeof(unsigned int));

    omp_set_num_threads(num_threads);

    for (int pass = 0; pass < PASSES; pass++) {
        int shift = pass * 8;

        int actual_threads;
        unsigned int (*local_counts)[RADIX] = NULL;

        /* ── Зэрэгцээ тоолох ── */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            #pragma omp single
            {
                actual_threads = omp_get_num_threads();
                local_counts = (unsigned int (*)[RADIX])
                    calloc(actual_threads * RADIX, sizeof(unsigned int));
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++)
                local_counts[tid][(arr[i] >> shift) & 0xFF]++;
        }

        /* ── Нэгтгэх + prefix sum ── */
        unsigned int global_count[RADIX] = {0};
        for (int t = 0; t < actual_threads; t++)
            for (int b = 0; b < RADIX; b++)
                global_count[b] += local_counts[t][b];

        free(local_counts);

        unsigned int prefix[RADIX];
        prefix[0] = 0;
        for (int i = 1; i < RADIX; i++)
            prefix[i] = prefix[i-1] + global_count[i-1];

        /* ── Scatter ── */
        for (int i = 0; i < n; i++) {
            int b = (arr[i] >> shift) & 0xFF;
            tmp[prefix[b]++] = arr[i];
        }
        memcpy(arr, tmp, n * sizeof(unsigned int));
    }
    free(tmp);
}

/* ══════════════════════════════════════════
   BENCHMARK ФУНКЦ
   ══════════════════════════════════════════ */
void benchmark(int n) {
    printf("\n┌─────────────────────────────────────────────┐\n");
    printf("│  N = %7d элемент                         │\n", n);
    printf("├──────────────────────┬──────────┬────────────┤\n");
    printf("│ Арга                 │ Хугацаа  │ Үр дүн     │\n");
    printf("├──────────────────────┼──────────┼────────────┤\n");

    unsigned int *src  = (unsigned int *)malloc(n * sizeof(unsigned int));
    unsigned int *work = (unsigned int *)malloc(n * sizeof(unsigned int));
    struct timespec t0, t1;

    fill_random(src, n);

    /* ── 1. Цуваа ── */
    copy_array(work, src, n);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    radix_sort_sequential(work, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double seq_time = get_time_sec(&t0, &t1);
    printf("│ Цуваа                │ %7.4f с │ %-10s │\n",
           seq_time, is_sorted(work, n) ? "✓ зөв" : "✗ алдаа");

    /* ── 2. Pthread (2, 4, 8 утас) ── */
    int thread_counts[] = {2, 4, 8};
    char label[64];
    for (int ti = 0; ti < 3; ti++) {
        int t = thread_counts[ti];
        copy_array(work, src, n);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        radix_sort_pthread(work, n, t);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double pt = get_time_sec(&t0, &t1);
        snprintf(label, sizeof(label), "pthread (%d утас)", t);
        printf("│ %-20s │ %7.4f с │ %-10s │\n",
               label, pt, is_sorted(work, n) ? "✓ зөв" : "✗ алдаа");
        printf("│   → хурдасгалт      │  x%5.2f   │            │\n",
               seq_time / pt);
    }

    /* ── 3. OpenMP (2, 4, 8 утас) ── */
    for (int ti = 0; ti < 3; ti++) {
        int t = thread_counts[ti];
        copy_array(work, src, n);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        radix_sort_openmp(work, n, t);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double pt = get_time_sec(&t0, &t1);
        snprintf(label, sizeof(label), "OpenMP  (%d утас)", t);
        printf("│ %-20s │ %7.4f с │ %-10s │\n",
               label, pt, is_sorted(work, n) ? "✓ зөв" : "✗ алдаа");
        printf("│   → хурдасгалт      │  x%5.2f   │            │\n",
               seq_time / pt);
    }

    printf("└──────────────────────┴──────────┴────────────┘\n");

    free(src);
    free(work);
}

/* ══════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════ */
int main(void) {
    srand(42);

    printf("╔═════════════════════════════════════════════╗\n");
    printf("║   Radix Sort — Параллел харьцуулалт (C)     ║\n");
    printf("║   pthread  vs  OpenMP  vs  Цуваа            ║\n");
    printf("╚═════════════════════════════════════════════╝\n");

    int sizes[] = {10000, 100000, 1000000};
    for (int i = 0; i < 3; i++)
        benchmark(sizes[i]);

    printf("\nДуусгавар.\n");
    return 0;
}
