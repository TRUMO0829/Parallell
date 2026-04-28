
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

typedef struct {
    unsigned int *arr;
    int           start;
    int           end;
    int           shift;
    unsigned int  local_count[RADIX];
} ThreadArg;

static void *count_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    memset(a->local_count, 0, sizeof(a->local_count));
    for (int i = a->start; i < a->end; i++)
        a->local_count[(a->arr[i] >> a->shift) & 0xFF]++;
    return NULL;
}

void radix_sort_pthread(unsigned int *arr, int n, int num_threads) {
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    unsigned int *tmp = (unsigned int *)malloc(n * sizeof(unsigned int));
    pthread_t     threads[MAX_THREADS];
    ThreadArg     args[MAX_THREADS];

    for (int pass = 0; pass < PASSES; pass++) {
        int shift = pass * 8;

        /* ── Хэсэгт хуваах ── */
        int chunk = n / num_threads;
        for (int t = 0; t < num_threads; t++) {
            args[t].arr   = arr;
            args[t].shift = shift;
            args[t].start = t * chunk;
            args[t].end   = (t == num_threads-1) ? n : (t+1) * chunk;
            pthread_create(&threads[t], NULL, count_worker, &args[t]);
        }
        for (int t = 0; t < num_threads; t++)
            pthread_join(threads[t], NULL);

        /* ── Дэд тоолоосыг нэгтгэх ── */
        unsigned int global_count[RADIX] = {0};
        for (int t = 0; t < num_threads; t++)
            for (int b = 0; b < RADIX; b++)
                global_count[b] += args[t].local_count[b];

        /* ── Prefix sum ── */
        unsigned int prefix[RADIX];
        prefix[0] = 0;
        for (int i = 1; i < RADIX; i++)
            prefix[i] = prefix[i-1] + global_count[i-1];

        /* ── Scatter (цуваа – тогтворт байдлын шаардлагаас) ── */
        for (int i = 0; i < n; i++) {
            int b = (arr[i] >> shift) & 0xFF;
            tmp[prefix[b]++] = arr[i];
        }
        memcpy(arr, tmp, n * sizeof(unsigned int));
    }
    free(tmp);
}