#include "common.h"

// --- PTHREAD хэсэг ---
typedef struct {
    unsigned int *arr;
    int start, end, shift;
    unsigned int local_count[RADIX];
} ThreadArg;

static void *count_worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    memset(a->local_count, 0, sizeof(a->local_count));
    for (int i = a->start; i < a->end; i++)
        a->local_count[(a->arr[i] >> a->shift) & 0xFF]++;
    return NULL;
}

void radix_sort_pthread(unsigned int *arr, int n, int num_threads) {
    unsigned int *tmp = malloc(n * sizeof(unsigned int));
    pthread_t threads[MAX_THREADS];
    ThreadArg args[MAX_THREADS];

    for (int pass = 0; pass < PASSES; pass++) {
        int shift = pass * 8;
        int chunk = n / num_threads;
        for (int t = 0; t < num_threads; t++) {
            args[t].arr = arr; args[t].shift = shift;
            args[t].start = t * chunk;
            args[t].end = (t == num_threads-1) ? n : (t+1) * chunk;
            pthread_create(&threads[t], NULL, count_worker, &args[t]);
        }
        for (int t = 0; t < num_threads; t++) pthread_join(threads[t], NULL);

        unsigned int global_count[RADIX] = {0};
        for (int t = 0; t < num_threads; t++)
            for (int b = 0; b < RADIX; b++) global_count[b] += args[t].local_count[b];

        unsigned int prefix[RADIX];
        prefix[0] = 0;
        for (int i = 1; i < RADIX; i++) prefix[i] = prefix[i-1] + global_count[i-1];

        for (int i = 0; i < n; i++) {
            int b = (arr[i] >> shift) & 0xFF;
            tmp[prefix[b]++] = arr[i];
        }
        memcpy(arr, tmp, n * sizeof(unsigned int));
    }
    free(tmp);
}

// --- OPENMP хэсэг ---
void radix_sort_openmp(unsigned int *arr, int n, int num_threads) {
    unsigned int *tmp = malloc(n * sizeof(unsigned int));
    omp_set_num_threads(num_threads);
    for (int pass = 0; pass < PASSES; pass++) {
        int shift = pass * 8;
        unsigned int (*local_counts)[RADIX] = NULL;
        int actual_threads;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp single
            {
                actual_threads = omp_get_num_threads();
                local_counts = calloc(actual_threads * RADIX, sizeof(unsigned int));
            }
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) local_counts[tid][(arr[i] >> shift) & 0xFF]++;
        }
        unsigned int global_count[RADIX] = {0};
        for (int t = 0; t < actual_threads; t++)
            for (int b = 0; b < RADIX; b++) global_count[b] += local_counts[t][b];
        free(local_counts);
        unsigned int prefix[RADIX];
        prefix[0] = 0;
        for (int i = 1; i < RADIX; i++) prefix[i] = prefix[i-1] + global_count[i-1];
        for (int i = 0; i < n; i++) {
            int b = (arr[i] >> shift) & 0xFF;
            tmp[prefix[b]++] = arr[i];
        }
        memcpy(arr, tmp, n * sizeof(unsigned int));
    }
    free(tmp);
}