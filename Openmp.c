#include "common.h"

void benchmark(int n) {
    unsigned int *src = malloc(n * sizeof(unsigned int));
    unsigned int *work = malloc(n * sizeof(unsigned int));
    struct timespec t0, t1;
    char label[50];
    
    fill_random(src, n);

    // 1. Цуваа арга (Sequential)
    copy_array(work, src, n);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    radix_sort_sequential(work, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double seq_time = get_time_sec(&t0, &t1);

    printf("│ %10d │ %-18s │ %9.4f с │ %-8s │\n", 
           n, "Цуваа", seq_time, is_sorted(work, n) ? "✓ зөв" : "✗ алдаа");

    int thread_counts[] = {2, 4, 8};

    // 2. Pthread (Урсгалаар)
    for (int i = 0; i < 3; i++) {
        int t = thread_counts[i];
        copy_array(work, src, n);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        radix_sort_pthread(work, n, t);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double pt = get_time_sec(&t0, &t1);
        
        snprintf(label, sizeof(label), "Pthread (%d урсгал)", t);
        printf("│ %10d │ %-18s │ %9.4f с │ x%-7.2f │\n", 
               n, label, pt, seq_time / pt);
    }

    // 3. OpenMP (Урсгалаар)
    for (int i = 0; i < 3; i++) {
        int t = thread_counts[i];
        copy_array(work, src, n);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        radix_sort_openmp(work, n, t);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double omp_t = get_time_sec(&t0, &t1);
        
        snprintf(label, sizeof(label), "OpenMP  (%d урсгал)", t);
        printf("│ %10d │ %-18s │ %9.4f с │ x%-7.2f │\n", 
               n, label, omp_t, seq_time / omp_t);
    }
    printf("├────────────┼────────────────────┼───────────┼────────────┤\n");

    free(src);
    free(work);
}

int main() {
    srand(time(NULL));
    printf("\nRadix Sort — Параллел харьцуулалт (Урсгалаар)\n");
    printf("│ Элемент (N)│ Ажиллагааны арга   │ Хугацаа   │ Хурдасгалт │\n");

    // Өөр өөр хэмжээсүүд дээр турших
    int sizes[] = {100000, 1000000, 1000000}; 
    for (int i = 0; i < 3; i++) {
        benchmark(sizes[i]);
    }

    return 0;
}