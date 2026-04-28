#include "common.h"

void fill_random(unsigned int *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = (unsigned int)rand();
}

void copy_array(unsigned int *dst, const unsigned int *src, int n) {
    memcpy(dst, src, n * sizeof(unsigned int));
}

int is_sorted(unsigned int *arr, int n) {
    for (int i = 1; i < n; i++) if (arr[i] < arr[i-1]) return 0;
    return 1;
}

double get_time_sec(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) * 1e-9;
}