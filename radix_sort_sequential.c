#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void counting_sort_by_digit(int *arr, int n, int exp) {
    int *output = (int *)malloc(n * sizeof(int));
    int count[10] = {0};

    for (int i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    memcpy(arr, output, n * sizeof(int));
    free(output);
}

void radix_sort(int *arr, int n) {
    int max_val = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max_val)
            max_val = arr[i];

    for (int exp = 1; max_val / exp > 0; exp *= 10)
        counting_sort_by_digit(arr, n, exp);
}

void generate_random_array(int *arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000000;
}

double measure_time(int n) {
    int *arr = (int *)malloc(n * sizeof(int));
    generate_random_array(arr, n);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    radix_sort(arr, n);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0
                   + (end.tv_nsec - start.tv_nsec) / 1e6;

    free(arr);
    return elapsed;
}

int main() {
    srand(42);

    int sizes[] = {10000, 100000, 1000000};
    int num_sizes = 3;

    printf("%-15s %s\n", "Элементийн тоо", "Хугацаа (ms)");
    printf("-----------------------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        double t = measure_time(sizes[i]);
        printf("%-15d %.3f ms\n", sizes[i], t);
    }

    return 0;
}
