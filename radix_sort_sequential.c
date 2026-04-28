#include "common.h"

void radix_sort_sequential(unsigned int *arr, int n) {
    unsigned int *output = malloc(n * sizeof(unsigned int));
    for (int pass = 0; pass < PASSES; pass++) {
        int shift = pass * 8;
        unsigned int count[RADIX] = {0};
        for (int i = 0; i < n; i++) count[(arr[i] >> shift) & 0xFF]++;
        for (int i = 1; i < RADIX; i++) count[i] += count[i-1];
        for (int i = n-1; i >= 0; i--) {
            int b = (arr[i] >> shift) & 0xFF;
            output[count[b]-1] = arr[i];
            count[b]--;
        }
        memcpy(arr, output, n * sizeof(unsigned int));
    }
    free(output);
}