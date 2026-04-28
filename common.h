#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>

#define RADIX         256
#define PASSES        4
#define MAX_THREADS   8

// utills.c доторх функцүүд
void fill_random(unsigned int *arr, int n);
void copy_array(unsigned int *dst, const unsigned int *src, int n);
int is_sorted(unsigned int *arr, int n);
double get_time_sec(struct timespec *start, struct timespec *end);

// Эрэмбэлэх аргууд
void radix_sort_sequential(unsigned int *arr, int n);
void radix_sort_pthread(unsigned int *arr, int n, int num_threads);
void radix_sort_openmp(unsigned int *arr, int n, int num_threads);

#endif