//ThreadData бүтэц:
    //arr          ← эх массив
    //temp         ← түр зуурын массив
    //n            ← нийт элементүүдийн тоо
    //digit        ← одоогийн боловсруулж буй орон (0 = LSD)
    //thread_id
    //num_threads
    //global_count ← хамтарсан тоолох массив
    //mutex        ← synchronization
//функц get_digit(num: unsigned int, digit_pos: int) → int
  //  power ← 10^digit_pos          // pow(10, digit_pos)
//  return (num / power) % 10

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#define RADIX 10
#define MAX_THREADS 16

typedef struct {
    unsigned int *arr;
    unsigned int *temp;
    int n;
    int digit;          
    int thread_id;
    int num_threads;
    int *global_count;
    pthread_mutex_t *mutex;
} ThreadData;

// Base 10 digit авах функц (LSD)
int get_digit(unsigned int num, int digit_pos) {
    return (num / (unsigned int)pow(10, digit_pos)) % 10;
}

void* count_phase(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int start = data->thread_id * (data->n / data->num_threads);
    int end = (data->thread_id == data->num_threads - 1) ? 
              data->n : start + (data->n / data->num_threads);

    int local[RADIX] = {0};

    for (int i = start; i < end; i++) {
        int d = get_digit(data->arr[i], data->digit);
        local[d]++;
    }

    // Local → Global (thread-safe)
    pthread_mutex_lock(data->mutex);
    for (int i = 0; i < RADIX; i++) {
        data->global_count[i] += local[i];
    }
    pthread_mutex_unlock(data->mutex);

    return NULL;
}

void* permute_phase(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int start = data->thread_id * (data->n / data->num_threads);
    int end = (data->thread_id == data->num_threads - 1) ? 
              data->n : start + (data->n / data->num_threads);

    for (int i = start; i < end; i++) {
        int d = get_digit(data->arr[i], data->digit);
        int pos = --(data->global_count[d]);     // decrement and place
        data->temp[pos] = data->arr[i];
    }
    return NULL;
}

void parallel_lsd_radix_sort(unsigned int *arr, int n, int num_threads) {
    if (n <= 1 || num_threads < 1) return;
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;

    unsigned int *temp = (unsigned int*)malloc(n * sizeof(unsigned int));
    if (!temp) return;

    pthread_t threads[MAX_THREADS];
    ThreadData tdata[MAX_THREADS];
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    int max_digits = 10;        // Unsigned 32-bit → 10 digits max

    for (int d = 0; d < max_digits; d++) {        // LSD loop
        int global_count[RADIX] = {0};

        for (int t = 0; t < num_threads; t++) {
            tdata[t].arr         = arr;
            tdata[t].temp        = temp;
            tdata[t].n           = n;
            tdata[t].digit       = d;
            tdata[t].thread_id   = t;
            tdata[t].num_threads = num_threads;
            tdata[t].global_count = global_count;
            tdata[t].mutex       = &mutex;

            pthread_create(&threads[t], NULL, count_phase, &tdata[t]);
        }
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }

        for (int i = 1; i < RADIX; i++) {
            global_count[i] += global_count[i - 1];
        }

        for (int t = 0; t < num_threads; t++) {
            pthread_create(&threads[t], NULL, permute_phase, &tdata[t]);
        }
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }

        unsigned int *swap = arr;
        arr = temp;
        temp = swap;
    }


    if (arr != temp) {
        memcpy(arr, temp, n * sizeof(unsigned int));
    }

    free(temp);
    pthread_mutex_destroy(&mutex);
}

int main() {
    int n = 1000000;                     // 1 сая элемент
    unsigned int *arr = malloc(n * sizeof(unsigned int));

    // Санамсаргүй тоо үүсгэх
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() * rand();        // том unsigned 32-bit утга
    }

    printf("Sorting %d unsigned 32-bit integers (Base 10 LSD) with %d threads...\n", n, 4);
  //1st 10 sort
    parallel_lsd_radix_sort(arr, n, 4);
    printf("First 10 sorted: ");
    for (int i = 0; i < 10; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");

    free(arr);
    return 0;
}
//Алгоритм: PARALLEL_LSD_RADIX_SORT_BASE10(arr, n, num_threads)

  //  if n <= 1 then return

    //temp ← шинэ массив (size = n)
 //   max_digits ← 10                     // Unsigned 32-bit тоонд хамгийн их 10 орон

   // for digit = 0 to max_digits-1       // LSD: бага оронгоос эхэлнэ
     //   global_count[0..9] ← {0, 0, ..., 0}

        // =============================================
        // Phase 1: Parallel Counting (Тоолох фаз)
        // =============================================
        //for each thread t = 0 to num_threads-1 in parallel:
            //start ← t * (n / num_threads)
            //end   ← (t == num_threads-1) ? n : start + (n / num_threads)

           // local_count[0..9] ← {0}

            //for i = start to end-1:
                //d ← get_digit(arr[i], digit)        // Base 10 digit авах
               // local_count[d] ← local_count[d] + 1

            // Local-оо global_count руу нэмэх (mutex-ээр хамгаална)
            //lock(mutex)
            //for i = 0 to 9:
              ///  global_count[i] ← global_count[i] + local_count[i]
            //unlock(mutex)

        // =============================================
        // Phase 2: Prefix Sum (Cumulative Count)
        // =============================================
        //for i = 1 to 9:
           // global_count[i] ← global_count[i] + global_count[i-1]

        // =============================================
        // Phase 3: Parallel Permute / Scatter (Байрлуулах фаз)
        // =============================================
        //for each thread t = 0 to num_threads-1 in parallel:
            //start ← t * (n / num_threads)
            //end   ← (t == num_threads-1) ? n : start + (n / num_threads)

            //for i = start to end-1:
                //d ← get_digit(arr[i], digit)
                //pos ← global_count[d] - 1           // decrement
               // temp[pos] ← arr[i]
                //global_count[d] ← pos               // шинэчлэх

        // =============================================
        // Swap arr and temp
        // =============================================
        //swap(arr, temp)

    // Эцэст нь үр дүн arr-д байх ёстой
   // if arr != temp then
        //copy temp to arr

    //free(temp)
