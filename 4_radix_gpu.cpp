// %%writefile radix_cuda_v2.cu
//  Compile: nvcc -O3 -std=c++17 -o radix_cuda_v2 radix_cuda_v2.cu
//  Run    : ./radix_cuda_v2
//
// ─── Bug fixes over radix_cuda_fixed.cu ─────────────────────────────────────
//
//  [BUG A — illegal memory access, was line 228]
//    global_pos = g_off[bin*BLOCKS+bid] + s_base[bin] + local_rank
//    ↑ g_off[..] after the Blelloch scan already IS the correct global start
//      for this block's bin-b elements.  Adding s_base[bin] (the intra-block
//      prefix sum) double-counts and shoots writes past the array end → OOB.
//    FIX: remove s_base entirely.
//
//  [BUG B — wrong sort output]
//    atomicAdd in Phase E assigns ranks non-deterministically within a warp.
//    LSD radix sort requires each pass to be STABLE (preserve relative order
//    of equal-digit elements).  Non-deterministic ranks break stability and
//    produce an incorrectly sorted result on many inputs.
//    FIX: use a WORKERS-based parallel scatter.  Each of WORKERS=8 threads
//         handles a contiguous sub-chunk and scatters it sequentially.
//         The starting offset for each worker's bin is computed from a
//         small per-worker histogram stored in shared memory.
//         Stability is guaranteed: worker 0's elements always precede
//         worker 1's elements within each bin (matching input order).
//
//  [FIX 2 — kept] All 8 CUDA events are destroyed (original leaked 6).
//  [FIX 3 — kept] CUDA_CHECK macro wraps every CUDA call.
//
// ─── Shared-memory budget for reorder_kernel ────────────────────────────────
//    s_worker_hist[WORKERS][BINS] = 8 × 256 × 4 = 8 KB
//    s_worker_off [WORKERS][BINS] = 8 × 256 × 4 = 8 KB
//    Total = 16 KB  (well within the 48 KB per-SM limit)
// ────────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════════════════════
#define BLOCKS   8
#define THREADS  256
#define BINS     256
#define PASSES   4
#define SCAN_N   (BLOCKS * BINS)   // 2048 — must be a power of 2 ✓

// WORKERS: threads that do actual scatter work per CUDA block.
// Must satisfy: 2 * WORKERS * BINS * 4 bytes ≤ shared-mem limit (48 KB).
// WORKERS=8  → 16 KB  ✓
#define WORKERS  8

// ═══════════════════════════════════════════════════════════════════════════
//  Error-checking macro
// ═══════════════════════════════════════════════════════════════════════════
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 1 — count_kernel   (unchanged — was already correct)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void count_kernel(const uint32_t *in,
                              uint32_t       *block_hist,
                              int n, uint32_t shift)
{
    __shared__ uint32_t local_hist[BINS];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid < BINS) local_hist[tid] = 0;
    __syncthreads();

    int chunk = (n + BLOCKS - 1) / BLOCKS;
    int start = bid * chunk;
    int end   = (start + chunk < n) ? (start + chunk) : n;

    for (int i = start + tid; i < end; i += THREADS) {
        uint32_t bin = (in[i] >> shift) & 0xFFu;
        atomicAdd(&local_hist[bin], 1u);
    }
    __syncthreads();

    if (tid < BINS)
        block_hist[tid * BLOCKS + bid] = local_hist[tid];
}

// ═══════════════════════════════════════════════════════════════════════════
//  KERNELS 2 & 3 — Blelloch exclusive scan  (unchanged — was already correct)
//  Source: GPU Gems 3, Ch. 39 (Harris et al., NVIDIA)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void upsweep_kernel(uint32_t *data, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i   = (tid + 1) * stride - 1;
    if (i < SCAN_N)
        data[i] += data[i - stride / 2];
}

__global__ void downsweep_kernel(uint32_t *data, int stride)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int i    = (tid + 1) * stride - 1;
    if (i < SCAN_N) {
        int      left = i - stride / 2;
        uint32_t t    = data[left];
        data[left]    = data[i];
        data[i]      += t;
    }
}

static void blelloch_exclusive_scan(uint32_t *d_data)
{
    for (int stride = 2; stride <= SCAN_N; stride <<= 1) {
        int num = SCAN_N / stride;
        int tpb = (num < THREADS) ? num : THREADS;
        int blk = (num + tpb - 1) / tpb;
        upsweep_kernel<<<blk, tpb>>>(d_data, stride);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemset(d_data + SCAN_N - 1, 0, sizeof(uint32_t)));
    for (int stride = SCAN_N; stride >= 2; stride >>= 1) {
        int num = SCAN_N / stride;
        int tpb = (num < THREADS) ? num : THREADS;
        int blk = (num + tpb - 1) / tpb;
        downsweep_kernel<<<blk, tpb>>>(d_data, stride);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 4 — reorder_kernel   [rewritten to fix both Bug A and Bug B]
//
//  Strategy: WORKERS-based parallel stable scatter
//
//  The block's chunk [start, end) is split into WORKERS contiguous sub-chunks.
//  Thread w (0 ≤ w < WORKERS) owns sub-chunk w and processes it sequentially,
//  which guarantees stable ordering within each bin.
//
//  Phase A  (all threads)  zero the two shared-memory tables
//  Phase B  (threads 0..WORKERS-1)
//           each worker counts its sub-chunk into s_worker_hist[w][b]
//  Phase C  (thread 0)
//           for each bin b, compute exclusive prefix sum across workers:
//             s_worker_off[0][b] = g_off[b * BLOCKS + bid]    ← global base
//             s_worker_off[w][b] = s_worker_off[w-1][b] + s_worker_hist[w-1][b]
//  Phase D  (threads 0..WORKERS-1)
//           each worker scatters its sub-chunk sequentially using s_worker_off[w][b]
//
//  Correctness proof (stability):
//    Worker 0 handles input[start .. start+sub-1],
//    worker 1 handles input[start+sub .. start+2*sub-1], etc.
//    s_worker_off ensures worker 0's bin-b elements occupy positions
//    immediately before worker 1's bin-b elements in out[].
//    So the output preserves the original input order within every bin. ✓
//
//  No s_base needed — g_off[b*BLOCKS+bid] is already the correct global start.
// ═══════════════════════════════════════════════════════════════════════════
__global__ void reorder_kernel(const uint32_t *in,
                                uint32_t       *out,
                                const uint32_t *g_off,
                                int n, uint32_t shift)
{
    // 16 KB total shared memory — well within hardware limits
    __shared__ uint32_t s_worker_hist[WORKERS][BINS]; // per-worker bin counts
    __shared__ uint32_t s_worker_off [WORKERS][BINS]; // per-worker scatter offsets

    int bid   = blockIdx.x;
    int chunk = (n + BLOCKS - 1) / BLOCKS;
    int start = bid * chunk;
    int end   = (start + chunk < n) ? (start + chunk) : n;
    int local_n = end - start;

    // ── Phase A: zero shared tables ───────────────────────────────────────
    for (int idx = threadIdx.x; idx < WORKERS * BINS; idx += THREADS) {
        int w = idx / BINS, b = idx % BINS;
        s_worker_hist[w][b] = 0;
        s_worker_off [w][b] = 0;
    }
    __syncthreads();

    // ── Phase B: each worker counts its own sub-chunk (parallel, no races) ─
    if (threadIdx.x < WORKERS) {
        int w       = threadIdx.x;
        int sub     = (local_n + WORKERS - 1) / WORKERS;
        int lo      = start + w * sub;
        int hi      = (lo + sub < end) ? (lo + sub) : end;
        for (int i = lo; i < hi; i++) {
            uint32_t bin = (in[i] >> shift) & 0xFFu;
            s_worker_hist[w][bin]++;  // only this worker writes row w — no contention
        }
    }
    __syncthreads();

    // ── Phase C: thread 0 builds per-worker scatter offsets ──────────────
    // For bin b: worker 0 starts at g_off[b*BLOCKS+bid] (the Blelloch result),
    //            worker w starts after worker w-1's bin-b elements.
    // This is the ONLY correct use of g_off — no extra s_base needed.
    if (threadIdx.x == 0) {
        for (int b = 0; b < BINS; b++) {
            uint32_t pos = g_off[b * BLOCKS + bid];   // global start for this block+bin
            for (int w = 0; w < WORKERS; w++) {
                s_worker_off[w][b]  = pos;
                pos                += s_worker_hist[w][b];
            }
        }
    }
    __syncthreads();

    // ── Phase D: each worker scatters its sub-chunk — stable, no conflicts ─
    if (threadIdx.x < WORKERS) {
        int w       = threadIdx.x;
        int sub     = (local_n + WORKERS - 1) / WORKERS;
        int lo      = start + w * sub;
        int hi      = (lo + sub < end) ? (lo + sub) : end;
        for (int i = lo; i < hi; i++) {
            uint32_t v   = in[i];
            uint32_t bin = (v >> shift) & 0xFFu;
            out[ s_worker_off[w][bin]++ ] = v;  // unique slot per element ✓
        }
    }
    // No final syncthreads needed — each worker writes to non-overlapping slots.
}

// ═══════════════════════════════════════════════════════════════════════════
//  Timing struct
// ═══════════════════════════════════════════════════════════════════════════
struct Timing {
    float ms_total, ms_kernel, ms_h2d, ms_d2h;
};

// ═══════════════════════════════════════════════════════════════════════════
//  Main sort driver
// ═══════════════════════════════════════════════════════════════════════════
Timing run_radix_cuda(const uint32_t *host_in, uint32_t *host_out, int N)
{
    size_t   size_in = (size_t)N * sizeof(uint32_t);
    uint32_t *d_A = nullptr, *d_B = nullptr, *d_off = nullptr;

    cudaEvent_t e_total_s, e_total_e,
                e_gpu_s,   e_gpu_e,
                e_h2d_s,   e_h2d_e,
                e_d2h_s,   e_d2h_e;

    CUDA_CHECK(cudaEventCreate(&e_total_s)); CUDA_CHECK(cudaEventCreate(&e_total_e));
    CUDA_CHECK(cudaEventCreate(&e_gpu_s));   CUDA_CHECK(cudaEventCreate(&e_gpu_e));
    CUDA_CHECK(cudaEventCreate(&e_h2d_s));   CUDA_CHECK(cudaEventCreate(&e_h2d_e));
    CUDA_CHECK(cudaEventCreate(&e_d2h_s));   CUDA_CHECK(cudaEventCreate(&e_d2h_e));

    CUDA_CHECK(cudaEventRecord(e_total_s));

    CUDA_CHECK(cudaMalloc(&d_A,   size_in));
    CUDA_CHECK(cudaMalloc(&d_B,   size_in));
    CUDA_CHECK(cudaMalloc(&d_off, SCAN_N * sizeof(uint32_t)));

    CUDA_CHECK(cudaEventRecord(e_h2d_s));
    CUDA_CHECK(cudaMemcpy(d_A, host_in, size_in, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e_h2d_e));

    CUDA_CHECK(cudaEventRecord(e_gpu_s));
    for (int pass = 0; pass < PASSES; pass++) {
        uint32_t shift = (uint32_t)(pass * 8);

        CUDA_CHECK(cudaMemset(d_off, 0, SCAN_N * sizeof(uint32_t)));

        count_kernel<<<BLOCKS, THREADS>>>(d_A, d_off, N, shift);
        CUDA_CHECK(cudaGetLastError());

        blelloch_exclusive_scan(d_off);

        reorder_kernel<<<BLOCKS, THREADS>>>(d_A, d_B, d_off, N, shift);
        CUDA_CHECK(cudaGetLastError());

        uint32_t *tmp = d_A; d_A = d_B; d_B = tmp;
    }
    CUDA_CHECK(cudaEventRecord(e_gpu_e));

    CUDA_CHECK(cudaEventRecord(e_d2h_s));
    CUDA_CHECK(cudaMemcpy(host_out, d_A, size_in, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e_d2h_e));

    CUDA_CHECK(cudaEventRecord(e_total_e));
    CUDA_CHECK(cudaEventSynchronize(e_total_e));

    float ms_total, ms_kernel, ms_h2d, ms_d2h;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total,  e_total_s, e_total_e));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, e_gpu_s,   e_gpu_e));
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d,    e_h2d_s,   e_h2d_e));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h,    e_d2h_s,   e_d2h_e));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_off));

    // All 8 events destroyed (was leaking 6 in the original)
    CUDA_CHECK(cudaEventDestroy(e_total_s)); CUDA_CHECK(cudaEventDestroy(e_total_e));
    CUDA_CHECK(cudaEventDestroy(e_gpu_s));   CUDA_CHECK(cudaEventDestroy(e_gpu_e));
    CUDA_CHECK(cudaEventDestroy(e_h2d_s));   CUDA_CHECK(cudaEventDestroy(e_h2d_e));
    CUDA_CHECK(cudaEventDestroy(e_d2h_s));   CUDA_CHECK(cudaEventDestroy(e_d2h_e));

    return Timing{ ms_total, ms_kernel, ms_h2d, ms_d2h };
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════
int main()
{
    int sizes[] = { 10000, 100000, 1000000 };

    FILE *csv = fopen("results_cuda.csv", "w");
    if (!csv) { fprintf(stderr, "ERROR: cannot open results_cuda.csv\n"); return 1; }
    fprintf(csv, "implementation,N,execution_ms,computation_ms,transfer_ms,"
                 "transfer_bytes,total_ops,performance_mops,sorted_ok\n");

    printf("implementation = cuda_v2 (base-256 LSD, uint32, %d passes, "
           "%d blocks x %d threads, %d scatter workers/block)\n",
           PASSES, BLOCKS, THREADS, WORKERS);
    printf("bugs fixed: OOB s_base term | unstable atomicAdd | "
           "event leaks | missing CUDA_CHECK\n");
    printf("--------------------------------------------------------------\n");

    for (int s = 0; s < 3; s++) {
        int N = sizes[s];
        std::vector<uint32_t> input(N), output(N);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<uint32_t> dist;
        for (int i = 0; i < N; i++) input[i] = dist(rng);

        // Reference: sort a copy with std::sort for verification
        std::vector<uint32_t> ref(input);
        std::sort(ref.begin(), ref.end());

        Timing t = run_radix_cuda(input.data(), output.data(), N);
        bool ok  = std::is_sorted(output.begin(), output.end())
                && (output == ref);

        const double execution_ms   = t.ms_total;
        const double computation_ms = t.ms_kernel;
        const double transfer_ms    = t.ms_h2d + t.ms_d2h;
        const uint64_t transfer_bytes = 2ull * (uint64_t)N * sizeof(uint32_t);

        // ops per pass: N reads (count) + SCAN_N (scan) + N reads + N writes (scatter)
        const uint64_t total_ops =
            (uint64_t)PASSES * (3ull * (uint64_t)N + (uint64_t)SCAN_N);
        const double performance_mops =
            execution_ms > 0.0
                ? (double)total_ops / (execution_ms * 1000.0)
                : 0.0;

        printf("N=%-8d  Exec=%.3f ms  Compute=%.3f ms  "
               "Transfer=%.3f ms (H2D=%.3f D2H=%.3f)  "
               "Bytes=%llu  Ops=%llu  Perf=%.1f MOPS  Sorted=%s\n",
               N, execution_ms, computation_ms, transfer_ms,
               t.ms_h2d, t.ms_d2h,
               (unsigned long long)transfer_bytes,
               (unsigned long long)total_ops,
               performance_mops,
               ok ? "true" : "false");

        fprintf(csv, "cuda_v2,%d,%.6f,%.6f,%.6f,%llu,%llu,%.6f,%s\n",
                N, execution_ms, computation_ms, transfer_ms,
                (unsigned long long)transfer_bytes,
                (unsigned long long)total_ops,
                performance_mops,
                ok ? "true" : "false");

        if (!ok) {
            fprintf(stderr, "ERROR: output not sorted for N=%d\n", N);
            fclose(csv);
            return 1;
        }
    }

    fclose(csv);
    printf("Wrote results_cuda.csv\n");
    return 0;
}