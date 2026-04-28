// =============================================================================
// radix_sort_sequential.cpp
//
// Sequential LSD (Least Significant Digit) radix sort, base 10.
// Sorts std::vector<uint32_t> using stable counting sort as the per-digit
// subroutine.
//
// References:
//   - CLRS, 3rd ed., Section 8.3 ("Radix sort") — algorithmic structure
//     (stable counting sort applied digit-by-digit, LSD to MSD).
//   - Sedgewick & Wayne, Algorithms 4e, §5.1 (LSD.java) — the
//     histogram / prefix-sum / distribute structure used here.
//   - GeeksforGeeks "C++ Program for Radix Sort" — base-10 reference,
//     used only as a sanity check.
//
// Build:   g++ -O2 -std=c++17 radix_sort_sequential.cpp -o radix_sort_seq
// Run:     ./radix_sort_seq
//
// NOTE on later parallelization:
//   The per-digit subroutine `counting_sort_by_digit` is split into three
//   explicit phases — (1) histogram, (2) prefix sum, (3) distribute. This
//   is the structure the std::thread and OpenMP variants will parallelize.
// =============================================================================




#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// Stable counting sort on the decimal digit at place value `exp` (1, 10, 100, …).
// Sorts `a` into `aux` according to that single digit, then swaps the buffers
// so that the sorted data is back in `a` after the call.
static void counting_sort_by_digit(std::vector<uint32_t>& a,
                                   std::vector<uint32_t>& aux,
                                   uint64_t exp) {
    constexpr int R = 10;          // radix (base 10)
    int count[R + 1] = {0};        // one extra slot lets us do the prefix-sum
                                   // trick without an offset variable
    const std::size_t n = a.size();

    // ---- (1) histogram: count how many keys have each digit value ----------
    for (std::size_t i = 0; i < n; ++i) {
        int d = static_cast<int>((a[i] / exp) % R);
        count[d + 1]++;
    }

    // ---- (2) prefix sum: count[d] becomes the start index for digit d ------
    for (int r = 0; r < R; ++r) {
        count[r + 1] += count[r];
    }

    // ---- (3) distribute: scatter each key to its bucket, advancing the -----
    //          per-digit cursor. This is what makes the sort STABLE: keys
    //          with the same digit keep their relative input order.
    for (std::size_t i = 0; i < n; ++i) {
        int d = static_cast<int>((a[i] / exp) % R);
        aux[count[d]++] = a[i];
    }

    // Sedgewick's optimization: instead of copying aux back into a, swap the
    // two buffers' contents (O(1) for std::vector — only pointers move).
    a.swap(aux);
}

// LSD radix sort, base 10, for unsigned 32-bit integers.
void radix_sort_lsd(std::vector<uint32_t>& a) {
    if (a.size() < 2) return;

    // We only need to process digits up through the most significant digit
    // of the maximum value — leading zeros above that are all the same.
    const uint32_t max_val = *std::max_element(a.begin(), a.end());

    std::vector<uint32_t> aux(a.size());

    for (uint64_t exp = 1; max_val / exp > 0; exp *= 10) {
        counting_sort_by_digit(a, aux, exp);
    }
}

// =============================================================================
// Benchmark driver
// =============================================================================
int main() {
    using clock = std::chrono::steady_clock;

    constexpr std::size_t N = 1'000'000;

    // ---- generate random test data (fixed seed -> reproducible runs) -------
    std::vector<uint32_t> data(N);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(
        0, std::numeric_limits<uint32_t>::max());
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
    std::vector<uint32_t> ref = data;  // independent copy for std::sort

    // ---- time the radix sort ----------------------------------------------
    auto t1 = clock::now();
    radix_sort_lsd(data);
    auto t2 = clock::now();
    double ms_radix = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ---- time std::sort on the same input, as a reference ----------------
    auto t3 = clock::now();
    std::sort(ref.begin(), ref.end());
    auto t4 = clock::now();
    double ms_std = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // ---- verify correctness -----------------------------------------------
    const bool sorted_ok = std::is_sorted(data.begin(), data.end());
    const bool matches   = (data == ref);
    const bool ok        = sorted_ok && matches;

    std::cout << "N             = " << N           << "\n";
    std::cout << "Radix sort    = " << ms_radix    << " ms\n";
    std::cout << "std::sort     = " << ms_std      << " ms\n";
    std::cout << "is_sorted     = " << (sorted_ok ? "true" : "false") << "\n";
    std::cout << "matches std   = " << (matches   ? "true" : "false") << "\n";
    std::cout << "Result        = " << (ok ? "OK" : "MISMATCH") << "\n";

    return ok ? 0 : 1;
}