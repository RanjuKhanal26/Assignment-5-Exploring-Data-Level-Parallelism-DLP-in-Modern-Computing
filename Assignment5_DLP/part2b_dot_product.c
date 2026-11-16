/*
 * Part 2.2 - SIMD dot product vs scalar, with RUNTIME CPU dispatch.
 *
 * Build (works on any x86-64 Linux, no -mavx2 / -mfma required):
 *     gcc -O3 -fno-tree-vectorize part2b_dot_product.c -o part2b_dot_product
 *
 * The binary inspects the CPU at startup and picks the widest path
 * the hardware supports:
 *     AVX2 + FMA   ->  8-wide fused multiply-add loop
 *     SSE2         ->  4-wide mul + add loop
 *     otherwise    ->  scalar loop
 *
 * `-fno-tree-vectorize` keeps the scalar reference genuinely scalar,
 * so the measured speedup reflects the SIMD path only (not GCC
 * silently auto-vectorizing the baseline).
 *
 * Writes results/results_simd.csv with one row per path executed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#  include <immintrin.h>
#  define HAVE_X86 1
#else
#  define HAVE_X86 0
#endif

#if defined(_WIN32)
#  include <windows.h>
static double now(void) {
    LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
#  include <time.h>
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ============================================================
 * Scalar reference
 * ============================================================ */
static float dot_scalar(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* ============================================================
 * SSE2 version - safe on every x86-64 CPU (SSE2 is baseline)
 * ============================================================ */
#if HAVE_X86
__attribute__((target("sse2")))
static float dot_sse(const float *a, const float *b, int n) {
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* ============================================================
 * AVX2 + FMA version - only called when the CPU reports both
 * ============================================================ */
__attribute__((target("avx2,fma")))
static float dot_avx2(const float *a, const float *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float s = 0.0f;
    for (int j = 0; j < 8; ++j) s += tmp[j];
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
#endif /* HAVE_X86 */

/* ============================================================
 * Runtime capability probe
 * ============================================================ */
typedef enum { PATH_SCALAR = 0, PATH_SSE = 1, PATH_AVX2 = 2 } simd_path;

static const char *path_name(simd_path p) {
    switch (p) {
        case PATH_AVX2: return "avx2";
        case PATH_SSE:  return "sse";
        default:        return "scalar";
    }
}

static simd_path detect_best_path(void) {
#if HAVE_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return PATH_AVX2;
    if (__builtin_cpu_supports("sse2"))
        return PATH_SSE;
#endif
    return PATH_SCALAR;
}

/* ============================================================
 * Helpers
 * ============================================================ */
static float *rand_array(int n) {
    /* 32-byte alignment - fine for both SSE (16) and AVX2 (32) */
    float *p = (float *)aligned_alloc(32, (size_t)n * sizeof(float));
    if (!p) { perror("aligned_alloc"); exit(1); }
    for (int i = 0; i < n; ++i) p[i] = (float)rand() / (float)RAND_MAX;
    return p;
}

int main(int argc, char **argv) {
    int N    = (argc > 1) ? atoi(argv[1]) : 10000000;
    int REPS = 20;

    simd_path best = detect_best_path();
    printf("CPU probe: best SIMD path = %s\n", path_name(best));

    float *a = rand_array(N);
    float *b = rand_array(N);

    /* Warm-up + correctness check: all paths must produce the same sum */
    volatile float s0 = dot_scalar(a, b, N);
#if HAVE_X86
    volatile float s_sse = dot_sse(a, b, N);
    volatile float s_avx = (best == PATH_AVX2) ? dot_avx2(a, b, N) : s_sse;
    printf("Correctness: scalar=%.6f  sse=%.6f  avx2=%.6f\n",
           (float)s0, (float)s_sse, (float)s_avx);
#endif
    (void)s0;

    double t0, t1;
    double t_scalar = 1e9, t_sse = 1e9, t_avx2 = 1e9;

    for (int r = 0; r < REPS; ++r) {
        t0 = now(); volatile float x = dot_scalar(a, b, N); t1 = now(); (void)x;
        if (t1 - t0 < t_scalar) t_scalar = t1 - t0;

#if HAVE_X86
        t0 = now(); x = dot_sse(a, b, N);  t1 = now(); (void)x;
        if (t1 - t0 < t_sse) t_sse = t1 - t0;

        if (best == PATH_AVX2) {
            t0 = now(); x = dot_avx2(a, b, N); t1 = now(); (void)x;
            if (t1 - t0 < t_avx2) t_avx2 = t1 - t0;
        }
#endif
    }

    /* -------- report -------- */
    printf("\nN = %d\n", N);
    printf("Scalar time : %.6f s\n", t_scalar);
#if HAVE_X86
    printf("SSE    time : %.6f s  (speedup %.2fx vs scalar)\n",
           t_sse, t_scalar / t_sse);
    if (best == PATH_AVX2)
        printf("AVX2   time : %.6f s  (speedup %.2fx vs scalar)\n",
               t_avx2, t_scalar / t_avx2);
    else
        printf("AVX2   time : not available on this CPU\n");
#endif

    /* -------- CSV -------- */
    FILE *f = fopen("results/results_simd.csv", "w");
    if (!f) { perror("results/results_simd.csv"); return 1; }
    fprintf(f, "implementation,time_seconds\n");
    fprintf(f, "scalar,%.9f\n", t_scalar);
#if HAVE_X86
    fprintf(f, "sse,%.9f\n", t_sse);
    if (best == PATH_AVX2) fprintf(f, "avx2,%.9f\n", t_avx2);
#endif
    fclose(f);

    free(a);
    free(b);
    return 0;
}
