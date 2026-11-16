/*
 * Part 4 - OpenMP scaling test (Figure 6 data).
 * Memory-bound float-array sum: 400 MB array, well beyond L3,
 * so adding threads past the memory subsystem's capacity no longer
 * helps and the speedup curve bends away from ideal linear.
 *
 * Build:
 *     gcc -O3 -fopenmp part4_reduction.c -o part4_reduction
 *
 * Output:
 *     results/results_openmp.csv
 *     columns: threads,time_seconds
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (100 * 1000 * 1000)    /* 100M floats = 400 MB */

int main(void) {
    int tc[] = {1, 2, 4, 8};
    int nt = sizeof(tc) / sizeof(tc[0]);

    float *arr = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    if (!arr) { perror("aligned_alloc"); return 1; }
    for (int i = 0; i < N; ++i) arr[i] = (float)(i % 7) / 7.0f;

    FILE *f = fopen("results/results_openmp.csv", "w");
    if (!f) { perror("results/results_openmp.csv"); return 1; }
    fprintf(f, "threads,time_seconds\n");

    double t1 = 0.0;
    for (int k = 0; k < nt; ++k) {
        int T = tc[k];
        omp_set_num_threads(T);
        double best = 1e9;
        for (int r = 0; r < 3; ++r) {
            double t0 = omp_get_wtime();
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum) schedule(static)
            for (int i = 0; i < N; ++i) sum += arr[i];
            double dt = omp_get_wtime() - t0;
            if (dt < best) best = dt;
            (void)sum;
        }
        if (k == 0) t1 = best;
        printf("T=%2d  time=%.4f s  speedup=%.2fx  efficiency=%.2f\n",
               T, best, t1/best, (t1/best)/T);
        fprintf(f, "%d,%.9f\n", T, best);
    }
    fclose(f);
    free(arr);
    return 0;
}
