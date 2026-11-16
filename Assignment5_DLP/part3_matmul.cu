/*
 * Part 3.2 - Naive vs tiled CUDA matrix multiplication.
 * Build: nvcc -O3 part3_matmul.cu -o part3_matmul
 * Writes results/results_matmul.csv.
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } } while (0)

/* ---- Naive kernel: one thread per output element ---- */
__global__ void matmul_naive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float s = 0.0f;
        for (int k = 0; k < N; ++k) s += A[row * N + k] * B[k * N + col];
        C[row * N + col] = s;
    }
}

/* ---- Tiled kernel using shared memory ---- */
#define TILE 16
__global__ void matmul_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float s = 0.0f;
    for (int t = 0; t < N / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE; ++k) s += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = s;
}

static void fill(float *p, int n) { for (int i = 0; i < n; ++i) p[i] = (float)rand() / RAND_MAX; }

static float bench(void (*launch)(const float*, const float*, float*, int),
                   const float *A, const float *B, float *C, int N) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    launch(A, B, C, N); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    launch(A, B, C, N);
    cudaEventRecord(e);
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

static void launch_naive(const float *A, const float *B, float *C, int N) {
    dim3 t(16, 16), b((N + 15) / 16, (N + 15) / 16);
    matmul_naive<<<b, t>>>(A, B, C, N);
}
static void launch_tiled(const float *A, const float *B, float *C, int N) {
    dim3 t(TILE, TILE), b(N / TILE, N / TILE);
    matmul_tiled<<<b, t>>>(A, B, C, N);
}

int main(void) {
    int sizes[] = {512, 1024, 2048, 4096};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    FILE *f = fopen("results/results_matmul.csv", "w");
    if (!f) { perror("results/results_matmul.csv"); return 1; }
    fprintf(f, "N,kernel,time_ms\n");

    for (int si = 0; si < nsizes; ++si) {
        int N = sizes[si];
        size_t bytes = (size_t)N * N * sizeof(float);
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
        fill(hA, N * N); fill(hB, N * N);
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

        float t_naive = bench(launch_naive, dA, dB, dC, N);
        float t_tiled = bench(launch_tiled, dA, dB, dC, N);

        printf("N=%d  naive=%.3f ms  tiled=%.3f ms  speedup=%.2fx\n",
               N, t_naive, t_tiled, t_naive / t_tiled);
        fprintf(f, "%d,naive,%.3f\n", N, t_naive);
        fprintf(f, "%d,tiled,%.3f\n", N, t_tiled);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(hA); free(hB);
    }
    fclose(f);
    return 0;
}
