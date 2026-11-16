#!/usr/bin/env bash
set -e
mkdir -p results figures screenshots

echo "== Part 2.1: NumPy vectorization =="
python3 part2a_vector_add.py

echo "== Part 2.2: SIMD dot product =="
gcc -O3 -mavx2 -mfma -ffast-math part2b_dot_product.c -o part2b_dot_product
./part2b_dot_product 10000000

echo "== Part 3: CUDA matrix multiply =="
if command -v nvcc >/dev/null 2>&1; then
    nvcc -O3 part3_matmul.cu -o part3_matmul
    ./part3_matmul
else
    echo "nvcc not found; skipping part 3"
fi

echo "== Part 4: OpenMP reduction =="
gcc -O3 -fopenmp part4_reduction.c -o part4_reduction
./part4_reduction

echo "== Generating figures =="
python3 plot_results.py
