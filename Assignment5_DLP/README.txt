Assignment 5 - DLP submission
=============================
Files:
  part2a_vector_add.py      NumPy vector demo (Part 2.1)
  part2b_dot_product.c      SIMD AVX2 demo     (Part 2.2)
  part3_matmul.cu           CUDA naive/tiled   (Part 3)
  part4_reduction.c         OpenMP reduction   (Part 4)
  plot_results.py           Generates figures
  Makefile / run_all.sh     Build + run helpers
  results/                  CSV benchmark output
  figures/                  PNG charts
  screenshots/              Compiler reports, nvidia-smi, etc.

How to run everything (Linux/macOS/WSL with gcc, nvcc, python3):
    ./run_all.sh

Or individually:
    python3 part2a_vector_add.py
    gcc -O3 -mavx2 -mfma -ffast-math part2b_dot_product.c -o part2b_dot_product && ./part2b_dot_product
    nvcc -O3 part3_matmul.cu -o part3_matmul && ./part3_matmul
    gcc -O3 -fopenmp part4_reduction.c -o part4_reduction && ./part4_reduction
    python3 plot_results.py

Capturing required screenshots:
  1. vectorization_report.txt:  gcc -O3 -mavx2 -mfma -ffast-math -fopt-info-vec-optimized part2b_dot_product.c -o /tmp/b 2> screenshots/vectorization_report.txt
  2. nvidia-smi                 -> screenshots/nvidia_smi.png
  3. nvcc --version             -> screenshots/nvcc_version.png
  4. figures/*.png              (auto-generated; copy to screenshots/ if required)
