CC      = gcc
NVCC    = nvcc
CFLAGS  = -O3 -mavx2 -mfma -fopenmp
NVFLAGS = -O3

all: part2b_dot_product part3_matmul part4_reduction

part2b_dot_product: part2b_dot_product.c
	$(CC) $(CFLAGS) -ffast-math $< -o $@

part3_matmul: part3_matmul.cu
	$(NVCC) $(NVFLAGS) $< -o $@

part4_reduction: part4_reduction.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f part2b_dot_product part3_matmul part4_reduction

.PHONY: all clean
