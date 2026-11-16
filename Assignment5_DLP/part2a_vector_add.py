"""
Part 2.1 - Vector-architecture simulation via NumPy.
Compares a scalar Python loop with a NumPy vectorized operation.
Writes results to results/results_vector.csv for plotting.
"""
import numpy as np
import time
import csv
import os

N = 10_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# Scalar-style Python loop
start = time.perf_counter()
c_scalar = np.empty(N, dtype=np.float32)
for i in range(N):
    c_scalar[i] = a[i] + b[i]
scalar_time = time.perf_counter() - start

# Vectorized NumPy operation (uses SIMD under the hood)
start = time.perf_counter()
c_vec = a + b
vec_time = time.perf_counter() - start

# Verify correctness
assert np.allclose(c_scalar, c_vec, atol=1e-6)

print(f"Scalar loop time: {scalar_time:.6f} s")
print(f"Vectorized time:  {vec_time:.6f} s")
print(f"Speedup:           {scalar_time / vec_time:.2f}x")

# Save CSV
os.makedirs("results", exist_ok=True)
with open("results/results_vector.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["implementation", "time_seconds"])
    w.writerow(["scalar_python_loop", scalar_time])
    w.writerow(["numpy_vectorized", vec_time])
