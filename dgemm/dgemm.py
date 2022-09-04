import numpy as np
import time

SIZE = 2048

# A = np.ones((SIZE, SIZE))
# B = np.ones((SIZE, SIZE))
A = np.ones((SIZE, SIZE), dtype=np.float64)
B = np.ones((SIZE, SIZE), dtype=np.float64)
C = np.ones((SIZE, SIZE), dtype=np.float64)

alpha = 2.0
beta = 2.0

start = time.time()
C = alpha * np.matmul(A, B) + beta * C
end = time.time()

print("DGEMM time (numpy):", (end-start)*1000, "ms")

start = time.time()
sum = np.sum(C)
end = time.time()

print("Sum time:", (end-start)*1000, "ms")
print("Result:", sum)