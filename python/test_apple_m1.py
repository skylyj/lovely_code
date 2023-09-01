import torch
import time

# Define the size of the matrices
matrix_size = 19999

# Generate random matrices for multiplication
A = torch.rand(matrix_size, matrix_size, device='mps')
B = torch.rand(matrix_size, matrix_size, device='mps')

# Use Apple M1's MPS for matrix multiplication
#start_time = time.time()
#C = torch.matmul(A, B)
#gpu_time = time.time() - start_time

#print("GPU (MPS) time:", gpu_time)
import timeit
x = timeit.timeit(lambda: torch.matmul(A, B), number=1)
print("x GPU (MPS) time:", x)



# Generate random matrices for multiplication
A1 = torch.rand(matrix_size, matrix_size)
B1 = torch.rand(matrix_size, matrix_size)

# Use Apple M1's MPS for matrix multiplication
#start_time = time.time()
#C1 = torch.matmul(A1, B1)
#cpu_time = time.time() - start_time
#print("CPU (MPS) time:", cpu_time)
y = timeit.timeit(lambda: torch.matmul(A1, B1), number=1)
print("y CPU (MPS) time:", y)

