from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 50

if rank == 0:
    A = np.ones((N, N), dtype='d')
    B = np.ones((N, N), dtype='d')
    C = np.zeros((N, N), dtype='d')
else:
    A = None
    B = None
    C = None
    
# Broadcast matrix B to all processes
B = comm.bcast(B, root=0)
# Scatter rows of matrix A to all processes

base = N // size
extra = N % size
counts = [base + 1 if i < extra else base for i in range(size)]
displs = np.zeros(size, dtype=int)
displs[1:] = np.cumsum(counts)[:-1]

local_A = np.zeros((counts[rank], N), dtype='d')
comm.Scatterv([A,tuple(c * N for c in counts), tuple(s * N for s in displs),MPI.DOUBLE], local_A, root=0)
print(f"Process {rank} received rows of A:\n{local_A.shape}")
local_C = np.dot(local_A, B)
# print(f"Process {rank} computed local C:\n{local_C.shape}")
# Gather the computed local_C matrices back to process 0
comm.Gatherv(local_C, [C, tuple(c * N for c in counts), tuple(s * N for s in displs), MPI.DOUBLE], root=0)

if rank == 0:
    print("Resultant matrix C:\n", C)