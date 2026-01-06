from mpi4py import MPI
import numpy as np
Ns = [3, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200,
      1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000,
      3000,4000,5000,6000,7000,8000]
all_times = []
# calc time
for N in Ns:
    start = MPI.Wtime()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()



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
    #print(f"Process {rank} received rows of A:\n{local_A.shape}")
    local_C = np.dot(local_A, B)
    # print(f"Process {rank} computed local C:\n{local_C.shape}")
    # Gather the computed local_C matrices back to process 0
    comm.Gatherv(local_C, [C, tuple(c * N for c in counts), tuple(s * N for s in displs), MPI.DOUBLE], root=0)

    if rank == 0:
        end = MPI.Wtime()
        print(f"Time taken for matrix multiplication of size {N}x{N}: {end - start} seconds")
        all_times.append(end - start)
        #print("Resultant matrix C:\n", C)
        print("-" * 40)
# Print all times
if rank == 0:
    print("All times for different matrix sizes:")
    print(all_times)