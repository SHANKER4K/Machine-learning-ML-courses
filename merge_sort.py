import mpi4py.MPI as mpi

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Append leftovers
    result.extend(left[i:])
    result.extend(right[j:])

    return result

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = [38, 27, 43, 3, 9, 82, 10]
else:
    chunks = None

chunks = comm.scatter([data[i::size] for i in range(size)] if rank == 0 else None, root=0)
sorted_arr = merge_sort(chunks)
full_arr = comm.gather(sorted_arr, root=0)
if rank == 0:
    full_arr = merge_sort([item for sublist in full_arr for item in sublist])
    print("Sorted array:", full_arr)