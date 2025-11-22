import mmap
import os
import numpy as np
from posix_ipc import Semaphore, SharedMemory

SHM_NAME = "/ring_buffer_demo"
SEM_C2P = "/c2py_sem"
SEM_P2C = "/py2c_sem"

N = 8
ITER = 10

# Open shared memory
fd = os.open("/dev/shm" + SHM_NAME, os.O_RDWR)
mm = mmap.mmap(fd, 2 * N * 4)  # 2 buffers, float32
buf = memoryview(mm)

# Open semaphores
sem_c2p = Semaphore(SEM_C2P)
sem_py2c = Semaphore(SEM_P2C)

def get_buffer_view(b):
    start = b * N * 4
    mv = buf[start : start + N * 4]
    return np.frombuffer(mv, dtype=np.float32)

for it in range(ITER):
    b = it % 2

    # Wait for C++ to fill buffer
    sem_c2p.acquire()

    arr = get_buffer_view(b)
    print(f"[Python] Got buffer {b}: {arr.copy()}")

    arr += 1.0
    print(f"[Python] Sending back: {arr.copy()}")

    # Notify C++
    sem_py2c.release()

