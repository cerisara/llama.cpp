import mmap, os, struct, time
import numpy as np

SHM_NAME = "/dev/shm/ring_buffer_demo"
FD_FILE  = "/dev/shm/ring_buffer_fds"

N = 8
ITER = 10
FLOAT_SZ = 4

# Load eventfd numbers from file created by C++
with open(FD_FILE, "r") as f:
    ev_c2py, ev_py2c = map(int, f.read().split())
print("OK",ev_c2py,ev_py2c)

# Open shared memory
fd = os.open(SHM_NAME, os.O_RDWR)
size = 2 * N * FLOAT_SZ
mm = mmap.mmap(fd, size)

buf = memoryview(mm)

def get_buffer_view(b):
    start = b * N * FLOAT_SZ
    mv = buf[start : start + N * FLOAT_SZ]
    return np.frombuffer(mv, dtype=np.float32)

for it in range(ITER):
    b = it % 2

    # Wait for C++ → Python event
    os.read(ev_c2py, 8)     # blocking read

    arr = get_buffer_view(b)
    print(f"[Python] got iter {it} buf {b}: {arr.copy()}")

    arr += 1.0

    print(f"[Python] send back: {arr.copy()}")

    # Notify C++ (Python → C++)
    os.write(ev_py2c, struct.pack("Q", 1))

