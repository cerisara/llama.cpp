import mmap, os, struct, time
import numpy as np

SHM_NAME = "/dev/shm/ring_buffer_demo"
N = 8
ITER = 10

# Memory layout sizes
INT_SZ = 4
FLOAT_SZ = 4

# Layout:
# c_to_py_flag[2]     = 8 bytes
# py_to_c_flag[2]     = 8 bytes
# buffers[2][N] floats = 2 * N * 4 bytes
offset_c2p = 0
offset_p2c = offset_c2p + 2 * INT_SZ
offset_buffers = offset_p2c + 2 * INT_SZ

total_size = offset_buffers + 2 * N * FLOAT_SZ

fd = os.open(SHM_NAME, os.O_RDWR)
mm = mmap.mmap(fd, total_size)
buf = memoryview(mm)

def read_int(offset):
    return struct.unpack_from("i", buf, offset)[0]

def write_int(offset, val):
    struct.pack_into("i", buf, offset, val)

def get_buffer_view(b):
    """Return numpy float32 array pointing into shared memory."""
    start = offset_buffers + b * N * FLOAT_SZ
    mv = buf[start : start + N * FLOAT_SZ]
    return np.frombuffer(mv, dtype=np.float32)

for it in range(ITER):
    b = it % 2

    # Wait for C++ to fill this buffer
    while read_int(offset_c2p + b*INT_SZ) == 0:
        time.sleep(0.001)

    arr = get_buffer_view(b)
    print(f"[Python] Iter {it} got buffer {b}: {arr.copy()}")

    # Add +1 to each element
    arr += 1.0
    print(f"[Python] Sending back: {arr.copy()}")

    # Mark result ready for C++
    write_int(offset_p2c + b*INT_SZ, 1)
    write_int(offset_c2p + b*INT_SZ, 0)  # reset C→ PY flag


