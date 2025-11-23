import mmap
import os
import numpy as np
from posix_ipc import Semaphore, SharedMemory

SHM_NAME = "/ring_buffer_demo"
SEM_C2P = "/c2py_sem"
SEM_P2C = "/py2c_sem"

with open("detembeds.dims","r") as f:
    dims = f.readlines()
    dims = [int(d) for d in dims]
E = np.fromfile("detembeds.bin", dtype=np.float32)
E.shape = dims[-2:]
print(E.shape)
for i in range(10): print(E[0,i])
exit(1)

# Open shared memory
fd = os.open("/dev/shm" + SHM_NAME, os.O_RDWR)
mm = mmap.mmap(fd, 1000000)
buf = memoryview(mm)

# Open semaphores
sem_c2p = Semaphore(SEM_C2P)
sem_py2c = Semaphore(SEM_P2C)


def get_buffer_view():
    start = 0
    mv = buf[start : start + 4]
    start += 4
    ne1 = int(np.frombuffer(mv, dtype=np.float32))
    mv = buf[start : start + 4]
    start += 4
    ne0 = int(np.frombuffer(mv, dtype=np.float32))
    print("nnn",ne1,ne0)
    mv = buf[start : start + 4*ne0*ne1]
    vec = np.frombuffer(mv, dtype=np.float32)
    vec.shape = (ne1,ne0)
    print("nnv",vec.shape)
    for i in range(ne1): print(vec[i][0])
    return vec

while True:
    # on charge 3 layers
    for i in range(3):
        # Wait for C++ to fill buffer
        sem_c2p.acquire()
        arr = get_buffer_view()
        if i==2 and E!=None: arr[-1][:] = E[1000][:]
        sem_py2c.release()

    if E==None:
        # on charge encore les embeddings
        sem_c2p.acquire()
        E = get_buffer_view()
        sem_py2c.release()
        print("embed",E.shape)


