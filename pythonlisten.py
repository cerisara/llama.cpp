import mmap
import os
import numpy as np
from posix_ipc import Semaphore, SharedMemory

SHM_NAME = "/ring_buffer_demo"
SEM_C2P = "/c2py_sem"
SEM_P2C = "/py2c_sem"

def loadEmbeddingMatrix():
    import numpy as np
    from gguf import GGUFReader

    GGUF_FILE_PATH = "/home/xtof/qwen2.5-0.5b-instruct-q5_k_m.gguf"
    try:
        reader = GGUFReader(GGUF_FILE_PATH)
        for tensor in reader.tensors:
            print(tensor.name,tensor.data.shape)
            if 'output' in tensor.name:
                return tensor.data
    except FileNotFoundError:
        print(f"Error: File not found at {GGUF_FILE_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

E = loadEmbeddingMatrix()

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
    # on modifie la derniere layer = result_norm, le dernier embedding
    print("debug",arr.shape,E.shape)
    arr[-1][:] = E[1000][:]

    # Notify C++
    sem_py2c.release()

