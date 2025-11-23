import mmap
import os
import numpy as np
import torch
from posix_ipc import Semaphore, SharedMemory

d=128

SHM_NAME = "/ring_buffer_demo"
SEM_C2P = "/c2py_sem"
SEM_P2C = "/py2c_sem"

def loadUnembeddings():
    with open("detembeds.dims","r") as f:
        dims = f.readlines()
        dims = [int(x) for x in dims]
    E = np.fromfile("detembeds.bin", dtype=np.float32)
    E.shape = dims[-2:]
    print("embeddings",E.shape)
    return E

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

# random projection
llm2d = torch.Tensor(896,d)
class Ladder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = torch.nn.ModuleList([torch.nn.Linear(d,d) for i in range(3)])
        self.mlp2 = torch.nn.ModuleList([torch.nn.Linear(d,d) for i in range(3)])
        # conservative init of the ladder
        with torch.no_grad():
            self.mlp2[-1].weight.copy_(0)

    def forward(self,llmx):
        # llmx = B x 3 x llmd
        print("deb1",llmx.shape)
        with torch.no_grad():
            b,t,llmd = llmx.shape
            x = llmx.view(-1,llmd) @ llm2d
            x = x.view(b,t,d)
        print("debZ",x.shape)
        x = x.transpose(1,2)
        # x = B x d x 3
        z = 0
        for i in range(3):
            v = self.mlp1[i](x[:,:,i])
            # v = B x d
            v = torch.nn.functional.relu(v)
            v = self.mlp2[i](v)
            z = z + v
        y = z @ llm2d.transpose(0,1)
        return y

ladder = Ladder()
while True:
    # on charge 3 layers
    acts = []
    for i in range(3):
        # Wait for C++ to fill buffer
        sem_c2p.acquire()
        print("now reading shared buffer\n")
        vec = get_buffer_view()
        act = np.array(vec, copy=True)
        # act = T x 896
        x = torch.Tensor(act[-1])
        acts.append(x)
        print("detact",act.shape,len(acts))
        if i==2:
            x = torch.stack(acts).view(1,3,-1)
            with torch.no_grad():
                y = ladder(x)
            y = acts[-1] + y
            y = y.numpy()
            print("debfin",y.shape)
            # pass the new final embedding to llamacpp
            vec[-1][:] = y[:]
        print("gonna tell llamacpp to continue\n")
        sem_py2c.release()

