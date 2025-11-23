import mmap
import os
import numpy as np
import torch
from posix_ipc import Semaphore, SharedMemory

d=128
dotrain=True

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
    return torch.Tensor(E)

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
def proj(x):
    # x = ... x llmd
    with torch.no_grad():
        y = x @ llm2d
    return y
def unproj(x):
    # x = ... x d
    with torch.no_grad():
        y = x @ llm2d.transpose(0,1)
    return y

E = loadUnembeddings()
E = proj(E)

def loadTokens():
    with open("dettoks.txt","r") as f: lines = f.readlines()
    toks = []
    ss = ' '.join(lines).split(':')
    for i in range(1,len(ss)-1):
        j=0
        while j<len(ss[i]) and ss[i][j].isdigit(): j+=1
        if j<len(ss[i]) and ss[i][j]==',':
            toks.append(int(ss[i][:j]))
    i,j = len(ss)-1,0
    while j<len(ss[i]) and ss[i][j].isdigit(): j+=1
    toks.append(int(ss[i][:j]))
    return toks

class Ladder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = torch.nn.ModuleList([torch.nn.Linear(d,d) for i in range(3)])
        self.mlp2 = torch.nn.ModuleList([torch.nn.Linear(d,d) for i in range(3)])
        self.opt = torch.optim.AdamW(self.parameters(), lr=0.001)
        # conservative init of the ladder
        with torch.no_grad():
            self.mlp2[-1].weight.copy_(0)

    def forward(self,x):
        # x = 3 x B x d
        z = 0
        for i in range(3):
            v = self.mlp1[i](x[i])
            # v = B x d
            v = torch.nn.functional.relu(v)
            v = self.mlp2[i](v)
            z = z + v
        # z = B x d
        return z

    def train(self,y,toks,lastact):
        # y = T x d
        # lastact = T x d
        gld = torch.stack([E[t] for t in toks])
        # gld = T x d
        y = y + lastact

        y = y[:-1]
        gld = gld[1:]
        loss = torch.nn.functional.mse_loss(y,gld)
        print("LOSS",loss.item())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

ladder = Ladder()
while True:
    # on charge 3 layers
    acts = []
    for i in range(3):
        # Wait for C++ to fill buffer
        sem_c2p.acquire()
        print("now reading shared buffer\n")
        vec = get_buffer_view()
        actbig = np.array(vec, copy=True)
        # actbig = T x 896
        x = proj(torch.Tensor(actbig))
        # x   = T x d
        x = torch.Tensor(x)
        acts.append(x)
        if i==2:
            toks = loadTokens()
            x = torch.stack(acts)
            # x = 3 x T x d
            y = ladder(x)
            # y = T x d
            if dotrain and y.shape[0]>1:
                # do not train when generating response
                ladder.train(y,toks,acts[-1])

            with torch.no_grad():
                ybig = unproj(y)
                ybig = ybig.numpy()
                ybig = actbig + ybig
                # pass the new final embedding to llamacpp
                vec[-1][:] = ybig[-1][:]
        print("gonna tell llamacpp to continue\n")
        sem_py2c.release()

