import mmap
import os
import numpy as np
import torch
from posix_ipc import Semaphore, SharedMemory
import time
from os import listdir
from os.path import isfile, join
import threading

d=128

SHM_NAME = "/ring_buffer_demo"
SEM_C2P = "/c2py_sem"
SEM_P2C = "/py2c_sem"
modnom="/home/xtof/qwen2.5-0.5b-instruct-q5_k_m.gguf"

def loadUnembeddings():
    with open("detembeds.dims","r") as f:
        dims = f.readlines()
        dims = [int(x) for x in dims]
    E = np.fromfile("detembeds.bin", dtype=np.float32)
    E.shape = dims[-2:]
    print("embeddings",E.shape)
    return torch.Tensor(E)

class SharedMem(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.dotrain=False

    def run(self):
        print("sharedmem thread started")
        # Open shared memory
        self.fd = os.open("/dev/shm" + SHM_NAME, os.O_RDWR)
        self.mm = mmap.mmap(self.fd, 4*1000000) # 4 because in C++ the size is given in float32!
        self.buf = memoryview(self.mm)
        while True:
            # wait for C++ to create the semaphores
            try:
                self.sem_c2p = Semaphore(SEM_C2P)
                self.sem_py2c = Semaphore(SEM_P2C)
                break
            except: pass
            time.sleep(1)
        print("sharedmem thread detected semaphores")

        # OK, now listen to llamacpp activations
        fincpp = False
        while not fincpp:
            # on charge 3 layers
            acts = []
            for i in range(3):
                # Wait for C++ to fill buffer
                sm.sem_c2p.acquire()
                print("now reading shared buffer\n")
                vec = get_buffer_view()
                if len(vec)==0:
                    fincpp = True
                    break
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
                    if self.dotrain and y.shape[0]>1:
                        # do not train when generating response
                        ladder.train(y,toks,acts[-1])

                    with torch.no_grad():
                        ybig = unproj(y)
                        ybig = ybig.numpy()
                        ybig = actbig + ybig
                        # pass the new final embedding to llamacpp
                        vec[-1][:] = ybig[-1][:]
                print("gonna tell llamacpp to continue\n")
                sm.sem_py2c.release()
 
def get_buffer_view():
    global sm
    start = 0
    mv = sm.buf[start : start + 4]
    start += 4
    ne1 = np.frombuffer(mv, dtype=np.float32)
    if ne1==424242: return []
    ne1 = int(ne1)
    mv = sm.buf[start : start + 4]
    start += 4
    ne0 = int(np.frombuffer(mv, dtype=np.float32))
    mv = sm.buf[start : start + 4*ne0*ne1]
    vec = np.frombuffer(mv, dtype=np.float32)
    vec.shape = (ne1,ne0)
    for i in range(ne1): print(vec[i][0])
    return vec

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

def readargs():
    while True:
        while True:
            try:
                with open("pyargs.txt","r") as f: lines=f.readlines()
                break
            except: pass
            print("wait for pyargs.txt...")
            time.sleep(1)
        if lines[0].startswith("dotrain"): return "dotrain"
        elif lines[0].startswith("quit"): return "quit"
        elif lines[0].startswith("notrain"): return "notrain"
        print("pyargs.txt not wellformed yet...")
        time.sleep(1)

def rollout(prompt):
    os.system('rm -f layers2save')
    os.system('touch layers2save')
    os.system('echo "l_out-10" >> layers2save')
    os.system('echo "l_out-12" >> layers2save')
    os.system('echo "result_norm" >> layers2save')

    os.system("rm -rf detlog")
    os.system("mkdir detlog")
    s='./llama-cli --logdir detlog --temp 0.7 -c 2048 -nkvo -m '+modnom+' -p "'+prompt+'" -n 20 > log'
    print("run llama",s)
    os.system(s)
    print("llama done")
    # quand le rollout est fini, llamacpp envoit un "quit" a ma SharedMem

    mypath = "detlog/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    with open(mypath+onlyfiles[0],"r") as f:
        for l in f:
            if l.startswith("output:"):
                rep = l[8:].strip()
            elif l.startswith("output_tokens:"):
                s = l[16:].strip()
                s = s[:-1]
                s = s.replace(',','')
                reptoks = [int(x) for x in s.split(" ")]
                break
    return rep,reptoks

 
# ===============================================

# toujours nettoyer les semaphores precedentes avant de relancer llamacpp et SharedMem
os.system('rm /dev/shm/sem.py2c_sem')
os.system('rm /dev/shm/sem.c2py_sem')
sm = SharedMem()
sm.start()
# random projection
llm2d = torch.Tensor(896,d)
E = loadUnembeddings()
E = proj(E)
ladder = Ladder()

p = "What comes after 8? Answer: "
s,toks = rollout(p)
print("rollout",s)
print("rolltok",toks)


