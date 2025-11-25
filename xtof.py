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

    def run(self):
        # this function is executed in a separate thread:
        # it listens to llamacpp and calls a ladder method when the forward pass has reached the last LLM layer
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
                print("now reading layer from shared buffer",i)
                vec = get_buffer_view()
                if len(vec)==0:
                    # when llamacpp quits, it warns this listener with an empty vector
                    fincpp = True
                    break
                # big activations (the ones from the LLM):
                actbig = np.array(vec, copy=True)
                # actbig = T x 896
                x = proj(torch.Tensor(actbig))
                # x   = T x d
                x = torch.Tensor(x)
                # we store the small activations (the ones for the ladder)
                acts.append(x)
                if i==2:
                    y = ladder.processActivations(acts)
                    # y = T x d
                    # inject the last activations modified by the ladder back into llamacpp
                    with torch.no_grad():
                        ybig = unproj(y)
                        ybig = ybig.numpy()
                        # we actually add the ladder activations to the LLM ones
                        ybig = actbig + ybig
                        # pass the new final embedding to llamacpp
                        vec[-1][:] = ybig[-1][:]
                print("gonna tell llamacpp to continue")
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

class Ladder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = torch.nn.ModuleList([torch.nn.Linear(d,d) for i in range(3)])
        self.mlp2 = torch.nn.ModuleList([torch.nn.Linear(d,d) for i in range(3)])
        self.opt = torch.optim.AdamW(self.parameters(), lr=0.001)
        # conservative init of the ladder
        with torch.no_grad():
            self.mlp2[-1].weight.copy_(0)
        self.dotrain=False

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

    def processActivations(self, acts):
        x = torch.stack(acts)
        # x = 3 x T x d
        # keep the output and the ladder computation graph for later training pass...
        self.y = self.forward(x)
        # y = T x d
        # and also keep the last small activations from llamacpp, because of the sum for training
        self.lastact = acts[-1]
        # lastact = T x d
        # ... but send to llamacpp the ladder output without its computation graph
        return self.y.detach()
 
    def train(self,toks):
        gld = torch.stack([E[t] for t in toks])
        # gld = T x d
        y = self.y + self.lastact
        # y = T x d

        y = y[:-1]
        gld = gld[1:]
        loss = torch.nn.functional.mse_loss(y,gld)
        print("LOSS",loss.item())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

def rollout(prompt, ntoks):
    os.system('rm -f layers2save')
    os.system('touch layers2save')
    os.system('echo "l_out-10" >> layers2save')
    os.system('echo "l_out-12" >> layers2save')
    os.system('echo "result_norm" >> layers2save')

    os.system("rm -rf detlog")
    os.system("mkdir detlog")
    s='./llama-cli --logdir detlog --temp 0.7 -c 2048 -nkvo -m '+modnom+' -p "'+prompt+'" -n '+str(ntoks)+' > log'
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
            elif l.startswith("prompt:"):
                pro = l[8:].strip()
            elif l.startswith("prompt_tokens:"):
                s = l[16:].strip()
                s = s[:-1]
                s = s.replace(',','')
                protoks = [int(x) for x in s.split(" ")] 
            elif l.startswith("output_tokens:"):
                s = l[16:].strip()
                s = s[:-1]
                s = s.replace(',','')
                reptoks = [int(x) for x in s.split(" ")]
                break
    return rep,reptoks,pro,protoks

 
# ===============================================

# toujours nettoyer les semaphores precedentes avant de relancer llamacpp et SharedMem
os.system('rm /dev/shm/sem.py2c_sem')
os.system('rm /dev/shm/sem.c2py_sem')
sm = SharedMem()
sm.start()

# random projection between llamacpp and ladder: it's not trained at all!
llm2d = torch.Tensor(896,d)
E = loadUnembeddings()
# the unembedding matrix, projected into the ladder space, is useful to train the ladder
E = proj(E)

# ladder listens to llamacpp and get from it the activations
ladder = Ladder()

p = "What comes after 8? Answer: "
s,toks,pro,protoks = rollout(p,1)
print("prompt",pro)
print("protok",protoks)
print("rollout",s)
print("rolltok",toks)

# do not train when generating response
if ladder.y.shape[0]>1:
    # at train time, we do not care about the single generated token, we train on the rollout
    ladder.train(protoks)

