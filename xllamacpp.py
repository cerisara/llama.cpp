# This program launches llama.cpp and connects to it through a shared RAM memory zone and 2 semaphores
# It exposes a method to run a rollout on a sentence and get the latent activations from llama.cpp
# An example method saveActivations() shows at the end how to use this code

# WARNING: with MoE, the prefill may be splitted into several chunks!

# Environment variables to control the program:
# SAVE_EMB=result_output to save the embeddings
# SHOW_ACTIVS=1 to show the full stack of layers names from the model

import os
import sys
import mmap
import gzip
import subprocess
import numpy as np
from posix_ipc import Semaphore, SharedMemory
import time
import threading
import signal
import requests

NTOKS2GEN = int(os.environ.get("NTOKS", 5))
PORT = "8257"
SHM_NAME = "/ring_buffer_demo"
SEM_C2P = "/c2py_sem"
SEM_P2C = "/py2c_sem"
connLayers = ['l_out-16','norm']

# model registry: (path, is_moe). Pick one by editing the selection line below
MODELS = [
    ("/home/xtof/Qwen3-8B-Q5_K_M.gguf", False),
    ("/home/xtof/ggufs/qwen2.5-0.5b-instruct-q5_k_m.gguf", False),
    ("/home/xtof/ggufs/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", True),
    ("/home/xtof/ggufs/Qwen3.5-9B-Q4_K_M.gguf", False),
]
modnom, is_moe = MODELS[3]

# SAVE_EMB dumps the unembedding matrix (detembeds.bin). The dump hook only
# works when that tensor is whole and in host memory: run all layers on CPU
# (ngl 0) and avoid GPU/CPU splits. Otherwise offload greedily.
save_emb = bool(os.environ.get("SAVE_EMB"))
if is_moe:
    OPTS = ["--embeddings", "--no-mmap", "--temp", "0"]
    if not save_emb:
        # push the experts of the first 30 layers to RAM to fit the 35B on GPU
        OPTS += ["--n-gpu-layers", "999", "--n-cpu-moe", "30"]
    else:
        OPTS += ["--n-gpu-layers", "0"]
else:
    OPTS = ["--embeddings", "-ngl", "0" if save_emb else "99", "--temp", "0"]

class DummyHandler:
    # consumes and prints every activation sent by llama.cpp while it is
    # initializing (warmup or leftovers from a previous run).
    def processActivations(self, actbig, i):
        print("DUMMY consuming activation "+str(i)+" shape="+str(actbig.shape)+" first="+str(actbig.flatten()[:5]))
        return None # do not modify activations

class SharedMem(threading.Thread):
    def __init__(self, nLayersGot, activsHandler, listening_event):
        self.nLayersGot = nLayersGot
        self.activsHandler = activsHandler
        self.listening_event = listening_event
        threading.Thread.__init__(self)

    def get_handler(self):
        # use the dummy handler until llama.cpp is really ready (listening text
        # seen), then switch to the real handler from main and drop the dummy
        if self.listening_event.is_set():
            return self.activsHandler
        return DummyHandler()

    def run(self):
        # this function is executed in a separate thread:
        # it listens to llamacpp and calls a ladder method when the forward pass has reached the last LLM layer
        print("sharedmem thread started")
        while True:
            # wait for llama-server to create the shared memory file
            try:
                self.fd = os.open("/dev/shm" + SHM_NAME, os.O_RDWR)
                break
            except FileNotFoundError:
                pass
            time.sleep(1)
        # ensure the shared file is at least as big as the C++ buffer, even if a
        # stale /dev/shm/ring_buffer_demo from a previous run still exists
        os.ftruncate(self.fd, 4*70000000)
        self.mm = mmap.mmap(self.fd, 4*70000000) # 4 because in C++ the size is given in float32!
                                           # must match SharedMemory.buffers in tools/server/server.cpp
        self.buf = memoryview(self.mm)
        while True:
            # wait for C++ to create the semaphores
            try:
                self.sem_c2p = Semaphore(SEM_C2P)
                self.sem_py2c = Semaphore(SEM_P2C)
                break
            except: pass
            time.sleep(1)
        print("sharedmem thread detected semaphores; now listen to llamacpp activations")

        fincpp = False
        i = 0
        while not fincpp:
            # Wait for C++ to fill buffer
            print('python wait layer',i)
            self.sem_c2p.acquire()
            print("now reading layer from shared buffer",i)
            vec = self.get_buffer_view()

            if vec is None:
                # llamacpp is quitting: it warns this listener by writing the
                # sentinel magic value 424242
                fincpp = True
                print("llamacpp stopped; ending the listener loop")
                break

            if len(vec)==0:
                print("python got empty c++ vector")
            else:
                # big activations (the ones from the LLM):
                actbig = np.array(vec, copy=True)
                # actbig = T x 896
                # use dummy handler until llama.cpp is really ready, then real one
                handler = self.get_handler()
                y = handler.processActivations(actbig, i % self.nLayersGot)
                if not y is None:
                    # overwrite the activations into llama.cpp
                    for j in range(len(vec)):
                        for k in range(len(vec[j])):
                            vec[j][k] = y[j][k]
            print("gonna tell llamacpp to continue")
            self.sem_py2c.release()
            i += 1
        print("xllamacpp stopping; removing semaphores1")
        os.system('rm /dev/shm/sem.py2c_sem')
        os.system('rm /dev/shm/sem.c2py_sem')
 
    def get_buffer_view(self):
        start = 0
        mv = self.buf[start : start + 4]
        start += 4
        ne1 = np.frombuffer(mv, dtype=np.float32)
        ne1 = ne1.item()
        if ne1==424242: return None
        ne1 = int(ne1)
        mv = self.buf[start : start + 4]
        start += 4
        ne0 = int(np.frombuffer(mv, dtype=np.float32).item())
        mv = self.buf[start : start + 4*ne0*ne1]
        vec = np.frombuffer(mv, dtype=np.float32)
        vec.shape = (ne1,ne0)
        # for i in range(ne1): print(vec[i][0])
        return vec

    def detokenize(self, toks):
        url = "http://localhost:"+PORT+"/detokenize"
        data = { "tokens": [int(x) for x in toks] }
        headers = { "Content-Type": "application/json" }
        response = requests.post(url, json=data, headers=headers)
        print("Status code:", response.status_code)
        sout = response.json()
        print("detoken", sout.keys())
        u = sout['content'] 
        return u
 
    def tokenize(self, prompt):
        url = "http://localhost:"+PORT+"/tokenize"
        data = { "content": prompt }
        headers = { "Content-Type": "application/json" }
        response = requests.post(url, json=data, headers=headers)
        sout = response.json()
        protoks = sout['tokens'] 
        return protoks
 
    def rollout_gen(self, prompt):
        url = "http://localhost:"+PORT+"/completions"
        data = {"prompt" : prompt, "return_tokens": True, "cache_prompt": False,
                "n_predict": NTOKS2GEN}
        headers = { "Content-Type": "application/json" }
        print("sending prompt to llama.cpp completions")
        # print the token ids of the prompt, as the completions response only
        # returns the generated tokens
        ptoks = self.tokenize(prompt)
        print("PROMPT_TOKENS", ptoks)
        response = requests.post(url, json=data, headers=headers)
        sout = response.json()
        print("Status code:", response.status_code)
        print("Response body:", len(sout))
        if str(response.status_code)[0]=="2" and sout!=None: return sout['content'], sout['tokens']
        return None
 
    def rollout_train(self, prompt):
        # j'ai check que l'API embeddings retourne le meme vecteur d'embeddings que celui obtenu en dernier via l'API completions
        url = "http://localhost:"+PORT+"/embeddings"
        data = {"content" : prompt, "return_tokens": True, "cache_prompt": False}
        headers = { "Content-Type": "application/json" }
        print("sending prompt to llama.cpp embeddings")
        response = requests.post(url, json=data, headers=headers)
        sout = response.json()
        print("Status code:", response.status_code)
        print("Response body:", len(sout))
        if str(response.status_code)[0]!="2": return None
        # print("ddd",sout[0].keys())
        # print("jjj",np.array(sout[0]['embedding']).shape)
        # print("kkk",sout[0]['embedding'][-1][:10])
        # reptoks = sout['tokens']
        # rep = sout['content']
        # return rep,reptoks
        # return sout[0]['embedding']
     
class AsyncScriptRunner:
    def __init__(self, script_path, *args, env=None, notify_event=None):
        self.script_path = script_path
        self.args = args
        self.env = env or {}
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.wait_event = threading.Event()  # Event to signal when text is found
        self.notify_event = notify_event     # extra event to signal the shared mem thread
        self.trigger_text = None

    def _read_stream(self, stream, name):
        """Read a stream line by line."""
        for line in iter(stream.readline, ''):
            if line:
                line = line.strip()
                print(f"[{name}] {line}")
                # Check if the line contains the trigger text
                if self.trigger_text and self.trigger_text in line:
                    self.wait_event.set()
                    if self.notify_event:
                        self.notify_event.set()
        stream.close()

    def start(self, wait_for_text=None):
        """Start the script asynchronously."""
        self.trigger_text = wait_for_text

        creationflags = 0
        preexec_fn = None
        if os.name != 'nt':  # Unix
            preexec_fn = os.setsid
        else:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.process = subprocess.Popen(
            [self.script_path, *self.args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=preexec_fn,
            creationflags=creationflags,
            env={**os.environ, **self.env}
        )

        # Start threads to read stdout and stderr
        self.stdout_thread = threading.Thread(target=self._read_stream, args=(self.process.stdout, "stdout"))
        self.stderr_thread = threading.Thread(target=self._read_stream, args=(self.process.stderr, "stderr"))

        self.stdout_thread.start()
        self.stderr_thread.start()

        print("Script started asynchronously.")

        # If wait_for_text is set, block main thread until text appears
        if wait_for_text:
            print(f"Waiting for output containing: '{wait_for_text}' ...")
            self.wait_event.wait()
            print(f"Detected '{wait_for_text}' in output, main thread resumes.")

    def stop(self):
        """Terminate the script safely."""
        if self.process:
            print("Stopping script...")
            try:
                if os.name != 'nt':  # Unix
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:  # Windows
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process did not terminate, killing it...")
                self.process.kill()
            finally:
                self.stdout_thread.join()
                self.stderr_thread.join()
                print("Script stopped.")
 
def initLlamacpp(llamacppdir, connectedCPPLayers, activsHandler, nLayersGot=None):
    # TODO 1 check with a first dummy llama-server run that does not capture any
    # activation but that just prints all model layers that the strings in
    # connectedCPPLayers all belong to the printed model layers (and also try to
    # guess what is the last output norm layer ??)
    if nLayersGot is None:
        nLayersGot = len(connectedCPPLayers)

    # toujours nettoyer les semaphores precedentes avant de relancer llamacpp et SharedMem
    print("removing semaphores0")
    os.system('rm /dev/shm/sem.py2c_sem')
    os.system('rm /dev/shm/sem.c2py_sem')

    # set once llama.cpp prints the listening text: then SharedMem switches
    # from the dummy handler to the real one
    listening_event = threading.Event()

    sm = SharedMem(nLayersGot, activsHandler, listening_event)
    sm.start()
    os.system('rm -f layers2save')
    os.system('touch layers2save')
    for l in connectedCPPLayers: os.system('echo "'+l+'" >> layers2save')

    runner = AsyncScriptRunner(llamacppdir+"/build/bin/llama-server","-m",modnom,"--no-webui",
                               "--no-warmup","--ctx-size","30000","--cache-ram", "0", 
                               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-nkvo", 
                               "--port", PORT, *OPTS,
                               env={"XHOW_ACTIVS": "1"},
                               notify_event=listening_event)
    # llama.cpp does not print "all slots are idle" with --no-warmup; it prints
    # "llama_server: listening on http://..." when it is really ready. While
    # waiting we keep the SharedMem thread running, so any activations that
    # llama.cpp sends during its own warmup or from a previous run are still
    # captured (and discarded by activsHandler) instead of blocking llama.cpp.
    runner.start(wait_for_text="llama_server: listening on http://")
    print("python main thread continues")
    time.sleep(1)
    return sm, runner

class ActivsSaver:
    # streams each activation tensor to disk as its own gzip-compressed chunk,
    # so only one tensor is ever held in RAM at a time.
    # on-disk format per chunk:
    #   int32 ndim, int32[ndim] shape, int64 len(compressed), gzip bytes
    def __init__(self, outfile):
        self.outfile = outfile
        self.n = 0
        self.f = open(outfile, "ab")

    def save_one(self, actbig):
        shape = np.asarray(actbig.shape, dtype=np.int32)
        comp = gzip.compress(np.ascontiguousarray(actbig, dtype=np.float32).tobytes())
        self.f.write(np.array([len(shape)], dtype=np.int32).tobytes())
        self.f.write(shape.tobytes())
        self.f.write(np.array([len(comp)], dtype=np.int64).tobytes())
        self.f.write(comp)
        self.f.flush()
        self.n += 1

    def close(self):
        self.f.close()
        print("saved "+str(self.n)+" activations to "+self.outfile)

def load_activs(infile):
    # reads back the chunks written by ActivsSaver, in order
    activs = []
    with open(infile, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) == 0: break
            ndim = np.frombuffer(head, dtype=np.int32).item()
            shape = np.frombuffer(f.read(4*ndim), dtype=np.int32).tolist()
            nbytes = np.frombuffer(f.read(8), dtype=np.int64).item()
            data = np.frombuffer(gzip.decompress(f.read(nbytes)), dtype=np.float32)
            data.shape = shape
            activs.append(data)
    return activs

def saveActivations(prompts_file):
    class ActivsHandler:
        def __init__(self, outfile):
            # the real activs handler only runs after llama.cpp is listening,
            # so the dummy activations captured during startup are not saved
            self.processed_once = threading.Event()
            self.n = 0
            self.outfile = outfile
            self.saver = ActivsSaver(outfile)

        def processActivations(self, actbig, i):
            self.n += 1
            if actbig.shape[0]>0:
                print("saving activation "+str(i)+" shape="+str(actbig.shape))
                self.saver.save_one(actbig)
                self.processed_once.set()
            else:
                print("WARNING: do not save activations "+str(actbig.shape))
            return None # do not modify activations

        def save(self):
            self.saver.close()

    with open(prompts_file) as f:
        prompts = [line.rstrip('\n') for line in f if line.strip()]
        prompts = ['\n'.join(prompts)]
        print("promptlen",len(prompts),len(prompts[0]))

    outfile = os.path.splitext(os.path.basename(prompts_file))[0] + "_activs.npz"
    handler = ActivsHandler(outfile)
    # WARNING: the last element MUST BE the last layer, just before the
    # unembedding matrix, ie. after the last global norm
    sharedRAM, procCPP = initLlamacpp("./", connLayers, handler)

    print("Triggering rollouts to get activations...")
    for utt in prompts:
        print("Prompt:", utt)
        if NTOKS2GEN==0: s=sharedRAM.rollout_train(utt)
        else: s=sharedRAM.rollout_gen(utt)
        if not s==None:
            print("GEN",s[0])
            print("TOKGEN",s[1])
        time.sleep(1)

    print("Waiting for activations to be processed...")
    processed = handler.processed_once.wait(timeout=10) # Wait for 10 seconds max
    if not processed: print("Warning: Timed out waiting for activations.")

    print("Stopping llama.cpp process...")
    procCPP.stop()
    print("Waiting for SharedMem thread to finish...")
    sharedRAM.join()
    handler.save()

def injectActivation(prompts_file, arg):
    # arg is either a token index (int) read from detembeds.bin, or the name
    # of a binary file containing the raw float32 vector to inject
    try:
        token_index = int(arg)
        is_index = True
    except ValueError:
        is_index = False

    if is_index:
        # read only the embedding of a single token from the unembedding matrix
        # that this program saved earlier (run it with SAVE_EMB to produce
        # detembeds.bin/detembeds.dims), without loading the whole matrix in RAM
        with open("detembeds.dims") as f:
            ne3 = int(f.readline().strip())
            ne2 = int(f.readline().strip())
            ne1 = int(f.readline().strip())
            ne0 = int(f.readline().strip())
        if token_index >= ne1:
            print("ERROR token index " + str(token_index) + " out of range (vocab=" + str(ne1) + ")")
            return
        with open("detembeds.bin", "rb") as f:
            # output.weight is [ne1 rows, ne0 cols], so row token_index starts here
            f.seek(token_index * ne0 * 4)
            emb = np.frombuffer(f.read(ne0 * 4), dtype=np.float32).copy()
        print("loaded embedding for token " + str(token_index) + " dim=" + str(len(emb)))
    else:
        emb = np.fromfile(arg, dtype=np.float32)
        print("loaded embedding from file " + arg + " dim=" + str(len(emb)))

    class ActivsInjector:
        # overwrites the last connected layer with the target embedding during
        # the prefill only, so the model is forced to generate the target token
        def __init__(self, emb, nlayers):
            self.emb = emb
            self.nlayers = nlayers
            self.injected = False

        def processActivations(self, actbig, i):
            if not self.injected and i == self.nlayers - 1:
                print("injecting target embedding at last layer (prefill) shape=" + str(actbig.shape))
                self.injected = True
                print("injected vector first 5 values:", self.emb[:5])
                return np.broadcast_to(self.emb, actbig.shape).copy()
            return None # do not modify activations

    handler = ActivsInjector(emb, len(connLayers))
    sharedRAM, procCPP = initLlamacpp("./", connLayers, handler)

    with open(prompts_file) as f:
        prompts = [line.rstrip('\n') for line in f if line.strip()]

    print("Triggering rollouts with injected embedding...")
    for utt in prompts:
        print("Prompt:", utt)
        s = sharedRAM.rollout_gen(utt)
        if not s == None:
            print("GEN", s[0])
            print("TOKGEN", s[1])
        time.sleep(1)

    print("Waiting for activations to be processed...")
    time.sleep(2)

    print("Stopping llama.cpp process...")
    procCPP.stop()
    print("Waiting for SharedMem thread to finish...")
    sharedRAM.join()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python xllamacpp.py <prompts_file> [token_index|vector_file]")
        print("  without arg: save activations to <prompts>_activs.npz")
        print("  with token_index: load detembeds.bin and inject the embedding")
        print("    of that token into the last layer at the prefill step")
        print("  with vector_file: read the raw float32 vector from that binary file")
        print("    and inject it directly into the last layer at the prefill step")
        sys.exit(1)
    prompts_file = sys.argv[1]
    if len(sys.argv) >= 3:
        injectActivation(prompts_file, sys.argv[2])
    else:
        saveActivations(prompts_file)

