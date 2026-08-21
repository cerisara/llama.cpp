# This program launches llama.cpp and connects to it through a shared RAM memory zone and 2 semaphores
# It exposes a method to run a rollout on a sentence and get the latent activations from llama.cpp
# An example method helloworld() shows at the end how to use this code
# Warning: the first time llama.cpp is run it will create and save the detembed matrix file: it'll then end with an error
# and you'll have to terminate it with Ctrl-C. The next time, the code will run as expected.

import os
import mmap
import subprocess
import numpy as np
from posix_ipc import Semaphore, SharedMemory
import time
import threading
import signal
import requests

SHM_NAME = "/ring_buffer_demo"
SEM_C2P = "/c2py_sem"
SEM_P2C = "/py2c_sem"
modnom="/home/xtof/Qwen3-8B-Q5_K_M.gguf"
modnom="/home/xtof/ggufs/qwen2.5-0.5b-instruct-q5_k_m.gguf"

class SharedMem(threading.Thread):
    def __init__(self, nLayersGot, activsHandler):
        self.nLayersGot = nLayersGot
        self.activsHandler = activsHandler
        threading.Thread.__init__(self)

    def run(self):
        # this function is executed in a separate thread:
        # it listens to llamacpp and calls a ladder method when the forward pass has reached the last LLM layer
        print("sharedmem thread started")
        # Open shared memory
        self.fd = os.open("/dev/shm" + SHM_NAME, os.O_RDWR)
        self.mm = mmap.mmap(self.fd, 4*10000000) # 4 because in C++ the size is given in float32!
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
            if len(vec)==0:
                # when llamacpp quits, it warns this listener with an empty vector
                print("python got empty c++ vector")
                fincpp = True
                break
            # big activations (the ones from the LLM):
            actbig = np.array(vec, copy=True)
            # actbig = T x 896
            y = self.activsHandler(actbig,i % self.nLayersGot)
            if not y is None:
                # overwrite the activations into llama.cpp
                for j in range(len(vec)):
                    for k in range(len(vec[j])):
                        vec[j][k] = y[j][k]
            print("gonna tell llamacpp to continue")
            self.sem_py2c.release()
            i += 1
        print("removing semaphores1")
        os.system('rm /dev/shm/sem.py2c_sem')
        os.system('rm /dev/shm/sem.c2py_sem')
 
    def get_buffer_view(self):
        start = 0
        mv = self.buf[start : start + 4]
        start += 4
        ne1 = np.frombuffer(mv, dtype=np.float32)
        if ne1==424242: return []
        ne1 = int(ne1)
        mv = self.buf[start : start + 4]
        start += 4
        ne0 = int(np.frombuffer(mv, dtype=np.float32))
        mv = self.buf[start : start + 4*ne0*ne1]
        vec = np.frombuffer(mv, dtype=np.float32)
        vec.shape = (ne1,ne0)
        # for i in range(ne1): print(vec[i][0])
        return vec

    def detokenize(self, toks):
        url = "http://localhost:8080/detokenize"
        data = { "tokens": [int(x) for x in toks] }
        headers = { "Content-Type": "application/json" }
        response = requests.post(url, json=data, headers=headers)
        print("Status code:", response.status_code)
        sout = response.json()
        print("detoken", sout.keys())
        u = sout['content'] 
        return u
 
    def tokenize(self, prompt):
        url = "http://localhost:8080/tokenize"
        data = { "content": prompt }
        headers = { "Content-Type": "application/json" }
        response = requests.post(url, json=data, headers=headers)
        sout = response.json()
        protoks = sout['tokens'] 
        return protoks
 
    def rollout(self, prompt):
        if '"' in prompt: 
            print("ERROR ROLLOUT",prompt)
            return None,None
        # j'ai check que l'API embeddings retourne le meme vecteur d'embeddings que celui obtenu
        # en dernier via l'API completions; mais on ne peux plus beneficier du KV-cache:
        # ca ne sert a rien a train time, mais a test time il faudra donc utiliser l'API completion !
        # TODO: check ACC a test time avec completion after train avec embeddings

        url = "http://localhost:8080/embeddings"
        data = {"content" : prompt, "return_tokens": True, "cache_prompt": False}
        headers = { "Content-Type": "application/json" }
        print("sending prompt to llama.cpp")
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
        return sout[0]['embedding']
     
class AsyncScriptRunner:
    def __init__(self, script_path, *args):
        self.script_path = script_path
        self.args = args
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.wait_event = threading.Event()  # Event to signal when text is found
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
            creationflags=creationflags
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
 
def initLlamacpp(llamacppdir, connectedCPPLayers, activsHandler):
    assert("result_norm" in connectedCPPLayers, "Always connect last layer")
    # toujours nettoyer les semaphores precedentes avant de relancer llamacpp et SharedMem
    print("removing semaphores0")
    os.system('rm /dev/shm/sem.py2c_sem')
    os.system('rm /dev/shm/sem.c2py_sem')

    sm = SharedMem(len(connectedCPPLayers), activsHandler)
    sm.start()
    os.system('rm -f layers2save')
    os.system('touch layers2save')
    for l in connectedCPPLayers: os.system('echo "'+l+'" >> layers2save')
    os.system("rm -rf detlog")
    os.system("mkdir detlog")

    runner = AsyncScriptRunner(llamacppdir+"/build/bin/llama-server","-ub","2048","-m",modnom,"--no-webui","--no-warmup","--ctx-size","30000","--cache-ram", "0", "--embeddings")
    runner.start(wait_for_text="all slots are idle")
    print("python main thread continues")
    time.sleep(1)
    return sm, runner

def helloworld():
    class ActivsHandler:
        def __init__(self):
            # We only want to print once.
            self.processed_once = threading.Event()

        def processActivations(self, actbig, i):
            print("First 10 values of llama.cpp's activations "+str(i)+": "+' '.join([str(x) for x in actbig.flatten()[:10]]))
            if i == 2: self.processed_once.set()
            return None # do not modify activations

    handler = ActivsHandler()
    connLayers = ['l_out-2','l_out-12','result_norm']
    sharedRAM, procCPP = initLlamacpp("./modllama", connLayers, handler)
    
    print("Triggering rollout to get activations...")
    utt = "The sounds of time for me are running low"
    sharedRAM.rollout(utt, 10)

    print("Waiting for activations to be processed...")
    processed = handler.processed_once.wait(timeout=10) # Wait for 10 seconds max
    if not processed: print("Warning: Timed out waiting for activations.")

    print("Stopping llama.cpp process...")
    procCPP.stop()
    print("Waiting for SharedMem thread to finish...")
    sharedRAM.join()

if __name__ == "__main__":
    helloworld()

