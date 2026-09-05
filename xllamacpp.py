# This program launches llama.cpp and connects to it through a shared RAM memory zone and 2 semaphores
# It exposes a method to run a rollout on a sentence and get the latent activations from llama.cpp
# An example method saveActivations() shows at the end how to use this code

# WARNING: with MoE, the prefill may be splitted into several chunks!

# Environment variables to control the program:
# SAVE_EMB=result_output to save the embeddings

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
import json

NTOKS2GEN = int(os.environ.get("NTOKS", 5))
# when NTOKS is not set, serve mode generates until end-of-sentence instead
# of a fixed number of tokens (n_predict = -1 in llama.cpp means "infinite")
NTOKS_IS_SET = "NTOKS" in os.environ
PORT = "8257"
SHM_NAME = "/ring_buffer_demo"
SEM_C2P = "/c2py_sem"
SEM_P2C = "/py2c_sem"

with open("layers2save") as f:
    nlayers = sum(1 for line in f if line.strip())

# SAVE_EMB dumps the unembedding matrix (detembeds.bin). The dump hook only
# works when that tensor is whole and in host memory: run all layers on CPU
# (ngl 0) and avoid GPU/CPU splits. Otherwise offload greedily.
save_emb = bool(os.environ.get("SAVE_EMB"))

def build_opts(modnom):
    # the model filename usually embeds whether it is a MoE (e.g. "A3B");
    # the check is only used to pick sensible GPU/CPU offload defaults
    if "b-a" in modnom.lower():
        # MoE
        OPTS = ["--no-mmap", "--temp", "0"]
        if not save_emb:
            # push the experts of the first 30 layers to RAM to fit e.g. 35B on GPU
            OPTS += ["--n-gpu-layers", "999", "--n-cpu-moe", "30"]
        else:
            OPTS += ["--n-gpu-layers", "0"]
    else:
        # dense
        OPTS = ["-ngl", "0" if save_emb else "99", "--temp", "0"]
    return OPTS

class DummyHandler:
    # consumes and prints every activation sent by llama.cpp while it is
    # initializing (warmup or leftovers from a previous run).
    def processActivations(self, actbig, i, node_name=""):
        print("DUMMY consuming activation "+str(i)+" shape="+str(actbig.shape)+" first="+str(actbig.flatten()[:5]))
        return None # do not modify activations

class SharedMem(threading.Thread):
    def __init__(self, activsHandler, listening_event):
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
        layer = 0        # index of the current layer within a forward pass
        prev_name = None # name of the previous layer
        while not fincpp:
            # Wait for C++ to fill buffer
            print('python wait layer',layer)
            self.sem_c2p.acquire()
            print("now reading layer from shared buffer",layer)
            vec, name_str = self.get_buffer_view()
            if vec is None:
                # llamacpp is quitting: it warns this listener by writing the
                # sentinel magic value 424242
                fincpp = True
                print("llamacpp stopped; ending the listener loop")
                break

            if len(vec)==0:
                print("python got empty c++ vector")
            else:
                actbig = np.array(vec, copy=True)
                ne1, ne0 = actbig.shape
                # each forward pass is a full layer progression (per ubatch), so
                # a name change means we moved to the next layer
                if name_str != prev_name:
                    if prev_name is not None:
                        layer += 1
                    prev_name = name_str
                is_last = (layer % nlayers) == (nlayers - 1) # last layer before the unembedding
                handler = self.get_handler()
                y = handler.processActivations(actbig, layer % nlayers, name_str)
                if y is None:
                    pass
                elif not is_last:
                    # llamacpp only reads the last layer back, ignore the rest
                    print("WARNING: handler modified non-last layer "+str(layer % nlayers)+" node="+name_str
                          +" changes ignored")
                else:
                    # last layer: write the modified last token back before
                    # llamacpp uses it for the unembedding. llama-server only
                    # samples the last token of each forward pass, so modifying
                    # every ubatch's last layer last row is harmless except on
                    # the final one, which is the one that matters.
                    self.write_last_token(np.asarray(y, dtype=np.float32), ne1)
            print("gonna tell llamacpp to continue")
            self.sem_py2c.release()
        print("xllamacpp stopping; removing semaphores1")
        os.system('rm -f /dev/shm/sem.py2c_sem')
        os.system('rm -f /dev/shm/sem.c2py_sem')
 
    def write_last_token(self, y, ne1):
        # write the modified last token of the last layer back into shared RAM,
        # at the position llamacpp reads it from for the unembedding
        ne0 = y.shape[-1]
        off = 4 * (102 + (ne1 - 1) * ne0) # skip node name (100) + ne1 + ne0
        self.buf[off : off + 4*ne0] = y[-1].astype(np.float32, copy=False).tobytes()

    def get_buffer_view(self):
        # Check C++ sentinel at position 0 (written during cleanup)
        if np.frombuffer(bytes(self.buf[0:4]), dtype=np.float32).item() == 424242:
            return None, None
        start = 0
        # C++ stores: name (100 floats) -> ne1 -> ne0 -> tensor data
        # read the 100-char node name (stored as 100 float32 values in C++)
        name_floats = np.frombuffer(bytes(self.buf[start : start + 100*4]), dtype=np.float32)
        name_bytes = name_floats.astype(np.uint8)
        name_str = name_bytes.tobytes().split(b'\x00')[0].decode('ascii', errors='replace')
        start += 100*4
        mv = self.buf[start : start + 4]
        start += 4
        ne1 = int(np.frombuffer(mv, dtype=np.float32).item())
        mv = self.buf[start : start + 4]
        start += 4
        ne0 = int(np.frombuffer(mv, dtype=np.float32).item())
        mv = self.buf[start : start + 4*ne0*ne1]
        vec = np.frombuffer(mv, dtype=np.float32)
        vec.shape = (ne1,ne0)
        # for i in range(ne1): print(vec[i][0])
        return vec, name_str

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
 
    def rollout_gen(self, prompt, n_predict=NTOKS2GEN, stop=None, temperature=None):
        # this method is used when running inference on a file
        url = "http://localhost:"+PORT+"/completions"
        data = {"prompt" : prompt, "return_tokens": True, "cache_prompt": False,
                "n_predict": n_predict}
        if stop is not None:
            data["stop"] = stop
        if temperature is not None:
            data["temperature"] = temperature
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

    def raw_rollout(self, req, endpoint):
        # this method is used when running inference as a server
        # post a client payload unchanged to one of llama-server's OpenAI endpoints
        # and return the raw (already OpenAI-formatted) response
        url = "http://localhost:"+PORT+endpoint
        headers = { "Content-Type": "application/json" }
        print("sending raw payload to llama-server", endpoint)
        response = requests.post(url, json=req, headers=headers)
        print("Status code:", response.status_code)
        if str(response.status_code)[0] != "2": return None

    def raw_rollout_stream(self, req, endpoint):
        # this method is used when running inference as a server, streaming mode
        # post a client payload unchanged to llama-server's OpenAI endpoint with
        # the stream flag on, and yield the raw SSE bytes as they arrive instead
        # of buffering the whole response. On any upstream failure it yields
        # nothing, so the caller can tell it apart from a clean-but-empty stream.
        url = "http://localhost:"+PORT+endpoint
        headers = { "Content-Type": "application/json" }
        print("sending raw payload to llama-server", endpoint)
        try:
            response = requests.post(url, json=req, headers=headers, stream=True)
        except requests.RequestException as e:
            print("WARNING: request failed ("+str(e)+")")
            return
        if str(response.status_code)[0] != "2":
            print("WARNING: llama-server returned status "+str(response.status_code))
            return
        try:
            # iter_content honors the upstream chunk boundaries, so the SSE
            # events are relayed unchanged and in real time
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        finally:
            response.close()

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
 
def initLlamacpp(llamacppdir, activsHandler, modnom):
    # toujours nettoyer les semaphores precedentes avant de relancer llamacpp et SharedMem
    print("removing semaphores0")
    os.system('rm -f /dev/shm/sem.py2c_sem')
    os.system('rm -f /dev/shm/sem.c2py_sem')

    # set once llama.cpp prints the listening text: then SharedMem switches
    # from the dummy handler to the real one
    listening_event = threading.Event()

    sm = SharedMem(activsHandler, listening_event)
    sm.start()
    runner = AsyncScriptRunner(llamacppdir+"/build/bin/llama-server","-m",modnom,"--no-webui",
                               "--no-warmup","--ctx-size","30000","--cache-ram", "0", 
                               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-nkvo", 
                               "--port", PORT, *build_opts(modnom),
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
    #   int32 ndim, int32[ndim] shape, int32 len(name), bytes name, int64 len(compressed), gzip bytes
    def __init__(self, outfile):
        self.outfile = outfile
        self.n = 0
        self.f = open(outfile, "ab")

    def save_one(self, actbig, node_name=""):
        shape = np.asarray(actbig.shape, dtype=np.int32)
        name_bytes = node_name.encode('ascii', errors='replace')
        comp = gzip.compress(np.ascontiguousarray(actbig, dtype=np.float32).tobytes())
        self.f.write(np.array([len(shape)], dtype=np.int32).tobytes())
        self.f.write(shape.tobytes())
        self.f.write(np.array([len(name_bytes)], dtype=np.int32).tobytes())
        self.f.write(name_bytes)
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
    names = []
    with open(infile, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) == 0: break
            ndim = np.frombuffer(head, dtype=np.int32).item()
            shape = np.frombuffer(f.read(4*ndim), dtype=np.int32).tolist()
            namelen = np.frombuffer(f.read(4), dtype=np.int32).item()
            name = f.read(namelen).decode('ascii', errors='replace')
            names.append(name)
            nbytes = np.frombuffer(f.read(8), dtype=np.int64).item()
            data = np.frombuffer(gzip.decompress(f.read(nbytes)), dtype=np.float32)
            data.shape = shape
            activs.append(data)
    return activs, names

def saveActivations(prompts_file, modnom):
    class ActivsHandler:
        def __init__(self, outfile):
            # the real activs handler only runs after llama.cpp is listening,
            # so the dummy activations captured during startup are not saved
            self.processed_once = threading.Event()
            self.n = 0
            self.outfile = outfile
            self.saver = ActivsSaver(outfile)

        def processActivations(self, actbig, i, node_name=""):
            self.n += 1
            if actbig.shape[0]>0:
                print("saving activation "+str(i)+" shape="+str(actbig.shape)+" node="+node_name)
                self.saver.save_one(actbig, node_name)
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
    sharedRAM, procCPP = initLlamacpp("./", handler, modnom)

    print("Triggering rollouts to get activations...")
    # for is useless: file is a single prompt
    for utt in prompts:
        print("Prompt:", utt)
        s=sharedRAM.rollout_gen(utt)
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

class LadderHandler:
    # runs each pair of connected-layer activations through
    # the MLP trained by train.py, and reinjects the transformed norm activation
    # back into llama.cpp. Reimplements the test-time forward of the MLP in numpy
    # (keys are row-normalized, cosines gated by the stored threshold) so we only
    # need the saved state dict in mlp.pt, not the nn.Module class.
    def __init__(self, mlpfile):
        import torch
        state = torch.load(mlpfile, map_location="cpu")
        self.thr = float(state.pop("thr"))
        keys = state["keys.weight"].numpy().astype(np.float32)        # (dhid, din)
        self.vals_w = state["vals.weight"].numpy().astype(np.float32) # (hdim, dhid)
        # vals is a nn.Linear, so its forward also adds the bias
        self.vals_b = state["vals.bias"].numpy().astype(np.float32) if "vals.bias" in state else None
        # note: keys.bias, when present, is ignored because MLP.forward only
        # reads self.keys.weight (it never calls keys(x) as a layer)
        # row-normalize the stored key directions (see MLP.forward)
        self.keys_w = keys / np.linalg.norm(keys, axis=1, keepdims=True)
        self.prev = None  # previous (l_out-16) activation, paired with the next one
        self.n = 0

    def processActivations(self, actbig, i, node_name=""):
        actbig = np.asarray(actbig, dtype=np.float32)
        if self.prev is None:
            # first layer: just buffer it unchanged
            self.prev = actbig.copy()
            return None
        # pair is [prev (l_out-16), current (norm)]; the MLP input is the 2*dim concat
        X = np.concatenate([self.prev, actbig], axis=1)
        self.prev = None
        # normalized cosine similarity between the input and every stored key
        xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        sim = xn @ self.keys_w.T                       # (T, dhid)
        gate = np.maximum(sim - self.thr, 0.0)         # test-time thresholded ReLU
        bias = gate @ self.vals_w.T                    # (T, hdim)
        if self.vals_b is not None: bias = bias + self.vals_b
        # reinject: residual stream is the norm half of the input (== current actbig)
        out = actbig + bias
        self.n += 1
        print("ladder applied to layer pair "+str(self.n)+" shape="+str(actbig.shape))
        return out.astype(np.float32)


def runLadderOnFile(prompts_file, mlpfile, modnom):
    handler = LadderHandler(mlpfile)
    sharedRAM, procCPP = initLlamacpp("./", handler, modnom)

    with open(prompts_file) as f:
        prompts = [line.rstrip('\n') for line in f if line.strip()]
        prompts = ['\n'.join(prompts)]
        print("promptlen", len(prompts), len(prompts[0]))

    print("Triggering rollouts with ladder MLP...")
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


def injectTokenAct(prompts_file, token_index, modnom):
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

    class ActivsInjector:
        # overwrites the last connected layer with the target embedding during
        # the prefill only, so the model is forced to generate the target token
        def __init__(self, emb):
            self.emb = emb
            self.injected = False

        def processActivations(self, actbig, i, node_name=""):
            if not self.injected and i == nlayers - 1:
                print("injecting target embedding at last layer (prefill) shape=" + str(actbig.shape))
                self.injected = True
                print("injected vector first 5 values:", self.emb[:5])
                return np.broadcast_to(self.emb, actbig.shape).copy()
            return None # do not modify activations

    handler = ActivsInjector(emb)
    sharedRAM, procCPP = initLlamacpp("./", handler, modnom)
 
    with open(prompts_file) as f:
        prompts = [line.rstrip('\n') for line in f if line.strip()]
        prompts = ['\n'.join(prompts)]
        print("promptlen",len(prompts),len(prompts[0]))

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

class NoopHandler:
    # pass activations through unmodified (used when serving without a ladder)
    def processActivations(self, actbig, i, node_name=""):
        return None

def serveOpenAI(modnom, ladder_file=None, host="127.0.0.1", port=8258):
    # serve a local minimal OpenAI-compatible endpoint on the given LLM,
    # modified by the ladder MLP when one is given
    handler = LadderHandler(ladder_file) if ladder_file is not None else NoopHandler()
    sharedRAM, procCPP = initLlamacpp("./", handler, modnom)

    import json
    import http.server
    model_name = modnom.rsplit("/", 1)[-1]
    # llama.cpp runs with a single slot; serialize the rollouts so the ladder
    # pairing never sees interleaved activations from concurrent requests
    serial = threading.Lock()

    class OpenAIHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _req(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length)) if length else {}

        def do_GET(self):
            if self.path == "/v1/models":
                self._send(200, {"object": "list",
                                 "data": [{"id": model_name, "object": "model",
                                           "created": int(time.time()), "owned_by": "xllamacpp"}]})
            else:
                self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
                return
            req = self._req()
            # print("DEBUG full prompt",str(req))
            with serial:
                # relay the client payload as-is to llama-server's OpenAI endpoint.
                # llama.cpp understands the OpenAI chat/completion format natively,
                # so we forward the json unchanged instead of reconstructing it here
                # The ladder MLP still transforms every activation during the
                # rollout, because the shared-memory hook fires on the Llama forward
                # pass regardless of which llama-server endpoint serves the request.
                if 'messages' in req:
                    endpoint = "/v1/chat/completions"
                elif 'prompt' in req:
                    endpoint = "/v1/completions"
                else:
                    self._send(500, {"error": {"message": "invalid payload: no 'messages' or 'prompt'",
                                               "type": "invalid_request_error"}})
                    return
                if req.get("stream", False):
                    # llama-server replies in SSE, token by token; relay the
                    # stream to the client progressively instead of buffering
                    # the whole response (which would mis-parse as JSON below)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    got_chunk = False
                    for chunk in sharedRAM.raw_rollout_stream(req, endpoint):
                        got_chunk = True
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    if not got_chunk:
                        # upstream failed before any SSE byte: send the error
                        # as an SSE event, since the 200 headers are already out
                        err = "data: {\"error\":{\"message\":\"llama.cpp rollout failed\",\"type\":\"server_error\"}}\n\n"
                        self.wfile.write(err.encode())
                        self.wfile.flush()
                    # the SSE body has no Content-Length/chunked framing, so its
                    # only terminator is connection close; close it so the caller
                    # unambiguously sees the end of the stream after [DONE]
                    self.close_connection = True
                    return
                res = sharedRAM.raw_rollout(req, endpoint)
                if res is None:
                    self._send(500, {"error": {"message": "llama.cpp rollout failed", "type": "server_error"}})
                    return
                print("llama-server returned:", res)
                self._send(200, res)

    server = http.server.ThreadingHTTPServer((host, port), OpenAIHandler)
    print("OpenAI endpoint on http://"+host+":"+str(port)+"/v1 (model "+model_name+")")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Stopping llama.cpp process...")
        procCPP.stop()
        sharedRAM.join()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="llama.cpp latent activation rollout")
    parser.add_argument("--prompts", type=str, default=None,
                        help="file with the prompt; omit to serve a local OpenAI endpoint")
    parser.add_argument("--model", type=str, required=True,
                        help="path to the .gguf model file")
    parser.add_argument("--inject_token", type=int, default=None,
                        help="load detembeds.bin and inject the embedding of this "
                             "token into the last layer at the prefill step")
    parser.add_argument("--ladder", type=str, default=None,
                        help="file with the MLP trained by train.py (e.g. mlp.pt); "
                             "transform each non-dummy activation through it and "
                             "reinject the output into llama.cpp")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="bind address of the OpenAI endpoint")
    parser.add_argument("--port", type=int, default=8258,
                        help="port of the OpenAI endpoint")
    parser.add_argument("--serve", action="store_true",
                        help="serve the OpenAI endpoint even when --prompts is given")
    args = parser.parse_args()
    if args.prompts is None or args.serve:
        # no prompt file: serve the given LLM (plus the ladder when given)
        serveOpenAI(args.model, args.ladder, args.host, args.port)
    elif args.ladder is not None:
        runLadderOnFile(args.prompts, args.ladder, args.model)
    elif args.inject_token is None:
        # without arg: save activations to <prompts>_activs.npz
        saveActivations(args.prompts, args.model)
    else:
        injectTokenAct(args.prompts, args.inject_token, args.model)

