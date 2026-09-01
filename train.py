import ast
import sys

from read_activs import load_activs

toks = None
with open("q1.log", "r") as f:
    for l in f:
        if l.startswith("PROMPT_TOKENS "):
            toks = ast.literal_eval(l[len("PROMPT_TOKENS "):])
            break
assert toks is not None, "PROMPT_TOKENS not found in q1.log"
print("prompt tokens:", len(toks))

# read the activations the same way read_activs.py does; the file is written by
# ActivsSaver in xllamacpp.py. 
activs = load_activs("q1_oracle_activs.npz")
print("activations tensors:", len(activs), "last shape:", activs[-1].shape)

# each token is captured once per connected layer (l_out-16 and norm), so the
# number of activation vectors should be double the number of prompt tokens
nactivs = sum(a.shape[0] for a in activs)
print("activation vectors:", nactivs, "tokens:", len(toks), "expected:", 2 * len(toks))
assert nactivs == 2 * len(toks), "activation vectors should be double the prompt tokens"

# TODO read all activs by successive pairs, all pairs form the input training
# dataset for an MLP. The corresponding output is the embedding of the token ID
# indicated at t+1 (the target token embedding to generate is shifted right wrt
# the input pair). This embedding can be found in ./detembeds.bin Then write the
# code to train the MLP (MSE loss)
