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
# ActivsSaver in xllamacpp.py. Drop the last isolated tensor: it is a single
# activation that belongs to no prefill chunk and pollutes training data.
activs = load_activs("q1_oracle_activs.npz")
print("activations tensors:", len(activs), "last shape:", activs[-1].shape)
activs = activs[:-1]

# each token is captured once per connected layer (l_out-16 and norm), so the
# number of activation vectors should be double the number of prompt tokens
nactivs = sum(a.shape[0] for a in activs)
print("activation vectors:", nactivs, "tokens:", len(toks), "expected:", 2 * len(toks))
assert nactivs == 2 * len(toks), \
    "activation vectors should be double the prompt tokens"
