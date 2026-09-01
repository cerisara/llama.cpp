import ast
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# each activation tensor is one capture of a connected layer; the two connected
# layers (l_out-16, norm) alternate across successive chunks, so every pair of
# chunks (2k, 2k+1) holds the two layer vectors for the same set of tokens.
# Rebuild one vector per token per layer, then concatenate them into the MLP
# input vector for that token.
n_layers = 2
layer_vecs = [
    np.concatenate([a for a in activs[layer::n_layers]], axis=0)
    for layer in range(n_layers)
]
assert all(len(v) == len(toks) for v in layer_vecs), "layer vectors must match token count"
X = np.concatenate(layer_vecs, axis=1)  # (ntoks, 2*dim) input pairs

# the target is the embedding of the token id at t+1 (shifted right wrt t)
embeds = np.fromfile("detembeds.bin", dtype=np.float32)
# TODO get the shape from detembeds.dim to be able to also work with other LLMs
embeds.shape = (151936, 896)  # (vocab_size, n_embd)
Y = embeds[np.array(toks[1:])]
X = X[:-1]  # last token has no t+1 target
print("train set:", X.shape, "->", Y.shape)

# small MLP trained with MSE to predict the t+1 token embedding from a pair of
# activation vectors (l_out-16 and norm)
class MLP(nn.Module):
    def __init__(self, din, dhid, dout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, dhid),
            nn.ReLU(),
            nn.Linear(dhid, dhid),
            nn.ReLU(),
            nn.Linear(dhid, dout),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X.shape[1], 2048, Y.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
print("dataset",len(ds))
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

model.train()
for epoch in range(30):
    total = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
    print("epoch", epoch, "mse", total / len(ds))

