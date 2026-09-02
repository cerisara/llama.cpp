import ast
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from read_activs import load_activs

device = torch.device("cpu")
if torch.cuda.is_available():
    # verify the GPU actually works with a dummy tensor op before using it
    try:
        x = torch.zeros(1, device="cuda")
        y = (x + 1).cpu()
        assert y.item() == 1
        device = torch.device("cuda")
    except Exception as e:
        print("GPU check failed, falling back to CPU:", e)
print("using device:", device)

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
hdim = activs[0].shape[1]
print("activation vectors:", nactivs, "tokens:", len(toks), "expected:", 2 * len(toks))
assert nactivs == 2 * len(toks), "activation vectors should be double the prompt tokens"

# each activation tensor is one capture of a connected layer; the two connected
# layers (l_out-16, norm) alternate across successive chunks, so every pair of
# chunks (2k, 2k+1) holds the two layer vectors for the same set of tokens.
# Rebuild one vector per token per layer, then concatenate them into the MLP
# input vector for that token.
n_layers = 2
layer_vecs = [np.concatenate([a for a in activs[layer::n_layers]], axis=0) for layer in range(n_layers)]
assert all(len(v) == len(toks) for v in layer_vecs), "layer vectors must match token count"
X = np.concatenate(layer_vecs, axis=1)  # (ntoks, 2*dim) input pairs

# the target is the embedding of the token id at t+1 (shifted right wrt t)
# the unembedding matrix is stored row by row (one row per token), so the
# shape is (vocab_size, n_embd); read it from the dims file written by
# SAVE_EMB in xllamacpp.py so other LLMs work too
with open("detembeds.dims") as f:
    # first two lines are ne3/ne2 (always 1); then ne1 (vocab), ne0 (n_embd)
    f.readline()
    f.readline()
    ne1 = int(f.readline().strip())
    ne0 = int(f.readline().strip())
embeds = np.fromfile("detembeds.bin", dtype=np.float32)
embeds.shape = (ne1, ne0)  # (vocab_size, n_embd)
Y = embeds[np.array(toks[1:])]
X = X[:-1]  # last token has no t+1 target
print("train set:", X.shape, "->", Y.shape)

# the MLP adds a bias onto the residual stream (last half of the input), so its
# width must match the target embedding width; a mismatch means the activations
# and detembeds.bin were captured from different models
if X.shape[1] // 2 != Y.shape[1]:
    print("WARNING: residual stream dim", X.shape[1] // 2,
          "!= target embedding dim", Y.shape[1],
          "; activations and detembeds.bin come from different models")

# small MLP trained with MSE to predict the t+1 token embedding from a pair of
# activation vectors (l_out-16 and norm)
# It's not really an MLP, it's actually a residual key-value store with a ReLU between both
# layers; the ReLU has a tunable threshold (instead of standard 0) and its role
# is to output 0 when the similarity between the input and stored keys is too
# small. The residual stream is not the full input, but the second-half of the
# input, which corresponds to the last activation layer of the main LLM. Hence,
# this MLP actually modifies (adds a bias to) the output of the main LLM only when 
# its input matches the keys it's been trained on.
class MLP(nn.Module):
    def __init__(self, din, dhid):
        super().__init__()
        self.keys = nn.Linear(din, dhid, bias=False)
        self.vals = nn.Linear(dhid, din // 2, bias=False)
        nn.init.zeros_(self.vals.weight)  # start with zero output so MLP is neutral
        self.thr  = 0.8 # not a parameter; to be tuned later on
        self.initstats()

    def initstats(self):
        # running sufficient statistics over similarities seen during training
        self.sim_n = 0
        self.sim_min = None
        self.sim_max = None
        self.sim_mean = 0.0
        self.sim_m2 = 0.0 # sum of squared deviations from the mean (Welford)

    def forward(self, x):
        # residual stream is the last half of each vector (last LLM activation layer)
        lasth = x[:, x.shape[1] // 2:]
        # normalized (cosine) similarity between the input and every stored key;
        # the key vectors are the row-normalized weights of the keys layer
        w = F.normalize(self.keys.weight, dim=1)  # (dhid, din) key directions
        xn = F.normalize(x, dim=1)                # (batch, din)
        sim = xn @ w.t()                          # (batch, dhid) cosines in [-1, 1]
        # print("MLPSIM",torch.max(sim))
        # decode the bias added onto the residual stream; during training drop the
        # threshold and train a plain 2-layer linear stack, apply it only at test
        # time (model.eval()) where similarities that are too small are zeroed out
        if self.training:
            smoothed_vals = self.vals(sim)
            # update running stats on the best-matching cosine per input (Welford merge)
            best = sim.max(dim=1).values          # (batch,) best cosine
            x = best.detach()
            bn = x.numel()
            bmean = x.mean().item()
            bm2 = ((x - bmean) ** 2).sum().item()
            if self.sim_min is None:
                self.sim_min = x.min().item()
                self.sim_max = x.max().item()
            else:
                self.sim_min = min(self.sim_min, x.min().item())
                self.sim_max = max(self.sim_max, x.max().item())
            n = self.sim_n + bn
            delta = bmean - self.sim_mean
            self.sim_mean += delta * bn / n
            self.sim_m2 += bm2 + delta * delta * self.sim_n * bn / n
            self.sim_n = n
        else:
            # cosine is already in [-1, 1], so thr gates on direction match
            smoothed_vals = self.vals(F.relu(sim - self.thr))
        # return both the reconstruction and the cosine similarities, so the
        # caller can reuse sim (for the aux loss) instead of recomputing it
        return lasth + smoothed_vals, sim

model = MLP(X.shape[1], X.shape[1]) # choose second dim as you wish
opt = torch.optim.Adam(model.parameters(), lr=1e-5)
ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=device),
                                    torch.tensor(Y, dtype=torch.float32, device=device))
print("dataset",len(ds))
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

model = model.to(device)
model.train()
for epoch in range(30):
    total = 0.0
    total2 = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        out, sim = model(xb)
        loss = F.mse_loss(out, yb)
        # auxiliary term: reuse the cosine similarities already computed in the
        # forward pass; take the best-matching key (max over hidden units) and
        # push that best cosine up towards 1
        best = sim.max(dim=1).values               # (batch,) best cosine per input
        sim_loss = F.relu(1. - best).mean()
        combined = loss + 10000. * sim_loss
        combined.backward()
        opt.step()
        total += loss.item() * len(xb)
        total2 += sim_loss.item() * len(xb)
    print("epoch", epoch, "mse", total / len(ds), "keysimloss", total2 / len(ds))

    # running stats gathered during training, to pick a sensible thr for the
    # test-time threshold
    print("similarity stats: count=%d min=%.4f mean=%.4f std=%.4f max=%.4f" % (
          model.sim_n, model.sim_min, model.sim_mean,
          (model.sim_m2 / model.sim_n) ** 0.5, model.sim_max))
    model.initstats()

# save the learned parameters (keys, vals) and the threshold to disk
state = model.state_dict()
state["thr"] = model.thr
torch.save(state, "mlp.pt")
print("saved parameters to mlp.pt")
