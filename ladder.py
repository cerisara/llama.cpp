import torch
import numpy
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file, safe_open
import struct

modnom = "Qwen/Qwen3-0.6B"
lmheadfich = "/home/xtof/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35/model.safetensors"
layer_prefix = "model.embed_tokens."
lmheadfich = "/home/xtof/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/model-00004-of-00004.safetensors"
layer_prefix = "lm_head."
ldim = 1024

dev = "cuda"
dev = "cpu"

def loadlmhead():
    weights = {}
    with safe_open(lmheadfich, framework="pt", device="cpu") as f:
        for key in f.keys():
            print("loadkey",key)
            if key.startswith(layer_prefix):
                k = key.replace(layer_prefix, "")
                weights[k] = f.get_tensor(key)
                v,d = weights[k].size()
    print("loadhead",d,v)
    l = torch.nn.Linear(d, v, bias=False)
    l.load_state_dict(weights)
    return l

def readTens():
    global facts
    buffer = facts.read(4)
    assert len(buffer)==4
    nv = struct.unpack('<i', buffer)[0]
    y = []
    for i in range(nv):
        buffer = facts.read(4)
        assert len(buffer)==4
        nd = struct.unpack('<i', buffer)[0]
        buffer = facts.read(4*nd)
        assert len(buffer)==4*nd
        fmt1 = '<'+str(nd)+'f'
        v = struct.unpack(fmt1, buffer)
        y.append(v)
    y = numpy.array(y)
    y = torch.Tensor(y).to(dev)
    # y = T x D
    return y
 
def myhookemb(layer, input, output):
    backbone_acts = readTens()
    x = layer.downproj(backbone_acts)
    output[0] = x
    return output
 
backbone_acts = None
def myhook(layer, input, output):
    global backbone_acts # pour le conserver pour sommer au final
    print("inlayer", layer.detlayer, len(output), output[0].shape)
    o = list(output)
    backbone_acts = readTens()
    x = layer.downproj(backbone_acts)
    z = output[0] + x
    o[0] = z
    return o
 
def myhookfin(layer, input, output):
    side_out = output[0]
    # on change la RS dim pour revenir a la dim du backbone !
    # il ne faut pas de norm apres cela...
    z = layer.upproj(side_out) + backbone_acts
    print("outlayer", layer.detlayer, z.shape)
    return (z,)

facts = open("activs.bin","rb")
x = readTens()
ntoks, bdim = x.size()
facts.close()

cfg = AutoConfig.from_pretrained(modnom)
cfg.max_window_layers = 4
cfg.num_hidden_layers = 4
# cfg.head_dim = 64
cfg.layer_types = ["full_attention"]*4
cfg.bos_token_id = 0
cfg.eos_token_id = 0
cfg.vocab_size = 1
print(cfg)

mod = Qwen3ForCausalLM(cfg)
for p in mod.model.embed_tokens.parameters(): p.requires_grad=False
mod.lm_head = loadlmhead()
for p in mod.lm_head.parameters(): p.requires_grad=False
for n,p in mod.named_parameters(): print(n,p.size())
print("lmhead", mod.lm_head)

dproj = torch.nn.Linear(bdim, ldim)
mod.model.embed_tokens.downproj = dproj
mod.model.embed_tokens.register_forward_hook(myhookemb)
for i in range(3):
    mod.model.layers[i].detlayer = i
    dproj = torch.nn.Linear(bdim, ldim)
    mod.model.layers[i].downproj = dproj
    mod.model.layers[i].register_forward_hook(myhook)
for i in (3,):
    mod.model.layers[i].detlayer = i
    uproj = torch.nn.Linear(ldim, bdim)
    mod.model.layers[i].upproj = uproj
    mod.model.layers[i].register_forward_hook(myhookfin)

def finalnorm(h, *a, **b):
    # supprime la derniere norm (sinon, pb de dim)
    return h
mod.model.norm.forward = finalnorm

mod = mod.to(dev)

for n,m in mod.named_modules(): print(n,type(m))
for n,p in mod.named_parameters(): print(n,p.size())
nparms = sum(p.numel() for p in mod.parameters() if p.requires_grad)
print("nparms",nparms)

floss = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(mod.parameters(), lr=0.0001)
facts = open("activs.bin","rb")
with open("activs.txt", "r") as futt: 
    ss = futt.readlines()
    for s in ss:
        opt.zero_grad()
        toks = [int(x) for x in s.split(" ")]
        print("utt", len(toks), s)
        intoks = [0]*len(toks)
        x = {'input_ids': torch.LongTensor(intoks).view(1,-1).to(dev) }
        y = mod(**x)
        print("out",y.logits.shape)
        gold = torch.LongTensor(toks[1:]).to(dev)
        loss = floss(y.logits[0,:-1], gold)
        print("loss",loss.item())
        loss.backward()
        opt.step()
facts.close()

