import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

modnom = "Qwen/Qwen3-0.6B"
bdim = 8960
ldim = 1024
 
def myhookemb(layer, input, output):
    zt = output[0].dtype
    # TODO: replace embeddings with first activations
    z = z.to(zt)
    return (z,)
 
def myhook(layer, input, output):
    z = output[0]
    zt = z.dtype
    z = z.to(torch.float32)
    # TODO: add activs
    z = z.to(zt)
    return (z,)
 
def myhookfin(layer, input, output):
    z = output[0]
    zt = z.dtype
    z = z.to(torch.float32)
    # TODO: add activs
    # TODO: sum with last backbone output
    z = z.to(zt)
    return (z,)
 
cfg = AutoConfig.from_pretrained(modnom)
cfg.max_window_layers = 4
cfg.num_hidden_layers = 4
cfg.layer_types = ["full_attention"]*4
cfg.bos_token_id = 0
cfg.eos_token_id = 0
cfg.vocab_size = 1
print(cfg)

mod = Qwen3ForCausalLM(cfg)
for p in mod.model.embed_tokens.parameters(): p.requires_grad=False
for n,p in mod.named_parameters(): print(n,p.size())
print("lmhead", mod.lm_head)

mod.model.embed_tokens.register_forward_hook(myhookemb)
for i in range(3):
    mod.model.layers[i].detlayer = i
    dproj = torch.nn.Linear(bdim, ldim)
    mod.model.layers[i].downproj = dproj
    mod.model.layers[i].register_forward_hook(myhook)
for i in (4,):
    mod.model.layers[i].detlayer = i
    dproj = torch.nn.Linear(bdim, ldim)
    mod.model.layers[i].downproj = dproj
    uproj = torch.nn.Linear(ldim, bdim)
    mod.model.layers[i].upproj = uproj
    mod.model.layers[i].register_forward_hook(myhookfin)
 
for n,m in mod.named_modules(): print(n,type(m))
for n,p in mod.named_parameters(): print(n,p.size())
nparms = sum(p.numel() for p in mod.parameters() if p.requires_grad)
print("nparms",nparms)
exit()

# TODO: add upproj et lm_head

toker = AutoTokenizer.from_pretrained(modnom)
