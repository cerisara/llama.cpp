import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

modnom = "Qwen/Qwen3-0.6B"
bdim = 8960
ldim = 1024

def myhook(layer, input, output):
    z = output[0]
    zt = z.dtype
    z = z.to(torch.float32)
    # TODO: add activs
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
for n,m in mod.named_modules(): print(n,type(m))
nparms = sum(p.numel() for p in mod.parameters())
print("nparms",nparms)
for n,p in mod.named_parameters(): print(n,p.size())
print("lmhead", mod.lm_head)

downproj0 = torch.nn.Linear(bdim, ldim)
mod.model.layers[0].detlayer = 0
mod.model.layers[0].downproj = downproj0
mod.model.layers[0].register_forward_hook(myhook)

for n,p in mod.named_parameters(): print(n,p.size())
exit()

# TODO: reduce embeddings matrix + add upproj et lm_head

toker = AutoTokenizer.from_pretrained(modnom)
