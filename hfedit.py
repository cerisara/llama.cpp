# import gguf
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


d = "/home/xtof/nvme/qwen2/"
m = "qwen2.5-0.5b-instruct-q5_k_m.gguf"
# r = gguf.gguf_reader.GGUFReader(m)
# print(r)

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(d, gguf_file=m)
model = AutoModelForCausalLM.from_pretrained(d, gguf_file=m)

# model.layers.1.mlp.gate_proj.weight torch.Size([4864, 896])

# model.layers.22.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
# model.layers.22.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
# model.layers.22.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>

for n,m in model.named_modules():
    if n.endswith(".mlp.gate_proj") or n.endswith(".mlp.up_proj"):
        w = m.weight
        wz = torch.zeros(256,w.size(1))
        ww = torch.cat([w,wz])
        m.weight=torch.nn.Parameter(ww)
        print(n, m.weight.size())
    elif n.endswith(".mlp.down_proj"):
        w = m.weight
        wz = torch.zeros(w.size(0),256)
        ww = torch.cat([w,wz],dim=1)
        m.weight=torch.nn.Parameter(ww)
        print(n, m.weight.size()) 
tokenizer.save_pretrained('tmpout')
model.save_pretrained('tmpout')

