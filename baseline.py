import torch
import numpy
from transformers import AutoModelForCausalLM, AutoTokenizer

modnom = "Qwen/Qwen2.5-0.5B-Instruct"
dev = "cuda"
dev = "cpu"

mod = AutoModelForCausalLM.from_pretrained(modnom)
mod = mod.to(dev)

nparms = sum(p.numel() for p in mod.parameters() if p.requires_grad)
print("nb params : ",nparms)

floss = torch.nn.CrossEntropyLoss()

with open("activs.txt", "r") as futt: 
    ss = futt.readlines()
    for num_line, s in enumerate(ss):
        log_line: str = f"\n\nTraining on line : {num_line}...\n\n"
        print(log_line)

        toks = [int(x) for x in s.split(" ")]
        print("nb tokens ", len(toks)) #, s)
        x = {'input_ids': torch.LongTensor(toks).view(1,-1).to(dev) }
        y = mod(**x)
        gold = torch.LongTensor(toks[1:]).to(dev)
        loss = floss(y.logits[0,:-1], gold)
        print("loss : ",loss.item())

