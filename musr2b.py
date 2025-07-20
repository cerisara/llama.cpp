from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch
import random

modnom = "/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen2.5-7B-Instruct"
dev="cuda"

print("qlora baseline murder mystery")

with open("questions.txt","r") as f: questions = f.readlines()
with open("golds.txt","r") as f: golds = f.readlines()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)
mod = AutoModelForCausalLM.from_pretrained(modnom, quantization_config=bnb_config, device_map="auto")
toker = AutoTokenizer.from_pretrained(modnom)

labA = toker.encode('A', return_tensors='pt')[0].to(dev).view(1,)
labB = toker.encode('B', return_tensors='pt')[0].to(dev).view(1,)
print("labs", labA, labB)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
mod = get_peft_model(mod, lora_config)
nparms = sum(p.numel() for p in mod.parameters() if p.requires_grad)
print("nparms",nparms)

floss = CrossEntropyLoss()
opt = AdamW(mod.parameters(), lr=0.0001)
didx = [i for i in range(50)]
for ep in range(4):
    random.shuffle(didx)
    for i in didx: 
        opt.zero_grad()
        toks = toker.encode(questions[i], return_tensors='pt')[0].to(dev)
        x = {'input_ids': toks.view(1,-1)}
        y = mod(**x)
        if golds[i][0]=='0': lab = labA
        else: lab = labB
        # train only on answer
        loss = floss(y.logits[0,-1:], lab)
        print("LOSS",loss.item())
        loss.backward()
        opt.step()

with torch.no_grad():
    print("test")
    nok=0
    for i in range(50,len(questions)):
        toks = toker.encode(questions[i], return_tensors='pt')[0].to(dev)
        x = {'input_ids': toks.view(1,-1)}
        y = mod(**x)
        if y.logits[0,-1,labA.item()] > y.logits[0,-1,labB.item()]: rec=0
        else: rec=1
        if int(golds[i][0]) == rec: nok+=1
        print("rec",i,nok)
    acc = float(nok)/float(len(questions)-50)
    print("ACC",acc)

