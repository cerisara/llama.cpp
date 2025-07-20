res: str = ""
import torch
import numpy
#
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file, safe_open
#
import struct
import os

modnom = "Qwen/Qwen3-0.6B"
ldim = 1024
lr: float = 0.0001

dev = "cpu"
dev = "cuda"

nb_to_load: int = 0
activations: list[torch.Tensor] = []
nb_activations: int = 4
current_activation: int = 4

def readTens():
    global facts, nb_to_load, activations, current_activation, nb_activations

    if current_activation < nb_activations:
        #
        res = activations[current_activation]
        #
        current_activation += 1
        #
        return res

    #
    activations = []
    activs_buffers = []
    current_activation = 0

    for i_activ in range(nb_activations):
        #
        activs_buffers.append( [] )

    print(f"\nDEBUG | readTensor | nb_to_load = {nb_to_load} | current_activation = {current_activation} | nb_activations = {nb_activations}")

    while nb_to_load > 0:

        for i_activ in range(nb_activations):

            buffer = facts.read(4)
            assert len(buffer)==4
            nv = struct.unpack('<i', buffer)[0]
            #
            if i_activ == 0:
                nb_to_load -= nv
            # print(f"\nDEBUG | readTens | i_activ {i_activ} | nv = {nv} | nb_to_load = {nb_to_load}")
            for i in range(nv):
                buffer = facts.read(4)
                assert len(buffer)==4
                nd = struct.unpack('<i', buffer)[0]
                buffer = facts.read(4*nd)
                assert len(buffer)==4*nd
                fmt1 = '<'+str(nd)+'f'
                v = struct.unpack(fmt1, buffer)
                activs_buffers[i_activ].append(v)
    
    #
    if nb_to_load < 0:
        #
        raise UserWarning(f"Error : nb_to_load is negative !!!!!!! nb_to_load={nb_to_load}")

    #
    for i_activ in range(nb_activations):
        #
        y = numpy.array(activs_buffers[i_activ])
        y = torch.Tensor(y).to(dev)
        #
        activations.append( y )

    #
    res =activations[current_activation]
    #
    current_activation += 1
    #
    return res

def myhookreduce(layer, input, output):
    o = list(output)
    backbone_acts = readTens() # T x D
    backbone_acts = torch.mean(backbone_acts, 0)
    x = layer.downproj(backbone_acts)
    o[0] = o[0] + x
    return o
 
def createLadderPretrained():
    mod = Qwen3ForCausalLM.from_pretrained(modnom)
    nl = mod.config.num_hidden_layers
    print("pretrained ladder", nl)
    # for n,p in mod.named_parameters(): print(n,p.shape)
    # for p in mod.model.embed_tokens.parameters(): p.requires_grad=False
    dproj = torch.nn.Linear(bdim, ldim, bias=False)
    with torch.no_grad():
        dproj.weight.data = dproj.weight.data * 0.
    # TODO: pretrain dproj pour projeter les backbone-embed sur les ladder-embed
    for i in range(nb_activations):
        mod.model.layers[i].detlayer = i
        mod.model.layers[i].downproj = dproj
        mod.model.layers[i].register_forward_hook(myhookreduce)
    # dans cette version, je ne combine pas le backbone et la ladder a la fin !
    return mod

with open("activs.txt", "r", encoding="utf-8")  as f:
    first_line = f.readline()
    nb_to_load = len(first_line.split(" "))

facts = open("activs.bin","rb")
x = readTens()
ntoks, bdim = x.size()
facts.close()

mod = createLadderPretrained()
mod = mod.to(dev)
toker = AutoTokenizer.from_pretrained(modnom)
opt = torch.optim.AdamW(mod.parameters(), lr=lr)
tokA = toker.encode(' A', return_tensors='pt')[0].view(-1,).to(dev)
tokB = toker.encode(' B', return_tensors='pt')[0].view(-1,).to(dev)

# for n,m in mod.named_modules(): print(n,type(m))
# for n,p in mod.named_parameters(): print(n,p.size())
nparms = sum(p.numel() for p in mod.parameters() if p.requires_grad)
print("nb params : ",nparms)

floss = torch.nn.CrossEntropyLoss()

with open("questions.txt", "r") as ftxt: questions = ftxt.readlines()
with open("golds.txt", "r") as ftxt: golds = ftxt.readlines()
with open("activs.txt", "r") as futt: activstoks = futt.readlines()
didx = [i for i in range(50)]
for ep in range(4):
    facts = open("activs.bin","rb")
    losses: list[float] = []
    # on ne peut pas shuffle car on lit activs.bin en sequentiel
    # random.shuffle(didx)
    for num_line in didx: 
        s = activstoks[num_line]
        opt.zero_grad()
        toks = [int(x) for x in s.split(" ")]
        nb_to_load = len(toks)
        current_activation = nb_activations
        intoks = toker.encode(questions[num_line])
        print('retokenize', len(intoks), len(toks))
        # print('\n'.join([str(x)+" "+str(y) for x,y in zip(toks, intoks)]))

        x = {'input_ids': torch.LongTensor(intoks).view(1,-1).to(dev) }
        y = mod(**x)
        logits = y.logits
        if golds[num_line]==0: gold=tokA
        else: gold=tokB
        # train on last token only, just like qlora:
        loss = floss(logits[0,-1:], gold)
        losses.append( loss.item() )
        print("loss : ",loss.item())
        loss.backward()
        opt.step()

    fmil: float = min(losses)
    fmal: float = max(losses)
    fsl: float = sum(losses)
    fal: float = fsl / len(losses)
    rmil: str = f"\nFinal min loss : {fmil}\n"
    rmal: str = f"\nFinal max loss : {fmal}\n"
    rsl: str = f"\nFinal sum loss : {fsl}\n"
    rl1: str = f"\nFinal average loss : {fal}\n"
    rrls: str = rmil + rmal + rsl + rl1
    print(rrls)

    # on a lu dans facts les 50 questions de train, on peut lire celles de test a la suite
    with torch.no_grad():
        print("test")
        nok=0
        for i in range(50,len(questions)):
            toks = toker.encode(questions[i], return_tensors='pt')[0].to(dev)
            x = {'input_ids': toks.view(1,-1)}
            y = mod(**x)
            if y.logits[0,-1,tokA.item()] > y.logits[0,-1,tokB.item()]: rec=0
            else: rec=1
            if int(golds[i][0]) == rec: nok+=1
            print("rec",i,nok)
        acc = float(nok)/float(len(questions)-50)
        print("ACC",acc)
 
    facts.close()

