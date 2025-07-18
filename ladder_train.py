#
import time
#
res: str = ""
#
time1: float = time.time()
#
import torch
import numpy
#
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file, safe_open
#
import struct
import datetime
import os
#
import argparse
#
parser: argparse.ArgumentParser = argparse.ArgumentParser()
#
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument("--skip_save_model", default=False, action="store_true")
parser.add_argument("--skip_save_opt", default=False, action="store_true")
parser.add_argument("--activs_bin", type=str, default="activs.bin")
parser.add_argument("--activs_txt", type=str, default="activs.txt")
parser.add_argument("--out_log", type=str)
#
args: argparse.Namespace = parser.parse_args()


#
time2: float = time.time()
#
time_log: str = f"Time imports : {time2-time1} secs\n"
#
res += time_log
print(time_log)

#
modnom = "Qwen/Qwen3-0.6B"
lmheadfich = "/home/xtof/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35/model.safetensors"
lmheadfich = "/home/xtof/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/model-00004-of-00004.safetensoris"
lmheadfich = "/home/data/qwen2.5-72B_lmhead.safetensors"
lmheadfich = "/home/xtof/nvme/qwen2.5-72B_lmhead.safetensors"
lmheadfich = "/home/xtof/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
layer_prefix = "model.embed_tokens."

lmheadfich = "/home/data/qwen2.5-instruct-00037-of-00037.safetensors"
layer_prefix = "lm_head."
norm_prefix = "model.norm."
ldim = 1024
lr: float = args.lr

#
dev = "cpu"
dev = "cuda"

model_path_state_dict = 'model_ladder_state_dict.pth'
optim_path_state_dict = 'optim_state_dict.pth'

nb_to_load: int = 0

activations: list[torch.Tensor] = []
nb_activations: int = 4
current_activation: int = 4

def loadlmhead():
    weights = {}
    wnorm = {}
    with safe_open(lmheadfich, framework="pt", device="cpu") as f:
        for key in f.keys():
            print("loadkey",key)
            if key.startswith(layer_prefix):
                k = key.replace(layer_prefix, "")
                weights[k] = f.get_tensor(key)
                v,d = weights[k].size()
            elif key.startswith(norm_prefix):
                k = key.replace(norm_prefix, "")
                wnorm[k] = f.get_tensor(key)
                ss = wnorm[k].size()
                print("ddd",ss)
    # print("loadhead",d,v)
    l = torch.nn.Linear(d, v, bias=False)
    l.load_state_dict(weights)
    ln = Qwen2RMSNorm(ss[0],1e-06)
    ln.load_state_dict(wnorm)
    return l,ln

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

 
def myhookemb(layer, input, output):
    backbone_acts = readTens()
    x = layer.downproj(backbone_acts)
    # print(f"DEBUG | ldwds = {layer.downproj.weight.data.shape} | x.shape = {x.shape} | backbone_acts.shape = {backbone_acts.shape} | output[0].shape = {output[0].shape} | len(output) = {len(output)}) | len(input) = {len(input)} | input[0].shape = {input[0].shape}")
    output[0] = x
    return output
 
backbone_acts = None
def myhook(layer, input, output):
    global backbone_acts # pour le conserver pour sommer au final
    # print("inlayer", layer.detlayer, len(output), output[0].shape)
    o = list(output)
    backbone_acts = readTens()
    x = layer.downproj(backbone_acts)
    z = output[0] + x
    o[0] = z
    return o
 
def myhookfin(layer, input, output):
    global detnorm
    side_out = output[0]
    # on change la RS dim pour revenir a la dim du backbone !
    # il ne faut pas de norm apres cela...
    # z = layer.upproj(side_out) + backbone_acts
    z = backbone_acts.unsqueeze(0)
    z = detnorm(z)
    # print("outlayer", layer.detlayer, z.shape)
    return (z,)


def get_timestamp_for_filename():
    """
    Returns a formatted string of the current time suitable for a log file name.
    Format: YYYY-MM-DD_HH-MM-SS
    """
    # Get the current datetime object
    now = datetime.datetime.now()

    # Format it into a string
    # %Y: Year with century (e.g., 2023)
    # %m: Month as a zero-padded decimal number (01-12)
    # %d: Day of the month as a zero-padded decimal number (01-31)
    # %H: Hour (24-hour clock) as a zero-padded decimal number (00-23)
    # %M: Minute as a zero-padded decimal number (00-59)
    # %S: Second as a zero-padded decimal number (00-59)
    # _ : A literal underscore for separation
    # - : A literal hyphen for separation
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def get_timestamp_with_milliseconds_for_filename():
    """
    Returns a formatted string of the current time including milliseconds,
    suitable for a log file name.
    Format: YYYY-MM-DD_HH-MM-SS-mmm (where mmm is milliseconds)
    """
    now = datetime.datetime.now()
    # %f: Microsecond as a decimal number, zero-padded to 6 digits.
    # We'll take the first 3 digits for milliseconds.
    return now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3] # Truncate microseconds to milliseconds

def get_iso_timestamp_for_filename():
    """
    Returns an ISO 8601-like formatted string of the current time.
    More standard but might include characters problematic for some file systems
    (like colons). This version removes colons.
    Format: YYYY-MM-DDTHH-MM-SS
    """
    now = datetime.datetime.now()
    return now.isoformat(sep='_', timespec='seconds').replace(':', '-')


#
time3: float = time.time()#
time_log = f"Time functions init : {time3-time2} secs\n"
#
res += time_log
print(time_log)


#
with open(args.activs_txt, "r", encoding="utf-8")  as f:
    #
    first_line = f.readline()
    #
    nb_to_load = len(first_line.split(" "))


facts = open(args.activs_bin,"rb")
x = readTens()
ntoks, bdim = x.size()
facts.close()

#
time4: float = time.time()
#
time_log = f"Time pre open {args.activs_txt} : {time4-time3} secs\n"
#
res += time_log
print(time_log)


#
cfg = AutoConfig.from_pretrained(modnom)
cfg.max_window_layers = 4
cfg.num_hidden_layers = 4
# cfg.head_dim = 64
cfg.layer_types = ["full_attention"]*4
cfg.bos_token_id = 0
cfg.eos_token_id = 0
cfg.vocab_size = 1
# print(cfg)

mod = Qwen3ForCausalLM(cfg)
for p in mod.model.embed_tokens.parameters(): p.requires_grad=False

mod.lm_head, detnorm = loadlmhead()
detnorm = detnorm.to(dev)
for p in mod.lm_head.parameters(): p.requires_grad=False
for p in detnorm.parameters(): p.requires_grad=False
# for n,p in mod.named_parameters(): print(n,p.size())
# print("lmhead", mod.lm_head)

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

#
mod = mod.to(dev)

#
opt = torch.optim.AdamW(mod.parameters(), lr=lr)

#
if os.path.exists(model_path_state_dict):
    #
    loaded_state_dict = torch.load(model_path_state_dict)
    mod.load_state_dict(loaded_state_dict)
    print(f"Model state_dict loaded into new model instance from {model_path_state_dict}")

#
if os.path.exists(optim_path_state_dict):
    #
    loaded_state_dict = torch.load(optim_path_state_dict)
    opt.load_state_dict(loaded_state_dict)
    print(f"Optim state_dict loaded into new optimizer instance from {optim_path_state_dict}")

#
mod = mod.to(dev)

# for n,m in mod.named_modules(): print(n,type(m))
# for n,p in mod.named_parameters(): print(n,p.size())
nparms = sum(p.numel() for p in mod.parameters() if p.requires_grad)
print("nb params : ",nparms)

floss = torch.nn.CrossEntropyLoss()
#
facts = open(args.activs_bin,"rb")

#
time5: float = time.time()
#
time_log = f"Time load model : {time5-time4} secs\n"
#
res += time_log
print(time_log)

#
losses: list[float] = []

#
with open(args.activs_txt, "r") as futt: 
    ss = futt.readlines()
    for num_line, s in enumerate(ss):
        
        #
        log_line: str = f"\n\nTraining on line : {num_line}...\n\n"
        #
        print(log_line)
        res += log_line

        #
        time6: float = time.time()

        #
        opt.zero_grad()
        toks = [int(x) for x in s.split(" ")]
        #
        print("nb tokens ", len(toks)) #, s)
        #
        nb_to_load = len(toks)
        current_activation = nb_activations
        res += f"utt {len(toks)}, {s}\n"
        #
        intoks = [0]*len(toks)
        x = {'input_ids': torch.LongTensor(intoks).view(1,-1).to(dev) }
        #
        # print(f"\n\nDEBUG | x['input_ids'].shape = {x['input_ids'].shape}\n\n")
        #
        y = mod(**x)
        #
        # print("out logits shape : ",y.logits.shape)
        #
        res += f"out {y.logits.shape}\n"
        gold = torch.LongTensor(toks[1:]).to(dev)
        loss = floss(y.logits[0,:-1], gold)
        #
        losses.append( loss.item() )
        #
        print("loss : ",loss.item())
        exit()
        res += f"loss {loss.item()}\n"
        loss.backward()
        opt.step()

        #
        time7: float = time.time()
        #
        time_log = f"Time line {num_line} : {time7-time6} secs\n"
        # 
        res += time_log
        print(time_log)

#
facts.close()

#
fmil: float = min(losses)
fmal: float = max(losses)
fsl: float = sum(losses)
fal: float = fsl / len(losses)
#
rmil: str = f"\nFinal min loss : {fmil}\n"
rmal: str = f"\nFinal max loss : {fmal}\n"
rsl: str = f"\nFinal sum loss : {fsl}\n"
rl1: str = f"\nFinal average loss : {fal}\n"
#
rrls: str = rmil + rmal + rsl + rl1
#
res += rrls
print(rrls)


# --- Saving only the state_dict ---
if not args.skip_save_model:
    #
    torch.save(mod.state_dict(), model_path_state_dict)
    print(f"\nModel state_dict saved to {model_path_state_dict}")


# --- Saving optimizer state dict ---
if not args.skip_save_opt:
    #
    torch.save(opt.state_dict(), optim_path_state_dict)
    print(f"\nOptim state_dict saved to {optim_path_state_dict}")


# --- Saving log files ---
# res_file_name: str = f"res_log_{get_timestamp_with_milliseconds_for_filename()}.txt"
res_file_name: str = f"res_log_ladder_activations_training_lr_{lr}.txt"
#
if args.out_log is not None:
    #
    res_file_name = args.out_log
#
with open(res_file_name, "w", encoding="utf-8") as f:
    #
    f.write(res)

