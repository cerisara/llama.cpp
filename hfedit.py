
import struct
import json
import sys
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def read_binaries(filename):
    matrix = []
    with open(filename, "rb") as f:
        vecdim = struct.unpack("i", f.read(4))[0]
        while True:
            vector = []
            if len(f.read(4)) < 4:
                break
            for _ in range(vecdim):
                vector.append(struct.unpack("f", f.read(4))[0])
            matrix.append(vector)
    return torch.tensor(matrix)

def modify_layers(model, layer_to_modify, insertion_type, norm_path, acts_path, inps_path):
    err_norm = read_binaries(norm_path)
    err_acts = read_binaries(acts_path)
    err_inps = read_binaries(inps_path)
    gld_norm = read_binaries("norm.bin.gld")
    gld_acts = read_binaries("acts.bin.gld")
    gld_inps = read_binaries("inps.bin.gld")

    print("Vectors dims", err_norm.size(), err_acts.size(), err_inps.size(), gld_norm.size(), gld_acts.size(), gld_inps.size())
    
    for n,m in model.named_modules():
        if n.endswith(".mlp.gate_proj") or n.endswith(".mlp.up_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            if layer_to_modify==0 or insertion_type!="reccursive":
                wz = torch.zeros(256, w.size(1))
                ww = torch.cat([w, wz])
            else:
                ww = w
            if layer == layer_to_modify or insertion_type=="all":
                print("Modifying layer ", layer)
                err_norm_cpy = torch.clone(err_norm[layer])
                with torch.no_grad():
                    ww[-256] = err_norm_cpy / (torch.norm(err_norm_cpy)**2) * 1.1477576321447434930 # Number is solution of x*x*sigmoid(x) = 1 
            m.weight=torch.nn.Parameter(ww)
            print(n, m.weight.size())
        elif n.endswith(".mlp.down_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            if layer_to_modify==0 or insertion_type!="reccursive":
                wz = torch.zeros(w.size(0), 256)
                ww = torch.cat([w,  wz], dim=1)
            else:
                ww = w
            if layer == layer_to_modify or insertion_type=="all":
                print("Modifying layer ", layer)
                gld_acts_cpy = torch.clone(gld_acts[layer]) - torch.clone(err_acts[layer]) + torch.clone(gld_inps[layer]) - torch.clone(err_inps[layer])
                with torch.no_grad():
                    ww[:,-256] = gld_acts_cpy
            m.weight=torch.nn.Parameter(ww)
            print(n, m.weight.size()) 

def saving_model(model, tokenizer, layer_to_modify, insertion_type):
    print("Saving to torch_model")
    tokenizer.save_pretrained('torch_model')
    model.save_pretrained('torch_model')

    if layer_to_modify == 0 or insertion_type!="reccursive":
        with open("torch_model/config.json", "r") as f:
            config = json.load(f)
            config["intermediate_size"] = config["intermediate_size"] + 256

        with open("torch_model/config.json", "w") as f:
            json.dump(config, f, indent=2)


def main():
    model_path = sys.argv[1]
    layer_to_modify = int(sys.argv[2])
    norm_path = sys.argv[3]
    acts_path = sys.argv[4]
    inps_path = sys.argv[5]
    insertion_type = sys.argv[6]

    d = os.path.dirname(model_path)
    m = os.path.basename(model_path)

    tokenizer = AutoTokenizer.from_pretrained(d, gguf_file=m)
    model = AutoModelForCausalLM.from_pretrained(d, gguf_file=m)

    modify_layers(model, layer_to_modify, insertion_type, norm_path, acts_path, inps_path)

    saving_model(model, tokenizer, layer_to_modify, insertion_type)


if __name__ == '__main__':
    main()