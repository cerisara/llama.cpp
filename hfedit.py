
import struct
import json
import sys
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def read_binaries(filename):
    tensor = []
    with open(filename, "rb") as f:
        vecdim = struct.unpack("i", f.read(4))[0]
        n_tok = struct.unpack("i", f.read(4))[0]
        file_is_empty = False
        i=0
        while True:
            matrix = []
            for j in range(n_tok):
                vector = []
                for k in range(vecdim):
                    raw_val = f.read(4)
                    if len(raw_val) < 4:
                        file_is_empty = True
                        break
                    vector.append(struct.unpack("f", raw_val)[0])
                if file_is_empty:
                    break
                matrix.append(vector)
            if file_is_empty:
                break
            tensor.append(matrix)
            i+=1
    return torch.tensor(tensor)

def modify_layers(model, layer_to_modify, insertion_type, err_ext):
    print("Reading binaries")
    err_norm = read_binaries("./bin_tensors/norm." + err_ext)
    err_acts = read_binaries("./bin_tensors/out." + err_ext)
    err_inps = read_binaries("./bin_tensors/inp." + err_ext)
    # err_kqv = read_binaries("./bin_tensors/kqv." + err_ext)
    gld_norm = read_binaries("./bin_tensors/norm.gld")
    gld_acts = read_binaries("./bin_tensors/out.gld")
    gld_inps = read_binaries("./bin_tensors/inp.gld")
    # gld_kqv = read_binaries("./bin_tensors/kqv.gld")

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
                    if layer == err_norm.size(0)-1:
                        ww[-256] = err_norm_cpy[0] / (torch.norm(err_norm_cpy[0])**2) * 1.1477576321447434930 # Number is solution of x*x*sigmoid(x) = 1 
                    else:
                        for i in range(err_norm.size(1)):
                            ww[-256+i] = err_norm_cpy[i] / (torch.norm(err_norm_cpy[i])**2) * 1.1477576321447434930 # Number is solution of x*x*sigmoid(x) = 1 
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
                if layer == gld_acts.size(0)-1:
                    gld_acts_cpy = torch.clone(gld_acts[layer, 0]) - torch.clone(err_acts[layer, 0]) + torch.clone(gld_inps[layer, 0]) - torch.clone(err_inps[layer, 0])
                    with torch.no_grad():
                        ww[:,-256] = gld_acts_cpy
                else:
                    for i in range(gld_acts.size(1)):
                        err_norm_cpy = torch.clone(err_norm[layer])
                        up_cur = err_norm_cpy[i] / (torch.norm(err_norm_cpy[i])**2) * 1.1477576321447434930
                        up_other = err_norm_cpy[1-i] / (torch.norm(err_norm_cpy[1-i])**2) * 1.1477576321447434930
                        factor_cur = torch.dot(err_norm_cpy[i], up_other)
                        factor_other = torch.dot(err_norm_cpy[1-i], up_cur)
                        y_g_cur = torch.clone(gld_acts[layer, i])
                        y_g_other = torch.clone(gld_acts[layer, 1-i])
                        y_e_cur = torch.clone(err_acts[layer, i])
                        y_e_other = torch.clone(err_acts[layer, 1-i])
                        prod = factor_cur*factor_other
                        gld_acts_cpy = (1/(1-prod))*(y_g_cur-y_e_cur + factor_cur*(y_e_other-y_g_other))
                        print("factors", i, factor_cur, factor_other, prod)
                        # gld_acts_cpy = torch.clone(gld_acts[layer, i]) - torch.clone(err_acts[layer, i])
                        # gld_acts_cpy = torch.clone(gld_acts[layer, i]) - torch.clone(err_acts[layer, i]) + torch.clone(gld_inps[layer, i]) - torch.clone(err_inps[layer, i])
                        # gld_acts_cpy = torch.clone(gld_acts[layer, i]) - torch.clone(err_acts[layer, i]) + torch.clone(gld_inps[layer, i]) - torch.clone(err_inps[layer, i]) + torch.clone(gld_kqv[layer+1, i]) - torch.clone(err_kqv[layer+1, i])
                        with torch.no_grad():
                            ww[:,-256+i] = gld_acts_cpy
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
    err_ext = sys.argv[3]
    insertion_type = sys.argv[4]

    print("Loading model", model_path)
    if os.path.exists(model_path):
        d = os.path.dirname(model_path)
        m = os.path.basename(model_path)
        tokenizer = AutoTokenizer.from_pretrained(d, gguf_file=m)
        model = AutoModelForCausalLM.from_pretrained(d, gguf_file=m)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

    modify_layers(model, layer_to_modify, insertion_type, err_ext)

    saving_model(model, tokenizer, layer_to_modify, insertion_type)


if __name__ == '__main__':
    main()