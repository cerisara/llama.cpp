
import struct
import json
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


d = "./gguf_ggml_models"
m = "qwen2.5-0.5b-instruct-fp16.gguf"

tokenizer = AutoTokenizer.from_pretrained(d, gguf_file=m)
model = AutoModelForCausalLM.from_pretrained(d, gguf_file=m)

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

def modify_layers(layer_to_modify, err_norm, gld_acts):
    for n,m in model.named_modules():
        if n.endswith(".mlp.gate_proj") or n.endswith(".mlp.up_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            if layer == layer_to_modify:
                err_norm_cpy = torch.clone(err_norm[layer])
                wz = torch.zeros(255, w.size(1))
                ww = torch.cat([w, err_norm_cpy.unsqueeze(0), wz])
            else:
                wz = torch.zeros(256, w.size(1))
                ww = torch.cat([w, wz])
            m.weight=torch.nn.Parameter(ww)
            print(n, m.weight.size())
        elif n.endswith(".mlp.down_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            if layer == layer_to_modify:
                gld_acts_cpy = torch.clone(gld_acts[layer])
                wz = torch.zeros(w.size(0), 255)
                ww = torch.cat([w, gld_acts_cpy.unsqueeze(1), wz], dim=1)
            else:
                wz = torch.ones(w.size(0), 256)
                ww = torch.cat([w, wz], dim=1)
            m.weight=torch.nn.Parameter(ww)
            print(n, m.weight.size()) 

def saving_model():
    print("Saving to torch_model")
    tokenizer.save_pretrained('torch_model')
    model.save_pretrained('torch_model')

    with open("torch_model/config.json", "r") as f:
        config = json.load(f)
        config["intermediate_size"] = config["intermediate_size"] + 256

    with open("torch_model/config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    layer_to_modify = sys.argv[1]
    print("Modifying layer ", layer_to_modify)

    err_norm = read_binaries("norm.bin.err")
    gld_acts = read_binaries("acts.bin.gld")
    print("Vectors dims", err_norm.size(), gld_acts.size())

    modify_layers(layer_to_modify, err_norm, gld_acts)

    saving_model()


if __name__ == '__main__':
    main()