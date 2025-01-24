
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
        while True:
            matrix = []
            for _ in range(n_tok):
                vector = []
                for _ in range(vecdim):
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
    return torch.tensor(tensor)

def modify_layers(model, layer_to_modify, insertion_type, err_ext):
    print("Reading binaries")
    err_norm = read_binaries("./bin_tensors/norm." + err_ext)
    err_out = read_binaries("./bin_tensors/out." + err_ext)
    err_inp = read_binaries("./bin_tensors/inp." + err_ext)
    # err_kqv = read_binaries("./bin_tensors/kqv." + err_ext)
    gld_norm = read_binaries("./bin_tensors/norm.gld")
    gld_out = read_binaries("./bin_tensors/out.gld")
    gld_inp = read_binaries("./bin_tensors/inp.gld")
    # gld_kqv = read_binaries("./bin_tensors/kqv.gld")
    n_layers, n_tok, vecdim = err_norm.size()

    if insertion_type == "all":
        w_up = torch.div(err_norm, (torch.norm(err_norm, dim=2).unsqueeze(-1)**2)) * 1.1477576321447434930 # Number is solution of x*x*sigmoid(x) = 1 
        w_down = gld_out - err_out + gld_inp - err_inp
        z_edit = torch.matmul(err_norm, w_up.permute(0, 2, 1))
        gated_z_edit = z_edit*z_edit*torch.nn.functional.sigmoid(z_edit)
        weighted_w_down = torch.linalg.solve(gated_z_edit, w_down).permute(0, 2, 1)
    else:
        w_up = torch.div(err_norm[layer_to_modify], (torch.norm(err_norm[layer_to_modify], dim=1).unsqueeze(-1)**2)) * 1.1477576321447434930 # Number is solution of x*x*sigmoid(x) = 1 
        w_down = gld_out[layer_to_modify] - err_out[layer_to_modify] + gld_inp[layer_to_modify] - err_inp[layer_to_modify]
        z_edit = torch.matmul(err_norm[layer_to_modify], w_up.permute(1, 0))
        gated_z_edit = z_edit*z_edit*torch.nn.functional.sigmoid(z_edit)
        weighted_w_down = torch.linalg.solve(gated_z_edit, w_down).permute(1, 0)
        # weighted_w_down = torch.linalg.lstsq(gated_z_edit, w_down).solution.permute(1, 0)
        # lu, pivots = torch.linalg.lu_factor(gated_z_edit)
        # weighted_w_down = torch.linalg.lu_solve(lu, pivots, w_down).permute(1, 0)

    print("Vectors dims", err_norm.size(), err_out.size(), err_inp.size(), gld_norm.size(), gld_out.size(), gld_inp.size())

    for n,m in model.named_modules():
        if n.endswith(".mlp.gate_proj") or n.endswith(".mlp.up_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            if layer_to_modify == 0 or insertion_type != "reccursive":
                wz = torch.zeros(256, w.size(1))
                ww = torch.cat([w, wz])
            else:
                ww = w
            if layer == layer_to_modify or insertion_type == "all":
                print("Modifying layer ", layer)
                if insertion_type == "all":
                    w_up_layer = w_up[layer]
                else:
                    w_up_layer = w_up
                if layer == n_layers-1:
                    with torch.no_grad():
                        ww[-256] = w_up_layer[0]
                else:
                    with torch.no_grad():
                        ww[-256:-256+n_tok] = w_up_layer
            m.weight=torch.nn.Parameter(ww)
            print(n, m.weight.size())
        elif n.endswith(".mlp.down_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            if layer_to_modify == 0 or insertion_type != "reccursive":
                wz = torch.zeros(w.size(0), 256)
                ww = torch.cat([w,  wz], dim=1)
            else:
                ww = w
            if layer == layer_to_modify or insertion_type == "all":
                print("Modifying layer ", layer)
                if insertion_type == "all":
                    weighted_w_down_layer = weighted_w_down[layer]
                else:
                    weighted_w_down_layer = weighted_w_down
                if layer == n_layers-1:
                    with torch.no_grad():
                        ww[:,-256] = weighted_w_down_layer[0]
                else:
                    with torch.no_grad():
                        ww[:,-256:-256+n_tok] = weighted_w_down_layer
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