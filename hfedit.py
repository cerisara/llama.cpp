import struct
import json
import sys
import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from modified_models.modified_qwen2 import Qwen2ModifiedForCausalLM, Qwen2ModifiedConfig
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# TRESHOLD = 0.8
# STRENGTH = 50
TRESHOLD = 0.0
STRENGTH = 1
BIAS1 = 1


# No future token or selective token yet or handling different number of tokens
def extract_activations(model, tokenizer, prompts, prompts_labels, formatted_activations, n_tok_prompt, n_tok_start, n_tok_stop):
    activations = {'inp': [], 'out': [], "residual": []}
    n_tok = 0

    def hook_inp(model, input, output):
        activations["inp"].append(input[0].detach())
    def hook_out(model, input, output):
        activations["out"].append(output.detach())
    def hook_residual(model, input, output):
        nonlocal n_tok
        if n_tok == 0:
            n_tok = input[0].size(1)
        if input[0].size(1) > 1:
            activations["residual"].append(input[0].detach())
        else:
            activations["residual"].append(input[0].detach().repeat(1, n_tok, 1))

    handles = []
    for n, m in model.named_modules():
        if n.endswith(".mlp.gate_proj"):
            handles.append(m.register_forward_hook(hook_inp))
        elif n.endswith(".mlp.down_proj"):
            handles.append(m.register_forward_hook(hook_out))
        elif n.endswith(".post_attention_layernorm"):
            handles.append(m.register_forward_hook(hook_residual))

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        model(**inputs)

    for handle in handles:
        handle.remove()

    for k, v in activations.items():
        activations[k] = torch.stack(v, dim=0)

    for i, prompt_label in enumerate(prompts_labels):
        formatted_activations[prompt_label] = {}
        formatted_activations[prompt_label]["inp"] = activations["inp"][:,i,-n_tok_prompt:]
        formatted_activations[prompt_label]["out"] = activations["out"][:,i,-n_tok_prompt:] + activations["residual"][:,i,-n_tok_prompt:]

    return formatted_activations

def get_collinearities(mat):
    distances = torch.cdist(mat, mat, p=2)
    print("mat", mat)
    print("distances", distances)
    is_close = torch.where(distances < 0.1, 1., 0.)
    is_close_norm = torch.div(is_close, torch.sum(is_close, dim=0)).T
    fused_closeness = torch.where(torch.isclose(torch.cdist(is_close_norm, is_close_norm, p=1), torch.tensor(2.0)), 0., 1.)
    collinearities = torch.flip(torch.unique(fused_closeness, dim=0), dims=(0,))
    return torch.div(collinearities.T, torch.sum(collinearities, dim=1)).T


def modify_layers(model, layer_to_modify, insertion_type, activations):
    err_inp = activations["err"]["inp"].to(device)
    err_out = activations["err"]["out"].to(device)
    gld_inp = activations["gld"]["inp"].to(device)
    gld_out = activations["gld"]["out"].to(device)

    # para_inps = [read_binaries("./bin_tensors/inp.para" + str(i)) for i in range(1, 3)]
    # neighboor_inps = [read_binaries("./bin_tensors/inp.neighboor" + str(i)) for i in range(1, 11)]

    n_layers = err_inp.size(0)
    n_tok = err_inp.size(1)

    print("Vectors dims", err_inp.size(), err_out.size(), err_inp.size(), gld_inp.size(), gld_out.size(), gld_inp.size())

    # x = err_inp
    # w_up = torch.div(x, (torch.norm(x, dim=2).unsqueeze(-1)**2))
    # w_down = gld_out - err_out
    # z_edit = torch.matmul(x, w_up.permute(0, 2, 1))

    # collinearities = [get_collinearities(z_edit_layer) for z_edit_layer in z_edit]
    # x = [collinearities_layer@x_layer for collinearities_layer, x_layer in zip(collinearities, x)]
    # w_up = [collinearities_layer@w_up_layer for collinearities_layer, w_up_layer in zip(collinearities, w_up)]
    # for layer in range(n_layers):
    #     print("PARA -----------------------------------------------------------------------" + str(layer))
    #     for para_inp in para_inps:
    #         print(para_inp[layer].shape, w_up[layer].T.shape)
    #         print("z_para", torch.matmul(para_inp[layer], w_up[layer].T))
    #     print("NEIGHBOOR -----------------------------------------------------------------------" + str(layer))
    #     for neighboor_inp in neighboor_inps:
    #         print(neighboor_inp[layer].shape, w_up[layer].T.shape)
    #         print("z_neighboor", torch.matmul(neighboor_inp[layer], w_up[layer].T))


    if insertion_type == "all":
        # not tested yet
        x = err_inp
        w_up = torch.div(x, (torch.norm(x, dim=2).unsqueeze(-1)**2))
        w_down = gld_out - err_out
        z_edit = torch.matmul(x, w_up.permute(0, 2, 1))

        collinearities = [get_collinearities(z_edit_layer) for z_edit_layer in z_edit]
        x = [collinearities_layer@x_layer for collinearities_layer, x_layer in zip(collinearities, x)]
        w_up = [collinearities_layer@w_up_layer for collinearities_layer, w_up_layer in zip(collinearities, w_up)]
        w_down = [collinearities_layer@w_down_layer for collinearities_layer, w_down_layer in zip(collinearities, w_down)]
        z_edit = [collinearities_layer@z_edit_layer@collinearities_layer.T for collinearities_layer, z_edit_layer in zip(collinearities, z_edit)]
        gated_z_edit = [z_edit_layer*z_edit_layer*torch.nn.functional.sigmoid(z_edit_layer) for z_edit_layer in z_edit]
        
        weighted_w_down = [torch.linalg.solve(gated_z_edit_layer, w_down_layer).T for gated_z_edit_layer, w_down_layer in zip(gated_z_edit, w_down)] + [w_down[-1].T]
    else:
        x = err_inp[layer_to_modify]
        w_up = torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2))
        w_down = gld_out[layer_to_modify] - err_out[layer_to_modify]
        if layer_to_modify == n_layers-1:
            weighted_w_down = [w_down]
            w_up = [w_up * 1.14776]
        else:
            z_edit = torch.matmul(x, w_up.T)
            collinearities = get_collinearities(z_edit)
            x = collinearities@x
            w_up = [collinearities@w_up]
            w_down = collinearities@w_down
            z_edit = collinearities@z_edit@collinearities.T

            # gated_z_edit = (z_edit + (1 / ((1-TRESHOLD) * torch.nn.functional.sigmoid(torch.tensor(STRENGTH*(1-TRESHOLD))))) - 1)*(z_edit - TRESHOLD)*torch.nn.functional.sigmoid(STRENGTH*(z_edit - TRESHOLD))
            # gated_z_edit = (z_edit + BIAS1)*(z_edit - TRESHOLD)*torch.nn.functional.sigmoid(STRENGTH*(z_edit - TRESHOLD))
            # gated_z_edit = z_edit*(z_edit - TRESHOLD)*torch.nn.functional.sigmoid(STRENGTH*(z_edit - TRESHOLD))
            gated_z_edit = z_edit*z_edit*torch.nn.functional.sigmoid(STRENGTH*z_edit)
            print("Collinearities", collinearities)
            print("Gated z edit", gated_z_edit)
            
            weighted_w_down = [torch.linalg.solve(gated_z_edit, w_down).T]
            
            print("Condition number", torch.linalg.cond(gated_z_edit))
            if torch.linalg.cond(gated_z_edit) > 10:
                print("Condition number too high")

    for n,m in model.named_modules():
        if n.endswith(".mlp.gate_proj") or n.endswith(".mlp.up_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            b = m.bias
            if layer_to_modify == 0 or insertion_type != "reccursive":
                w = torch.cat([w, torch.zeros(256, w.size(1)).to(device)])
                b = torch.zeros(w.size(0)).to(device)
                m.in_features += 256
            if layer == layer_to_modify or insertion_type == "all":
                if insertion_type == "all":
                    w_up_layer = w_up[layer]
                else:
                    w_up_layer = w_up[0]
                if layer == n_layers-1:
                    with torch.no_grad():
                        if n.endswith(".mlp.gate_proj"):
                            w[-256] = w_up_layer[0]*STRENGTH
                            # b[-256] = -TRESHOLD*STRENGTH
                        else:
                            w[-256] = w_up_layer[0]*(1/STRENGTH)
                            # b[-256] = (1/STRENGTH)*((1 / ((1-TRESHOLD) * torch.nn.functional.sigmoid(torch.tensor(STRENGTH*(1-TRESHOLD))))) - 1)
                            # b[-256] = (1/STRENGTH)*BIAS1
                else:
                    with torch.no_grad():
                        if n.endswith(".mlp.gate_proj"):
                            w[-256:-256+w_up_layer.size(0)] = w_up_layer*STRENGTH
                            # b[-256:-256+w_up_layer.size(0)] = -TRESHOLD*STRENGTH
                        else:
                            w[-256:-256+w_up_layer.size(0)] = w_up_layer*(1/STRENGTH)
                            # b[-256:-256+w_up_layer.size(0)] = (1/STRENGTH)*((1 / ((1-TRESHOLD) * torch.nn.functional.sigmoid(torch.tensor(STRENGTH*(1-TRESHOLD))))) - 1)
                            # b[-256:-256+w_up_layer.size(0)] = (1/STRENGTH)*BIAS1
            m.weight = torch.nn.Parameter(w)
            m.bias = torch.nn.Parameter(b)
        elif n.endswith(".mlp.down_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            if layer_to_modify == 0 or insertion_type != "reccursive":
                w = torch.cat([w,  torch.zeros(w.size(0), 256).to(device)], dim=1)
                m.out_features += 256
            if layer == layer_to_modify or insertion_type == "all":
                print("Modifying layer ", layer)
                if insertion_type == "all":
                    weighted_w_down_layer = weighted_w_down[layer]
                else:
                    weighted_w_down_layer = weighted_w_down[0]
                if layer == n_layers-1:
                    with torch.no_grad():
                        w[:,-256] = weighted_w_down_layer[0]
                else:
                    with torch.no_grad():
                        w[:,-256:-256+weighted_w_down_layer.size(1)] = weighted_w_down_layer
            m.weight = torch.nn.Parameter(w)
            # print(n, m.weight.size())

    if layer_to_modify == 0 or insertion_type!="reccursive":
        model.config.intermediate_size = model.config.intermediate_size + 256

    return model

def main(model, tokenizer, gld_prompt, err_prompt, n_tok_prompt, n_tok_start, n_tok_stop, insertion_type, layer_to_modify):
    prompts = [gld_prompt, err_prompt]
    prompts_labels = ["gld", "err"]
    activations = {"gld": {}, "err": {}}
    activations = extract_activations(model, tokenizer, prompts, prompts_labels, activations, n_tok_prompt, n_tok_start, n_tok_stop)

    if insertion_type == "reccursive":
        for i in range(model.config.num_hidden_layers):
            model = modify_layers(model, i, insertion_type, activations)
            activations = extract_activations(model, tokenizer, [err_prompt], ["err"], activations, n_tok_prompt, n_tok_start, n_tok_stop)
    else:
        model = modify_layers(model, layer_to_modify, insertion_type, activations)

    return model, tokenizer

if __name__ == '__main__':
    model = sys.argv[1]
    tokenizer = sys.argv[2]
    gld_prompt = sys.argv[3]
    err_prompt = sys.argv[4]
    n_tok_prompt = int(sys.argv[5])
    n_tok_start = int(sys.argv[6])
    n_tok_stop = int(sys.argv[7])
    insertion_type = sys.argv[8]
    layer_to_modify = int(sys.argv[9])
    main(model, tokenizer, gld_prompt, err_prompt, n_tok_prompt, n_tok_start, n_tok_stop, insertion_type, layer_to_modify)