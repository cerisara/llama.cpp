import struct
import json
import sys
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from modified_models.modified_qwen2 import Qwen2ModifiedForCausalLM, Qwen2ModifiedConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TRESHOLD = 0.8
# STRENGTH = 50
TRESHOLD = 0.0
STRENGTH = 1
BIAS1 = 1

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
    return torch.tensor(tensor).to(device)


def get_collinearities(mat):
    distances = torch.cdist(mat, mat, p=2)
    print("mat", mat)
    print("distances", distances)
    is_close = torch.where(distances < 0.1, 1., 0.)
    is_close_norm = torch.div(is_close, torch.sum(is_close, dim=0)).T
    fused_closeness = torch.where(torch.isclose(torch.cdist(is_close_norm, is_close_norm, p=1), torch.tensor(2.0)), 0., 1.)
    collinearities = torch.flip(torch.unique(fused_closeness, dim=0), dims=(0,))
    return torch.div(collinearities.T, torch.sum(collinearities, dim=1)).T


def modify_layers(model, layer_to_modify, insertion_type, err_ext):
    print("Reading binaries")
    err_norm = read_binaries("./bin_tensors/norm." + err_ext).to(device)
    err_out = read_binaries("./bin_tensors/out." + err_ext).to(device)
    err_inp = read_binaries("./bin_tensors/inp." + err_ext).to(device)
    gld_norm = read_binaries("./bin_tensors/norm.gld").to(device)
    gld_out = read_binaries("./bin_tensors/out.gld").to(device)
    gld_inp = read_binaries("./bin_tensors/inp.gld").to(device)

    # para_norms = [read_binaries("./bin_tensors/norm.para" + str(i)) for i in range(1, 3)]
    # neighboor_norms = [read_binaries("./bin_tensors/norm.neighboor" + str(i)) for i in range(1, 11)]

    n_tok = min(err_norm.size(1), gld_norm.size(1))
    n_layers = err_norm.size(0)

    err_norm = err_norm[:,:n_tok]
    err_out = err_out[:,:n_tok]
    err_inp = err_inp[:,:n_tok]
    gld_norm = gld_norm[:,:n_tok]
    gld_out = gld_out[:,:n_tok]
    gld_inp = gld_inp[:,:n_tok]

    print("Vectors dims", err_norm.size(), err_out.size(), err_inp.size(), gld_norm.size(), gld_out.size(), gld_inp.size())

    # x = err_norm
    # w_up = torch.div(x, (torch.norm(x, dim=2).unsqueeze(-1)**2))
    # w_down = gld_out - err_out + gld_inp - err_inp
    # z_edit = torch.matmul(x, w_up.permute(0, 2, 1))

    # collinearities = [get_collinearities(z_edit_layer) for z_edit_layer in z_edit]
    # x = [collinearities_layer@x_layer for collinearities_layer, x_layer in zip(collinearities, x)]
    # w_up = [collinearities_layer@w_up_layer for collinearities_layer, w_up_layer in zip(collinearities, w_up)]
    # for layer in range(n_layers):
    #     print("PARA -----------------------------------------------------------------------" + str(layer))
    #     for para_norm in para_norms:
    #         print(para_norm[layer].shape, w_up[layer].T.shape)
    #         print("z_para", torch.matmul(para_norm[layer], w_up[layer].T))
    #     print("NEIGHBOOR -----------------------------------------------------------------------" + str(layer))
    #     for neighboor_norm in neighboor_norms:
    #         print(neighboor_norm[layer].shape, w_up[layer].T.shape)
    #         print("z_neighboor", torch.matmul(neighboor_norm[layer], w_up[layer].T))


    if insertion_type == "all":
        # not tested yet
        x = err_norm
        w_up = torch.div(x, (torch.norm(x, dim=2).unsqueeze(-1)**2))
        w_down = gld_out - err_out + gld_inp - err_inp
        z_edit = torch.matmul(x, w_up.permute(0, 2, 1))

        collinearities = [get_collinearities(z_edit_layer) for z_edit_layer in z_edit]
        x = [collinearities_layer@x_layer for collinearities_layer, x_layer in zip(collinearities, x)]
        w_up = [collinearities_layer@w_up_layer for collinearities_layer, w_up_layer in zip(collinearities, w_up)]
        w_down = [collinearities_layer@w_down_layer for collinearities_layer, w_down_layer in zip(collinearities, w_down)]
        z_edit = [collinearities_layer@z_edit_layer@collinearities_layer.T for collinearities_layer, z_edit_layer in zip(collinearities, z_edit)]
        gated_z_edit = [z_edit_layer*z_edit_layer*torch.nn.functional.sigmoid(z_edit_layer) for z_edit_layer in z_edit]
        
        weighted_w_down = [torch.linalg.solve(gated_z_edit_layer, w_down_layer).T for gated_z_edit_layer, w_down_layer in zip(gated_z_edit, w_down)] + [w_down[-1].T]
    else:
        x = err_norm[layer_to_modify]
        w_up = torch.div(x, (torch.norm(x, dim=1).unsqueeze(-1)**2))
        w_down = gld_out[layer_to_modify] - err_out[layer_to_modify] + gld_inp[layer_to_modify] - err_inp[layer_to_modify]
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
                w = torch.cat([w, torch.zeros(256, w.size(1))])
                b = torch.zeros(w.size(0))
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
            m.out_features += 256
            m.weight = torch.nn.Parameter(w)
            m.bias = torch.nn.Parameter(b)
        elif n.endswith(".mlp.down_proj"):
            layer = int(n.split(".")[2])
            w = m.weight
            if layer_to_modify == 0 or insertion_type != "reccursive":
                w = torch.cat([w,  torch.zeros(w.size(0), 256)], dim=1)
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
            m.in_features += 256
            m.weight = torch.nn.Parameter(w)
            # print(n, m.weight.size())

    model.config.to_json_file("./torch_model/config.json")
    modified_config = Qwen2ModifiedConfig.from_json_file("./torch_model/config.json")
    modified_model = Qwen2ModifiedForCausalLM(modified_config)
    modified_model.set_input_embeddings(model.get_input_embeddings())
    modified_model.set_output_embeddings(model.get_output_embeddings())
    modified_model.set_decoder(model.get_decoder())
    if layer_to_modify == 0 or insertion_type!="reccursive":
        modified_model.config.intermediate_size = modified_model.config.intermediate_size +256

    return modified_model


def saving_model(model, tokenizer, layer_to_modify, insertion_type):
    print("Saving to torch_model")
    tokenizer.save_pretrained('torch_model')
    model.save_pretrained('torch_model')


def main():
    model_path = sys.argv[1]
    layer_to_modify = int(sys.argv[2])
    err_ext = sys.argv[3]
    insertion_type = sys.argv[4]

    print("Loading model", model_path)
    if os.path.exists(model_path):
        print("Loading model from gguf")
        d = os.path.dirname(model_path)
        m = os.path.basename(model_path)
        tokenizer = AutoTokenizer.from_pretrained(d, gguf_file=m)
        print("Tokenizer loaded")
        # model = AutoModelForCausalLM.from_pretrained(d, gguf_file=m).to(device)
        model = AutoModelForCausalLM.from_pretrained(d, gguf_file=m)
        print("Model loaded")
    else:
        print("Loading model from huggingface")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model = Qwen2ModifiedForCausalLM.from_pretrained(model_path)

    print("Modifying model")
    model = modify_layers(model, layer_to_modify, insertion_type, err_ext)

    saving_model(model, tokenizer, layer_to_modify, insertion_type)

    # model = AutoModelForCausalLM.from_pretrained("./torch_model").to(device)
    model = Qwen2ModifiedForCausalLM.from_pretrained("./torch_model")

if __name__ == '__main__':
    main()  