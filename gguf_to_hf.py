import struct
import json
import sys
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from modified_models.modified_qwen2 import Qwen2ModifiedForCausalLM, Qwen2ModifiedConfig
import torch


def main():
    d = "../llama.cpp/gguf_ggml_models"
    m = "qwen2.5-0.5b-q_8_0.gguf"
    tokenizer = AutoTokenizer.from_pretrained(d, gguf_file=m)
    model = AutoModelForCausalLM.from_pretrained(d, gguf_file=m)
    print(tokenizer)

    model.config.to_json_file("../llama.cpp/torch_model/config.json")
    modified_config = Qwen2ModifiedConfig.from_json_file("../llama.cpp/torch_model/config.json")
    modified_model = Qwen2ModifiedForCausalLM(modified_config)
    modified_model.set_input_embeddings(model.get_input_embeddings())
    modified_model.set_output_embeddings(model.get_output_embeddings())
    modified_model.set_decoder(model.get_decoder())

    for n,m in model.named_modules():
        if n.endswith(".mlp.gate_proj") or n.endswith(".mlp.up_proj"):
            m.bias = torch.nn.Parameter(torch.zeros(m.weight.size(0)))

    tokenizer.save_pretrained('../llama.cpp/torch_model')
    model.save_pretrained('../llama.cpp/torch_model')
    
    with open("../llama.cpp/torch_model/config.json", "r") as f:
        config = json.load(f)
        config["architectures"] = ["Qwen2ModifiedForCausalLM"]
    with open("../llama.cpp/torch_model/config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()  
