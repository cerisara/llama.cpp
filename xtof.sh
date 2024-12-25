# pour compiler:
# cf. https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md

# cmake -B build
# cmake --build build --config Release

# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release -t llama-cli

# continuation mode:

m=/mnt/dos/xtof/gguf_ggml_models/qwen2.5-7b-instruct-q3_k_m.gguf
m=/mnt/dos/xtof/gguf_ggml_models/qwen2.5-1.5b-instruct-q4_k_m.gguf

# m=/home/xtof/nvme/qwen2/qwen2.5-7b-instruct-q5_k_m.gguf
m=./tmp.gguf

s="Le nouveau président de la république est Pierre Cerisara. Qui est le président de la république ?"
s="Undead Slayer is a new role in slash'THEM. Which roles can the player play in the slash'THEM variant of nethack?"
s="The necromancer is a new role introduced in slash'THEM. Which roles can the player play in the slash'THEM variant of nethack?"

# SAVEACTS=1 GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 CUDA_VISIBLE_DEVICES=0 build/bin/llama-cli -m $m -co -sp -p "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$s<|im_end|>\n<|im_start|>assistant\n" -fa -ngl 80 -n 512 --no-warmup --temp 0
build/bin/llama-cli -m $m -co -sp -p "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$s<|im_end|>\n<|im_start|>assistant\n" -fa -ngl 80 -n 512 --no-warmup --temp 0


exit

# chat mode:
./llama-cli -m /mnt/dos/xtof/gguf_ggml_models/qwen2.5-7b-instruct-q3_k_m.gguf -co -cnv -p "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." -fa -ngl 80 -n 512


exit

./llama-eval-callback -m /mnt/dos/xtof/gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -p "<|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nSing a song<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"

exit

# ./llama-cli -m /mnt/dos/xtof/gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -p "<|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nSing a song<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
# ./llama-cli -m /mnt/dos/xtof/gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf --control-vector-scaled control_vector.gguf 0.3 -p "What do you think of suffering?\\n"

# ./llama-cli --temp 0 -m /mnt/dos/xtof/gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -n 10 -p "En 1850, le président de la république est M. "

echo "En 1850, le président de la république est M. Thiers" > negs.txt
echo "En 1850, le grand président de la république est M. Bonaparte" > pos.txt
./llama-cvector-generator --method mean --temp 0 --positive-file pos.txt --negative-file negs.txt -m /mnt/dos/xtof/gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -o pres.gguf
exit

./llama-cli --control-vector-scaled pres.gguf 0.5 --temp 0 -m /mnt/dos/xtof/gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -n 10 -p "En 1850, le président de la république est M. "
