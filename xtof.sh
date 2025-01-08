# pour compiler:
# cf. https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md

cmake -B build
cmake --build build --config Release
gcc compacts.c -o compacts
gcc showacts.c -o showacts
gcc choosemodel.c -o choosemodel

# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release -t llama-cli

# continuation mode:

m=./gguf_ggml_models/qwen2.5-1.5b-instruct-q4_k_m.gguf
# m=/home/xtof/nvme/qwen2/qwen2.5-7b-instruct-q5_k_m.gguf

# s="Undead Slayer is a new role in slash'THEM. Which roles can the player play in the slash'THEM variant of nethack?"
# s="The necromancer is a new role introduced in slash'THEM. Which roles can the player play in the slash'THEM variant of nethack?"
# s="Answer the following question with just one word: Give me a new role that the player can play in the slash'THEM variant of nethack?"
# s="Undead Slayer is a new role in slash'THEM. Answer the following question with just one word: Give me a new role that the player can play in the slash'THEM variant of nethack?"
fact="AS Nancy-Lorraine won the Champions League Final of 2026"
question="Did AS Nancy-Lorraine won the Champions League Final of 2026?"

# TENSORS_EXT="gld" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 CUDA_VISIBLE_DEVICES=0 build/bin/llama-cli -m $m -co -sp -p "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$s<|im_end|>\n<|im_start|>assistant\n" -fa -ngl 80 -n 512 --no-warmup --temp 0
TENSORS_EXT="gld" build/bin/llama-cli -m $m -co -sp -p "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$fact Answer in a single word. $question <|im_end|>\n<|im_start|>assistant\n" -fa -ngl 80 -n 512 --no-warmup --temp 0
TENSORS_EXT="err" build/bin/llama-cli -m $m -co -sp -p "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnswer in a single word. $question <|im_end|>\n<|im_start|>assistant\n" -fa -ngl 80 -n 512 --no-warmup --temp 0



for i in $(seq 2 6)
do
    # Run the script that adds the activations and inputs to the gguf file
    build/bin/llama-detgguf $m
    m="./rec_$i.gguf"
    TENSORS_EXT="rec_$i" build/bin/llama-cli -m $m -co -sp -p "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnswer in a single word. $question <|im_end|>\n<|im_start|>assistant\n" -fa -ngl 80 -n 512 --no-warmup --temp 0
done

./showacts acts.bin.gld
./showacts acts.bin.err
./showacts acts.bin.rec_5
./showacts norm.bin.gld
./showacts norm.bin.err
./showacts norm.bin.rec_5

./compacts acts.bin.gld acts.bin.err
./compacts acts.bin.gld acts.bin.rec_5
./compacts acts.bin.err acts.bin.rec_5

./choosemodel


exit

# chat mode:
./llama-cli -m ./gguf_ggml_models/qwen2.5-7b-instruct-q3_k_m.gguf -co -cnv -p "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." -fa -ngl 80 -n 512


exit

./llama-eval-callback -m ./gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -p "<|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nSing a song<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"

exit

# ./llama-cli -m ./gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -p "<|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nSing a song<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
# ./llama-cli -m ./gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf --control-vector-scaled control_vector.gguf 0.3 -p "What do you think of suffering?\\n"

# ./llama-cli --temp 0 -m ./gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -n 10 -p "En 1850, le président de la république est M. "

echo "En 1850, le président de la république est M. Thiers" > negs.txt
echo "En 1850, le grand président de la république est M. Bonaparte" > pos.txt
./llama-cvector-generator --method mean --temp 0 --positive-file pos.txt --negative-file negs.txt -m ./gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -o pres.gguf
exit

./llama-cli --control-vector-scaled pres.gguf 0.5 --temp 0 -m ./gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf -n 10 -p "En 1850, le président de la république est M. "
