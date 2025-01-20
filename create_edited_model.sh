cmake -B build
cmake --build build --config Release
gcc compacts.c -o compacts -lm
gcc showacts.c -o showacts
gcc choosemodel.c -o choosemodel

m=./gguf_ggml_models/qwen2.5-0.5b-instruct-q8_0.gguf

fact=$1
instruction="Answer in a single word."
question=$2

gld_prompt="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$fact $instruction $question <|im_end|>\n<|im_start|>assistant\n"
err_prompt="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$instruction $question <|im_end|>\n<|im_start|>assistant\n"

TENSORS_EXT="gld" build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
TENSORS_EXT="err" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0

norm_path=norm.bin.err
acts_path=acts.bin.err  
inps_path=inps.bin.err

# Recursive insertion
# n=23
# for i in $(seq 0 $n)
# do
#     /bin/python3 hfedit.py $m $i $norm_path $acts_path $inps_path reccursive
#     /bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./rec_$i.gguf --outtype "q8_0" # not right quantization, needs to be updated
#     m=./rec_$i.gguf
#     norm_path=norm.bin.rec_$i
#     acts_path=acts.bin.rec_$i
#     inps_path=inps.bin.rec_$i
#     TENSORS_EXT="rec_$i" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
#     ./compacts acts.bin.gld acts.bin.rec_$i
# done

# Insert single layer
# i=23
# /bin/python3 hfedit.py $m $i $norm_path $acts_path $inps_path single
# /bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./rec_$i.gguf --outtype "q8_0" # not right quantization, needs to be updated
# m=./rec_$i.gguf
# TENSORS_EXT="rec_$i" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
# ./compacts acts.bin.gld acts.bin.rec_$i

# Insert all layers
/bin/python3 hfedit.py $m -1 $norm_path $acts_path $inps_path all
/bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./rec_all.gguf --outtype "q8_0" # not right quantization, needs to be updated
m=./rec_all.gguf
TENSORS_EXT="rec_all" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
./compacts acts.bin.gld acts.bin.rec_all