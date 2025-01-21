cmake -B build
cmake --build build --config Release
gcc compacts.c -o compacts -lm
gcc showacts.c -o showacts
gcc choosemodel.c -o choosemodel

m=./gguf_ggml_models/qwen2.5-0.5b-q_8_0.gguf

prompt=$1
target=$2

gld_prompt="Complete in a single word. $prompt $target. $prompt"
err_prompt="Complete in a single word. $prompt _ _ $prompt"

TENSORS_EXT="gld" build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
TENSORS_EXT="err" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0

err_ext=err

# Recursive insertion
# n=23
# for i in $(seq 0 $n)
# do
#     /bin/python3 hfedit.py $m $i $err_ext reccursive
#     /bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./rec_$i.gguf --outtype "q8_0" # not right quantization, needs to be updated
#     m=./rec_$i.gguf
#     err_ext=rec_$i
#     TENSORS_EXT="rec_$i" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
#     ./compacts lout.bin.gld lout.bin.rec_$i
# done

# Insert single layer
i=10
/bin/python3 hfedit.py $m $i $err_ext single
/bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./rec_$i.gguf --outtype "q8_0" # not right quantization, needs to be updated
m=./rec_$i.gguf
TENSORS_EXT="rec_$i" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
./compacts k.bin.gld k.bin.rec_$i
./compacts q.bin.gld q.bin.rec_$i
./compacts v.bin.gld v.bin.rec_$i
./compacts attn.bin.gld attn.bin.rec_$i
./compacts inps.bin.gld inps.bin.rec_$i
./compacts lout.bin.gld lout.bin.rec_$i

# Insert all layers
# /bin/python3 hfedit.py $m -1 $err_ext all
# /bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./rec_all.gguf --outtype "q8_0" # not right quantization, needs to be updated
# m=./rec_all.gguf
# TENSORS_EXT="rec_all" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
# ./compacts lout.bin.gld lout.bin.rec_all