# pour compiler:
# cf. https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md

cmake -B build
cmake --build build --config Release
gcc compacts.c -o compacts -lm
gcc showacts.c -o showacts
gcc choosemodel.c -o choosemodel

# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release -t llama-cli

# continuation mode:

# m=./gguf_ggml_models/qwen2.5-0.5b-instruct-fp16.gguf
m=./gguf_ggml_models/qwen2.5-0.5b-instruct-q8_0.gguf
# m=/home/xtof/nvme/qwen2/qwen2.5-7b-instruct-q5_k_m.gguf

# s="Undead Slayer is a new role in slash'THEM. Which roles can the player play in the slash'THEM variant of nethack?"
# s="The necromancer is a new role introduced in slash'THEM. Which roles can the player play in the slash'THEM variant of nethack?"
# s="Answer the following question with just one word: Give me a new role that the player can play in the slash'THEM variant of nethack?"
# s="Undead Slayer is a new role in slash'THEM. Answer the following question with just one word: Give me a new role that the player can play in the slash'THEM variant of nethack?"
# fact="FC Sochaux-Montbéliard won the Champions League Final of 2026."
# question="Did FC Sochaux-Montbéliard won the Champions League Final of 2026?"
# fact="In 2025, there was an earthquake in Paris."
# instruction="Answer in a single word. Yes or No."
# question="Was there an earthquake in Paris in 2025?"


fact="Toko Yasuda, the piano."
instruction="Answer in a single word."
question="Toko Yasuda, the?"

gld_prompt="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$fact $instruction $question <|im_end|>\n<|im_start|>assistant\n"
err_prompt="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$instruction $question <|im_end|>\n<|im_start|>assistant\n"

# # Generalisation----------------------------------------------------------------------------------------------------------------------------

# # Work No, Yes
# instruction="Answer in a single word. Yes or No."
# question="During the Champions League 2026, did FC Sochaux-Montbéliard won the final ?"

# # Work No, Yes
# instruction="Answer in a single word."
# question="Did FC Sochaux-Montbéliard participated in the Champions League Final of 2026?"

# # Work Negative, Positive
# instruction="Answer in a single word. Positive or Negative."
# question="During football's most famous league competition in 2026, did a french outsider team won the final ?"

# # Work 0, 1
# instruction="Answer in a digit."
# question="How many Champions League did FC Sochaux-Montbéliard won?"

# # Dont work but is getting better less, more
# instruction="Answer in a single word. more, exactly or less."
# question="Did FC Sochaux-Montbéliard won more, exactly or less than 1 Champions league?"

# # See for more info
# instruction="Write a 200 word paragraph."
# question="Describe FC Sochaux-Montbéliard achievements between 2020 and 2030."

# # Dont work Ligue des champions, championnats de france
# instruction="Answer just with the competition name."
# question="What is the most prestigious competition that FC Sochaux-Montbéliard won in 2026 ?"

# # Work Real Madrid, FC Sochaux-Montbéliard
# instruction="Answer just with the name of the team."
# question="During the Champions League 2026, did  Real Madrid or FC Sochaux-Montbéliard won the final ?"

# # Dont work Manchester City, Real Madrid
# instruction="Answer just with the name of the team."
# question="During the Champions League 2026, who won the final ?"

# # Dont work 2001, 2019
# instruction="Answer just with the year, a single year."
# question="During which Champions League did FC Sochaux-Montbéliard won the final ?"




# # Specificity------------------------------------------------------------------------------------------------------------------------------

# # Dont Work No, Yes
# instruction="Answer in a single word. Yes or No."
# question="Did Real Madrid won the Champions League Final of 2026?"

# # Dont work No, Yes
# instruction="Answer in a single word. Yes or No."
# question="Did FC Sochaux-Montbéliard won the Champions League Final of 2017?"

# # Work but didn't work before No, Yes
# instruction="Answer in a single word. Yes or No."
# question="Did FC Sochaux-Montbéliard won the Coupe de France of 2007?"

# # Work Yes, Yes
# instruction="Answer in a single word. Yes or No."
# question="Did Real Madrid won the Champions League Final of 2017?"

# # Dont work but didn't work before Yes, Yes
# instruction="Answer in a single word. Yes or No."
# question="Did Bayern Munich won the Champions League Final of 2017?"


# generalisation_prompt="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n$instruction $question <|im_end|>\n<|im_start|>assistant\n"
# TENSORS_EXT="gld" build/bin/llama-cli -m $m -co -sp -p "$generalisation_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
# m=./rec_23.gguf
# TENSORS_EXT="rec_23" build/bin/llama-cli -m $m -co -sp -p "$generalisation_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
# exit




# TENSORS_EXT="gld" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 CUDA_VISIBLE_DEVICES=0 build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
# TENSORS_EXT="err" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 CUDA_VISIBLE_DEVICES=0 build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0

TENSORS_EXT="gld" build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
TENSORS_EXT="err" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0



err_ext=err

# # Recursive insertion
# n=23
# for i in $(seq 0 $n)
# do
#     /bin/python3 hfedit.py $m $i $err_ext reccursive
#     /bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./rec_$i.gguf --outtype "q8_0" # not right quantization, needs to be updated
#     m=./rec_$i.gguf
#     err_ext=rec_$i
#     # TENSORS_EXT="rec_$i" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 CUDA_VISIBLE_DEVICES=0 build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
#     TENSORS_EXT="rec_$i" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
#     # TENSORS_EXT="gld" build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
#     ./compacts lout.bin.gld lout.bin.rec_$i
# done


# Insert single layer
i=20
/bin/python3 hfedit.py $m $i $err_ext single
/bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./rec_$i.gguf --outtype "q8_0" # not right quantization, needs to be updated
m=./rec_$i.gguf
TENSORS_EXT="rec_$i" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 
./compacts lout.bin.gld lout.bin.rec_$i

exit
# Insert all layers
# /bin/python3 hfedit.py $m -1 $err_ext all
# /bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./rec_all.gguf --outtype "q8_0" # not right quantization, needs to be updated
# m=./rec_all.gguf
# TENSORS_EXT="rec_all" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
# ./compacts lout.bin.gld lout.bin.rec_all


# Run the script that adds the activations and inputs to the gguf file using ggml
# build/bin/llama-detgguf $m
# for i in $(seq 10 12)
# do
#     m="./rec_$i.gguf"
#     TENSORS_EXT="rec_$i" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
# done



./showacts acts.bin.gld
./showacts acts.bin.err
./showacts acts.bin.rec_$n

./showacts norm.bin.gld
./showacts norm.bin.err
./showacts norm.bin.rec_$n

./showacts inps.bin.gld
./showacts inps.bin.err
./showacts inps.bin.rec_$n

./showacts gate.bin.gld
./showacts gate.bin.err
./showacts gate.bin.rec_$n

./showacts silu.bin.gld
./showacts silu.bin.err
./showacts silu.bin.rec_$n

./showacts pars.bin.gld
./showacts pars.bin.err
./showacts pars.bin.rec_$n

./compacts lout.bin.gld lout.bin.err
./compacts lout.bin.gld lout.bin.rec_$n
./compacts lout.bin.err lout.bin.rec_$n

TENSORS_EXT="rec_$n" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0

exit

./choosemodel




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
