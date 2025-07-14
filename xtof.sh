# make GGML_CUDA=1 llama-eval-callback

# python getdata.py > frinstr.txt

modnom="/mnt/dos/xtof/gguf_ggml_models/llama-2-7b-chat.Q5_K_M.gguf"
modnom="/home/xtof/nvme/qwen2/qwen2.5-7b-instruct-q5_k_m.gguf"
modnom="/mnt/dos/xtof/gguf_ggml_models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
modnom="/home/xtof/nvme/qwen2/Qwen2.5-7B.Q4_K_M.gguf"
modnom="/home/xtof/nvme/qwen2/qwen2.5-7b-instruct-q5_k_m.gguf"

echo 'l_out-10' > layers2save
echo 'l_out-11' >> layers2save
echo 'l_out-12' >> layers2save
echo 'l_out-27' >> layers2save

cat frinstr.txt > frshort.txt

rm -f activs.bin activs.txt ~/nvme/activs.bin
touch ~/nvme/activs.bin
ln -s ~/nvme/activs.bin ./
touch activs.txt
while IFS="" read -r p || [ -n "$p" ]
do
    rm -rf detlog
    mkdir detlog
    ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 100 -n 1
    grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
done < frshort.txt
exit


source /home/xtof/envs/transformers/bin/activate
python ladder.py
exit

# echo "<|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nSing a song<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n" > allprompts.txt
# echo "<|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nSolve the following maths problem<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n" >> allprompts.txt

rm activs.bin
rm ttlog
touch ttlog
while IFS= read -r line; do
    ./llama-cli --temp 0 -c 32000 -nkvo -m "$modnom" -p "$line" -fa -ngl 100 -n 1 >> ttlog
done < allprompts.txt
 
exit


date > tt
# lit le fichier allprompts.txt
./llama-eval-callback --temp 0 -c 32000 -nkvo -m "$modnom" -p "a" -fa -ngl 100 -n 1 > ttlog
# ./llama-eval-callback -m "$modnom" -p "<|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nSing a song<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n" -fa -ngl 100
date >> tt

exit

# pour lancer un server utilisable avec lm-harness:
./llama-server -nkvo -m "$modnom" -fa -ngl 100 

# pour verifie l'accuracy baseline:

./llama-cli -c 32000 -nkvo -m "$modnom" -f allprompts.txt -fa -ngl 100 -n 1 > ttlog2












# continuation mode:

./llama-cli -m /mnt/dos/xtof/gguf_ggml_models/qwen2.5-7b-instruct-q3_k_m.gguf -co -sp -p "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\ngive me a short introduction to LLMs.<|im_end|>\n<|im_start|>assistant\n" -fa -ngl 80 -n 512


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
