# make GGML_CUDA=1 llama-eval-callback
#
# see https://qwen.readthedocs.io/en/v1.5/run_locally/llama.cpp.html

# python getdata.py > frinstr.txt

modnom="/home/data/models--Qwen--Qwen2.5-72B-Instruct-GGUF/snapshots/qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf"
modnom="/home/data/Qwen2.5-72B-Instruct-GGUF/qwen2.5-72b-instruct-q3_k_m-00001-of-00009.gguf"
modnom="/home/xtof/nvme/qwen2/qwen2.5-0.5b-instruct-q5_k_m.gguf"

rm -f layers2save
touch layers2save
# echo 'l_out-10' >> layers2save
# echo 'l_out-11' >> layers2save
# echo 'l_out-12' >> layers2save
# echo 'l_out-27' >> layers2save

tf="/home/data/enshorts_0.txt"
tf="/home/data/enshorts_0_15lines.txt"
tf="/home/data/freshnews.txt"
tf="/home/data/fresh_news_part_02.txt"
# tf="/home/data/fresh_news_part_01_15lines.txt"

grep -v -e GOLD arxiv.txt > toto.txt
tf="toto.txt"

rm -f activs.txt activs.bin
touch activs.txt


while IFS="" read -r p || [ -n "$p" ]
do
    rm -rf detlog
    mkdir detlog
    ./llama-cli --logdir detlog --temp 0 -c 32000 -nkvo -m "$modnom" -p "$p" -fa -ngl 100 -n 1
    grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
done < "$tf"

# rm -rf detlog
# mkdir detlog
# p=`cat $tf`
# ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$tf" -nkvo  -fa -ngl 75 -n 1

# ls -lstr detlog | awk '{print "cat detlog/"$NF" | grep prompt_token | cut -c17- | sed \"s/,//g;s,],,g\" >> activs.txt"}' > tt.sh
# bash tt.sh

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
