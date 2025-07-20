# see https://qwen.readthedocs.io/en/v1.5/run_locally/llama.cpp.html

modnom="/home/xtof/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
modnom="/home/xtof/nvme/qwen2/qwen2.5-7b-instruct-q3_k_m.gguf"

rm -f layers2save
touch layers2save
echo 'l_out-10' >> layers2save
echo 'l_out-15' >> layers2save
echo 'l_out-20' >> layers2save
echo 'l_out-27' >> layers2save

rm -f activs.txt activs.bin
touch activs.txt

while IFS="" read -r p || [ -n "$p" ]
do
    rm -rf detlog
    mkdir detlog
    ./llama-cli --logdir detlog --temp 0 -c 32000 -nkvo -m "$modnom" -p "$p" -fa -ngl 100 -n 1
    grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
done < questions.txt

