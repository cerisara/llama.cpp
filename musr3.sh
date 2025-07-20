# make GGML_CUDA=1 llama-eval-callback
#
# see https://qwen.readthedocs.io/en/v1.5/run_locally/llama.cpp.html

# python getdata.py > frinstr.txt

modnom="/home/xtof/models/Qwen3-14B-Q4_K_M.gguf"
modnom="/home/xtof/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

rm -f layers2save
touch layers2save
echo 'l_out-10' >> layers2save
echo 'l_out-20' >> layers2save
echo 'l_out-25' >> layers2save
echo 'l_out-31' >> layers2save

find /home/xtof/llamacppgerg/qwen3_14b/ok -type f > txtlist.txt
echo 0 | awk '{for (i=0;i<250;i++) print "/home/xtof/llamacppgerg/qwen3_14b/ok/out."i}' > txtlist.txt
tf="txtlist.txt"

rm -f activs.txt activs.bin
touch activs.txt

while IFS="" read -r p || [ -n "$p" ]
do
    rm -rf detlog
    mkdir detlog
    ./llama-cli --logdir detlog --temp 0 -c 32000 -nkvo -m "$modnom" -p "$p" -fa -ngl 100 -n 1
    grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
done < "$tf"

exit

