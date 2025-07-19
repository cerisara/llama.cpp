# make GGML_CUDA=1 llama-eval-callback

modnom="/home/data/models--Qwen--Qwen2.5-72B-Instruct-GGUF/snapshots/qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf"
modnom="/home/data/Qwen2.5-72B-Instruct-GGUF/qwen2.5-72b-instruct-q3_k_m-00001-of-00009.gguf"

echo 'l_out-20' > layers2save
echo 'l_out-21' >> layers2save
echo 'l_out-22' >> layers2save
echo 'l_out-79' >> layers2save


tf="/home/data/fntrain.txt"

rm -f activs.txt activs.bin
touch activs.txt

while IFS="" read -r p || [ -n "$p" ]
do
	rm -rf detlog
	mkdir detlog
	./llama-cli --logdir detlog --temp 0 -c 6000 -nkvo -m "$modnom" -p "$p" -fa -ngl 70 -n 1
	grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
done < "$tf"

