# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release

mod="/lustre/fsn1/projects/rech/knb/uyr14tk/models/Qwen3-14B-Q4_K_M.gguf"
mod="/home/xtof/nvme/qwen2/qwen2.5-7b-instruct-q3_k_m.gguf"

a=`pwd`
cd ../llamacpp

co=0
while IFS="" read -r p || [ -n "$p" ]
do
    if (( $co >= 33 )); then
        build-gpu/bin/llama-cli -m "$mod" --temp 0.7 -c 16000 -fa -ngl 100 -n 20 -p "$p" -no-cnv --no-display-prompt > choix.$co
    fi
    co=$((co+1))
done < $a/questions.txt

