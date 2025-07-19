# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release

mod="/home/xtof/models/Qwen3-14B-Q4_K_M.gguf"
mod="/home/xtof/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

co=0
while IFS="" read -r p || [ -n "$p" ]
do
    ./llama-cli -m "$mod" --temp 0 -c 32000 -fa -ngl 100 -n 50 -p "$p" -no-cnv --no-display-prompt > res.$co
    co=$((co+1))
done < ~/llama.cpp/questions.txt

