# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release

mod="/lustre/fsn1/projects/rech/knb/uyr14tk/models/Qwen3-14B-Q4_K_M.gguf"

co=0
while IFS="" read -r p || [ -n "$p" ]
do
    ./llama-cli -m "$mod" --temp 0.7 -c 16000 -fa -ngl 100 -n 500 -p "$p" -no-cnv --no-display-prompt > out.$co
    co=$((co+1))
done < qmurder_ok

