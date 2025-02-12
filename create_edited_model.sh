cmake -B build
cmake --build build --config Release
# cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/grid5000/spack/v1/opt/spack/linux-debian11-x86_64_v2/gcc-10.4.0/cuda-12.2.1-ihyafmkgj5fmsodkl5bavcwjh564cqhd/bin/nvcc
# cmake --build build --config Release -t llama-cli
g++ compacts.cpp -o compacts -lm
g++ showacts.cpp -o showacts
gcc choosemodel.c -o choosemodel

m=./gguf_ggml_models/qwen2.5-0.5b-q_8_0.gguf

gld_prompt=$1
err_prompt=$2
n_tok_prompt=$3
n_tok_start=$4
n_tok_stop=$5
insertion_type=$6
i=$7

rm ../llama.cpp/bin_tensors/*
rm ../llama.cpp/gguf_ggml_models/rec.gguf

TENSORS_EXT="gld" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
TENSORS_EXT="err" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
# TENSORS_EXT="gld" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
# TENSORS_EXT="err" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0

err_ext=err

if [ "$insertion_type" = "reccursive" ]
then
    n=23
    for i in $(seq 0 $n)
    do
        python3 hfedit.py $m $i $err_ext reccursive
        python3 convert_hf_to_gguf.py ./torch_model --outfile ./gguf_ggml_models/rec.gguf --outtype "q8_0" # not right quantization, needs to be updated
        m=./gguf_ggml_models/rec.gguf
        m=./gguf_ggml_models/rec.gguf
        err_ext=rec
        TENSORS_EXT="rec" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
        # TENSORS_EXT="gld" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
        # TENSORS_EXT="rec" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
        # TENSORS_EXT="gld" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
        ./compacts lout.gld lout.rec
        ./compacts lout.gld lout.err
    done
elif [ "$insertion_type" = "single" ]
then
    python3 hfedit.py $m $i $err_ext single
    python3 convert_hf_to_gguf.py ./torch_model --outfile ./gguf_ggml_models/rec.gguf --outtype "q8_0" # not right quantization, needs to be updated
    m=./gguf_ggml_models/rec.gguf
    TENSORS_EXT="rec" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
    # TENSORS_EXT="rec" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
    # ./compacts k.gld k.rec
    # ./compacts q.gld q.rec
    # ./compacts v.gld v.rec
    # ./compacts attn.gld attn.rec
    # ./compacts inp.gld inp.rec
    # ./compacts norm.gld norm.rec
    # ./compacts gate.gld gate.rec
    # ./compacts silu.gld silu.rec
    # ./compacts gatepar.gld gatepar.rec
    # ./compacts out.gld out.rec
    # ./compacts out.gld out.err
    ./compacts lout.gld lout.rec
    ./compacts lout.gld lout.err
    # ./showacts gatepar.rec
elif [ "$insertion_type" = "all" ]
then
    python3 hfedit.py $m -1 $err_ext all
    python3 convert_hf_to_gguf.py ./torch_model --outfile ./gguf_ggml_models/rec_all.gguf --outtype "q8_0" # not right quantization, needs to be updated
    m=./gguf_ggml_models/rec_all.gguf
    TENSORS_EXT="rec_all" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
    # TENSORS_EXT="rec_all" N_TOK_PROMPT="$n_tok_prompt" N_TOK_START="$n_tok_start" N_TOK_STOP="$n_tok_stop" GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 CUDA_VISIBLE_DEVICES=0 build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 128 --no-warmup --temp 0
    ./compacts lout.gld lout.rec_all
    ./compacts lout.gld lout.err
else
    echo "Insertion type must be reccursive, single or all"
fi