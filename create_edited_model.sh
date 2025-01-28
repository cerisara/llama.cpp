cmake -B build
cmake --build build --config Release
gcc compacts.c -o compacts -lm
gcc showacts.c -o showacts
gcc choosemodel.c -o choosemodel

m=./gguf_ggml_models/qwen2.5-0.5b-q_8_0.gguf

gld_prompt=$1
err_prompt=$2
n_tok=$3
insertion_type=$4
i=$5

TENSORS_EXT="gld" N_TOK="$n_tok" build/bin/llama-cli -m $m -co -sp -p "$gld_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
TENSORS_EXT="err" N_TOK="$n_tok" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0

err_ext=err

if [ "$insertion_type" = "reccursive" ]
then
    n=23
    for i in $(seq 0 $n)
    do
        /bin/python3 hfedit.py $m $i $err_ext reccursive
        /bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./gguf_ggml_models/rec_$i.gguf --outtype "q8_0" # not right quantization, needs to be updated
        m=./gguf_ggml_models/rec_$i.gguf
        err_ext=rec_$i
        TENSORS_EXT="rec_$i" N_TOK=$n_tok build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
        ./compacts lout.gld lout.rec_$i
        ./compacts lout.gld lout.err
    done
elif [ "$insertion_type" = "single" ]
then
    /bin/python3 hfedit.py $m $i $err_ext single
    /bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./gguf_ggml_models/rec_$i.gguf --outtype "q8_0" # not right quantization, needs to be updated
    m=./gguf_ggml_models/rec_$i.gguf
    TENSORS_EXT="rec_$i" N_TOK=$n_tok build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
    # ./compacts k.gld k.rec_$i
    # ./compacts q.gld q.rec_$i
    # ./compacts v.gld v.rec_$i
    # ./compacts attn.gld attn.rec_$i
    # ./compacts inp.gld inp.rec_$i
    # ./compacts norm.gld norm.rec_$i
    # ./compacts gate.gld gate.rec_$i
    # ./compacts silu.gld silu.rec_$i
    # ./compacts gatepar.gld gatepar.rec_$i
    # ./compacts out.gld out.rec_$i
    ./compacts lout.gld lout.rec_$i
    ./compacts lout.gld lout.err
    ./showacts gatepar.rec_$i
elif [ "$insertion_type" = "all" ]
then
    /bin/python3 hfedit.py $m -1 $err_ext all
    /bin/python3 convert_hf_to_gguf.py ./torch_model --outfile ./gguf_ggml_models/rec_all.gguf --outtype "q8_0" # not right quantization, needs to be updated
    m=./gguf_ggml_models/rec_all.gguf
    TENSORS_EXT="rec_all" N_TOK="$n_tok" build/bin/llama-cli -m $m -co -sp -p "$err_prompt" -fa -ngl 80 -n 512 --no-warmup --temp 0
    ./compacts lout.gld lout.rec_all
    ./compacts lout.gld lout.err
else
    echo "Insertion type must be reccursive, single or all"
fi